// Cap-safe energy windows (rule: calibrate/CAP_SAFE_WINDOWS.md). Shared by micro_energy.cu and micro_tensor.cu. The caller provides three callables:
//   power(unsigned* mw) -> int     NVML board power in mW, 0 on success
//   rebuild(int blocks)            (re)allocate the footprint and the CUDA graph for `blocks` blocks
//   spin(double seconds)           replay the current graph back-to-back for `seconds`
// The choice formula `cap_next_blocks` is mirrored exactly by calibrate/cal/capsafe.py, which re-derives every recorded choice from the recorded probe steps.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

struct CapStep { int blocks; double mean_mw; size_t samples; };
struct CapRecord {
  bool enabled = false, forced = false, scaled = false, converged = true;
  double fraction = 0, idle_mw = 0, threshold_mw = 0; unsigned limit_mw = 0; int blocks_full = 0, chosen = 0, sm_count = 0, rule_version = 2; double probe_elapsed_s = 0;
  std::vector<CapStep> steps;
};

static const int CAP_MAX_PROBES = 8;
static double cap_q(double x) { return std::floor(x * 1000.0 + 0.5) / 1000.0; }   // inputs are quantised to 0.001 mW so that the recorded values reproduce the choice exactly
static double cap_threshold_mw(double fraction, unsigned limit_mw) { return cap_q(fraction * (double)limit_mw); }
static double cap_settle_measure_s(double warmup_s) { return std::min(6.0, std::max(1.0, 0.1 * warmup_s)); }

// Next block count after the last probe in `steps` (rule version 2, CAP_SAFE_WINDOWS.md Amendment 1). Unchanged when the last probe is at or below the threshold or already at one block.
// Above the threshold: more blocks than SMs -> one block per SM; at one block per SM -> linear proposal in dynamic power over active SMs (blocks < SMs, whole SMs idle);
// already below one block per SM -> halve. Always a strict decrease, at least one block.
static int cap_next_blocks(const std::vector<CapStep>& steps, int sm, double idle_mw, double threshold_mw) {
  const int b = steps.back().blocks; const double power_mw = steps.back().mean_mw;
  if (power_mw <= threshold_mw || b <= 1) return b;
  if (b > sm) return sm;
  if (b < sm) return std::max(1, b / 2);
  const double dyn = power_mw - idle_mw;
  const double k = dyn > 0 ? std::floor(sm * (threshold_mw - idle_mw) / dyn) : (double)(b - 1);
  return (int)std::max(1.0, std::min((double)(b - 1), k));
}

static unsigned long long cap_now_ns() { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Mean power over the samples taken at or after `t_from_ns`, sampling every 10 ms while `body` runs on the calling thread.
template <class PowerFn, class Body>
static bool cap_sample_mean(PowerFn power, Body body, unsigned long long t_from_ns, double* mean_mw, size_t* n) {
  std::atomic<bool> run(true), ok(true); std::vector<std::pair<unsigned long long, unsigned>> s; s.reserve(4096);
  std::thread th([&] { while (run.load()) { unsigned mw = 0; if (power(&mw) != 0) { ok.store(false); return; } s.push_back({cap_now_ns(), mw}); std::this_thread::sleep_for(std::chrono::milliseconds(10)); } });
  body(); run.store(false); th.join();
  double sum = 0; size_t c = 0; for (const auto& x : s) if (x.first >= t_from_ns) { sum += x.second; ++c; }
  if (!ok.load() || c == 0) return false;
  *mean_mw = cap_q(sum / c); *n = c; return true;
}

template <class PowerFn, class RebuildFn, class SpinFn>
static CapRecord cap_decide(double fraction, unsigned limit_mw, int blocks_full, int sm_count, double warmup_s, PowerFn power, RebuildFn rebuild, SpinFn spin) {
  CapRecord r; r.enabled = true; r.fraction = fraction; r.limit_mw = limit_mw; r.blocks_full = blocks_full; r.sm_count = sm_count; r.threshold_mw = cap_threshold_mw(fraction, limit_mw);
  const unsigned long long t0 = cap_now_ns(); size_t n = 0;
  if (!cap_sample_mean(power, [&] { std::this_thread::sleep_for(std::chrono::milliseconds(2000)); }, 0, &r.idle_mw, &n)) { fprintf(stderr, "REFUSED: NVML power unreadable during the idle measurement\n"); exit(1); }
  const double sm = cap_settle_measure_s(warmup_s); int b = blocks_full;
  for (int step = 1; step <= CAP_MAX_PROBES; ++step) {
    double mean = 0; const unsigned long long ts = cap_now_ns();
    if (!cap_sample_mean(power, [&] { spin(2 * sm); }, ts + (unsigned long long)(sm * 1e9), &mean, &n)) { fprintf(stderr, "REFUSED: NVML power unreadable during the probe\n"); exit(1); }
    r.steps.push_back({b, mean, n});
    const int nb = cap_next_blocks(r.steps, sm_count, r.idle_mw, r.threshold_mw);
    if (nb == b) { r.chosen = b; r.converged = mean <= r.threshold_mw; break; }          // at or below the threshold (converged), or already at one block (cannot reduce further)
    if (step == CAP_MAX_PROBES) { r.chosen = b; r.converged = false; break; }            // the last probe still exceeded the threshold: the last measured grid is used, no unmeasured reduction
    rebuild(nb); b = nb;
  }
  r.scaled = r.chosen != blocks_full; r.probe_elapsed_s = (double)(cap_now_ns() - t0) * 1e-9;
  return r;
}

static std::string cap_json(const CapRecord& r) {
  if (!r.enabled) return "{\"enabled\":false}";
  char b[384]; std::string s = "{\"enabled\":true";
  snprintf(b, sizeof b, ",\"rule_version\":%d,\"sm_count\":%d,\"fraction\":%.4f,\"limit_mw\":%u,\"threshold_mw\":%.3f,\"idle_mw\":%.3f,\"blocks_full\":%d,\"blocks_chosen\":%d,\"scaled\":%s,\"forced\":%s,\"converged\":%s,\"probe_elapsed_s\":%.1f,\"probe_steps\":[",
           r.rule_version, r.sm_count, r.fraction, r.limit_mw, r.threshold_mw, r.idle_mw, r.blocks_full, r.chosen, r.scaled ? "true" : "false", r.forced ? "true" : "false", r.converged ? "true" : "false", r.probe_elapsed_s);
  s += b;
  for (size_t i = 0; i < r.steps.size(); ++i) { snprintf(b, sizeof b, "%s{\"blocks\":%d,\"mean_mw\":%.3f,\"samples\":%zu}", i ? "," : "", r.steps[i].blocks, r.steps[i].mean_mw, r.steps[i].samples); s += b; }
  return s + "]}";
}
