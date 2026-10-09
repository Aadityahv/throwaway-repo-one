// Tensor-core stage of the portable calibrator (optional stage `tensor`). Synthetic work only: bf16 mma.sync.m16n8k16 with fp32 accumulate. Two modes:
//
//   micro_tensor issue  <out_dir>                              pipe cells: dependent latency (1 warp, 1 chain) and issue cost (32 warps per SM, 4 independent chains) of the tensor MMA
//   micro_tensor energy <variant> <warmup_s> <window_s> <out_dir> [cap_fraction [forced_blocks]]   one energy window, variant tc_w1 | tc_w4 | tc_w16 = 1, 4 or 16 warps per SM; NVML power sampled in-process;
//                                                    with cap_fraction the window follows the cap-safe rule (calibrate/CAP_SAFE_WINDOWS.md), without it the full grid (one block per SM) as before
//
// The program never changes clocks, power limits or persistence mode and refuses unless the device guard of cal_common.h passes and NVML reports the same UUID. The energy counter is read only at
// the window edges (reading it in a fast sampler loop corrupts it, see blackwell_experiments/a1_step_load/RESULT_A1.md). Operands are bf16 pairs in [0.5, 2): finite, bounded, toggling.
#include "cal_common.h"
#include "cap_safe.h"
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned smid() { unsigned x; asm volatile("mov.u32 %0, %%smid;" : "=r"(x)); return x; }

// ---- issue / latency cells: all operands are bf16 1.0, so every accumulator element is exactly 16 * (number of MMAs it has absorbed)
template <int S, int U>
__global__ void __launch_bounds__(1024) issue_kernel(float* out, unsigned long long* cycles, int loops) {
  float acc[S][4];
  #pragma unroll
  for (int j = 0; j < S; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f; }
  const unsigned one = 0x3F803F80u;
  const unsigned long long begin = clock64();
  #pragma unroll 1
  for (int k = 0; k < loops; ++k) {
    #pragma unroll
    for (int u = 0; u < U; ++u) {
      #pragma unroll
      for (int j = 0; j < S; ++j) mma16816(acc[j], one, one, one, one, one, one);
    }
  }
  const unsigned long long elapsed = clock64() - begin;
  const unsigned lane = threadIdx.x & 31, warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  #pragma unroll
  for (int j = 0; j < S; ++j) { out[((size_t)(blockIdx.x * blockDim.x + threadIdx.x)) * S * 4 + j * 4] = acc[j][0]; out[((size_t)(blockIdx.x * blockDim.x + threadIdx.x)) * S * 4 + j * 4 + 3] = acc[j][3]; }
  if (lane == 0) cycles[warp] = elapsed;
}
template <int S, int U> static void issue_launch(int blocks, int threads, float* out, unsigned long long* cycles, int loops, cudaStream_t st) { issue_kernel<S, U><<<blocks, threads, 0, st>>>(out, cycles, loops); }

// ---- energy windows: one global load per trip (L2-resident), then 32 MMAs on four independent accumulator chains with operand registers derived from the loaded word
__global__ void init_pairs(unsigned* p, size_t words) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u + 12345u; x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    const unsigned h0 = 0x3F00u | (x & 0x7Fu), h1 = 0x3F00u | ((x >> 8) & 0x7Fu);     // bf16 in [0.5, 1)
    p[i] = (h1 << 16) | h0;
  }
}
template <int M>
__global__ void __launch_bounds__(512) mma_energy(const unsigned* __restrict__ in, unsigned* __restrict__ res, unsigned T, unsigned kin_mask) {
  const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x, lanes = gridDim.x * blockDim.x;
  float acc[4][4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f; }
  unsigned x = lane;
#pragma unroll 1
  for (unsigned k = 0; k < T; ++k) {
    unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(in + (size_t)(k & kin_mask) * lanes + lane) : "memory");
    x ^= v;
    const unsigned a0 = v, a1 = v ^ 0x00800080u, a2 = v ^ 0x00400040u, a3 = v ^ 0x00200020u, b0 = v ^ 0x00100010u, b1 = v ^ 0x00080008u;
    #pragma unroll
    for (int m = 0; m < M; ++m) mma16816(acc[m & 3], a0, a1, a2, a3, b0, b1);
  }
  float s = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  res[lane] = __float_as_uint(s) ^ x;
}

typedef void* NvmlDev;
struct Nvml { int (*init)(); int (*by_uuid)(const char*, NvmlDev*); int (*power)(NvmlDev, unsigned*); int (*temp)(NvmlDev, int, unsigned*); int (*clock)(NvmlDev, int, unsigned*); int (*energy)(NvmlDev, unsigned long long*); int (*limit)(NvmlDev, unsigned*); NvmlDev dev = nullptr; bool energy_ok = false; };
static Nvml load_nvml(const std::string& uuid) {
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) { fprintf(stderr, "REFUSED: libnvidia-ml.so.1 not found\n"); exit(1); }
  Nvml n; n.init = (int(*)())dlsym(h, "nvmlInit_v2"); n.by_uuid = (int(*)(const char*, NvmlDev*))dlsym(h, "nvmlDeviceGetHandleByUUID"); n.power = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
  n.temp = (int(*)(NvmlDev, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature"); n.clock = (int(*)(NvmlDev, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo"); n.energy = (int(*)(NvmlDev, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
  n.limit = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
  if (!n.init || !n.by_uuid || !n.power || !n.temp || !n.clock || !n.limit) { fprintf(stderr, "REFUSED: NVML symbols missing\n"); exit(1); }
  if (n.init() != 0 || n.by_uuid(uuid.c_str(), &n.dev) != 0) { fprintf(stderr, "REFUSED: NVML cannot open %s\n", uuid.c_str()); exit(1); }
  unsigned long long e0 = 0; n.energy_ok = n.energy && n.energy(n.dev, &e0) == 0; return n;
}
struct Sample { unsigned long long ns; unsigned power_mw, temp_c, gclk, mclk, util; };
static unsigned long long now_ns() { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int mode_issue(const std::string& out_dir, const DeviceInfo& dev) {
  struct Cell { int S, threads, blocks; };
  const Cell cells[2] = {{1, 32, 1}, {4, 1024, dev.sm}};
  const int points[6][2] = {{127, 1}, {127, 4}, {127, 16}, {509, 1}, {509, 4}, {509, 16}};
  std::ofstream f(out_dir + "/issue.json"); f << "{\"schema\":\"tensor_issue/1\",\"sm_count\":" << dev.sm << ",\"cells\":[";
  bool first = true;
  float* out = nullptr; unsigned long long* cycles = nullptr; CK(cudaMalloc(&out, (size_t)dev.sm * 1024 * 16 * 4)); CK(cudaMalloc(&cycles, (size_t)dev.sm * 32 * 8));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  for (const Cell& c : cells) for (const auto& p : points) {
    const int loops = p[0], unroll = p[1]; std::vector<double> med;
    for (int rep = 0; rep < 4; ++rep) {    // repetition 0 is a discarded warm-up
      CK(cudaMemset(out, 0, (size_t)c.blocks * c.threads * c.S * 4 * 4)); CK(cudaMemset(cycles, 0, (size_t)c.blocks * (c.threads / 32) * 8));
      if (c.S == 1) { if (unroll == 1) issue_launch<1, 1>(c.blocks, c.threads, out, cycles, loops, st); else if (unroll == 4) issue_launch<1, 4>(c.blocks, c.threads, out, cycles, loops, st); else issue_launch<1, 16>(c.blocks, c.threads, out, cycles, loops, st); }
      else { if (unroll == 1) issue_launch<4, 1>(c.blocks, c.threads, out, cycles, loops, st); else if (unroll == 4) issue_launch<4, 4>(c.blocks, c.threads, out, cycles, loops, st); else issue_launch<4, 16>(c.blocks, c.threads, out, cycles, loops, st); }
      CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
      std::vector<unsigned long long> cy((size_t)c.blocks * (c.threads / 32)); CK(cudaMemcpy(cy.data(), cycles, cy.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<float> h((size_t)c.blocks * c.threads * c.S * 4); CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
      const float expect = 16.0f * (float)loops * (float)unroll; bool ok = true;
      for (size_t i = 0; i < (size_t)c.blocks * c.threads; ++i) for (int j = 0; j < c.S; ++j) { if (h[i * c.S * 4 + j * 4] != expect || h[i * c.S * 4 + j * 4 + 3] != expect) ok = false; }
      if (!ok) { fprintf(stderr, "REFUSED: tensor issue cell accumulators differ from the exact expectation (%g)\n", expect); return 1; }
      std::sort(cy.begin(), cy.end()); if (rep > 0) med.push_back((double)cy[cy.size() / 2]);
    }
    std::sort(med.begin(), med.end());
    f << (first ? "" : ",") << "{\"streams\":" << c.S << ",\"threads\":" << c.threads << ",\"blocks\":" << c.blocks << ",\"loops\":" << loops << ",\"unroll\":" << unroll << ",\"cycles_median\":" << med[med.size() / 2] << ",\"cycles_min\":" << med.front() << ",\"cycles_max\":" << med.back() << "}"; first = false;
  }
  f << "]}\n"; CK(cudaFree(out)); CK(cudaFree(cycles)); CK(cudaStreamDestroy(st));
  printf("{\"mode\":\"tensor_issue\",\"ok\":true}\n"); return 0;
}

static int mode_energy(const std::string& variant, double warmup_s, double window_s, const std::string& out_dir, const DeviceInfo& dev, double cap_fraction, int force_blocks) {
  int warps = variant == "tc_w1" ? 1 : variant == "tc_w4" ? 4 : variant == "tc_w16" ? 16 : 0; if (!warps) { fprintf(stderr, "REFUSED: unknown variant %s\n", variant.c_str()); return 1; }
  constexpr int M = 32; Nvml nvml = load_nvml(dev.uuid);
  const int threads = warps * 32, blocks_full = dev.sm; const unsigned kin_mask = 7, T = 2048;   // 8 sweeps of one word per lane: a few MB, L2-resident
  if (cap_fraction < 0.0 || cap_fraction >= 1.0) { fprintf(stderr, "REFUSED: cap fraction must be 0 (off) or in (0,1)\n"); return 1; }
  if (force_blocks < 0 || force_blocks > blocks_full || (force_blocks > 0 && cap_fraction <= 0.0)) { fprintf(stderr, "REFUSED: forced block count %d outside 1..%d or without a cap fraction\n", force_blocks, blocks_full); return 1; }
  cudaStream_t st; CK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  struct Rig { int blocks = 0; size_t lanes = 0, in_words = 0; int batch = 1; unsigned *din = nullptr, *dres = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr; std::vector<unsigned> res_first; } rig;
  auto destroy = [&](Rig& g) { if (g.exec) { CK(cudaGraphExecDestroy(g.exec)); CK(cudaGraphDestroy(g.graph)); CK(cudaFree(g.din)); CK(cudaFree(g.dres)); g.exec = nullptr; } };
  auto build = [&](int blocks) {
    destroy(rig); Rig g; g.blocks = blocks; g.lanes = (size_t)blocks * threads; g.in_words = g.lanes * (kin_mask + 1);
    CK(cudaMalloc(&g.din, g.in_words * 4)); CK(cudaMalloc(&g.dres, g.lanes * 4));
    init_pairs<<<dev.sm * 8, 256>>>(g.din, g.in_words); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    mma_energy<M><<<blocks, threads, 0, st>>>(g.din, g.dres, T, kin_mask); CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
    g.res_first.resize(g.lanes); CK(cudaMemcpy(g.res_first.data(), g.dres, g.lanes * 4, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e0, st)); for (int i = 0; i < 5; ++i) mma_energy<M><<<blocks, threads, 0, st>>>(g.din, g.dres, T, kin_mask);
    CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); const double per_launch_s = std::max(1e-6, ms / 5.0 / 1000.0);
    g.batch = (int)std::min(2000.0, std::max(1.0, std::ceil(0.05 / per_launch_s)));
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < g.batch; ++i) mma_energy<M><<<blocks, threads, 0, st>>>(g.din, g.dres, T, kin_mask);
    CK(cudaStreamEndCapture(st, &g.graph)); CK(cudaGraphInstantiateWithFlags(&g.exec, g.graph, 0));
    rig = g; };
  auto spin = [&](double seconds) { unsigned long long w0 = cap_now_ns(); long long wg = 0; while ((cap_now_ns() - w0) * 1e-9 < seconds) { CK(cudaGraphLaunch(rig.exec, st)); if (++wg % 4 == 0) CK(cudaStreamSynchronize(st)); } CK(cudaStreamSynchronize(st)); };
  unsigned limit_mw = 0; if (nvml.limit(nvml.dev, &limit_mw) != 0) limit_mw = 0;
  CapRecord cap;
  if (force_blocks > 0) { cap.enabled = true; cap.forced = true; cap.fraction = cap_fraction; cap.limit_mw = limit_mw; cap.threshold_mw = cap_threshold_mw(cap_fraction, limit_mw); cap.blocks_full = blocks_full; cap.chosen = force_blocks; cap.scaled = force_blocks != blocks_full; build(force_blocks); spin(warmup_s); }
  else if (cap_fraction > 0.0) {
    if (!limit_mw) { fprintf(stderr, "REFUSED: NVML reported no enforced power limit, the cap-safe rule needs it\n"); return 1; }
    build(blocks_full);
    auto power = [&](unsigned* mw) { return nvml.power(nvml.dev, mw); };
    cap = cap_decide(cap_fraction, limit_mw, blocks_full, dev.sm, warmup_s, power, [&](int b) { build(b); }, spin);
    spin(std::max(warmup_s - cap.probe_elapsed_s, 0.5 * warmup_s));   // the probe steps count as warm-up, never less than half of it at the chosen grid
  } else { build(blocks_full); spin(warmup_s); }
  const int blocks = rig.blocks; const size_t lanes = rig.lanes; const int batch = rig.batch; cudaGraphExec_t exec = rig.exec; const std::vector<unsigned> res_first = rig.res_first; unsigned* dres = rig.dres;
  std::atomic<bool> run(true), ok(true); std::vector<Sample> samples; samples.reserve((size_t)((window_s + 6) * 400));
  std::thread sampler([&] { while (run.load()) { Sample s{now_ns(), 0, 0, 0, 0, 0}; if (nvml.power(nvml.dev, &s.power_mw) || nvml.temp(nvml.dev, 0, &s.temp_c) || nvml.clock(nvml.dev, 0, &s.gclk) || nvml.clock(nvml.dev, 2, &s.mclk)) { ok.store(false); return; }
      samples.push_back(s); std::this_thread::sleep_for(std::chrono::milliseconds(5)); } });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  unsigned long long en0 = 0, en1 = 0; if (nvml.energy_ok) nvml.energy(nvml.dev, &en0);
  CK(cudaStreamSynchronize(st)); const unsigned long long t_begin = now_ns(); long long graphs = 0;
  while ((now_ns() - t_begin) * 1e-9 < window_s) { CK(cudaGraphLaunch(exec, st)); ++graphs; if (graphs % 4 == 0) CK(cudaStreamSynchronize(st)); }
  CK(cudaStreamSynchronize(st)); const unsigned long long t_end = now_ns(); if (nvml.energy_ok) nvml.energy(nvml.dev, &en1);
  std::this_thread::sleep_for(std::chrono::milliseconds(300)); run.store(false); sampler.join();
  std::vector<unsigned> res_last(lanes); CK(cudaMemcpy(res_last.data(), dres, lanes * 4, cudaMemcpyDeviceToHost));
  const bool same = res_first == res_last; bool nonzero = false; for (size_t i = 0; i < res_last.size() && !nonzero; ++i) nonzero = res_last[i] != 0;
  { std::ofstream f(out_dir + "/samples.csv"); f << "monotonic_ns,board_power_mw,temperature_c,graphics_clock_mhz,memory_clock_mhz,utilization_percent\n"; for (const auto& s : samples) f << s.ns << ',' << s.power_mw << ',' << s.temp_c << ',' << s.gclk << ',' << s.mclk << ",0\n"; }
  const long long launches = graphs * batch; char row[4096];
  snprintf(row, sizeof row, "{\"mode\":\"tensor_energy_window\",\"design\":\"%s\",\"M\":%d,\"warps_per_sm\":%d,\"blocks\":%d,\"threads\":%d,\"lanes\":%zu,\"trips_per_launch\":%u,\"read_bytes_per_launch\":%.0f,\"write_bytes_per_launch\":0,"
    "\"graph_batch\":%d,\"graphs\":%lld,\"launches\":%lld,\"t_begin_ns\":%llu,\"t_end_ns\":%llu,\"window_seconds\":%.6f,\"warmup_seconds\":%.1f,\"samples\":%zu,\"nvml_sampler_ok\":%s,\"energy_counter_mj\":%lld,\"energy_counter_ok\":%s,"
    "\"power_limit_mw\":%u,\"output_deterministic\":%s,\"output_nonzero\":%s,\"device\":\"%s\",\"sm_count\":%d,\"cap_safe\":%s}\n",
    variant.c_str(), M, warps, blocks, threads, lanes, T, (double)lanes * T * 4, batch, graphs, launches, t_begin, t_end, (t_end - t_begin) * 1e-9, warmup_s, samples.size(), ok.load() ? "true" : "false",
    nvml.energy_ok ? (long long)(en1 - en0) : -1LL, nvml.energy_ok ? "true" : "false", limit_mw, same ? "true" : "false", nonzero ? "true" : "false", dev.name.c_str(), dev.sm, cap_json(cap).c_str());
  fputs(row, stdout); { std::ofstream f(out_dir + "/window.json"); f << row; }
  destroy(rig); CK(cudaStreamDestroy(st));
  return (ok.load() && same && nonzero) ? 0 : 3;
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "issue") { DeviceInfo dev = cal_init_device(); return mode_issue(argv[2], dev); }
  if (argc >= 6 && argc <= 8 && std::string(argv[1]) == "energy") { DeviceInfo dev = cal_init_device(); return mode_energy(argv[2], atof(argv[3]), atof(argv[4]), argv[5], dev, argc > 6 ? atof(argv[6]) : 0.0, argc > 7 ? atoi(argv[7]) : 0); }
  fprintf(stderr, "usage: micro_tensor issue <out_dir> | micro_tensor energy <tc_w1|tc_w4|tc_w16> <warmup_s> <window_s> <out_dir> [cap_fraction [forced_blocks]]\n"); return 1;
}
