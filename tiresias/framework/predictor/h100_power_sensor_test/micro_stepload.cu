// A1: NVML step-load test. Synthetic compute-only load (FP32 FMA, no memory traffic) switched on and off for a list of durations while NVML board power and the
// cumulative energy counter are polled in-process as fast as the library answers. Records the true load edges (host steady clock around the launch/sync loop)
// beside every sample, so the analysis can measure the update period, the response lag to a step, and the shortest window whose energy agrees with a long one.
// Never changes clocks, power limits or persistence mode; refuses unless the device guard of cal_common.h passes and NVML reports the same UUID. One GPU only.
//
//   micro_stepload <out_dir> <idle_s> <on_s> [<on_s> ...]      (CAL_EXPECT_UUID, CUDA_VISIBLE_DEVICES, CAL_BOOKING_REF set as for the calibrator)
#include "cal_common.h"
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>

typedef void* NvmlDev;
struct Nvml { int (*init)(); int (*by_uuid)(const char*, NvmlDev*); int (*power)(NvmlDev, unsigned*); int (*energy)(NvmlDev, unsigned long long*); int (*limit)(NvmlDev, unsigned*); NvmlDev dev = nullptr; };
static Nvml load_nvml(const std::string& uuid) {
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) { fprintf(stderr, "REFUSED: libnvidia-ml.so.1 not found\n"); exit(1); }
  Nvml n; n.init = (int(*)())dlsym(h, "nvmlInit_v2"); n.by_uuid = (int(*)(const char*, NvmlDev*))dlsym(h, "nvmlDeviceGetHandleByUUID");
  n.power = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage"); n.energy = (int(*)(NvmlDev, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
  n.limit = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
  if (!n.init || !n.by_uuid || !n.power || !n.energy || !n.limit) { fprintf(stderr, "REFUSED: NVML symbols missing\n"); exit(1); }
  if (n.init() != 0 || n.by_uuid(uuid.c_str(), &n.dev) != 0) { fprintf(stderr, "REFUSED: NVML cannot open %s\n", uuid.c_str()); exit(1); }
  return n;
}
struct Sample { unsigned long long ns, power_ns_end; unsigned power_mw; unsigned long long energy_mj; };  // ns = time before the calls, power_ns_end = after the power call
static unsigned long long now_ns() { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__global__ void __launch_bounds__(256) fma_load(float* res, unsigned T) {
  float acc[4] = {1.0f, 1.0f, 1.0f, 1.0f}; const float vf = 0.75f + (threadIdx.x & 7) * 0.03125f;
#pragma unroll 1
  for (unsigned k = 0; k < T; ++k) {
#pragma unroll
    for (int f = 0; f < 64; ++f) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(acc[f & 3]) : "f"(0.9990234375f), "f"(vf));
  }
  res[blockIdx.x * blockDim.x + threadIdx.x] = acc[0] + acc[1] + acc[2] + acc[3];
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: micro_stepload <out_dir> <idle_s> <on_s>...\n"); return 1; }
  const std::string out_dir = argv[1]; const double idle_s = atof(argv[2]); std::vector<double> ons; for (int i = 3; i < argc; ++i) ons.push_back(atof(argv[i]));
  DeviceInfo dev = cal_init_device(); Nvml nvml = load_nvml(dev.uuid); unsigned limit_mw = 0; nvml.limit(nvml.dev, &limit_mw);
  const int blocks = dev.sm * std::min(4, dev.max_threads_sm / 256); float* res = nullptr; CK(cudaMalloc(&res, (size_t)blocks * 256 * 4));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  unsigned T = 2000; { cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); fma_load<<<blocks, 256, 0, st>>>(res, T); CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st)); fma_load<<<blocks, 256, 0, st>>>(res, T); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    T = (unsigned)std::max(1.0, T * 2.0 / std::max(0.05f, ms)); }  // one launch is about 2 ms
  const char* sl = getenv("SAMPLER_SLEEP_US"); const int sleep_us = sl ? atoi(sl) : 300; const char* cp = getenv("COUNTER_POLL_MS"); const int counter_poll_ms = cp ? atoi(cp) : 0;  // 0: read the energy counter in every sampler iteration; N: at most every N ms; the edge reads below always happen
  unsigned long long last_counter_ns = 0;
  std::atomic<bool> run(true), ok(true); std::vector<Sample> samples; samples.reserve(400000);
  std::thread sampler([&] { while (run.load()) { Sample s{}; s.ns = now_ns(); if (nvml.power(nvml.dev, &s.power_mw)) { ok.store(false); return; } s.power_ns_end = now_ns();
      if (counter_poll_ms == 0 || (s.ns - last_counter_ns) * 1e-6 >= counter_poll_ms) { if (nvml.energy(nvml.dev, &s.energy_mj)) { ok.store(false); return; } last_counter_ns = s.ns; } samples.push_back(s); std::this_thread::sleep_for(std::chrono::microseconds(sleep_us)); } });
  struct Edge { double on_s; unsigned long long t_on_ns, t_off_ns, e_before_mj, e_on_mj, e_off_mj, e_after_mj; }; std::vector<Edge> edges;
  std::this_thread::sleep_for(std::chrono::duration<double>(idle_s));
  for (double on : ons) {
    CK(cudaStreamSynchronize(st)); unsigned long long eb = 0, e_on = 0, e_off = 0, e_after = 0; nvml.energy(nvml.dev, &eb); const unsigned long long t_on = now_ns(); nvml.energy(nvml.dev, &e_on); long long n = 0;
    while ((now_ns() - t_on) * 1e-9 < on) { fma_load<<<blocks, 256, 0, st>>>(res, T); if (++n % 2 == 0) CK(cudaStreamSynchronize(st)); }
    CK(cudaStreamSynchronize(st)); const unsigned long long t_off = now_ns(); nvml.energy(nvml.dev, &e_off);
    std::this_thread::sleep_for(std::chrono::duration<double>(idle_s)); nvml.energy(nvml.dev, &e_after); edges.push_back({on, t_on, t_off, eb, e_on, e_off, e_after});
  }
  run.store(false); sampler.join();
  { std::ofstream f(out_dir + "/samples.csv"); f << "monotonic_ns,power_call_end_ns,board_power_mw,energy_counter_mj\n"; for (const auto& s : samples) f << s.ns << ',' << s.power_ns_end << ',' << s.power_mw << ',' << s.energy_mj << '\n'; }
  { std::ofstream f(out_dir + "/edges.json"); f << "{\"device\":\"" << dev.name << "\",\"power_limit_mw\":" << limit_mw << ",\"idle_s\":" << idle_s << ",\"kernel_trips\":" << T << ",\"blocks\":" << blocks << ",\"nvml_ok\":" << (ok.load() ? "true" : "false") << ",\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) f << (i ? "," : "") << "{\"on_s\":" << edges[i].on_s << ",\"t_on_ns\":" << edges[i].t_on_ns << ",\"t_off_ns\":" << edges[i].t_off_ns << ",\"e_before_mj\":" << edges[i].e_before_mj << ",\"e_on_mj\":" << edges[i].e_on_mj << ",\"e_off_mj\":" << edges[i].e_off_mj << ",\"e_after_mj\":" << edges[i].e_after_mj << "}"; f << "]}\n"; }
  CK(cudaFree(res)); CK(cudaStreamDestroy(st)); printf("{\"mode\":\"step_load\",\"samples\":%zu,\"steps\":%zu,\"nvml_ok\":%s}\n", samples.size(), edges.size(), ok.load() ? "true" : "false");
  return ok.load() ? 0 : 3;
}
