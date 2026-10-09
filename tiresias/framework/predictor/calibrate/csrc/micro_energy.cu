// Energy calibration window: one design (a fixed mix of priced instruction classes and memory traffic), run in CUDA graphs for a warm-up and then a timed window while NVML
// power/temperature/clock samples are recorded in-process. Synthetic work only. Output: <out_dir>/samples.csv, <out_dir>/window.json and one JSON row on stdout.
// The program never changes clocks, power limits or persistence mode. It refuses unless the device guard of cal_common.h passes and NVML reports the same UUID.
//
//   micro_energy <design> <warmup_s> <window_s> <out_dir> [cap_fraction [forced_blocks]]
// With cap_fraction the window follows the cap-safe rule (calibrate/CAP_SAFE_WINDOWS.md, csrc/cap_safe.h); without it the full grid is used as before.
//
// Every priced instruction is an `asm volatile` PTX instruction, so the compiler can neither merge nor delete it; the Python side verifies the real SASS opcode counts of the
// built kernel against the design (and refuses on any spill, predicate-guarded loop instruction or count mismatch) before any energy is attributed. Loads read each
// address once per sweep (logical bytes == distinct bytes for DRAM-sized footprints); L2-sized footprints are re-swept across launches.
#include "cal_common.h"
#include "cap_safe.h"
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

typedef void* NvmlDev;
struct NvmlUtil { unsigned gpu, memory; };
struct Nvml {
  int (*init)(); int (*by_uuid)(const char*, NvmlDev*); int (*power)(NvmlDev, unsigned*); int (*temp)(NvmlDev, int, unsigned*); int (*clock)(NvmlDev, int, unsigned*);
  int (*energy)(NvmlDev, unsigned long long*); int (*limit)(NvmlDev, unsigned*); int (*util)(NvmlDev, NvmlUtil*);
  NvmlDev dev = nullptr; bool energy_ok = false;
};
static Nvml load_nvml(const std::string& uuid) {
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) { fprintf(stderr, "REFUSED: libnvidia-ml.so.1 not found\n"); exit(1); }
  Nvml n;
  n.init = (int(*)())dlsym(h, "nvmlInit_v2"); n.by_uuid = (int(*)(const char*, NvmlDev*))dlsym(h, "nvmlDeviceGetHandleByUUID");
  n.power = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage"); n.temp = (int(*)(NvmlDev, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature");
  n.clock = (int(*)(NvmlDev, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo"); n.energy = (int(*)(NvmlDev, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
  n.limit = (int(*)(NvmlDev, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit"); n.util = (int(*)(NvmlDev, NvmlUtil*))dlsym(h, "nvmlDeviceGetUtilizationRates");
  if (!n.init || !n.by_uuid || !n.power || !n.temp || !n.clock || !n.limit || !n.util) { fprintf(stderr, "REFUSED: NVML symbols missing\n"); exit(1); }
  if (n.init() != 0 || n.by_uuid(uuid.c_str(), &n.dev) != 0) { fprintf(stderr, "REFUSED: NVML cannot open %s\n", uuid.c_str()); exit(1); }
  unsigned long long e0 = 0; n.energy_ok = n.energy && n.energy(n.dev, &e0) == 0;
  return n;
}

struct Sample { unsigned long long ns; unsigned power_mw, temp_c, gclk, mclk, util; };
static unsigned long long now_ns() { return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the kernel: all doses are compile-time; footprints and trip count are run-time ----
__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
// Input words are floats in [0.5, 1) with pseudo-random mantissas (operand toggling like real data), also read as raw bits by the integer doses.
__global__ void init_input(unsigned* p, size_t words) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) p[i] = 0x3f000000u | (hash32((unsigned)i * 2654435761u + 12345u) & 0x007fffffu);
}

template <int R, int F, int A, int I, int S, int H, int B, int Z, int W>
__global__ void __launch_bounds__(256) energy_fixture(const unsigned* __restrict__ in, unsigned* __restrict__ out, unsigned* __restrict__ res, unsigned T, unsigned kin_mask, unsigned kout_mask) {
  __shared__ volatile unsigned shared[256];
  const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x, lanes = gridDim.x * blockDim.x;
  float acc[4] = {1.0f, 1.0f, 1.0f, 1.0f}, add[4] = {1.0f, 1.0f, 1.0f, 1.0f}; unsigned iv[4] = {lane, lane + 1, lane + 2, lane + 3}, x = lane, special_bits = 0; float sf = 0.75f;
#pragma unroll 1
  for (unsigned k = 0; k < T; ++k) {
    const size_t rbase = ((size_t)(k & kin_mask) * (R > 0 ? R : 1)) * lanes + lane;
    const size_t wbase = ((size_t)(k & kout_mask) * (W > 0 ? W : 1)) * lanes + lane;
    unsigned v0 = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      unsigned v;
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(in + rbase + (size_t)r * lanes) : "memory");
      if (r == 0) v0 = v;
      x ^= v;  // every loaded value is consumed: ptxas deletes dead loads even from asm volatile (found by the compile-only SASS check)
      const float vf = __uint_as_float(v);
#pragma unroll
      for (int f = 0; f < F; ++f) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(acc[f & 3]) : "f"(0.9990234375f), "f"(vf));
#pragma unroll
      for (int a = 0; a < A; a += 2) {  // add then subtract the same operand: bounded values, two FADD per pair
        asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(add[(a >> 1) & 3]) : "f"(vf));
        asm volatile("sub.rn.f32 %0, %0, %1;" : "+f"(add[(a >> 1) & 3]) : "f"(vf));
      }
#pragma unroll
      for (int i = 0; i < I; i += 2) {
        asm volatile("xor.b32 %0, %0, %1;" : "+r"(iv[(i >> 1) & 3]) : "r"(v));
        asm volatile("add.u32 %0, %0, %1;" : "+r"(iv[(i >> 1) & 3]) : "r"(v));
      }
#pragma unroll
      for (int s = 0; s < S; ++s) { asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(sf) : "f"(-sf)); special_bits = __float_as_uint(sf); }
#pragma unroll
      for (int h = 0; h < H; ++h) { shared[threadIdx.x] = x ^ v; x = shared[threadIdx.x]; }
#pragma unroll
      for (int b = 0; b < B; ++b) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(x));
#pragma unroll
      for (int z = 0; z < Z; ++z) asm volatile("bar.sync 0;" ::: "memory");
    }
#pragma unroll
    for (int w = 0; w < W; ++w) asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(out + wbase + (size_t)w * lanes), "r"(x ^ v0 ^ (unsigned)w) : "memory");
  }
  res[4 * (size_t)lane] = __float_as_uint(acc[0] + acc[1] + acc[2] + acc[3]);
  res[4 * (size_t)lane + 1] = __float_as_uint(add[0] + add[1] + add[2] + add[3]);
  res[4 * (size_t)lane + 2] = iv[0] ^ iv[1] ^ iv[2] ^ iv[3] ^ x;
  res[4 * (size_t)lane + 3] = special_bits;
}

struct Design { const char* name; int R, F, A, I, S, H, B, Z, W; const char* in_tier; const char* out_tier; int light; void (*launch)(dim3, dim3, const unsigned*, unsigned*, unsigned*, unsigned, unsigned, unsigned, cudaStream_t); };
#define LAUNCHER(R,F,A,I,S,H,B,Z,W) [](dim3 g, dim3 b, const unsigned* in, unsigned* out, unsigned* res, unsigned T, unsigned ki, unsigned ko, cudaStream_t s) { energy_fixture<R,F,A,I,S,H,B,Z,W><<<g,b,0,s>>>(in,out,res,T,ki,ko); }
// name, R F A I S H B Z W, input footprint tier, output footprint tier, light-geometry flag. Tiers: small (L2-resident, little traffic), l2 (about a quarter of L2), dram (four times L2), none.
static const Design DESIGNS[] = {
  {"fma16", 1,16,0,0,0,0,0,0,0, "small","none",0, LAUNCHER(1,16,0,0,0,0,0,0,0)}, {"fma64", 1,64,0,0,0,0,0,0,0, "small","none",0, LAUNCHER(1,64,0,0,0,0,0,0,0)},
  {"fpadd16", 1,0,16,0,0,0,0,0,0, "small","none",0, LAUNCHER(1,0,16,0,0,0,0,0,0)}, {"fpadd64", 1,0,64,0,0,0,0,0,0, "small","none",0, LAUNCHER(1,0,64,0,0,0,0,0,0)},
  {"int16", 1,0,0,16,0,0,0,0,0, "small","none",0, LAUNCHER(1,0,0,16,0,0,0,0,0)}, {"int64", 1,0,0,64,0,0,0,0,0, "small","none",0, LAUNCHER(1,0,0,64,0,0,0,0,0)},
  {"sfu16", 1,0,0,0,16,0,0,0,0, "small","none",0, LAUNCHER(1,0,0,0,16,0,0,0,0)}, {"sfu64", 1,0,0,0,64,0,0,0,0, "small","none",0, LAUNCHER(1,0,0,0,64,0,0,0,0)},
  {"shared16", 1,0,0,0,0,16,0,0,0, "small","none",0, LAUNCHER(1,0,0,0,0,16,0,0,0)}, {"shared64", 1,0,0,0,0,64,0,0,0, "small","none",0, LAUNCHER(1,0,0,0,0,64,0,0,0)},
  {"shfl16", 1,0,0,0,0,0,16,0,0, "small","none",0, LAUNCHER(1,0,0,0,0,0,16,0,0)}, {"shfl64", 1,0,0,0,0,0,64,0,0, "small","none",0, LAUNCHER(1,0,0,0,0,0,64,0,0)},
  {"bar16", 1,0,0,0,0,0,0,16,0, "small","none",0, LAUNCHER(1,0,0,0,0,0,0,16,0)}, {"bar64", 1,0,0,0,0,0,0,64,0, "small","none",0, LAUNCHER(1,0,0,0,0,0,0,64,0)},
  {"rd_l2_8", 8,0,0,0,0,0,0,0,0, "l2","none",0, LAUNCHER(8,0,0,0,0,0,0,0,0)}, {"rd_l2_16", 16,0,0,0,0,0,0,0,0, "l2","none",0, LAUNCHER(16,0,0,0,0,0,0,0,0)},
  {"rd_dram_8", 8,0,0,0,0,0,0,0,0, "dram","none",0, LAUNCHER(8,0,0,0,0,0,0,0,0)}, {"rd_dram_16", 16,0,0,0,0,0,0,0,0, "dram","none",0, LAUNCHER(16,0,0,0,0,0,0,0,0)},
  {"wr_l2_8", 1,0,0,0,0,0,0,0,8, "small","l2",0, LAUNCHER(1,0,0,0,0,0,0,0,8)}, {"wr_dram_8", 1,0,0,0,0,0,0,0,8, "small","dram",0, LAUNCHER(1,0,0,0,0,0,0,0,8)},
  {"wr_dram_16", 1,0,0,0,0,0,0,0,16, "small","dram",0, LAUNCHER(1,0,0,0,0,0,0,0,16)}, {"copy_dram_8", 8,0,0,0,0,0,0,0,8, "dram","dram",0, LAUNCHER(8,0,0,0,0,0,0,0,8)},
  {"light_int64", 1,0,0,64,0,0,0,0,0, "small","none",1, LAUNCHER(1,0,0,64,0,0,0,0,0)},
  {"mixed_a", 1,8,0,0,8,8,0,0,0, "small","none",0, LAUNCHER(1,8,0,0,8,8,0,0,0)}, {"mixed_b", 1,24,0,0,4,12,0,0,0, "small","none",0, LAUNCHER(1,24,0,0,4,12,0,0,0)},
  {"mixed_app", 8,0,8,8,0,0,4,4,0, "dram","none",0, LAUNCHER(8,0,8,8,0,0,4,4,0)},
};
static const Design* find_design(const std::string& n) { for (const auto& d : DESIGNS) if (n == d.name) return &d; return nullptr; }
static size_t pow2_at_least(size_t v) { size_t p = 1; while (p < v) p <<= 1; return p; }

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--list") { for (const auto& d : DESIGNS) printf("%s\n", d.name); return 0; }
  if (argc < 5) { fprintf(stderr, "usage: micro_energy <design> <warmup_s> <window_s> <out_dir> [cap_fraction [forced_blocks]] | --list\n"); return 1; }
  const Design* d = find_design(argv[1]); if (!d) { fprintf(stderr, "REFUSED: unknown design %s\n", argv[1]); return 1; }
  const double warmup_s = atof(argv[2]), window_s = atof(argv[3]); const std::string out_dir = argv[4];
  DeviceInfo dev = cal_init_device(); Nvml nvml = load_nvml(dev.uuid);
  const int bpsm = std::min(4, dev.max_threads_sm / 256); const int threads = d->light ? 32 : 256, blocks_full = d->light ? dev.sm : dev.sm * bpsm;
  const double cap_fraction = argc > 5 ? atof(argv[5]) : 0.0; const int force_blocks = argc > 6 ? atoi(argv[6]) : 0;   // optional: cap-safe rule (CAP_SAFE_WINDOWS.md); absent = previous behaviour
  if (argc > 5 && (cap_fraction <= 0.0 || cap_fraction >= 1.0)) { fprintf(stderr, "REFUSED: cap fraction must be in (0,1)\n"); return 1; }
  if (force_blocks < 0 || force_blocks > blocks_full || (force_blocks > 0 && cap_fraction <= 0.0)) { fprintf(stderr, "REFUSED: forced block count %d outside 1..%d or without a cap fraction\n", force_blocks, blocks_full); return 1; }
  const bool dram_in = !strcmp(d->in_tier, "dram"), dram_out = !strcmp(d->out_tier, "dram");
  cudaStream_t st; CK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  // everything that depends on the grid: footprints (in "sweeps": one sweep reads R*lanes words, each address once; kin_mask+1 sweeps form the input footprint), buffers, trip count, graph batch.
  struct Rig { int blocks = 0; size_t lanes = 0, kin_p = 1, kout_p = 1, in_words = 0, out_words = 0; unsigned T = 0, kin_mask = 0, kout_mask = 0; int batch = 1;
               unsigned *din = nullptr, *dout = nullptr, *dres = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr; std::vector<unsigned> res_first; } rig;
  auto destroy = [&](Rig& g) { if (g.exec) { CK(cudaGraphExecDestroy(g.exec)); CK(cudaGraphDestroy(g.graph)); CK(cudaFree(g.din)); CK(cudaFree(g.dout)); CK(cudaFree(g.dres)); g.exec = nullptr; } };
  auto build = [&](int blocks) {
    destroy(rig); Rig g; g.blocks = blocks; g.lanes = (size_t)blocks * threads;
    auto sweeps = [&](const char* tier, int per) -> size_t {
      if (!strcmp(tier, "none")) return 1;
      if (!strcmp(tier, "small")) return 8;  // 8 sweeps: a few MB, L2-resident on every supported GPU
      const size_t sweep_bytes = (size_t)std::max(per, 1) * g.lanes * 4;
      const double target = !strcmp(tier, "l2") ? 0.25 * dev.l2 : 4.0 * dev.l2;
      return std::max<size_t>(1, (size_t)std::ceil(target / sweep_bytes)); };
    const size_t kin_n = sweeps(d->in_tier, d->R), kout_n = sweeps(d->out_tier, d->W);
    g.kin_p = pow2_at_least(kin_n); g.kout_p = pow2_at_least(kout_n);  // masks need powers of two; the footprint is rounded up to the next one
    g.T = (dram_in || dram_out) ? (unsigned)std::max(g.kin_p, g.kout_p) : (!strcmp(d->in_tier, "l2") || !strcmp(d->out_tier, "l2")) ? (unsigned)std::max(g.kin_p, g.kout_p) * 8 : 4096u;
    g.kin_mask = (unsigned)g.kin_p - 1; g.kout_mask = (unsigned)g.kout_p - 1;
    g.in_words = (size_t)std::max(d->R, 1) * g.kin_p * g.lanes; g.out_words = (size_t)std::max(d->W, 1) * g.kout_p * g.lanes;
    if ((g.in_words + g.out_words) * 4 + 4 * g.lanes * 4 > dev.total_mem / 2) { fprintf(stderr, "REFUSED: footprint exceeds half of device memory\n"); exit(1); }
    CK(cudaMalloc(&g.din, g.in_words * 4)); CK(cudaMalloc(&g.dout, g.out_words * 4)); CK(cudaMalloc(&g.dres, g.lanes * 4 * 4));
    init_input<<<dev.sm * 8, 256>>>(g.din, g.in_words); CK(cudaGetLastError()); CK(cudaMemset(g.dout, 0, g.out_words * 4)); CK(cudaDeviceSynchronize());
    // choose the graph batch so one graph replay is about 50 ms
    d->launch(dim3(blocks), dim3(threads), g.din, g.dout, g.dres, g.T, g.kin_mask, g.kout_mask, st); CK(cudaGetLastError()); CK(cudaStreamSynchronize(st));
    g.res_first.resize(g.lanes * 4); CK(cudaMemcpy(g.res_first.data(), g.dres, g.lanes * 16, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e0, st)); for (int i = 0; i < 5; ++i) d->launch(dim3(blocks), dim3(threads), g.din, g.dout, g.dres, g.T, g.kin_mask, g.kout_mask, st);
    CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); const double per_launch_s = std::max(1e-6, ms / 5.0 / 1000.0);
    g.batch = (int)std::min(2000.0, std::max(1.0, std::ceil(0.05 / per_launch_s)));
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < g.batch; ++i) d->launch(dim3(blocks), dim3(threads), g.din, g.dout, g.dres, g.T, g.kin_mask, g.kout_mask, st);
    CK(cudaStreamEndCapture(st, &g.graph)); CK(cudaGraphInstantiateWithFlags(&g.exec, g.graph, 0));
    rig = g; };
  auto spin = [&](double seconds) { unsigned long long w0 = cap_now_ns(); long long wg = 0; while ((cap_now_ns() - w0) * 1e-9 < seconds) { CK(cudaGraphLaunch(rig.exec, st)); if (++wg % 4 == 0) CK(cudaStreamSynchronize(st)); } CK(cudaStreamSynchronize(st)); };  // bounded queue: the duration is `seconds`, not seconds plus a backlog
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
  const int blocks = rig.blocks; const size_t lanes = rig.lanes, kin_p = rig.kin_p, kout_p = rig.kout_p, in_words = rig.in_words, out_words = rig.out_words; const unsigned T = rig.T, kin_mask = rig.kin_mask, kout_mask = rig.kout_mask;
  const int batch = rig.batch; cudaGraphExec_t exec = rig.exec; const std::vector<unsigned> res_first = rig.res_first; unsigned* dres = rig.dres;
  (void)kin_mask; (void)kout_mask;
  // measured window with an in-process NVML sampler (about 200 Hz) that starts before and ends after the window
  std::atomic<bool> run(true), ok(true); std::vector<Sample> samples; samples.reserve((size_t)((window_s + 6) * 400));
  std::thread sampler([&] { while (run.load()) { Sample s{now_ns(), 0, 0, 0, 0, 0}; NvmlUtil u{0, 0};
      if (nvml.power(nvml.dev, &s.power_mw) || nvml.temp(nvml.dev, 0, &s.temp_c) || nvml.clock(nvml.dev, 0, &s.gclk) || nvml.clock(nvml.dev, 2, &s.mclk) || nvml.util(nvml.dev, &u)) { ok.store(false); return; }
      s.util = u.gpu; samples.push_back(s); std::this_thread::sleep_for(std::chrono::milliseconds(5)); } });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  unsigned long long en0 = 0, en1 = 0; if (nvml.energy_ok) nvml.energy(nvml.dev, &en0);
  CK(cudaStreamSynchronize(st)); const unsigned long long t_begin = now_ns(); long long graphs = 0;
  while ((now_ns() - t_begin) * 1e-9 < window_s) { CK(cudaGraphLaunch(exec, st)); ++graphs; if (graphs % 4 == 0) CK(cudaStreamSynchronize(st)); }
  CK(cudaStreamSynchronize(st)); const unsigned long long t_end = now_ns(); if (nvml.energy_ok) nvml.energy(nvml.dev, &en1);
  std::this_thread::sleep_for(std::chrono::milliseconds(300)); run.store(false); sampler.join();
  std::vector<unsigned> res_last(lanes * 4); CK(cudaMemcpy(res_last.data(), dres, lanes * 16, cudaMemcpyDeviceToHost));
  bool same = res_first == res_last; bool nonzero = false; for (size_t i = 0; i < res_last.size() && !nonzero; ++i) nonzero = res_last[i] != 0;
  { std::ofstream f(out_dir + "/samples.csv"); f << "monotonic_ns,board_power_mw,temperature_c,graphics_clock_mhz,memory_clock_mhz,utilization_percent\n";
    for (const auto& s : samples) f << s.ns << ',' << s.power_mw << ',' << s.temp_c << ',' << s.gclk << ',' << s.mclk << ',' << s.util << '\n'; }
  const long long launches = graphs * batch; const double read_bytes = (double)lanes * T * d->R * 4, write_bytes = (double)lanes * T * d->W * 4;
  char row[4096];
  snprintf(row, sizeof row, "{\"mode\":\"energy_window\",\"design\":\"%s\",\"R\":%d,\"F\":%d,\"A\":%d,\"I\":%d,\"S\":%d,\"H\":%d,\"B\":%d,\"Z\":%d,\"W\":%d,\"in_tier\":\"%s\",\"out_tier\":\"%s\",\"blocks\":%d,\"threads\":%d,\"lanes\":%zu,"
    "\"trips_per_launch\":%u,\"kin_sweeps\":%zu,\"kout_sweeps\":%zu,\"in_footprint_bytes\":%zu,\"out_footprint_bytes\":%zu,\"read_bytes_per_launch\":%.0f,\"write_bytes_per_launch\":%.0f,"
    "\"graph_batch\":%d,\"graphs\":%lld,\"launches\":%lld,\"t_begin_ns\":%llu,\"t_end_ns\":%llu,\"window_seconds\":%.6f,\"warmup_seconds\":%.1f,\"samples\":%zu,\"nvml_sampler_ok\":%s,"
    "\"energy_counter_mj\":%lld,\"energy_counter_ok\":%s,\"power_limit_mw\":%u,\"output_deterministic\":%s,\"output_nonzero\":%s,\"device\":\"%s\",\"l2_bytes\":%zu,\"sm_count\":%d,\"cap_safe\":%s}\n",
    d->name, d->R, d->F, d->A, d->I, d->S, d->H, d->B, d->Z, d->W, d->in_tier, d->out_tier, blocks, threads, lanes, T, kin_p, kout_p, in_words * 4, out_words * 4, read_bytes, write_bytes,
    batch, graphs, launches, t_begin, t_end, (t_end - t_begin) * 1e-9, warmup_s, samples.size(), ok.load() ? "true" : "false", nvml.energy_ok ? (long long)(en1 - en0) : -1LL,
    nvml.energy_ok ? "true" : "false", limit_mw, same ? "true" : "false", nonzero ? "true" : "false", dev.name.c_str(), dev.l2, dev.sm, cap_json(cap).c_str());
  fputs(row, stdout); { std::ofstream f(out_dir + "/window.json"); f << row; }
  destroy(rig); CK(cudaStreamDestroy(st));
  return (ok.load() && same && nonzero) ? 0 : 3;
}
