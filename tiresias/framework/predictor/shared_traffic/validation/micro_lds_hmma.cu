// LDS / HMMA interleave microbenchmark (Blackwell sm_120). Question: inside one barrier phase, do shared-memory loads (LDS) and tensor-core MMAs (HMMA) issue concurrently
// (cost = max of the two single-class costs), serially (cost = sum), or in between, and does a dependent accumulator chain expose HMMA latency?
// One kernel, 4 warps per block, conflict-free 32-bit ld.shared and bf16 mma.sync.m16n8k16 (fp32 accumulate). Per loop iteration a warp issues LDS_N loads and 4 HMMA.
// Modes: 0 LDS only, 1 HMMA only, 2 interleaved (one LDS group between consecutive HMMA). chains: independent accumulator chains (4 = independent, 1 = every HMMA depends on the previous).
// Resident blocks per SM (bps) 1, 2, 4: grid = SM count x bps (all resident). Timing: clock64 of every warp (cycles of the whole SM from earliest start to latest end).
// No sub-second energy, no clock/power change, no profiler: pure cycle counting. Each configuration: one short probe pass to pick the iteration count (about 0.2 s), then one measured pass.
// Output: one JSON line per configuration. Build: nvcc -O3 -arch=sm_120 micro_lds_hmma.cu -o micro_lds_hmma
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);}}while(0)

__device__ __forceinline__ void hmma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned lds(unsigned addr) { unsigned v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr)); return v; }

template <int MODE, int LDS_N, int CHAINS>
__global__ void k(long long iters, unsigned long long* t0, unsigned long long* t1, float* sink) {
  __shared__ unsigned buf[4096];
  int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  for (int i = tid; i < 4096; i += blockDim.x) buf[i] = i * 2654435761u;
  __syncthreads();
  unsigned a[4] = {0x3f803f80u + tid, 0x3f003f00u, 0x3f803f00u, 0x3f003f80u}, b[2] = {0x3f803f80u, 0x3f003f00u + lane};
  float acc[4][4]; for (int c = 0; c < 4; c++) for (int j = 0; j < 4; j++) acc[c][j] = 0.f;
  unsigned s = 0; unsigned base = (unsigned)__cvta_generic_to_shared(buf) + lane * 4;     // lane-consecutive words: conflict free
  unsigned long long c0 = clock64();
  for (long long it = 0; it < iters; it++) {
    #pragma unroll
    for (int h = 0; h < 4; h++) {
      if (MODE != 0) { hmma(acc[CHAINS == 1 ? 0 : h], a, b); }
      if (MODE != 1) {
        #pragma unroll
        for (int l = 0; l < (LDS_N + 3 - h) / 4; l++) s ^= lds(base + ((h * 4 + l * 16 + (it & 3)) & 63) * 128);
      }
    }
  }
  unsigned long long c1 = clock64();
  if (lane == 0) { t0[blockIdx.x * 4 + w] = c0; t1[blockIdx.x * 4 + w] = c1; }
  float r = (float)s; for (int c = 0; c < 4; c++) for (int j = 0; j < 4; j++) r += acc[c][j];
  sink[blockIdx.x * blockDim.x + tid] = r;     // unconditional per-thread store: nothing in the loop can be eliminated as dead code (run 1 had the LDS-only loop removed by the compiler)
}

typedef void (*KF)(long long, unsigned long long*, unsigned long long*, float*);
static KF pick(int mode, int lds_n, int chains) {
  if (mode == 0) return lds_n == 8 ? (KF)k<0, 8, 4> : (KF)k<0, 11, 4>;
  if (mode == 1) return chains == 4 ? (KF)k<1, 8, 4> : (KF)k<1, 8, 1>;
  if (lds_n == 8) return chains == 4 ? (KF)k<2, 8, 4> : (KF)k<2, 8, 1>;
  return chains == 4 ? (KF)k<2, 11, 4> : (KF)k<2, 11, 1>;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int sms = p.multiProcessorCount; int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  fprintf(stderr, "device %s SMs %d clock_khz %d\n", p.name, sms, clk_khz);
  unsigned long long *t0, *t1; float* sink; int maxblocks = sms * 4;
  CK(cudaMalloc(&t0, maxblocks * 4 * 8)); CK(cudaMalloc(&t1, maxblocks * 4 * 8)); CK(cudaMalloc(&sink, maxblocks * 128 * 4));
  struct Cfg { int mode, lds_n, chains; };
  const int ratios[2] = {8, 11};                      // LDS per 4 HMMA: 2:1 (tensor matmul loop) and 2.75:1 (attention loop)
  for (int bps = 1; bps <= 4; bps *= (bps == 1 ? 2 : 2)) {
    int blocks = sms * bps; if (bps == 3) continue;
    for (int ri = 0; ri < 2; ri++) for (int ch = 4; ch >= 1; ch -= 3) {
      Cfg cfgs[3] = {{0, ratios[ri], 4}, {1, ratios[ri], ch}, {2, ratios[ri], ch}};
      for (int ci = 0; ci < 3; ci++) {
        if (cfgs[ci].mode == 0 && ch == 1) continue;      // LDS-only does not depend on the chain
        Cfg c = cfgs[ci]; KF f = pick(c.mode, c.lds_n, c.chains);
        long long probe_it = 2000; f<<<blocks, 128>>>(probe_it, t0, t1, sink); CK(cudaDeviceSynchronize());
        unsigned long long h0[1024 * 4], h1[1024 * 4];
        CK(cudaMemcpy(h0, t0, blocks * 4 * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h1, t1, blocks * 4 * 8, cudaMemcpyDeviceToHost));
        double cyc = 0; for (int i = 0; i < blocks * 4; i++) { double d = (double)(h1[i] - h0[i]); if (d > cyc) cyc = d; }
        double per_it = cyc / probe_it; long long iters = (long long)(0.2 * clk_khz * 1000.0 / per_it); if (iters < 4000) iters = 4000;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        f<<<blocks, 128>>>(iters, t0, t1, sink); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
        CK(cudaMemcpy(h0, t0, blocks * 4 * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h1, t1, blocks * 4 * 8, cudaMemcpyDeviceToHost));
        unsigned long long mn = ~0ull, mx = 0;
        for (int i = 0; i < blocks * 4; i++) { if (h0[i] < mn) mn = h0[i]; if (h1[i] > mx) mx = h1[i]; }   // clock64 is per SM: spans are taken per block below
        double worst = 0; for (int i = 0; i < blocks * 4; i++) { double d = (double)(h1[i] - h0[i]); if (d > worst) worst = d; }
        printf("{\"mode\":\"%s\",\"lds_per_4hmma\":%d,\"chains\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"iters\":%lld,\"ms\":%.3f,\"cycles_per_iter_per_sm\":%.3f}\n",
               c.mode == 0 ? "lds" : c.mode == 1 ? "hmma" : "mix", c.lds_n, c.mode == 0 ? 0 : c.chains, bps, bps * 4, iters, ms, worst / iters);
        fflush(stdout);
      }
    }
  }
  return 0;
}
