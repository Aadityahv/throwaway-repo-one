// Dependent fragment-load -> consumer microbenchmark (Blackwell sm_120). Question: inside one barrier phase whose warps first load operand fragments from shared memory and then feed them to a
// consumer pipe (tensor-core mma, or FP32 FFMA), how much of the SMALLER of the two pipe costs is NOT hidden behind the larger one, as a function of resident warps per SM and of the load:consumer ratio?
// The earlier LDS/HMMA microbenchmark (validation/micro_lds_hmma.cu) used independent loads and found full overlap at >= 2 resident blocks; the real kernels (ablation, raw/ablation.jsonl) do not overlap fully.
// This benchmark is generic (not a copy of any evaluated kernel): a loop of  {barrier; two or four fragment-load + consume steps; barrier},  in four modes:
//   0 both (loads feed the consumer), 1 loads only (consumer replaced by an xor fold), 2 consumer only (fragments loaded once, kept in registers), 3 empty (the two barriers only).
// Per iteration cycles per SM (clock64 of every warp, worst warp): T0 = empty, l = loads - T0, c = consumer - T0, m = both - T0;  beta = (m - max(l, c)) / min(l, c): 0 = the pipes fully overlap, 1 = fully serial.
// Consumers: C0 = bf16 mma.sync.m16n8k16 with TM x TN tiles (fragment loads exactly as a tiled kernel, pitch-40 shared tile, conflict free): (TM,TN) = (4,4) 24 LDS:16 HMMA, (2,4) 16:8, (1,4) 12:4;
//            C1 = FP32 FFMA on a TM x TN register tile fed by 128-bit shared loads: (8,8) 4 LDS.128:64 FFMA, (4,4) 2:16.
// Geometries (warps per block W, resident blocks per SM bps held by padding dynamic shared memory): W=4: bps 1,2,4,8; W=8: bps 1,2,4  => 4..32 warps per SM. One resident wave (188*bps blocks).
// No energy, no clock/power setting, no profiler. Output: one JSON line per configuration. Build: nvcc -O3 -arch=sm_120 -std=c++17 micro_dep_frag.cu -o micro_dep_frag
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);}}while(0)

__device__ __forceinline__ void hmma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int CONS, int TM, int TN, int MODE>
__global__ void __launch_bounds__(256) kd(long long iters, unsigned long long* t0, unsigned long long* t1, float* sink) {
  __shared__ __align__(16) __nv_bfloat16 As[64 * 40];
  __shared__ __align__(16) __nv_bfloat16 Bs[64 * 40];
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  for (unsigned i = tid; i < 64 * 40; i += blockDim.x) { As[i] = __float2bfloat16(((i * 37u) & 255) / 256.0f); Bs[i] = __float2bfloat16(((i * 91u) & 255) / 256.0f); }
  __syncthreads();
  constexpr int NA = CONS == 0 ? TM * TN * 4 : TM * TN;
  float acc[NA];
  #pragma unroll
  for (int i = 0; i < NA; i++) acc[i] = 0.f;
  unsigned xs = 0;
  unsigned af[CONS == 0 ? TM : 1][4], bf[CONS == 0 ? TN : 1][2];
  float fa[CONS == 1 ? TM : 1], fb[CONS == 1 ? TN : 1];
  const float4* Fa = reinterpret_cast<const float4*>(As);   // 64*40*2 = 5120 B = 320 float4
  const float4* Fb = reinterpret_cast<const float4*>(Bs);
  auto load0 = [&](unsigned ks) {
    #pragma unroll
    for (unsigned i = 0; i < TM; ++i) {
      const unsigned r0 = (warp * (TM * 16) + i * 16 + gid) & 63;
      const unsigned* p0 = reinterpret_cast<const unsigned*>(&As[r0 * 40 + ks + tig * 2]);
      const unsigned* p1 = reinterpret_cast<const unsigned*>(&As[((r0 + 8) & 63) * 40 + ks + tig * 2]);
      af[i][0] = p0[0]; af[i][1] = p1[0]; af[i][2] = p0[4]; af[i][3] = p1[4];
    }
    #pragma unroll
    for (unsigned j = 0; j < TN; ++j) {
      const unsigned c0 = (warp * (TN * 8) + j * 8 + gid) & 63;
      const unsigned* p = reinterpret_cast<const unsigned*>(&Bs[c0 * 40 + ks + tig * 2]);
      bf[j][0] = p[0]; bf[j][1] = p[4];
    }
  };
  auto load1 = [&](unsigned u) {
    #pragma unroll
    for (unsigned q = 0; q < TM / 4; ++q) { const float4 v = Fa[((u * (TM / 4) + q) * 32 + lane) & 255]; fa[q * 4] = v.x; fa[q * 4 + 1] = v.y; fa[q * 4 + 2] = v.z; fa[q * 4 + 3] = v.w; }
    #pragma unroll
    for (unsigned q = 0; q < TN / 4; ++q) { const float4 v = Fb[((u * (TN / 4) + q) * 32 + lane) & 255]; fb[q * 4] = v.x; fb[q * 4 + 1] = v.y; fb[q * 4 + 2] = v.z; fb[q * 4 + 3] = v.w; }
  };
  if constexpr (MODE == 2) { if constexpr (CONS == 0) load0(0); else load1(0); }
  unsigned long long c0 = clock64();
  for (long long it = 0; it < iters; it++) {
    __syncthreads();
    if constexpr (MODE != 3) {
      if constexpr (CONS == 0) {
        #pragma unroll
        for (unsigned ks = 0; ks < 32; ks += 16) {
          if constexpr (MODE != 2) load0(ks);
          if constexpr (MODE == 1) {
            #pragma unroll
            for (unsigned i = 0; i < TM; ++i) xs ^= af[i][0] ^ af[i][1] ^ af[i][2] ^ af[i][3];
            #pragma unroll
            for (unsigned j = 0; j < TN; ++j) xs ^= bf[j][0] ^ bf[j][1];
          } else {
            #pragma unroll
            for (unsigned i = 0; i < TM; ++i) {
              #pragma unroll
              for (unsigned j = 0; j < TN; ++j) hmma(&acc[(i * TN + j) * 4], af[i][0], af[i][1], af[i][2], af[i][3], bf[j][0], bf[j][1]);
            }
          }
        }
      } else {
        #pragma unroll
        for (unsigned u = 0; u < 4; ++u) {
          if constexpr (MODE != 2) load1(u);
          if constexpr (MODE == 1) {
            #pragma unroll
            for (unsigned i = 0; i < TM; ++i) xs ^= __float_as_uint(fa[i]);
            #pragma unroll
            for (unsigned j = 0; j < TN; ++j) xs ^= __float_as_uint(fb[j]);
          } else {
            #pragma unroll
            for (unsigned i = 0; i < TM; ++i) {
              #pragma unroll
              for (unsigned j = 0; j < TN; ++j) acc[i * TN + j] = fmaf(fa[i], fb[j], acc[i * TN + j]);
            }
          }
        }
      }
    } else xs += tid;
    __syncthreads();
  }
  unsigned long long c1 = clock64();
  if (lane == 0) { t0[blockIdx.x * 8 + warp] = c0; t1[blockIdx.x * 8 + warp] = c1; }
  float r = (float)xs; for (int i = 0; i < NA; i++) r += acc[i];
  sink[blockIdx.x * blockDim.x + tid] = r;
}

typedef void (*KF)(long long, unsigned long long*, unsigned long long*, float*);
template <int CONS, int TM, int TN> static KF pick(int mode) {
  switch (mode) { case 0: return kd<CONS, TM, TN, 0>; case 1: return kd<CONS, TM, TN, 1>; case 2: return kd<CONS, TM, TN, 2>; default: return kd<CONS, TM, TN, 3>; }
}

static double run_cfg(KF f, int blocks, int W, int pad, long long iters, unsigned long long* t0, unsigned long long* t1, float* sink) {
  f<<<blocks, W * 32, pad>>>(iters, t0, t1, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());   // run 1 had no launch check: a launch with more than 48 KB of shared memory failed silently and stale timestamps were read
  static unsigned long long h0[188 * 8 * 8], h1[188 * 8 * 8];
  CK(cudaMemcpy(h0, t0, (size_t)blocks * 8 * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h1, t1, (size_t)blocks * 8 * 8, cudaMemcpyDeviceToHost));
  double worst = 0; for (int b = 0; b < blocks; b++) for (int w = 0; w < W; w++) { double d = (double)(h1[b * 8 + w] - h0[b * 8 + w]); if (d > worst) worst = d; }
  return worst / (double)iters;
}

template <int CONS, int TM, int TN> static void run_shape(int sms, int clk_khz, unsigned long long* t0, unsigned long long* t1, float* sink) {
  const int geo[7][2] = {{4, 1}, {4, 2}, {4, 4}, {4, 8}, {8, 1}, {8, 2}, {8, 4}};
  for (int g = 0; g < 7; g++) {
    int W = geo[g][0], bps = geo[g][1]; double cyc[4]; int pad = 0, nat = 0; long long iters = 0;
    for (int mode = 3; mode >= 0; mode--) {                       // empty first
      KF f = pick<CONS, TM, TN>(mode); CK(cudaFuncSetAttribute((const void*)f, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)); int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f, W * 32, 0)); if (mode == 3) nat = nb;
      if (nb < bps) { fprintf(stderr, "skip: natural occupancy %d < %d\n", nb, bps); cyc[mode] = -1; continue; }
      int p = 0; while (nb > bps && p < 60000) { p += 256; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f, W * 32, p)); }
      if (mode == 3) pad = p; else pad = p;
      int blocks = sms * bps;
      if (iters == 0) { double per = run_cfg(f, blocks, W, p, 300, t0, t1, sink); iters = 300; (void)per; }
      double probe = run_cfg(f, blocks, W, p, 500, t0, t1, sink); long long it = (long long)(0.08 * clk_khz * 1000.0 / (probe > 1 ? probe : 1)); if (it < 2000) it = 2000;
      cyc[mode] = run_cfg(f, blocks, W, p, it, t0, t1, sink);
    }
    if (cyc[0] < 0 || cyc[1] < 0 || cyc[2] < 0 || cyc[3] < 0) continue;
    double l = cyc[1] - cyc[3], c = cyc[2] - cyc[3], m = cyc[0] - cyc[3], mx = l > c ? l : c, mn = l > c ? c : l;
    printf("{\"consumer\":\"%s\",\"TM\":%d,\"TN\":%d,\"warps_per_block\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"cycles_empty\":%.1f,\"cycles_loads\":%.1f,\"cycles_consumer\":%.1f,\"cycles_both\":%.1f,\"net_loads\":%.1f,\"net_consumer\":%.1f,\"net_both\":%.1f,\"both_over_max\":%.3f,\"beta\":%.3f}\n",
           CONS == 0 ? "hmma" : "ffma", TM, TN, W, bps, W * bps, cyc[3], cyc[1], cyc[2], cyc[0], l, c, m, m / mx, (m - mx) / mn); fflush(stdout);
  }
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int sms = p.multiProcessorCount; int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  fprintf(stderr, "device %s SMs %d clock_khz %d\n", p.name, sms, clk_khz);
  unsigned long long *t0, *t1; float* sink; CK(cudaMalloc(&t0, (size_t)sms * 8 * 8 * 8)); CK(cudaMalloc(&t1, (size_t)sms * 8 * 8 * 8)); CK(cudaMalloc(&sink, (size_t)sms * 8 * 256 * 4));
  run_shape<0, 4, 4>(sms, clk_khz, t0, t1, sink); run_shape<0, 2, 4>(sms, clk_khz, t0, t1, sink); run_shape<0, 1, 4>(sms, clk_khz, t0, t1, sink);
  run_shape<1, 8, 8>(sms, clk_khz, t0, t1, sink); run_shape<1, 4, 4>(sms, clk_khz, t0, t1, sink);
  return 0;
}
