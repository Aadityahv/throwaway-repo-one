// Ablation microbenchmark of the two tensor-core kernels under diagnosis (Blackwell sm_120): the tensor-core matrix multiply tc_gemm (fresh_g/src/tc_kernels.cuh) and the fused attention
// attn_fwd (fresh_h/src/attn_kernels.cuh). Question: where does the runtime that the static model does not predict (L2-resident cells 0.63-0.81x, fused-attention DRAM cells 0.78-0.88x) come from?
// The kernels are copied here with compile-time switches that REMOVE one part of the loop body at a time, so that per-phase times can be compared with the model's per-phase terms:
//   NOLOAD  tile(s) loaded once before the loop; the loop keeps both barriers and the compute (compute phase alone)
//   NOCOMP  loop keeps the global loads, the shared stores and both barriers; compute replaced by one shared load per thread (load phase alone)
//   NOBAR   both __syncthreads in the loop removed (a data race, timing only): warps of a block drift and overlap load and compute
//   NOHMMA  compute keeps the fragment shared loads, no mma (LDS only);  NOLDS  compute keeps the mma on register fragments (HMMA only)       [matmul]
//   VPLAIN  attention: V stored un-transposed with STS.128 (conflict free) instead of 8 STS.U16 with 8-way bank conflicts                     [attention]
//   QKONLY / QKSM / PVONLY  attention compute pieces: QK^T only; QK^T plus online softmax; P V only                                         [attention]
// The variant with no switch is the original arithmetic, checked bit-for-bit against the original kernel (headers included unchanged) and against its timing (control).
// Resident blocks per SM are held at the original kernel's value by padding dynamic shared memory (cudaOccupancy), so a variant that needs fewer registers does not change the occupancy.
// Timing: CUDA graph of back-to-back launches (batch sized to >= 5 ms), 5 windows of about 60 ms, median per launch; same method as the project's application timing harness.
// No energy, no profiler, no clock/power setting. Output: one JSON line per (kernel, cell, variant). Build: nvcc -O3 -arch=sm_120 -std=c++17 -I<fresh_g/src> -I<fresh_h/src> ablation_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tc_kernels.cuh"
#include "attn_kernels.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %d\n",cudaGetErrorString(e),__LINE__); exit(1);}}while(0)

__global__ void fill_bf16(__nv_bfloat16* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u + seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    p[i] = __float2bfloat16(((h & 0xffff) / 32768.0f - 1.0f) * 0.5f);
  }
}
__global__ void count_diff(const float* a, const float* b, size_t n, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) c += (__float_as_uint(a[i]) != __float_as_uint(b[i]));
  atomicAdd(out, c);
}

// ------------------------------------------------------------------------------------------------ tensor-core matrix multiply variants
enum { NOLOAD = 1, NOCOMP = 2, NOBAR = 4, NOHMMA = 8, NOLDS = 16, VPLAIN = 8, QKONLY = 16, QKSM = 32, PVONLY = 64 };

template <int BM, int BN, int WM, int WN, int V>
__global__ void __launch_bounds__(WM * WN * 32) tcv(uint4* __restrict__ A, uint4* __restrict__ B, float* __restrict__ C, unsigned N, unsigned K) {
  constexpr int BK = 32, PITCH = BK + 8;
  constexpr unsigned T = WM * WN * 32, TM = BM / WM / 16, TN = BN / WN / 8;
  __shared__ __align__(16) __nv_bfloat16 As[BM * PITCH];
  __shared__ __align__(16) __nv_bfloat16 Bs[BN * PITCH];
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const unsigned wm = warp / WN, wn = warp % WN;
  const unsigned kv = K / 8;
  uint4* Ab = A + blockIdx.y * BM * kv;
  uint4* Bb = B + blockIdx.x * BN * kv;
  float acc[TM][TN][4];
  #pragma unroll
  for (unsigned i = 0; i < TM; ++i) {
    #pragma unroll
    for (unsigned j = 0; j < TN; ++j) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f; }
  }
  unsigned xs = 0;
  unsigned af[TM][4], bf[TN][2];
  auto load_tiles = [&](unsigned k0) {
    #pragma unroll
    for (unsigned i = 0; i < (BM * (BK / 8)) / T; ++i) {
      const unsigned idx = tid + i * T, r = idx >> 2, c = idx & 3;
      *reinterpret_cast<uint4*>(&As[r * PITCH + c * 8]) = Ab[r * kv + (k0 >> 3) + c];
    }
    #pragma unroll
    for (unsigned i = 0; i < (BN * (BK / 8)) / T; ++i) {
      const unsigned idx = tid + i * T, r = idx >> 2, c = idx & 3;
      *reinterpret_cast<uint4*>(&Bs[r * PITCH + c * 8]) = Bb[r * kv + (k0 >> 3) + c];
    }
  };
  auto load_frags = [&](unsigned ks) {
    #pragma unroll
    for (unsigned i = 0; i < TM; ++i) {
      const unsigned r0 = wm * (TM * 16) + i * 16 + gid;
      const unsigned* p0 = reinterpret_cast<const unsigned*>(&As[r0 * PITCH + ks + tig * 2]);
      const unsigned* p1 = reinterpret_cast<const unsigned*>(&As[(r0 + 8) * PITCH + ks + tig * 2]);
      af[i][0] = p0[0]; af[i][1] = p1[0]; af[i][2] = p0[4]; af[i][3] = p1[4];
    }
    #pragma unroll
    for (unsigned j = 0; j < TN; ++j) {
      const unsigned c0 = wn * (TN * 8) + j * 8 + gid;
      const unsigned* p = reinterpret_cast<const unsigned*>(&Bs[c0 * PITCH + ks + tig * 2]);
      bf[j][0] = p[0]; bf[j][1] = p[4];
    }
  };
  if constexpr (V & NOLOAD) { load_tiles(0); __syncthreads(); }
  if constexpr (V & NOLDS) { load_frags(0); }
  for (unsigned k0 = 0; k0 < K; k0 += BK) {
    if constexpr (!(V & NOLOAD)) load_tiles(k0);
    if constexpr (!(V & NOBAR)) __syncthreads();
    if constexpr (V & NOCOMP) {
      xs ^= *reinterpret_cast<const unsigned*>(&As[(tid & (BM - 1)) * PITCH + ((k0 >> 5) & 3) * 2]) ^ *reinterpret_cast<const unsigned*>(&Bs[(tid & (BN - 1)) * PITCH + ((k0 >> 5) & 3) * 2]);
    } else {
      #pragma unroll
      for (unsigned ks = 0; ks < BK; ks += 16) {
        if constexpr (!(V & NOLDS)) load_frags(ks);
        if constexpr (V & NOHMMA) {
          #pragma unroll
          for (unsigned i = 0; i < TM; ++i) xs ^= af[i][0] ^ af[i][1] ^ af[i][2] ^ af[i][3];
          #pragma unroll
          for (unsigned j = 0; j < TN; ++j) xs ^= bf[j][0] ^ bf[j][1];
        } else {
          #pragma unroll
          for (unsigned i = 0; i < TM; ++i) {
            #pragma unroll
            for (unsigned j = 0; j < TN; ++j) mma_bf16_16816(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bf[j][0], bf[j][1]);
          }
        }
      }
    }
    if constexpr (!(V & NOBAR)) __syncthreads();
  }
  if constexpr (V != 0) acc[0][0][0] += (float)xs;
  float* Cb = C + blockIdx.y * BM * N + blockIdx.x * BN;
  #pragma unroll
  for (unsigned i = 0; i < TM; ++i) {
    #pragma unroll
    for (unsigned j = 0; j < TN; ++j) {
      const unsigned r = wm * (TM * 16) + i * 16 + gid, c = wn * (TN * 8) + j * 8 + tig * 2;
      *reinterpret_cast<float2*>(&Cb[r * N + c]) = make_float2(acc[i][j][0], acc[i][j][1]);
      *reinterpret_cast<float2*>(&Cb[(r + 8) * N + c]) = make_float2(acc[i][j][2], acc[i][j][3]);
    }
  }
}

// ------------------------------------------------------------------------------------------------ fused attention variants
template <int WARPS, int V>
__global__ void __launch_bounds__(WARPS * 32) atv(uint4* __restrict__ Q, uint4* __restrict__ K, uint4* __restrict__ Vv, float* __restrict__ O, unsigned S) {
  constexpr unsigned D = 64, KT = 64, PITCH = 72, T = WARPS * 32, QT = 16 * WARPS;
  __shared__ __align__(16) __nv_bfloat16 Ks[KT * PITCH];
  __shared__ __align__(16) __nv_bfloat16 Vt[D * PITCH];
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const unsigned bh = blockIdx.y, q0 = blockIdx.x * QT + warp * 16;
  const unsigned* Qw = reinterpret_cast<const unsigned*>(Q) + bh * S * (D / 2);
  uint4* Kb = K + bh * S * (D / 8); uint4* Vb = Vv + bh * S * (D / 8);
  float* Ob = O + bh * S * D;
  unsigned qa[4][4];
  #pragma unroll
  for (unsigned ks = 0; ks < 4; ++ks) {
    qa[ks][0] = Qw[(q0 + gid) * 32 + ks * 8 + tig];       qa[ks][1] = Qw[(q0 + gid + 8) * 32 + ks * 8 + tig];
    qa[ks][2] = Qw[(q0 + gid) * 32 + ks * 8 + tig + 4];   qa[ks][3] = Qw[(q0 + gid + 8) * 32 + ks * 8 + tig + 4];
  }
  float o[8][4];
  #pragma unroll
  for (unsigned j = 0; j < 8; ++j) { o[j][0] = 0.f; o[j][1] = 0.f; o[j][2] = 0.f; o[j][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const float sc = 0.125f * 1.4426950409f;
  unsigned xs = 0;
  auto load_tile = [&](unsigned kt) {
    #pragma unroll
    for (unsigned i = 0; i < (KT * (D / 8)) / T; ++i) {
      const unsigned idx = tid + i * T, r = idx >> 3, c = idx & 7;
      *reinterpret_cast<uint4*>(&Ks[r * PITCH + c * 8]) = Kb[(kt + r) * (D / 8) + c];
      const uint4 v = Vb[(kt + r) * (D / 8) + c];
      if constexpr (V & VPLAIN) { *reinterpret_cast<uint4*>(&Vt[r * PITCH + c * 8]) = v; }
      else {
        const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (unsigned e = 0; e < 8; ++e) Vt[(c * 8 + e) * PITCH + r] = vp[e];
      }
    }
  };
  if constexpr (V & NOLOAD) { load_tile(0); __syncthreads(); }
  for (unsigned kt = 0; kt < S; kt += KT) {
    if constexpr (!(V & NOLOAD)) load_tile(kt);
    if constexpr (!(V & NOBAR)) __syncthreads();
    if constexpr (V & NOCOMP) {
      xs ^= *reinterpret_cast<const unsigned*>(&Ks[(tid & 63) * PITCH + ((kt >> 6) & 3) * 2]) ^ *reinterpret_cast<const unsigned*>(&Vt[(tid & 63) * PITCH + ((kt >> 6) & 3) * 2]);
    } else {
      float s[8][4];
      #pragma unroll
      for (unsigned j = 0; j < 8; ++j) { s[j][0] = 0.f; s[j][1] = 0.f; s[j][2] = 0.f; s[j][3] = 0.f; }
      if constexpr (!(V & PVONLY)) {
        #pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
          #pragma unroll
          for (unsigned j = 0; j < 8; ++j) {
            const unsigned* p = reinterpret_cast<const unsigned*>(&Ks[(j * 8 + gid) * PITCH + ks * 16 + tig * 2]);
            mma_bf16(s[j], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], p[0], p[4]);
          }
        }
      }
      if constexpr (V & QKONLY) {
        float fs = 0.f;
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) fs += s[j][0] + s[j][1] + s[j][2] + s[j][3];
        o[0][0] += fs;
      } else if constexpr (!(V & PVONLY)) {
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) { mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1])); mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0 * sc), n1 = fmaxf(m1, mx1 * sc);
        const float c0 = at_ex2(m0 - n0), c1 = at_ex2(m1 - n1);
        m0 = n0; m1 = n1; l0 *= c0; l1 *= c1;
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          s[j][0] = at_ex2(fmaf(s[j][0], sc, -n0)); s[j][1] = at_ex2(fmaf(s[j][1], sc, -n0));
          s[j][2] = at_ex2(fmaf(s[j][2], sc, -n1)); s[j][3] = at_ex2(fmaf(s[j][3], sc, -n1));
          l0 += s[j][0] + s[j][1]; l1 += s[j][2] + s[j][3];
          o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1;
        }
      }
      if constexpr (V & QKSM) {
        float fs = 0.f;
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) fs += s[j][0] + s[j][1] + s[j][2] + s[j][3];
        o[0][0] += fs;
      } else if constexpr (!(V & QKONLY)) {
        #pragma unroll
        for (unsigned kk = 0; kk < 4; ++kk) {
          unsigned a0, a1, a2, a3;
          if constexpr (V & PVONLY) { a0 = qa[kk][0]; a1 = qa[kk][1]; a2 = qa[kk][2]; a3 = qa[kk][3]; }
          else { a0 = pack_bf16(s[2 * kk][0], s[2 * kk][1]); a1 = pack_bf16(s[2 * kk][2], s[2 * kk][3]); a2 = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]); a3 = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]); }
          #pragma unroll
          for (unsigned j = 0; j < 8; ++j) {
            const unsigned* p = reinterpret_cast<const unsigned*>(&Vt[(j * 8 + gid) * PITCH + kk * 16 + tig * 2]);
            mma_bf16(o[j], a0, a1, a2, a3, p[0], p[4]);
          }
        }
      }
    }
    if constexpr (!(V & NOBAR)) __syncthreads();
  }
  if constexpr (V != 0) o[0][0] += (float)xs;
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = at_rcp(l0), i1 = at_rcp(l1);
  #pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    *reinterpret_cast<float2*>(&Ob[(q0 + gid) * D + j * 8 + tig * 2]) = make_float2(o[j][0] * i0, o[j][1] * i0);
    *reinterpret_cast<float2*>(&Ob[(q0 + gid + 8) * D + j * 8 + tig * 2]) = make_float2(o[j][2] * i1, o[j][3] * i1);
  }
}

// ------------------------------------------------------------------------------------------------ host
typedef void (*TCF)(uint4*, uint4*, float*, unsigned, unsigned);
typedef void (*ATF)(uint4*, uint4*, uint4*, float*, unsigned);
struct Var { const char* name; int flags; };
static const Var MMV[] = {{"full", 0}, {"noload", NOLOAD}, {"nocomp", NOCOMP}, {"nobar", NOBAR}, {"noload_ldsonly", NOLOAD | NOHMMA}, {"noload_hmmaonly", NOLOAD | NOLDS}};
static const Var ATV[] = {{"full", 0}, {"noload", NOLOAD}, {"nocomp", NOCOMP}, {"nobar", NOBAR}, {"vplain", VPLAIN}, {"noload_qkonly", NOLOAD | QKONLY}, {"noload_qksm", NOLOAD | QKSM}, {"noload_pvonly", NOLOAD | PVONLY}};
template <int BM, int BN, int WM, int WN> static TCF tc_pick(int i) {
  switch (i) { case 0: return tcv<BM, BN, WM, WN, 0>; case 1: return tcv<BM, BN, WM, WN, NOLOAD>; case 2: return tcv<BM, BN, WM, WN, NOCOMP>; case 3: return tcv<BM, BN, WM, WN, NOBAR>;
    case 4: return tcv<BM, BN, WM, WN, NOLOAD | NOHMMA>; default: return tcv<BM, BN, WM, WN, NOLOAD | NOLDS>; }
}
template <int W> static ATF at_pick(int i) {
  switch (i) { case 0: return atv<W, 0>; case 1: return atv<W, NOLOAD>; case 2: return atv<W, NOCOMP>; case 3: return atv<W, NOBAR>; case 4: return atv<W, VPLAIN>;
    case 5: return atv<W, NOLOAD | QKONLY>; case 6: return atv<W, NOLOAD | QKSM>; default: return atv<W, NOLOAD | PVONLY>; }
}

template <class F> static int pad_to(F f, int threads, int target, int* nat, int* regs, int* local) {
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (const void*)f)); *regs = fa.numRegs; *local = (int)fa.localSizeBytes;
  int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f, threads, 0)); *nat = nb;
  int pad = 0;
  while (nb > target && pad < 60000) { pad += 256; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f, threads, pad)); }
  return pad;
}

// median per-launch microseconds of `launch` (graph batch >= 5 ms, 5 windows of ~60 ms)
template <class L> static void time_it(L launch, double* med, double* mn, double* mx) {
  cudaStream_t st; CK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; i++) launch(st); CK(cudaStreamSynchronize(st));
  CK(cudaEventRecord(e0, st)); for (int i = 0; i < 10; i++) launch(st); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); double t1 = ms * 1000.0 / 10;
  int batch = (int)(5000.0 / t1) + 1; if (batch > 4000) batch = 4000;
  cudaGraph_t g; cudaGraphExec_t ge; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); for (int i = 0; i < batch; i++) launch(st); CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0)); CK(cudaGraphLaunch(ge, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  int reps = (int)(60000.0 / (t1 * batch)) + 1; std::vector<double> v;
  for (int w = 0; w < 5; w++) { CK(cudaEventRecord(e0, st)); for (int r = 0; r < reps; r++) CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms * 1000.0 / ((double)reps * batch)); }
  std::sort(v.begin(), v.end()); *med = v[2]; *mn = v[0]; *mx = v[4];
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(st)); CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}

static size_t mismatches(const float* a, const float* b, size_t n) {
  unsigned long long* d; CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8)); count_diff<<<1024, 256>>>(a, b, n, d); CK(cudaDeviceSynchronize());
  unsigned long long h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); CK(cudaFree(d)); return (size_t)h;
}

template <int BM, int BN, int WM, int WN> static void run_tc(const char* regime, const char* cand, int M, int N, int K, int target_bps, bool only_some) {
  __nv_bfloat16 *A, *B; float *C, *Cref; size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
  CK(cudaMalloc(&A, na * 2)); CK(cudaMalloc(&B, nb * 2)); CK(cudaMalloc(&C, nc * 4)); CK(cudaMalloc(&Cref, nc * 4));
  fill_bf16<<<1024, 256>>>(A, na, 1); fill_bf16<<<1024, 256>>>(B, nb, 2); CK(cudaDeviceSynchronize());
  dim3 grid(N / BN, M / BM), block(WM * WN * 32); const int thr = WM * WN * 32;
  // original kernel: control and reference output
  { int nat, regs, local; int pad = pad_to(tc_gemm<BM, BN, WM, WN>, thr, target_bps, &nat, &regs, &local);
    auto L = [&](cudaStream_t s) { tc_gemm<BM, BN, WM, WN><<<grid, block, pad, s>>>((uint4*)A, (uint4*)B, Cref, N, K); };
    double med, mn, mx; time_it(L, &med, &mn, &mx);
    printf("{\"kernel\":\"tc_gemm\",\"cell\":\"%s/%s\",\"variant\":\"original_header\",\"blocks_per_sm_natural\":%d,\"pad\":%d,\"regs\":%d,\"local_bytes\":%d,\"us_median\":%.3f,\"us_min\":%.3f,\"us_max\":%.3f}\n", regime, cand, nat, pad, regs, local, med, mn, mx); fflush(stdout); }
  for (int vi = 0; vi < (int)(sizeof(MMV) / sizeof(MMV[0])); vi++) {
    TCF f = tc_pick<BM, BN, WM, WN>(vi); int nat, regs, local; int pad = pad_to(f, thr, target_bps, &nat, &regs, &local);
    auto L = [&](cudaStream_t s) { f<<<grid, block, pad, s>>>((uint4*)A, (uint4*)B, C, N, K); };
    double med, mn, mx; time_it(L, &med, &mn, &mx);
    long long diff = -1; if (vi == 0) { CK(cudaMemset(C, 0, nc * 4)); L(0); CK(cudaDeviceSynchronize()); diff = (long long)mismatches(C, Cref, nc); }
    printf("{\"kernel\":\"tc_gemm\",\"cell\":\"%s/%s\",\"variant\":\"%s\",\"blocks_per_sm_natural\":%d,\"pad\":%d,\"regs\":%d,\"local_bytes\":%d,\"us_median\":%.3f,\"us_min\":%.3f,\"us_max\":%.3f,\"mismatch_vs_original\":%lld}\n", regime, cand, MMV[vi].name, nat, pad, regs, local, med, mn, mx, diff); fflush(stdout);
  }
  CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(Cref));
}

template <int W> static void run_at(const char* regime, int BH, int S, int target_bps) {
  size_t nq = (size_t)BH * S * 64, no = nq; __nv_bfloat16 *Q, *K, *V; float *O, *Oref;
  CK(cudaMalloc(&Q, nq * 2)); CK(cudaMalloc(&K, nq * 2)); CK(cudaMalloc(&V, nq * 2)); CK(cudaMalloc(&O, no * 4)); CK(cudaMalloc(&Oref, no * 4));
  fill_bf16<<<1024, 256>>>(Q, nq, 3); fill_bf16<<<1024, 256>>>(K, nq, 4); fill_bf16<<<1024, 256>>>(V, nq, 5); CK(cudaDeviceSynchronize());
  dim3 grid(S / (16 * W), BH), block(W * 32); const int thr = W * 32; char cand[8]; snprintf(cand, 8, "c%d", W == 4 ? 1 : 2);
  { int nat, regs, local; int pad = pad_to(attn_fwd<W>, thr, target_bps, &nat, &regs, &local);
    auto L = [&](cudaStream_t s) { attn_fwd<W><<<grid, block, pad, s>>>((uint4*)Q, (uint4*)K, (uint4*)V, Oref, S); };
    double med, mn, mx; time_it(L, &med, &mn, &mx);
    printf("{\"kernel\":\"attn_fwd\",\"cell\":\"%s/%s\",\"variant\":\"original_header\",\"blocks_per_sm_natural\":%d,\"pad\":%d,\"regs\":%d,\"local_bytes\":%d,\"us_median\":%.3f,\"us_min\":%.3f,\"us_max\":%.3f}\n", regime, cand, nat, pad, regs, local, med, mn, mx); fflush(stdout); }
  for (int vi = 0; vi < (int)(sizeof(ATV) / sizeof(ATV[0])); vi++) {
    ATF f = at_pick<W>(vi); int nat, regs, local; int pad = pad_to(f, thr, target_bps, &nat, &regs, &local);
    auto L = [&](cudaStream_t s) { f<<<grid, block, pad, s>>>((uint4*)Q, (uint4*)K, (uint4*)V, O, S); };
    double med, mn, mx; time_it(L, &med, &mn, &mx);
    long long diff = -1; if (vi == 0) { CK(cudaMemset(O, 0, no * 4)); L(0); CK(cudaDeviceSynchronize()); diff = (long long)mismatches(O, Oref, no); }
    printf("{\"kernel\":\"attn_fwd\",\"cell\":\"%s/%s\",\"variant\":\"%s\",\"blocks_per_sm_natural\":%d,\"pad\":%d,\"regs\":%d,\"local_bytes\":%d,\"us_median\":%.3f,\"us_min\":%.3f,\"us_max\":%.3f,\"mismatch_vs_original\":%lld}\n", regime, cand, ATV[vi].name, nat, pad, regs, local, med, mn, mx, diff); fflush(stdout);
  }
  CK(cudaFree(Q)); CK(cudaFree(K)); CK(cudaFree(V)); CK(cudaFree(O)); CK(cudaFree(Oref));
}

int main(int argc, char** argv) {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  fprintf(stderr, "device %s SMs %d clock_khz %d\n", p.name, p.multiProcessorCount, clk);
  const char* which = argc > 1 ? argv[1] : "all";                       // all | tc | attn
  bool tc = strcmp(which, "attn") != 0, at = strcmp(which, "tc") != 0;
  if (tc) {
    // regimes of fresh_g/cells_g.py (M,N,K); c1 = 128x128 tile 8 warps (2 resident blocks per SM in the static analysis), c2 = 64x64 tile 4 warps (8)
    struct R { const char* n; int m, n_, k; } rs[] = {{"small", 512, 512, 512}, {"medium", 1024, 1024, 2048}, {"large", 8192, 8192, 512}};
    for (auto& r : rs) { run_tc<128, 128, 2, 4>(r.n, "c1", r.m, r.n_, r.k, 2, false); run_tc<64, 64, 2, 2>(r.n, "c2", r.m, r.n_, r.k, 8, false); }
  }
  if (at) {
    // regimes of fresh_h/cells_h.py (BH, S); c1 = 4 warps (4 resident blocks per SM), c2 = 8 warps (2)
    struct R { const char* n; int bh, s; } rs[] = {{"small", 16, 512}, {"medium", 64, 1024}, {"large", 256, 2048}};
    for (auto& r : rs) { run_at<4>(r.n, r.bh, r.s, 4); run_at<8>(r.n, r.bh, r.s, 2); }
  }
  return 0;
}
