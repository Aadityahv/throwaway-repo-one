// Index arithmetic is 32-bit (all sizes of the set keep offsets below 2^31). Machine-learning kernels of the unseen-kernel set (written for this set; no pinned sample). float32, row-major, no data-dependent control flow: sizes are
// chosen so that every launch is exactly tiled (no partial blocks, no bounds guards). Transcendentals are explicit PTX approximations (ex2, rcp, rsqrt), so
// the compiled code has no fix-up branches. Candidate pairs compute the same result with different thread mappings (equal work).
#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float k_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float k_rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float k_rsq(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// GELU, tanh form: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), tanh(u) = 1 - 2 / (exp(2u) + 1)
__device__ __forceinline__ float k_gelu(float x) {
    const float u = 0.7978845608f * fmaf(0.044715f * x, x * x, x);
    const float t = fmaf(-2.0f, k_rcp(k_ex2(2.8853900818f * u) + 1.0f), 1.0f);
    return 0.5f * x * (1.0f + t);
}
// SwiGLU gate: silu(a) * b, silu(a) = a / (1 + exp(-a))
__device__ __forceinline__ float k_silu(float a) { return a * k_rcp(1.0f + k_ex2(-1.4426950409f * a)); }

__global__ void __launch_bounds__(256) gelu_s(float* __restrict__ x, float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    y[i] = k_gelu(x[i]);
}
__global__ void __launch_bounds__(256) gelu_v4(float4* __restrict__ x, float4* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const float4 v = x[i];
    y[i] = make_float4(k_gelu(v.x), k_gelu(v.y), k_gelu(v.z), k_gelu(v.w));
}
__global__ void __launch_bounds__(256) swiglu_s(float* __restrict__ a, float* __restrict__ b, float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    y[i] = k_silu(a[i]) * b[i];
}
__global__ void __launch_bounds__(256) swiglu_v4(float4* __restrict__ a, float4* __restrict__ b, float4* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const float4 u = a[i], v = b[i];
    y[i] = make_float4(k_silu(u.x) * v.x, k_silu(u.y) * v.y, k_silu(u.z) * v.z, k_silu(u.w) * v.w);
}

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w, one block per row. Block sum: warp shuffles, one shared slot per warp, every warp reduces the slots.
template <int T>
__device__ __forceinline__ float block_sum(float s, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float t = (lane < T / 32) ? red[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    return t;
}
template <int T>
__global__ void __launch_bounds__(T) rmsnorm_s(float* __restrict__ x, float* __restrict__ w, float* __restrict__ y, int cols, float inv_cols, float eps) {
    __shared__ float red[T / 32];
    float* xr = x + blockIdx.x * cols; float* yr = y + blockIdx.x * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += T) { const float v = xr[c]; s = fmaf(v, v, s); }
    const float inv = k_rsq(fmaf(block_sum<T>(s, red), inv_cols, eps));
    for (int c = threadIdx.x; c < cols; c += T) yr[c] = xr[c] * inv * w[c];
}
template <int T>
__global__ void __launch_bounds__(T) rmsnorm_v4(float4* __restrict__ x, float4* __restrict__ w, float4* __restrict__ y, int cols4, float inv_cols, float eps) {
    __shared__ float red[T / 32];
    float4* xr = x + blockIdx.x * cols4; float4* yr = y + blockIdx.x * cols4;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols4; c += T) { const float4 v = xr[c]; s = fmaf(v.x, v.x, s); s = fmaf(v.y, v.y, s); s = fmaf(v.z, v.z, s); s = fmaf(v.w, v.w, s); }
    const float inv = k_rsq(fmaf(block_sum<T>(s, red), inv_cols, eps));
    for (int c = threadIdx.x; c < cols4; c += T) { const float4 v = xr[c], g = w[c]; yr[c] = make_float4(v.x * inv * g.x, v.y * inv * g.y, v.z * inv * g.z, v.w * inv * g.w); }
}

// Rotary position embedding, half-split convention, head_dim 128 (64 pairs), layout [batch][seq][head][dim], tables [seq][64].
// rope_all: one block per (position, batch), HEADS*64 threads, one rotated pair per thread.  rope_one: one block per (position, batch*HEADS + head), 64 threads.
template <int HEADS>
__global__ void __launch_bounds__(HEADS * 64) rope_all(float* __restrict__ x, float* __restrict__ cs, float* __restrict__ sn, float* __restrict__ y) {
    const int t = threadIdx.x, h = t >> 6, i = t & 63;
    const int base = ((blockIdx.y * gridDim.x + blockIdx.x) * HEADS + h) * 128;
    const float c = cs[blockIdx.x * 64 + i], s = sn[blockIdx.x * 64 + i];
    const float x0 = x[base + i], x1 = x[base + i + 64];
    y[base + i] = x0 * c - x1 * s;
    y[base + i + 64] = fmaf(x0, s, x1 * c);
}
template <int HEADS>
__global__ void __launch_bounds__(64) rope_one(float* __restrict__ x, float* __restrict__ cs, float* __restrict__ sn, float* __restrict__ y) {
    const int i = threadIdx.x, b = blockIdx.y / HEADS, h = blockIdx.y % HEADS;
    const int base = ((b * gridDim.x + blockIdx.x) * HEADS + h) * 128;
    const float c = cs[blockIdx.x * 64 + i], s = sn[blockIdx.x * 64 + i];
    const float x0 = x[base + i], x1 = x[base + i + 64];
    y[base + i] = x0 * c - x1 * s;
    y[base + i + 64] = fmaf(x0, s, x1 * c);
}

// FP32 matrix multiply C[M,N] = A[M,K] B[K,N], row-major, one BMxBN tile per block, TMxTN outputs per thread, 256 threads. M % BM == N % BN == K % BK == 0.
// All index arithmetic is unsigned 32-bit so the compiler zero-extends offsets (IMAD.WIDE.U32) instead of emitting 64-bit sign-extension sequences.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) sgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, unsigned N, unsigned K) {
    constexpr unsigned T = (BM / TM) * (BN / TN);
    __shared__ float As[BK][BM + 1];
    __shared__ __align__(16) float Bs[BK][BN];
    const unsigned tid = threadIdx.x, tx = tid % (BN / TN), ty = tid / (BN / TN);
    float* Ab = A + blockIdx.y * BM * K;
    float* Bb = B + blockIdx.x * BN;
    float* Cb = C + blockIdx.y * BM * N + blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned i = 0; i < (BM * BK) / T; ++i) { const unsigned idx = tid + i * T, r = idx / BK, c = idx % BK; As[c][r] = Ab[r * K + k0 + c]; }
        #pragma unroll
        for (unsigned i = 0; i < (BK * BN) / T; ++i) { const unsigned idx = tid + i * T, r = idx / BN, c = idx % BN; Bs[r][c] = Bb[(k0 + r) * N + c]; }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) Cb[(ty * TM + i) * N + tx * TN + j] = acc[i][j];
    }
}
