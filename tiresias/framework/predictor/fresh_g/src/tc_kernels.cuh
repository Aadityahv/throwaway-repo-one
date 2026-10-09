// Tensor-core matrix multiply of the unseen-kernel set (written for this set): C[M,N] (fp32) = A[M,K] (bf16) * B[N,K]^T (bf16), fp32 accumulate. This is the transformer linear layer
// y = x W^T with the weight W stored [out][in], so both operands are K-contiguous. One BMxBN output tile per block, warps arranged WMxWN, each warp computing a (BM/WM)x(BN/WN) sub-tile
// with mma.sync.m16n8k16 (bf16). Global tiles are staged in shared memory (uint4 loads and stores, row pitch BK+8 bf16 = 80 bytes: conflict-free 32-bit fragment loads), and the fragments are
// read with ordinary 32-bit shared loads in the documented mma fragment layout (no ldmatrix, no cp.async), so only the global/shared/mma instruction classes appear. M % BM == N % BN == K % BK == 0.
// Index arithmetic is unsigned 32-bit (zero-extended offsets), as in the FP32 kernels of the set.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) tc_gemm(uint4* __restrict__ A, uint4* __restrict__ B, float* __restrict__ C, unsigned N, unsigned K) {
    constexpr int BK = 32, PITCH = BK + 8;                 // bf16 elements per shared row (80 bytes)
    constexpr unsigned T = WM * WN * 32, TM = BM / WM / 16, TN = BN / WN / 8;   // m16 tiles and n8 tiles per warp
    __shared__ __align__(16) __nv_bfloat16 As[BM * PITCH];
    __shared__ __align__(16) __nv_bfloat16 Bs[BN * PITCH];
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const unsigned wm = warp / WN, wn = warp % WN;
    const unsigned kv = K / 8;                              // uint4 per row of A / B (8 bf16 each)
    uint4* Ab = A + blockIdx.y * BM * kv;
    uint4* Bb = B + blockIdx.x * BN * kv;
    float acc[TM][TN][4];
    #pragma unroll
    for (unsigned i = 0; i < TM; ++i) {
        #pragma unroll
        for (unsigned j = 0; j < TN; ++j) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f; }
    }
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (unsigned i = 0; i < (BM * (BK / 8)) / T; ++i) {          // A tile: BM rows x 4 uint4
            const unsigned idx = tid + i * T, r = idx >> 2, c = idx & 3;
            *reinterpret_cast<uint4*>(&As[r * PITCH + c * 8]) = Ab[r * kv + (k0 >> 3) + c];
        }
        #pragma unroll
        for (unsigned i = 0; i < (BN * (BK / 8)) / T; ++i) {
            const unsigned idx = tid + i * T, r = idx >> 2, c = idx & 3;
            *reinterpret_cast<uint4*>(&Bs[r * PITCH + c * 8]) = Bb[r * kv + (k0 >> 3) + c];
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < BK; ks += 16) {
            unsigned af[TM][4], bf[TN][2];
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
            #pragma unroll
            for (unsigned i = 0; i < TM; ++i) {
                #pragma unroll
                for (unsigned j = 0; j < TN; ++j) mma_bf16_16816(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bf[j][0], bf[j][1]);
            }
        }
        __syncthreads();
    }
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
