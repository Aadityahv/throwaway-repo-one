// New kernels of the prospective test of the attention / tensor-core runtime proposal (written for this test; the existing tc_kernels.cuh and attn_kernels.cuh are included unchanged).
//   tc_gemm_bias_relu<BM,BN,WM,WN>: the main loop of tc_gemm (same instructions, same shared-memory layout) followed by a plain epilogue after the loop: y = max(acc + bias[n], 0).
//   attn_nomax<WARPS>: attention without a running maximum: S = Q K^T (MMA stage), P = ex2(scale * S) with the row sums accumulated (exp stage), O += P V (MMA stage); no shuffles or max
//     reduction inside the loop, no output correction; the row sums are reduced over the 4 lanes of a row after the loop. Inputs are bounded (|scale * S| <= 11.6) so no overflow occurs.
// Index arithmetic is unsigned 32-bit as in the existing kernels.
#pragma once
#include "tc_kernels.cuh"
#include "attn_kernels.cuh"

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) tc_gemm_bias_relu(uint4* __restrict__ A, uint4* __restrict__ B, const float* __restrict__ bias, float* __restrict__ C, unsigned N, unsigned K) {
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
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
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
    const float* bb = bias + blockIdx.x * BN;
    #pragma unroll
    for (unsigned i = 0; i < TM; ++i) {
        #pragma unroll
        for (unsigned j = 0; j < TN; ++j) {
            const unsigned r = wm * (TM * 16) + i * 16 + gid, c = wn * (TN * 8) + j * 8 + tig * 2;
            const float b0 = bb[c], b1 = bb[c + 1];
            *reinterpret_cast<float2*>(&Cb[r * N + c]) = make_float2(fmaxf(acc[i][j][0] + b0, 0.f), fmaxf(acc[i][j][1] + b1, 0.f));
            *reinterpret_cast<float2*>(&Cb[(r + 8) * N + c]) = make_float2(fmaxf(acc[i][j][2] + b0, 0.f), fmaxf(acc[i][j][3] + b1, 0.f));
        }
    }
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) attn_nomax(uint4* __restrict__ Q, uint4* __restrict__ K, uint4* __restrict__ V, float* __restrict__ O, unsigned S) {
    constexpr unsigned D = 64, KT = 64, PITCH = 72, T = WARPS * 32, QT = 16 * WARPS;
    __shared__ __align__(16) __nv_bfloat16 Ks[KT * PITCH];     // [key][d]
    __shared__ __align__(16) __nv_bfloat16 Vt[D * PITCH];      // [d][key]
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const unsigned bh = blockIdx.y, q0 = blockIdx.x * QT + warp * 16;
    const unsigned* Qw = reinterpret_cast<const unsigned*>(Q) + bh * S * (D / 2);
    uint4* Kb = K + bh * S * (D / 8); uint4* Vb = V + bh * S * (D / 8);
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
    float l0 = 0.f, l1 = 0.f;
    const float sc = 0.125f * 1.4426950409f;                 // 1/sqrt(64) * log2(e)
    for (unsigned kt = 0; kt < S; kt += KT) {
        #pragma unroll
        for (unsigned i = 0; i < (KT * (D / 8)) / T; ++i) {
            const unsigned idx = tid + i * T, r = idx >> 3, c = idx & 7;
            *reinterpret_cast<uint4*>(&Ks[r * PITCH + c * 8]) = Kb[(kt + r) * (D / 8) + c];
            const uint4 v = Vb[(kt + r) * (D / 8) + c]; const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
            #pragma unroll
            for (unsigned e = 0; e < 8; ++e) Vt[(c * 8 + e) * PITCH + r] = vp[e];
        }
        __syncthreads();
        float s[8][4];
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) { s[j][0] = 0.f; s[j][1] = 0.f; s[j][2] = 0.f; s[j][3] = 0.f; }
        #pragma unroll
        for (unsigned ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const unsigned* p = reinterpret_cast<const unsigned*>(&Ks[(j * 8 + gid) * PITCH + ks * 16 + tig * 2]);
                mma_bf16(s[j], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], p[0], p[4]);
            }
        }
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            s[j][0] = at_ex2(s[j][0] * sc); s[j][1] = at_ex2(s[j][1] * sc);
            s[j][2] = at_ex2(s[j][2] * sc); s[j][3] = at_ex2(s[j][3] * sc);
            l0 += s[j][0] + s[j][1]; l1 += s[j][2] + s[j][3];
        }
        #pragma unroll
        for (unsigned kk = 0; kk < 4; ++kk) {
            const unsigned a0 = pack_bf16(s[2 * kk][0], s[2 * kk][1]), a1 = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
            const unsigned a2 = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), a3 = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const unsigned* p = reinterpret_cast<const unsigned*>(&Vt[(j * 8 + gid) * PITCH + kk * 16 + tig * 2]);
                mma_bf16(o[j], a0, a1, a2, a3, p[0], p[4]);
            }
        }
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = at_rcp(l0), i1 = at_rcp(l1);
    #pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        *reinterpret_cast<float2*>(&Ob[(q0 + gid) * D + j * 8 + tig * 2]) = make_float2(o[j][0] * i0, o[j][1] * i0);
        *reinterpret_cast<float2*>(&Ob[(q0 + gid + 8) * D + j * 8 + tig * 2]) = make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}
