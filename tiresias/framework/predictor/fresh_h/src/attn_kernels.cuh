// Fused attention forward of the unseen-kernel set (written for this set): O = softmax(Q K^T / sqrt(64)) V per (batch*head), non-causal, head_dim 64, bf16 Q/K/V [BH][S][64], fp32 O [BH][S][64].
// One block handles 16*WARPS query rows (each warp 16 rows), loops over key tiles of 64: S = Q K^T with mma.m16n8k16 (Q fragments held in registers, K fragments read with 32-bit shared loads from a
// pitch-72 tile), online softmax in the log2 domain (running row max and sum, ex2.approx, correction of the output accumulator), the probabilities are repacked from the accumulator layout into
// bf16 A fragments, and O += P V with V staged transposed ([d][key]) so that its fragments are 32-bit shared loads too. No ldmatrix, no cp.async, no data-dependent control flow; index arithmetic is
// unsigned 32-bit. S % 64 == 0 and S % (16*WARPS) == 0.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float at_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float at_rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<unsigned*>(&v); }

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attn_fwd(uint4* __restrict__ Q, uint4* __restrict__ K, uint4* __restrict__ V, float* __restrict__ O, unsigned S) {
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
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float sc = 0.125f * 1.4426950409f;                 // 1/sqrt(64) * log2(e)
    for (unsigned kt = 0; kt < S; kt += KT) {
        #pragma unroll
        for (unsigned i = 0; i < (KT * (D / 8)) / T; ++i) {            // 512 uint4 per tile for K and for V
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
