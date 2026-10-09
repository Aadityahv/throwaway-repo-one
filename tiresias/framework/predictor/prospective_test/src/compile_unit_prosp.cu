#include "prosp_kernels.cuh"
template __global__ void tc_gemm<128, 64, 2, 2>(uint4*, uint4*, float*, unsigned, unsigned);
template __global__ void tc_gemm<256, 128, 4, 4>(uint4*, uint4*, float*, unsigned, unsigned);
template __global__ void attn_fwd<2>(uint4*, uint4*, uint4*, float*, unsigned);
template __global__ void attn_fwd<16>(uint4*, uint4*, uint4*, float*, unsigned);
template __global__ void attn_nomax<4>(uint4*, uint4*, uint4*, float*, unsigned);
template __global__ void attn_nomax<8>(uint4*, uint4*, uint4*, float*, unsigned);
template __global__ void tc_gemm_bias_relu<128, 128, 2, 4>(uint4*, uint4*, const float*, float*, unsigned, unsigned);
template __global__ void tc_gemm_bias_relu<64, 64, 2, 2>(uint4*, uint4*, const float*, float*, unsigned, unsigned);
