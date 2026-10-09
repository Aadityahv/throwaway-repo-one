#include "tc_kernels.cuh"
template __global__ void tc_gemm<128, 128, 2, 4>(uint4*, uint4*, float*, unsigned, unsigned);
template __global__ void tc_gemm<64, 64, 2, 2>(uint4*, uint4*, float*, unsigned, unsigned);
