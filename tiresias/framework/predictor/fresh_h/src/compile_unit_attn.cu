#include "attn_kernels.cuh"
template __global__ void attn_fwd<4>(uint4*, uint4*, uint4*, float*, unsigned);
template __global__ void attn_fwd<8>(uint4*, uint4*, uint4*, float*, unsigned);
