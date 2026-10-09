// Compile unit for the static pipeline: explicit instantiations of every template kernel of the set.
#include "ml_kernels.cuh"
template __global__ void rmsnorm_s<256>(float*, float*, float*, int, float, float);
template __global__ void rmsnorm_v4<128>(float4*, float4*, float4*, int, float, float);
template __global__ void rope_all<8>(float*, float*, float*, float*);
template __global__ void rope_one<8>(float*, float*, float*, float*);
template __global__ void sgemm<64, 64, 16, 4, 4>(float*, float*, float*, unsigned, unsigned);
template __global__ void sgemm<128, 128, 8, 8, 8>(float*, float*, float*, unsigned, unsigned);
