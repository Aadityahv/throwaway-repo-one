#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "transpose.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z4copyPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z13copySharedMemPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z14transposeNaivePfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z18transposeCoalescedPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z24transposeNoBankConflictsPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z17transposeDiagonalPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z20transposeFineGrainedPfS_ii(float *, float *, int, int);
extern __attribute__((visibility("hidden"))) void __device_stub__Z22transposeCoarseGrainedPfS_ii(float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z4copyPfS_ii(float *__par0, float *__par1, int __par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(float *, float *, int, int))copy)));}
# 81 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void copy( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 82 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z4copyPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 91 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z13copySharedMemPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))copySharedMem))); }
# 93 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void copySharedMem( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 94 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z13copySharedMemPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 119 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z14transposeNaivePfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeNaive))); }
# 126 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeNaive( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 127 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z14transposeNaivePfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 137 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z18transposeCoalescedPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeCoalesced))); }
# 141 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeCoalesced( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 142 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z18transposeCoalescedPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 164 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z24transposeNoBankConflictsPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeNoBankConflicts))); }
# 168 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeNoBankConflicts( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 169 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z24transposeNoBankConflictsPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 191 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z17transposeDiagonalPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeDiagonal))); }
# 205 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeDiagonal( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 206 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z17transposeDiagonalPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 244 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z20transposeFineGrainedPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeFineGrained))); }
# 255 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeFineGrained( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 256 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z20transposeFineGrainedPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 274 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
__attribute__((visibility("hidden"))) void __device_stub__Z22transposeCoarseGrainedPfS_ii( float *__par0,  float *__par1,  int __par2,  int __par3) {  __cudaLaunchPrologue(4); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int, int))transposeCoarseGrained))); }
# 276 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
void transposeCoarseGrained( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 277 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
{__device_stub__Z22transposeCoarseGrainedPfS_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 299 "/home/user/cudasamples_pinned_5443602d/cpp/6_Performance/transpose/transpose.cu"
}
# 1 "/home/user/ipdps_cuda_compile_51afb2f3/output/nvcc/compile_events/0003_nvcc/intermediates/transpose.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T35) {  __nv_dummy_param_ref(__T35); __nv_save_fatbinhandle_for_managed_rt(__T35); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeCoarseGrained), _Z22transposeCoarseGrainedPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeFineGrained), _Z20transposeFineGrainedPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeDiagonal), _Z17transposeDiagonalPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeNoBankConflicts), _Z24transposeNoBankConflictsPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeCoalesced), _Z18transposeCoalescedPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))transposeNaive), _Z14transposeNaivePfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))copySharedMem), _Z13copySharedMemPfS_ii, (-1)); __cudaRegisterEntry(__T35, ((void ( *)(float *, float *, int, int))copy), _Z4copyPfS_ii, (-1)); __cudaRegisterVariable(__T35, __shadow_var(_ZN43_INTERNAL_e301b184_12_transpose_cu_1e3911aa4cuda3std3__45__cpo9iter_swapE,::cuda::std::__4::__cpo::iter_swap), 0, 1UL, 0, 0); __cudaRegisterVariable(__T35, __shadow_var(_ZN43_INTERNAL_e301b184_12_transpose_cu_1e3911aa4cuda3std9execution3__43seqE,::cuda::std::execution::__4::seq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T35, __shadow_var(_ZN43_INTERNAL_e301b184_12_transpose_cu_1e3911aa4cuda3std9execution3__43parE,::cuda::std::execution::__4::par), 0, 1UL, 0, 0); __cudaRegisterVariable(__T35, __shadow_var(_ZN43_INTERNAL_e301b184_12_transpose_cu_1e3911aa4cuda3std9execution3__49par_unseqE,::cuda::std::execution::__4::par_unseq), 0, 1UL, 0, 0); __cudaRegisterVariable(__T35, __shadow_var(_ZN43_INTERNAL_e301b184_12_transpose_cu_1e3911aa4cuda3std9execution3__45unseqE,::cuda::std::execution::__4::unseq), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
