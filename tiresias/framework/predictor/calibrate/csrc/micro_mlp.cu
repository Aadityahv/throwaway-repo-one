// Portable calibration port of micro_v3/micro_mlp.cu (the Blackwell-only original is kept untouched as the record). Device identity, SM count, L2 size and shared memory
// come from cal_common.h (live device properties); footprints scale with the L2 size. Output format is unchanged.
// Memory-level-parallelism microbenchmark (Blackwell GPU 1 only): copy kernels where each thread issues M independent
// W-byte loads then M stores. Varies element width W, loads per thread M and resident blocks per SM (via dynamic shared
// memory), in an L2-resident tier and a DRAM-sized tier. Measures bandwidth against bytes in flight per SM. No operator.
#include "cal_common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
template<class T> struct Ty;
template<> struct Ty<unsigned>{ static __device__ __forceinline__ unsigned ld(const unsigned* p){return __ldcg(p);} static __device__ __forceinline__ void st(unsigned* p,unsigned v){__stcg(p,v);} };
template<> struct Ty<uint2>{ static __device__ __forceinline__ uint2 ld(const uint2* p){return __ldcg(p);} static __device__ __forceinline__ void st(uint2* p,uint2 v){__stcg(p,v);} };
template<> struct Ty<uint4>{ static __device__ __forceinline__ uint4 ld(const uint4* p){return __ldcg(p);} static __device__ __forceinline__ void st(uint4* p,uint4 v){__stcg(p,v);} };
template<class T,int M>
__global__ void copy_mlp(const T* __restrict__ in,T* __restrict__ out,size_t n_threads){
  extern __shared__ char pad[];
  size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(tid>=n_threads)return;
  T v[M];
  #pragma unroll
  for(int k=0;k<M;++k)v[k]=Ty<T>::ld(in+tid+static_cast<size_t>(k)*n_threads);
  #pragma unroll
  for(int k=0;k<M;++k)Ty<T>::st(out+tid+static_cast<size_t>(k)*n_threads,v[k]);
}
template<class T,int M> static void run(const DeviceInfo& D,const char* tier,int bps,const T* in,T* out,size_t bytes,cudaStream_t s){
  const int threads=128;size_t n_el=bytes/sizeof(T),n_threads=n_el/M;int blocks=static_cast<int>(n_threads/threads);
  const int max_res=std::min(D.max_threads_sm/threads,D.max_blocks_sm);size_t smem=0;if(bps<max_res){ size_t usable=D.smem_sm>D.smem_reserved_per_block*bps?D.smem_sm-D.smem_reserved_per_block*bps:D.smem_sm;smem=usable/bps;smem=(smem/1024)*1024;if(smem>D.smem_optin)smem=(D.smem_optin/1024)*1024; }
  auto k=copy_mlp<T,M>;CK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(smem)));
  int occ=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,threads,smem));
  k<<<blocks,threads,smem,s>>>(in,out,n_threads);CK(cudaStreamSynchronize(s));
  std::vector<double> us;cudaEvent_t a,e;CK(cudaEventCreate(&a));CK(cudaEventCreate(&e));int reps=bytes>(64u<<20)?5:50;
  for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));for(int i=0;i<reps;++i)k<<<blocks,threads,smem,s>>>(in,out,n_threads);CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));us.push_back(ms*1e3/reps);}
  std::sort(us.begin(),us.end());
  printf("{\"mode\":\"mlp\",\"tier\":\"%s\",\"W\":%d,\"M\":%d,\"requested_blocks_per_sm\":%d,\"achieved_blocks_per_sm\":%d,\"blocks\":%d,\"bytes_each_way\":%zu,\"us\":%.4f}\n",
         tier,static_cast<int>(sizeof(T)),M,bps,occ,blocks,bytes,us[us.size()/2]);
}
template<class T> static void sweep(const DeviceInfo& D,const char* tier,const T* in,T* out,size_t bytes,cudaStream_t s){
  const int max_res=std::min(D.max_threads_sm/128,D.max_blocks_sm);std::vector<int> bs{1,2,4,8,max_res};bs.erase(std::remove_if(bs.begin(),bs.end(),[&](int b){return b>max_res;}),bs.end());std::sort(bs.begin(),bs.end());bs.erase(std::unique(bs.begin(),bs.end()),bs.end());
  for(int bps:bs){ run<T,1>(D,tier,bps,in,out,bytes,s);run<T,2>(D,tier,bps,in,out,bytes,s);run<T,4>(D,tier,bps,in,out,bytes,s);run<T,8>(D,tier,bps,in,out,bytes,s); }
}
int main(){
  DeviceInfo D=cal_init_device();
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  struct Tier{const char* name;size_t bytes;};
  const size_t l2b=std::max<size_t>(4ull<<20,((D.l2/8)>>20)<<20);const size_t drb=std::max<size_t>(256ull<<20,(4*D.l2)); // L2 tier: one eighth of L2 per buffer; DRAM tier: >= 4x L2 per buffer
  for(Tier t:{Tier{"l2",l2b},Tier{"dram",drb}}){
    char *in,*out;CK(cudaMalloc(&in,t.bytes));CK(cudaMalloc(&out,t.bytes));CK(cudaMemset(in,0x3f,t.bytes));CK(cudaMemset(out,0,t.bytes));
    sweep<unsigned>(D,t.name,reinterpret_cast<unsigned*>(in),reinterpret_cast<unsigned*>(out),t.bytes,s);
    sweep<uint2>(D,t.name,reinterpret_cast<uint2*>(in),reinterpret_cast<uint2*>(out),t.bytes,s);
    sweep<uint4>(D,t.name,reinterpret_cast<uint4*>(in),reinterpret_cast<uint4*>(out),t.bytes,s);
    CK(cudaFree(in));CK(cudaFree(out));
  }
  return 0;
}
