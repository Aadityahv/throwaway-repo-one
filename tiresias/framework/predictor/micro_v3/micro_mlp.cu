// Memory-level-parallelism microbenchmark (Blackwell GPU 1 only): copy kernels where each thread issues M independent
// W-byte loads then M stores. Varies element width W, loads per thread M and resident blocks per SM (via dynamic shared
// memory), in an L2-resident tier and a DRAM-sized tier. Measures bandwidth against bytes in flight per SM. No operator.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char* w){ if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);} }
#define CK(x) check((x),#x)
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
template<class T,int M> static void run(const char* tier,int bps,const T* in,T* out,size_t bytes,cudaStream_t s){
  const int threads=128;size_t n_el=bytes/sizeof(T),n_threads=n_el/M;int blocks=static_cast<int>(n_threads/threads);
  size_t smem=0;if(bps<12){ smem=static_cast<size_t>((100*1024)/bps);smem=(smem/1024)*1024;if(smem>99*1024)smem=99*1024; }
  auto k=copy_mlp<T,M>;CK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(smem)));
  int occ=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,threads,smem));
  k<<<blocks,threads,smem,s>>>(in,out,n_threads);CK(cudaStreamSynchronize(s));
  std::vector<double> us;cudaEvent_t a,e;CK(cudaEventCreate(&a));CK(cudaEventCreate(&e));int reps=bytes>(64u<<20)?5:50;
  for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));for(int i=0;i<reps;++i)k<<<blocks,threads,smem,s>>>(in,out,n_threads);CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));us.push_back(ms*1e3/reps);}
  std::sort(us.begin(),us.end());
  printf("{\"mode\":\"mlp\",\"tier\":\"%s\",\"W\":%d,\"M\":%d,\"requested_blocks_per_sm\":%d,\"achieved_blocks_per_sm\":%d,\"blocks\":%d,\"bytes_each_way\":%zu,\"us\":%.4f}\n",
         tier,static_cast<int>(sizeof(T)),M,bps,occ,blocks,bytes,us[us.size()/2]);
}
template<class T> static void sweep(const char* tier,const T* in,T* out,size_t bytes,cudaStream_t s){
  for(int bps:{1,2,4,8,12}){ run<T,1>(tier,bps,in,out,bytes,s);run<T,2>(tier,bps,in,out,bytes,s);run<T,4>(tier,bps,in,out,bytes,s);run<T,8>(tier,bps,in,out,bytes,s); }
}
int main(){
  const char* vis=getenv("CUDA_VISIBLE_DEVICES");
  if(!vis||std::string(vis)!=UUID){fprintf(stderr,"REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n",UUID);return 1;}
  int n=0;CK(cudaGetDeviceCount(&n));if(n!=1){fprintf(stderr,"REFUSED: exactly one device required\n");return 1;}
  CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
  char live[41];auto* u=reinterpret_cast<unsigned char*>(p.uuid.bytes);
  snprintf(live,sizeof live,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(std::string(live)!=UUID||p.major!=12||p.minor!=0||p.multiProcessorCount!=188){fprintf(stderr,"REFUSED: hardware identity differs\n");return 1;}
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  struct Tier{const char* name;size_t bytes;};
  for(Tier t:{Tier{"l2",16ull<<20},Tier{"dram",256ull<<20}}){
    char *in,*out;CK(cudaMalloc(&in,t.bytes));CK(cudaMalloc(&out,t.bytes));CK(cudaMemset(in,0x3f,t.bytes));CK(cudaMemset(out,0,t.bytes));
    sweep<unsigned>(t.name,reinterpret_cast<unsigned*>(in),reinterpret_cast<unsigned*>(out),t.bytes,s);
    sweep<uint2>(t.name,reinterpret_cast<uint2*>(in),reinterpret_cast<uint2*>(out),t.bytes,s);
    sweep<uint4>(t.name,reinterpret_cast<uint4*>(in),reinterpret_cast<uint4*>(out),t.bytes,s);
    CK(cudaFree(in));CK(cudaFree(out));
  }
  return 0;
}
