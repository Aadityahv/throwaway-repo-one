// Shared-memory bank-conflict microbenchmark (Blackwell GPU 1 only): cycles per warp-level shared load/store versus the
// word stride between consecutive lanes (stride 1 = conflict-free, stride 2^k = 2^k-way conflict, odd strides conflict-free,
// stride 0 = broadcast). No operator involved. Prints one JSON object per measurement.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char* w){ if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);} }
#define CK(x) check((x),#x)
template<int STRIDE,int STORE,int S>
__global__ void smem_kernel(unsigned* out,unsigned long long* cycles,int loops){
  __shared__ unsigned data[4096];
  for(int i=threadIdx.x;i<4096;i+=blockDim.x)data[i]=i;
  __syncthreads();
  const unsigned lane=threadIdx.x%32;unsigned acc=0;
  const unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(&data[0]));
  unsigned long long t0=clock64();
  #pragma unroll 1
  for(int k=0;k<loops;++k){
    #pragma unroll
    for(int j=0;j<S;++j){
      unsigned idx=(lane*STRIDE+j*1)&4095u; unsigned p=base+4*idx; unsigned v;
      if(STORE)asm volatile("st.volatile.shared.u32 [%0], %1;"::"r"(p),"r"(lane+j):"memory");
      else {asm volatile("ld.volatile.shared.u32 %0, [%1];":"=r"(v):"r"(p):"memory");acc+=v;}
    }
  }
  unsigned long long el=clock64()-t0;
  out[blockIdx.x*blockDim.x+threadIdx.x]=acc;
  if(threadIdx.x%32==0)cycles[(blockIdx.x*blockDim.x+threadIdx.x)/32]=el;
}
template<int STRIDE,int STORE> static void run(unsigned* out,unsigned long long* cyc,cudaStream_t s){
  const int blocks=188,threads=1024,loops=509,S=4;auto k=smem_kernel<STRIDE,STORE,S>;
  k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));
  std::vector<double> cy;std::vector<unsigned long long> h(blocks*threads/32);
  for(int r=0;r<5;++r){k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(h.data(),cyc,h.size()*8,cudaMemcpyDeviceToHost));
    double m=0;int w=threads/32;for(int b=0;b<blocks;++b){unsigned long long mx=0;for(int i=0;i<w;++i)mx=std::max(mx,h[b*w+i]);m+=mx;}cy.push_back(m/blocks);}
  std::sort(cy.begin(),cy.end());
  printf("{\"mode\":\"smem\",\"stride\":%d,\"store\":%d,\"streams\":%d,\"loops\":%d,\"warps_per_sm\":%d,\"last_warp_cycles\":%.1f,\"cycles_per_warp_instruction_per_sm\":%.4f}\n",
         STRIDE,STORE,S,loops,threads/32,cy[cy.size()/2],cy[cy.size()/2]/(static_cast<double>(loops)*S*(threads/32)));
}
#define BOTH(ST) run<ST,0>(out,cyc,s);run<ST,1>(out,cyc,s)
int main(){
  const char* vis=getenv("CUDA_VISIBLE_DEVICES");
  if(!vis||std::string(vis)!=UUID){fprintf(stderr,"REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n",UUID);return 1;}
  int n=0;CK(cudaGetDeviceCount(&n));if(n!=1){fprintf(stderr,"REFUSED: exactly one device required\n");return 1;}
  CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
  char live[41];auto* u=reinterpret_cast<unsigned char*>(p.uuid.bytes);
  snprintf(live,sizeof live,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(std::string(live)!=UUID||p.major!=12||p.minor!=0||p.multiProcessorCount!=188){fprintf(stderr,"REFUSED: hardware identity differs\n");return 1;}
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));unsigned* out;unsigned long long* cyc;CK(cudaMalloc(&out,188*1024*4));CK(cudaMalloc(&cyc,188*32*8));
  BOTH(0);BOTH(1);BOTH(2);BOTH(3);BOTH(4);BOTH(5);BOTH(8);BOTH(16);BOTH(32);BOTH(33);
  return 0;
}
