// Plain (non-volatile) shared-memory throughput microbenchmark (Blackwell GPU 1 only), companion of micro_smem.cu which used ld.volatile.
// Measures cycles per warp-level shared load for plain ld.shared of 32/64/128 bits, with and without an interleaved FFMA per load.
// (original header) Shared-memory bank-conflict microbenchmark: cycles per warp-level shared load/store versus the
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

// WIDTH: 32, 64 or 128; STRIDE in 32-bit words between lanes; FMA: 0 or 1 interleaved dependent-free FFMA per load.
template<int WIDTH,int STRIDE,int FMA,int S>
__global__ void smem_kernel(float* out,unsigned long long* cycles,int loops){
  __shared__ __align__(16) unsigned data[4096];
  for(int i=threadIdx.x;i<4096;i+=blockDim.x)data[i]=i;
  __syncthreads();
  const unsigned lane=threadIdx.x%32;
  const unsigned base=static_cast<unsigned>(__cvta_generic_to_shared(&data[0]));
  float a0=1.f,a1=2.f,a2=3.f,a3=4.f,f=1.0001f,g=0.5f;
  unsigned long long t0=clock64();
  #pragma unroll 1
  for(int k=0;k<loops;++k){
    #pragma unroll
    for(int j=0;j<S;++j){
      unsigned idx=(lane*STRIDE+((k*S+j)*WIDTH/32*0)+j*4+ (k&1)*128)&4095u; if(WIDTH==64)idx&=~1u; if(WIDTH==128)idx&=~3u;
      unsigned p=base+4*idx;
      if(WIDTH==32){unsigned v;asm("ld.shared.u32 %0, [%1];":"=r"(v):"r"(p));a0+=__uint_as_float(v);}
      if(WIDTH==64){unsigned v0,v1;asm("ld.shared.v2.u32 {%0,%1}, [%2];":"=r"(v0),"=r"(v1):"r"(p));a0+=__uint_as_float(v0);a1+=__uint_as_float(v1);}
      if(WIDTH==128){unsigned v0,v1,v2,v3;asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3):"r"(p));a0+=__uint_as_float(v0);a1+=__uint_as_float(v1);a2+=__uint_as_float(v2);a3+=__uint_as_float(v3);}
      if(FMA){f=fmaf(f,g,a0);g=fmaf(g,f,a1);}
    }
  }
  unsigned long long el=clock64()-t0;
  out[blockIdx.x*blockDim.x+threadIdx.x]=a0+a1+a2+a3+f+g;
  if(threadIdx.x%32==0)cycles[(blockIdx.x*blockDim.x+threadIdx.x)/32]=el;
}
template<int WIDTH,int STRIDE,int FMA> static void run(float* out,unsigned long long* cyc,cudaStream_t s){
  const int blocks=188,threads=1024,loops=509,S=4;auto k=smem_kernel<WIDTH,STRIDE,FMA,S>;
  k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));
  std::vector<double> cy;std::vector<unsigned long long> h(blocks*threads/32);
  for(int r=0;r<5;++r){k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(h.data(),cyc,h.size()*8,cudaMemcpyDeviceToHost));
    double m=0;int w=threads/32;for(int b=0;b<blocks;++b){unsigned long long mx=0;for(int i=0;i<w;++i)mx=std::max(mx,h[b*w+i]);m+=mx;}cy.push_back(m/blocks);}
  std::sort(cy.begin(),cy.end());
  printf("{\"mode\":\"smem_plain\",\"width\":%d,\"stride_words\":%d,\"ffma_per_load\":%d,\"streams\":%d,\"loops\":%d,\"warps_per_sm\":%d,\"last_warp_cycles\":%.1f,\"cycles_per_warp_instruction_per_sm\":%.4f}\n",
         WIDTH,STRIDE,FMA,S,loops,threads/32,cy[cy.size()/2],cy[cy.size()/2]/(static_cast<double>(loops)*S*(threads/32)));
}
int main(){
  const char* vis=getenv("CUDA_VISIBLE_DEVICES");
  if(!vis||std::string(vis)!=UUID){fprintf(stderr,"REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n",UUID);return 1;}
  int n=0;CK(cudaGetDeviceCount(&n));if(n!=1){fprintf(stderr,"REFUSED: exactly one device required\n");return 1;}
  CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
  char live[41];auto* u=reinterpret_cast<unsigned char*>(p.uuid.bytes);
  snprintf(live,sizeof live,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(std::string(live)!=UUID||p.major!=12||p.minor!=0||p.multiProcessorCount!=188){fprintf(stderr,"REFUSED: hardware identity differs\n");return 1;}
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));float* out;unsigned long long* cyc;CK(cudaMalloc(&out,188*1024*4));CK(cudaMalloc(&cyc,188*32*8));
  run<32,0,0>(out,cyc,s);run<32,1,0>(out,cyc,s);run<32,2,0>(out,cyc,s);run<32,4,0>(out,cyc,s);run<32,32,0>(out,cyc,s);run<32,33,0>(out,cyc,s);
  run<32,1,1>(out,cyc,s);run<32,0,1>(out,cyc,s);
  run<64,2,0>(out,cyc,s);run<64,4,0>(out,cyc,s);run<64,2,1>(out,cyc,s);
  run<128,4,0>(out,cyc,s);run<128,8,0>(out,cyc,s);run<128,4,1>(out,cyc,s);
  return 0;
}
