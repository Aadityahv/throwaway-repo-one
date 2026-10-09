// Portable calibration port of micro_v3/micro_smem.cu and micro_smem_plain.cu (modes volatile and plain) (the Blackwell-only original is kept untouched as the record). Device identity, SM count, L2 size and shared memory
// come from cal_common.h (live device properties); footprints scale with the L2 size. Output format is unchanged.
// Shared-memory bank-conflict microbenchmark (Blackwell GPU 1 only): cycles per warp-level shared load/store versus the
// word stride between consecutive lanes (stride 1 = conflict-free, stride 2^k = 2^k-way conflict, odd strides conflict-free,
// stride 0 = broadcast). No operator involved. Prints one JSON object per measurement.
#include "cal_common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
template<int STRIDE,int STORE,int S>
__global__ void smem_vol_kernel(unsigned* out,unsigned long long* cycles,int loops){
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
template<int STRIDE,int STORE> static void run_vol(const DeviceInfo& D,unsigned* out,unsigned long long* cyc,cudaStream_t s){
  const int blocks=D.sm,threads=1024,loops=509,S=4;auto k=smem_vol_kernel<STRIDE,STORE,S>;
  k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));
  std::vector<double> cy;std::vector<unsigned long long> h(blocks*threads/32);
  for(int r=0;r<5;++r){k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(h.data(),cyc,h.size()*8,cudaMemcpyDeviceToHost));
    double m=0;int w=threads/32;for(int b=0;b<blocks;++b){unsigned long long mx=0;for(int i=0;i<w;++i)mx=std::max(mx,h[b*w+i]);m+=mx;}cy.push_back(m/blocks);}
  std::sort(cy.begin(),cy.end());
  printf("{\"mode\":\"smem\",\"stride\":%d,\"store\":%d,\"streams\":%d,\"loops\":%d,\"warps_per_sm\":%d,\"last_warp_cycles\":%.1f,\"cycles_per_warp_instruction_per_sm\":%.4f}\n",
         STRIDE,STORE,S,loops,threads/32,cy[cy.size()/2],cy[cy.size()/2]/(static_cast<double>(loops)*S*(threads/32)));
}

template<int WIDTH,int STRIDE,int FMA,int S>
__global__ void smem_plain_kernel(float* out,unsigned long long* cycles,int loops){
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
template<int WIDTH,int STRIDE,int FMA> static void run_plain(const DeviceInfo& D,float* out,unsigned long long* cyc,cudaStream_t s){
  const int blocks=D.sm,threads=1024,loops=509,S=4;auto k=smem_plain_kernel<WIDTH,STRIDE,FMA,S>;
  k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));
  std::vector<double> cy;std::vector<unsigned long long> h(blocks*threads/32);
  for(int r=0;r<5;++r){k<<<blocks,threads,0,s>>>(out,cyc,loops);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(h.data(),cyc,h.size()*8,cudaMemcpyDeviceToHost));
    double m=0;int w=threads/32;for(int b=0;b<blocks;++b){unsigned long long mx=0;for(int i=0;i<w;++i)mx=std::max(mx,h[b*w+i]);m+=mx;}cy.push_back(m/blocks);}
  std::sort(cy.begin(),cy.end());
  printf("{\"mode\":\"smem_plain\",\"width\":%d,\"stride_words\":%d,\"ffma_per_load\":%d,\"streams\":%d,\"loops\":%d,\"warps_per_sm\":%d,\"last_warp_cycles\":%.1f,\"cycles_per_warp_instruction_per_sm\":%.4f}\n",
         WIDTH,STRIDE,FMA,S,loops,threads/32,cy[cy.size()/2],cy[cy.size()/2]/(static_cast<double>(loops)*S*(threads/32)));
}
int main(int argc,char** argv){
  DeviceInfo D=cal_init_device();std::string mode=argc>1?argv[1]:"";
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  if(mode=="volatile"){
    unsigned* out;unsigned long long* cyc;CK(cudaMalloc(&out,static_cast<size_t>(D.sm)*1024*4));CK(cudaMalloc(&cyc,static_cast<size_t>(D.sm)*32*8));
    #define BOTHV(ST) run_vol<ST,0>(D,out,cyc,s);run_vol<ST,1>(D,out,cyc,s)
    BOTHV(0);BOTHV(1);BOTHV(2);BOTHV(3);BOTHV(4);BOTHV(5);BOTHV(8);BOTHV(16);BOTHV(32);BOTHV(33);
  } else if(mode=="plain"){
    float* out;unsigned long long* cyc;CK(cudaMalloc(&out,static_cast<size_t>(D.sm)*1024*4));CK(cudaMalloc(&cyc,static_cast<size_t>(D.sm)*32*8));
    run_plain<32,0,0>(D,out,cyc,s);run_plain<32,1,0>(D,out,cyc,s);run_plain<32,2,0>(D,out,cyc,s);run_plain<32,4,0>(D,out,cyc,s);run_plain<32,32,0>(D,out,cyc,s);run_plain<32,33,0>(D,out,cyc,s);
    run_plain<32,1,1>(D,out,cyc,s);run_plain<32,0,1>(D,out,cyc,s);
    run_plain<64,2,0>(D,out,cyc,s);run_plain<64,4,0>(D,out,cyc,s);run_plain<64,2,1>(D,out,cyc,s);
    run_plain<128,4,0>(D,out,cyc,s);run_plain<128,8,0>(D,out,cyc,s);run_plain<128,4,1>(D,out,cyc,s);
  } else {fprintf(stderr,"usage: micro_smem volatile|plain\n");return 1;}
  return 0;
}
