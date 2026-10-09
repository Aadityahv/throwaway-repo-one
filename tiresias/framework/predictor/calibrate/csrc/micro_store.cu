// Store/triad fixtures of the stream-term calibration (replaces the legacy `triad` and `store` windows that were taken from the repository's operator calibration harness).
// Measures (a) 4-byte stores with word stride 1, 7 and 33 (index = (tid * stride) & mask: stride 1 is fully coalesced, 7 and 33 spread each warp request over more sectors and lines),
// (b) a triad (three float4 reads, one float4 write per thread) in an L2-resident region and a DRAM-sized region, and (c) the per-launch floor of a tiny kernel.
// L2-resident regions are 4 MiB per array (below every supported L2); DRAM regions are 256 MiB per array. Prints one JSON object per measurement.
#include "cal_common.h"
template<int STRIDE> __global__ void store_kernel(float* out,size_t n_threads,unsigned long long mask){
  size_t tid=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(tid>=n_threads)return;
  asm volatile("st.global.cg.f32 [%0], %1;"::"l"(out+((tid*STRIDE)&mask)),"f"(1.0f):"memory");
}
__global__ void triad_kernel(const float4* __restrict__ a,const float4* __restrict__ b,const float4* __restrict__ c,float4* __restrict__ o,size_t n){
  size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  uint4 x,y,z;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(a+i):"memory");
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(y.x),"=r"(y.y),"=r"(y.z),"=r"(y.w):"l"(b+i):"memory");
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(z.x),"=r"(z.y),"=r"(z.z),"=r"(z.w):"l"(c+i):"memory");
  float4 r=make_float4(__uint_as_float(x.x)+__uint_as_float(y.x)*__uint_as_float(z.x),__uint_as_float(x.y)+__uint_as_float(y.y)*__uint_as_float(z.y),
                       __uint_as_float(x.z)+__uint_as_float(y.z)*__uint_as_float(z.z),__uint_as_float(x.w)+__uint_as_float(y.w)*__uint_as_float(z.w));
  asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"::"l"(o+i),"f"(r.x),"f"(r.y),"f"(r.z),"f"(r.w):"memory");
}
template<class F> static double timed_us(F&& launch,int reps,bool graphed,cudaStream_t s){
  cudaEvent_t a,e;CK(cudaEventCreate(&a));CK(cudaEventCreate(&e));std::vector<double> v;launch();CK(cudaStreamSynchronize(s));
  if(graphed){
    cudaGraph_t g;cudaGraphExec_t x;CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));for(int k=0;k<reps;++k)launch();CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiateWithFlags(&x,g,0));
    CK(cudaGraphLaunch(x,s));CK(cudaStreamSynchronize(s));
    for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));CK(cudaGraphLaunch(x,s));CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));v.push_back(ms*1e3/reps);}
    CK(cudaGraphExecDestroy(x));CK(cudaGraphDestroy(g));
  } else {
    for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));for(int k=0;k<reps;++k)launch();CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));v.push_back(ms*1e3/reps);}
  }
  std::sort(v.begin(),v.end());return v[v.size()/2];
}
int main(){
  DeviceInfo D=cal_init_device();(void)D;
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  const int T=256;
  // ---- stores: below (4 MiB region, 2^20 threads, graph of 100) and above (1 GiB region, 2^28 threads, 5 launches)
  for(int tier=0;tier<2;++tier){
    size_t words=tier==0?(1ull<<20):(1ull<<28);float* out;CK(cudaMalloc(&out,words*4));CK(cudaMemset(out,0,words*4));
    int blocks=static_cast<int>((words+T-1)/T);unsigned long long mask=words-1;const int reps=tier==0?100:5;
    auto run=[&](int stride,double us){printf("{\"mode\":\"store\",\"fixture\":\"store\",\"tier\":\"%s\",\"stride\":%d,\"threads\":%zu,\"bytes_written\":%zu,\"us\":%.4f}\n",tier==0?"below":"above",stride,words,words*4,us);fflush(stdout);};
    run(1,timed_us([&]{store_kernel<1><<<blocks,T,0,s>>>(out,words,mask);},reps,tier==0,s));
    if(tier==0){
      run(7,timed_us([&]{store_kernel<7><<<blocks,T,0,s>>>(out,words,mask);},reps,true,s));
      run(33,timed_us([&]{store_kernel<33><<<blocks,T,0,s>>>(out,words,mask);},reps,true,s));
    }
    CK(cudaFree(out));
  }
  // ---- triad: below (4 MiB per array) and above (256 MiB per array)
  for(int tier=0;tier<2;++tier){
    size_t bytes=tier==0?(4ull<<20):(256ull<<20);size_t n=bytes/16;float4 *a,*b,*c,*o;
    CK(cudaMalloc(&a,bytes));CK(cudaMalloc(&b,bytes));CK(cudaMalloc(&c,bytes));CK(cudaMalloc(&o,bytes));CK(cudaMemset(a,0x3f,bytes));CK(cudaMemset(b,0x3f,bytes));CK(cudaMemset(c,0x3f,bytes));CK(cudaMemset(o,0,bytes));
    int blocks=static_cast<int>((n+T-1)/T);const int reps=tier==0?100:5;
    double us=timed_us([&]{triad_kernel<<<blocks,T,0,s>>>(a,b,c,o,n);},reps,tier==0,s);
    printf("{\"mode\":\"store\",\"fixture\":\"triad\",\"tier\":\"%s\",\"bytes_read\":%zu,\"bytes_written\":%zu,\"us\":%.4f}\n",tier==0?"below":"above",3*bytes,bytes,us);fflush(stdout);
    if(tier==0){ // tiny launch floor: the same triad over 12 KiB per array... one block, one wave
      size_t nt=256;double u2=timed_us([&]{triad_kernel<<<1,T,0,s>>>(a,b,c,o,nt);},1000,true,s);
      printf("{\"mode\":\"store\",\"fixture\":\"tiny\",\"tier\":\"below\",\"bytes_read\":%zu,\"bytes_written\":%zu,\"us\":%.4f}\n",3*nt*16,nt*16,u2);fflush(stdout);
    }
    CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(c));CK(cudaFree(o));
  }
  return 0;
}
