// Portable calibration port of micro_v3/micro_stream3.cu (the Blackwell-only original is kept untouched as the record). Device identity, SM count, L2 size and shared memory
// come from cal_common.h (live device properties); footprints scale with the L2 size. Output format is unchanged.
// Streaming mix microbenchmark v3 (L2 tiers graph-launched, two sizes) (volatile loads: read-only kernels are kept; L2 tier 8 MiB buffers) (Blackwell GPU 1 only). R float4 reads and W float4 writes per element from distinct buffers.
// Varies read/write mix, footprint tier (L2-resident vs DRAM-sized) and grid size (thread coarsening). No operator involved.
// Prints one JSON object per measurement. Refuses unless exactly the approved device UUID is visible.
#include "cal_common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
template<int R,int W>
__global__ void stream_kernel(const float4* __restrict__ in0,const float4* __restrict__ in1,const float4* __restrict__ in2,
                              float4* __restrict__ out0,float4* __restrict__ out1,size_t n_vec,int per_thread){
  size_t base=(static_cast<size_t>(blockIdx.x)*blockDim.x*per_thread)+threadIdx.x;
  #pragma unroll
  for(int u=0;u<4;++u){ if(u>=per_thread)break;
    size_t i=base+static_cast<size_t>(u)*blockDim.x; if(i>=n_vec)return;
    float4 acc=make_float4(0,0,0,0);
    if(R>=1){float4 v;uint4 vu;asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(vu.x),"=r"(vu.y),"=r"(vu.z),"=r"(vu.w):"l"(in0+i):"memory");v=make_float4(__uint_as_float(vu.x),__uint_as_float(vu.y),__uint_as_float(vu.z),__uint_as_float(vu.w));acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w;}
    if(R>=2){float4 v;uint4 vu;asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(vu.x),"=r"(vu.y),"=r"(vu.z),"=r"(vu.w):"l"(in1+i):"memory");v=make_float4(__uint_as_float(vu.x),__uint_as_float(vu.y),__uint_as_float(vu.z),__uint_as_float(vu.w));acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w;}
    if(R>=3){float4 v;uint4 vu;asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(vu.x),"=r"(vu.y),"=r"(vu.z),"=r"(vu.w):"l"(in2+i):"memory");v=make_float4(__uint_as_float(vu.x),__uint_as_float(vu.y),__uint_as_float(vu.z),__uint_as_float(vu.w));acc.x+=v.x;acc.y+=v.y;acc.z+=v.z;acc.w+=v.w;}
    if(W==0 && (acc.x+acc.y+acc.z+acc.w)==1234567.0f)out0[i]=acc; // never taken: keeps read-only loads live
    if(W>=1)asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"::"l"(out0+i),"f"(acc.x),"f"(acc.y),"f"(acc.z),"f"(acc.w):"memory");
    if(W>=2)asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"::"l"(out1+i),"f"(acc.x),"f"(acc.y),"f"(acc.z),"f"(acc.w):"memory");
  }
}
struct Bufs{ float4 *in[3],*out[2]; };
template<int R,int W> static void run(const char* tier,size_t n_vec,int per_thread,int threads,Bufs b,cudaStream_t s){
  int blocks=static_cast<int>((n_vec+static_cast<size_t>(threads)*per_thread-1)/(static_cast<size_t>(threads)*per_thread));
  auto launch=[&]{stream_kernel<R,W><<<blocks,threads,0,s>>>(b.in[0],b.in[1],b.in[2],b.out[0],b.out[1],n_vec,per_thread);};
  launch();CK(cudaStreamSynchronize(s));
  std::vector<double> us;cudaEvent_t a,e;CK(cudaEventCreate(&a));CK(cudaEventCreate(&e));
  const bool graphed=std::string(tier).rfind("l2",0)==0; // L2 tiers: 200-launch CUDA graph (matches how cells are timed; no host launch limit)
  if(graphed){
    cudaGraph_t g;cudaGraphExec_t x;const int reps=200;
    CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));for(int k=0;k<reps;++k)launch();CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiateWithFlags(&x,g,0));
    CK(cudaGraphLaunch(x,s));CK(cudaStreamSynchronize(s));
    for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));CK(cudaGraphLaunch(x,s));CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));us.push_back(ms*1e3/reps);}
    CK(cudaGraphExecDestroy(x));CK(cudaGraphDestroy(g));
  } else {
    int reps=5;
    for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));for(int k=0;k<reps;++k)launch();CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));us.push_back(ms*1e3/reps);}
  }
  std::sort(us.begin(),us.end());
  printf("{\"mode\":\"stream2\",\"tier\":\"%s\",\"R\":%d,\"W\":%d,\"threads\":%d,\"per_thread\":%d,\"blocks\":%d,\"bytes_read\":%zu,\"bytes_written\":%zu,\"us\":%.4f}\n",
         tier,R,W,threads,per_thread,blocks,static_cast<size_t>(R)*n_vec*16,static_cast<size_t>(W)*n_vec*16,us[us.size()/2]);
}
template<int R,int W> static void sweep(const char* tier,size_t n_vec,Bufs b,cudaStream_t s){
  for(int pt:{1,2,4})run<R,W>(tier,n_vec,pt,256,b,s);
}
int main(){
  DeviceInfo D=cal_init_device();
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  // Tiers: per-buffer element counts. L2-resident: 4 MiB per buffer (<= 20 MiB total). DRAM-sized: 256 MiB per buffer (<= 1.25 GiB total).
  struct T{const char* name;size_t bytes;};
  const size_t unit=std::max<size_t>(1ull<<20,((D.l2*6/128)>>20)<<20); // 6 MiB per buffer at 128 MiB L2 (five buffers stay well inside L2)
  const size_t drb=std::max<size_t>(256ull<<20,(2*D.l2));
  for(T t:{T{"l2a",unit},T{"l2b",2*unit},T{"dram",drb}}){
    Bufs b;size_t nv=t.bytes/16;
    for(int i=0;i<3;++i){CK(cudaMalloc(&b.in[i],t.bytes));CK(cudaMemset(b.in[i],0x3f,t.bytes));}
    for(int i=0;i<2;++i){CK(cudaMalloc(&b.out[i],t.bytes));CK(cudaMemset(b.out[i],0,t.bytes));}
    sweep<1,0>(t.name,nv,b,s);sweep<2,0>(t.name,nv,b,s);sweep<0,1>(t.name,nv,b,s);sweep<1,1>(t.name,nv,b,s);sweep<2,1>(t.name,nv,b,s);sweep<3,1>(t.name,nv,b,s);sweep<1,2>(t.name,nv,b,s);
    for(int i=0;i<3;++i)CK(cudaFree(b.in[i]));for(int i=0;i<2;++i)CK(cudaFree(b.out[i]));
  }
  return 0;
}
