// Portable calibration port of micro_v3/micro_v3.cu (chain and reuse modes) (the Blackwell-only original is kept untouched as the record). Device identity, SM count, L2 size and shared memory
// come from cal_common.h (live device properties); footprints scale with the L2 size. Output format is unchanged.
// Static runtime v3 composition microbenchmarks (Blackwell GPU 1 only). No energy, profiler or settings changes.
//   chain : K dependent kernels per launch, 1000 launches in one CUDA graph -> per-launch time vs K (kernel gap)
//   mix   : kernels mixing fp32 / integer / shuffle / MUFU / shared-load per step -> tests max-over-pipes vs sum
//   reuse : repeated passes over a per-block footprint -> L1 capacity and L1-served pass cost
// Prints one JSON object per measurement. Refuses unless exactly the approved device is visible.
#include "cal_common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

__global__ void tiny(unsigned* out){ out[blockIdx.x*blockDim.x+threadIdx.x]=threadIdx.x; }

__global__ void reuse_kernel(const float4* in,float* out,unsigned long long* cycles,int vec_per_block,int passes){
  const float4* base=in+static_cast<size_t>(blockIdx.x)*vec_per_block;
  float s=0.f;
  unsigned long long t0=clock64();
  for(int p=0;p<passes;++p){
    for(int i=threadIdx.x;i<vec_per_block;i+=blockDim.x){
      float4 v;asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(base+i):"memory");
      s+=v.x+v.y+v.z+v.w;
    }
  }
  unsigned long long el=clock64()-t0;
  out[blockIdx.x*blockDim.x+threadIdx.x]=s;
  if(threadIdx.x%32==0)cycles[(blockIdx.x*blockDim.x+threadIdx.x)/32]=el;
}

struct Timer{ cudaEvent_t a,b; Timer(){CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));} };
static double median(std::vector<double> v){ std::sort(v.begin(),v.end());return v[v.size()/2]; }

template<class F> static double graph_time_us(F&& launch_once,int reps,int samples,cudaStream_t s){
  cudaGraph_t g;cudaGraphExec_t e;Timer t;
  CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<reps;++i)launch_once();
  CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiateWithFlags(&e,g,0));
  CK(cudaGraphLaunch(e,s));CK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for(int k=0;k<samples;++k){ CK(cudaEventRecord(t.a,s));CK(cudaGraphLaunch(e,s));CK(cudaEventRecord(t.b,s));CK(cudaStreamSynchronize(s));
    float ms;CK(cudaEventElapsedTime(&ms,t.a,t.b));v.push_back(ms*1e3/reps); }
  CK(cudaGraphExecDestroy(e));CK(cudaGraphDestroy(g));
  return median(v);
}

int main(int argc,char** argv){
  DeviceInfo D=cal_init_device();
  std::string mode=argc>1?argv[1]:"";  // chain: kernel gap by grid size; reuse: per-SM re-read footprint (L1 capacity and bandwidth)
cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  unsigned* out;unsigned long long* cyc;CK(cudaMalloc(&out,static_cast<size_t>(D.sm)*1024*4*4));CK(cudaMalloc(&cyc,static_cast<size_t>(D.sm)*1024/32*8*4));
  if(mode=="chain"){
    for(int grid:{1,D.sm,2048})for(int K=1;K<=4;++K){
      double us=graph_time_us([&]{for(int k=0;k<K;++k)tiny<<<grid,128,0,stream>>>(out);},1000,5,stream);
      printf("{\"mode\":\"chain\",\"grid\":%d,\"kernels_per_launch\":%d,\"us_per_launch\":%.4f}\n",grid,K,us);
    }
  } else if(mode=="reuse"){
    const int threads=256;size_t cap=static_cast<size_t>(D.sm)*8*262144;float4* in;CK(cudaMalloc(&in,cap));std::vector<float> h(cap/4);for(size_t i=0;i<h.size();++i)h[i]=(i%97)*0.01f;CK(cudaMemcpy(in,h.data(),cap,cudaMemcpyHostToDevice));
    for(int bps:{1,2,4})for(int kb:{4,16,32,48,64,96,128,192,256}){
      int blocks=D.sm*bps;size_t bytes=static_cast<size_t>(kb)*1024;if(static_cast<size_t>(blocks)*bytes>cap)continue;int vec=bytes/16;
      reuse_kernel<<<blocks,threads,0,stream>>>(in,reinterpret_cast<float*>(out),cyc,vec,1);CK(cudaStreamSynchronize(stream));
      double t1,t5;
      for(int pass:{1,21}){ std::vector<double> us;for(int r=0;r<25;++r){   /* 25 samples and a 21-pass launch: single-launch event times are 7 to 16 us, so one extra pass is a small difference; the earlier median of 5 samples with a 5-pass launch moved L1 bandwidth by 20% between runs */Timer t;CK(cudaEventRecord(t.a,stream));reuse_kernel<<<blocks,threads,0,stream>>>(in,reinterpret_cast<float*>(out),cyc,vec,pass);CK(cudaEventRecord(t.b,stream));CK(cudaStreamSynchronize(stream));float ms;CK(cudaEventElapsedTime(&ms,t.a,t.b));us.push_back(ms*1e3);} (pass==1?t1:t5)=median(us); }
      printf("{\"mode\":\"reuse\",\"blocks_per_sm\":%d,\"footprint_kb_per_block\":%d,\"threads\":%d,\"us_1pass\":%.4f,\"us_21pass\":%.4f,\"extra_pass_us\":%.4f}\n",bps,kb,threads,t1,t5,(t5-t1)/20);
    }
  } else {fprintf(stderr,"usage: micro_launch chain|reuse\n");return 1;}
  return 0;
}
