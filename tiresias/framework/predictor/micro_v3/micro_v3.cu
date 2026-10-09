// Static runtime v3 composition microbenchmarks (Blackwell GPU 1 only). No energy, profiler or settings changes.
//   chain : K dependent kernels per launch, 1000 launches in one CUDA graph -> per-launch time vs K (kernel gap)
//   mix   : kernels mixing fp32 / integer / shuffle / MUFU / shared-load per step -> tests max-over-pipes vs sum
//   reuse : repeated passes over a per-block footprint -> L1 capacity and L1-served pass cost
// Prints one JSON object per measurement. Refuses unless exactly the approved device is visible.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char* w){ if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);} }
#define CK(x) check((x),#x)

__global__ void tiny(unsigned* out){ out[blockIdx.x*blockDim.x+threadIdx.x]=threadIdx.x; }

template<int NF,int NI,int NS,int NM,int NL,int S>
__global__ void mix_kernel(unsigned* out,unsigned long long* cycles,int loops,float a,float b){
  __shared__ unsigned data[1024];
  for(int i=threadIdx.x;i<1024;i+=blockDim.x)data[i]=(i*7+1)&1023;
  __syncthreads();
  unsigned xi[S],ys[S],xl[S];float xf[S],xm[S];
  #pragma unroll
  for(int j=0;j<S;++j){ xi[j]=(threadIdx.x%32)+100*j; ys[j]=(threadIdx.x%32)+7*j; xl[j]=(threadIdx.x*S+j)&1023;
    xf[j]=1.0f+(threadIdx.x%32)/1024.0f+j/4.0f; xm[j]=0.25f+j/8.0f; }
  const unsigned c=__float_as_uint(b);
  unsigned long long t0=clock64();
  #pragma unroll 1
  for(int k=0;k<loops;++k){
    #pragma unroll
    for(int j=0;j<S;++j){
      #pragma unroll
      for(int n=0;n<NF;++n)asm volatile("fma.rn.f32 %0, %0, %1, %2;":"+f"(xf[j]):"f"(a),"f"(-b));
      #pragma unroll
      for(int n=0;n<NI;++n)asm volatile("{ .reg .u32 t; xor.b32 t, %0, %1; add.u32 %0, %0, t; }":"+r"(xi[j]):"r"(c));
      #pragma unroll
      for(int n=0;n<NS;++n)asm volatile("shfl.sync.bfly.b32 %0, %0, 16, 31, 0xffffffff;":"+r"(ys[j]));
      #pragma unroll
      for(int n=0;n<NM;++n){ float v=-xm[j];asm volatile("ex2.approx.ftz.f32 %0, %0;":"+f"(v));xm[j]=v; }
      #pragma unroll
      for(int n=0;n<NL;++n){ unsigned p=static_cast<unsigned>(__cvta_generic_to_shared(&data[0]))+4*(xl[j]&1023);asm volatile("ld.volatile.shared.u32 %0, [%1];":"=r"(xl[j]):"r"(p):"memory"); }
    }
  }
  unsigned long long el=clock64()-t0;
  unsigned acc=0;
  #pragma unroll
  for(int j=0;j<S;++j)acc+=xi[j]+ys[j]+xl[j]+__float_as_uint(xf[j])+__float_as_uint(xm[j]);
  out[blockIdx.x*blockDim.x+threadIdx.x]=acc;
  if(threadIdx.x%32==0)cycles[(blockIdx.x*blockDim.x+threadIdx.x)/32]=el;
}

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
  CK(cudaStreamEndCapture(s,&g));CK(cudaGraphInstantiate(&e,g,0));
  CK(cudaGraphLaunch(e,s));CK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for(int k=0;k<samples;++k){ CK(cudaEventRecord(t.a,s));CK(cudaGraphLaunch(e,s));CK(cudaEventRecord(t.b,s));CK(cudaStreamSynchronize(s));
    float ms;CK(cudaEventElapsedTime(&ms,t.a,t.b));v.push_back(ms*1e3/reps); }
  CK(cudaGraphExecDestroy(e));CK(cudaGraphDestroy(g));
  return median(v);
}

template<int NF,int NI,int NS,int NM,int NL> static void run_mix(const char* name,unsigned* out,unsigned long long* cyc,cudaStream_t s){
  const int blocks=188,threads=1024,loops=509,S=4;const float a=1.0f+ldexpf(1.0f,-20),b=ldexpf(1.0f,-20);
  auto k=mix_kernel<NF,NI,NS,NM,NL,S>;
  CK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,0));
  k<<<blocks,threads,0,s>>>(out,cyc,loops,a,b);CK(cudaStreamSynchronize(s));
  std::vector<double> us,cy;std::vector<unsigned long long> h(blocks*threads/32);
  for(int r=0;r<5;++r){ Timer t;CK(cudaEventRecord(t.a,s));k<<<blocks,threads,0,s>>>(out,cyc,loops,a,b);CK(cudaEventRecord(t.b,s));CK(cudaStreamSynchronize(s));
    float ms;CK(cudaEventElapsedTime(&ms,t.a,t.b));us.push_back(ms*1e3);
    CK(cudaMemcpy(h.data(),cyc,h.size()*8,cudaMemcpyDeviceToHost));double m=0;int w=threads/32;for(int bl=0;bl<blocks;++bl){unsigned long long mx=0;for(int i=0;i<w;++i)mx=std::max(mx,h[bl*w+i]);m+=mx;}cy.push_back(m/blocks); }
  printf("{\"mode\":\"mix\",\"name\":\"%s\",\"NF\":%d,\"NI\":%d,\"NS\":%d,\"NM\":%d,\"NL\":%d,\"streams\":%d,\"loops\":%d,\"warps_per_sm\":%d,\"event_us\":%.4f,\"last_warp_cycles\":%.1f}\n",name,NF,NI,NS,NM,NL,S,loops,threads/32,median(us),median(cy));
}
#define MIX(NAME,NF,NI,NS,NM,NL) run_mix<NF,NI,NS,NM,NL>(NAME,out,cyc,stream)

int main(int argc,char** argv){
  const char* vis=getenv("CUDA_VISIBLE_DEVICES");
  if(!vis||std::string(vis)!=UUID){fprintf(stderr,"REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n",UUID);return 1;}
  int n=0;CK(cudaGetDeviceCount(&n));if(n!=1){fprintf(stderr,"REFUSED: exactly one device required\n");return 1;}
  CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
  char live[41];auto* u=reinterpret_cast<unsigned char*>(p.uuid.bytes);
  snprintf(live,sizeof live,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if(std::string(live)!=UUID||p.major!=12||p.minor!=0||p.multiProcessorCount!=188){fprintf(stderr,"REFUSED: hardware identity differs\n");return 1;}
  std::string mode=argc>1?argv[1]:"";cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  unsigned* out;unsigned long long* cyc;CK(cudaMalloc(&out,188*1024*4*4));CK(cudaMalloc(&cyc,188*1024/32*8*4));
  if(mode=="chain"){
    for(int grid:{1,188,2048})for(int K=1;K<=4;++K){
      double us=graph_time_us([&]{for(int k=0;k<K;++k)tiny<<<grid,128,0,stream>>>(out);},1000,5,stream);
      printf("{\"mode\":\"chain\",\"grid\":%d,\"kernels_per_launch\":%d,\"us_per_launch\":%.4f}\n",grid,K,us);
    }
  } else if(mode=="mix"){
    MIX("fp32",4,0,0,0,0);MIX("fp32_int",4,4,0,0,0);MIX("fp32_shfl",8,0,1,0,0);MIX("fp32_mufu",8,0,0,1,0);MIX("shfl_mufu",0,0,1,1,0);
    MIX("fp32_lds",4,0,0,0,1);MIX("lds_only",0,0,0,0,1);MIX("shfl_only",0,0,1,0,0);MIX("mufu_only",0,0,0,1,0);MIX("int_only",0,4,0,0,0);MIX("int_lds",0,4,0,0,1);MIX("all",4,2,1,1,1);
  } else if(mode=="reuse"){
    const int threads=256;size_t cap=188ull*8*262144;float4* in;CK(cudaMalloc(&in,cap));std::vector<float> h(cap/4);for(size_t i=0;i<h.size();++i)h[i]=(i%97)*0.01f;CK(cudaMemcpy(in,h.data(),cap,cudaMemcpyHostToDevice));
    for(int bps:{1,2,4})for(int kb:{4,16,32,48,64,96,128,192,256}){
      int blocks=188*bps;size_t bytes=static_cast<size_t>(kb)*1024;if(static_cast<size_t>(blocks)*bytes>cap)continue;int vec=bytes/16;
      reuse_kernel<<<blocks,threads,0,stream>>>(in,reinterpret_cast<float*>(out),cyc,vec,1);CK(cudaStreamSynchronize(stream));
      double t1,t5;
      for(int pass:{1,5}){ std::vector<double> us;for(int r=0;r<5;++r){Timer t;CK(cudaEventRecord(t.a,stream));reuse_kernel<<<blocks,threads,0,stream>>>(in,reinterpret_cast<float*>(out),cyc,vec,pass);CK(cudaEventRecord(t.b,stream));CK(cudaStreamSynchronize(stream));float ms;CK(cudaEventElapsedTime(&ms,t.a,t.b));us.push_back(ms*1e3);} (pass==1?t1:t5)=median(us); }
      printf("{\"mode\":\"reuse\",\"blocks_per_sm\":%d,\"footprint_kb_per_block\":%d,\"threads\":%d,\"us_1pass\":%.4f,\"us_5pass\":%.4f,\"extra_pass_us\":%.4f}\n",bps,kb,threads,t1,t5,(t5-t1)/4);
    }
  } else {fprintf(stderr,"usage: micro_v3 chain|mix|reuse\n");return 1;}
  return 0;
}
