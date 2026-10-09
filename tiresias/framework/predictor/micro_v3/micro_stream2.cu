// Streaming mix microbenchmark v2 (volatile loads: read-only kernels are kept; L2 tier 8 MiB buffers) (Blackwell GPU 1 only). R float4 reads and W float4 writes per element from distinct buffers.
// Varies read/write mix, footprint tier (L2-resident vs DRAM-sized) and grid size (thread coarsening). No operator involved.
// Prints one JSON object per measurement. Refuses unless exactly the approved device UUID is visible.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char* w){ if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);} }
#define CK(x) check((x),#x)
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
  int reps=n_vec*16<(size_t)1<<26?20:5;
  for(int r=0;r<5;++r){CK(cudaEventRecord(a,s));for(int k=0;k<reps;++k)launch();CK(cudaEventRecord(e,s));CK(cudaStreamSynchronize(s));float ms;CK(cudaEventElapsedTime(&ms,a,e));us.push_back(ms*1e3/reps);}
  std::sort(us.begin(),us.end());
  printf("{\"mode\":\"stream2\",\"tier\":\"%s\",\"R\":%d,\"W\":%d,\"threads\":%d,\"per_thread\":%d,\"blocks\":%d,\"bytes_read\":%zu,\"bytes_written\":%zu,\"us\":%.4f}\n",
         tier,R,W,threads,per_thread,blocks,static_cast<size_t>(R)*n_vec*16,static_cast<size_t>(W)*n_vec*16,us[us.size()/2]);
}
template<int R,int W> static void sweep(const char* tier,size_t n_vec,Bufs b,cudaStream_t s){
  for(int pt:{1,2,4})run<R,W>(tier,n_vec,pt,256,b,s);
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
  // Tiers: per-buffer element counts. L2-resident: 4 MiB per buffer (<= 20 MiB total). DRAM-sized: 256 MiB per buffer (<= 1.25 GiB total).
  struct T{const char* name;size_t bytes;};
  for(T t:{T{"l2",8ull<<20},T{"dram",256ull<<20}}){
    Bufs b;size_t nv=t.bytes/16;
    for(int i=0;i<3;++i){CK(cudaMalloc(&b.in[i],t.bytes));CK(cudaMemset(b.in[i],0x3f,t.bytes));}
    for(int i=0;i<2;++i){CK(cudaMalloc(&b.out[i],t.bytes));CK(cudaMemset(b.out[i],0,t.bytes));}
    sweep<1,0>(t.name,nv,b,s);sweep<2,0>(t.name,nv,b,s);sweep<0,1>(t.name,nv,b,s);sweep<1,1>(t.name,nv,b,s);sweep<2,1>(t.name,nv,b,s);sweep<3,1>(t.name,nv,b,s);sweep<1,2>(t.name,nv,b,s);
    for(int i=0;i<3;++i)CK(cudaFree(b.in[i]));for(int i=0;i<2;++i)CK(cudaFree(b.out[i]));
  }
  return 0;
}
