// Phase-overlap microbenchmark (Blackwell GPU 1 only). No operator involved.
// Each block runs PAIRS iterations of {memory-only phase; barrier; compute-only phase; barrier}. Per configuration three launches
// are timed: memory phases only (Tm), compute phases only (Tc, compute work tuned so Tc ~ Tm), both (T). With perfect overlap of
// different blocks' phases T ~ max(Tm, Tc); with none, T ~ Tm + Tc. alpha = (Tm + Tc - T) / (Tm + Tc - max(Tm, Tc)).
// Configurations: PAIRS in {1,2,8,32,128}, resident blocks per SM B in {1,2,3,6} (set by dynamic shared memory), WAVES in {1,4}.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char* w){ if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);} }
#define CK(x) check((x),#x)
static constexpr int THREADS=256, LOADS=64, BUF_FLOATS=(32<<20)/4;
__global__ void phase_kernel(const float* __restrict__ buf,float* out,int pairs,int do_mem,int do_cmp,int citers){
  extern __shared__ float dummy[];
  float acc=0.f,a0=1.f,a1=2.f,a2=3.f,a3=4.f,a4=5.f,a5=6.f,a6=7.f,a7=8.f; const float c=1.0000001f,d=0.9999999f;
  for(int p=0;p<pairs;++p){
    if(do_mem){
      size_t off=((size_t)(blockIdx.x*pairs+p)*THREADS*LOADS)%(size_t)BUF_FLOATS;
      #pragma unroll
      for(int j=0;j<LOADS;++j) acc+=__ldcg(&buf[(off+j*THREADS+threadIdx.x)%BUF_FLOATS]);
    }
    __syncthreads();
    if(do_cmp){
      for(int i=0;i<citers;++i){a0=fmaf(a0,c,d);a1=fmaf(a1,c,d);a2=fmaf(a2,c,d);a3=fmaf(a3,c,d);a4=fmaf(a4,c,d);a5=fmaf(a5,c,d);a6=fmaf(a6,c,d);a7=fmaf(a7,c,d);}
    }
    __syncthreads();
  }
  if(threadIdx.x==0&&dummy==nullptr) acc+=1.f;
  out[blockIdx.x*THREADS+threadIdx.x]=acc+a0+a1+a2+a3+a4+a5+a6+a7;
}
static double time_launch(const float* buf,float* out,int blocks,size_t smem,int pairs,int dm,int dc,int ci,cudaStream_t s){
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  phase_kernel<<<blocks,THREADS,smem,s>>>(buf,out,pairs,dm,dc,ci);CK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for(int r=0;r<7;++r){CK(cudaEventRecord(e0,s));for(int k=0;k<10;++k)phase_kernel<<<blocks,THREADS,smem,s>>>(buf,out,pairs,dm,dc,ci);CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));float ms;CK(cudaEventElapsedTime(&ms,e0,e1));v.push_back(ms*1e3/10);}
  std::sort(v.begin(),v.end());CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));return v[v.size()/2];
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
  float* buf;float* out;CK(cudaMalloc(&buf,(size_t)BUF_FLOATS*4));CK(cudaMalloc(&out,(size_t)188*6*4*THREADS*4));CK(cudaMemset(buf,0,(size_t)BUF_FLOATS*4));
  CK(cudaFuncSetAttribute(phase_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,98304));
  const int Bs[4]={1,2,3,6};const size_t sm[4]={96*1024,48*1024,32*1024,16*1024};const int PAIRS[5]={1,2,8,32,128};const int WAVES[2]={1,4};
  for(int bi=0;bi<4;++bi)for(int pi=0;pi<5;++pi)for(int wi=0;wi<2;++wi){
    int B=Bs[bi],pairs=PAIRS[pi],W=WAVES[wi];int blocks=188*B*W;size_t smem=sm[bi];
    int act=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act,phase_kernel,THREADS,smem));
    double Tm=time_launch(buf,out,blocks,smem,pairs,1,0,0,s);
    // tune compute iterations so Tc ~ Tm (bisection on citers)
    int lo=1,hi=1<<16;double Tc=0;int ci=1;
    for(int it=0;it<14;++it){ci=(lo+hi)/2;Tc=time_launch(buf,out,blocks,smem,pairs,0,1,ci,s);if(Tc<Tm)lo=ci;else hi=ci;if(hi-lo<=1)break;}
    ci=hi;Tc=time_launch(buf,out,blocks,smem,pairs,0,1,ci,s);
    double T=time_launch(buf,out,blocks,smem,pairs,1,1,ci,s);
    double full=std::max(Tm,Tc),serial=Tm+Tc;
    printf("{\"mode\":\"overlap\",\"resident_blocks_per_sm\":%d,\"occupancy_api_blocks\":%d,\"pairs\":%d,\"waves\":%d,\"blocks\":%d,\"citers\":%d,\"T_mem_us\":%.3f,\"T_cmp_us\":%.3f,\"T_both_us\":%.3f,\"alpha\":%.4f}\n",
           B,act,pairs,W,blocks,ci,Tm,Tc,T,(serial-T)/(serial-full));
    fflush(stdout);
  }
  return 0;
}
