// Portable calibration port of micro_v3/micro_overlap.cu. Phase-overlap microbenchmark: each block runs PAIRS iterations of {memory-only phase; barrier; compute-only phase; barrier}.
// Per configuration three launches are timed: memory phases only (Tm), compute phases only (Tc, tuned so Tc ~ Tm), both (T). alpha = (Tm + Tc - T) / (Tm + Tc - max(Tm, Tc))
// (0 = other blocks' phases fully serialised, 1 = fully overlapped). Resident blocks per SM B are set through dynamic shared memory; the achieved B is recorded from the occupancy API.
// Device facts (SM count, shared memory, thread limits, L2) come from cal_common.h; nothing is hard-coded.
#include "cal_common.h"
static constexpr int THREADS=256, LOADS=64;
template<int LG> __global__ void phase_kernel(const float* __restrict__ buf,float* out,int pairs,int do_mem,int do_cmp,int citers){
  constexpr int buf_floats=1<<LG;   // compile-time constant: identical code generation to the original Blackwell-only program (a runtime mask changes the combined-phase timing by 5 to 11%)
  extern __shared__ float dummy[];
  float acc=0.f,a0=1.f,a1=2.f,a2=3.f,a3=4.f,a4=5.f,a5=6.f,a6=7.f,a7=8.f; const float c=1.0000001f,d=0.9999999f;
  for(int p=0;p<pairs;++p){
    if(do_mem){
      size_t off=((size_t)(blockIdx.x*pairs+p)*THREADS*LOADS)&(size_t)(buf_floats-1);      // buf_floats is a power of two: a mask, not a runtime modulo (which would make the memory phase integer-bound)
      #pragma unroll
      for(int j=0;j<LOADS;++j) acc+=__ldcg(&buf[(off+j*THREADS+threadIdx.x)&(buf_floats-1)]);
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
static int g_lg=23;   // log2 of the buffer size in 32-bit words (set once from the L2 size)
#define DISPATCH_LG(CALL) switch(g_lg){case 20:{CALL(20);}break;case 21:{CALL(21);}break;case 22:{CALL(22);}break;case 23:{CALL(23);}break;default:fprintf(stderr,"unsupported buffer size log2 %d\n",g_lg);exit(2);}
static void launch_k(int blocks,size_t smem,const float* buf,float* out,int pairs,int dm,int dc,int ci,cudaStream_t s){
#define L(LG) phase_kernel<LG><<<blocks,THREADS,smem,s>>>(buf,out,pairs,dm,dc,ci)
  DISPATCH_LG(L)
#undef L
}
static double time_launch(const float* buf,float* out,int blocks,size_t smem,int pairs,int dm,int dc,int ci,int bf,cudaStream_t s){
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  launch_k(blocks,smem,buf,out,pairs,dm,dc,ci,s);CK(cudaStreamSynchronize(s));
  std::vector<double> v;
  for(int r=0;r<7;++r){CK(cudaEventRecord(e0,s));for(int k=0;k<10;++k)launch_k(blocks,smem,buf,out,pairs,dm,dc,ci,s);CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));float ms;CK(cudaEventElapsedTime(&ms,e0,e1));v.push_back(ms*1e3/10);}
  std::sort(v.begin(),v.end());CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));return v[v.size()/2];
}
int main(int argc,char** argv){
  DeviceInfo D=cal_init_device();const bool quick=argc>1&&std::string(argv[1])=="quick";   // quick: three configurations, seconds, for checks only
  cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  const int max_res=std::min(D.max_threads_sm/THREADS,D.max_blocks_sm);
  std::vector<int> Bs{1,2,3,6,max_res};Bs.erase(std::remove_if(Bs.begin(),Bs.end(),[&](int b){return b>max_res||b<1;}),Bs.end());std::sort(Bs.begin(),Bs.end());Bs.erase(std::unique(Bs.begin(),Bs.end()),Bs.end());
  std::vector<int> PAIRS=quick?std::vector<int>{32}:std::vector<int>{1,2,8,32,128};std::vector<int> WAVES=quick?std::vector<int>{1}:std::vector<int>{1,4};
  if(quick){std::vector<int> q;for(int b:Bs)if(b==1||b==2||b==max_res)q.push_back(b);Bs=q;}
  size_t bufb=4ull<<20;while((bufb<<1)<=std::min<size_t>(32ull<<20,D.l2/2))bufb<<=1;const int bf=static_cast<int>(bufb/4);g_lg=0;while((1<<g_lg)<bf)++g_lg;   // largest power of two <= min(32 MiB, L2/2), at least 4 MiB
  float* buf;float* out;CK(cudaMalloc(&buf,static_cast<size_t>(bf)*4));CK(cudaMalloc(&out,static_cast<size_t>(D.sm)*max_res*4*THREADS*4));CK(cudaMemset(buf,0,static_cast<size_t>(bf)*4));
  #define SA(LG) CK(cudaFuncSetAttribute(phase_kernel<LG>,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(D.smem_optin)))
  DISPATCH_LG(SA)
#undef SA
  for(int B:Bs)for(int pairs:PAIRS)for(int W:WAVES){
    size_t per=D.smem_sm/B;per=per>D.smem_reserved_per_block?per-D.smem_reserved_per_block:0;per=(per/1024)*1024;if(per>D.smem_optin)per=(D.smem_optin/1024)*1024;
    int act=0;
#define OC(LG) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act,phase_kernel<LG>,THREADS,per))
    DISPATCH_LG(OC)
#undef OC
    if(act!=B){printf("{\"mode\":\"overlap_skipped\",\"requested_blocks_per_sm\":%d,\"occupancy_api_blocks\":%d,\"smem_per_block\":%zu}\n",B,act,per);continue;}
    int blocks=D.sm*B*W;
    double Tm=time_launch(buf,out,blocks,per,pairs,1,0,0,bf,s);
    int lo=1,hi=1<<16;double Tc=0;int ci=1;
    for(int it=0;it<14;++it){ci=(lo+hi)/2;Tc=time_launch(buf,out,blocks,per,pairs,0,1,ci,bf,s);if(Tc<Tm)lo=ci;else hi=ci;if(hi-lo<=1)break;}
    ci=hi;Tc=time_launch(buf,out,blocks,per,pairs,0,1,ci,bf,s);
    double T=time_launch(buf,out,blocks,per,pairs,1,1,ci,bf,s);
    double full=std::max(Tm,Tc),serial=Tm+Tc;
    printf("{\"mode\":\"overlap\",\"resident_blocks_per_sm\":%d,\"occupancy_api_blocks\":%d,\"pairs\":%d,\"waves\":%d,\"blocks\":%d,\"citers\":%d,\"T_mem_us\":%.3f,\"T_cmp_us\":%.3f,\"T_both_us\":%.3f,\"alpha\":%.4f}\n",
           B,act,pairs,W,blocks,ci,Tm,Tc,T,(serial-T)/(serial-full));
    fflush(stdout);
  }
  return 0;
}
