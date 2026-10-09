// Same barriers and alpha definition as micro_overlap.cu. No operator code.
// Types: global loads, conflict-free plain shared loads, 8 independent FP32 FMA chains.
#include "micro_assistant_b_guard.h"
#include <algorithm>
#include <cmath>
#include <vector>
static constexpr int THREADS=256,BUF_FLOATS=(32<<20)/4;
__global__ __launch_bounds__(256,6) void phases(const float*buf,float*out,int pairs,int type_a,int type_b,int work_a,int work_b){
 extern __shared__ float sh[];sh[threadIdx.x]=1.f;__syncthreads();
 unsigned sink=0;float a0=float(threadIdx.x+1),a1=float(threadIdx.x+2),a2=float(threadIdx.x+3),a3=float(threadIdx.x+4),a4=float(threadIdx.x+5),a5=float(threadIdx.x+6),a6=float(threadIdx.x+7),a7=float(threadIdx.x+8);
 unsigned ptr=(unsigned)__cvta_generic_to_shared(sh+threadIdx.x);
 for(int p=0;p<pairs;++p){for(int phase=0;phase<2;++phase){int typ=phase?type_b:type_a,work=phase?work_b:work_a;
  if(typ==0){size_t off=((size_t)(blockIdx.x*pairs+p)*THREADS*work)%BUF_FLOATS;
   for(int j=0;j<work;++j)sink^=__float_as_uint(__ldcg(buf+(off+(size_t)j*THREADS+threadIdx.x)%BUF_FLOATS));
  }else if(typ==1){
   for(int j=0;j<work;++j){float x;asm volatile("ld.shared.f32 %0, [%1];":"=f"(x):"r"(ptr):"memory");sink^=__float_as_uint(x);}
  }else if(typ==2){
   for(int j=0;j<work;++j){a0=fmaf(a0,1.0000001f,0.9999999f);a1=fmaf(a1,1.0000001f,0.9999999f);a2=fmaf(a2,1.0000001f,0.9999999f);a3=fmaf(a3,1.0000001f,0.9999999f);a4=fmaf(a4,1.0000001f,0.9999999f);a5=fmaf(a5,1.0000001f,0.9999999f);a6=fmaf(a6,1.0000001f,0.9999999f);a7=fmaf(a7,1.0000001f,0.9999999f);}
  }
  __syncthreads();
 }}out[(size_t)blockIdx.x*THREADS+threadIdx.x]=__uint_as_float(sink)+a0+a1+a2+a3+a4+a5+a6+a7;
}
struct Timing{double us;std::vector<double>raw;};
static Timing time_launch(const float*buf,float*out,int blocks,size_t smem,int pairs,int ta,int tb,int wa,int wb,cudaStream_t s){
 cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
 phases<<<blocks,THREADS,smem,s>>>(buf,out,pairs,ta,tb,wa,wb);CK(cudaGetLastError());CK(cudaStreamSynchronize(s));std::vector<double>v;
 for(int r=0;r<5;++r){CK(cudaEventRecord(e0,s));phases<<<blocks,THREADS,smem,s>>>(buf,out,pairs,ta,tb,wa,wb);CK(cudaGetLastError());CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));float ms;CK(cudaEventElapsedTime(&ms,e0,e1));v.push_back(ms*1000);}
 auto raw=v;std::sort(v.begin(),v.end());fprintf(stderr,"PROBE blocks=%d smem=%zu pairs=%d types=%d,%d work=%d,%d us=%.6f\n",blocks,smem,pairs,ta,tb,wa,wb,v[2]);CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));return {v[2],raw};
}
int main(){auto p=guard();cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));std::vector<float>h(BUF_FLOATS,1.f);float*buf,*out;CK(cudaMalloc(&buf,(size_t)BUF_FLOATS*4));CK(cudaMalloc(&out,(size_t)p.multiProcessorCount*6*4*THREADS*4));CK(cudaMemcpy(buf,h.data(),h.size()*4,cudaMemcpyHostToDevice));
 const char*names[]={"global","shared_load","fp32"};
 for(int B:{1,2,3,6}){size_t smem=smem_for(phases,B);for(int pairs:{8,32})for(int W:{1,4})for(int pair=0;pair<3;++pair){
  int ta=pair==1?1:0,tb=pair==2?1:2,wa=64,wb=64,blocks=p.multiProcessorCount*B*W;
  // Scale synthetic work only until every standalone and combined timing exceeds the frozen floor.
  Timing A,Bt,M;int adjustments=0;
  for(;;){A=time_launch(buf,out,blocks,smem,pairs,ta,-1,wa,0,s);
   if(A.us<75){wa*=2;if(++adjustments>14)return 4;continue;}
   // Match the two standalone phase costs by a fixed eight-step multiplicative search.
   for(int step=0;step<8;++step){Bt=time_launch(buf,out,blocks,smem,pairs,-1,tb,0,wb,s);double ratio=A.us/Bt.us;if(std::abs(ratio-1)<.05)break;wb=std::max(1,std::min(1<<20,(int)std::ceil(wb*std::max(.5,std::min(2.,ratio)))));}
   Bt=time_launch(buf,out,blocks,smem,pairs,-1,tb,0,wb,s);M=time_launch(buf,out,blocks,smem,pairs,ta,tb,wa,wb,s);
   double floor=*std::min_element(A.raw.begin(),A.raw.end());floor=std::min(floor,*std::min_element(Bt.raw.begin(),Bt.raw.end()));floor=std::min(floor,*std::min_element(M.raw.begin(),M.raw.end()));
   if(floor>=50)break;wa*=2;wb*=2;if(++adjustments>14)return 4;
  }
  std::vector<float>got((size_t)blocks*THREADS);CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));int mem_work=(ta<2?wa:0)+(tb<2?wb:0),cmp_work=(ta==2?wa:0)+(tb==2?wb:0);
  std::vector<float>expected(THREADS);
  for(int t=0;t<THREADS;++t){float chains[8];for(int k=0;k<8;++k)chains[k]=float(t+k+1);
   for(int j=0;j<pairs*cmp_work;++j)for(float&x:chains)x=std::fmaf(x,1.0000001f,0.9999999f);
   expected[t]=float((pairs*mem_work)%2);for(float x:chains)expected[t]+=x;
  }
  bool finite=true;for(size_t i=0;i<got.size();++i)if(got[i]!=expected[i%THREADS]){fprintf(stderr,"BAD_OUTPUT index=%zu got=%.9g want=%.9g\n",i,got[i],expected[i%THREADS]);finite=false;break;}
  double alpha=(A.us+Bt.us-M.us)/std::min(A.us,Bt.us);
  printf("{\"mode\":\"overlap_mix\",\"pair\":\"%s+%s\",\"resident_blocks_per_sm\":%d,\"occupancy_api_blocks\":%d,\"smem_bytes\":%zu,\"pairs\":%d,\"waves\":%d,\"blocks\":%d,\"work_a\":%d,\"work_b\":%d,\"T_a_us\":%.6f,\"T_b_us\":%.6f,\"T_both_us\":%.6f,\"alpha\":%.6f,\"exact_output_check\":%s,\"raw_a_us\":[%.6f,%.6f,%.6f,%.6f,%.6f],\"raw_b_us\":[%.6f,%.6f,%.6f,%.6f,%.6f],\"raw_both_us\":[%.6f,%.6f,%.6f,%.6f,%.6f]}\n",names[ta],names[tb],B,B,smem,pairs,W,blocks,wa,wb,A.us,Bt.us,M.us,alpha,finite?"true":"false",A.raw[0],A.raw[1],A.raw[2],A.raw[3],A.raw[4],Bt.raw[0],Bt.raw[1],Bt.raw[2],Bt.raw[3],Bt.raw[4],M.raw[0],M.raw[1],M.raw[2],M.raw[3],M.raw[4]);fflush(stdout);if(!finite)return 3;
 }}CK(cudaFree(buf));CK(cudaFree(out));CK(cudaStreamDestroy(s));return 0;
}
