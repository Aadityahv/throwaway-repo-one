// Synthetic exact cross-block sharing: all blocks in a group read the same tile once.
// Full degree x footprint x order x resident grid; no operator or profiler.
#include "micro_assistant_b_guard.h"
#include <algorithm>
#include <cmath>
#include <vector>
static constexpr int THREADS=256,TILE_FLOATS=16384; // 64 KiB tile, 64 loads per thread
__global__ void sharing(const float*buf,float*out,int tiles,int degree,int far){
 extern __shared__ float reserve[];
 int tile=far?blockIdx.x%tiles:blockIdx.x/degree;float sum=0;
 #pragma unroll 1
 for(int j=0;j<TILE_FLOATS/THREADS;++j)sum+=__ldcg(buf+(size_t)tile*TILE_FLOATS+j*THREADS+threadIdx.x);
 for(int off=16;off>0;off/=2)sum+=__shfl_down_sync(0xffffffff,sum,off);
 if(threadIdx.x%32==0)out[(size_t)blockIdx.x*8+threadIdx.x/32]=sum;
}
int main(){auto p=guard();cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 const size_t maxBytes=(size_t)p.l2CacheSize*4;std::vector<float>host(maxBytes/4,1.f);float*buf,*out;
 CK(cudaMalloc(&buf,maxBytes));CK(cudaMalloc(&out,maxBytes/65536*32*8*4));
 cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
 for(int B:{1,2,3,6}){size_t smem=smem_for(sharing,B);for(double mult:{1.5,2.,4.})for(int degree:{1,2,4,8,16,32})for(int far:{0,1}){
  size_t bytes=(size_t)(mult*p.l2CacheSize);int tiles=bytes/65536,blocks=tiles*degree;
  std::vector<double>v;std::vector<double>raw;
  for(int r=0;r<6;++r){
   // The same >L2 input is uploaded outside timing; no cache-flush/fill kernel.
   CK(cudaMemcpyAsync(buf,host.data(),bytes,cudaMemcpyHostToDevice,s));CK(cudaStreamSynchronize(s));
   CK(cudaEventRecord(e0,s));sharing<<<blocks,THREADS,smem,s>>>(buf,out,tiles,degree,far);CK(cudaGetLastError());CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));
   float ms;CK(cudaEventElapsedTime(&ms,e0,e1));if(r){v.push_back(ms*1000);raw.push_back(ms*1000);}
  }
  std::vector<float>got((size_t)blocks*8);CK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));bool correct=std::all_of(got.begin(),got.end(),[](float x){return x==2048.f;});
  // Same grid and instructions, 32 MiB cyclic L2 control: estimates the non-DRAM resource floor.
  std::vector<double>control;CK(cudaMemcpyAsync(buf,host.data(),32<<20,cudaMemcpyHostToDevice,s));CK(cudaStreamSynchronize(s));
  for(int r=0;r<6;++r){CK(cudaEventRecord(e0,s));sharing<<<blocks,THREADS,smem,s>>>(buf,out,512,1,1);CK(cudaGetLastError());CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));float ms;CK(cudaEventElapsedTime(&ms,e0,e1));if(r)control.push_back(ms*1000);}
  auto craw=control;std::sort(control.begin(),control.end());
  std::sort(v.begin(),v.end());printf("{\"mode\":\"dram_sharing\",\"resident_blocks_per_sm\":%d,\"smem_bytes\":%zu,\"sharing_degree\":%d,\"footprint_over_l2\":%.1f,\"footprint_bytes\":%zu,\"tile_bytes\":65536,\"blocks\":%d,\"order\":\"%s\",\"compulsory_bytes\":%zu,\"no_credit_bytes\":%zu,\"output_bytes\":%zu,\"time_us\":%.6f,\"l2_control_us\":%.6f,\"l2_control_raw_us\":[%.6f,%.6f,%.6f,%.6f,%.6f],\"correct\":%s,\"raw_us\":[%.6f,%.6f,%.6f,%.6f,%.6f]}\n",B,smem,degree,mult,bytes,blocks,far?"far":"adjacent",bytes,bytes*degree,got.size()*4,v[2],control[2],craw[0],craw[1],craw[2],craw[3],craw[4],correct?"true":"false",raw[0],raw[1],raw[2],raw[3],raw[4]);fflush(stdout);if(!correct)return 3;
 }}CK(cudaFree(buf));CK(cudaFree(out));CK(cudaStreamDestroy(s));return 0;
}
