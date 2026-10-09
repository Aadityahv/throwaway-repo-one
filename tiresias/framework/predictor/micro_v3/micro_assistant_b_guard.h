#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
static constexpr const char* UUID="GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static void check(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",w,cudaGetErrorString(e));exit(2);}}
#define CK(x) check((x),#x)
static cudaDeviceProp guard(){
 const char*vis=getenv("CUDA_VISIBLE_DEVICES");if(!vis||std::string(vis)!=UUID){fprintf(stderr,"REFUSED: CUDA_VISIBLE_DEVICES must be exactly %s\n",UUID);exit(1);}
 int n=0;CK(cudaGetDeviceCount(&n));if(n!=1){fprintf(stderr,"REFUSED: exactly one device required\n");exit(1);}CK(cudaSetDevice(0));cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
 char live[41];auto*u=reinterpret_cast<unsigned char*>(p.uuid.bytes);
 snprintf(live,sizeof live,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
 if(std::string(live)!=UUID||p.major!=12||p.minor!=0||p.multiProcessorCount!=188||p.l2CacheSize!=134217728){fprintf(stderr,"REFUSED: hardware differs from HARDWARE_GROUND_TRUTH.md\n");exit(1);}return p;
}
template<class K>static size_t smem_for(K kernel,int B){
 CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,98304));
 // Select by metadata only, never by timing. Largest capacity with exactly requested occupancy.
 for(size_t bytes=98304;bytes>=8192;bytes-=128){int act=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act,kernel,256,bytes));if(act==B)return bytes;}
 fprintf(stderr,"REFUSED: requested resident blocks unavailable\n");exit(1);
}
