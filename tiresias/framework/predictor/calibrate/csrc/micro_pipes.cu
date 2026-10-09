// Portable calibration port of repair/next_iteration/benchmark.cu (kept untouched as the record). Isolated pipe, latency, barrier and pointer-chase timings;
// no energy, profiler or system configuration. The reviewed-packet authorization of the original is replaced by the allow-list guard of cal_common.h; the
// geometry limits now use the live SM count; the pointer-chase table size may be any multiple of 4096 bytes between 64 KiB and 8 MiB (the orchestrator sizes
// the tiers from the L2 size).
#include "cal_common.h"
#include "oracle.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }  // throws; the exception handler in main reports REFUSED
static int integer(const char* p) { std::string s(p);size_t end=0;int n=std::stoi(s,&end);if(end!=s.size() || n<0)throw std::runtime_error("invalid integer argument");return n; }
__device__ __forceinline__ unsigned smid() { unsigned x;asm volatile("mov.u32 %0, %%smid;":"=r"(x));return x; }
__device__ __forceinline__ unsigned shared_pointer(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

template<int F,int S,int U> __device__ __forceinline__ void one_step(unsigned (&x)[S],unsigned shared,float a,float b) {
#pragma unroll
  for(int j=0;j<S;++j) {
    if constexpr(F==0)asm volatile("bar.sync 0;":::"memory"); // block barrier per step; loop overhead comes from the unroll slope (M445 redesign)
    else if constexpr(F==1)asm volatile("{ .reg .u32 t; xor.b32 t, %0, %1; add.u32 %0, %0, t; }":"+r"(x[j]):"r"(__float_as_uint(b))); // x = x + (x ^ c): two dependent integer ops per step that the compiler cannot merge (M445)
    else if constexpr(F==2) { float v=__uint_as_float(x[j]);asm volatile("add.rn.f32 %0, %0, %1;":"+f"(v):"f"(b));x[j]=__float_as_uint(v); }
    else if constexpr(F==3) { float v=__uint_as_float(x[j]);asm volatile("fma.rn.f32 %0, %0, %1, %2;":"+f"(v):"f"(a),"f"(-b));x[j]=__float_as_uint(v); }
    else if constexpr(F==4) { float v=-__uint_as_float(x[j]);asm volatile("ex2.approx.ftz.f32 %0, %0;":"+f"(v));x[j]=__float_as_uint(v); }
    else if constexpr(F==5)asm volatile("shfl.sync.bfly.b32 %0, %0, 16, 31, 0xffffffff;":"+r"(x[j]));
    else if constexpr(F==6) { unsigned p=shared+4*x[j];asm volatile("ld.shared.u32 %0, [%1];":"=r"(x[j]):"r"(p):"memory"); }
    else if constexpr(F==7) { unsigned p=shared+4*(threadIdx.x*S+j);asm volatile("add.u32 %0, %0, 1; st.volatile.shared.u32 [%1], %0; ld.volatile.shared.u32 %0, [%1];":"+r"(x[j]):"r"(p):"memory"); } // volatile: no store-to-load forwarding or store merging (M445)
  }
}
template<int F,int S,int U> __global__ void compute_kernel(unsigned* out,unsigned long long* cycles,unsigned* sms,int loops,float a,float b) {
  extern __shared__ unsigned data[];
  unsigned x[S];
#pragma unroll
  for(int j=0;j<S;++j) {
    if constexpr(F==6)x[j]=threadIdx.x*S+j;
    else if constexpr(F==1 || F==5 || F==7)x[j]=(threadIdx.x%32)+100*j;
    else x[j]=__float_as_uint(1.0f+(threadIdx.x%32)/1024.0f+j/4.0f);
    if constexpr(F==6 || F==7)data[threadIdx.x*S+j]=x[j];
  }
  if constexpr(F==6 || F==7)__syncthreads();
  unsigned long long begin=clock64();
#pragma unroll 1
  for(int k=0;k<loops;++k) {
#pragma unroll
    for(int u=0;u<U;++u)one_step<F,S,U>(x,shared_pointer(data),a,b);
  }
  unsigned long long elapsed=clock64()-begin;
#pragma unroll
  for(int j=0;j<S;++j)out[(blockIdx.x*blockDim.x+threadIdx.x)*S+j]=x[j];
  if(threadIdx.x%32==0)cycles[(blockIdx.x*blockDim.x+threadIdx.x)/32]=elapsed;
  if(threadIdx.x==0)sms[blockIdx.x]=smid();
}
template<class T,bool CA> __global__ void memory_kernel(const T* input,unsigned* out,unsigned long long* cycles,unsigned* sms,unsigned table,unsigned stride,unsigned steps) {
  constexpr unsigned W=sizeof(T);
  const char* base=reinterpret_cast<const char*>(input)+static_cast<size_t>(blockIdx.x)*table;
  unsigned off=static_cast<unsigned>((static_cast<unsigned long long>(threadIdx.x)*stride*W)%table);
  unsigned long long begin=clock64();
#pragma unroll 1
  for(unsigned k=0;k<steps;++k) {
    T value;
    if constexpr(W==4 && CA)asm volatile("ld.global.ca.u32 %0, [%1];":"=r"(value):"l"(base+off):"memory");
    else if constexpr(W==4)asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(value):"l"(base+off):"memory");
    else if constexpr(CA)asm volatile("ld.global.ca.u64 %0, [%1];":"=l"(value):"l"(base+off):"memory");
    else asm volatile("ld.global.cg.u64 %0, [%1];":"=l"(value):"l"(base+off):"memory");
    // Fold both halves so ptxas keeps the full-width load (the ring stores a zero high word): M445.
    if constexpr(W==8)off=static_cast<unsigned>(value^(value>>32));
    else off=static_cast<unsigned>(value);
  }
  unsigned long long elapsed=clock64()-begin;
  size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  out[2*i]=off;out[2*i+1]=0;
  if(threadIdx.x%32==0)cycles[i/32]=elapsed;
  if(threadIdx.x==0)sms[blockIdx.x]=smid();
}
struct Case { bool memory;int f,s,u,k,t,blocks,table,width,stride,sms;bool ca; };
struct Buffers { unsigned* out;unsigned long long* cycles;unsigned* sms;void* input; };
template<int F,int S> static void compute_dispatch(const Case& c,Buffers d,cudaStream_t stream,float a,float b) {
  int shared=(F==6 || F==7)?c.t*S*4:0;
  if(c.u==1)compute_kernel<F,S,1><<<c.blocks,c.t,shared,stream>>>(d.out,d.cycles,d.sms,c.k,a,b);
  else if(c.u==4)compute_kernel<F,S,4><<<c.blocks,c.t,shared,stream>>>(d.out,d.cycles,d.sms,c.k,a,b);
  else if(c.u==16)compute_kernel<F,S,16><<<c.blocks,c.t,shared,stream>>>(d.out,d.cycles,d.sms,c.k,a,b);
  else throw std::runtime_error("unreviewed unroll");
}
template<int F> static void stream_dispatch(const Case& c,Buffers d,cudaStream_t stream,float a,float b) {
  if(c.s==1)compute_dispatch<F,1>(c,d,stream,a,b);
  else if(c.s==4)compute_dispatch<F,4>(c,d,stream,a,b);
  else throw std::runtime_error("unreviewed stream count");
}
static void launch(const Case& c,Buffers d,cudaStream_t stream,bool warmup,float a,float b) {
  if(c.memory) {
    unsigned steps=c.table/32*(warmup?1:2);
    if(c.width==4 && c.ca)memory_kernel<unsigned,true><<<c.blocks,c.t,0,stream>>>(static_cast<unsigned*>(d.input),d.out,d.cycles,d.sms,c.table,c.stride,steps);
    else if(c.width==4)memory_kernel<unsigned,false><<<c.blocks,c.t,0,stream>>>(static_cast<unsigned*>(d.input),d.out,d.cycles,d.sms,c.table,c.stride,steps);
    else if(c.ca)memory_kernel<unsigned long long,true><<<c.blocks,c.t,0,stream>>>(static_cast<unsigned long long*>(d.input),d.out,d.cycles,d.sms,c.table,c.stride,steps);
    else memory_kernel<unsigned long long,false><<<c.blocks,c.t,0,stream>>>(static_cast<unsigned long long*>(d.input),d.out,d.cycles,d.sms,c.table,c.stride,steps);
  } else switch(c.f) {
#define DISPATCH(N) case N:stream_dispatch<N>(c,d,stream,a,b);break;
    DISPATCH(0) DISPATCH(1) DISPATCH(2) DISPATCH(3) DISPATCH(4) DISPATCH(5) DISPATCH(6) DISPATCH(7)
#undef DISPATCH
    default:throw std::runtime_error("unknown family");
  }
  check(cudaGetLastError());
}
static void write_bytes(std::ofstream& f,const void* p,size_t bytes) { f.write(static_cast<const char*>(p),bytes);if(!f)throw std::runtime_error("artifact write failed"); }
static std::string uuid_string(const cudaUUID_t& id) {
  std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');
  for(int j=0;j<16;++j) { if(j==4 || j==6 || j==8 || j==10)s<<'-';s<<std::setw(2)<<static_cast<unsigned>(static_cast<unsigned char>(id.bytes[j])); }
  return s.str();
}
int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--describe") { std::cout<<"{\"schema\":\"isolated_calibration/1\",\"compiled_kernels\":52,\"samples\":3,\"warmup_launches\":1}\n";return 0; }
    if(argc!=14)throw std::runtime_error("require exact cell arguments");
    DeviceInfo D=cal_init_device();
    std::string kind=argv[2],policy=argv[11];
    if((kind!="memory" && kind!="compute") || (policy!="ca" && policy!="cg"))throw std::runtime_error("unknown kind/policy");
    Case c{kind=="memory",integer(argv[3]),integer(argv[4]),integer(argv[5]),integer(argv[6]),integer(argv[7]),integer(argv[8]),integer(argv[9]),integer(argv[10]),integer(argv[12]),integer(argv[13]),policy=="ca"};
    const std::vector<int> ts{32,64,128,256,384,512,768,1024},strides{1,2,4,5,7,8,16,23,32,64};
    if(std::find(ts.begin(),ts.end(),c.t)==ts.end() || c.sms!=D.sm || (c.blocks!=1 && c.blocks!=D.sm))throw std::runtime_error("unreviewed launch geometry");
    if(c.memory) {
      if(c.f!=8 || c.s!=2 || c.u!=1 || c.k!=1 || c.blocks!=D.sm || c.table%4096!=0 || c.table<65536 || c.table>(8u<<20) || (c.width!=4 && c.width!=8) || std::find(strides.begin(),strides.end(),c.stride)==strides.end())throw std::runtime_error("unreviewed memory case");
    } else if(c.f>7 || (c.s!=1 && c.s!=4) || (c.u!=1 && c.u!=4 && c.u!=16) || (c.k!=127 && c.k!=509) || c.table!=0 || c.width!=4 || !c.ca || c.stride!=1)throw std::runtime_error("unreviewed compute case");
    std::filesystem::path dir(argv[1]);std::filesystem::create_directories(dir);
    if(std::filesystem::exists(dir/"result.json") || std::filesystem::exists(dir/"sample0.bin"))throw std::runtime_error("cell already consumed");
    size_t lanes=static_cast<size_t>(c.blocks)*c.t,warps=lanes/32,input_bytes=static_cast<size_t>(c.blocks)*c.table;
    Buffers d{};check(cudaMalloc(&d.out,lanes*c.s*4));check(cudaMalloc(&d.cycles,warps*8));check(cudaMalloc(&d.sms,c.blocks*4));
    std::vector<unsigned char> input(input_bytes),readback(input_bytes);
    if(c.memory) {
      for(int off=0;off<c.table;off+=c.width) {
        unsigned long long next=(off+32)%c.table;std::memcpy(input.data()+off,&next,c.width);
      }
      for(int block=1;block<c.blocks;++block)std::memcpy(input.data()+static_cast<size_t>(block)*c.table,input.data(),c.table);
      std::ofstream pattern(dir/"input_pattern.bin",std::ios::binary);write_bytes(pattern,input.data(),c.table);pattern.close();
      check(cudaMalloc(&d.input,input_bytes));check(cudaMemcpy(d.input,input.data(),input_bytes,cudaMemcpyHostToDevice));
      check(cudaMemcpy(readback.data(),d.input,input_bytes,cudaMemcpyDeviceToHost));if(input!=readback)throw std::runtime_error("initial immutable input mismatch");
    }
    const float a=1.0f+std::ldexp(1.0f,-20),b=std::ldexp(1.0f,-20);
    unsigned reference[32][4]{};
    if(!c.memory)for(int lane=0;lane<32;++lane)for(int j=0;j<c.s;++j)reference[lane][j]=isolated::expected(c.f,lane,j,c.s,c.u*c.k,a,b);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    launch(c,d,stream,true,a,b);check(cudaStreamSynchronize(stream));
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));check(cudaEventRecordWithFlags(start,stream,cudaEventRecordExternal));launch(c,d,stream,false,a,b);check(cudaEventRecordWithFlags(stop,stream,cudaEventRecordExternal));check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,0));
    size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));std::vector<cudaGraphNode_t> graph_nodes(nodes);check(cudaGraphGetNodes(graph,graph_nodes.data(),&nodes));
    int kernel_nodes=0;for(auto node:graph_nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeKernel)++kernel_nodes;}
    if(nodes!=3 || kernel_nodes!=1)throw std::runtime_error("graph must contain two events and one kernel");
    std::vector<unsigned> out(lanes*c.s,std::numeric_limits<unsigned>::max()),sms(c.blocks,std::numeric_limits<unsigned>::max());std::vector<unsigned long long> cycles(warps,0);
    check(cudaMemcpy(d.out,out.data(),out.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(d.cycles,cycles.data(),cycles.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(d.sms,sms.data(),sms.size()*4,cudaMemcpyHostToDevice));
    float ms[3]{};double maximum_error=0;
    for(int sample=0;sample<3;++sample) {
      check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));check(cudaEventElapsedTime(&ms[sample],start,stop));
      if(!std::isfinite(ms[sample]) || ms[sample]<=0)throw std::runtime_error("nonpositive event interval");
      check(cudaMemcpy(out.data(),d.out,out.size()*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(cycles.data(),d.cycles,cycles.size()*8,cudaMemcpyDeviceToHost));check(cudaMemcpy(sms.data(),d.sms,sms.size()*4,cudaMemcpyDeviceToHost));
      std::ofstream raw(dir/("sample"+std::to_string(sample)+".bin"),std::ios::binary);write_bytes(raw,out.data(),out.size()*4);write_bytes(raw,cycles.data(),cycles.size()*8);write_bytes(raw,sms.data(),sms.size()*4);raw.close();
      for(size_t i=0;i<lanes;++i)for(int j=0;j<c.s;++j) {
        unsigned ref=c.memory?(j==0?isolated::offset(i%c.t,c.stride,c.width,c.table):0):c.f==6?(i%c.t)*c.s+j:reference[i%32][j];
        if(!(c.memory?out[i*c.s+j]==ref:isolated::agrees(c.f,out[i*c.s+j],ref)))throw std::runtime_error("output oracle mismatch; raw output retained");
        if(!c.memory && c.f==4)maximum_error=std::max(maximum_error,static_cast<double>(std::fabs(isolated::value(out[i*c.s+j])-isolated::value(ref))));
      }
      if(std::any_of(cycles.begin(),cycles.end(),[](auto x){return x==0;}) || std::any_of(sms.begin(),sms.end(),[](auto x){return x==std::numeric_limits<unsigned>::max();}))throw std::runtime_error("invalid device clock or SM assignment");
    }
    if(c.memory){check(cudaMemcpy(readback.data(),d.input,input_bytes,cudaMemcpyDeviceToHost));if(input!=readback)throw std::runtime_error("final immutable input mismatch");}
    std::ofstream result(dir/"result.json");result<<std::setprecision(12)<<"{\"schema\":\"isolated_calibration_cell/1\",\"event_ms\":["<<ms[0]<<','<<ms[1]<<','<<ms[2]<<"],\"kernel_launches\":4,\"kernel_graph_nodes\":1,\"graph_nodes\":3,\"output_words_checked\":"<<out.size()*3<<",\"input_readback_bytes_checked\":"<<input_bytes*2<<",\"approximate_maximum_absolute_error\":"<<maximum_error<<",\"alpha_bits\":"<<isolated::bits(a)<<",\"beta_bits\":"<<isolated::bits(b)<<",\"device_uuid\":\""<<D.uuid<<"\",\"sm_count\":"<<D.sm<<",\"timing_admitted\":false,\"physical_latency_admitted\":false}\n";result.close();if(!result)throw std::runtime_error("result write failed");
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaStreamDestroy(stream));check(cudaFree(d.out));check(cudaFree(d.cycles));check(cudaFree(d.sms));if(d.input)check(cudaFree(d.input));return 0;
  } catch(const std::exception& e){std::cerr<<"REFUSED: "<<e.what()<<'\n';return 1;}
}
