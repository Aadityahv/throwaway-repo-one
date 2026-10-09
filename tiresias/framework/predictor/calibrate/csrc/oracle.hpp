#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace isolated {
inline unsigned bits(float value) { unsigned b;std::memcpy(&b,&value,4);return b; }
inline float value(unsigned b) { float v;std::memcpy(&v,&b,4);return v; }
inline unsigned initial(int family,unsigned thread,int stream,int streams) {
  if(family==6)return thread*streams+stream;
  if(family==1 || family==5 || family==7)return (thread%32)+100*stream;
  return bits(1.0f+(thread%32)/1024.0f+stream/4.0f);
}
inline unsigned expected(int family,unsigned thread,int stream,int streams,unsigned steps,float a,float b) {
  unsigned x=initial(family,thread,stream,streams);
  if(family==0 || family==6)return x;
  if(family==1) { unsigned c=bits(b);for(unsigned k=0;k<steps;++k)x=x+(x^c);return x; }
  if(family==7)return x+steps;
  if(family==5)return initial(family,(thread&~31u)|((thread%32)^((steps&1)?16:0)),stream,streams);
  float f=value(x);
  for(unsigned k=0;k<steps;++k) {
    if(family==2)f=f+b;
    else if(family==3)f=std::fma(f,a,-b);
    else if(family==4)f=std::exp2(-f);
    else throw std::runtime_error("unknown family");
  }
  if(!std::isfinite(f))throw std::runtime_error("nonfinite scalar oracle");
  return bits(f);
}
inline bool agrees(int family,unsigned actual,unsigned reference) {
  if(family!=4)return actual==reference;
  float a=value(actual),b=value(reference);
  // Explicit bounded approximate-instruction oracle; never exact-bit admission.
  return std::isfinite(a) && a>0 && a<1 && std::fabs(a-b)<=1e-5f;
}
inline unsigned offset(unsigned thread,unsigned stride,unsigned width,unsigned table) {
  return static_cast<unsigned>((static_cast<std::uint64_t>(thread)*stride*width)%table);
}
}
