// CPU-only reference, cached after the full-grid correctness gate. Never caches GPU output.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace energy_oracle {
struct Mix { int r, f, s, h; };
inline unsigned input_bits(size_t i) { return 0x3f000000u | (unsigned(i * 2654435761u) & 0x007fffffu); }
inline float as_float(unsigned x) { float f; std::memcpy(&f,&x,4); return f; }
inline unsigned as_bits(float f) { unsigned x; std::memcpy(&x,&f,4); return x; }
inline unsigned fold(unsigned checksum, unsigned v) { return (checksum*1664525u+1013904223u)^v; }

inline std::vector<unsigned> compute(size_t n, size_t lanes, Mix m) {
    if (!n || !lanes || n % lanes || m.r <= 0 || m.f < 0 || m.s < 0 || m.h < 0)
        throw std::runtime_error("invalid oracle controls");
    std::vector<unsigned> result(3*lanes);
    for (size_t lane=0;lane<lanes;++lane) {
        unsigned checksum=0,other=unsigned(lane); float special=0;
        for (size_t index=lane;index<n;index+=lanes) {
            for(int r=0;r<m.r;++r) {
                unsigned v=input_bits(index+r); float fp=as_float(v);
                for(int f=0;f<m.f;++f)fp=std::fma(fp,0.9990234375f,0.0009765625f);
                if(m.s) {
                    special=fp;
                    for(int s=0;s<m.s;++s)special=float(std::exp2(-double(special)));
                }
                unsigned sink=m.f?as_bits(fp):v;
                if(m.s) {
                    float scaled=special*16.0f;
                    float ulp=std::nextafter(special,std::numeric_limits<float>::infinity())-special;
                    double margin=std::fabs(double(scaled)-std::floor(double(scaled))-.5);
                    if(margin<=10*double(ulp)*16)throw std::runtime_error("REFUSED: SFU quantized sink lacks propagated error margin");
                    sink=unsigned(std::floor(double(scaled)+.5));
                }
                checksum=fold(checksum,sink);
                for(int h=0;h<m.h;++h)other+=other^v;
            }
        }
        result[3*lane]=checksum;result[3*lane+1]=other;result[3*lane+2]=as_bits(special);
    }
    return result;
}

inline std::vector<unsigned> cached(const std::string& path,size_t n,size_t lanes,Mix m) {
    const uint64_t header[8]={0x454e455247590004ULL,uint64_t(n),uint64_t(lanes),uint64_t(m.r),uint64_t(m.f),uint64_t(m.s),uint64_t(m.h),uint64_t(3*lanes)};
    std::vector<unsigned> result(3*lanes);
    std::ifstream in(path,std::ios::binary);
    if(in.good()) {
        uint64_t found[8];in.read(reinterpret_cast<char*>(found),sizeof(found));
        if(!in || std::memcmp(found,header,sizeof(header)))throw std::runtime_error("REFUSED: stale oracle cache controls/version");
        in.read(reinterpret_cast<char*>(result.data()),result.size()*sizeof(unsigned));
        if(!in || in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("REFUSED: truncated/extended oracle cache");
        return result;
    }
    result=compute(n,lanes,m);
    std::ofstream out(path,std::ios::binary);
    if(!out)throw std::runtime_error("cannot create CPU oracle cache");
    out.write(reinterpret_cast<const char*>(header),sizeof(header));
    out.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(unsigned));
    if(!out)throw std::runtime_error("short CPU oracle cache write");
    return result;
}

inline bool matches(const std::vector<unsigned>& actual,const std::vector<unsigned>& expected,Mix m) {
    if(actual.size()!=expected.size() || actual.size()%3) return false;
    for(size_t i=0;i<actual.size();i+=3) {
        if(actual[i]!=expected[i] || actual[i+1]!=expected[i+1])return false;
        if(!m.s) { if(actual[i+2]!=0)return false; }
        else {
            float x=as_float(actual[i+2]),y=as_float(expected[i+2]);
            float ulp=std::nextafter(y,std::numeric_limits<float>::infinity())-y;
            // f(x)=2^-x has derivative <0.7 on [0.5,1]. At most 2 device
            // +1 CPU-reference ulp per step: geometric propagation <3/(1-.7)=10 ulp.
            if(!std::isfinite(x) || std::fabs(x-y)>10*ulp)return false;
        }
    }
    return true;
}
}
