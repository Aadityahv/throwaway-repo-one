#include "oracle.hpp"
#include <cassert>
#include <cstdio>

int main(int argc,char** argv) {
    using namespace energy_oracle;
    assert(argc==2);std::string path=argv[1];std::remove(path.c_str());
    auto memory=compute(1,1,Mix{16,0,0,0});
    // Direct XOR over an even dose would be zero. The retained checksum must not be.
    assert(memory[0]!=0);
    auto arithmetic=compute(1,1,Mix{1,1,0,0});
    assert(arithmetic[0]==((1013904223u)^as_bits(0.50048828125f)));
    auto integer=compute(1,1,Mix{1,0,0,1});
    assert(integer[1]==input_bits(0));
    auto original=cached(path,4,2,Mix{1,1,1,1});
    assert(original==cached(path,4,2,Mix{1,1,1,1}));
    assert(matches(original,original,Mix{1,1,1,1}));
    auto wrong=original;wrong[0]^=1;assert(!matches(wrong,original,Mix{1,1,1,1}));
    wrong=original;wrong[2]=0x7fc00000;assert(!matches(wrong,original,Mix{1,1,1,1}));
    bool refused=false;try{cached(path,6,2,Mix{1,1,1,1});}catch(const std::runtime_error&){refused=true;}assert(refused);
    std::ofstream corrupt(path,std::ios::binary|std::ios::trunc);corrupt<<"truncated";corrupt.close();
    refused=false;try{cached(path,4,2,Mix{1,1,1,1});}catch(const std::runtime_error&){refused=true;}assert(refused);
    std::remove(path.c_str());puts("CPU oracle: exact FMA/integer anchors, noncancelling sink, cache controls and corruption/NaN guards pass");
}
