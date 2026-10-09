// Synthetic calibration only. No application-energy labels or family corrections.
// Shared graph timing/trace contract is the already reviewed application harness.
#include "../../fresh_e/gpu/drivers/driver_common.h"
#include "oracle.hpp"
#include <stdexcept>
#include <limits>

static constexpr const char* approved_uuid = "GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894";
static constexpr int blocks = 188, threads = 256;
static constexpr size_t lanes = blocks * threads;
static constexpr size_t verified_l2 = 134217728;

template<int R, int F, int S, int H>
__global__ void energy_fixture(const unsigned* input, unsigned* out, unsigned n) {
    __shared__ volatile unsigned shared[threads];
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned checksum = 0, other = unsigned(lane), special = 0;
    // n is an exact multiple of the full launch width: uniform loop, no partial warp.
    // All non-global instruction families must be counted from the actual binary,
    // including checksum, loop, address and conversion work.
#pragma unroll 1
    for (unsigned index = lane; index < n; index += unsigned(lanes)) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            unsigned v;
            // Distinct adjacent addresses preserve the native memory dose:
            // ptxas can merge repeated non-volatile PTX loads of one address.
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(input + index + r) : "memory");
            float fp = __uint_as_float(v);
#pragma unroll
            for (int f = 0; f < F; ++f)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fp) : "f"(0.9990234375f), "f"(0.0009765625f));
            float sf = fp;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                // Bounded dependent recurrence, not repeated dead assignments.
                // Values remain in [0.5,1], preventing overflow/subnormal inputs.
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(sf) : "f"(-sf));
                special = __float_as_uint(sf);
            }
#pragma unroll
            for (int h = 0; h < H; ++h) {
                other += other ^ v;
                shared[threadIdx.x] = other;
                other = shared[threadIdx.x]; // own slot, volatile: cannot forward/remove
            }
            // Make every SFU iteration observable without comparing approximate
            // bits exactly: reference checks the rounding margin before use.
            unsigned sink = S ? __float2uint_rn(sf * 16.0f) : F ? __float_as_uint(fp) : v;
            // Memory-only dependence slows combined lookup/DRAM issue.
            // Every operation is observable and counted from the native binary.
#pragma unroll
            for (int pause = 0; pause < (R > 1 ? 8 : 0); ++pause)
                checksum = (checksum * 1664525u + 1013904223u) ^ sink;
            checksum = (checksum * 1664525u + 1013904223u) ^ sink;
        }
    }
    out[3 * lane] = checksum;
    out[3 * lane + 1] = other;
    out[3 * lane + 2] = special;
}

using Mix = energy_oracle::Mix;
static Mix choose(const std::string& mix, int dose) {
    if (mix == "memory") return {dose, 0, 0, 0};
    if (mix == "arithmetic") return {1, dose, 0, 0};
    if (mix == "special_function") return {1, 0, dose, 0};
    if (mix == "other_shared") return {1, 0, 0, dose};
    if (mix == "mixed") return {1, 8, 8, 8};
    if (mix == "mixed_altered") return {1, 24, 4, 12};
    throw std::runtime_error("unknown activity mix");
}

static void launch(const Mix& m, const unsigned* in, unsigned* out, size_t n, cudaStream_t stream) {
    if (n > std::numeric_limits<unsigned>::max() - lanes)
        throw std::runtime_error("REFUSED: unsigned loop induction overflow");
#define RUN(R,F,S,H) energy_fixture<R,F,S,H><<<blocks,threads,0,stream>>>(in,out,unsigned(n))
    if (m.r == 16) { RUN(16,0,0,0); }
    else if (m.r == 64) { RUN(64,0,0,0); }
    else if (m.f == 16) { RUN(1,16,0,0); }
    else if (m.f == 64) { RUN(1,64,0,0); }
    else if (m.s == 16) { RUN(1,0,16,0); }
    else if (m.s == 64) { RUN(1,0,64,0); }
    else if (m.h == 16) { RUN(1,0,0,16); }
    else if (m.h == 64) { RUN(1,0,0,64); }
    else if (m.f == 8) { RUN(1,8,8,8); }
    else if (m.f == 24) { RUN(1,24,4,12); }
    else throw std::runtime_error("unsupported template combination");
#undef RUN
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) throw std::runtime_error("usage: calibration MIX {16|64} FOOTPRINT out.bin [--repeat N --graph-batch B --trace-dir DIR]");
        int dose = int(unseen::parse_ll(argv[2], "dose"));
        if (dose != 16 && dose != 64) throw std::runtime_error("unreviewed dose");
        Mix m = choose(argv[1], dose);
        std::string foot(argv[3]);
        if (std::string(argv[1]) != "memory" || foot != "dram_candidate")
            throw std::runtime_error("REFUSED: only the two proposed DRAM memory controls are in scope");
        size_t target = foot == "small_candidate" ? lanes * 4 : foot == "l2_candidate" ? verified_l2 / 4 : foot == "dram_candidate" ? 4 * verified_l2 : 0;
        if (!target) throw std::runtime_error("unreviewed footprint");
        size_t n = (foot == "dram_candidate" ? (target + lanes * 4 - 1) / (lanes * 4) : target / (lanes * 4)) * lanes;
        auto trailer = unseen::parse_trailer(argc, argv, 5);
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        const char* expected = std::getenv("UNSEEN_EXPECT_UUID");
        const char* booking = std::getenv("ENERGY_BOOKING_REF");
        if (!visible || std::string(visible) != "1" || !expected || std::string(expected) != approved_uuid || !booking || !*booking)
            throw std::runtime_error("REFUSED: Blackwell GPU 1 visibility, exact UUID and ENERGY_BOOKING_REF required");
        unseen::guard_single_approved_device();
        cudaDeviceProp prop; UD_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (prop.multiProcessorCount != blocks || size_t(prop.l2CacheSize) != verified_l2 || prop.major != 12 || prop.minor != 0)
            throw std::runtime_error("REFUSED: live hardware differs from frozen ground truth");
        std::vector<unsigned> input(n + m.r - 1), output(3 * lanes);
        for (size_t i = 0; i < input.size(); ++i) input[i] = energy_oracle::input_bits(i);
        unsigned *di, *dout;
        UD_CUDA_CHECK(cudaMalloc(&di, input.size() * 4)); UD_CUDA_CHECK(cudaMalloc(&dout, output.size() * 4));
        UD_CUDA_CHECK(cudaMemcpy(di, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        cudaStream_t stream; UD_CUDA_CHECK(cudaStreamCreate(&stream));
        unseen::remove_stale_outputs(argv[4]);
        unseen::execute(stream, trailer, [&](cudaStream_t s) { launch(m, di, dout, n, s); });
        UD_CUDA_CHECK(cudaMemcpy(output.data(), dout, output.size() * 4, cudaMemcpyDeviceToHost));
        // Oracle after timed block; inputs reloaded each call, output never fed back.
        // Full CPU expected outputs created at correctness gate, then cached separately
        // from actual GPU output. The supervisor binds this cache's hash before energy.
        auto expected_output = energy_oracle::cached(std::string(argv[4])+".oracle",n,lanes,m);
        bool ok = energy_oracle::matches(output,expected_output,m);
        unseen::maybe_write_output(argv[4],output.data(),output.size()*sizeof(unsigned));
        UD_CUDA_CHECK(cudaFree(di)); UD_CUDA_CHECK(cudaFree(dout)); UD_CUDA_CHECK(cudaStreamDestroy(stream));
        return unseen::finish(argv[4], ok, unseen::fmt("energy calibration n=%zu bytes=%zu R=%d F=%d S=%d H=%d", n, n*4, m.r, m.f, m.s, m.h));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "REFUSED: %s\n", e.what()); return 2;
    }
}
