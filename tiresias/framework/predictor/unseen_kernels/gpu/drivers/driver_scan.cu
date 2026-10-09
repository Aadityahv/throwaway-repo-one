// Driver for the pinned cuda-samples prefix-sum scan (cpp/2_Concepts_and_Techniques/scan/scan.cu; the sample's
// main lives in main.cpp and is not used). The sample source is included byte-unmodified.
//
//   driver_scan <N> <arrayLength> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// One operator call = the three kernels of the sample's scanExclusiveLarge(), launched directly on the driver's
// private stream (scanExclusiveLarge itself launches on the default stream, which cannot be graph-captured):
//   nb = N / (4 * 256)
//   scanExclusiveShared  <<<nb, 256>>>((uint4*)dst, (uint4*)src, 1024)
//   scanExclusiveShared2 <<<ceil(nb / 256), 256>>>(buf, dst, src, nb, arrayLength / 1024)
//   uniformUpdate        <<<nb, 256>>>((uint4*)dst, buf)
// buf holds nb unsigned ints. The result is an exclusive prefix sum of each consecutive array of arrayLength
// elements (N / arrayLength arrays). N % arrayLength == 0, arrayLength a power of two in [2048, 262144],
// N % 1024 == 0, N <= 64 Mi (the same asserts as the sample).
// Input: unsigned ints with value % 16 (splitmix64 stream seeded kSeed+1). dst is pre-filled with 0xFF bytes.
// Check: exact per-array exclusive prefix sums with 32-bit wraparound, every element compared.
#include "driver_common.h"

#define main SAMPLE_main_disabled
#include "scan.cu"
#undef main

#ifdef THREADBLOCK_SIZE
#if THREADBLOCK_SIZE != 256
#error "the cells assume THREADBLOCK_SIZE == 256 (cells.py geometry)"
#endif
#endif

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 4 && argc != 10) {
        std::fprintf(stderr, "usage: %s <N> <arrayLength> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long N = parse_ll(argv[1], "N");
    const long long L = parse_ll(argv[2], "arrayLength");
    const char *out_path = argv[3];
    const long long kTB = 256;               // THREADBLOCK_SIZE
    const long long kMaxBatch = 64LL << 20;  // MAX_BATCH_ELEMENTS of the sample
    if (N <= 0 || N > kMaxBatch) fatal("N out of range (1..64Mi)");
    if (L < 2048 || L > 262144 || (L & (L - 1)) != 0) fatal("arrayLength must be a power of two in [2048, 262144]");
    if (N % L != 0 || N % (4 * kTB) != 0) fatal("N must be a multiple of arrayLength and of 1024");
    const Trailer tr = parse_trailer(argc, argv, 4);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const size_t n = (size_t)N;
    const size_t bytes = n * sizeof(unsigned);
    const unsigned nb = (unsigned)(N / (4 * kTB));
    const unsigned top_blocks = (unsigned)ceil_div_ll(nb, kTB);
    const unsigned top_array_length = (unsigned)(L / (4 * kTB));  // arrayLength / 1024
    std::vector<unsigned> hSrc(n), hDst(n);
    {
        SplitMix r1(kSeed + 1);
        for (size_t i = 0; i < n; ++i) hSrc[i] = r1.small_uint();
    }
    unsigned *dSrc = nullptr, *dDst = nullptr, *dBuf = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dSrc, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dDst, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dBuf, (size_t)nb * sizeof(unsigned)));
    UD_CUDA_CHECK(cudaMemcpy(dSrc, hSrc.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemset(dDst, 0xFF, bytes));
    UD_CUDA_CHECK(cudaMemset(dBuf, 0xFF, (size_t)nb * sizeof(unsigned)));
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    auto launch_all = [&](cudaStream_t s) {
        scanExclusiveShared<<<nb, (unsigned)kTB, 0, s>>>((uint4 *)dDst, (uint4 *)dSrc, 4u * (unsigned)kTB);
        scanExclusiveShared2<<<top_blocks, (unsigned)kTB, 0, s>>>(dBuf, dDst, dSrc, nb, top_array_length);
        uniformUpdate<<<nb, (unsigned)kTB, 0, s>>>((uint4 *)dDst, dBuf);
    };
    execute(stream, tr, launch_all);

    // ---- correctness, after the timed loop and the final sync, never inside the timed region ----
    UD_CUDA_CHECK(cudaMemcpy(hDst.data(), dDst, bytes, cudaMemcpyDeviceToHost));
    maybe_write_output(out_path, hDst.data(), bytes);
    long long mismatches = 0, first_bad = -1;
    for (size_t base = 0; base < n; base += (size_t)L) {
        unsigned running = 0;  // 32-bit wraparound
        for (size_t i = base; i < base + (size_t)L; ++i) {
            if (hDst[i] != running) {
                if (mismatches == 0) first_bad = (long long)i;
                ++mismatches;
            }
            running += hSrc[i];
        }
    }
    const std::string msg = fmt("scan N=%lld arrayLength=%lld mismatches=%lld first_bad=%lld repeat=%lld graph_batch=%lld", N,
                                L, mismatches, first_bad, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dSrc));
    UD_CUDA_CHECK(cudaFree(dDst));
    UD_CUDA_CHECK(cudaFree(dBuf));
    return finish(out_path, mismatches == 0, msg);
}
