// Driver for the pinned cuda-samples separable convolution
// (cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu; the sample's main lives in
// main.cpp and is not used). The sample source is included byte-unmodified.
//
//   driver_conv <W> <H> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// setConvolutionKernel(h_Kernel) is called once, before any capture (kernel length 17 = 2 * KERNEL_RADIUS + 1,
// integer weights with value % 16). One operator call is the sample's convolutionRowsGPU() followed by
// convolutionColumnsGPU(), launched directly on the driver's private stream:
//   convolutionRowsKernel   <<<dim3(W / 128, H / 4),  dim3(16, 4)>>>(tmp, src, W, H, W)
//   convolutionColumnsKernel<<<dim3(W / 16,  H / 64), dim3(16, 8)>>>(dst, tmp, W, H, W)
// (pitch = W). W % 128 == 0 and H % 64 == 0 (no partial tail block).
// Input: image of integers with value % 16 (splitmix64 stream seeded kSeed+1), kernel weights from stream
// kSeed+2, stored as floats, so every partial sum of both passes is below 2^24 and the float result is exact.
// tmp and dst are pre-filled with 0xFF bytes (NaN).
// Check: exact two-pass reference in integer arithmetic, zero padding outside the image, using the gold
// convention of the sample (dst[x] = sum_{k=-R..R} src[x+k] * kernel[R-k], rows pass then columns pass);
// every element compared for exact equality.
#include "driver_common.h"

#define main SAMPLE_main_disabled
#include "convolutionSeparable.cu"
#undef main

#ifdef KERNEL_RADIUS
#if KERNEL_RADIUS != 8
#error "the cells assume KERNEL_RADIUS == 8 (cells.py geometry)"
#endif
#endif

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 4 && argc != 10) {
        std::fprintf(stderr, "usage: %s <W> <H> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long W = parse_ll(argv[1], "W");
    const long long H = parse_ll(argv[2], "H");
    const char *out_path = argv[3];
    constexpr int R = 8;
    constexpr int KLEN = 2 * R + 1;  // 17
    if (W <= 0 || H <= 0 || W > 65535 * 16 || H > 65535 * 4) fatal("W/H out of range");
    if (W % 128 != 0 || H % 64 != 0) fatal("W must be a multiple of 128 and H of 64 (no partial tail block)");
    const Trailer tr = parse_trailer(argc, argv, 4);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const size_t n = (size_t)W * (size_t)H;
    const size_t bytes = n * sizeof(float);
    std::vector<float> hSrc(n), hDst(n);
    float hKernel[KLEN];
    {
        SplitMix r1(kSeed + 1), r2(kSeed + 2);
        for (size_t i = 0; i < n; ++i) hSrc[i] = (float)r1.small_uint();
        for (int i = 0; i < KLEN; ++i) hKernel[i] = (float)r2.small_uint();
    }
    float *dSrc = nullptr, *dTmp = nullptr, *dDst = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dSrc, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dTmp, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dDst, bytes));
    UD_CUDA_CHECK(cudaMemcpy(dSrc, hSrc.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemset(dTmp, 0xFF, bytes));
    UD_CUDA_CHECK(cudaMemset(dDst, 0xFF, bytes));
    setConvolutionKernel(hKernel);  // once, before capture (cudaMemcpyToSymbol)
    UD_CUDA_CHECK(cudaGetLastError());
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    const dim3 rows_blocks((unsigned)(W / 128), (unsigned)(H / 4)), rows_threads(16, 4);
    const dim3 cols_blocks((unsigned)(W / 16), (unsigned)(H / 64)), cols_threads(16, 8);
    const int w_int = (int)W, h_int = (int)H;
    auto launch_all = [&](cudaStream_t s) {
        convolutionRowsKernel<<<rows_blocks, rows_threads, 0, s>>>(dTmp, dSrc, w_int, h_int, w_int);
        convolutionColumnsKernel<<<cols_blocks, cols_threads, 0, s>>>(dDst, dTmp, w_int, h_int, w_int);
    };
    execute(stream, tr, launch_all);

    // ---- correctness, after the timed loop and the final sync, never inside the timed region ----
    UD_CUDA_CHECK(cudaMemcpy(hDst.data(), dDst, bytes, cudaMemcpyDeviceToHost));
    maybe_write_output(out_path, hDst.data(), bytes);
    std::vector<int> ker(KLEN);
    for (int i = 0; i < KLEN; ++i) ker[i] = (int)hKernel[i];
    std::vector<int> tmp(n, 0);
    const long long w = W, h = H;
    // rows pass
    for (long long y = 0; y < h; ++y) {
        const float *srow = hSrc.data() + (size_t)y * w;
        int *trow = tmp.data() + (size_t)y * w;
        for (long long x = 0; x < w; ++x) {
            int acc = 0;
            const long long k_lo = std::max<long long>(-R, -x);
            const long long k_hi = std::min<long long>(R, w - 1 - x);
            for (long long k = k_lo; k <= k_hi; ++k) acc += (int)srow[x + k] * ker[R - k];
            trow[x] = acc;
        }
    }
    // columns pass; the accumulation runs over whole rows so the inner loop is contiguous
    long long mismatches = 0, first_bad = -1;
    std::vector<int> acc_row((size_t)w);
    for (long long y = 0; y < h; ++y) {
        std::fill(acc_row.begin(), acc_row.end(), 0);
        const long long k_lo = std::max<long long>(-R, -y);
        const long long k_hi = std::min<long long>(R, h - 1 - y);
        for (long long k = k_lo; k <= k_hi; ++k) {
            const int *trow = tmp.data() + (size_t)(y + k) * w;
            const int coeff = ker[R - k];
            for (long long x = 0; x < w; ++x) acc_row[(size_t)x] += trow[x] * coeff;
        }
        for (long long x = 0; x < w; ++x) {
            if (hDst[(size_t)(y * w + x)] != (float)acc_row[(size_t)x]) {  // NaN also mismatches
                if (mismatches == 0) first_bad = y * w + x;
                ++mismatches;
            }
        }
    }
    const std::string msg = fmt("conv W=%lld H=%lld mismatches=%lld first_bad=%lld repeat=%lld graph_batch=%lld", W, H,
                                mismatches, first_bad, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dSrc));
    UD_CUDA_CHECK(cudaFree(dTmp));
    UD_CUDA_CHECK(cudaFree(dDst));
    return finish(out_path, mismatches == 0, msg);
}
