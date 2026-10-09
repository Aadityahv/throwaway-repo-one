// Driver for the pinned cuda-samples fast Walsh transform (cpp/5_Domain_Specific/fastWalshTransform/fastWalshTransform_kernel.cuh holds fwtBatch1Kernel,
// fwtBatch2Kernel and the host front-end fwtBatchGPU; fastWalshTransform.cu, which owns the sample's main, is not used). The header is included
// byte-unmodified.
//
//   driver_fwt <log2N> <batches> <inplace|pingpong> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// One operator call = the launch sequence of the sample's fwtBatchGPU(d_Data, M = batches, log2N), re-issued on the driver's private stream (the sample's
// own launches use the default stream, which cannot be graph-captured): radix-4 global passes fwtBatch2Kernel<<<(N/(4*256), M), 256>>>(d, d, N/4)
// while log2N > 11 (log2N -= 2, N >>= 2, M <<= 2 per pass), then fwtBatch1Kernel<<<M, N/4, N*4>>>(d, d, log2N). In place, as in the sample.
// Mode inplace is the sample's own call (used for runtime timing). Mode pingpong (used for ENERGY only) makes the first radix-4 pass read a separate pristine input
// buffer and write the working buffer (all later passes in place), so repeated calls always start from the same finite data instead of overflowing; SASS and
// bytes per kernel are unchanged, only the first pass's source pointer differs. Input: values in {-1, 0, 1} (splitmix64 seeded kSeed+1). The transform is unnormalised, so repeated in-place calls grow the values by sqrt(N)
// per call and overflow to inf/NaN after a few calls; floating-point instruction timing does not depend on the data. Therefore the correctness check
// runs on a FIRST single call, before the timed region, on the pristine input; the data is then restored and the timed loop runs.
// Check: relative L2 error ||gpu - ref||_2 / ||ref||_2 < 1e-5 against an in-place double-precision CPU Walsh-Hadamard transform of every batch.
#include "driver_common.h"

#include <helper_cuda.h>
#define main SAMPLE_main_disabled
#include "fastWalshTransform_kernel.cuh"
#undef main

static void wht_cpu(std::vector<double> &x, size_t base, size_t n) {
    for (size_t h = 1; h < n; h <<= 1)
        for (size_t i = 0; i < n; i += h << 1)
            for (size_t j = i; j < i + h; ++j) {
                const double a = x[base + j], b = x[base + j + h];
                x[base + j] = a + b;
                x[base + j + h] = a - b;
            }
}

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 5 && argc != 11) {
        std::fprintf(stderr, "usage: %s <log2N> <batches> <inplace|pingpong> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long L = parse_ll(argv[1], "log2N"), M = parse_ll(argv[2], "batches");
    const bool pingpong = std::strcmp(argv[3], "pingpong") == 0;
    if (!pingpong && std::strcmp(argv[3], "inplace") != 0) fatal("mode must be inplace or pingpong");
    const char *out_path = argv[4];
    if (L < 12 || L > 27 || M < 1 || M > 64) fatal("log2N must be in [12, 27] and batches in [1, 64]");
    const Trailer tr = parse_trailer(argc, argv, 5);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const size_t N = (size_t)1 << L, total = N * (size_t)M, bytes = total * sizeof(float);
    std::vector<float> hIn(total), hOut(total);
    {
        SplitMix r1(kSeed + 1);
        for (size_t i = 0; i < total; ++i) hIn[i] = (float)((int)(r1.next() >> 33) % 3 - 1);
    }
    float *dData = nullptr, *dIn = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dData, bytes));
    if (pingpong) {
        UD_CUDA_CHECK(cudaMalloc(&dIn, bytes));
        UD_CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));
    }
    UD_CUDA_CHECK(cudaMemcpy(dData, hIn.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    auto launch_all = [&](cudaStream_t s) {
        const int THREAD_N = 256;
        int l = (int)L;
        long long n = (long long)1 << l, m = M;
        dim3 grid((unsigned)(((long long)1 << l) / (4 * THREAD_N)), (unsigned)M, 1);
        bool first = true;
        for (; l > ELEMENTARY_LOG2SIZE; l -= 2, n >>= 2, m <<= 2, first = false)
            fwtBatch2Kernel<<<grid, THREAD_N, 0, s>>>(dData, (pingpong && first) ? dIn : dData, (int)(n / 4));
        fwtBatch1Kernel<<<(unsigned)m, (unsigned)(n / 4), (size_t)n * sizeof(float), s>>>(dData, dData, l);
    };

    // ---- correctness on one pristine call, before the timed region
    launch_all(stream);
    UD_CUDA_CHECK(cudaGetLastError());
    UD_CUDA_CHECK(cudaStreamSynchronize(stream));
    UD_CUDA_CHECK(cudaMemcpy(hOut.data(), dData, bytes, cudaMemcpyDeviceToHost));
    maybe_write_output(out_path, hOut.data(), bytes);
    double num = 0.0, den = 0.0;
    {
        std::vector<double> ref(total);
        for (size_t i = 0; i < total; ++i) ref[i] = (double)hIn[i];
        for (long long b = 0; b < M; ++b) wht_cpu(ref, (size_t)b * N, N);
        for (size_t i = 0; i < total; ++i) {
            const double d = (double)hOut[i] - ref[i];
            num += d * d;
            den += ref[i] * ref[i];
        }
    }
    const double rel = std::sqrt(num / (den > 0 ? den : 1.0));
    // ---- restore the pristine input, then the timed loop (values overflow during it; instruction timing is data independent)
    UD_CUDA_CHECK(cudaMemcpy(dData, hIn.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaDeviceSynchronize());
    execute(stream, tr, launch_all);
    const bool ok = rel < 1e-5;
    const std::string msg = fmt("fastWalshTransform mode=%s log2N=%lld batches=%lld rel_l2_error=%.3g repeat=%lld graph_batch=%lld", pingpong ? "pingpong" : "inplace", L, M, rel, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dData));
    if (dIn) UD_CUDA_CHECK(cudaFree(dIn));
    return finish(out_path, ok, msg);
}
