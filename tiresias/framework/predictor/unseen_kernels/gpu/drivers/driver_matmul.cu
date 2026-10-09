// Driver for the pinned cuda-samples tiled matrix multiply (cpp/0_Introduction/matrixMul/matrixMul.cu).
// The sample source is included byte-unmodified; only its main() is renamed away so this file can own main().
//
//   driver_matmul <N> <tile> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// One operator call = MatrixMulCUDA<tile><<<dim3(N/tile, N/tile), dim3(tile, tile)>>>(C, A, B, wA=N, wB=N),
// the geometry and arguments of the sample's MatrixMultiply() for square N x N matrices. tile in {16, 32}.
// Inputs: A, B uniform [-1, 1), splitmix64 streams seeded kSeed+1 (A) and kSeed+2 (B). C is pre-filled with
// 0xFF bytes (NaN) so a kernel that does not run fails the check.
// Check: 4096 deterministically sampled output elements against a double-precision dot product; tolerance
// 1e-4 * sum|a_k b_k| + 1e-6 per element.
#include "driver_common.h"

#define main SAMPLE_main_disabled
#include "matrixMul.cu"
#undef main

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 4 && argc != 10) {
        std::fprintf(stderr, "usage: %s <N> <tile> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long N = parse_ll(argv[1], "N");
    const long long tile = parse_ll(argv[2], "tile");
    const char *out_path = argv[3];
    if (N <= 0 || N > 16384) fatal("N out of range (1..16384)");
    if (tile != 16 && tile != 32) fatal("tile must be 16 or 32");
    if (N % tile != 0) fatal("N must be a multiple of tile (the sample asserts the same)");
    const Trailer tr = parse_trailer(argc, argv, 4);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const size_t elems = (size_t)N * (size_t)N;
    const size_t bytes = elems * sizeof(float);
    std::vector<float> hA(elems), hB(elems), hC(elems);
    {
        SplitMix ra(kSeed + 1), rb(kSeed + 2);
        for (size_t i = 0; i < elems; ++i) hA[i] = ra.uniform(-1.0f, 1.0f);
        for (size_t i = 0; i < elems; ++i) hB[i] = rb.uniform(-1.0f, 1.0f);
    }
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dA, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dB, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dC, bytes));
    UD_CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemset(dC, 0xFF, bytes));
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    const dim3 threads((unsigned)tile, (unsigned)tile);
    const dim3 grid((unsigned)(N / tile), (unsigned)(N / tile));
    const int n_int = (int)N;
    auto launch_all = [&](cudaStream_t s) {
        if (tile == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, s>>>(dC, dA, dB, n_int, n_int);
        } else {
            MatrixMulCUDA<32><<<grid, threads, 0, s>>>(dC, dA, dB, n_int, n_int);
        }
    };
    execute(stream, tr, launch_all);

    // ---- correctness, after the timed loop and the final sync, never inside the timed region ----
    UD_CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    maybe_write_output(out_path, hC.data(), bytes);
    const int kSamples = 4096;
    SplitMix pick(kSeed + 3);
    long long bad = 0;
    double worst = 0.0;  // largest |diff| / tolerance
    for (int k = 0; k < kSamples; ++k) {
        const size_t row = (size_t)(pick.next() % (uint64_t)N);
        const size_t col = (size_t)(pick.next() % (uint64_t)N);
        double acc = 0.0, abs_acc = 0.0;
        for (size_t kk = 0; kk < (size_t)N; ++kk) {
            const double p = (double)hA[row * N + kk] * (double)hB[kk * N + col];
            acc += p;
            abs_acc += std::fabs(p);
        }
        const double tol = 1e-4 * abs_acc + 1e-6;
        const double diff = std::fabs((double)hC[row * N + col] - acc);
        if (!(diff <= tol)) ++bad;  // NaN fails
        const double ratio = std::isnan(diff) ? (double)INFINITY : diff / tol;
        if (ratio > worst) worst = ratio;
    }
    const std::string msg =
        fmt("matmul N=%lld tile=%lld sampled=%d bad=%lld worst_diff_over_tol=%.3g repeat=%lld graph_batch=%lld", N, tile,
            kSamples, bad, worst, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dA));
    UD_CUDA_CHECK(cudaFree(dB));
    UD_CUDA_CHECK(cudaFree(dC));
    return finish(out_path, bad == 0, msg);
}
