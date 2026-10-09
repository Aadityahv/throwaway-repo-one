// Driver for the pinned cuda-samples scalar product (cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_kernel.cuh holds scalarProdGPU;
// scalarProd.cu, which owns the sample's main, is not used). The kernel header is included byte-unmodified.
//
//   driver_sp <vectorN> <elementN> <grid> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// One operator call = scalarProdGPU<<<grid, 256>>>(dC, dA, dB, vectorN, elementN): vectorN pairs of vectors of elementN floats, one dot product each.
// vectorN % grid == 0, elementN % 1024 == 0 (ACCUM_N), 256 threads as in the sample's host code (ThreadN = 256).
// Inputs: A and B uniform [0,1) (splitmix64 streams seeded kSeed+1 / +2); dC is pre-filled with 0xFF bytes (NaN).
// Check: per-vector |gpu - ref| <= 1e-4 * sum_i |a_i b_i| against a double-precision CPU dot product, every vector compared.
#include "driver_common.h"

#include <helper_cuda.h>
#define main SAMPLE_main_disabled
#include "scalarProd_kernel.cuh"
#undef main

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 5 && argc != 11) {
        std::fprintf(stderr, "usage: %s <vectorN> <elementN> <grid> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long V = parse_ll(argv[1], "vectorN"), E = parse_ll(argv[2], "elementN"), G = parse_ll(argv[3], "grid");
    const char *out_path = argv[4];
    if (V <= 0 || E <= 0 || G <= 0 || V % G != 0 || E % 1024 != 0) fatal("need vectorN % grid == 0 and elementN % 1024 == 0");
    const Trailer tr = parse_trailer(argc, argv, 5);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const size_t n = (size_t)V * (size_t)E;
    std::vector<float> hA(n), hB(n), hC((size_t)V);
    {
        SplitMix r1(kSeed + 1), r2(kSeed + 2);
        for (size_t i = 0; i < n; ++i) hA[i] = r1.uniform01();
        for (size_t i = 0; i < n; ++i) hB[i] = r2.uniform01();
    }
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dA, n * sizeof(float)));
    UD_CUDA_CHECK(cudaMalloc(&dB, n * sizeof(float)));
    UD_CUDA_CHECK(cudaMalloc(&dC, (size_t)V * sizeof(float)));
    UD_CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)V * sizeof(float)));
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    auto launch_all = [&](cudaStream_t s) { scalarProdGPU<<<(unsigned)G, 256, 0, s>>>(dC, dA, dB, (int)V, (int)E); };
    execute(stream, tr, launch_all);

    UD_CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    maybe_write_output(out_path, hC.data(), (size_t)V * sizeof(float));
    long long bad = 0, first_bad = -1;
    double worst = 0.0;
    for (long long v = 0; v < V; ++v) {
        double ref = 0.0, mag = 0.0;
        for (long long i = 0; i < E; ++i) {
            const double p = (double)hA[(size_t)(v * E + i)] * (double)hB[(size_t)(v * E + i)];
            ref += p;
            mag += std::fabs(p);
        }
        const double err = std::fabs((double)hC[(size_t)v] - ref) / (mag > 0 ? mag : 1.0);
        worst = std::max(worst, err);
        if (!(err <= 1e-4)) {
            if (bad == 0) first_bad = v;
            ++bad;
        }
    }
    const std::string msg = fmt("scalarProd vectorN=%lld elementN=%lld grid=%lld bad=%lld first_bad=%lld worst_err_over_mag=%.3g repeat=%lld graph_batch=%lld", V, E, G,
                                bad, first_bad, worst, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dA));
    UD_CUDA_CHECK(cudaFree(dB));
    UD_CUDA_CHECK(cudaFree(dC));
    return finish(out_path, bad == 0, msg);
}
