// Driver for the pinned cuda-samples Black-Scholes kernel (cpp/5_Domain_Specific/BlackScholes/
// BlackScholes_kernel.cuh holds BlackScholesGPU; BlackScholes.cu, which owns the sample's main, is not used).
// The kernel header is included byte-unmodified.
//
//   driver_bs <optN> <block> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//
// One operator call = BlackScholesGPU<<<(optN/2)/block, block>>>((float2*)call, (float2*)put, (float2*)stock,
// (float2*)strike, (float2*)years, 0.02f, 0.30f, optN): one thread per two options, as in the sample's host code
// (the sample uses block 128; 64 is the second candidate). block in {64, 128}; optN even and (optN/2) % block == 0.
// Inputs as in the sample: stock U[5,30], strike U[1,100], years U[0.25,10], splitmix64 streams seeded
// kSeed+1 / +2 / +3. call and put are pre-filled with 0xFF bytes (NaN).
// Check: L1-norm relative error (sum |ref - gpu| / sum |ref| over call and put together) < 1e-5 against a
// double-precision CPU Black-Scholes using the same cumulative-normal polynomial as the sample's gold code.
#include "driver_common.h"

#define main SAMPLE_main_disabled
#include "BlackScholes_kernel.cuh"
#undef main

namespace {

// Same formulas as the sample's BlackScholes_gold.cpp, written independently, in double precision.
double cnd_cpu(double d) {
    const double A1 = 0.31938153, A2 = -0.356563782, A3 = 1.781477937, A4 = -1.821255978, A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;
    const double K = 1.0 / (1.0 + 0.2316419 * std::fabs(d));
    double cnd = RSQRT2PI * std::exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0) cnd = 1.0 - cnd;
    return cnd;
}

void black_scholes_cpu(double &call, double &put, float Sf, float Xf, float Tf, float Rf, float Vf) {
    const double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    const double d2 = d1 - V * sqrtT;
    const double cnd1 = cnd_cpu(d1);
    const double cnd2 = cnd_cpu(d2);
    const double expRT = std::exp(-R * T);
    call = S * cnd1 - X * expRT * cnd2;
    put = X * expRT * (1.0 - cnd2) - S * (1.0 - cnd1);
}

}  // namespace

int main(int argc, char **argv) {
    using namespace unseen;
    if (argc != 4 && argc != 10) {
        std::fprintf(stderr, "usage: %s <optN> <block> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]);
        return 2;
    }
    const long long optN = parse_ll(argv[1], "optN");
    const long long block = parse_ll(argv[2], "block");
    const char *out_path = argv[3];
    if (optN <= 0 || optN > (1LL << 30)) fatal("optN out of range");
    if (block != 64 && block != 128) fatal("block must be 64 or 128");
    if (optN % 2 != 0 || (optN / 2) % block != 0) fatal("(optN/2) must be a multiple of block (no partial tail block)");
    const Trailer tr = parse_trailer(argc, argv, 4);
    remove_stale_outputs(out_path);
    guard_single_approved_device();

    const float kRiskfree = 0.02f, kVolatility = 0.30f;  // RISKFREE / VOLATILITY of the sample
    const size_t n = (size_t)optN;
    const size_t bytes = n * sizeof(float);
    std::vector<float> hStock(n), hStrike(n), hYears(n), hCall(n), hPut(n);
    {
        SplitMix r1(kSeed + 1), r2(kSeed + 2), r3(kSeed + 3);
        for (size_t i = 0; i < n; ++i) hStock[i] = r1.uniform(5.0f, 30.0f);
        for (size_t i = 0; i < n; ++i) hStrike[i] = r2.uniform(1.0f, 100.0f);
        for (size_t i = 0; i < n; ++i) hYears[i] = r3.uniform(0.25f, 10.0f);
    }
    float *dCall = nullptr, *dPut = nullptr, *dStock = nullptr, *dStrike = nullptr, *dYears = nullptr;
    UD_CUDA_CHECK(cudaMalloc(&dCall, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dPut, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dStock, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dStrike, bytes));
    UD_CUDA_CHECK(cudaMalloc(&dYears, bytes));
    UD_CUDA_CHECK(cudaMemcpy(dStock, hStock.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemcpy(dStrike, hStrike.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemcpy(dYears, hYears.data(), bytes, cudaMemcpyHostToDevice));
    UD_CUDA_CHECK(cudaMemset(dCall, 0xFF, bytes));
    UD_CUDA_CHECK(cudaMemset(dPut, 0xFF, bytes));
    UD_CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    UD_CUDA_CHECK(cudaStreamCreate(&stream));
    const unsigned blocks = (unsigned)((optN / 2) / block);
    const int optN_int = (int)optN;
    auto launch_all = [&](cudaStream_t s) {
        BlackScholesGPU<<<blocks, (unsigned)block, 0, s>>>((float2 *)dCall, (float2 *)dPut, (float2 *)dStock,
                                                           (float2 *)dStrike, (float2 *)dYears, kRiskfree,
                                                           kVolatility, optN_int);
    };
    execute(stream, tr, launch_all);

    // ---- correctness, after the timed loop and the final sync, never inside the timed region ----
    UD_CUDA_CHECK(cudaMemcpy(hCall.data(), dCall, bytes, cudaMemcpyDeviceToHost));
    UD_CUDA_CHECK(cudaMemcpy(hPut.data(), dPut, bytes, cudaMemcpyDeviceToHost));
    if (2 * bytes <= kMaxOutputFileBytes) {
        // out.bin = call array followed by put array
        std::vector<float> both(2 * n);
        std::copy(hCall.begin(), hCall.end(), both.begin());
        std::copy(hPut.begin(), hPut.end(), both.begin() + n);
        maybe_write_output(out_path, both.data(), 2 * bytes);
    }
    double sum_delta = 0.0, sum_ref = 0.0, max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) {
        double rc, rp;
        black_scholes_cpu(rc, rp, hStock[i], hStrike[i], hYears[i], kRiskfree, kVolatility);
        if (!std::isfinite(hCall[i]) || !std::isfinite(hPut[i])) finite = false;
        const double dc = std::fabs(rc - (double)hCall[i]);
        const double dp = std::fabs(rp - (double)hPut[i]);
        sum_delta += dc + dp;
        sum_ref += std::fabs(rc) + std::fabs(rp);
        max_abs = std::max(max_abs, std::max(dc, dp));
    }
    const double l1 = sum_delta / sum_ref;
    const bool ok = finite && std::isfinite(l1) && l1 < 1e-5;
    const std::string msg = fmt("bs optN=%lld block=%lld l1_rel=%.3g max_abs=%.3g finite=%d repeat=%lld graph_batch=%lld",
                                optN, block, l1, max_abs, finite ? 1 : 0, tr.repeat, tr.graph_batch);
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    UD_CUDA_CHECK(cudaFree(dCall));
    UD_CUDA_CHECK(cudaFree(dPut));
    UD_CUDA_CHECK(cudaFree(dStock));
    UD_CUDA_CHECK(cudaFree(dStrike));
    UD_CUDA_CHECK(cudaFree(dYears));
    return finish(out_path, ok, msg);
}
