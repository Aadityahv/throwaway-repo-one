// Driver of the prospective test (kernels in src/prosp_kernels.cuh plus the existing tc_kernels.cuh and attn_kernels.cuh). One binary, one family per invocation:
//   driver_prosp tcgemm2 <variant 0|1> <M> <N> <K>   <out.bin> [--repeat N --graph-batch B --trace-dir DIR]   variant 0: tc_gemm 128x64 (4 warps), 1: 256x128 (16 warps)
//   driver_prosp attn2   <variant 0|1> <BH> <S>      <out.bin> [...]   variant 0: attn_fwd<2> (32 query rows), 1: attn_fwd<16> (256 query rows)
//   driver_prosp attnnm  <variant 0|1> <BH> <S>      <out.bin> [...]   attention without running maximum: variant 0: 4 warps, 1: 8 warps
//   driver_prosp tcrelu  <variant 0|1> <M> <N> <K>   <out.bin> [...]   matrix multiply with bias and ReLU epilogue: variant 0: 128x128 (8 warps), 1: 64x64 (4 warps)
// Inputs, checks and trailer follow driver_ml.cu (splitmix64 streams, 0xFF-filled outputs, strided double-precision CPU reference check).
#include "driver_common.h"
#include "prosp_kernels.cuh"

using namespace unseen;

template <class T> static T* dalloc(size_t n) { T* p = nullptr; UD_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }
static void up(void* d, const void* h, size_t bytes) { UD_CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); }

struct Result { long long bad = 0, checked = 0, first_bad = -1; double worst = 0; };
static void cmp(Result& r, long long idx, double got, double ref, double tol_abs, double tol_rel) {
    ++r.checked;
    const double err = std::fabs(got - ref), lim = tol_abs + tol_rel * std::fabs(ref);
    const bool ok = std::isfinite(got) && err <= lim;
    if (std::isfinite(got)) r.worst = std::max(r.worst, err / lim);
    if (!ok) { if (r.bad == 0) r.first_bad = idx; ++r.bad; }
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <family> <variant> <dims...> <out.bin> [--repeat N --graph-batch B --trace-dir DIR]\n", argv[0]); return 2; }
    const std::string fam = argv[1];
    const int variant = (int)parse_ll(argv[2], "variant");
    if (variant != 0 && variant != 1) fatal("variant must be 0 or 1");
    int ndims = (fam == "tcgemm2" || fam == "tcrelu") ? 3 : (fam == "attn2" || fam == "attnnm") ? 2 : -1;
    if (ndims < 0) fatal("unknown family");
    if (argc != 3 + ndims + 1 && argc != 3 + ndims + 1 + 6) fatal("wrong number of arguments for this family");
    long long d[3] = {0, 0, 0};
    for (int i = 0; i < ndims; ++i) d[i] = parse_ll(argv[3 + i], "dim");
    const char* out_path = argv[3 + ndims];
    const Trailer tr = parse_trailer(argc, argv, 4 + ndims);
    remove_stale_outputs(out_path);
    guard_single_approved_device();
    cudaStream_t stream; UD_CUDA_CHECK(cudaStreamCreate(&stream));
    Result R; std::string desc;

    if (fam == "attn2" || fam == "attnnm") {
        const bool nomax = fam == "attnnm";
        const long long BH = d[0], S = d[1]; constexpr int D = 64;
        const int warps = nomax ? (variant == 0 ? 4 : 8) : (variant == 0 ? 2 : 16);
        const long long qt = 16 * warps;
        if (BH <= 0 || S <= 0 || S % 64 || S % qt) fatal("S must be a multiple of 64 and of the query tile");
        const size_t n = (size_t)BH * S * D;
        std::vector<__nv_bfloat16> hq(n), hk(n), hv(n); std::vector<float> ho(n);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2), r3(kSeed + 3); for (auto& v : hq) v = __float2bfloat16_rn(r1.uniform(-1.0f, 1.0f)); for (auto& v : hk) v = __float2bfloat16_rn(r2.uniform(-1.0f, 1.0f)); for (auto& v : hv) v = __float2bfloat16_rn(r3.uniform(-0.5f, 0.5f)); }
        __nv_bfloat16 *dq = dalloc<__nv_bfloat16>(n), *dk = dalloc<__nv_bfloat16>(n), *dv = dalloc<__nv_bfloat16>(n); float* dO = dalloc<float>(n);
        up(dq, hq.data(), n * 2); up(dk, hk.data(), n * 2); up(dv, hv.data(), n * 2); UD_CUDA_CHECK(cudaMemset(dO, 0xFF, n * 4));
        const dim3 grid((unsigned)(S / qt), (unsigned)BH);
        auto launch = [&](cudaStream_t s) {
            uint4 *q = (uint4*)dq, *k = (uint4*)dk, *v = (uint4*)dv;
            if (!nomax) { if (variant == 0) attn_fwd<2><<<grid, 64, 0, s>>>(q, k, v, dO, (unsigned)S); else attn_fwd<16><<<grid, 512, 0, s>>>(q, k, v, dO, (unsigned)S); }
            else { if (variant == 0) attn_nomax<4><<<grid, 128, 0, s>>>(q, k, v, dO, (unsigned)S); else attn_nomax<8><<<grid, 256, 0, s>>>(q, k, v, dO, (unsigned)S); }
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(ho.data(), dO, n * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, ho.data(), n * 4);
        std::vector<std::pair<long long, long long>> rows;
        for (long long b = 0; b < BH; b += std::max(1LL, BH / 6)) for (long long r = 0; r < S; r += std::max(1LL, S / 11) + 1) rows.push_back({b, r});
        rows.push_back({BH - 1, 0}); rows.push_back({BH - 1, S - 1});
        std::vector<double> sc((size_t)S);
        for (auto& br : rows) {
            const size_t qb = ((size_t)br.first * S + br.second) * D; double mx = nomax ? 0.0 : -1e300;   // the no-maximum kernel exponentiates the raw scores (bounded, no overflow)
            for (long long k = 0; k < S; ++k) { double dot = 0; const size_t kb = ((size_t)br.first * S + k) * D; for (int j = 0; j < D; ++j) dot += (double)__bfloat162float(hq[qb + j]) * (double)__bfloat162float(hk[kb + j]); sc[k] = dot * 0.125; if (!nomax) mx = std::max(mx, sc[k]); }
            double den = 0; for (long long k = 0; k < S; ++k) { sc[k] = std::exp(sc[k] - mx); den += sc[k]; }
            for (int j = 0; j < D; ++j) { double acc = 0; for (long long k = 0; k < S; ++k) acc += sc[k] * (double)__bfloat162float(hv[((size_t)br.first * S + k) * D + j]); cmp(R, (long long)(qb + j), ho[qb + j], acc / den, 6e-3, 0.0); }
        }
        desc = fmt("%s variant=%d BH=%lld S=%lld D=64", fam.c_str(), variant, BH, S);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dO);
    } else {   // tcgemm2, tcrelu
        const bool relu = fam == "tcrelu";
        const long long M = d[0], N = d[1], K = d[2];
        const long long bm = relu ? (variant == 0 ? 128 : 64) : (variant == 0 ? 128 : 256), bn = relu ? bm : (variant == 0 ? 64 : 128);
        if (M <= 0 || N <= 0 || K <= 0 || M % bm || N % bn || K % 32) fatal("M and N must be multiples of the tile and K of 32");
        std::vector<__nv_bfloat16> ha((size_t)M * K), hb((size_t)N * K); std::vector<float> hc((size_t)M * N), hbias((size_t)N);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2), r3(kSeed + 3); for (auto& v : ha) v = __float2bfloat16_rn(r1.uniform(-0.5f, 0.5f)); for (auto& v : hb) v = __float2bfloat16_rn(r2.uniform(-0.5f, 0.5f)); for (auto& v : hbias) v = r3.uniform(-1.0f, 1.0f); }
        __nv_bfloat16 *da = dalloc<__nv_bfloat16>(ha.size()), *db = dalloc<__nv_bfloat16>(hb.size()); float *dc = dalloc<float>(hc.size()), *dbias = dalloc<float>(hbias.size());
        up(da, ha.data(), ha.size() * 2); up(db, hb.data(), hb.size() * 2); up(dbias, hbias.data(), hbias.size() * 4); UD_CUDA_CHECK(cudaMemset(dc, 0xFF, hc.size() * 4));
        const dim3 grid((unsigned)(N / bn), (unsigned)(M / bm));
        auto launch = [&](cudaStream_t s) {
            uint4 *a = (uint4*)da, *b = (uint4*)db;
            if (!relu) { if (variant == 0) tc_gemm<128, 64, 2, 2><<<grid, 128, 0, s>>>(a, b, dc, (unsigned)N, (unsigned)K); else tc_gemm<256, 128, 4, 4><<<grid, 512, 0, s>>>(a, b, dc, (unsigned)N, (unsigned)K); }
            else { if (variant == 0) tc_gemm_bias_relu<128, 128, 2, 4><<<grid, 256, 0, s>>>(a, b, dbias, dc, (unsigned)N, (unsigned)K); else tc_gemm_bias_relu<64, 64, 2, 2><<<grid, 128, 0, s>>>(a, b, dbias, dc, (unsigned)N, (unsigned)K); }
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hc.data(), hc.size() * 4);
        const size_t total = (size_t)M * N; const size_t step = std::max<size_t>(1, total / 4096);
        auto check = [&](size_t idx) {                       // magnitude-relative rule: |gpu - ref| <= 1e-4 * (sum_k |a_k b_k| + |bias|)
            const long long m = (long long)(idx / N), nn = (long long)(idx % N); double ref = 0, mag = 0;
            for (long long k = 0; k < K; ++k) { const double p = (double)__bfloat162float(ha[(size_t)m * K + k]) * (double)__bfloat162float(hb[(size_t)nn * K + k]); ref += p; mag += std::fabs(p); }
            if (relu) { ref = std::max(ref + (double)hbias[nn], 0.0); mag += std::fabs((double)hbias[nn]); }
            const double err = std::fabs((double)hc[idx] - ref) / (mag > 0 ? mag : 1.0); ++R.checked; R.worst = std::max(R.worst, err);
            if (!(std::isfinite(hc[idx]) && err <= 1e-4)) { if (R.bad == 0) R.first_bad = (long long)idx; ++R.bad; }
        };
        for (size_t idx = 0; idx < total; idx += step) check(idx);
        for (size_t idx = total > 64 ? total - 64 : 0; idx < total; ++idx) check(idx);
        desc = fmt("%s variant=%d M=%lld N=%lld K=%lld", fam.c_str(), variant, M, N, K);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dbias);
    }
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    const std::string msg = fmt("%s checked=%lld bad=%lld first_bad=%lld worst=%.3g repeat=%lld graph_batch=%lld", desc.c_str(), R.checked, R.bad, R.first_bad, R.worst, tr.repeat, tr.graph_batch);
    return finish(out_path, R.bad == 0, msg);
}
