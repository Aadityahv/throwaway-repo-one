// Driver of the unseen machine-learning kernel set (src/ml_kernels.cuh). One binary, one family per invocation:
//
//   driver_ml gelu    <variant 0|1> <n>                 <out.bin> [--repeat N --graph-batch B --trace-dir DIR]
//   driver_ml swiglu  <variant 0|1> <n>                 <out.bin> [...]
//   driver_ml rmsnorm <variant 0|1> <rows> <cols>       <out.bin> [...]
//   driver_ml rope    <variant 0|1> <seq> <batch>       <out.bin> [...]      (8 heads, head_dim 128)
//   driver_ml sgemm   <variant 0|1> <M> <N> <K>         <out.bin> [...]
//   driver_ml attn    <variant 0|1> <BH> <S>            <out.bin> [...]      (fused attention, head_dim 64, bf16 Q/K/V [BH][S][64], fp32 O)
//   driver_ml tcgemm  <variant 0|1> <M> <N> <K>         <out.bin> [...]      (bf16 inputs A[M][K], B[N][K]; fp32 output C[M][N] = A B^T)
//
// variant 0 / 1 are the two candidates of each cell (see cells_f.py). One operator call = one kernel launch. Trailer and device guard as in driver_common.h.
// Inputs come from splitmix64 streams (kSeed + buffer id); outputs are pre-filled with 0xFF bytes (NaN). The check compares a fixed strided sample of the output
// (every 4093rd element plus the last 1024; whole rows or rotated vectors for the row kernels; 4096 outputs for the matrix multiply) with a double-precision CPU reference.
#include "driver_common.h"
#include "ml_kernels.cuh"
#include "tc_kernels.cuh"   // tensor-core matrix multiply (fresh set G); src/ of fresh_g is on the include path
#include "attn_kernels.cuh" // fused attention (fresh set H); src/ of fresh_h is on the include path

using namespace unseen;

template <class T> static T* dalloc(size_t n) { T* p = nullptr; UD_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }
static void up(void* d, const void* h, size_t bytes) { UD_CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); }
static const long long kStride = 4093;

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
    int ndims = fam == "gelu" || fam == "swiglu" ? 1 : (fam == "sgemm" || fam == "tcgemm") ? 3 : fam == "rmsnorm" || fam == "rope" || fam == "attn" ? 2 : -1;
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

    if (fam == "gelu" || fam == "swiglu") {
        const long long n = d[0];
        if (n <= 0 || n % 1024 != 0) fatal("n must be a positive multiple of 1024");
        const bool two = fam == "swiglu";
        std::vector<float> ha(n), hb(two ? n : 0), hy(n);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2); for (long long i = 0; i < n; ++i) ha[i] = r1.uniform(-4.0f, 4.0f); if (two) for (long long i = 0; i < n; ++i) hb[i] = r2.uniform(-1.0f, 1.0f); }
        float *da = dalloc<float>(n), *db = two ? dalloc<float>(n) : nullptr, *dy = dalloc<float>(n);
        up(da, ha.data(), n * 4); if (two) up(db, hb.data(), n * 4); UD_CUDA_CHECK(cudaMemset(dy, 0xFF, n * 4));
        auto launch = [&](cudaStream_t s) {
            if (!two) { if (variant == 0) gelu_s<<<(unsigned)(n / 256), 256, 0, s>>>(da, dy); else gelu_v4<<<(unsigned)(n / 1024), 256, 0, s>>>((float4*)da, (float4*)dy); }
            else      { if (variant == 0) swiglu_s<<<(unsigned)(n / 256), 256, 0, s>>>(da, db, dy); else swiglu_v4<<<(unsigned)(n / 1024), 256, 0, s>>>((float4*)da, (float4*)db, (float4*)dy); }
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hy.data(), (size_t)n * 4);
        auto check = [&](long long i) {
            const double x = ha[i];
            const double ref = two ? (x / (1.0 + std::exp(-x))) * (double)hb[i] : 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            cmp(R, i, hy[i], ref, 5e-5, 5e-5);
        };
        for (long long i = 0; i < n; i += kStride) check(i);
        for (long long i = std::max(0LL, n - 1024); i < n; ++i) check(i);
        desc = fmt("%s variant=%d n=%lld", fam.c_str(), variant, n);
        cudaFree(da); if (db) cudaFree(db); cudaFree(dy);
    } else if (fam == "rmsnorm") {
        const long long rows = d[0], cols = d[1];
        if (rows <= 0 || cols <= 0 || cols % 512 != 0) fatal("rows > 0 and cols a positive multiple of 512 required");
        const size_t n = (size_t)rows * cols;
        std::vector<float> hx(n), hw(cols), hy(n);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2); for (size_t i = 0; i < n; ++i) hx[i] = r1.uniform(-2.0f, 2.0f); for (long long i = 0; i < cols; ++i) hw[i] = r2.uniform(0.5f, 1.5f); }
        float *dx = dalloc<float>(n), *dw = dalloc<float>(cols), *dy = dalloc<float>(n);
        up(dx, hx.data(), n * 4); up(dw, hw.data(), cols * 4); UD_CUDA_CHECK(cudaMemset(dy, 0xFF, n * 4));
        const float inv_cols = 1.0f / (float)cols, eps = 1e-5f;
        auto launch = [&](cudaStream_t s) {
            if (variant == 0) rmsnorm_s<256><<<(unsigned)rows, 256, 0, s>>>(dx, dw, dy, (int)cols, inv_cols, eps);
            else rmsnorm_v4<128><<<(unsigned)rows, 128, 0, s>>>((float4*)dx, (float4*)dw, (float4*)dy, (int)(cols / 4), inv_cols, eps);
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hy.data(), n * 4);
        std::vector<long long> rs; for (long long r = 0; r < rows; r += std::max(1LL, rows / 61)) rs.push_back(r); rs.push_back(rows - 1);
        for (long long r : rs) {
            double ss = 0; for (long long c = 0; c < cols; ++c) { const double v = hx[(size_t)r * cols + c]; ss += v * v; }
            const double inv = 1.0 / std::sqrt(ss / (double)cols + 1e-5);
            for (long long c = 0; c < cols; ++c) cmp(R, r * cols + c, hy[(size_t)r * cols + c], (double)hx[(size_t)r * cols + c] * inv * hw[c], 5e-5, 5e-5);
        }
        desc = fmt("rmsnorm variant=%d rows=%lld cols=%lld", variant, rows, cols);
        cudaFree(dx); cudaFree(dw); cudaFree(dy);
    } else if (fam == "rope") {
        const long long seq = d[0], batch = d[1]; constexpr int H = 8;
        if (seq <= 0 || batch <= 0) fatal("seq and batch must be positive");
        const size_t n = (size_t)batch * seq * H * 128, nt = (size_t)seq * 64;
        std::vector<float> hx(n), hc(nt), hs(nt), hy(n);
        { SplitMix r1(kSeed + 1); for (size_t i = 0; i < n; ++i) hx[i] = r1.uniform(-1.0f, 1.0f); }
        for (long long s = 0; s < seq; ++s) for (int i = 0; i < 64; ++i) { const double th = (double)s * std::pow(10000.0, -(double)i / 64.0); hc[(size_t)s * 64 + i] = (float)std::cos(th); hs[(size_t)s * 64 + i] = (float)std::sin(th); }
        float *dx = dalloc<float>(n), *dc = dalloc<float>(nt), *ds = dalloc<float>(nt), *dy = dalloc<float>(n);
        up(dx, hx.data(), n * 4); up(dc, hc.data(), nt * 4); up(ds, hs.data(), nt * 4); UD_CUDA_CHECK(cudaMemset(dy, 0xFF, n * 4));
        auto launch = [&](cudaStream_t s) {
            if (variant == 0) rope_all<H><<<dim3((unsigned)seq, (unsigned)batch), H * 64, 0, s>>>(dx, dc, ds, dy);
            else rope_one<H><<<dim3((unsigned)seq, (unsigned)(batch * H)), 64, 0, s>>>(dx, dc, ds, dy);
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hy.data(), n * 4);
        auto check = [&](size_t idx) {                       // idx indexes an element; rebuild (token, head, dim) and the rotated partner
            const size_t vec = idx / 128; const int dd = (int)(idx % 128), i = dd % 64; const size_t tok = vec / H; const long long s = (long long)(tok % (size_t)seq);
            const double c = hc[(size_t)s * 64 + i], sn = hs[(size_t)s * 64 + i]; const size_t b0 = vec * 128;
            const double x0 = hx[b0 + i], x1 = hx[b0 + i + 64];
            const double ref = dd < 64 ? x0 * c - x1 * sn : x1 * c + x0 * sn;
            cmp(R, (long long)idx, hy[idx], ref, 1e-5, 1e-5);
        };
        for (size_t i = 0; i < n; i += (size_t)kStride) check(i);
        for (size_t i = n > 1024 ? n - 1024 : 0; i < n; ++i) check(i);
        desc = fmt("rope variant=%d seq=%lld batch=%lld heads=8", variant, seq, batch);
        cudaFree(dx); cudaFree(dc); cudaFree(ds); cudaFree(dy);
    } else if (fam == "attn") {
        const long long BH = d[0], S = d[1]; constexpr int D = 64;
        const long long qt = variant == 0 ? 64 : 128;
        if (BH <= 0 || S <= 0 || S % 64 || S % qt) fatal("S must be a multiple of 64 and of the query tile");
        const size_t n = (size_t)BH * S * D;
        std::vector<__nv_bfloat16> hq(n), hk(n), hv(n); std::vector<float> ho(n);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2), r3(kSeed + 3); for (auto& v : hq) v = __float2bfloat16_rn(r1.uniform(-1.0f, 1.0f)); for (auto& v : hk) v = __float2bfloat16_rn(r2.uniform(-1.0f, 1.0f)); for (auto& v : hv) v = __float2bfloat16_rn(r3.uniform(-0.5f, 0.5f)); }
        __nv_bfloat16 *dq = dalloc<__nv_bfloat16>(n), *dk = dalloc<__nv_bfloat16>(n), *dv = dalloc<__nv_bfloat16>(n); float* dO = dalloc<float>(n);
        up(dq, hq.data(), n * 2); up(dk, hk.data(), n * 2); up(dv, hv.data(), n * 2); UD_CUDA_CHECK(cudaMemset(dO, 0xFF, n * 4));
        auto launch = [&](cudaStream_t s) {
            if (variant == 0) attn_fwd<4><<<dim3((unsigned)(S / 64), (unsigned)BH), 128, 0, s>>>((uint4*)dq, (uint4*)dk, (uint4*)dv, dO, (unsigned)S);
            else attn_fwd<8><<<dim3((unsigned)(S / 128), (unsigned)BH), 256, 0, s>>>((uint4*)dq, (uint4*)dk, (uint4*)dv, dO, (unsigned)S);
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(ho.data(), dO, n * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, ho.data(), n * 4);
        std::vector<std::pair<long long, long long>> rows;                 // (bh, row): evenly spaced rows of evenly spaced heads, plus the first and last row of the last head
        for (long long b = 0; b < BH; b += std::max(1LL, BH / 6)) for (long long r = 0; r < S; r += std::max(1LL, S / 11) + 1) rows.push_back({b, r});
        rows.push_back({BH - 1, 0}); rows.push_back({BH - 1, S - 1});
        std::vector<double> sc((size_t)S);
        for (auto& br : rows) {
            const size_t qb = ((size_t)br.first * S + br.second) * D; double mx = -1e300;
            for (long long k = 0; k < S; ++k) { double dot = 0; const size_t kb = ((size_t)br.first * S + k) * D; for (int j = 0; j < D; ++j) dot += (double)__bfloat162float(hq[qb + j]) * (double)__bfloat162float(hk[kb + j]); sc[k] = dot * 0.125; mx = std::max(mx, sc[k]); }
            double den = 0; for (long long k = 0; k < S; ++k) { sc[k] = std::exp(sc[k] - mx); den += sc[k]; }
            for (int j = 0; j < D; ++j) { double acc = 0; for (long long k = 0; k < S; ++k) acc += sc[k] * (double)__bfloat162float(hv[((size_t)br.first * S + k) * D + j]); cmp(R, (long long)(qb + j), ho[qb + j], acc / den, 6e-3, 0.0); }
        }
        desc = fmt("attn variant=%d BH=%lld S=%lld D=64", variant, BH, S);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dO);
    } else if (fam == "tcgemm") {
        const long long M = d[0], N = d[1], K = d[2];
        const long long bm = variant == 0 ? 128 : 64;
        if (M <= 0 || N <= 0 || K <= 0 || M % bm || N % bm || K % 32) fatal("M and N must be multiples of the tile and K of 32");
        std::vector<__nv_bfloat16> ha((size_t)M * K), hb((size_t)N * K); std::vector<float> hc((size_t)M * N);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2); for (auto& v : ha) v = __float2bfloat16_rn(r1.uniform(-0.5f, 0.5f)); for (auto& v : hb) v = __float2bfloat16_rn(r2.uniform(-0.5f, 0.5f)); }
        __nv_bfloat16 *da = dalloc<__nv_bfloat16>(ha.size()), *db = dalloc<__nv_bfloat16>(hb.size()); float* dc = dalloc<float>(hc.size());
        up(da, ha.data(), ha.size() * 2); up(db, hb.data(), hb.size() * 2); UD_CUDA_CHECK(cudaMemset(dc, 0xFF, hc.size() * 4));
        auto launch = [&](cudaStream_t s) {
            if (variant == 0) tc_gemm<128, 128, 2, 4><<<dim3((unsigned)(N / 128), (unsigned)(M / 128)), 256, 0, s>>>((uint4*)da, (uint4*)db, dc, (unsigned)N, (unsigned)K);
            else tc_gemm<64, 64, 2, 2><<<dim3((unsigned)(N / 64), (unsigned)(M / 64)), 128, 0, s>>>((uint4*)da, (uint4*)db, dc, (unsigned)N, (unsigned)K);
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hc.data(), hc.size() * 4);
        const size_t total = (size_t)M * N; const size_t step = std::max<size_t>(1, total / 4096);
        auto check = [&](size_t idx) {                       // inputs are the bf16-rounded values; magnitude-relative rule, accumulation in fp32 on the GPU
            const long long m = (long long)(idx / N), nn = (long long)(idx % N); double ref = 0, mag = 0;
            for (long long k = 0; k < K; ++k) { const double p = (double)__bfloat162float(ha[(size_t)m * K + k]) * (double)__bfloat162float(hb[(size_t)nn * K + k]); ref += p; mag += std::fabs(p); }
            const double err = std::fabs((double)hc[idx] - ref) / (mag > 0 ? mag : 1.0); ++R.checked; R.worst = std::max(R.worst, err);
            if (!(std::isfinite(hc[idx]) && err <= 1e-4)) { if (R.bad == 0) R.first_bad = (long long)idx; ++R.bad; }
        };
        for (size_t idx = 0; idx < total; idx += step) check(idx);
        for (size_t idx = total > 64 ? total - 64 : 0; idx < total; ++idx) check(idx);
        desc = fmt("tcgemm variant=%d M=%lld N=%lld K=%lld", variant, M, N, K);
        cudaFree(da); cudaFree(db); cudaFree(dc);
    } else {   // sgemm
        const long long M = d[0], N = d[1], K = d[2];
        const long long bm = variant == 0 ? 64 : 128, bk = variant == 0 ? 16 : 8;
        if (M <= 0 || N <= 0 || K <= 0 || M % bm || N % bm || K % bk) fatal("M and N must be multiples of the tile and K of the k-step");
        std::vector<float> ha((size_t)M * K), hb((size_t)K * N), hc((size_t)M * N);
        { SplitMix r1(kSeed + 1), r2(kSeed + 2); for (auto& v : ha) v = r1.uniform(-0.5f, 0.5f); for (auto& v : hb) v = r2.uniform(-0.5f, 0.5f); }
        float *da = dalloc<float>(ha.size()), *db = dalloc<float>(hb.size()), *dc = dalloc<float>(hc.size());
        up(da, ha.data(), ha.size() * 4); up(db, hb.data(), hb.size() * 4); UD_CUDA_CHECK(cudaMemset(dc, 0xFF, hc.size() * 4));
        auto launch = [&](cudaStream_t s) {
            if (variant == 0) sgemm<64, 64, 16, 4, 4><<<dim3((unsigned)(N / 64), (unsigned)(M / 64)), 256, 0, s>>>(da, db, dc, (unsigned)N, (unsigned)K);
            else sgemm<128, 128, 8, 8, 8><<<dim3((unsigned)(N / 128), (unsigned)(M / 128)), 256, 0, s>>>(da, db, dc, (unsigned)N, (unsigned)K);
        };
        execute(stream, tr, launch);
        UD_CUDA_CHECK(cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost));
        maybe_write_output(out_path, hc.data(), hc.size() * 4);
        const size_t total = (size_t)M * N; const size_t step = std::max<size_t>(1, total / 4096);
        auto check = [&](size_t idx) {                       // magnitude-relative rule of the scalar-product driver: |gpu - ref| <= 1e-4 * sum_k |a_k b_k|
            const long long m = (long long)(idx / N), nn = (long long)(idx % N); double ref = 0, mag = 0;
            for (long long k = 0; k < K; ++k) { const double p = (double)ha[(size_t)m * K + k] * (double)hb[(size_t)k * N + nn]; ref += p; mag += std::fabs(p); }
            const double err = std::fabs((double)hc[idx] - ref) / (mag > 0 ? mag : 1.0); ++R.checked; R.worst = std::max(R.worst, err);
            if (!(std::isfinite(hc[idx]) && err <= 1e-4)) { if (R.bad == 0) R.first_bad = (long long)idx; ++R.bad; }
        };
        for (size_t idx = 0; idx < total; idx += step) check(idx);
        for (size_t idx = total > 64 ? total - 64 : 0; idx < total; ++idx) check(idx);
        desc = fmt("sgemm variant=%d M=%lld N=%lld K=%lld", variant, M, N, K);
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }
    UD_CUDA_CHECK(cudaStreamDestroy(stream));
    const std::string msg = fmt("%s checked=%lld bad=%lld first_bad=%lld worst=%.3g repeat=%lld graph_batch=%lld", desc.c_str(), R.checked, R.bad, R.first_bad, R.worst, tr.repeat, tr.graph_batch);
    return finish(out_path, R.bad == 0, msg);
}
