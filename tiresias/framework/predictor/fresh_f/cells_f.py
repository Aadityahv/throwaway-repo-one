"""Cell definitions of the unseen machine-learning kernel set (Blackwell): GELU, SwiGLU gate, RMSNorm, rotary embedding, FP32 matrix multiply. Rules fixed here, before any
timing or energy (same as unseen_kernels/cells.py): footprint = bytes of every distinct buffer the launch touches; tier = L2 iff footprint / L2 < 1, no cell between 0.4 and 1.5 of
the verified L2 capacity; logical bytes = compulsory bytes (each distinct buffer read counted once, each written counted once); sizes keep every launch exactly tiled."""
from __future__ import annotations

FLOAT = 4
REGIMES = ("small", "medium", "large", "xlarge")
GELU_N = {"small": 1 << 20, "medium": 1 << 22, "large": 1 << 26, "xlarge": 1 << 27}
SWIGLU_N = {"small": 1 << 19, "medium": 1 << 21, "large": 1 << 25, "xlarge": 1 << 26}
RMS_RC = {"small": (1024, 1024), "medium": (2048, 2048), "large": (16384, 4096), "xlarge": (32768, 4096)}       # (rows, cols)
ROPE_SB = {"small": (256, 2), "medium": (2048, 2), "large": (4096, 16), "xlarge": (4096, 32)}                    # (seq, batch); 8 heads, head_dim 128
GEMM_MNK = {"small": (512, 512, 512), "medium": (1024, 1024, 2048), "large": (8192, 8192, 512), "xlarge": (8192, 8192, 1024)}
HEADS, HD = 8, 128
EPS = 1e-5


def _cell(family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls):
    ratio = footprint / l2
    assert not (0.4 <= ratio <= 1.5), (family, regime, cand, ratio)
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="this file (cells_f.py)", blocks_per_sm="", active_sm_fraction="",
               shape_a=str(shape_a), shape_b=str(shape_b), actual_logical_bytes_per_launch=str(logical), tier=tier)
    return dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), family=family, operator_id=op, regime=regime, candidate_id=cand, kernels=kernels, footprint_bytes=footprint,
                l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical, shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)


def define_cells(hw):
    l2 = hw["l2_bytes"]
    assert l2 == 134217728, "verified Blackwell L2 capacity changed; re-check the sizes"
    cells = []
    for regime in REGIMES:                                   # GELU: y = gelu(x); c1 one element per thread, c2 float4 per thread
        n = GELU_N[regime]
        for cand, kid, per in (("c1", "gelu_s", 256), ("c2", "gelu_v4", 1024)):
            k = dict(kid=kid, grid=[n // per, 1, 1], block=[256, 1, 1], args={})
            cells.append(_cell("gelu", "fresh_f_ml_gelu", regime, cand, [k], 2 * n * FLOAT, 2 * n * FLOAT, l2, n, per, dict(n=n, elements_per_thread=per // 256)))
    for regime in REGIMES:                                   # SwiGLU gate: y = silu(a) * b
        n = SWIGLU_N[regime]
        for cand, kid, per in (("c1", "swiglu_s", 256), ("c2", "swiglu_v4", 1024)):
            k = dict(kid=kid, grid=[n // per, 1, 1], block=[256, 1, 1], args={})
            cells.append(_cell("swiglu", "fresh_f_ml_swiglu", regime, cand, [k], 3 * n * FLOAT, 3 * n * FLOAT, l2, n, per, dict(n=n, elements_per_thread=per // 256)))
    for regime in REGIMES:                                   # RMSNorm over rows x cols; c1 256 threads scalar, c2 128 threads float4
        r, c = RMS_RC[regime]
        for cand, kid, thr, args in (("c1", "rms_s", 256, dict(cols=c, inv_cols=1.0 / c, eps=EPS)), ("c2", "rms_v4", 128, dict(cols4=c // 4, inv_cols=1.0 / c, eps=EPS))):
            assert c % (4 * thr) == 0 if kid == "rms_v4" else c % thr == 0
            k = dict(kid=kid, grid=[r, 1, 1], block=[thr, 1, 1], args=args)
            fp = (2 * r * c + c) * FLOAT
            cells.append(_cell("rmsnorm", "fresh_f_ml_rmsnorm", regime, cand, [k], fp, fp, l2, r, c, dict(rows=r, cols=c, threads=thr)))
    for regime in REGIMES:                                   # rotary embedding, 8 heads, head_dim 128; c1 one block per (position, batch), c2 one block per (position, batch, head)
        s, b = ROPE_SB[regime]
        x = b * s * HEADS * HD * FLOAT; tab = s * (HD // 2) * FLOAT
        for cand, kid, grid, thr in (("c1", "rope_all", [s, b, 1], HEADS * 64), ("c2", "rope_one", [s, b * HEADS, 1], 64)):
            k = dict(kid=kid, grid=grid, block=[thr, 1, 1], args={})
            fp = 2 * x + 2 * tab
            cells.append(_cell("rope", "fresh_f_ml_rotary_embedding", regime, cand, [k], fp, fp, l2, s, b, dict(seq=s, batch=b, heads=HEADS)))
    for regime in REGIMES:                                   # FP32 matrix multiply C = A B, c1 64x64 tile (4x4 per thread), c2 128x128 tile (8x8 per thread)
        m, n, kk = GEMM_MNK[regime]
        for cand, kid, bm, bk in (("c1", "sg64", 64, 16), ("c2", "sg128", 128, 8)):
            assert m % bm == 0 and n % bm == 0 and kk % bk == 0
            k = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[256, 1, 1], args=dict(N=n, K=kk))
            fp = (m * kk + kk * n + m * n) * FLOAT
            cells.append(_cell("sgemm", "fresh_f_ml_fp32_matmul", regime, cand, [k], fp, fp, l2, m, n, dict(M=m, N=n, K=kk, tile=bm)))
    return cells


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fresh_f_lib as L
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
    cs = define_cells(hw)
    for c in cs:
        print("%-62s %-4s %.3f fp=%.1f MiB blocks=%d" % (c["cell_id"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20, c["kernels"][-1]["grid"][0] * c["kernels"][-1]["grid"][1]))
    print(len(cs), "cells")
