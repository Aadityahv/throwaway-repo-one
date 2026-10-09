"""Cell definitions of the tensor-core matrix multiply set (Blackwell): C[M,N] (fp32) = A[M,K] (bf16) B[N,K]^T (bf16), the transformer linear layer. Rules fixed here, before any timing or energy
(same as cells_f.py): footprint = bytes of every distinct buffer (A and B are 2 bytes per element, C 4); tier = L2 iff footprint / L2 < 1, no cell between 0.4 and 1.5 of the verified L2 capacity;
logical bytes = compulsory bytes; sizes tile exactly. Candidates: c1 128x128x32 block tile with 8 warps (2x4, 64x32 per warp), c2 64x64x32 with 4 warps (2x2, 32x32 per warp)."""
from __future__ import annotations

REGIMES = ("small", "medium", "large", "xlarge")
GEMM_MNK = {"small": (512, 512, 512), "medium": (1024, 1024, 2048), "large": (8192, 8192, 512), "xlarge": (8192, 8192, 1024)}


def _cell(family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls):
    ratio = footprint / l2
    assert not (0.4 <= ratio <= 1.5), (family, regime, cand, ratio)
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="this file (cells_g.py)", blocks_per_sm="", active_sm_fraction="",
               shape_a=str(shape_a), shape_b=str(shape_b), actual_logical_bytes_per_launch=str(logical), tier=tier)
    return dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), family=family, operator_id=op, regime=regime, candidate_id=cand, kernels=kernels, footprint_bytes=footprint,
                l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical, shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)


def define_cells(hw):
    l2 = hw["l2_bytes"]
    assert l2 == 134217728, "verified Blackwell L2 capacity changed; re-check the sizes"
    cells = []
    for regime in REGIMES:
        m, n, k = GEMM_MNK[regime]
        for cand, kid, bm, thr in (("c1", "tc128", 128, 256), ("c2", "tc64", 64, 128)):
            assert m % bm == 0 and n % bm == 0 and k % 32 == 0
            kern = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k))
            fp = (m * k + n * k) * 2 + m * n * 4
            cells.append(_cell("tcgemm", "fresh_g_ml_tensor_matmul", regime, cand, [kern], fp, fp, l2, m, n, dict(M=m, N=n, K=k, tile=bm)))
    return cells


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fresh_g_lib as L
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
    for c in define_cells(hw): print("%-60s %-4s %.3f fp=%.1f MiB blocks=%d" % (c["cell_id"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20, c["kernels"][-1]["grid"][0] * c["kernels"][-1]["grid"][1]))
