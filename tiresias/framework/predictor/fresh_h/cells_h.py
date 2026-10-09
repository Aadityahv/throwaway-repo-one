"""Cell definitions of the fused attention set (Blackwell): O = softmax(Q K^T / 8) V per (batch*head), head_dim 64, bf16 Q/K/V, fp32 O. Rules fixed before any timing or energy (same as cells_f.py):
footprint = bytes of every distinct buffer (Q, K, V 2 bytes per element, O 4); tier = L2 iff footprint / L2 < 1, no cell between 0.4 and 1.5 of the verified L2; logical bytes = compulsory bytes
(each of Q, K, V read once, O written once). Candidates: c1 64 query rows per block (4 warps), c2 128 query rows per block (8 warps); key tile 64."""
from __future__ import annotations

REGIMES = ("small", "medium", "large", "xlarge")
BH_S = {"small": (16, 512), "medium": (64, 1024), "large": (256, 2048), "xlarge": (512, 2048)}      # (batch*heads, sequence length)
D = 64


def _cell(family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls):
    ratio = footprint / l2
    assert not (0.4 <= ratio <= 1.5), (family, regime, cand, ratio)
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="this file (cells_h.py)", blocks_per_sm="", active_sm_fraction="",
               shape_a=str(shape_a), shape_b=str(shape_b), actual_logical_bytes_per_launch=str(logical), tier=tier)
    return dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), family=family, operator_id=op, regime=regime, candidate_id=cand, kernels=kernels, footprint_bytes=footprint,
                l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical, shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)


def define_cells(hw):
    l2 = hw["l2_bytes"]
    assert l2 == 134217728, "verified Blackwell L2 capacity changed; re-check the sizes"
    cells = []
    for regime in REGIMES:
        bh, s = BH_S[regime]
        for cand, kid, warps in (("c1", "at4", 4), ("c2", "at8", 8)):
            qt = 16 * warps; assert s % qt == 0 and s % 64 == 0
            kern = dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s))
            fp = 3 * bh * s * D * 2 + bh * s * D * 4
            cells.append(_cell("attn", "fresh_h_ml_fused_attention", regime, cand, [kern], fp, fp, l2, bh, s, dict(BH=bh, S=s, D=D, query_rows_per_block=qt)))
    return cells


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fresh_h_lib as L
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
    for c in define_cells(hw): print("%-60s %-4s %.3f fp=%.1f MiB blocks=%d" % (c["cell_id"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20, c["kernels"][-1]["grid"][0] * c["kernels"][-1]["grid"][1]))
