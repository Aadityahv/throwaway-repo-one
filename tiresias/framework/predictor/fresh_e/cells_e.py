"""Cell definitions for the fresh-set E test (Blackwell): scalar product and fast Walsh transform from the pinned cuda-samples (5443602d).
Geometry follows each sample's host code. Rules (fixed here, before any timing): footprint = bytes of every distinct buffer the launch sequence touches;
tier = L2 iff footprint / L2 capacity < 1, with no cell between 0.4 and 1.5 of capacity; logical bytes per launch = for every kernel launch, the
compulsory bytes (each distinct buffer it reads counted once, each it writes counted once), summed over launches. Sizes keep every loop uniform
(no partial blocks)."""
from __future__ import annotations

FLOAT = 4
REGIMES = ("small", "medium", "large", "xlarge")
SP = {"small": (256, 4096), "medium": (256, 16384), "large": (256, 24576), "xlarge": (512, 65536)}   # (vectorN, elementN); elementN multiple of ACCUM_N = 1024
SP_GRID = {"c1": 128, "c2": 256}                                                                      # blocks of 256 threads; vectorN divisible by grid
FWT_LOG2 = {"small": 18, "medium": 22, "large": 23, "xlarge": 26}                                    # c1: one batch of 2^L; c2: two batches of 2^(L-1)
ELEMENTARY = 11


def _cell(family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls):
    ratio = footprint / l2
    assert not (0.4 <= ratio <= 1.5), (family, regime, cand, ratio)
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="host code of the sample (cells_e.py)", blocks_per_sm="",
               active_sm_fraction="", shape_a=str(shape_a), shape_b=str(shape_b), actual_logical_bytes_per_launch=str(logical), tier=tier)
    return dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), family=family, operator_id=op, regime=regime, candidate_id=cand, kernels=kernels,
                footprint_bytes=footprint, l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical, shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)


def fwt_launches(log2n, m):
    """The sample's fwtBatchGPU(): radix-4 global passes while log2N > 11, then the in-shared kernel."""
    ks, n, mm = [], 1 << log2n, m
    grid = ((1 << log2n) // (4 * 256), m, 1)
    l = log2n
    while l > ELEMENTARY:
        ks.append(dict(kid="fwt2", grid=list(grid), block=[256, 1, 1], args=dict(stride=n // 4)))
        l -= 2; n >>= 2; mm <<= 2
    ks.append(dict(kid="fwt1", grid=[mm, 1, 1], block=[n // 4, 1, 1], dynamic_smem=n * FLOAT, args=dict(log2N=l)))
    return ks


def define_cells(hw):
    l2 = hw["l2_bytes"]
    assert l2 == 134217728, "verified Blackwell L2 capacity changed; re-check the sizes"
    cells = []
    for regime in REGIMES:
        v, e = SP[regime]
        for cand, g in SP_GRID.items():
            assert v % g == 0 and e % 1024 == 0
            k = dict(kid="sp", grid=[g, 1, 1], block=[256, 1, 1], args=dict(vectorN=v, elementN=e))
            fp = (2 * v * e + v) * FLOAT
            cells.append(_cell("sp", "fresh_e_cuda_samples_scalar_product", regime, cand, [k], fp, fp, l2, v, e, dict(vectorN=v, elementN=e, grid=g)))
    for regime in REGIMES:
        l = FWT_LOG2[regime]
        for cand, (lg, m) in (("c1", (l, 1)), ("c2", (l - 1, 2))):
            ks = fwt_launches(lg, m)
            buf = (1 << lg) * m * FLOAT
            fp = buf
            logical = len(ks) * 2 * buf
            cells.append(_cell("fwt", "fresh_e_cuda_samples_fast_walsh_transform", regime, cand, ks, fp, logical, l2, lg, m, dict(log2N=lg, batches=m, launches=len(ks))))
    return cells


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fresh_e_lib as L
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
    for c in define_cells(hw):
        print("%-66s %-4s %.3f fp=%.1f MiB logical=%.1f MiB kernels=%d" % (c["cell_id"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20, c["logical_bytes_per_launch"] / 2**20, len(c["kernels"])))
