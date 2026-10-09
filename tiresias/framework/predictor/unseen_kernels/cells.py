"""Cell definitions for the unseen-kernel test (Blackwell). Geometry comes from each sample's host code (pinned
cuda-samples 5443602d); sizes are chosen from the verified L2 capacity, read live from HARDWARE_GROUND_TRUTH.md.

Rules (fixed here, before any timing or energy):
- footprint = bytes of every distinct buffer the launch sequence touches; tier = L2 iff footprint / L2 capacity < 1,
  with no cell between 0.4 and 1.5 of capacity, so no tier is marginal.
- logical bytes per launch = for every kernel in the launch sequence, the compulsory bytes (each distinct buffer
  the kernel reads counted once, each it writes counted once), summed over the kernels. This charges every kernel of
  a multi-kernel launch, which the earlier fresh sets did not do for the padded copy kernel (a flaw found in round B).
- Sizes are chosen so that no block is a partial tail block (all divisibility asserts of the samples hold with margin).
- Candidates: matrix multiply and Black-Scholes differ in launch block size; scan and convolution have compile-time
  block sizes, so their two candidates are two input shapes with the same footprint.
"""
from __future__ import annotations

FLOAT = 4
MM_N = {"small": 352, "medium": 1056, "large": 2080, "xlarge": 4256}               # all multiples of 32
BS_OPTN = {"small": 262144, "medium": 1048576, "large": 2621440, "xlarge": 12582912}  # optN/2 divisible by 128
SCAN_N = {"small": 1 << 18, "medium": 1 << 21, "large": 3 << 21, "xlarge": 1 << 25}
SCAN_L = {"c1": 1 << 12, "c2": 1 << 18}                                              # arrayLength, power of two
CONV_WH = {"small": (512, 384), "medium": (1280, 1024), "large": (2304, 1792), "xlarge": (6144, 4096)}
REGIMES = ("small", "medium", "large", "xlarge")


def _cell(family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls):
    ratio = footprint / l2
    assert not (0.4 <= ratio <= 1.5), (family, regime, cand, ratio)  # no marginal tier
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="host code of the sample (cells.py)",
               blocks_per_sm="", active_sm_fraction="", shape_a=str(shape_a), shape_b=str(shape_b),
               actual_logical_bytes_per_launch=str(logical), tier=tier)
    return dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), family=family, operator_id=op, regime=regime, candidate_id=cand,
                kernels=kernels, footprint_bytes=footprint, l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical,
                shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)


def define_cells(hw):
    l2 = hw["l2_bytes"]
    assert l2 == 134217728, "verified Blackwell L2 capacity changed; re-check the sizes"
    cells = []
    # ---- tiled matrix multiply: C(N x N) = A(N x N) B(N x N); grid (N/bs, N/bs), block (bs, bs)
    for regime in REGIMES:
        n = MM_N[regime]
        for cand, bs in (("c1", 16), ("c2", 32)):
            assert n % bs == 0
            k = dict(kid="mm%d" % bs, grid=[n // bs, n // bs, 1], block=[bs, bs, 1], args=dict(wA=n, wB=n))
            fp = 3 * n * n * FLOAT
            cells.append(_cell("matmul", "unseen_cuda_samples_matrix_multiply", regime, cand, [k], fp, 3 * n * n * FLOAT, l2, n, bs,
                               dict(N=n, tile=bs)))
    # ---- Black-Scholes: one thread per two options; grid DIV_UP(optN/2, block); block 128 (sample) or 64
    for regime in REGIMES:
        opt = BS_OPTN[regime]
        for cand, blk in (("c1", 128), ("c2", 64)):
            assert (opt // 2) % blk == 0
            k = dict(kid="bs", grid=[opt // 2 // blk, 1, 1], block=[blk, 1, 1], args=dict(riskfree=0.02, volatility=0.30, optN=opt))
            fp = 5 * opt * FLOAT
            cells.append(_cell("bs", "unseen_cuda_samples_black_scholes", regime, cand, [k], fp, 5 * opt * FLOAT, l2, opt, blk,
                               dict(optN=opt, block=blk)))
    # ---- prefix-sum scan: bottom-level scan, scan of block sums, uniform update (3 kernels), THREADBLOCK_SIZE 256
    for regime in REGIMES:
        n = SCAN_N[regime]
        for cand in ("c1", "c2"):
            length = SCAN_L[cand]
            assert n % length == 0 and 2048 <= length <= 262144 and n % 1024 == 0 and n <= 64 * 1048576
            nb = n // 1024                         # blocks of the bottom scan and of the update
            top_blocks = -(-nb // 256)
            ks = [dict(kid="scan_bottom", grid=[nb, 1, 1], block=[256, 1, 1], args=dict(size=1024)),
                  dict(kid="scan_top", grid=[top_blocks, 1, 1], block=[256, 1, 1], args=dict(N=nb, arrayLength=length // 1024)),
                  dict(kid="scan_update", grid=[nb, 1, 1], block=[256, 1, 1], args={})]
            fp = 2 * n * FLOAT + nb * FLOAT        # src + dst + block-sum buffer
            logical = (2 * n + 2 * n + 0) * FLOAT  # bottom: read n write n; update: read n write n (block sums negligible)
            cells.append(_cell("scan", "unseen_cuda_samples_scan", regime, cand, ks, fp, logical, l2, n, length,
                               dict(n=n, array_length=length, batch=n // length)))
    # ---- separable convolution (radius 8): rows kernel then columns kernel through a temporary image
    for regime in REGIMES:
        w0, h0 = CONV_WH[regime]
        for cand, (w, h) in (("c1", (w0, h0)), ("c2", (h0, w0))):
            assert w % 128 == 0 and h % 64 == 0
            ks = [dict(kid="conv_rows", grid=[w // 128, h // 4, 1], block=[16, 4, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w)),
                  dict(kid="conv_cols", grid=[w // 16, h // 64, 1], block=[16, 8, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w))]
            fp = 3 * w * h * FLOAT                 # input, temporary, output
            logical = 4 * w * h * FLOAT            # rows: read+write; columns: read+write
            cells.append(_cell("conv", "unseen_cuda_samples_separable_convolution", regime, cand, ks, fp, logical, l2, w, h,
                               dict(imageW=w, imageH=h)))
    return cells


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import unseen_pipeline as UP
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    for c in define_cells(hw):
        print("%-62s %-4s %.3f  MiB fp=%.1f logical=%.1f" % (c["cell_id"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20, c["logical_bytes_per_launch"] / 2**20))
