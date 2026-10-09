#!/usr/bin/env python3
"""Cell list of the H100 evaluation (CPU only; no measured value of any kind is read).

Reads the H100 section of HARDWARE_GROUND_TRUTH.md programmatically (never a literal), derives every problem size from the L2 capacity and the SM count with the rule
written in FREEZE.md, and writes `cells_h100.json` byte-for-byte reproducibly:

    python make_cells_h100.py            # (re)write cells_h100.json
    python make_cells_h100.py --check    # exit 1 unless the committed file equals what this script produces

Three kernel sets of the pinned NVIDIA cuda-samples (revision 5443602d), the same sources and launch geometry rules as the Blackwell evaluation sets:
  unseen kernels (matrix multiply, Black-Scholes, scan, separable convolution)  geometry as unseen_kernels/cells.py
  set E (scalar product, fast Walsh transform)                                  geometry as fresh_e/cells_e.py
  set D (transposes, copies, reductions; vector add has no CUDA 12.1 build)      geometry as fresh_d/fresh_d_lib.py
The kernel-launch builders below mirror those files; test_cells_h100.py proves it by reproducing the committed Blackwell cell lists from Blackwell's own dimensions.

Size rule (the same tier rule as every Blackwell set, with the sizes derived instead of typed):
  footprint = bytes of every distinct buffer the launch sequence touches (as in the Blackwell cell files); tier = L2 iff footprint / L2 < 1; no cell between 0.4 and 1.5 of L2.
  Each size regime has a target footprint = TARGET_RATIO[regime] x L2 (small 1/64, medium 1/8, large 3/8, xlarge 2). Per family the allowed dimensions are those the kernels'
  divisibility asserts permit AND whose ratio satisfies the regime's hard band (L2 regimes < 0.4, xlarge >= 1.5); the dimension whose footprint is nearest to the target in
  log-ratio is taken (ties: the smaller). Two candidates per cell as in the Blackwell sets (block size, grid or shape variant).
  Scalar-product / matmul / scan grids: c1 = largest power of two <= SM count (128 on H100; also 128 on Blackwell's 188 SMs), c2 = twice that.
No L1-resident tier is defined: HARDWARE_GROUND_TRUTH.md has no verified H100 L1 capacity (and says not to choose a footprint from the shared-memory field); see FREEZE.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
REPO = SR.parents[2]
GROUND_TRUTH = REPO / "HARDWARE_GROUND_TRUTH.md"
OUT = HERE / "cells_h100.json"
SECTION = "## H100"
SCHEMA = "h100_eval_cells/1"
SOURCE_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
FLOAT = 4
SP_VECTOR_N = 256        # scalar product: number of vector pairs; divisible by both candidate grids (128 and 256)

REGIMES = ("small", "medium", "large", "xlarge")
D_REGIMES = ("l2", "dram")
# target footprint / L2 per regime; hard bands: tier L2 needs ratio < 0.4 (the 0.4 to 1.5 gap is forbidden), xlarge/dram needs ratio >= 1.5
TARGET_RATIO = {"small": 1 / 64, "medium": 1 / 8, "large": 3 / 8, "xlarge": 2.0, "l2": 1 / 8, "dram": 2.0}
GAP = (0.4, 1.5)
GT_ROWS_USED = ("Name", "SM count", "L2 cache size", "Max threads per block / per SM", "Max resident blocks per SM", "Shared mem per SM", "Reserved shared memory per block")


# ------------------------------------------------------------------------------------------------ hardware (read from the file)
def read_h100_hardware(text: str | None = None):
    """-> (hw dict with the keys of extract_features.load_hardware, {row label: the file's row text} for the rows quoted in FREEZE.md)."""
    sys.path.insert(0, str(SR))
    import extract_features as X
    if text is None:
        text = GROUND_TRUTH.read_text(encoding="utf-8")
    hw = X.load_hardware(text, SECTION)
    missing = [k for k, v in hw.items() if v is None]
    if missing:
        raise SystemExit("HARDWARE_GROUND_TRUTH.md H100 section is missing: %s. Verify them live and add them first; nothing is assumed." % ", ".join(missing))
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(SECTION))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    quoted = {}
    for l in lines[start:end]:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if len(cells) >= 3:
                for label in GT_ROWS_USED:
                    if cells[0].startswith(label) and label not in quoted:
                        quoted[label] = "| %s | %s | %s |" % (cells[0], cells[1], cells[2])
    if len(quoted) != len(GT_ROWS_USED):
        raise SystemExit("could not quote every needed H100 row: missing %s" % sorted(set(GT_ROWS_USED) - set(quoted)))
    return hw, quoted


def grid_candidates(sm_count: int):
    """c1 = largest power of two <= SM count, c2 = twice that."""
    g = 1 << (sm_count.bit_length() - 1)
    return g, 2 * g


# ------------------------------------------------------------------------------------------------ the dimension rule
def allowed(ratio: float, regime: str) -> bool:
    if GAP[0] <= ratio < GAP[1]:
        return False
    if regime in ("xlarge", "dram"):
        return ratio >= GAP[1]
    return ratio < GAP[0]


def pick(regime: str, l2: int, candidates):
    """candidates: iterable of (sort_key_dimension, footprint_bytes). Nearest allowed footprint to TARGET_RATIO x L2 in log ratio; ties -> smaller dimension."""
    target = TARGET_RATIO[regime] * l2
    best = None
    for dim, fp in candidates:
        if not allowed(fp / l2, regime):
            continue
        key = (abs(math.log(fp / target)), dim)
        if best is None or key < best[0]:
            best = (key, dim, fp)
    if best is None:
        raise SystemExit("no allowed dimension for regime %s" % regime)
    return best[1], best[2]


# ------------------------------------------------------------------------------------------------ kernel-launch builders (mirror the Blackwell cell files)
def mm_kernels(n, bs):
    assert n % bs == 0
    return [dict(kid="mm%d" % bs, grid=[n // bs, n // bs, 1], block=[bs, bs, 1], args=dict(wA=n, wB=n))]


def bs_kernels(opt, blk):
    assert (opt // 2) % blk == 0
    return [dict(kid="bs", grid=[opt // 2 // blk, 1, 1], block=[blk, 1, 1], args=dict(riskfree=0.02, volatility=0.30, optN=opt))]


def scan_kernels(n, length):
    assert n % length == 0 and 2048 <= length <= 262144 and n % 1024 == 0 and n <= 64 * 1048576
    nb = n // 1024
    top_blocks = -(-nb // 256)
    return [dict(kid="scan_bottom", grid=[nb, 1, 1], block=[256, 1, 1], args=dict(size=1024)),
            dict(kid="scan_top", grid=[top_blocks, 1, 1], block=[256, 1, 1], args=dict(N=nb, arrayLength=length // 1024)),
            dict(kid="scan_update", grid=[nb, 1, 1], block=[256, 1, 1], args={})]


def conv_kernels(w, h):
    assert w % 128 == 0 and h % 64 == 0
    return [dict(kid="conv_rows", grid=[w // 128, h // 4, 1], block=[16, 4, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w)),
            dict(kid="conv_cols", grid=[w // 16, h // 64, 1], block=[16, 8, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w))]


def sp_kernels(vector_n, element_n, grid):
    assert vector_n % grid == 0 and element_n % 1024 == 0
    return [dict(kid="sp", grid=[grid, 1, 1], block=[256, 1, 1], args=dict(vectorN=vector_n, elementN=element_n))]


def fwt_kernels(log2n, m):
    """The sample's fwtBatchGPU(): radix-4 global passes while log2N > 11, then the in-shared kernel (as fresh_e/cells_e.py)."""
    ks, n, mm = [], 1 << log2n, m
    grid = ((1 << log2n) // (4 * 256), m, 1)
    l = log2n
    while l > 11:
        ks.append(dict(kid="fwt2", grid=list(grid), block=[256, 1, 1], args=dict(stride=n // 4)))
        l -= 2
        n >>= 2
        mm <<= 2
    ks.append(dict(kid="fwt1", grid=[mm, 1, 1], block=[n // 4, 1, 1], dynamic_smem=n * FLOAT, args=dict(log2N=l)))
    return ks


# ------------------------------------------------------------------------------------------------ cell assembly
def _cell(set_name, family, op, regime, cand, kernels, footprint, logical, l2, shape_a, shape_b, controls, **extra):
    ratio = footprint / l2
    assert allowed(ratio, regime), (family, regime, cand, ratio)
    main = kernels[-1]
    grid_blocks = main["grid"][0] * main["grid"][1] * main["grid"][2]
    threads = main["block"][0] * main["block"][1] * main["block"][2]
    tier = "L2" if ratio < 1.0 else "DRAM"
    dev = dict(grid_blocks=str(grid_blocks), block_threads=str(threads), geometry_source="host code of the sample (make_cells_h100.py)", blocks_per_sm="",
               active_sm_fraction="", shape_a=str(shape_a), shape_b=str(shape_b), actual_logical_bytes_per_launch=str(logical), tier=tier)
    cell = dict(cell_id="h100/%s/%s/%s" % (op, regime, cand), set=set_name, family=family, operator_id=op, regime=regime, candidate_id=cand, kernels=kernels,
                footprint_bytes=footprint, l2_ratio=ratio, tier=tier, logical_bytes_per_launch=logical, shape_a=shape_a, shape_b=shape_b, controls=controls, _dev=dev)
    cell.update(extra)
    return cell


def unseen_cells(hw):
    l2, sm = hw["l2_bytes"], hw["sm_count"]
    cells = []
    for regime in REGIMES:
        n, _ = pick(regime, l2, [(n, 3 * n * n * FLOAT) for n in range(32, 20001, 32)])                         # N multiple of 32 (both tile sizes)
        for cand, bs in (("c1", 16), ("c2", 32)):
            fp = 3 * n * n * FLOAT
            cells.append(_cell("unseen", "matmul", "unseen_cuda_samples_matrix_multiply", regime, cand, mm_kernels(n, bs), fp, fp, l2, n, bs, dict(N=n, tile=bs)))
    for regime in REGIMES:
        opt, _ = pick(regime, l2, [(o, 5 * o * FLOAT) for o in range(256, 40_000_000, 256)])                      # optN/2 divisible by 128 (and 64)
        for cand, blk in (("c1", 128), ("c2", 64)):
            fp = 5 * opt * FLOAT
            cells.append(_cell("unseen", "bs", "unseen_cuda_samples_black_scholes", regime, cand, bs_kernels(opt, blk), fp, fp, l2, opt, blk, dict(optN=opt, block=blk)))
    scan_len = {"c1": 1 << 12, "c2": 1 << 18}
    for regime in REGIMES:
        n, _ = pick(regime, l2, [(n, 2 * n * FLOAT + (n // 1024) * FLOAT) for n in range(1 << 18, 64 * 1048576 + 1, 1 << 18)])   # multiple of the longest arrayLength
        for cand in ("c1", "c2"):
            length = scan_len[cand]
            ks = scan_kernels(n, length)
            nb = n // 1024
            fp = 2 * n * FLOAT + nb * FLOAT
            logical = (2 * n + 2 * n) * FLOAT
            cells.append(_cell("unseen", "scan", "unseen_cuda_samples_scan", regime, cand, ks, fp, logical, l2, n, length, dict(n=n, array_length=length, batch=n // length)))
    for regime in REGIMES:
        shapes = [((w, h), 3 * w * h * FLOAT) for h in range(128, 12289, 128) for w in range(h, 2 * h + 1, 128)]
        # nearest footprint; ties by the smaller aspect gap, then the smaller height
        target = TARGET_RATIO[regime] * l2
        best = None
        for (w, h), fp in shapes:
            if not allowed(fp / l2, regime):
                continue
            key = (abs(math.log(fp / target)), w - h, h)
            if best is None or key < best[0]:
                best = (key, w, h)
        w0, h0 = best[1], best[2]
        for cand, (w, h) in (("c1", (w0, h0)), ("c2", (h0, w0))):
            fp = 3 * w * h * FLOAT
            cells.append(_cell("unseen", "conv", "unseen_cuda_samples_separable_convolution", regime, cand, conv_kernels(w, h), fp, 4 * w * h * FLOAT, l2, w, h, dict(imageW=w, imageH=h)))
    return cells


def set_e_cells(hw):
    l2, sm = hw["l2_bytes"], hw["sm_count"]
    g1, g2 = grid_candidates(sm)
    vector_n = SP_VECTOR_N
    assert vector_n % g1 == 0 and vector_n % g2 == 0, "vectorN must be divisible by both candidate grids (%d, %d)" % (g1, g2)
    cells = []
    for regime in REGIMES:
        e, _ = pick(regime, l2, [(e, (2 * vector_n * e + vector_n) * FLOAT) for e in range(1024, 400_000, 1024)])
        for cand, g in (("c1", g1), ("c2", g2)):
            fp = (2 * vector_n * e + vector_n) * FLOAT
            cells.append(_cell("set_e", "sp", "fresh_e_cuda_samples_scalar_product", regime, cand, sp_kernels(vector_n, e, g), fp, fp, l2, vector_n, e, dict(vectorN=vector_n, elementN=e, grid=g)))
    for regime in REGIMES:
        lg, _ = pick(regime, l2, [(l, (1 << l) * FLOAT) for l in range(13, 31)])
        for cand, (l, m) in (("c1", (lg, 1)), ("c2", (lg - 1, 2))):
            ks = fwt_kernels(l, m)
            buf = (1 << l) * m * FLOAT
            cells.append(_cell("set_e", "fwt", "fresh_e_cuda_samples_fast_walsh_transform", regime, cand, ks, buf, len(ks) * 2 * buf, l2, l, m, dict(log2N=l, batches=m, launches=len(ks))))
    return cells


# set D: kernels in candidate order per operator (as fresh_d_lib.OPERATORS; vector add has no CUDA 12.1 build and is listed under `excluded_before_freeze`)
D_OPERATORS = {
    "fresh_d_cuda_samples_transpose": ["transposeNaive", "transposeCoalesced", "transposeNoBankConflicts", "transposeDiagonal"],
    "fresh_d_cuda_samples_copy": ["copy", "copySharedMem"],
    "fresh_d_cuda_samples_transposefine": ["transposeFineGrained", "transposeCoarseGrained"],
    "fresh_d_cuda_samples_reduction": ["reduce0", "reduce1", "reduce6"],
    "fresh_d_cuda_samples_reduce2": ["reduce2"],
}
TILE = 32
TILE_VARIANT = {"transposeNaive": 2, "transposeCoalesced": 3, "transposeNoBankConflicts": 4, "transposeDiagonal": 7, "copy": 0, "copySharedMem": 1,
                "transposeCoarseGrained": 5, "transposeFineGrained": 6}
REDUCE_WHICH = {"reduce0": 0, "reduce1": 1, "reduce2": 2, "reduce6": 6}
TILE_RUNNER = {"transposeNaive": ("transpose_runner", "transpose_mod8192", "full_transpose"), "transposeCoalesced": ("transpose_runner", "transpose_mod8192", "full_transpose"),
               "transposeNoBankConflicts": ("transpose_runner", "transpose_mod8192", "full_transpose"), "transposeDiagonal": ("transpose_runner", "transpose_mod8192", "full_transpose"),
               "copy": ("tile_family", "copy_uniform", "identity"), "copySharedMem": ("tile_family", "copy_uniform", "identity"),
               "transposeFineGrained": ("tile_family", "transpose_mod2p24", "kernel_semantics:transposeFineGrained"),
               "transposeCoarseGrained": ("tile_family", "transpose_mod2p24", "kernel_semantics:transposeCoarseGrained")}
REDUCE_THREADS = 256
REDUCE6_BLOCKS = 64      # the retained reduce6<256, true> instantiation needs n a power of two (set_d_port.py note); the sample launches 64 blocks


def d_kernel_entry(kernel, geometry):
    """Kernel entry in the pipeline's form (as port_common/set_d_port.py cells()); sample blocks first, middle and last."""
    grid, block = geometry["grid"], geometry["block"]
    nblocks = grid[0] * grid[1] * grid[2]
    sampled = sorted({0, nblocks // 2, nblocks - 1})
    if kernel.startswith("reduce"):
        args, dyn = dict(n=geometry["n"]), REDUCE_THREADS * FLOAT
    else:
        args, dyn = dict(width=geometry["dim_x"], height=geometry["dim_y"]), 0
    return dict(kid="d_" + kernel, grid=list(grid), block=list(block), args=args, dynamic_smem=dyn, sample_blocks=sampled)


def set_d_cells(hw):
    l2 = hw["l2_bytes"]
    cells = []
    tile_dim = {r: pick(r, l2, [(d, 2 * d * d * FLOAT) for d in range(TILE, 30001, TILE)])[0] for r in D_REGIMES}
    red_n = {r: pick(r, l2, [(n, 4 * n + 4 * (n // REDUCE_THREADS)) for n in range(REDUCE_THREADS, 1 << 29, REDUCE_THREADS)])[0] for r in D_REGIMES}
    red6_n = {r: pick(r, l2, [(1 << k, 4 * (1 << k) + 4 * REDUCE6_BLOCKS) for k in range(12, 30)])[0] for r in D_REGIMES}
    for op, kernels in D_OPERATORS.items():
        for regime in D_REGIMES:
            for i, kernel in enumerate(kernels):
                cand = "c%d" % (i + 1)
                if kernel.startswith("reduce"):
                    n = red6_n[regime] if kernel == "reduce6" else red_n[regime]
                    blocks = REDUCE6_BLOCKS if kernel == "reduce6" else n // REDUCE_THREADS
                    geometry = dict(n=n, threads=REDUCE_THREADS, blocks=blocks, grid=[blocks, 1, 1], block=[REDUCE_THREADS, 1, 1])
                    logical = 4 * n + 4 * blocks
                    shape_a, shape_b = n, 1
                    timing = dict(runner="reduction_runner", numeric_args=[n, REDUCE_THREADS, blocks, REDUCE_WHICH[kernel]], input_kind="reduction_uniform", elements=n,
                                  oracle="sum_double", file_args=["in.bin", "out.bin"])
                else:
                    dim = tile_dim[regime]
                    geometry = dict(dim_x=dim, dim_y=dim, grid=[dim // TILE, dim // TILE, 1], block=[TILE, 16, 1], blocks=(dim // TILE) ** 2, threads=TILE * 16)
                    logical = 2 * dim * dim * FLOAT
                    shape_a, shape_b = (dim, dim) if op.endswith("transpose") else ((dim * dim, 1) if op.endswith("copy") else (dim, 1))
                    runner, kind, oracle = TILE_RUNNER[kernel]
                    timing = dict(runner=runner, numeric_args=[dim, dim, TILE_VARIANT[kernel]], input_kind=kind, elements=dim * dim, oracle=oracle, file_args=["in.bin", "out.bin"])
                cells.append(_cell("set_d", "reduction" if kernel.startswith("reduce") else "tile", op, regime, cand, [d_kernel_entry(kernel, geometry)], logical, logical, l2,
                                   shape_a, shape_b, dict(kernel=kernel), kernel=kernel, geometry=geometry, timing=timing))
    return cells


def excluded_before_freeze(hw):
    """Cells that exist in the Blackwell sets but cannot be part of the H100 list, stated before any measurement (not a data-dependent drop)."""
    l2 = hw["l2_bytes"]
    return [dict(kernel="vecAdd", operator_id="fresh_d_cuda_samples_vector_add", regimes=list(D_REGIMES), candidates=["c1", "c2"], count=4,
                 reason="the pinned vector-add sample does not compile with the project's CUDA 12.1 toolchain on Cluster (fatal error: cuda/cmath: No such file or directory), "
                        "so neither a static analysis from a 12.1 build nor a measurement binary exists; port_common/README.md",
                 counted_as_failure=False, note="a scope limit of the toolchain, listed in the paper; it is not a cell the method failed to predict")]


def build_document(hw=None, quoted=None):
    if hw is None:
        hw, quoted = read_h100_hardware()
    cells = unseen_cells(hw) + set_e_cells(hw) + set_d_cells(hw)
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids))
    g1, g2 = grid_candidates(hw["sm_count"])
    doc = dict(
        schema=SCHEMA, gpu="NVIDIA H100 80GB HBM3 (sm_90)", source_revision=SOURCE_REVISION,
        purpose="Frozen cell list of the H100 evaluation: runtime accuracy on held-out kernels, end-to-end energy accuracy, utility. No measured value is in this file.",
        hardware_from_ground_truth=dict(section=SECTION, file="HARDWARE_GROUND_TRUTH.md", values={k: hw[k] for k in sorted(hw)}, rows_quoted=quoted),
        derivation=dict(
            tier_rule="tier = L2 iff footprint / L2 capacity < 1; no cell with ratio in [0.4, 1.5)",
            target_ratio={k: TARGET_RATIO[k] for k in sorted(TARGET_RATIO)},
            hard_band="L2 regimes (small, medium, large, l2): ratio < 0.4; DRAM regimes (xlarge, dram): ratio >= 1.5",
            selection="per family, the allowed dimension whose footprint is nearest to target_ratio x L2 in log ratio; ties to the smaller",
            grid_candidates=dict(c1=g1, c2=g2, rule="c1 = largest power of two <= SM count; c2 = twice that"),
            l1_tier="none: HARDWARE_GROUND_TRUTH.md has no verified H100 L1 capacity (see FREEZE.md)"),
        excluded_before_freeze=excluded_before_freeze(hw),
        cells=cells)
    return doc


def render(doc) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def markdown_table(doc) -> str:
    """The size table of FREEZE.md: one row per (kernel set, operator, regime); the footprint, its ratio to the L2 capacity and the launch geometry of the first and last kernel launched."""
    rows = ["| Set | Operator | Regime | Tier | Footprint (bytes) | MiB | Footprint / L2 | Controls | Launch (first ... last kernel of one call) |", "|---|---|---|---|---:|---:|---:|---|---|"]
    seen = set()
    for c in doc["cells"]:
        key = (c["operator_id"], c["regime"])
        if key in seen:
            continue
        seen.add(key)
        ks = c["kernels"]
        sel = [ks[0]] if len(ks) == 1 else [ks[0], ks[-1]]
        launch = " ... ".join("%s grid %s block %s" % (k["kid"], "x".join(str(g) for g in k["grid"] if g != 1) or "1", "x".join(str(b) for b in k["block"] if b != 1) or "1") for k in sel)
        ctl = ", ".join("%s=%s" % kv for kv in sorted(c["controls"].items()) if kv[0] not in ("block", "grid", "launches", "tile"))
        rows.append("| %s | %s | %s | %s | %d | %.2f | %.4f | %s | %s |" % (c["set"], c["operator_id"].replace("_cuda_samples", "").replace("unseen_", "").replace("fresh_e_", "").replace("fresh_d_", ""),
                                                                            c["regime"], c["tier"], c["footprint_bytes"], c["footprint_bytes"] / 2**20, c["l2_ratio"], ctl, launch))
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true", help="print the size table used in FREEZE.md")
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed cells_h100.json equals the regenerated one")
    ap.add_argument("--summary", action="store_true", help="print the cell table")
    a = ap.parse_args(argv)
    doc = build_document()
    text = render(doc)
    if a.markdown:
        print(markdown_table(doc))
        return 0
    if a.summary:
        for c in doc["cells"]:
            print("%-70s %-6s %-4s ratio=%.4f fp=%.2f MiB logical=%.2f MiB kernels=%d" % (c["cell_id"], c["set"], c["tier"], c["l2_ratio"], c["footprint_bytes"] / 2**20,
                                                                                          c["logical_bytes_per_launch"] / 2**20, len(c["kernels"])))
    if a.check:
        if not OUT.exists() or OUT.read_bytes() != text.encode("utf-8"):
            print("cells_h100.json differs from what make_cells_h100.py produces from HARDWARE_GROUND_TRUTH.md", file=sys.stderr)
            return 1
        print("cells_h100.json reproduced byte-for-byte: %d cells" % len(doc["cells"]))
        return 0
    OUT.write_bytes(text.encode("utf-8"))
    print("wrote %s: %d cells" % (OUT, len(doc["cells"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
