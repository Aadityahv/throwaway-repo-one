#!/usr/bin/env python3
"""Ada evaluation cell lists (CPU only; no measured value is read). Byte-reproducible from the Ada section of HARDWARE_GROUND_TRUTH.md.

    python make_cells_ada.py            # (re)write cells_ada_<group>.json
    python make_cells_ada.py --check    # exit 1 unless the committed files equal what this script produces
    python make_cells_ada.py --summary  # cell counts per set and tier
    python make_cells_ada.py --table    # one row per (operator, regime): tier, footprint, footprint / L2

Groups and files (same kernels, candidates and launch builders as Blackwell; H100 generators reused unchanged):
  samples     cells_ada_samples.json     unseen classic kernels (32), scalar product and fast Walsh transform (16), CUDA samples at new shapes incl. vector add (28)  = 76
  ml          cells_ada_ml.json          GELU, SwiGLU gate, RMSNorm, rotary, FP32 matrix multiply (40)
  tensor      cells_ada_tensor.json      tensor-core matrix multiply (8), fused attention (8)
  validation  cells_ada_validation.json  the 12 validation cells (6 shapes x 2 candidates)
  prospective cells_ada_prospective.json the 24 prospective cells (tensor matmul new tiles, attention new blocks, attention without maximum, matmul with bias and ReLU)

Size rules (every hardware value is read from the Ada section at run time):
  samples     the H100 generator (h100_eval_freeze/make_cells_h100.py: unseen_cells, set_e_cells, set_d_cells) with Ada's L2 and SM count. Footprint = bytes of every distinct buffer; per regime a target
              footprint = TARGET_RATIO x L2 (small 1/64, medium 1/8, large 3/8, xlarge 2; set D l2 1/8, dram 2), the allowed dimension nearest to it in log ratio (bands: L2 tier < 0.4, DRAM tier >= 1.5 of L2).
              Grid candidates c1/c2 = largest power of two <= SM count and twice that (64 and 128 on Ada's 100 SMs). Vector add (set D) uses the same rule: n elements, multiple of 1024, footprint 12 n bytes.
  ml, tensor  h100_eval_freeze/ml_sets/make_cells_ml.py (define_ml, define_tensor) with Ada's L2: regime targets = footprint / L2 of the Blackwell cell of the same regime (Blackwell L2 read from its
              section), the scaled dimension is the multiple of its tiling unit nearest the target in log ratio.
  validation, prospective   the Blackwell cell's footprint / L2 ratio is the target and one dimension is re-derived from Ada's L2 (nearest multiple of its tiling unit, same bands); every other dimension
              keeps its structural role. Validation: FP32 and tensor matmul scale M (M=N for FP32; K, N fixed), attention scales batch*heads (S=1536), scalar product scales elementN (vectorN 768),
              RMSNorm scales rows (2560 columns), separable convolution scales the height with the Blackwell aspect ratio (width a multiple of 128). Prospective: matmul scales M=N (multiple of 256, K per
              regime fixed), attention scales batch*heads (S per regime fixed). A cell that cannot land in its Blackwell tier is listed under `unplaceable` (never shrunk after a refusal).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
from pathlib import Path

import board_ada as B

HERE = B.HERE
SR = B.SR
sys.path.insert(0, str(B.FZ / "ml_sets"))
import make_cells_h100 as MC  # noqa: E402
import make_cells_ml as MM  # noqa: E402

SCHEMA = "ada_eval_cells/1"
FLOAT = 4
GROUPS = ("samples", "ml", "tensor", "validation", "prospective")
FILES = {g: "cells_ada_%s.json" % g for g in GROUPS}
GAP = MM.GAP
RULE_TEXT = __doc__.split("Size rules")[1].strip()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lib(name, rel):
    sys.path.insert(0, str(SR / rel))
    return _load(name + "_for_ada", SR / rel / (name + ".py"))


CE = _lib("cells_e", "fresh_e")
CK = _lib("cells", "unseen_kernels")
CF, CG, CH = MM.CF, MM.CG, MM.CH


def _rename(c):
    c = dict(c)
    c["cell_id"] = "%s/%s" % (B.PREFIX, c["cell_id"].partition("/")[2])
    if "_dev" in c:
        c["_dev"] = dict(c["_dev"], geometry_source="make_cells_ada.py (derived from Ada's L2 and SM count in HARDWARE_GROUND_TRUTH.md)")
    if "l2_ratio" in c:
        c["footprint_over_l2"] = c.pop("l2_ratio")
    return c


# ---------------------------------------------------------------------------------------------- samples
def vecadd_cells(hw):
    """Vector add of set D (candidates 128 and 1024 threads), same size rule as the other set D families."""
    l2 = hw["l2_bytes"]
    cells = []
    for regime in MC.D_REGIMES:
        n, _ = MC.pick(regime, l2, [(n, 12 * n) for n in range(1024, 1 << 29, 1024)])
        for cand, thr in (("c1", 128), ("c2", 1024)):
            assert n % thr == 0
            blocks = n // thr
            geometry = dict(n=n, threads=thr, blocks=blocks, grid=[blocks, 1, 1], block=[thr, 1, 1])
            logical = 3 * n * FLOAT
            entry = dict(kid="d_vecAdd", grid=[blocks, 1, 1], block=[thr, 1, 1], args=dict(vectorLength=n), dynamic_smem=0, sample_blocks=sorted({0, blocks // 2, blocks - 1}))
            timing = dict(runner="copy_runner", numeric_args=[n, thr], input_kind="vecadd_uniform_pair", elements=n, oracle="vector_add", file_args=["x.bin", "y.bin", "out.bin"])
            cells.append(MC._cell("set_d", "vecadd", "fresh_d_cuda_samples_vector_add", regime, cand, [entry], logical, logical, l2, n, 1, dict(kernel="vecAdd"), kernel="vecAdd",
                                  geometry=geometry, timing=timing))
    return cells


def samples_cells(hw):
    cells = MC.unseen_cells(hw) + MC.set_e_cells(hw) + vecadd_cells(hw) + MC.set_d_cells(hw)
    out = []
    for c in cells:
        c = dict(c)
        c["cell_id"] = c["cell_id"].replace("h100/", "%s/" % B.PREFIX, 1)
        c["footprint_over_l2"] = c.pop("l2_ratio")
        out.append(c)
    return out


# ---------------------------------------------------------------------------------------------- ml and tensor
def _blackwell_scoped(fn):
    """make_cells_ml.blackwell_targets() uses a typed Blackwell L2 constant; replace it by the Blackwell section's value (read, not typed) for the duration of the call."""
    bw_l2, _ = B.forbidden_blackwell_values()
    old = MM.BLACKWELL_L2
    MM.BLACKWELL_L2 = bw_l2
    try:
        return fn()
    finally:
        MM.BLACKWELL_L2 = old


def ml_cells(hw):
    return _blackwell_scoped(lambda: MM.define_ml(hw["l2_bytes"], B.PREFIX))


def tensor_cells(hw):
    return _blackwell_scoped(lambda: MM.define_tensor(hw["l2_bytes"], B.PREFIX))


# ---------------------------------------------------------------------------------------------- validation and prospective: ratio-preserving
def _blackwell_cells():
    bw, _ = B.forbidden_blackwell_values()
    hwb = dict(l2_bytes=bw)
    sys.path.insert(0, str(SR / "shared_traffic" / "validation"))
    sys.path.insert(0, str(SR / "prospective_test"))
    import cells_validation as V
    import cells_prosp as P
    val = []
    for lib in ("f", "g", "h", "e", "classic"):
        val.extend(V.define(lib, hwb))
    return val, P.define_cells(hwb)


def _ratio_table(cells):
    """(operator_id, regime) -> (footprint / L2, tier) of the Blackwell cell (both candidates share it)."""
    return {(c["operator_id"], c["regime"]): (c["l2_ratio"], c["tier"]) for c in cells}


def scale(unit, footprint_of, ratio, tier, l2):
    """(dimension, footprint): j*unit nearest ratio x L2 in log ratio among those inside the tier's band; ties -> smaller. None if no multiple lands in the band."""
    target = ratio * l2
    best = None
    j = 1
    while True:
        fp = footprint_of(j * unit)
        if fp > 8 * target:
            break
        if MM.tier_ok(fp / l2, tier):
            key = (abs(math.log(fp / target)), j)
            if best is None or key < best[0]:
                best = (key, j * unit, fp)
        j += 1
    return None if best is None else (best[1], best[2])


def validation_cells(hw, bw_val):
    l2 = hw["l2_bytes"]
    g1, g2 = MC.grid_candidates(hw["sm_count"])
    R = _ratio_table(bw_val)
    cells, unplaceable = [], []

    def fail(op, regime, why):
        unplaceable.append(dict(operator_id=op, regime=regime, reason=why))

    r, t = R[("validation_fp32_matmul", "v1")]
    k_ = 512
    got = scale(128, lambda m: (m * k_ + k_ * m + m * m) * FLOAT, r, t, l2)
    if got:
        m, fp = got
        for cand, kid, bm, bk in (("c1", "sg64", 64, 16), ("c2", "sg128", 128, 8)):
            assert m % bm == 0 and k_ % bk == 0
            k = dict(kid=kid, grid=[m // bm, m // bm, 1], block=[256, 1, 1], args=dict(N=m, K=k_))
            cells.append(CF._cell("sgemm", "validation_fp32_matmul", "v1", cand, [k], fp, fp, l2, m, m, dict(M=m, N=m, K=k_, tile=bm)))
    else:
        fail("validation_fp32_matmul", "v1", "no M in the tier band")
    r, t = R[("validation_rmsnorm", "v6")]
    c_ = 2560
    got = scale(64, lambda rows: (2 * rows * c_ + c_) * FLOAT, r, t, l2)
    if got:
        rows, fp = got
        for cand, kid, thr, args in (("c1", "rms_s", 256, dict(cols=c_, inv_cols=1.0 / c_, eps=CF.EPS)), ("c2", "rms_v4", 128, dict(cols4=c_ // 4, inv_cols=1.0 / c_, eps=CF.EPS))):
            k = dict(kid=kid, grid=[rows, 1, 1], block=[thr, 1, 1], args=args)
            cells.append(CF._cell("rmsnorm", "validation_rmsnorm", "v6", cand, [k], fp, fp, l2, rows, c_, dict(rows=rows, cols=c_, threads=thr)))
    else:
        fail("validation_rmsnorm", "v6", "no row count in the tier band")
    r, t = R[("validation_tensor_matmul", "v2")]
    n_, k_ = 4096, 512
    got = scale(128, lambda m: (m * k_ + n_ * k_) * 2 + m * n_ * 4, r, t, l2)
    if got:
        m, fp = got
        for cand, kid, bm, thr in (("c1", "tc128", 128, 256), ("c2", "tc64", 64, 128)):
            assert m % bm == 0 and n_ % bm == 0 and k_ % 32 == 0
            kern = dict(kid=kid, grid=[n_ // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n_, K=k_))
            cells.append(CG._cell("tcgemm", "validation_tensor_matmul", "v2", cand, [kern], fp, fp, l2, m, n_, dict(M=m, N=n_, K=k_, tile=bm)))
    else:
        fail("validation_tensor_matmul", "v2", "no M in the tier band")
    r, t = R[("validation_fused_attention", "v3")]
    s = 1536
    got = scale(1, lambda bh: 3 * bh * s * CH.D * 2 + bh * s * CH.D * 4, r, t, l2)
    if got:
        bh, fp = got
        for cand, kid, warps in (("c1", "at4", 4), ("c2", "at8", 8)):
            qt = 16 * warps
            assert s % qt == 0 and s % 64 == 0
            kern = dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s))
            cells.append(CH._cell("attn", "validation_fused_attention", "v3", cand, [kern], fp, fp, l2, bh, s, dict(BH=bh, S=s, D=CH.D, query_rows_per_block=qt)))
    else:
        fail("validation_fused_attention", "v3", "no batch*heads in the tier band")
    r, t = R[("validation_scalar_product", "v4")]
    v_ = 768
    got = scale(1024, lambda e: (2 * v_ * e + v_) * FLOAT, r, t, l2)
    if got and all(v_ % g == 0 for g in (g1, g2)):
        e, fp = got
        for cand, g in (("c1", g1), ("c2", g2)):
            k = dict(kid="sp", grid=[g, 1, 1], block=[256, 1, 1], args=dict(vectorN=v_, elementN=e))
            cells.append(CE._cell("sp", "validation_scalar_product", "v4", cand, [k], fp, fp, l2, v_, e, dict(vectorN=v_, elementN=e, grid=g)))
    else:
        fail("validation_scalar_product", "v4", "no elementN in the tier band, or vectorN not divisible by the grids")
    r, t = R[("validation_separable_convolution", "v5")]
    w_b, h_b = next((c["shape_a"], c["shape_b"]) for c in bw_val if c["operator_id"] == "validation_separable_convolution" and c["candidate_id"] == "c1")
    aspect = w_b / h_b
    best = None
    for h in range(128, 30001, 128):
        w = max(128, int(round(h * aspect / 128.0)) * 128)
        fp = 3 * w * h * FLOAT
        if MM.tier_ok(fp / l2, t):
            key = (abs(math.log(fp / (r * l2))), h)
            if best is None or key < best[0]:
                best = (key, w, h, fp)
    if best:
        _, w0, h0, fp = best
        for cand, (w, h) in (("c1", (w0, h0)), ("c2", (h0, w0))):
            assert w % 128 == 0 and h % 64 == 0
            ks = [dict(kid="conv_rows", grid=[w // 128, h // 4, 1], block=[16, 4, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w)),
                  dict(kid="conv_cols", grid=[w // 16, h // 64, 1], block=[16, 8, 1], sampling="classes", args=dict(imageW=w, imageH=h, pitch=w))]
            cells.append(CK._cell("conv", "validation_separable_convolution", "v5", cand, ks, fp, 4 * w * h * FLOAT, l2, w, h, dict(imageW=w, imageH=h)))
    else:
        fail("validation_separable_convolution", "v5", "no image height in the tier band")
    return [_rename(c) for c in cells], unplaceable


def prospective_cells(hw, bw_prosp):
    import cells_prosp as P
    l2 = hw["l2_bytes"]
    R = _ratio_table(bw_prosp)
    cells, unplaceable = [], []
    for regime, (m0, n0, k) in P.GEMM.items():
        r, t = R[("prosp_tensor_matmul_new_tiles", regime)]
        got = scale(256, lambda m: (m * k + m * k) * 2 + m * m * 4, r, t, l2)
        if not got:
            for op in ("prosp_tensor_matmul_new_tiles", "prosp_tensor_matmul_bias_relu"):
                unplaceable.append(dict(operator_id=op, regime=regime, reason="no M=N multiple of 256 in the tier band"))
            continue
        m, fp = got
        n = m
        for cand, kid, bm, bn, thr in (("c1", "tc128x64", 128, 64, 128), ("c2", "tc256x128", 256, 128, 512)):
            assert m % bm == 0 and n % bn == 0 and k % 32 == 0
            cells.append(CG._cell("tcgemm2", "prosp_tensor_matmul_new_tiles", regime, cand, [dict(kid=kid, grid=[n // bn, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k))], fp, fp, l2, m, n,
                                  dict(M=m, N=n, K=k, tile_m=bm, tile_n=bn)))
        for cand, kid, bm, thr in (("c1", "tcr128", 128, 256), ("c2", "tcr64", 64, 128)):
            assert m % bm == 0
            cells.append(CG._cell("tcrelu", "prosp_tensor_matmul_bias_relu", regime, cand, [dict(kid=kid, grid=[n // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k))], fp, fp, l2, m, n,
                                  dict(M=m, N=n, K=k, tile=bm)))
    for regime, (bh0, s) in P.ATT.items():
        r, t = R[("prosp_fused_attention_new_blocks", regime)]
        got = scale(1, lambda bh: 3 * bh * s * P.D * 2 + bh * s * P.D * 4, r, t, l2)
        if not got:
            for op in ("prosp_fused_attention_new_blocks", "prosp_attention_no_max"):
                unplaceable.append(dict(operator_id=op, regime=regime, reason="no batch*heads in the tier band"))
            continue
        bh, fp = got
        for fam, op, kids in (("attn2", "prosp_fused_attention_new_blocks", (("c1", "at2", 2), ("c2", "at16", 16))), ("attnnm", "prosp_attention_no_max", (("c1", "nm4", 4), ("c2", "nm8", 8)))):
            for cand, kid, warps in kids:
                qt = 16 * warps
                assert s % qt == 0 and s % 64 == 0
                cells.append(CH._cell(fam, op, regime, cand, [dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s))], fp, fp, l2, bh, s,
                                      dict(BH=bh, S=s, D=P.D, query_rows_per_block=qt)))
    return [_rename(c) for c in cells], unplaceable


# ---------------------------------------------------------------------------------------------- assembly
SET_OF_GROUP = {"ml": None, "tensor": None, "validation": "validation", "prospective": "prospective"}


def build():
    hw, quoted = B.read_ada_hardware()
    bw_val, bw_prosp = _blackwell_cells()
    val, val_bad = validation_cells(hw, bw_val)
    pro, pro_bad = prospective_cells(hw, bw_prosp)
    ml = [_rename(c) if c["cell_id"].startswith("blackwell") else c for c in ml_cells(hw)]
    tc = [_rename(c) if c["cell_id"].startswith("blackwell") else c for c in tensor_cells(hw)]
    groups = dict(samples=(samples_cells(hw), []), ml=(ml, []), tensor=(tc, []), validation=(val, val_bad), prospective=(pro, pro_bad))
    g1, g2 = MC.grid_candidates(hw["sm_count"])
    out = {}
    for g, (cells, bad) in groups.items():
        ids = [c["cell_id"] for c in cells]
        assert len(ids) == len(set(ids)), g
        for c in cells:
            c["group"] = g
            if g in ("validation", "prospective"):
                c["set"] = g
            assert not (GAP[0] <= c["footprint_over_l2"] < GAP[1]), c["cell_id"]
            assert c["tier"] == ("L2" if c["footprint_over_l2"] < 1 else "DRAM"), c["cell_id"]
            assert c["cell_id"].startswith(B.PREFIX + "/"), c["cell_id"]
        out[g] = dict(schema=SCHEMA, board=B.PREFIX, gpu=B.GPU_NAME, group=g,
                      hardware_from_ground_truth=dict(section=B.SECTION, file="HARDWARE_GROUND_TRUTH.md", values={k: hw[k] for k in sorted(hw)}, rows_quoted=quoted,
                                                      note="the resident-blocks row is parsed under the short label by an in-memory alias; the file is unchanged"),
                      derivation=dict(rule=RULE_TEXT, grid_candidates=dict(c1=g1, c2=g2), tier_rule="tier = L2 iff footprint / L2 < 1; no cell with ratio in [0.4, 1.5)",
                                      l1_tier="none: no verified Ada L1 capacity row (same decision as H100)"),
                      unplaceable=bad, cells=cells)
    return out


def dump(doc):
    return json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n"


def load_group(g):
    return json.loads((HERE / FILES[g]).read_text(encoding="utf-8"))


def load_cells(groups=GROUPS):
    cells = []
    for g in groups:
        cells.extend(load_group(g)["cells"])
    return cells


def summary(docs):
    lines = []
    for g in GROUPS:
        d = docs[g]
        cnt = collections.Counter((c.get("set", g), c["tier"]) for c in d["cells"])
        lines.append("%-12s %3d cells, unplaceable %d: %s" % (g, len(d["cells"]), len(d["unplaceable"]), ", ".join("%s/%s=%d" % (k[0], k[1], v) for k, v in sorted(cnt.items()))))
        for u in d["unplaceable"]:
            lines.append("   UNPLACEABLE %s %s: %s" % (u["operator_id"], u["regime"], u["reason"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    docs = build()
    if a.summary:
        print(summary(docs))
    if a.table:
        seen = set()
        for g in GROUPS:
            for c in docs[g]["cells"]:
                k = (g, c["operator_id"], c["regime"])
                if k in seen:
                    continue
                seen.add(k)
                print("%-11s %-46s %-7s %-4s fp=%12d  %.4f of L2" % (g, c["operator_id"], c["regime"], c["tier"], c["footprint_bytes"], c["footprint_over_l2"]))
    if a.check:
        bad = [FILES[g] for g in GROUPS if not (HERE / FILES[g]).is_file() or (HERE / FILES[g]).read_bytes() != dump(docs[g]).encode("utf-8")]
        if bad:
            print("DIFFERS from the generator output: " + ", ".join(bad), file=sys.stderr)
            return 1
        print("all cell files reproduced byte-for-byte: %d cells" % sum(len(docs[g]["cells"]) for g in GROUPS))
        return 0
    if not (a.summary or a.table):
        for g in GROUPS:
            (HERE / FILES[g]).write_bytes(dump(docs[g]).encode("utf-8"))
            print("wrote", FILES[g], len(docs[g]["cells"]), "cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
