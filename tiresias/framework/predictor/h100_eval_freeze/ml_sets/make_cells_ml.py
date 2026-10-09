#!/usr/bin/env python3
"""Cell lists of the unseen machine-learning kernel set (GELU, SwiGLU gate, RMSNorm, rotary embedding, FP32 matrix multiply: 40 cells), and of the tensor-core matrix multiply and
fused attention sets (16 cells), for the H100 (and, prepared only, the A100). CPU only; no measured value of any kind is read.

    python make_cells_ml.py --arch h100            # (re)write cells_ml_h100.json and cells_tensor_h100.json
    python make_cells_ml.py --arch h100 --check    # exit 1 unless the committed files equal what this script produces
    python make_cells_ml.py --arch h100 --markdown # the size table of FREEZE.md

The L2 capacity (and the SM count, memory) are READ from the H100 or A100 section of HARDWARE_GROUND_TRUTH.md on every run, never typed. Shapes are NEVER copied from Blackwell. The regime
rule is the one the Blackwell sets use (fresh_f/cells_f.py, fresh_g/cells_g.py, fresh_h/cells_h.py): per kernel, four size regimes (small, medium: L2 tier; large, xlarge: DRAM tier) whose
footprint / L2 ratios are those of the Blackwell cells of the same regime, with the same two hard bands as every set (L2 tier needs footprint / L2 < 0.4, DRAM tier needs >= 1.5; no cell between
0.4 and 1.5). Footprint = bytes of every distinct buffer the launch touches. The dimension each regime scales is the one stated per family below; every other dimension keeps its structural
role (hidden size, batch, head count, K rule) and the scaled dimension is the multiple of its tiling unit whose footprint is nearest the target in log ratio (ties: the smaller).

  GELU, SwiGLU gate    n elements, multiple of 1024 (c2 = float4 per thread, 256 threads)
  RMSNorm              rows multiple of 64; columns = hidden size per regime (1024, 2048, 4096, 4096), a multiple of 512 so both candidates tile exactly
  rotary embedding     sequence multiple of 16; batch per regime (2, 2, 16, 32); 8 heads, head_dim 128
  FP32 matrix multiply M = N multiple of 128 (both tile candidates); K rule per regime: small K=M, medium K=2M, large K=512, xlarge K=1024
  tensor matmul        same rule, bf16 inputs fp32 output, K multiple of 32
  fused attention      batch*heads integer; sequence per regime (512, 1024, 2048, 2048); head_dim 64
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FZ = HERE.parent
SR = FZ.parent
REPO = SR.parents[2]
GROUND_TRUTH = REPO / "HARDWARE_GROUND_TRUTH.md"
SCHEMA = "ml_eval_cells/1"
REGIMES = ("small", "medium", "large", "xlarge")
GAP = (0.4, 1.5)
FLOAT = 4
BLACKWELL_L2 = 134217728      # only used to evaluate the Blackwell cell files' own ratios (the regime targets); the verified Blackwell value of HARDWARE_GROUND_TRUTH.md
SECTION = {"h100": "## H100", "a100": "## A100"}
PREFIX = {"h100": "h100", "a100": "a100"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CF, CG, CH = (_load("cells_%s_for_ml" % s, SR / ("fresh_%s" % s) / ("cells_%s.py" % s)) for s in "fgh")


def read_hardware(arch):
    sys.path.insert(0, str(SR))
    import extract_features as X
    text = GROUND_TRUTH.read_text(encoding="utf-8")
    hw = X.load_hardware(text, SECTION[arch])
    missing = [k for k, v in hw.items() if v is None]
    if missing:
        raise SystemExit("HARDWARE_GROUND_TRUTH.md %s section is missing: %s. Verify them live and add them first; nothing is assumed." % (SECTION[arch], ", ".join(missing)))
    return hw


def blackwell_targets():
    """(family, regime) -> footprint / L2 of the Blackwell cell of that regime (the two candidates have equal footprints)."""
    hw = dict(l2_bytes=BLACKWELL_L2)
    t = {}
    for mod in (CF, CG, CH):
        for c in mod.define_cells(hw):
            t[(c["family"], c["regime"])] = c["l2_ratio"]
    return t


def tier_ok(ratio, tier):
    if GAP[0] <= ratio < GAP[1]:
        return False
    return ratio < GAP[0] if tier == "L2" else ratio >= GAP[1]


def pick_dim(unit, footprint_of, target_fp, l2, tier):
    """Dimension j*unit (j >= 1) whose footprint is nearest target_fp in log ratio among those allowed by the hard band of `tier` (ties: smaller). footprint_of must be nondecreasing."""
    lo, hi = 1, 1
    while footprint_of(hi * unit) <= target_fp:
        hi *= 2
        if hi > 1 << 40:
            raise SystemExit("size search diverged")
    while lo < hi:                              # largest j with footprint <= target (or 1)
        mid = (lo + hi + 1) // 2
        if footprint_of(mid * unit) <= target_fp:
            lo = mid
        else:
            hi = mid - 1
    best = None
    for j in range(max(1, lo - 1), lo + 3):
        fp = footprint_of(j * unit)
        if not tier_ok(fp / l2, tier):
            continue
        key = (abs(math.log(fp / target_fp)), j)
        if best is None or key < best[0]:
            best = (key, j * unit, fp)
    if best is None:
        raise SystemExit("no dimension satisfies the %s band near target %d" % (tier, target_fp))
    return best[1], best[2]


def tier_of(regime):
    return "L2" if regime in ("small", "medium") else "DRAM"


K_RULE = {"small": lambda m: m, "medium": lambda m: 2 * m, "large": lambda m: 512, "xlarge": lambda m: 1024}


def define_ml(l2, pfx):
    """The 40 cells of the machine-learning kernel set (5 kernels x 4 regimes x 2 candidates)."""
    T = blackwell_targets()
    cells = []
    for regime in REGIMES:
        n, fp = pick_dim(1024, lambda n: 2 * n * FLOAT, T[("gelu", regime)] * l2, l2, tier_of(regime))
        for cand, kid, per in (("c1", "gelu_s", 256), ("c2", "gelu_v4", 1024)):
            k = dict(kid=kid, grid=[n // per, 1, 1], block=[256, 1, 1], args={})
            cells.append(CF._cell("gelu", "fresh_f_ml_gelu", regime, cand, [k], fp, fp, l2, n, per, dict(n=n, elements_per_thread=per // 256)))
    for regime in REGIMES:
        n, fp = pick_dim(1024, lambda n: 3 * n * FLOAT, T[("swiglu", regime)] * l2, l2, tier_of(regime))
        for cand, kid, per in (("c1", "swiglu_s", 256), ("c2", "swiglu_v4", 1024)):
            k = dict(kid=kid, grid=[n // per, 1, 1], block=[256, 1, 1], args={})
            cells.append(CF._cell("swiglu", "fresh_f_ml_swiglu", regime, cand, [k], fp, fp, l2, n, per, dict(n=n, elements_per_thread=per // 256)))
    COLS = {"small": 1024, "medium": 2048, "large": 4096, "xlarge": 4096}
    for regime in REGIMES:
        c = COLS[regime]
        r, fp = pick_dim(64, lambda r: (2 * r * c + c) * FLOAT, T[("rmsnorm", regime)] * l2, l2, tier_of(regime))
        for cand, kid, thr, args in (("c1", "rms_s", 256, dict(cols=c, inv_cols=1.0 / c, eps=CF.EPS)), ("c2", "rms_v4", 128, dict(cols4=c // 4, inv_cols=1.0 / c, eps=CF.EPS))):
            assert c % (4 * thr) == 0 if kid == "rms_v4" else c % thr == 0
            k = dict(kid=kid, grid=[r, 1, 1], block=[thr, 1, 1], args=args)
            cells.append(CF._cell("rmsnorm", "fresh_f_ml_rmsnorm", regime, cand, [k], fp, fp, l2, r, c, dict(rows=r, cols=c, threads=thr)))
    BATCH = {"small": 2, "medium": 2, "large": 16, "xlarge": 32}
    for regime in REGIMES:
        b = BATCH[regime]
        s, fp = pick_dim(16, lambda s: 2 * b * s * CF.HEADS * CF.HD * FLOAT + 2 * s * (CF.HD // 2) * FLOAT, T[("rope", regime)] * l2, l2, tier_of(regime))
        for cand, kid, grid, thr in (("c1", "rope_all", [s, b, 1], CF.HEADS * 64), ("c2", "rope_one", [s, b * CF.HEADS, 1], 64)):
            k = dict(kid=kid, grid=grid, block=[thr, 1, 1], args={})
            cells.append(CF._cell("rope", "fresh_f_ml_rotary_embedding", regime, cand, [k], fp, fp, l2, s, b, dict(seq=s, batch=b, heads=CF.HEADS)))
    for regime in REGIMES:
        kf = K_RULE[regime]
        m, fp = pick_dim(128, lambda m: (m * kf(m) + kf(m) * m + m * m) * FLOAT, T[("sgemm", regime)] * l2, l2, tier_of(regime))
        n, kk = m, kf(m)
        for cand, kid, bm, bk in (("c1", "sg64", 64, 16), ("c2", "sg128", 128, 8)):
            assert m % bm == 0 and n % bm == 0 and kk % bk == 0
            k = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[256, 1, 1], args=dict(N=n, K=kk))
            cells.append(CF._cell("sgemm", "fresh_f_ml_fp32_matmul", regime, cand, [k], fp, fp, l2, m, n, dict(M=m, N=n, K=kk, tile=bm)))
    return _finish(cells, "fresh_f", pfx)


def define_tensor(l2, pfx):
    """The 16 cells of the tensor-core matrix multiply (8) and fused attention (8) sets."""
    T = blackwell_targets()
    cells = []
    for regime in REGIMES:
        kf = K_RULE[regime]
        m, fp = pick_dim(128, lambda m: (m * kf(m) + kf(m) * m) * 2 + m * m * 4, T[("tcgemm", regime)] * l2, l2, tier_of(regime))
        n, k_ = m, kf(m)
        for cand, kid, bm, thr in (("c1", "tc128", 128, 256), ("c2", "tc64", 64, 128)):
            assert m % bm == 0 and n % bm == 0 and k_ % 32 == 0
            kern = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k_))
            cells.append(CG._cell("tcgemm", "fresh_g_ml_tensor_matmul", regime, cand, [kern], fp, fp, l2, m, n, dict(M=m, N=n, K=k_, tile=bm)))
    SEQ = {"small": 512, "medium": 1024, "large": 2048, "xlarge": 2048}
    for regime in REGIMES:
        s = SEQ[regime]
        bh, fp = pick_dim(1, lambda bh: 3 * bh * s * CH.D * 2 + bh * s * CH.D * 4, T[("attn", regime)] * l2, l2, tier_of(regime))
        for cand, kid, warps in (("c1", "at4", 4), ("c2", "at8", 8)):
            qt = 16 * warps
            assert s % qt == 0 and s % 64 == 0
            kern = dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s))
            cells.append(CH._cell("attn", "fresh_h_ml_fused_attention", regime, cand, [kern], fp, fp, l2, bh, s, dict(BH=bh, S=s, D=CH.D, query_rows_per_block=qt)))
    for c in cells:
        c["set"] = {"tcgemm": "fresh_g", "attn": "fresh_h"}[c["family"]]
    return _finish(cells, None, pfx)


def _finish(cells, set_name, pfx):
    out = []
    for c in cells:
        c = dict(c)
        c["cell_id"] = pfx + c["cell_id"][len("blackwell"):]
        c["_dev"] = dict(c["_dev"], geometry_source="make_cells_ml.py (derived from the board's L2 in HARDWARE_GROUND_TRUTH.md)")
        if set_name:
            c["set"] = set_name
        c["footprint_over_l2"] = c.pop("l2_ratio")
        out.append(c)
    return out


def build(arch):
    hw = read_hardware(arch)
    l2 = hw["l2_bytes"]
    pfx = PREFIX[arch]
    ml, tc = define_ml(l2, pfx), define_tensor(l2, pfx)
    assert len(ml) == 40 and len(tc) == 16
    for c in ml + tc:
        assert not (GAP[0] <= c["footprint_over_l2"] < GAP[1]), c["cell_id"]
        assert c["tier"] == ("L2" if c["footprint_over_l2"] < 1 else "DRAM"), c["cell_id"]
    head = dict(schema=SCHEMA, arch=arch, hardware_from_ground_truth=hw, l2_bytes=l2, ground_truth_section=SECTION[arch].lstrip("# "),
                regime_targets_footprint_over_l2={"%s/%s" % k: round(v, 6) for k, v in sorted(blackwell_targets().items())},
                note="Shapes derived from this board's L2 in HARDWARE_GROUND_TRUTH.md; the regime targets are the footprint/L2 ratios of the Blackwell cells of the same regime, never Blackwell shapes.")
    return {"ml": dict(head, cells=ml), "tensor": dict(head, cells=tc)}


def dump(doc):
    return json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n"


def paths(arch):
    return {"ml": HERE / ("cells_ml_%s.json" % arch), "tensor": HERE / ("cells_tensor_%s.json" % arch)}


def markdown(arch):
    docs = build(arch)
    lines = ["| Set | Operator | Regime | Tier | Footprint (bytes) | MiB | Footprint / L2 | Controls | Launch (candidate 1; candidate 2 in `cells_*.json`) |", "|---|---|---|---|---:|---:|---:|---|---|"]
    for key in ("ml", "tensor"):
        for c in docs[key]["cells"]:
            if c["candidate_id"] != "c1":
                continue
            k = c["kernels"][0]
            lines.append("| %s | %s | %s | %s | %d | %.2f | %.4f | %s | %s grid %s block %s |" % (c["set"], c["family"], c["regime"], c["tier"], c["footprint_bytes"], c["footprint_bytes"] / 2**20,
                         c["footprint_over_l2"], ", ".join("%s=%s" % kv for kv in sorted(c["controls"].items())), k["kid"], "x".join(map(str, k["grid"])), "x".join(map(str, k["block"]))))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(SECTION), required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    if a.markdown:
        print(markdown(a.arch))
        return 0
    docs = build(a.arch)
    bad = []
    for key, p in paths(a.arch).items():
        text = dump(docs[key])
        if a.check:
            if not p.is_file() or p.read_bytes() != text.encode("utf-8"):
                bad.append(p.name)
        else:
            p.write_bytes(text.encode("utf-8"))
            print("wrote", p, len(docs[key]["cells"]), "cells")
    if bad:
        print("DIFFERS from the generator output: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
