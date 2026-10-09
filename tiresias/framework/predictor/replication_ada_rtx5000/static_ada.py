#!/usr/bin/env python3
"""Static tables, footprints and port-support check of every Ada cell, through the cross-GPU port on Ada's own CUDA 13.2 SASS (CPU only; reads no measured value). One command:

    python static_ada.py [--jobs 4] [--only <substring>] [--force]     # build what has SASS, mark the rest "pending build", merge, write the support report
    python static_ada.py --merge-only                                  # merge the cached per-cell files (deterministic)
    python static_ada.py --status                                      # which SASS exists, which cells are pending

Origins (which SASS the cell's kernels come from; chosen per cell from its kernel ids):
  base   the nine CUDA-samples builds in port_ada/compiled_ada_cuda13.2/ (exist): classic kernels, scalar product, fast Walsh transform, set D (transposes, copies, reductions, vector add), validation scalar product and convolution
  ml     driver_ml.sass / driver_ml.res in port_ada/compiled_ada_cuda13.2_eval/ (the combined machine-learning, tensor-core matrix multiply and attention driver; NOT built yet): sets F, G, H and validation F, G, H cells
  prosp  driver_prosp.sass / driver_prosp.res in the same folder (the prospective-test driver; NOT built yet)
A cell whose SASS does not exist is "pending_build" (never a refusal and never dropped); the freeze refuses while any cell is pending. A cell the static side refuses keeps its exact reason and is a failure.

Outputs (deterministic, no timestamps) in static/: features_<group>.json, phases_<group>.json, phases_unique_<group>.json, bank_conflicts_<group>.json, static_support_<group>.json, footprints_<group>.json,
plus PORT_SUPPORT_ADA.md. Per-cell cache: build/<group>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import math
import sys
import time
from pathlib import Path

import board_ada as B
import make_cells_ada as MA

HERE = B.HERE
SR = B.SR
for p in (SR / "port_common", SR / "shared_traffic", SR / "bank"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTDIR = HERE / "static"
BUILD = HERE / "build"
SUPPORTED = ("supported", "supported_with_assumptions")
BASE_STEMS = ("matrixMul", "BlackScholes", "scan", "convolutionSeparable", "scalarProd", "fastWalshTransform", "vectorAdd", "reduction", "transpose")
# prospective-test kernels: needle (symbol fragment) and parameter list as in prospective_test/prosp_lib.py (test_static_ada.py compares them)
_G = [("A", "ptr"), ("B", "ptr"), ("C", "ptr"), ("N", "i32"), ("K", "i32")]
_GB = [("A", "ptr"), ("B", "ptr"), ("bias", "ptr"), ("C", "ptr"), ("N", "i32"), ("K", "i32")]
_AT = [("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("S", "i32")]
PROSP_KERNELS = {
    "tc128x64": ("tc_gemmILi128ELi64E", _G), "tc256x128": ("tc_gemmILi256E", _G), "at2": ("attn_fwdILi2E", _AT), "at16": ("attn_fwdILi16E", _AT),
    "nm4": ("attn_nomaxILi4E", _AT), "nm8": ("attn_nomaxILi8E", _AT), "tcr128": ("tc_gemm_bias_reluILi128E", _GB), "tcr64": ("tc_gemm_bias_reluILi64E", _GB),
}
PROSP_STEM = "driver_prosp"


def corners(grid):
    ids = set()
    for xyz in itertools.product(*[sorted({0, g - 1}) for g in grid]):
        ids.add(xyz[0] + xyz[1] * grid[0] + xyz[2] * grid[0] * grid[1])
    return sorted(ids)


def origin_of(cell):
    import port_ml_ext as PM
    kids = {k["kid"] for k in cell["kernels"]}
    if kids & set(PROSP_KERNELS):
        return "prosp"
    if kids <= set(PM.KERNELS):
        return "ml"
    return "base"


def sass_state():
    """{origin: (available, [missing files])}."""
    base = [str(B.SASS_BASE / "sass" / (s + ".sass")) for s in BASE_STEMS] + [str(B.SASS_BASE / "log" / (s + ".res")) for s in BASE_STEMS]
    out = {"base": [f for f in base if not Path(f).is_file()]}
    out["ml"] = [str(B.SASS_EVAL / ("driver_ml." + e)) for e in ("sass", "res") if not (B.SASS_EVAL / ("driver_ml." + e)).is_file()]
    out["prosp"] = [str(B.SASS_EVAL / (PROSP_STEM + "." + e)) for e in ("sass", "res") if not (B.SASS_EVAL / (PROSP_STEM + "." + e)).is_file()]
    return {k: (not v, v) for k, v in out.items()}


def _setup(origin):
    sys.path.insert(0, str(SR / "bank"))
    import bank_unseen as BU
    if origin == "base":
        import port_ext as PX
        import set_d_port as SD
        PX.install(B.ARCH, B.SASS_BASE)
        SD.register(blackwell=False)
        UP = PX.UP
    else:
        import port_ml_ext as PM
        PM.install(B.ARCH, B.SASS_EVAL)
        UP = PM.UP
        if origin == "prosp":
            import port_ext as PX
            for kid, (needle, params) in PROSP_KERNELS.items():
                UP.KERNELS[kid] = dict(sass=PROSP_STEM, needle=needle, role="main", params=params)
                PX.A.PARAM_FIELDS[kid] = params
            UP._KERNEL_CACHE.clear()
    if not getattr(BU.sum_rows, "_with_histogram", False):
        orig = BU.sum_rows

        def sum_rows_w(rows):
            out = orig(rows)
            h = collections.Counter()
            for r in rows:
                h.update(r["shared_request_wavefront_histogram"])
            out["shared_request_wavefront_histogram"] = {str(k): v for k, v in sorted(h.items(), key=lambda kv: int(kv[0]))}
            return out
        sum_rows_w._with_histogram = True
        BU.sum_rows = sum_rows_w
    return UP, BU


def cache_path(group, cell_id):
    return BUILD / group / (cell_id.replace("/", "__") + ".json")


def build_one(args):
    group, cell_id = args
    import footprint as FP
    cell = next(c for c in MA.load_group(group)["cells"] if c["cell_id"] == cell_id)
    origin = origin_of(cell)
    UP, BU = _setup(origin)
    C, P = UP.C, UP.P
    hw, _ = B.read_ada_hardware()
    t = time.time()
    try:
        rec, prow, urow = UP.build_cell(cell, hw)
        rec = dict(rec)
        rec.pop("build_seconds", None)
        if prow["status"] == "conditional_static_phases" and urow["status"] == "ok":
            try:
                bank = dict(kernels=[BU.analyse_kernel(k["kid"], k, fk) for k, fk in zip(cell["kernels"], prow["kernels"])])
            except (C.Refusal, P.Refusal) as ex:
                bank = dict(status="refused", reason=str(ex))
        else:
            bank = dict(status="refused", reason="no phase or first-touch table for this cell (%s / %s)" % (prow["status"], urow["status"]))
        try:
            kernels = []
            nblocks_of = lambda g: math.prod(g)
            UP.KW["hw"] = hw
            for kern in cell["kernels"]:
                consts, grid, block, prov = UP.binding(kern["kid"], kern)
                sub = dict(kern, sample_blocks=corners(grid)) if kern.get("sampling") == "classes" else kern
                sigs, obs = UP.trace_kernel(kern["kid"], sub, consts, grid, block)
                fp = FP.kernel_footprint(UP.U, obs, nblocks_of(grid))
                fp["kernel_id"] = kern["kid"]
                fp["sampled_block_ids"] = UP.sample_blocks(grid, sub)
                kernels.append(fp)
            fpdoc = dict(cell_id=cell_id, status="ok", kernels=kernels)
        except Exception as ex:   # recorded with its reason, never replaced by a default footprint
            fpdoc = dict(cell_id=cell_id, status="refused", reason="%s: %s" % (type(ex).__name__, ex), kernels=[])
        res = dict(cell=cell, origin=origin, features=rec, phases=prow, unique=urow, bank=bank, footprints=fpdoc)
        msg = "%s %s %s %s fp:%s" % (rec["status"], prow["status"], urow["status"], "bank ok" if "kernels" in bank else "bank refused", fpdoc["status"])
    except (C.Refusal, P.Refusal) as ex:
        res = dict(cell=cell, origin=origin, refused=str(ex))
        msg = "REFUSED " + str(ex)[:200]
    p = cache_path(group, cell_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return cell_id, "%s %.0fs" % (msg, time.time() - t)


def merge(group, hw):
    cells = MA.load_group(group)["cells"]
    state = sass_state()
    feats, phases, uniq, bank, support, fps = [], {}, {}, {}, {}, {}
    pending = []
    for c in cells:
        cid = c["cell_id"]
        p = cache_path(group, cid)
        if not p.exists():
            origin = origin_of(c)
            if state[origin][0]:
                raise SystemExit("missing build file for %s although its SASS exists; run static_ada.py without --merge-only" % cid)
            why = "pending build: %s missing (BUILD_MANIFEST.md)" % ", ".join(Path(f).name for f in state[origin][1][:2])
            feats.append(dict(cell_id=cid, status="pending_build", missing_features=[dict(feature="SASS", reason=why)]))
            phases[cid] = dict(status="pending_build", reason=why, kernels=[])
            uniq[cid] = dict(status="pending_build", reason=why, kernels=[])
            bank[cid] = dict(status="pending_build", reason=why)
            fps[cid] = dict(cell_id=cid, status="pending_build", reason=why, kernels=[])
            support[cid] = dict(supported=False, pending_build=True, origin=origin, reason=why)
            pending.append(cid)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "refused" in d:
            reason = d["refused"]
            feats.append(dict(cell_id=cid, status="missing_features", missing_features=[dict(feature="static pipeline", reason=reason)]))
            phases[cid] = dict(status="unsupported", reason=reason, kernels=[])
            uniq[cid] = dict(status="unsupported", reason=reason, kernels=[])
            bank[cid] = dict(status="refused", reason=reason)
            fps[cid] = dict(cell_id=cid, status="refused", reason=reason, kernels=[])
            support[cid] = dict(supported=False, pending_build=False, origin=d["origin"], reason=reason)
            continue
        feats.append(d["features"]); phases[cid] = d["phases"]; uniq[cid] = d["unique"]; bank[cid] = d["bank"]; fps[cid] = d["footprints"]
        ok = (d["features"]["status"] in SUPPORTED and d["phases"]["status"] == "conditional_static_phases" and d["unique"]["status"] == "ok" and "kernels" in d["bank"])
        why = [] if ok else ["features %s%s" % (d["features"]["status"], (": " + "; ".join(str(m.get("reason", m))[:120] for m in d["features"].get("missing_features", [])[:2])) if d["features"].get("missing_features") else ""),
                               "phases %s %s" % (d["phases"]["status"], d["phases"].get("reason") or ""), "first-touch %s %s" % (d["unique"]["status"], d["unique"].get("reason") or ""),
                               "bank %s" % (d["bank"].get("reason") or "ok")]
        support[cid] = dict(supported=ok, pending_build=False, origin=d["origin"], status=d["features"]["status"], reason=None if ok else " | ".join(why))
    feats.sort(key=lambda r: r["cell_id"])
    OUTDIR.mkdir(exist_ok=True)
    w = lambda name, obj: (OUTDIR / ("%s_%s.json" % (name, group))).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    w("features", dict(schema="static_runtime_features_ada/1", hardware_from_ground_truth=hw, rows=feats, toolchain="Ada's own CUDA 13.2 SASS (port_ada/compiled_ada_cuda13.2 and compiled_ada_cuda13.2_eval)",
                       note="Label-free static features of the Ada cells. No runtime, energy or power value was read."))
    w("phases", dict(schema="static_dynamic_barrier_phases/1", rows=phases))
    w("phases_unique", dict(schema="first_touch_unique_sectors/1", rows=uniq))
    w("bank_conflicts", dict(set="ada_eval", schema="bank_conflicts/1", rows=bank))
    w("footprints", dict(schema="ada_footprints/1", arch=B.ARCH, group=group, cells=len(fps), refused=sorted(c for c, v in fps.items() if v["status"] not in ("ok",)), rows=fps,
                         note="Grid-wide unique footprints per pointer argument from the interpreted address traces of the sampled blocks (shared_traffic/footprint.py), sm_89 port. No measured value was read."))
    n_ok = sum(1 for v in support.values() if v["supported"])
    w("static_support", dict(schema="ada_static_support/1", cells=len(support), supported=n_ok, pending_build=sorted(pending),
                             unsupported=sorted(c for c, v in support.items() if not v["supported"] and not v["pending_build"]), rows=support,
                             note="Unsupported cells stay in the list and are failures; pending cells wait for SASS from the Ada host."))
    return dict(group=group, cells=len(support), supported=n_ok, pending=len(pending), unsupported={c: v["reason"] for c, v in support.items() if not v["supported"] and not v["pending_build"]})


def report(rows):
    lines = ["# Ada port-support report (sm_89, Ada's own CUDA 13.2 SASS)", "", "Generated by `static_ada.py`; deterministic. `pending build` = SASS not yet produced on the Ada host (BUILD_MANIFEST.md); `unsupported` = the port refuses the cell (counted as a failure).", "",
             "| Group | Cells | Supported | Pending build | Unsupported |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| %s | %d | %d | %d | %d |" % (r["group"], r["cells"], r["supported"], r["pending"], len(r["unsupported"])))
    kids = {}
    for g in MA.GROUPS:
        for c in MA.load_group(g)["cells"]:
            kids.setdefault(origin_of(c), set()).update(k["kid"] for k in c["kernels"])
    st = sass_state()
    lines += ["", "## Kernels by SASS origin", "", "| Origin | Kernels | SASS |", "|---|---:|---|"]
    for o in ("base", "ml", "prosp"):
        lines.append("| %s | %d (%s) | %s |" % (o, len(kids.get(o, ())), ", ".join(sorted(kids.get(o, ()))), "present" if st[o][0] else "**pending build**: " + ", ".join(Path(m).name for m in st[o][1])))
    for r in rows:
        for c, why in sorted(r["unsupported"].items()):
            lines.append("")
            lines.append("- UNSUPPORTED `%s`: %s" % (c, (why or "")[:300]))
    (OUTDIR / "PORT_SUPPORT_ADA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    state = sass_state()
    todo_all = []
    for g in MA.GROUPS:
        for c in MA.load_group(g)["cells"]:
            todo_all.append((g, c, origin_of(c)))
    if a.status:
        for o, (ok, miss) in state.items():
            print("%-6s SASS %s%s" % (o, "present" if ok else "MISSING", "" if ok else ": " + ", ".join(Path(m).name for m in miss)))
        cnt = collections.Counter((g, o, state[o][0]) for g, c, o in todo_all)
        for (g, o, ok), n in sorted(cnt.items()):
            print("%-12s origin %-6s %3d cells %s" % (g, o, n, "buildable" if ok else "pending build"))
        return 0
    if not a.merge_only:
        todo = [(g, c["cell_id"]) for g, c, o in todo_all if state[o][0] and (not a.only or a.only in c["cell_id"]) and (a.force or not cache_path(g, c["cell_id"]).exists())]
        print("%d cells to build" % len(todo), flush=True)
        if a.jobs > 1 and len(todo) > 1:
            with cf.ProcessPoolExecutor(max_workers=a.jobs) as pool:
                for cid, msg in pool.map(build_one, todo):
                    print("%-72s %s" % (cid, msg), flush=True)
        else:
            for t in todo:
                print("%-72s %s" % build_one(t), flush=True)
        if a.only:
            return 0
    hw, _ = B.read_ada_hardware()
    rows = [merge(g, hw) for g in MA.GROUPS]
    report(rows)
    for r in rows:
        print("%-12s %3d cells: %3d supported, %3d pending build, %d unsupported" % (r["group"], r["cells"], r["supported"], r["pending"], len(r["unsupported"])))
        for c, why in sorted(r["unsupported"].items()):
            print("   UNSUPPORTED %s: %s" % (c, (why or "")[:200]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
