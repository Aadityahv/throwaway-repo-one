#!/usr/bin/env python3
"""Static pipeline for the unseen-kernel test (pinned CUDA samples, Blackwell, CPU only).

Builds, per cell, the three tables the frozen runtime predictor reads (features row, barrier-phase row,
first-touch row) from the retained SASS of kernels that no model has seen, using the committed helpers by
import (phases.py, reuse/phases_unique.py, pytorch_features/adapter.py, pt_interp.py); none of them is edited.
The only new pieces are the kernel registry (parameter layouts, from the samples' source), the geometry of
each launch (from the samples' host code), and the cell definitions in cells.py.

No runtime, energy or power value is read, and nothing is executed on a GPU. One result file per cell is
written to build/cells/ so that slow cells can run in parallel processes and be resumed.

Usage: python3 unseen_pipeline.py --kernel matmul [--only <cell_id>] [--list]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR))
sys.path.insert(0, str(SR / "coalescing"))
sys.path.insert(0, str(SR / "pytorch_features"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


U = _load("phases_unique_for_unseen", SR / "reuse/phases_unique.py")  # imported, never edited; patches its own copy of phases.py
PH, C, A, P = U.PH, U.C, U.A, U.P
X = PH.X

SASS_DIR = HERE / "compiled/sass"
LOG_DIR = HERE / "compiled/log"
ISO_DIR = HERE / "isolated"
CELL_DIR = HERE / "build/cells"

# ---------------------------------------------------------------------------- kernel registry
# Parameter lists follow the samples' kernel signatures (cpp/... at commit 5443602d). Every constant-bank access in
# the SASS must land on a declared field start (adapter.abi_check), which independently checks these layouts.
KERNELS = {
    "mm16": dict(sass="matrixMul", needle="MatrixMulCUDAILi16E", role="main",
                 params=[("C", "ptr"), ("A", "ptr"), ("B", "ptr"), ("wA", "i32"), ("wB", "i32")]),
    "mm32": dict(sass="matrixMul", needle="MatrixMulCUDAILi32E", role="main",
                 params=[("C", "ptr"), ("A", "ptr"), ("B", "ptr"), ("wA", "i32"), ("wB", "i32")]),
    "bs": dict(sass="BlackScholes", needle="BlackScholesGPU", role="main",
               params=[("call", "ptr"), ("put", "ptr"), ("stock", "ptr"), ("strike", "ptr"), ("years", "ptr"),
                       ("riskfree", "f32"), ("volatility", "f32"), ("optN", "i32")]),
    "scan_bottom": dict(sass="scan", needle="scanExclusiveSharedP5uint4", role="main",
                        params=[("dst", "ptr"), ("src", "ptr"), ("size", "i32")]),
    "scan_top": dict(sass="scan", needle="scanExclusiveShared2", role="main",
                     params=[("buf", "ptr"), ("dst", "ptr"), ("src", "ptr"), ("N", "i32"), ("arrayLength", "i32")]),
    "scan_update": dict(sass="scan", needle="uniformUpdate", role="main",
                        params=[("data", "ptr"), ("buffer", "ptr")]),
    "conv_rows": dict(sass="convolutionSeparable", needle="convolutionRowsKernel", role="main",
                      params=[("dst", "ptr"), ("src", "ptr"), ("imageW", "i32"), ("imageH", "i32"), ("pitch", "i32")]),
    "conv_cols": dict(sass="convolutionSeparable", needle="convolutionColumnsKernel", role="main",
                      params=[("dst", "ptr"), ("src", "ptr"), ("imageW", "i32"), ("imageH", "i32"), ("pitch", "i32")]),
}
for _kid, _spec in KERNELS.items():
    A.PARAM_FIELDS[_kid] = _spec["params"]  # registers the layout with the committed adapter (in memory only)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def isolated_text(sass_name, needle):
    txt = (SASS_DIR / (sass_name + ".sass")).read_text()
    hits = [f for f in re.split(r"\n\s*Function : ", txt)[1:] if needle in f.split("\n")[0]]
    if len(hits) != 1:
        raise SystemExit("kernel %r matched %d functions in %s" % (needle, len(hits), sass_name))
    return "Function : " + hits[0]


def resource_record(sass_name, needle):
    lines = (LOG_DIR / (sass_name + ".res")).read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Function ") and needle in ln:
            res = lines[i + 1].strip()
            m = A.RES_RE.search(res)
            return ln.strip().split()[1].rstrip(":"), res, int(m.group(1)), int(m.group(3))
    raise SystemExit("no resource line for " + needle)


_KERNEL_CACHE = {}


def kernel_assets(kid):
    if kid in _KERNEL_CACHE:
        return _KERNEL_CACHE[kid]
    s = KERNELS[kid]
    text = isolated_text(s["sass"], s["needle"])
    ISO_DIR.mkdir(exist_ok=True)
    (ISO_DIR / (kid + ".isolated.sass")).write_text(text)
    mangled, res_line, reg, shared = resource_record(s["sass"], s["needle"])
    sites = P.parse(text)
    ok, problems = A.abi_check(kid, sites)
    rec = dict(mangled=mangled, res_usage_line=res_line, registers_per_thread=reg, static_shared_bytes=shared)
    _KERNEL_CACHE[kid] = dict(text=text, sites=sites, idx=rec, abi_ok=ok, abi_problems=problems,
                              sha=sha(text.encode()), cubin_sha=sha((HERE / "compiled/cubin" / (s["sass"] + ".cubin")).read_bytes()))
    return _KERNEL_CACHE[kid]


# ---------------------------------------------------------------------------- private interpreter extension
# Five opcodes outside the committed whitelists (see PHASE0_GATE.md). Four need no new semantics: the committed branches
# already implement them under a sibling name (ISETP with LE and U32 modifiers, LOP3.LUT for ULOP3.LUT) or give an unknown
# float/data result (MUFU, a constant-cache load, as for the whitelisted MUFU.EX2 and LDG.E.128.CONSTANT). USHF.R.U32.HI is
# the uniform-register form of SHF.R.U32.HI and is routed to that branch. The on-disk interpreter is never edited.
UNSEEN_EXT_OPS = frozenset(["LDG.E.64.CONSTANT", "MUFU.LG2", "ISETP.LE.U32.AND", "ULOP3.LUT", "USHF.R.U32.HI"])
_NEEDLE = "            elif o == 'SHF.R.U32.HI':\n"
_REPL = "            elif o in ('SHF.R.U32.HI', 'USHF.R.U32.HI'):\n"


def unseen_interp_class():
    import inspect
    import textwrap
    src = inspect.getsource(P.Interp._lane_gen)
    if src.count(_NEEDLE) != 1:
        raise SystemExit("SHF.R.U32.HI insertion point changed in pt_interp.py")
    env = dict(P.Interp._lane_gen.__globals__)
    exec(compile(textwrap.dedent(src.replace(_NEEDLE, _REPL)), "<unseen-ushf>", "exec"), env)

    class UnseenInterp(P.Interp):
        _lane_gen = env["_lane_gen"]

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.KNOWN_OPS = set(self.KNOWN_OPS) | UNSEEN_EXT_OPS

    return UnseenInterp


INTERP = unseen_interp_class()


# ---------------------------------------------------------------------------- binding
def binding(kid, kern):
    """({offset: value}, grid, block, provenance) for one launch. Pointer arguments are distinct aligned bases."""
    grid, block = list(kern["grid"]), list(kern["block"])
    c = {A.IMPLICIT["blockDim.x"]: block[0], A.IMPLICIT["blockDim.y"]: block[1], A.IMPLICIT["blockDim.z"]: block[2],
         A.IMPLICIT["gridDim.x"]: grid[0], A.IMPLICIT["gridDim.y"]: grid[1], A.IMPLICIT["gridDim.z"]: grid[2]}
    prov, nptr = {}, 0
    for (name, off, size), (_, ty) in zip(A.abi_layout(kid), KERNELS[kid]["params"]):
        if ty == "ptr":
            value = C.PTR_BASE0 + nptr * C.PTR_STRIDE
            nptr += 1
            src = "distinct aligned synthetic base %d (pointer value; only address differences matter)" % (nptr - 1)
        elif ty == "f32":
            value = A.f32_bits(kern["args"][name])
            src = "launch argument %r as binary32" % kern["args"][name]
        else:
            value = kern["args"][name]
            src = "launch argument from the sample's host code"
        c[off] = value & 0xFFFFFFFF
        if size == 8:
            c[off + 4] = (value >> 32) & 0xFFFFFFFF
        prov[name] = dict(offset=hex(off), value=value, source=src)
    return c, grid, block, prov


# ---------------------------------------------------------------------------- per-kernel analyses
def sample_blocks(grid, kern):
    """Blocks whose phase signature is traced. Cells may name an explicit list (block classes); default: first and last."""
    n = grid[0] * grid[1] * grid[2]
    return kern.get("sample_blocks") or sorted({0, n - 1})


def run_exact(kid, kern, consts, grid, block):
    """Work counts at exact block coordinates (the committed A.run_kernel uses block-index intervals, which cannot decide
    loop exits that depend on the block index). Every sampled block must give identical per-lane visits and events; otherwise
    the grid is not uniform and the cell is refused (class weighting would be needed)."""
    a = kernel_assets(kid)
    threads = math.prod(block)
    results = []
    for b in sample_blocks(grid, kern):
        I = INTERP(C.D, ext=True, fchk_fast_path=True)
        cm = {o: P.V.exact(v) for o, v in consts.items()}

        def lane(l, b=b):
            return {"SR_CTAID.X": P.V.exact(b % grid[0]), "SR_CTAID.Y": P.V.exact((b // grid[0]) % grid[1]),
                    "SR_CTAID.Z": P.V.exact(b // (grid[0] * grid[1])), "SR_CgaCtaId": P.V.exact(0),
                    "SR_TID.X": P.V.exact(l % block[0]), "SR_TID.Y": P.V.exact((l // block[0]) % block[1]),
                    "SR_TID.Z": P.V.exact(l // (block[0] * block[1])), "SR_LANEID": P.V.exact(l % 32)}
        try:
            res = I.run_block(a["sites"], cm, lane, threads)
        except P.Refusal as ex:
            return {"status": "refused", "reason": str(ex)}
        events = collections.Counter()
        for e, _, _ in res:
            events.update(e)
        results.append(dict(visits=[v for _, v, _ in res], events=events, unknown_pcs=set(I.unknown_pcs),
                            assumption_log={"%s@%#x" % k: v for k, v in sorted(I.assumption_log.items())},
                            shared_addresses=(min(I.shared_touched), max(I.shared_touched)) if I.shared_touched else None))
    first = results[0]
    for r in results[1:]:
        if r["visits"] != first["visits"] or r["events"] != first["events"]:
            return {"status": "refused", "reason": "work differs between sampled blocks (non-uniform grid; class weighting needed)"}
    return dict(status="ok", threads=threads, **first)


def trace_kernel(kid, kern, consts, grid, block):
    """Phase signatures and the observers (for first-touch) at the sampled blocks."""
    a = kernel_assets(kid)
    threads = math.prod(block)
    sigs, observers = [], []
    del U.CAPTURED[:]
    for b in sample_blocks(grid, kern):
        obs = U.PH.Observer(threads)
        obs.block_x = block[0]
        interp = INTERP(C.D, ext=True, fchk_fast_path=True, trace=obs.pytorch)

        def coords(l, b=b):
            return {k: P.V.exact(v) for k, v in {
                "SR_CTAID.X": b % grid[0], "SR_CTAID.Y": (b // grid[0]) % grid[1], "SR_CTAID.Z": b // (grid[0] * grid[1]),
                "SR_CgaCtaId": 0, "SR_TID.X": l % block[0], "SR_TID.Y": (l // block[0]) % block[1],
                "SR_TID.Z": l // (block[0] * block[1]), "SR_LANEID": l % 32}.items()}
        interp.run_block(a["sites"], {k: P.V.exact(v) for k, v in consts.items()}, coords, threads)
        sigs.append(obs.finish(math.prod(grid)))
        observers.append(obs)
    return sigs, list(U.CAPTURED)



# ---------------------------------------------------------------------------- block classes (kernels with border guards)
def dim_classes(g):
    """{class: (block count, representative indices)} along one grid dimension: first, last, interior. Interior has up to
    three representatives (1, g // 2, g - 2) whose analyses must agree."""
    if g == 1:
        return {"only": (1, [0])}
    d = {"first": (1, [0]), "last": (1, [g - 1])}
    if g > 2:
        d["interior"] = (g - 2, sorted({1, g // 2, g - 2}))
    return d


def class_list(grid):
    assert grid[2] == 1
    out = []
    for xn, (xw, xr) in dim_classes(grid[0]).items():
        for yn, (yw, yr) in dim_classes(grid[1]).items():
            out.append(dict(name=xn + "/" + yn, weight=xw * yw, reps=[x + y * grid[0] for y in yr for x in xr]))
    assert sum(c["weight"] for c in out) == grid[0] * grid[1]
    return out


def _add_rows(rows):
    """Elementwise sum of per-class phase rows (each already multiplied by its class block count)."""
    out = []
    for i in range(len(rows[0])):
        r = dict(index=i, repetitions=1, issue_warp_instructions=dict(sum((collections.Counter(c[i]["issue_warp_instructions"]) for c in rows), collections.Counter())))
        for key in ("read_sectors", "write_sectors", "lines", "read_bytes", "write_bytes"):
            vals = [c[i][key] for c in rows]
            r[key] = None if any(v is None for v in vals) else sum(vals)
        for key in ("critical_path_compute_instructions", "dependent_global_load_depth"):
            r[key] = max(c[i][key] for c in rows)
        out.append(r)
    return out


def analyse_kernel_classes(kid, kern, consts, grid, block):
    """Class-weighted work, phases and first-touch. Representatives of one class must agree exactly; classes are summed with
    their block counts as weights. Returns (interior_run, work_override, structure_override, phase_row, unique_entry)."""
    a = kernel_assets(kid)
    threads = math.prod(block)
    nblocks = math.prod(grid)
    per_class, run_main = [], None
    for cl in class_list(grid):
        sub = dict(kern, sample_blocks=cl["reps"])
        run = run_exact(kid, sub, consts, grid, block)         # also checks that the representatives agree
        if run["status"] != "ok":
            raise C.Refusal("class %s: %s" % (cl["name"], run.get("reason")))
        sigs, obs = trace_kernel(kid, sub, consts, grid, block)
        if any(x != sigs[0] for x in sigs[1:]):
            raise C.Refusal("class %s: representatives have different phase signatures" % cl["name"])
        an = [U.analyse_block(o) for o in obs]
        key = lambda x: ([(r["first_touch"], r["unique_in_phase"], r["req_read"], r["req_write"], r["lines"]) for r in x["rows"]], len(x["cum_read"]), x["unknown"])
        if any(key(x) != key(an[0]) for x in an[1:]):
            raise C.Refusal("class %s: representatives have different first-touch analyses" % cl["name"])
        if an[0]["unknown"]:
            raise C.Refusal("class %s: data-dependent global address; first-touch unknown" % cl["name"])
        w = cl["weight"]
        kc = dict(kern, grid=[w, 1, 1])                      # pseudo-launch of w identical blocks, only to reuse the work builders
        blk_c = A.kernel_features(X, C.D, KW["hw"], kid, "main", kc, a["idx"], a["sites"], run, {"status": "not_run"}, consts, {},
                                  a["abi_ok"], a["abi_problems"], a["sha"], {}, kern.get("dynamic_smem", 0))
        if blk_c["work"] is None:
            raise C.Refusal("class %s: work refused" % cl["name"])
        per_class.append(dict(cl=cl, run=run, obs=obs[0], an=an[0], blk=blk_c, rows=obs[0].finish(w)))
        if cl["name"] in ("interior/interior", "interior/only", "only/interior", "only/only"):
            run_main = run
    run_main = run_main or max(per_class, key=lambda c: c["cl"]["weight"])["run"]
    for c in per_class:
        if c["rows"]["barrier_sequence"] != per_class[0]["rows"]["barrier_sequence"]:
            raise C.Refusal("block classes have different barrier sequences")
        if len(c["rows"]["phases"]) != len(per_class[0]["rows"]["phases"]):
            raise C.Refusal("block classes have different phase counts")
    agg = A.per_launch_totals([c["blk"] for c in per_class])
    phases = _add_rows([c["rows"]["phases"] for c in per_class])
    prow = dict(phases=phases, barrier_sequence=per_class[0]["rows"]["barrier_sequence"],
                unknown_guard_upper_bounds=sorted({g for c in per_class for g in c["rows"]["unknown_guard_upper_bounds"]}),
                sampled_blocks=[b for c in per_class for b in c["cl"]["reps"]],
                grid_extrapolation="block classes (first/interior/last per dimension) weighted by class size; representatives of a class compared exactly")
    ents = []
    for i, ph in enumerate(phases):
        tot = lambda k: sum(c["cl"]["weight"] * c["an"]["rows"][i][k] for c in per_class)
        for c in per_class:
            r = c["an"]["rows"][i]
            assert r["req_read"] * c["cl"]["weight"] == c["rows"]["phases"][i]["read_sectors"] and r["req_write"] * c["cl"]["weight"] == c["rows"]["phases"][i]["write_sectors"] and r["lines"] * c["cl"]["weight"] == c["rows"]["phases"][i]["lines"]
        ents.append(dict(index=i, repetitions=1, read_sectors=ph["read_sectors"], write_sectors=ph["write_sectors"], lines=ph["lines"],
                         read_sectors_total_requested=ph["read_sectors"],
                         read_sectors_unique_in_phase_per_block=tot("unique_in_phase") / nblocks,
                         read_sectors_first_touch_per_block=tot("first_touch") / nblocks,
                         read_sectors_first_touch=tot("first_touch"),
                         read_sectors_readback_of_prior_phase_writes_per_block=tot("readback_of_prior_phase_writes") / nblocks,
                         read_sectors_also_written_same_phase_per_block=tot("read_also_written_same_phase") / nblocks))
    uniq = sum(c["cl"]["weight"] * len(c["an"]["cum_read"]) for c in per_class) / nblocks
    uent = dict(status="ok", blocks=nblocks, phases=ents, weighting="block classes",
                total_requested_read_sectors_per_block=sum(e["read_sectors_total_requested"] for e in ents) / nblocks,
                unique_read_sectors_per_block=uniq, unique_read_bytes_per_block=uniq * 32,
                first_touch_read_sectors_per_block=sum(e["read_sectors_first_touch_per_block"] for e in ents),
                note="per-block fields are averages over the grid; totals are exact class-weighted sums")
    w = dict(per_class[0]["blk"]["work"])
    w.update(total_lane_instructions=agg["total_lane_instructions"], total_lane_instructions_per_thread=agg["total_lane_instructions"] / (threads * nblocks),
             families=agg["families"], opcode_lane_counts=agg["opcode_lane_counts"], executed_global_load_bytes=agg["executed_global_load_bytes_lane_level"],
             executed_global_store_bytes=agg["executed_global_store_bytes_lane_level"], all_counts_exact=agg["all_counts_exact"],
             executed_global_bytes_per_thread=(agg["executed_global_load_bytes_lane_level"] + agg["executed_global_store_bytes_lane_level"]) / (threads * nblocks),
             six_class_counts=agg["six_class_counts"],
             opcodes_with_unknown_guard_upper_bound=sorted({o for c in per_class for o in c["blk"]["work"]["opcodes_with_unknown_guard_upper_bound"]}),
             note=w["note"] + " Class-weighted sum over first/interior/last blocks (grid is not uniform).")
    st = dict(per_class[0]["blk"]["structure"])
    st["barrier_releases_per_launch_estimate"] = agg["barrier_releases_per_launch_estimate"]
    return run_main, w, st, prow, uent


KW = {}


def build_cell(cell, hw):
    """-> (features_row, phases_row, unique_row) for one cell dict from cells.py."""
    t0 = time.time()
    kblocks, kphases, kunique = [], [], []
    for kern in cell["kernels"]:
        kid = kern["kid"]
        a = kernel_assets(kid)
        consts, grid, block, prov = binding(kid, kern)
        KW["hw"] = hw
        classed = kern.get("sampling") == "classes"
        if classed:
            run, cw, cst, cprow, cuent = analyse_kernel_classes(kid, kern, consts, grid, block)
        else:
            run = run_exact(kid, kern, consts, grid, block)
        blk = A.kernel_features(X, C.D, hw, kid, "main", kern, a["idx"], a["sites"], run,
                                {"status": "not_run", "reason": "frozen derive.run_lane comparison skipped for new kernels"},
                                consts, prov, a["abi_ok"], a["abi_problems"], a["sha"],
                                dict(logical_bytes_per_launch=cell["logical_bytes_per_launch"], tier=cell["tier"],
                                     shape_a=cell.get("shape_a"), shape_b=cell.get("shape_b")), kern.get("dynamic_smem", 0))
        if classed:
            blk["work"], blk["structure"] = cw, cst
        blk["shape_problems"] = []
        blk["cubin_sha256"] = a["cubin_sha"]
        blk["kernel_symbol"] = a["idx"]["mangled"]
        kblocks.append(blk)
        if classed:
            kphases.append(dict(kernel_id=kid, **cprow))
            kunique.append((kid, None, cuent))
        else:
            sigs, obs = trace_kernel(kid, kern, consts, grid, block)
            base = sigs[0]
            for s in sigs[1:]:
                if s != base:
                    raise C.Refusal("sampled blocks have different phase signatures for %s: set sampling='classes'" % kid)
            kphases.append(dict(kernel_id=kid, **base))
            kunique.append((kid, obs, base))
    unknown = any(p["read_sectors"] is None or p["write_sectors"] is None for k in kphases for p in k["phases"])
    prow = dict(status="unsupported" if unknown else "conditional_static_phases", kernels=kphases,
                reason="data-dependent global address; transaction counts remain null" if unknown else None)
    uentries = []
    for (kid, obs, base), kp, blk in zip(kunique, kphases, kblocks):
        if obs is None:
            e = dict(base, blocks_per_sm=(blk.get("occupancy") or {}).get("blocks_per_sm"))
        else:
            e = U.kernel_entry(obs, kp, (blk.get("occupancy") or {}).get("blocks_per_sm"))
        e["kernel_id"] = kid
        uentries.append(e)
    ok = all(e["status"] == "ok" for e in uentries)
    bad = next((e for e in uentries if e["status"] != "ok"), None)
    urow = dict(status="ok" if ok else bad["status"], reason=None if ok else bad["reason"],
                kernels_per_launch=len(kblocks), kernels=uentries if ok else [])
    main, secondary = kblocks[-1], kblocks[:-1]
    d = cell["_dev"]
    rec = {"cell_id": cell["cell_id"], "operator_id": cell["operator_id"], "cell": "%s/%s" % (cell["regime"], cell["candidate_id"]),
           "corpus": "cuda_samples_unseen", "kernels_per_launch": len(kblocks)}
    rec["inputs"] = {"isolated_sass_sha256": main["isolated_sass_sha256"], "cubin_sha256": main["cubin_sha256"],
                     "kernel_symbol": main["kernel_symbol"], "secondary_kernel_symbols": [s["kernel_symbol"] for s in secondary]}
    geom = X.dev_geometry(d, hw)
    rec["geometry"] = geom
    rec["resources"], rec["occupancy"], rec["work"], rec["structure"] = main["resources"], main["occupancy"], main["work"], main["structure"]
    mem = X.dev_memory(d)
    if main.get("memory_executed"):
        mem.update({k: v for k, v in main["memory_executed"].items() if k.startswith(("executed_", "global_"))})
    mem["tier_source_note"] = "tier from the footprint-over-verified-L2 rule written in cells.py, not a counter"
    mem["executed_bytes_scope"] = "main kernel only; per-launch sums are in per_launch_totals"
    rec["memory"] = mem
    rec["main_kernel"] = {k: main[k] for k in ("kernel_id", "role", "kernel_symbol", "abi", "interpreter_outcomes", "shape_problems")}
    rec["secondary_kernels"] = secondary
    rec["per_launch_totals"] = A.per_launch_totals(kblocks)
    miss = []
    for b in kblocks:
        for m_ in b["missing_features"]:
            miss.append(dict(m_, kernel=b["kernel_id"]))
        if not b["abi"]["all_constant_bank_accesses_on_declared_fields"]:
            miss.append({"feature": "parameter ABI", "reason": "; ".join(b["abi"]["problems"]), "kernel": b["kernel_id"]})
    rec["missing_features"] = miss
    rec["status"] = "supported" if not miss and not any(
        (b["work"] or {}).get("opcodes_with_unknown_guard_upper_bound") for b in kblocks) else "supported_with_assumptions" if not miss else "missing_features"
    rec["build_seconds"] = round(time.time() - t0, 1)
    return rec, prow, urow


def main():
    import cells as CELLS  # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    hw = X.load_hardware(X.read_text(X.GROUND_TRUTH))
    cells = [c for c in CELLS.define_cells(hw) if c["family"] == args.kernel and (not args.only or c["cell_id"] == args.only)]
    if args.list:
        for c in cells:
            print(c["cell_id"], c["tier"], "%.3f" % c["l2_ratio"], [k["kid"] for k in c["kernels"]])
        return
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    for c in cells:
        out = CELL_DIR / (c["cell_id"].replace("/", "__") + ".json")
        if out.exists():
            print("skip (exists)", c["cell_id"], flush=True)
            continue
        try:
            rec, prow, urow = build_cell(c, hw)
            doc = dict(features=rec, phases=prow, unique=urow, cell={k: v for k, v in c.items() if k != "_dev"})
            print(c["cell_id"], rec["status"], prow["status"], urow["status"], "%.0fs" % rec["build_seconds"], flush=True)
        except (C.Refusal, P.Refusal) as ex:
            doc = dict(refused=str(ex), cell={k: v for k, v in c.items() if k != "_dev"})
            print(c["cell_id"], "REFUSED", str(ex)[:160], flush=True)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
