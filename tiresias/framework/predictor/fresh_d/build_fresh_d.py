#!/usr/bin/env python3
"""Build fresh set D (28 CUDA Samples cells at new geometries) with a regression gate. CPU only, no GPU, no measured value.

Order:
 1. REGRESSION GATE. The wrapper (fresh_d_lib.analyse, binding replaced by explicit-geometry binding) is run over development cells
    that cover every kernel of the set; its derivation counts, coalescing totals, feature row, phase row and first-touch row must
    equal the frozen development rows (candidate_counts.json / collective_counts.json, coalescing/static_sectors_frozen.json,
    features_blackwell.json, phases_blackwell.json, reuse/phases_unique_blackwell.json) EXACTLY (canonical JSON). Result:
    gate_fresh_d.json. A fresh cell of an operator with any failing gate cell is refused, never built.
 2. NEW CELLS. The 28 cells of fresh_cells_d.json are analysed (a process pool, one cell per task; results cached under cache/).
    Three reduce3/reduce4/reduce5 cells at one new n are analysed as refusal probes only (kept out of the tables).
 3. Tables written next to this file in the schemas of features_blackwell.json, phases_blackwell.json and
    reuse/phases_unique_blackwell.json: features_fresh_d.json, phases_fresh_d.json, phases_unique_fresh_d.json, plus fresh_cells_d.json.

Usage (background; reduce6 cells take tens of minutes):
    python3 build_fresh_d.py --jobs 8            # gate, then build
    python3 build_fresh_d.py --gate-only
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_d_lib as L  # noqa: E402

X, C, SR = L.X, L.C, L.SR
CACHE = HERE / "cache"

GATE_CELLS = [  # development cells; at least one per kernel of the set, plus larger geometries and reproduced refusals
    "final_cuda_samples_copy/small/c1", "final_cuda_samples_copy/small/c4", "final_cuda_samples_copy/medium/c2",
    "final_cuda_samples_copy/large/c1", "final_cuda_samples_copy/large/c4",
    "train_cuda_samples_transpose/small/c1", "train_cuda_samples_transpose/small/c2", "train_cuda_samples_transpose/small/c3",
    "train_cuda_samples_transpose/small/c4", "train_cuda_samples_transpose/medium/c2", "train_cuda_samples_transpose/large/c4",
    "alt_cuda_samples_copy/small/c1", "alt_cuda_samples_copy/small/c2", "alt_cuda_samples_copy/large/c2",
    "alt_cuda_samples_transposefine/small/c1", "alt_cuda_samples_transposefine/small/c2", "alt_cuda_samples_transposefine/large/c1",
    "alt_cuda_samples_reduction/small/c1", "alt_cuda_samples_reduction/small/c2", "alt_cuda_samples_reduction/small/c3",
    "alt_cuda_samples_reduction/medium/c3", "alt_cuda_samples_reduction/large/c1", "alt_cuda_samples_reduction/large/c2",
    "alt_cuda_samples_reduction/large/c3",
    "train_cuda_samples_reduction/small/c1", "train_cuda_samples_reduction/large/c1",
    "train_cuda_samples_reduction/small/c2", "train_cuda_samples_reduction/small/c3",   # refusals the frozen tables record
]
# which gate cells vouch for which new operator (every kernel of the operator needs at least one)
GATE_FOR_OPERATOR = {
    "fresh_d_cuda_samples_vector_add": ["final_cuda_samples_copy/"],
    "fresh_d_cuda_samples_transpose": ["train_cuda_samples_transpose/"],
    "fresh_d_cuda_samples_copy": ["alt_cuda_samples_copy/"],
    "fresh_d_cuda_samples_transposefine": ["alt_cuda_samples_transposefine/"],
    "fresh_d_cuda_samples_reduction": ["alt_cuda_samples_reduction/"],
    "fresh_d_cuda_samples_reduce2": ["train_cuda_samples_reduction/small/c1", "train_cuda_samples_reduction/large/c1"],
}
PROBE_N = 6_291_456
PROBES = [("reduce3", "small/c2"), ("reduce4", "small/c3"), ("reduce5", "small/c4")]   # origin cells of train_cuda_samples_reduction


def canon(x):
    return json.dumps(x, sort_keys=True)


def frozen_tables():
    ft = {r["cell_id"]: r for r in json.loads((SR / "features_blackwell.json").read_text())["rows"]}
    ph = json.loads((SR / "phases_blackwell.json").read_text())["rows"]
    un = json.loads((SR / "reuse/phases_unique_blackwell.json").read_text())["rows"]
    co = {r["cell_id"]: r for r in json.loads((SR / "coalescing/static_sectors_frozen.json").read_text())["rows"]}
    cand = {(r["operator_id"], r["cell"]): r for r in json.loads((L.OPD / "candidate_counts.json").read_text())["rows"]}
    coll = {(r["operator_id"], r["cell"]): r for r in json.loads((L.OPD / "collective_counts.json").read_text())["rows"]}
    return ft, ph, un, co, cand, coll


def compare_dev(out, cid, frozen):
    ft, ph, un, co, cand, coll = frozen
    oid, cell = cid.split("/", 2)[1], cid.split("/", 2)[2]
    ref = cand[(oid, cell)]
    if ref["status"] != "candidate_class_counts":
        ref = coll[(oid, cell)]
    drop = ("operator_id", "cell", "corpus")
    der_equal = canon(out["derivation"]) == canon({k: v for k, v in ref.items() if k not in drop})
    cf = co[cid]
    coal_equal = (out["coalescing"].get("status") == cf["status"] and
                  all(canon(out["coalescing"][k]) == canon(cf[k]) for k in out["coalescing"]))
    return {"derivation": der_equal, "coalescing": coal_equal, "features": canon(out["features"]) == canon(ft[cid]),
            "phases": canon(out["phases"]) == canon(ph[cid]), "first_touch": canon(out["unique"]) == canon(un[cid]),
            "phase_status": out["phases"]["status"], "first_touch_status": out["unique"]["status"],
            "derivation_path": out["derivation_path"]}


def gate_task(cid):
    dev = L.load_dev()
    t = time.time()
    spec = L.dev_spec("blackwell/" + cid, dev)
    out = L.analyse(spec)
    res = compare_dev(out, "blackwell/" + cid, frozen_tables())
    res.update(cell=cid, seconds=round(time.time() - t, 1), kernel_symbol=spec["kernel_symbol"])
    res["all_equal"] = all(res[k] for k in ("derivation", "coalescing", "features", "phases", "first_touch"))
    return res


def gate_by_operator(results):
    status = {}
    for op, prefixes in GATE_FOR_OPERATOR.items():
        mine = [r for r in results if any(r["cell"].startswith(p) for p in prefixes)]
        status[op] = {"gate_cells": [r["cell"] for r in mine], "passed": bool(mine) and all(r["all_equal"] for r in mine)}
    return status


# ----------------------------------------------------------------------------------------------- new cells
def cell_spec(cell, sm_count):
    return dict(cell_id=cell["cell_id"], operator_id=cell["operator_id"], cell_name="%s/%s" % (cell["regime"], cell["candidate_id"]),
                family=cell["kernel_family"], origin=[cell["origin_operator_id"], cell["origin_cell"]], geometry=cell["geometry"],
                dev_like=L.fresh_dev_like(cell, sm_count))


def new_task(spec):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / (spec["cell_id"].replace("/", "__") + ".json")
    key = L.sha(L.HERE / "fresh_d_lib.py")
    if path.exists():
        doc = json.loads(path.read_text())
        if doc.get("lib_sha256") == key:
            return doc["out"]
    t = time.time()
    out = L.analyse(spec)
    out["seconds"] = round(time.time() - t, 1)
    path.write_text(json.dumps({"lib_sha256": key, "out": out}, sort_keys=True) + "\n")
    return out


def probe_specs(sm_count):
    rows = L.retained_rows()
    specs = []
    for kernel, origin_cell in PROBES:
        origin = rows[("train_cuda_samples_reduction", origin_cell)]
        geometry = dict(n=PROBE_N, threads=256, blocks=PROBE_N // 256, grid=[PROBE_N // 256, 1, 1], block=[256, 1, 1])
        logical = 4 * PROBE_N + 4 * geometry["blocks"]
        cell = dict(cell_id="blackwell/fresh_d_probe_%s/l2/c1" % kernel, operator_id="fresh_d_probe_" + kernel, regime="l2", candidate_id="c1",
                    kernel_family="reduction", tier="L2", logical_bytes_per_launch=logical, grid_blocks=geometry["blocks"], block_threads=256,
                    shape_a=PROBE_N, shape_b=1)
        specs.append((kernel, origin["isolated_function_section"], dict(cell_id=cell["cell_id"], operator_id=cell["operator_id"], cell_name="l2/c1",
                      family="reduction", origin=["train_cuda_samples_reduction", origin_cell], geometry=geometry,
                      dev_like=L.fresh_dev_like(cell, sm_count))))
    return specs


def assemble(cells, results, gate_ops, l2_bytes, hw, probes):
    by_id = {r["cell_id"]: r for r in results}
    rows = {c["cell_id"]: by_id[c["cell_id"]] for c in cells if c["cell_id"] in by_id}
    base_ph = json.loads((SR / "phases_blackwell.json").read_text())
    X.READ_HASHES.clear()
    for p in (L.OPD / "derive.py", L.OPD / "collective.py", X.MI / "retained_count/derive.py", L.D.CONFIG, X.MI.parent / "adapters/adapters.json", X.GROUND_TRUTH):
        X.read_bytes(p)
    rr = L.retained_rows()
    for c in cells:
        o = rr[(c["origin_operator_id"], c["origin_cell"])]
        for key in ("cubin_path", "disassembly_path", "retained_source_path"):
            X.read_bytes(L.CUDA_ROOT / o[key])
        X.read_bytes(L.CUDA_ROOT / "retention_manifest.json")
        for ref in o["retained_compiler_events"]:
            X.read_bytes(L.CUDA_ROOT / ref["path"])
    frows = sorted((rows[c["cell_id"]]["features"] for c in cells if c["cell_id"] in rows), key=lambda r: r["cell_id"])
    features = {
        "schema": "static_runtime_features_blackwell/1",
        "note": ("Label-free static features. No runtime, energy or power column was read. Lane-level counts are not warp-issued counts. "
                 "Derivation outputs are candidates, not profiler-validated or scientifically admitted. "
                 "FRESH SET D (fresh_cells_d.json): retained CUDA Samples cubins bound to new explicit geometry by fresh_d_lib.py; "
                 "tier, bytes and grid come from the fresh cell table, not from the development table."),
        "extract_features_py_sha256": L.sha(L.SR / "extract_features.py"),
        "hardware_from_ground_truth": hw,
        "hardware_constants_not_in_ground_truth": list(X.HW_NOT_IN_GROUND_TRUTH),
        "coverage": X.coverage(frows),
        "input_sha256": dict(sorted(X.READ_HASHES.items())),
        "rows": frows,
    }
    paths = [SR / "phases.py", SR / "extract_features.py", SR / "coalescing/coalescing.py", L.OPD / "derive.py", L.OPD / "collective.py",
             SR / "reuse/phases_unique.py", HERE / "fresh_d_lib.py", HERE / "build_fresh_d.py"]
    phases = {"schema": base_ph["schema"], "input_sha256": {str(p.relative_to(X.REPO)): L.sha(p) for p in paths},
              "assumptions": base_ph["assumptions"] + [
                  "Fresh set D: retained CUDA Samples evidence of a development cell with the same kernel is hash-verified under that cell's own identity; "
                  "only the launch binding (grid, block and scalar arguments) is new and explicit (fresh_d_lib.bind_fresh).",
                  "Frozen full-launch coalescing totals that phases.retained() cross-checks are, for fresh cells, the totals of coalescing.analyse_cell() on the same new cell."],
              "transaction_profiler_validation": False,
              "rows": {cid: r["phases"] for cid, r in rows.items()}}
    unique = {"schema": "first_touch_unique_sectors/1", "sector_bytes": 32,
              "derived_from": "tiresias/framework/predictor/fresh_d/phases_fresh_d.json (this package; frozen development tables unchanged)",
              "uses_measured_runtime_or_energy": False, "rows": {cid: r["unique"] for cid, r in rows.items()}}
    return features, phases, unique


def pool_task(task):
    kind, payload = task
    if kind == "gate":
        return kind, gate_task(payload)
    return kind, new_task(payload)


def refused_outputs(cell, reason, hw):
    dev_like = L.fresh_dev_like(cell, hw["sm_count"])
    return {"cell_id": cell["cell_id"], "derivation_path": None, "derivation": {"status": "refused", "reason": reason},
            "coalescing": {"status": "refused", "reason": reason}, "phases": {"status": "unsupported", "reason": reason, "kernels": []},
            "unique": {"status": "unsupported", "reason": reason, "kernels": []},
            "features": {"cell_id": cell["cell_id"], "operator_id": cell["operator_id"], "cell": "%s/%s" % (cell["regime"], cell["candidate_id"]), "corpus": "cuda",
                         "status": "missing_features", "missing_features": [{"feature": "all", "reason": reason}],
                         "geometry": X.dev_geometry(dev_like, L.l2_bytes_from_ground_truth()[0]), "memory": X.dev_memory(dev_like)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=HERE, help="where tables are written (default: next to this file)")
    ap.add_argument("--exclude", default="", help="comma-separated substrings: skip cells/gate cells containing one (TESTING the assembly only; output is partial)")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    excl = [e for e in args.exclude.split(",") if e]
    hw, l2_bytes = L.l2_bytes_from_ground_truth()
    t0 = time.time()
    cells = L.define_cells(l2_bytes)
    cells = [c for c in cells if not any(e in c["cell_id"] for e in excl)]
    gate_cells = [g for g in GATE_CELLS if not any(e in g for e in excl)]
    pspecs = probe_specs(hw["sm_count"])
    # one pool for everything (no barrier between the gate and the new cells); most expensive first (reduce6 cost ~ trips = n/32768)
    tasks = [("gate", g) for g in gate_cells]
    if not args.gate_only:
        tasks += [("cell", cell_spec(c, hw["sm_count"])) for c in cells] + [("cell", p[2]) for p in pspecs]

    def cost(t):
        kind, x = t
        if kind == "gate":
            return 1e6 if x.endswith("large/c3") else 5e4 if "reduction/medium/c3" in x else 1
        g = x["geometry"]
        return (g["n"] / 32768 * 1e3 if x["origin"][1] == "small/c3" and x["origin"][0].startswith("alt_") else 1) if x["family"] == "reduction" else 1
    tasks.sort(key=lambda t: -cost(t))
    ctx = mp.get_context("fork")
    gate_results, cell_results = [], {}
    with ctx.Pool(args.jobs) as pool:
        for kind, r in pool.imap_unordered(pool_task, tasks):
            if kind == "gate":
                print("gate %-55s %s (%.0fs)" % (r["cell"], "PASS byte-exact" if r["all_equal"] else "FAIL " + str(r), r["seconds"]), flush=True)
                gate_results.append(r)
            else:
                cell_results[r["cell_id"]] = r
                print("cell %-62s phases=%s first_touch=%s features=%s (%.0fs)" % (r["cell_id"], r["phases"]["status"], r["unique"]["status"], r["features"]["status"], r.get("seconds", 0)), flush=True)
    gate_results.sort(key=lambda r: GATE_CELLS.index(r["cell"]))
    ops = gate_by_operator(gate_results)
    gate = {"schema": "fresh_d_regression_gate/1", "fresh_d_lib_py_sha256": L.sha(HERE / "fresh_d_lib.py"),
            "rule": "every development gate cell must reproduce its frozen derivation counts, coalescing totals, feature row, phase row and first-touch row byte-exactly (canonical JSON)",
            "cells": gate_results, "operators": ops, "all_passed": all(r["all_equal"] for r in gate_results), "partial_test_run": bool(excl)}
    (outdir / "gate_fresh_d.json").write_text(json.dumps(gate, indent=1, sort_keys=True) + "\n")
    print("GATE all_passed=%s operators=%s (%.0fs)" % (gate["all_passed"], {k: v["passed"] for k, v in ops.items()}, time.time() - t0), flush=True)
    if args.gate_only:
        return 0 if gate["all_passed"] else 1
    results, refused = [], []
    for c in cells:
        if ops[c["operator_id"]]["passed"]:
            results.append(cell_results[c["cell_id"]])
        else:
            refused.append(c["cell_id"])
            results.append(refused_outputs(c, "regression gate failed for operator " + c["operator_id"], hw))
    probe_out = []
    for (kernel, symbol, spec) in pspecs:
        out = cell_results[spec["cell_id"]]
        probe_out.append({"kernel": kernel, "symbol": symbol, "n": PROBE_N, "derivation_path": out["derivation_path"],
                          "derivation_status": out["derivation"]["status"], "derivation_reason": out["derivation"].get("reason"),
                          "coalescing_status": out["coalescing"].get("status"), "phases_status": out["phases"]["status"],
                          "phases_reason": out["phases"].get("reason"), "first_touch_status": out["unique"]["status"],
                          "first_touch_reason": out["unique"].get("reason"), "features_status": out["features"]["status"]})
        print("probe", probe_out[-1], flush=True)
    by_id = {r["cell_id"]: r for r in results}
    rr = L.retained_rows()
    for c in cells:
        r = by_id[c["cell_id"]]
        origin = rr[(c["origin_operator_id"], c["origin_cell"])]
        c["gate_passed"] = ops[c["operator_id"]]["passed"]
        c["kernel_symbol"] = origin["isolated_function_section"]
        c["retained_cubin_sha256"] = origin["cubin_sha256"]
        c["retained_isolated_sass_sha256"] = origin["disassembly_sha256"]
        c["retained_source_sha256"] = origin["source_sha256"]
        c["timing"] = L.timing_spec(c)
        c.update(features_status=r["features"]["status"], features_missing=r["features"].get("missing_features"),
                 phases_status=r["phases"]["status"], phases_reason=r["phases"].get("reason"), first_touch_status=r["unique"]["status"],
                 first_touch_reason=r["unique"].get("reason"), coalescing_status=r["coalescing"].get("status"), derivation_path=r["derivation_path"],
                 derived_fully=(r["features"]["status"] == "supported" and r["phases"]["status"] == "conditional_static_phases" and r["unique"]["status"] == "ok"))
        c["in_timing_list"] = bool(c["derived_fully"])
    features, phases, unique = assemble(cells, results, ops, l2_bytes, hw, probe_out)
    (outdir / "features_fresh_d.json").write_text(json.dumps(features, indent=1, sort_keys=True) + "\n")
    (outdir / "phases_fresh_d.json").write_text(json.dumps(phases, indent=1, sort_keys=True) + "\n")
    (outdir / "phases_unique_fresh_d.json").write_text(json.dumps(unique, indent=1, sort_keys=True) + "\n")
    doc = {"schema": "fresh_cuda_samples_cells/1", "set": "D",
           "purpose": "Prospective test cells of the static runtime model: pinned CUDA Samples kernels at new geometries; no measured value of any kind is in this file.",
           "hardware_l2_bytes_from_ground_truth": l2_bytes, "source_revision": L.SOURCE_REVISION,
           "tier_rule": "L2 iff footprint / verified L2 capacity < 1 (HARDWARE_GROUND_TRUTH.md Blackwell); footprint = logical bytes per launch",
           "logical_bytes_rule": {"vector_add": "3*n*4", "transposes / copy / tile kernels": "2*dim*dim*4", "reductions": "4*n + 4*blocks (blocks = launched grid)"},
           "cells": cells, "refused_by_gate": refused, "refusal_probes": probe_out,
           "coverage": dict(collections.Counter("%s derived_fully=%s" % (c["operator_id"], c["derived_fully"]) for c in cells))}
    (outdir / "fresh_cells_d.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "derived_fully": sum(c["derived_fully"] for c in cells), "features": features["coverage"]["by_status"],
                      "phases": dict(collections.Counter(r["status"] for r in phases["rows"].values())),
                      "first_touch": dict(collections.Counter(r["status"] for r in unique["rows"].values())), "seconds": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
