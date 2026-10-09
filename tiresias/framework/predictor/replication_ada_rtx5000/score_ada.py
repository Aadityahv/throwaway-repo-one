#!/usr/bin/env python3
"""Score the frozen Ada predictions against the measurements, once. Mirrors h100_eval_freeze/score_h100.py (whose scoring functions are imported unchanged) with Ada's own cells, cap and cell names.

    python score_ada.py --profile main|tensor --timing <timing_ada_result.json> [<more timing files of the same profile> ...] [--energy-raw <dir> ...] [--baselines baselines_ada.json] --out <score file>
    python score_ada.py --joint --timing-main <file> [<file> ...] --timing-tensor <file> [<file> ...] --energy-raw <dir> [<dir> ...] [--window-gates <file> ...] --out <score file>

--joint scores both profiles together over all 168 cells (as score_board.py does for the H100 replication) and adds the held-out breakdown, the reference controls, the clock-throttle check and the
decision summary. A profile's timing may come from several runs (resume runs): every file must record the frozen predictions sha256, no cell may be timed twice, and the merged cells must be exactly the
profile's cells; every energy directory is read, no cell may have two energy rows.


Conventions of the Blackwell and H100 scoring (tiresias/evaluation/results/build_eval_cells.py, make_paper_figures.py): median and 90th percentile of the absolute relative error; a refused or unsupported cell is a
FAILURE (infinite error; a percentile that reaches one is reported null); energy "below the limit" = mean window power below 95% of Ada's enforced power limit, read from HARDWARE_GROUND_TRUTH.md (250 W); windows above
98.5% of the limit are excluded for every method alike (the calibrator's admission rule). Reads ONLY the committed predictions (sha256 equal to freeze_record_ada*.json) and the measurement files.
Per-wave L2 refusals, unmeasured-opcode refusals (F2FP, FMNMX) and unsupported cells are failures in every table; per set and per group (samples, ml, tensor, validation, prospective) rows are added.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import board_ada as B
import cell_sets_ada as CS

HERE = B.HERE
sys.path.insert(0, str(B.FZ))
import score_h100 as S  # noqa: E402  (scoring functions reused unchanged)

Refusal = S.Refusal
sha = S.sha


def load_frozen(pred_path, record_path):
    pred_path, record_path = Path(pred_path), Path(record_path)
    doc = json.loads(pred_path.read_text(encoding="utf-8"))
    if doc.get("kind") != "ada_frozen_predictions" or doc.get("warning"):
        raise Refusal("%s is not an Ada frozen-predictions file" % pred_path.name)
    if not record_path.is_file():
        raise Refusal("%s is missing" % record_path.name)
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    if sha(pred_path) != rec.get("predictions_sha256"):
        raise Refusal("%s differs from the sha256 recorded at the freeze" % pred_path.name)
    return doc, rec


def read_energy(dirs):
    rows = {}
    for d in dirs:
        p = Path(d) / "application_energy_raw.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                cid = "%s/%s/%s/%s" % (B.PREFIX, r["parent_id"], r["regime"], r["candidate_id"])
                if cid in rows:
                    raise Refusal("two energy rows for %s" % cid)
                total, interval = float(r["board_energy_j_total"]), float(r["counted_launch_interval_s"])
                rows[cid] = dict(energy_j=float(r["board_energy_j_per_launch"]), mean_power_w=total / interval, launches=int(r["launch_count"]), window_s=interval, gpu_uuid=r.get("gpu_uuid"), correctness_check=r.get("correctness_check"), source_dir=str(d))
    return rows


def merge_timings(paths, rec, profile, cell_ids=None):
    """One timing record per cell from one or more timing files of a profile (the first run plus resume runs). Refuses unless every file recorded the frozen predictions sha256 of the freeze record and
    the profile, no cell is timed in two files (or twice in one), and, when cell_ids is given, the merged cells are exactly those cells."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not paths:
        raise Refusal("no timing file given for profile %s" % profile)
    cells, gate_refused, engines, metas, seen = [], [], {}, [], {}
    for i, p in enumerate(paths):
        t = json.loads(Path(p).read_text(encoding="utf-8"))
        meta = t.get("meta") or {}
        got = ((meta.get("freeze")) or {}).get("predictions_sha256")
        if got != rec["predictions_sha256"]:
            raise Refusal("%s was not taken against the frozen predictions (recorded predictions sha256 %r)" % (Path(p).name, got))
        if meta.get("profile") is not None and meta.get("profile") != profile:
            raise Refusal("%s is of profile %r, not %r" % (Path(p).name, meta.get("profile"), profile))
        for r in t["cells"]:
            if r["cell_id"] in seen:
                raise Refusal("cell %s is timed twice (%s and %s)" % (r["cell_id"], seen[r["cell_id"]], Path(p).parent.name + "/" + Path(p).name))
            seen[r["cell_id"]] = Path(p).parent.name + "/" + Path(p).name
            cells.append(r)
        gate_refused += t.get("gate_refused_cells", [])
        for k, v in (t.get("engines") or {}).items():
            engines["%d:%s" % (i, k)] = v
        metas.append(dict(file=str(p), sha256=sha(p), cells=len(t["cells"]), gpu_uuid=((meta.get("gpu")) or {}).get("uuid"), hipc_commit=meta.get("hipc_commit")))
    if cell_ids is not None and set(seen) != set(cell_ids):
        raise Refusal("the merged timing cells are not exactly the %s profile's cells (missing %d: %s; extra %d: %s)" % (profile, len(set(cell_ids) - set(seen)), sorted(set(cell_ids) - set(seen))[:3], len(set(seen) - set(cell_ids)), sorted(set(seen) - set(cell_ids))[:3]))
    return dict(meta=dict(gpu=dict(uuid=metas[0]["gpu_uuid"]), files=metas, profile=profile, uuids=sorted({m["gpu_uuid"] for m in metas if m["gpu_uuid"]})), cells=cells, gate_refused_cells=gate_refused, engines=engines)


def score(pred_path, record_path, timing_path, energy_dirs=(), baselines_path=None, profile="main", cells_list=None):
    pred, rec = load_frozen(pred_path, record_path)
    if (pred.get("profile") or "main") != profile:
        raise Refusal("the predictions are of profile %r, not %r" % (pred.get("profile"), profile))
    limit = B.power_limit_w()
    cal_cap = (pred.get("calibration") or {}).get("energy_cap_w")
    if cal_cap is not None and abs(float(cal_cap) - limit) > 1e-6:
        raise Refusal("the calibration's enforced cap (%s W) differs from Ada's power limit in HARDWARE_GROUND_TRUTH.md (%s W)" % (cal_cap, limit))
    pred = dict(pred, calibration=dict(pred["calibration"], energy_cap_w=limit))
    cells_list = cells_list if cells_list is not None else CS.load_cells(profile)
    cells_by_id = {c["cell_id"]: c for c in cells_list}
    cell_ids = sorted(cells_by_id)
    if set(pred["cells"]) != set(cell_ids):
        raise Refusal("the predictions do not cover exactly the cells of the %s profile" % profile)
    timing = merge_timings(timing_path, rec, profile, cell_ids)
    rt, err, sgn = S.runtime_scores(pred, timing, cell_ids)
    rt["per_set"] = S.group_stats(err, cells_by_id, lambda c: c["set"])
    rt["per_group"] = S.group_stats(err, cells_by_id, lambda c: c["group"])
    rt["per_family"] = S.group_stats(err, cells_by_id, lambda c: c["family"])
    rt["per_tier"] = S.group_stats(err, cells_by_id, lambda c: c["tier"])
    cal_uuid = (pred.get("calibration") or {}).get("uuid")
    meas_uuid = (((timing.get("meta") or {}).get("gpu")) or {}).get("uuid")
    if len(timing["meta"]["uuids"]) > 1:
        raise Refusal("the timing files of profile %s were taken on different GPUs: %s" % (profile, timing["meta"]["uuids"]))
    tp = [timing_path] if isinstance(timing_path, (str, Path)) else list(timing_path)
    rep = dict(schema="ada_score/1", profile=profile, inputs=dict(predictions_sha256=rec["predictions_sha256"], freeze_commit=rec["freeze_commit"], timing_sha256=[sha(p) for p in tp], score_ada_py_sha256=sha(__file__)),
               runtime=rt, power_limit_w=limit, below_limit_fraction=S.BELOW_CAP_FRACTION,
               gpu=dict(calibration_gpu_uuid=cal_uuid, measurement_gpu_uuid=meas_uuid, same_gpu=None if not (cal_uuid and meas_uuid) else cal_uuid == meas_uuid))
    r1 = rt["supported_cells"] and rt["supported_cells"]["median_pct"] is not None and rt["supported_cells"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT
    r2 = rt["unsupported_as_failure"] and rt["unsupported_as_failure"]["median_pct"] is not None and rt["unsupported_as_failure"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT
    r3 = bool(rt["roofline_on_supported"] and rt["supported_cells"]) and rt["supported_cells"]["median_pct"] < rt["roofline_on_supported"]["median_pct"]
    criteria = dict(runtime_median_le_15_supported=bool(r1), runtime_median_le_15_unsupported_as_failure=bool(r2), runtime_beats_roofline_median=bool(r3))
    if energy_dirs:
        meas = read_energy(energy_dirs)
        en = S.energy_scores(pred, meas, timing, cell_ids, cells_by_id, baselines_path)
        rep["energy"] = en
        criteria.update(en["criteria"])
        rep["decision_utility"] = S.decision_utility(pred, meas, timing, cells_by_id)
        rep["cost"] = S.cost(pred, timing, energy_dirs)
    rep["criteria"] = criteria
    rep["stop_rule"] = "If a criterion is false, FREEZE.md's stop rule applies: report as measured, no re-tuning on these cells; later model changes are retrospective and labelled so."
    return rep


def find_calibration_document(pred):
    """The Ada calibration document the predictions were made from, hash-checked against the predictions."""
    p = B.ADA_CALIBRATION
    if not p.is_file() or sha(p) != pred["calibration"]["sha256"]:
        raise Refusal("the calibration document %s is missing or does not match the sha256 in the predictions" % p.name)
    return p


def _subset(ids, rt_per_cell, en_rows):
    """Runtime and energy statistics over a subset of cells, from the per-cell rows of the joint report. A cell without a prediction is an infinite error (failure) in the 'as failure' variants."""
    inf = math.inf
    rt_sup = [abs(rt_per_cell[c]["error_pct"]) for c in ids if c in rt_per_cell and rt_per_cell[c]["error_pct"] is not None]
    rt_all = [abs(rt_per_cell[c]["error_pct"]) if rt_per_cell[c]["error_pct"] is not None else inf for c in ids if c in rt_per_cell]
    en = {c: en_rows[c] for c in ids if c in en_rows}
    ab = lambda x: abs(x) if x is not None else inf
    out = dict(cells=len(ids), runtime_supported=S.stats(rt_sup), runtime_unsupported_as_failure=S.stats(rt_all), energy_cells_scored=len(en))
    for name, key in (("energy_static_runtime", "ours_static_error_pct"), ("energy_measured_runtime", "ours_measured_runtime_error_pct")):
        out[name] = dict(all_supported=S.stats([ab(r[key]) for r in en.values() if r[key] is not None]), below_cap_supported=S.stats([ab(r[key]) for r in en.values() if r[key] is not None and r["below_cap"]]),
                         all_unsupported_as_failure=S.stats([ab(r[key]) for r in en.values()]))
    return out


def _decision_extra(du):
    rows = du["rows"]
    gap = [r for r in rows if r["measured_energy_gap_pct"] > 1.0]
    mean = lambda v: None if not v else round(sum(v) / len(v), 3)
    ours_only = sum(1 for r in gap if r["ours_correct"] and not r["fastest_correct"])
    fast_only = sum(1 for r in gap if r["fastest_correct"] and not r["ours_correct"])
    n = ours_only + fast_only
    p = None
    if n:   # exact two-sided sign test on the groups where exactly one of the two is right
        k = min(ours_only, fast_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(groups=len(rows), groups_with_measured_gap_above_1pct=len(gap), ours_top1_with_gap=sum(r["ours_correct"] for r in gap), fastest_top1_with_gap=sum(r["fastest_correct"] for r in gap),
                mean_regret_pct=dict(ours=mean([r["ours_regret_pct"] for r in rows if r["ours_regret_pct"] is not None]), fastest=mean([r["fastest_regret_pct"] for r in rows]), first_candidate=mean([r["first_candidate_regret_pct"] for r in rows])),
                median_regret_pct_with_gap=dict(ours=None if not gap else round(S.percentile([r["ours_regret_pct"] for r in gap if r["ours_regret_pct"] is not None], 0.5), 3), fastest=None if not gap else round(S.percentile([r["fastest_regret_pct"] for r in gap], 0.5), 3)),
                only_ours_right=ours_only, only_fastest_right=fast_only, sign_test_two_sided_p=None if p is None else round(p, 4),
                groups_without_prediction_for_ours=sum(1 for r in rows if r["ours"] is None))


def score_joint(timing_main, timing_tensor, energy_dirs, window_gates=(), pred_dir=None):
    """Both profiles together over all 168 cells (see the module docstring)."""
    sys.path.append(str(B.SR / "replication_h100_a100"))      # for controls() of the H100 replication scorer, reused unchanged
    import score_board as SB
    d = Path(pred_dir) if pred_dir else HERE
    preds, recs, timings, cells_all = {}, {}, {}, {}
    for prof, tp in (("main", timing_main), ("tensor", timing_tensor)):
        pr = CS.profile(prof)
        preds[prof], recs[prof] = load_frozen(d / pr["predictions"], d / pr["record"])
        if (preds[prof].get("profile") or "main") != prof:
            raise Refusal("%s is not of profile %s" % (pr["predictions"], prof))
        ids = [c["cell_id"] for c in CS.load_cells(prof)]
        for c in CS.load_cells(prof):
            cells_all[c["cell_id"]] = c
        if set(preds[prof]["cells"]) != set(ids):
            raise Refusal("the %s predictions do not cover exactly the cells of the profile" % prof)
        timings[prof] = merge_timings(tp, recs[prof], prof, ids)
    if preds["main"]["calibration"]["sha256"] != preds["tensor"]["calibration"]["sha256"]:
        raise Refusal("the profiles were frozen from different calibration documents")
    limit = B.power_limit_w()
    for prof in preds:
        cap = (preds[prof].get("calibration") or {}).get("energy_cap_w")
        if cap is not None and abs(float(cap) - limit) > 1e-6:
            raise Refusal("the %s calibration's cap (%s W) differs from Ada's power limit (%s W)" % (prof, cap, limit))
    pred = json.loads(json.dumps(preds["main"]))
    pred["cells"].update(preds["tensor"]["cells"])
    pred["roofline_runtime_s"].update(preds["tensor"]["roofline_runtime_s"])
    pred["calibration"]["energy_cap_w"] = limit
    pred["_cells_by_id"] = cells_all
    timing = dict(cells=timings["main"]["cells"] + timings["tensor"]["cells"], gate_refused_cells=timings["main"]["gate_refused_cells"] + timings["tensor"]["gate_refused_cells"],
                  engines={**{"main" + k: v for k, v in timings["main"]["engines"].items()}, **{"tensor" + k: v for k, v in timings["tensor"]["engines"].items()}},
                  meta=dict(gpu=timings["main"]["meta"]["gpu"], uuids=sorted(set(timings["main"]["meta"]["uuids"]) | set(timings["tensor"]["meta"]["uuids"]))))
    if len({c["cell_id"] for c in timing["cells"]}) != len(timing["cells"]) or len(timing["cells"]) != len(cells_all):
        raise Refusal("a cell is timed twice across the profiles")
    cell_ids = sorted(cells_all)
    rt, err, sgn = S.runtime_scores(pred, timing, cell_ids)
    for k, f in (("per_set", lambda c: c["set"]), ("per_group", lambda c: c["group"]), ("per_family", lambda c: c["family"]), ("per_tier", lambda c: c["tier"])):
        rt[k] = S.group_stats(err, cells_all, f)
    rt["per_profile"] = {p: S.stats([e for c, e in err.items() if CS.needs_tensor(cells_all[c]) == (p == "tensor")]) for p in ("main", "tensor")}
    cal_uuid = pred["calibration"].get("uuid")
    meas_uuids = timing["meta"]["uuids"]
    same = None if not (cal_uuid and meas_uuids) else all(u == cal_uuid for u in meas_uuids)
    meas = read_energy(energy_dirs)
    stray = sorted(set(meas) - set(cells_all))
    if stray:
        raise Refusal("energy rows for cells outside the 168: %s" % stray[:3])
    row_uuids = sorted({r["gpu_uuid"] for r in meas.values() if r["gpu_uuid"]})
    rep = dict(schema="ada_score_joint/1", scope=dict(cells=len(cell_ids)),
               inputs=dict(predictions_sha256={p: recs[p]["predictions_sha256"] for p in recs}, freeze_commit={p: recs[p]["freeze_commit"] for p in recs},
                           timing_files={p: timings[p]["meta"]["files"] for p in timings}, energy_dirs=[str(x) for x in energy_dirs], score_ada_py_sha256=sha(__file__)),
               runtime=rt, power_limit_w=limit, below_limit_fraction=S.BELOW_CAP_FRACTION,
               gpu=dict(calibration_gpu_uuid=cal_uuid, measurement_gpu_uuids_timing=meas_uuids, measurement_gpu_uuids_energy_rows=row_uuids, same_gpu=same and row_uuids == [cal_uuid] if same is not None else None,
                        statement=("calibration and measurement (timing and every energy window) ran on the same GPU" if same and row_uuids == [cal_uuid] else
                                   "calibration and measurement ran on DIFFERENT GPUs: every error includes GPU-to-GPU variation" if same is False or row_uuids != [cal_uuid] else
                                   "a GPU UUID is missing: cannot say whether the runs used the same GPU")),
               calibration=dict(source=pred["calibration"].get("source"), file=pred["calibration"].get("file"), sha256=pred["calibration"].get("sha256"), complete=pred["calibration"].get("complete")))
    r1 = bool(rt["supported_cells"] and rt["supported_cells"]["median_pct"] is not None and rt["supported_cells"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT)
    r2 = bool(rt["unsupported_as_failure"] and rt["unsupported_as_failure"]["median_pct"] is not None and rt["unsupported_as_failure"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT)
    r3 = bool(rt["roofline_on_supported"] and rt["supported_cells"]) and rt["supported_cells"]["median_pct"] < rt["roofline_on_supported"]["median_pct"]
    criteria = dict(runtime_median_le_15_supported=r1, runtime_median_le_15_unsupported_as_failure=r2, runtime_beats_roofline_median=bool(r3))
    # energy
    en = S.energy_scores(pred, meas, timing, cell_ids, cells_all, None)
    methods = {}
    for prof in ("main", "tensor"):
        b = json.loads((d / CS.profile(prof)["baselines"]).read_text(encoding="utf-8"))
        for m, v in b["methods"].items():
            methods.setdefault(m, dict(status=v.get("status"), reason=v.get("reason"), profiles=[]))["profiles"].append(prof)
    en["published_methods"] = dict(methods_recorded=methods, note="no published-method row was run on Ada (baselines_ada*.json record each as not run, with the reason); the controls below are reference controls computed from the same measurements, not competitors")
    T = {r["cell_id"]: r for r in timing["cells"] if r.get("correct")}
    labelled = [c for c in cell_ids if c in meas and c in T]
    cells = [c for c in labelled if meas[c]["mean_power_w"] <= S.ADMIT_FRACTION_OF_CAP * limit]
    below = lambda c: meas[c]["mean_power_w"] < S.BELOW_CAP_FRACTION * limit
    docp = find_calibration_document(pred)
    doc = json.loads(docp.read_text(encoding="utf-8"))
    en["controls"] = SB.controls(pred, doc, meas, T, cells, below, limit)
    en["rows_with_failed_correctness_check"] = sorted(c for c, r in meas.items() if r.get("correctness_check") not in (None, "", "True"))
    rep["energy"] = en
    criteria.update(en["criteria"])
    # clock-throttle flags (from the window gates) and their effect
    thr = set()
    for g in window_gates:
        for c in json.loads(Path(g).read_text(encoding="utf-8")).get("clock_throttled", []):
            thr.add("%s/%s" % (B.PREFIX, c.split("/", 1)[1]) if "/" in c else c)
    thr &= set(en["per_cell"])
    if window_gates:
        pc = en["per_cell"]
        side = lambda ids, key: S.stats([abs(pc[c][key]) for c in ids if pc[c][key] is not None])
        yes, no = sorted(c for c in pc if c in thr), sorted(c for c in pc if c not in thr)
        rep["clock_throttled_windows"] = dict(note="a throttled flag is reported, not an exclusion", flagged_total_in_gates=len(thr), among_scored_cells=len(yes),
                                                   static_runtime_error=dict(throttled=side(yes, "ours_static_error_pct"), not_throttled=side(no, "ours_static_error_pct")),
                                                   measured_runtime_error=dict(throttled=side(yes, "ours_measured_runtime_error_pct"), not_throttled=side(no, "ours_measured_runtime_error_pct")),
                                                   cells=yes, window_gate_files=[str(g) for g in window_gates])
    rep["decision_utility"] = S.decision_utility(pred, meas, timing, cells_all)
    rep["decision_utility"]["summary"] = _decision_extra(rep["decision_utility"])
    stages = doc.get("stages", [])
    rep["cost"] = S.cost(dict(pred, calibration=dict(pred["calibration"], board_seconds_recorded=round(sum(float(st.get("seconds") or 0) for st in stages), 2),
                                                     stages_without_recorded_seconds=[st.get("name") or st.get("program") for st in stages if st.get("seconds") is None])), timing, energy_dirs)
    # held-out breakdown: only the 24 prospective cells are out of sample for the model's rule development
    held = sorted(c for c in cell_ids if cells_all[c]["set"] == "prospective")
    rest = sorted(c for c in cell_ids if c not in set(held))
    valid = sorted(c for c in cell_ids if cells_all[c]["set"] == "validation")
    rpc = rt["per_cell"]
    rep["held_out"] = dict(note="The whole Ada result is out of sample for Ada's data (frozen before any Ada cell was timed). Only the 24 prospective cells are also held out from the development of the model's rules (CURRENT_MODELS.md); "
                                "they are all tensor-core or attention kernels, a harder mix, so the gap to the rest is not purely a measure of optimism.",
                           prospective=_subset(held, rpc, en["per_cell"]), all_other_cells=_subset(rest, rpc, en["per_cell"]), validation=_subset(valid, rpc, en["per_cell"]),
                           tensor_group=_subset([c for c in cell_ids if CS.needs_tensor(cells_all[c])], rpc, en["per_cell"]))
    rep["criteria"] = criteria
    rep["stop_rule"] = "If a criterion is false, FREEZE.md's stop rule applies: report as measured, no re-tuning on these cells; later model changes are retrospective and labelled so."
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(CS.PROFILES), default="main")
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--timing", type=Path, nargs="*", default=[])
    ap.add_argument("--joint", action="store_true", help="score both profiles together over all 168 cells")
    ap.add_argument("--timing-main", type=Path, nargs="*", default=[])
    ap.add_argument("--timing-tensor", type=Path, nargs="*", default=[])
    ap.add_argument("--window-gates", type=Path, nargs="*", default=[], help="window_gate_*.json files of the energy runs (clock-throttle flags)")
    ap.add_argument("--energy-raw", type=Path, nargs="*", default=[])
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise Refusal("%s exists; scored once, never overwritten" % a.out)
    if a.joint:
        rep = score_joint(a.timing_main, a.timing_tensor, a.energy_raw, a.window_gates)
        a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(rep["criteria"], indent=1))
        print("runtime supported:", rep["runtime"]["supported_cells"], "| unsupported as failure:", rep["runtime"]["unsupported_as_failure"])
        return 0
    if not a.timing:
        raise Refusal("--timing is required (or --joint with --timing-main and --timing-tensor)")
    pr = CS.profile(a.profile)
    rep = score(a.predictions or HERE / pr["predictions"], a.record or HERE / pr["record"], a.timing, a.energy_raw, a.baselines, a.profile)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep["criteria"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
