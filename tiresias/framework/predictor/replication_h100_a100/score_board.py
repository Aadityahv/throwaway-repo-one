#!/usr/bin/env python3
"""Score the frozen A100/H100 replication predictions against the measurements, once (CPU only). Reads ONLY the committed predictions (checked against their freeze records) and the measurement files.

    python score_board.py --board h100 --timing-main <timing result> --timing-tensor <timing result> --energy-raw <dir> [<dir> ...] --out <score.json>

Both profiles are scored together over the 168 intended cells (main 124 + tensor 44); a profile whose timing result is not given is reported as not measured. Refuses unless every predictions file is a
frozen file of this board whose sha256 equals its freeze record AND the matching timing result was taken against those predictions (its recorded predictions sha256). Criteria, exclusions and the stop
rule are those of the replication plan and of the earlier H100 scorer (score_h100.py, whose helpers this file reuses unchanged):

  runtime   median and 90th percentile of |predicted/measured - 1| over cells with a prediction; the same with every unsupported cell counted as a failure (infinite error); against the
            calibration's roofline reference; per family, set and tier. Criterion: median at most 15 percent, with and without failures, and better than the roofline reference.
  energy    ours at the static runtime and at the measured runtime (min(cap x t, base x t + terms)), below cap (mean window power under 95 percent of the enforced limit) and over all cells;
            cells measured above 98.5 percent of the limit are excluded alike for every method and listed. Criteria: static within 5 points (median) and 10 points (90th percentile) of the
            measured-runtime route, below cap.
  controls  computed from the same measurements, labelled reference controls, not competitors (published-method rows were not run on this board, by recorded decision of 4 October 2026):
            calibration-mean power x measured runtime; one energy anchor per kernel family (measured mean power of its first candidate at its first L2-tier size) x measured runtime.
  decisions the lower-energy candidate of each equal-work group chosen by ours, by the faster measured candidate and by the first candidate; regret, top-1, groups with a measured gap above 1 percent.
  cost      calibration board seconds, median service time of one energy window as measured, break-even cell count.
The calibration board and the measurement board are compared and the statement is written into the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SR / "h100_eval_freeze"))
import board as BD  # noqa: E402
import cell_sets_board as CS  # noqa: E402
import score_h100 as S  # noqa: E402  (percentile, stats, energy_at, runtime_scores, group_stats, groups_of, decision_utility, cost, constants)

Refusal = S.Refusal
sha = S.sha


def load_frozen(board, profile, tree=None):
    d = Path(tree) if tree else CS.board_dir(board)
    n = CS.names(board, profile)
    pred_path, rec_path = d / n["predictions"], d / n["record"]
    if not pred_path.is_file() or not rec_path.is_file():
        raise Refusal("%s or %s is missing" % (n["predictions"], n["record"]))
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if pred.get("kind") != "cluster_frozen_predictions" or pred.get("warning") or pred.get("board") != board or pred.get("profile") != profile:
        raise Refusal("%s is not a frozen-predictions file of board %s profile %s" % (n["predictions"], board, profile))
    if sha(pred_path) != rec.get("predictions_sha256"):
        raise Refusal("%s differs from the sha256 recorded at the freeze" % n["predictions"])
    return pred, rec


def check_timing(timing, rec, profile):
    got = (((timing.get("meta") or {}).get("freeze")) or {}).get("predictions_sha256")
    if got != rec["predictions_sha256"]:
        raise Refusal("the %s timing result was not taken against the frozen predictions (its recorded predictions sha256 is %r)" % (profile, got))
    if (timing.get("meta") or {}).get("profile") != profile:
        raise Refusal("the %s timing result is of profile %r" % (profile, (timing.get("meta") or {}).get("profile")))


def find_calibration_document(pred):
    """The calibration document the predictions were made from, found in the archived runs and hash-checked against the predictions."""
    name, want = pred["calibration"]["file"], pred["calibration"]["sha256"]
    for p in sorted((SR / "calibrate" / "runs").rglob(name)):
        if sha(p) == want:
            return p
    raise Refusal("calibration document %s with sha256 %s... was not found under calibrate/runs/" % (name, want[:12]))


def controls(pred, doc, meas, T, cells, below, cap):
    """Reference controls from the same measurements (see the module docstring). Cells without energy are not scored; unsupported never applies (controls need no model)."""
    fit = [w for w in doc["constants"]["energy"]["windows"] if w.get("role") == "fit" and w.get("admitted")]
    if not fit:
        raise Refusal("the calibration document has no admitted fit windows")
    p_const = sum(w["power_w"] for w in fit) / len(fit)
    cell_by = pred["_cells_by_id"]
    anchor = {}
    for fam in sorted({c["family"] for c in cell_by.values()}):
        cand = [c for c in cell_by.values() if c["family"] == fam and c["candidate_id"] == "c1" and c["tier"] == "L2" and c["cell_id"] in meas and c["cell_id"] in T]
        if cand:
            first = sorted(cand, key=lambda c: (c["regime"], c["cell_id"]))[0]
            anchor[fam] = meas[first["cell_id"]]["mean_power_w"]
    err_c, err_a = {}, {}
    for c in cells:
        m, t = meas[c]["energy_j"], T[c]["per_launch_runtime_s"]
        err_c[c] = abs(p_const * t / m - 1) * 100
        fam = cell_by[c]["family"]
        if fam in anchor:
            err_a[c] = abs(anchor[fam] * t / m - 1) * 100
    sub = lambda d, f: [v for c, v in d.items() if f(c)]
    return dict(rule="reference controls computed from the same measurements; not published methods, not competitors",
                constant_power_times_measured_runtime=dict(constant_power_w=round(p_const, 3), fit_windows=len(fit), below_cap=S.stats(sub(err_c, below)), all=S.stats(list(err_c.values()))),
                one_energy_anchor_times_measured_runtime=dict(anchor_power_w_by_family={k: round(v, 2) for k, v in anchor.items()}, below_cap=S.stats(sub(err_a, below)), all=S.stats(list(err_a.values()))))


def merge_profiles(preds, timings):
    base = json.loads(json.dumps(preds[0]))
    for p in preds[1:]:
        if p["calibration"]["sha256"] != base["calibration"]["sha256"]:
            raise Refusal("the profiles were frozen from different calibration documents")
        base["cells"].update(p["cells"])
        base["roofline_runtime_s"].update(p["roofline_runtime_s"])
    tm = dict(cells=[], gate_refused_cells=[], engines={}, meta=(timings[0] or {}).get("meta") if timings and timings[0] else None)
    for t in timings:
        if t:
            tm["cells"] += t["cells"]
            tm["gate_refused_cells"] += t.get("gate_refused_cells", [])
            for k, v in (t.get("engines") or {}).items():
                tm["engines"][(t.get("meta") or {}).get("profile", "") + ":" + k] = v
    return base, tm


def read_energy_board(board, dirs):
    """Energy rows keyed by the board's own cell ids. score_h100.read_energy (reused unchanged for everything else) hard-codes the prefix "h100/", so on any other board every energy row failed to
    match its cell and the scorer reported zero energy cells (found on the A100 run of 7 October 2026, before any energy number was read). Same fields and the same refusal on a duplicate row."""
    import csv
    rows = {}
    for d in dirs:
        p = Path(d) / "application_energy_raw.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                cid = "%s/%s/%s/%s" % (board, r["parent_id"], r["regime"], r["candidate_id"])
                if cid in rows:
                    raise Refusal("two energy rows for %s" % cid)
                total, interval = float(r["board_energy_j_total"]), float(r["counted_launch_interval_s"])
                rows[cid] = dict(energy_j=float(r["board_energy_j_per_launch"]), mean_power_w=total / interval, launches=int(r["launch_count"]), window_s=interval)
    return rows


def score(board, timing_main, timing_tensor, energy_dirs, tree=None):
    profiles = {"main": timing_main, "tensor": timing_tensor}
    preds, recs, timings, cells_all = [], {}, [], {}
    for prof, tpath in profiles.items():
        pred, rec = load_frozen(board, prof, tree)
        preds.append(pred)
        recs[prof] = rec
        t = None
        if tpath:
            t = json.loads(Path(tpath).read_text(encoding="utf-8"))
            check_timing(t, rec, prof)
        timings.append(t)
        for c in CS.load_cells(board, prof):
            cells_all[c["cell_id"]] = c
        if set(pred["cells"]) != {c["cell_id"] for c in CS.load_cells(board, prof)}:
            raise Refusal("the %s predictions do not cover exactly the cells of the profile" % prof)
    pred, timing = merge_profiles(preds, timings)
    pred["_cells_by_id"] = cells_all
    cell_ids = sorted(cells_all)
    not_measured_profiles = [p for p, t in zip(profiles, timings) if t is None]
    cells_measured_scope = [c for c in cell_ids if not any(cells_all[c] in CS.load_cells(board, p) for p in not_measured_profiles)] if not_measured_profiles else cell_ids
    rt, err, sgn = S.runtime_scores(pred, timing, cells_measured_scope)
    rt["per_set"] = S.group_stats(err, cells_all, lambda c: c["set"])
    rt["per_family"] = S.group_stats(err, cells_all, lambda c: c["family"])
    rt["per_tier"] = S.group_stats(err, cells_all, lambda c: c["tier"])
    rt["per_profile"] = {p: S.stats([e for c, e in err.items() if CS.needs_tensor(cells_all[c]) == (p == "tensor")]) for p in profiles}
    cal_uuid = pred["calibration"].get("uuid")
    meas_uuid = (((timing.get("meta") or {}).get("gpu")) or {}).get("uuid")
    same = None if not (cal_uuid and meas_uuid) else cal_uuid == meas_uuid
    rep = dict(schema="cluster_score/1", board=board, scope=dict(cells=len(cell_ids), profiles_not_measured=not_measured_profiles),
               inputs=dict(predictions_sha256={p: recs[p]["predictions_sha256"] for p in profiles}, freeze_commit={p: recs[p]["freeze_commit"] for p in profiles},
                           timing_sha256={p: (sha(profiles[p]) if profiles[p] else None) for p in profiles}, score_board_py_sha256=sha(__file__)), runtime=rt,
               gpu=dict(calibration_gpu_uuid=cal_uuid, measurement_gpu_uuid=meas_uuid, same_gpu=same,
                        statement=("calibration and measurement ran on the same GPU" if same else
                                   "calibration and measurement ran on DIFFERENT GPUs of the same model: every error below includes GPU-to-GPU variation, and the paper says so" if same is False else
                                   "a GPU UUID is missing: cannot say whether the two runs used the same GPU")),
               calibration=dict(source=pred["calibration"].get("source"), file=pred["calibration"].get("file"), sha256=pred["calibration"].get("sha256"), complete=pred["calibration"].get("complete")))
    r1 = bool(rt["supported_cells"] and rt["supported_cells"]["median_pct"] is not None and rt["supported_cells"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT)
    r2 = bool(rt["unsupported_as_failure"] and rt["unsupported_as_failure"]["median_pct"] is not None and rt["unsupported_as_failure"]["median_pct"] <= S.RUNTIME_MEDIAN_MAX_PCT)
    r3 = bool(rt["roofline_on_supported"] and rt["supported_cells"]) and rt["supported_cells"]["median_pct"] < rt["roofline_on_supported"]["median_pct"]
    criteria = dict(runtime_median_le_15_supported=r1, runtime_median_le_15_unsupported_as_failure=r2, runtime_beats_roofline_median=bool(r3))
    if energy_dirs:
        meas = read_energy_board(board, energy_dirs)
        en = S.energy_scores(pred, meas, timing, cell_ids, cells_all, None)
        en.pop("published_methods", None)
        en["published_methods"] = "not run on this board: recorded decision of 4 October 2026; the controls below are reference controls computed from the same measurements"
        T = {r["cell_id"]: r for r in timing["cells"] if r.get("correct")}
        cap = pred["calibration"]["energy_cap_w"]
        labelled = [c for c in cell_ids if c in meas and c in T]
        cells = [c for c in labelled if meas[c]["mean_power_w"] <= S.ADMIT_FRACTION_OF_CAP * cap]
        below = lambda c: meas[c]["mean_power_w"] < S.BELOW_CAP_FRACTION * cap
        doc = json.loads(find_calibration_document(pred).read_text(encoding="utf-8"))
        en["controls"] = controls(pred, doc, meas, T, cells, below, cap)
        rep["energy"] = en
        criteria.update(en["criteria"])
        rep["decision_utility"] = S.decision_utility(pred, meas, timing, cells_all)
        rep["cost"] = S.cost(dict(pred, calibration=dict(pred["calibration"], board_seconds_recorded=round(sum(float(st.get("seconds") or 0) for st in doc.get("stages", [])), 2),
                                                          stages_without_recorded_seconds=[st.get("name") or st.get("program") for st in doc.get("stages", []) if st.get("seconds") is None])), timing, energy_dirs)
    rep["criteria"] = criteria
    rep["stop_rule"] = "If a criterion is false, the stop rule applies: report as measured, no re-tuning on these cells; later model changes are retrospective and labelled so."
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True, choices=sorted(BD.CONFIG))
    ap.add_argument("--timing-main", type=Path, default=None)
    ap.add_argument("--timing-tensor", type=Path, default=None)
    ap.add_argument("--energy-raw", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tree", type=Path, default=None, help="directory holding the frozen files (default: the board directory)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise Refusal("%s exists; scored once, never overwritten" % a.out)
    if not (a.timing_main or a.timing_tensor):
        raise Refusal("at least one timing result is required")
    rep = score(a.board, a.timing_main, a.timing_tensor, a.energy_raw, a.tree)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep["criteria"], indent=1))
    print("runtime supported:", rep["runtime"]["supported_cells"], "| unsupported as failure:", rep["runtime"]["unsupported_as_failure"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
