#!/usr/bin/env python3
"""Score the frozen H100 predictions against the measurements, once. Reads ONLY the committed predictions (checked against freeze_record_h100.json) and the measurement files.

    python score_h100.py --timing <timing_h100_result.json> [--energy-raw <dir with application_energy_raw.csv> ...] [--baselines baselines_h100.json] --out score_h100.json

Refuses unless predictions_h100.json is an H100 frozen-predictions file whose sha256 equals the one in freeze_record_h100.json AND the timing result was taken against those
predictions (timing meta.freeze.predictions_sha256 equal). Criteria, cell exclusions and the stop rule are those of FREEZE.md; the numbers below are the same as the Blackwell sets'.

Runtime (a cell whose correctness check failed in the measurement is excluded and listed; a cell the static side could not handle is a FAILURE with infinite error):
  supported_cells        median and 90th percentile of |predicted/measured - 1| over cells with a prediction
  unsupported_as_failure the same over all measured-correct cells with every unsupported cell counted as an infinite error (a percentile that reaches an infinite error is reported null)
Energy (per cell: board energy per launch of the counted window): ours at the static runtime, ours at the measured runtime (the same terms re-evaluated at the measured runtime,
  min(cap x t, base x t + sum of terms)); below cap (mean window power < 0.95 x the enforced limit of the calibration) and all cells; published-method rows only when a frozen
  baselines file is given (baselines_h100.json, committed before measurement); unsupported or unmeasured cells are failures in the "as failure" variants.
Decisions and cost: see decision_utility() and cost().
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNTIME_MEDIAN_MAX_PCT = 15.0
ENERGY_MEDIAN_MARGIN_PP = 5.0
ENERGY_P90_MARGIN_PP = 10.0
BELOW_CAP_FRACTION = 0.95
ADMIT_FRACTION_OF_CAP = 0.985      # the calibrator's own admission rule: a window above 98.5% of the enforced limit is not admitted


class Refusal(SystemExit):
    def __init__(self, m):
        super().__init__("REFUSED: " + m)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def percentile(values, q):
    v = sorted(values)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if not math.isfinite(v[hi]):
        return None
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def stats(values):
    v = list(values)
    if not v:
        return None
    r = lambda x: None if x is None else round(x, 2)
    return dict(n=len(v), median_pct=r(percentile(v, 0.5)), p90_pct=r(percentile(v, 0.9)), failures=sum(1 for x in v if not math.isfinite(x)))


def energy_at(row, runtime_s, cap_w):
    """Ours re-evaluated at another runtime: min(cap x t, base power x t + sum of the instruction and traffic terms) (the terms do not depend on t)."""
    return min(cap_w * runtime_s, row["base_power_w"] * runtime_s + sum(row["term_j"].values()))


def load_frozen(pred_path, record_path):
    pred_path, record_path = Path(pred_path), Path(record_path)
    doc = json.loads(pred_path.read_text(encoding="utf-8"))
    if doc.get("kind") != "h100_frozen_predictions" or doc.get("warning"):
        raise Refusal("%s is not an H100 frozen-predictions file" % pred_path.name)
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
                cid = "h100/%s/%s/%s" % (r["parent_id"], r["regime"], r["candidate_id"])
                if cid in rows:
                    raise Refusal("two energy rows for %s" % cid)
                total, interval = float(r["board_energy_j_total"]), float(r["counted_launch_interval_s"])
                rows[cid] = dict(energy_j=float(r["board_energy_j_per_launch"]), mean_power_w=total / interval, launches=int(r["launch_count"]), window_s=interval)
    return rows


def runtime_scores(pred, timing, cells):
    P, T = pred["cells"], {r["cell_id"]: r for r in timing["cells"]}
    excluded = sorted(c for c, r in T.items() if not r.get("correct"))
    gate_refused = [x["cell_id"] for x in timing.get("gate_refused_cells", [])]
    scored = [c for c in cells if c in T and T[c].get("correct")]
    not_measured = [c for c in cells if c not in T and c not in gate_refused]
    err, sgn, per = {}, {}, {}
    for c in scored:
        t = T[c]["per_launch_runtime_s"]
        if P[c]["status"] == "predicted":
            e = (P[c]["runtime_s"] / t - 1) * 100
            err[c], sgn[c] = abs(e), e
            per[c] = dict(measured_s=t, predicted_s=P[c]["runtime_s"], error_pct=round(e, 2))
        else:
            err[c] = math.inf
            per[c] = dict(measured_s=t, predicted_s=None, error_pct=None, failure=P[c].get("reason"))
    roof = {c: abs(pred["roofline_runtime_s"][c] / T[c]["per_launch_runtime_s"] - 1) * 100 for c in scored if c in pred.get("roofline_runtime_s", {})}
    sup = [c for c in scored if P[c]["status"] == "predicted"]
    out = dict(cells_in_list=len(cells), measured_correct=len(scored), excluded_correctness_failed=excluded, refused_by_sass_gate=gate_refused, not_measured=not_measured,
               supported_cells=stats([err[c] for c in sup]), unsupported_as_failure=stats([err[c] for c in scored]),
               signed_median_pct=None if not sgn else round(percentile(list(sgn.values()), 0.5), 2),
               roofline_on_supported=stats([roof[c] for c in sup if c in roof]), per_cell=per)
    return out, err, sgn


def group_stats(err, cells_by_id, key):
    groups = {}
    for c, e in err.items():
        groups.setdefault(key(cells_by_id[c]), []).append(e)
    return {k: stats(v) for k, v in sorted(groups.items())}


def score(pred_path, record_path, timing_path, energy_dirs=(), baselines_path=None, cells_path=None, profile="main"):
    pred, rec = load_frozen(pred_path, record_path)
    timing = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    tsha = (((timing.get("meta") or {}).get("freeze")) or {}).get("predictions_sha256")
    if tsha != rec["predictions_sha256"]:
        raise Refusal("the timing result was not taken against the frozen predictions (its recorded predictions sha256 is %r)" % (tsha,))
    if cells_path:
        cells_list = json.loads(Path(cells_path).read_text(encoding="utf-8"))["cells"]
    else:
        import cell_sets as CS
        if (pred.get("profile") or "main") != profile:
            raise Refusal("the predictions are of profile %r, not %r" % (pred.get("profile") or "main", profile))
        cells_list, _ = CS.load_cells(profile)
    cells_by_id = {c["cell_id"]: c for c in cells_list}
    cell_ids = sorted(cells_by_id)
    if set(pred["cells"]) != set(cell_ids):
        raise Refusal("the predictions do not cover exactly the cells of the %s profile (cells_h100.json plus ml_sets/ for main, ml_sets/cells_tensor_h100.json for tensor)" % profile)
    rt, err, sgn = runtime_scores(pred, timing, cell_ids)
    rt["per_set"] = group_stats(err, cells_by_id, lambda c: c["set"])
    rt["per_family"] = group_stats(err, cells_by_id, lambda c: c["family"])
    rt["per_tier"] = group_stats(err, cells_by_id, lambda c: c["tier"])
    cal_uuid = (pred.get("calibration") or {}).get("uuid")
    meas_uuid = (((timing.get("meta") or {}).get("gpu")) or {}).get("uuid")
    same = None if not (cal_uuid and meas_uuid) else cal_uuid == meas_uuid
    rep = dict(schema="h100_score/1", inputs=dict(predictions_sha256=rec["predictions_sha256"], freeze_commit=rec["freeze_commit"], timing_sha256=sha(timing_path),
                                                   score_h100_py_sha256=sha(__file__)), runtime=rt,
               gpu=dict(calibration_gpu_uuid=cal_uuid, measurement_gpu_uuid=meas_uuid, same_gpu=same,
                        statement=("calibration and measurement ran on the same GPU" if same else
                                   "calibration and measurement ran on DIFFERENT GPUs of the same model: every error below includes GPU-to-GPU variation, and the paper says so" if same is False else
                                   "a GPU UUID is missing from the predictions or the timing result: cannot say whether the two runs used the same GPU")),
               calibration=dict(source=(pred.get("calibration") or {}).get("source"), file=(pred.get("calibration") or {}).get("file"), sha256=(pred.get("calibration") or {}).get("sha256"),
                                complete=(pred.get("calibration") or {}).get("complete"),
                                energy_windows_not_admitted=(pred.get("calibration") or {}).get("energy_windows_not_admitted")))
    r1 = rt["supported_cells"] and rt["supported_cells"]["median_pct"] is not None and rt["supported_cells"]["median_pct"] <= RUNTIME_MEDIAN_MAX_PCT
    r2 = rt["unsupported_as_failure"] and rt["unsupported_as_failure"]["median_pct"] is not None and rt["unsupported_as_failure"]["median_pct"] <= RUNTIME_MEDIAN_MAX_PCT
    r3 = bool(rt["roofline_on_supported"] and rt["supported_cells"]) and rt["supported_cells"]["median_pct"] < rt["roofline_on_supported"]["median_pct"]
    criteria = dict(runtime_median_le_15_supported=bool(r1), runtime_median_le_15_unsupported_as_failure=bool(r2), runtime_beats_roofline_median=bool(r3))
    if energy_dirs:
        en = energy_scores(pred, read_energy(energy_dirs), timing, cell_ids, cells_by_id, baselines_path)
        rep["energy"] = en
        criteria.update(en["criteria"])
        rep["decision_utility"] = decision_utility(pred, read_energy(energy_dirs), timing, cells_by_id)
        rep["cost"] = cost(pred, timing, energy_dirs)
    rep["criteria"] = criteria
    rep["stop_rule"] = "If a criterion is false, FREEZE.md's stop rule applies: report as measured, no re-tuning on these cells; later model changes are retrospective and labelled so."
    return rep


def energy_scores(pred, meas, timing, cell_ids, cells_by_id, baselines_path):
    P = pred["cells"]
    cap = pred["calibration"]["energy_cap_w"]
    T = {r["cell_id"]: r for r in timing["cells"] if r.get("correct")}
    labelled = [c for c in cell_ids if c in meas and c in T]
    # Cells measured above the admission rule (mean power above 98.5% of the enforced limit) are excluded from the energy scoring of our model AND of every published method alike; counted and listed.
    at_cap = [c for c in labelled if meas[c]["mean_power_w"] > ADMIT_FRACTION_OF_CAP * cap]
    cells = [c for c in labelled if c not in at_cap]
    below = lambda c: meas[c]["mean_power_w"] < BELOW_CAP_FRACTION * cap
    e_static, e_meas, rows = {}, {}, {}
    for c in cells:
        m = meas[c]["energy_j"]
        if P[c]["status"] == "predicted":
            e_static[c] = abs(P[c]["energy_j"] / m - 1) * 100
            e_meas[c] = abs(energy_at(P[c], T[c]["per_launch_runtime_s"], cap) / m - 1) * 100
        else:
            e_static[c] = e_meas[c] = math.inf
        rows[c] = dict(measured_j=m, mean_power_w=round(meas[c]["mean_power_w"], 1), below_cap=bool(below(c)),
                       ours_static_error_pct=None if not math.isfinite(e_static[c]) else round((P[c]["energy_j"] / m - 1) * 100, 2),
                       ours_measured_runtime_error_pct=None if not math.isfinite(e_meas[c]) else round((energy_at(P[c], T[c]["per_launch_runtime_s"], cap) / m - 1) * 100, 2))
    sub = lambda d, f: [v for c, v in d.items() if f(c)]
    sup = lambda c: P[c]["status"] == "predicted"
    out = dict(cells_with_energy=len(labelled), cells_scored=len(cells), excluded_above_the_cap_rule=dict(fraction_of_cap=ADMIT_FRACTION_OF_CAP, count=len(at_cap), cells=at_cap,
                                                                                                          note="excluded for every method alike; their measured energies are in the raw files"),
               cells_without_energy=[c for c in cell_ids if c not in meas], cap_w=cap, below_cap_fraction=BELOW_CAP_FRACTION,
               ours_static_runtime=dict(below_cap=stats(sub(e_static, lambda c: below(c) and sup(c))), all=stats(sub(e_static, sup)),
                                        below_cap_unsupported_as_failure=stats(sub(e_static, below)), all_unsupported_as_failure=stats(list(e_static.values()))),
               ours_measured_runtime=dict(below_cap=stats(sub(e_meas, lambda c: below(c) and sup(c))), all=stats(sub(e_meas, sup)),
                                          below_cap_unsupported_as_failure=stats(sub(e_meas, below)), all_unsupported_as_failure=stats(list(e_meas.values()))),
               per_cell=rows)
    sb, mb = out["ours_static_runtime"]["below_cap"], out["ours_measured_runtime"]["below_cap"]
    sf, mf = out["ours_static_runtime"]["below_cap_unsupported_as_failure"], out["ours_measured_runtime"]["below_cap_unsupported_as_failure"]
    ok = lambda a, b, pp, k: bool(a and b and a[k] is not None and b[k] is not None and a[k] <= b[k] + pp)
    out["criteria"] = dict(energy_median_within_5pp_of_measured_runtime_below_cap=ok(sb, mb, ENERGY_MEDIAN_MARGIN_PP, "median_pct"),
                           energy_p90_within_10pp_of_measured_runtime_below_cap=ok(sb, mb, ENERGY_P90_MARGIN_PP, "p90_pct"),
                           energy_median_within_5pp_unsupported_as_failure=ok(sf, mf, ENERGY_MEDIAN_MARGIN_PP, "median_pct"))
    if baselines_path:
        out["published_methods"] = published_rows(baselines_path, meas, T, cells, below, cap, pred)
    else:
        out["published_methods"] = "not scored: no frozen baselines_h100.json was given (published-method rows are required before the paper reports an energy comparison)"
    return out


def form_energy(form, runtime_s, cap_w):
    """Energy of a published-method row at a given runtime, from the frozen form written by baselines_h100.py."""
    kind = form["kind"]
    if kind == "constant_plus_variable":                    # component model: power = constant + variable terms / t, capped
        return min(cap_w * runtime_s, form["constant_power_w"] * runtime_s + form["variable_j"])
    if kind == "native_power":                               # static-feature power model: energy = native power x runtime
        return form["power_w"] * runtime_s
    raise Refusal("unknown baseline form %r" % kind)


def published_rows(path, meas, T, cells, below, cap, pred):
    """Frozen published-method rows (baselines_h100.json, committed before measurement). Each method that ran is scored with its own runtime input (the measured runtime) and with ours
    (our static prediction); a cell the method cannot be applied to, or for which we have no static runtime, is a failure. Methods recorded as not run are listed with their reason."""
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    P = pred["cells"]
    out, not_run = {}, {}
    for method, m in b["methods"].items():
        if m.get("status") != "run":
            not_run[method] = dict(status=m.get("status"), reason=m.get("reason"), blockers=m.get("blockers"))
            continue
        entry = dict(label=m.get("label"), route=m.get("route"))
        for variant, runtime_of in (("measured_runtime", lambda c: T[c]["per_launch_runtime_s"]), ("our_static_runtime", lambda c: P[c]["runtime_s"] if P[c]["status"] == "predicted" else None)):
            err = {}
            for c in cells:
                form, t = (m["cells"].get(c) or None), runtime_of(c)
                err[c] = math.inf if (form is None or t is None) else abs(form_energy(form, t, cap) / meas[c]["energy_j"] - 1) * 100
            entry[variant] = dict(below_cap=stats([e for c, e in err.items() if below(c)]), all_scored_cells=stats(list(err.values())),
                                  cells_where_ours_is_lower=None)
        out[method] = entry
    return dict(file=Path(path).name, sha256=sha(path), methods_run=out, methods_not_run=not_run)


def groups_of(cells_by_id):
    g = {}
    for c in cells_by_id.values():
        g.setdefault((c["operator_id"], c["regime"]), []).append(c["cell_id"])
    return {k: sorted(v) for k, v in g.items() if len(v) >= 2}


def decision_utility(pred, meas, timing, cells_by_id):
    """Choose the lower-energy candidate of each (operator, size) group (equal work, different implementation). Separate from accuracy: top-1 regret and energy saved.
    Compared: ours without execution; the faster candidate by measured runtime ('time the kernel'); the first candidate (c1); and an anchor-power rule ('one energy anchor':
    each candidate's energy = the anchor power of its operator x its measured runtime, which can only choose the faster one when the anchor power is shared)."""
    P = pred["cells"]
    T = {r["cell_id"]: r["per_launch_runtime_s"] for r in timing["cells"] if r.get("correct")}
    out = dict(groups=0, rows=[])
    totals = dict(ours=[], fastest=[], first=[])
    for (op, regime), ids in sorted(groups_of(cells_by_id).items()):
        ids = [c for c in ids if c in meas and c in T]
        if len(ids) < 2:
            continue
        best = min(ids, key=lambda c: meas[c]["energy_j"])
        pick_ours = min((c for c in ids if P[c]["status"] == "predicted"), key=lambda c: P[c]["energy_j"], default=None)
        pick_fast = min(ids, key=lambda c: T[c])
        first = sorted(ids)[0]
        reg = lambda c: None if c is None else round((meas[c]["energy_j"] / meas[best]["energy_j"] - 1) * 100, 3)
        out["rows"].append(dict(operator=op, regime=regime, candidates=len(ids), best=best, ours=pick_ours, ours_regret_pct=reg(pick_ours), fastest=pick_fast, fastest_regret_pct=reg(pick_fast),
                                first_candidate_regret_pct=reg(first), ours_correct=pick_ours == best, fastest_correct=pick_fast == best,
                                measured_energy_gap_pct=round((max(meas[c]["energy_j"] for c in ids) / meas[best]["energy_j"] - 1) * 100, 3)))
    rows = out["rows"]
    out["groups"] = len(rows)
    if rows:
        out["ours_top1_correct"] = sum(r["ours_correct"] for r in rows)
        out["fastest_top1_correct"] = sum(r["fastest_correct"] for r in rows)
        out["ours_median_regret_pct"] = round(percentile([r["ours_regret_pct"] for r in rows if r["ours_regret_pct"] is not None], 0.5), 3)
        out["fastest_median_regret_pct"] = round(percentile([r["fastest_regret_pct"] for r in rows], 0.5), 3)
        out["median_measured_energy_gap_pct"] = round(percentile([r["measured_energy_gap_pct"] for r in rows], 0.5), 3)
    out["note"] = "Energy saved by a decision is a different claim from estimation accuracy and from measurement cost saved; none establishes another."
    return out


def cost(pred, timing, energy_dirs):
    """Measurement cost, counted separately from accuracy: one-time calibration board time (from the calibration document's stages, when recorded in the predictions), the median service
    time of one valid energy window as measured in this run (successive log entries), and the break-even number of cells."""
    import datetime
    stamps = []
    for d in energy_dirs:
        for p in Path(d).glob("*energy*log.jsonl"):
            for ln in p.read_text().splitlines():
                try:
                    r = json.loads(ln)
                    stamps.append((datetime.datetime.strptime(r.get("utc") or r.get("logged_utc"), "%Y-%m-%dT%H:%M:%SZ").timestamp(), r.get("status")))
                except (ValueError, TypeError):
                    continue
    stamps.sort()
    gaps = [b[0] - a[0] for a, b in zip(stamps, stamps[1:]) if b[1] == "raw"]
    service = percentile(gaps, 0.5) if gaps else None
    cal = pred.get("calibration") or {}
    cal_s = cal.get("board_seconds_recorded") if not cal.get("stages_without_recorded_seconds") else None     # only when every stage recorded its time
    out = dict(calibration_stages_without_recorded_seconds=cal.get("stages_without_recorded_seconds"), calibration_board_seconds_recorded=cal.get("board_seconds_recorded"), energy_window_service_time_s_median=None if service is None else round(service, 1), valid_windows_timed=len(gaps), one_time_calibration_board_s=cal_s,
               timing_stage_wall_s={k: v.get("wall_s") for k, v in (timing.get("engines") or {}).items()})
    if service and cal_s:
        out["break_even_cells"] = round(cal_s / service, 1)
        out["meaning"] = "calibration board time divided by the service time of one measured window: beyond this many cells, predicting is cheaper in board time than measuring each cell. Static analysis uses no board time."
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=("main", "tensor"), default="main", help="main: 72 CUDA-samples + 40 machine-learning-kernel cells; tensor: the 16 tensor-core matrix multiply and fused attention cells")
    ap.add_argument("--predictions", type=Path, default=None); ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--timing", type=Path, required=True); ap.add_argument("--energy-raw", type=Path, nargs="*", default=[]); ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise Refusal("%s exists; scored once, never overwritten" % a.out)
    import cell_sets as CS
    pred_path = a.predictions or HERE / CS.profile(a.profile)["predictions"]
    record_path = a.record or HERE / CS.profile(a.profile)["record"]
    rep = score(pred_path, record_path, a.timing, a.energy_raw, a.baselines, profile=a.profile)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep["criteria"], indent=1))
    print("runtime supported:", rep["runtime"]["supported_cells"], "unsupported as failure:", rep["runtime"]["unsupported_as_failure"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
