#!/usr/bin/env python3
"""Score the frozen unseen-kernel predictions once, against measured runtime and energy (Blackwell). CPU only.

Frozen with the predictions, before any measurement. Definitions (they govern anything the prose leaves open):
  * Measured runtime t_meas of a cell = median over the timing windows of cuda_seconds / launches (run_unseen.py time).
  * Measured energy E_meas = board_energy_j_per_launch of the harness row (status raw) of the energy run; the energy
    window's own per-launch time t_win = counted_launch_interval_s / launch_count gives the measured power E_meas / t_win.
  * Below-cap cell: E_meas / t_win < 0.95 * cap (cap 600 W), as for the development cells.
  * A cell without a supported prediction, a correct timing, or an accepted energy row counts as a failure (infinite error)
    in every "unsupported as failure" statistic; it is never dropped. Statistics without that suffix use the cells that
    have both a prediction and a measurement and report how many that is.
  * Error = |predicted / measured - 1| * 100.
Criteria (PREREGISTRATION_UNSEEN_KERNELS.md):
  1 runtime : median error of the static runtime model <= 15 and below the roofline's median on the same cells
  2 energy  : on below-cap cells, median error of the decomposed energy model at predicted runtime is within 5 percentage
              points of the same model at measured runtime
  3 compute : on below-cap cells, that model's median error is below the runtime-plus-traffic formula's at predicted runtime
Run once:  python3 score_unseen.py --timing <timing json> --energy-csv <application_energy_raw.csv> --out <result json>
Self-test: python3 score_unseen.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
sys.path.insert(0, str(HERE))
import energy_model as EM  # noqa: E402

CAP_FRACTION = 0.95


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(pred, meas):
    return abs(pred / meas - 1) * 100


def stats(values):
    """median / 90th percentile; infinite values (failures) sort last, a percentile that lands on one is reported as null."""
    v = sorted(values)
    if not v:
        return None
    def q(p):
        pos = (len(v) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        if not math.isfinite(v[hi]):
            return None
        return round(float(v[lo] + (v[hi] - v[lo]) * (pos - lo)), 2)
    return dict(n=len(v), median_pct=q(.5), p90_pct=q(.9), failures=int(sum(not math.isfinite(x) for x in v)))


def load_inputs():
    cells = {c["cell_id"]: c for c in json.loads((FROZEN / "cells_unseen.json").read_text())["cells"]}
    pred = {m: json.loads((FROZEN / ("predictions_unseen_%s.json" % m)).read_text()) for m in ("v2", "v3d", "v3e")}
    en = json.loads((FROZEN / "energy_predictions_unseen.json").read_text())
    base = json.loads((FROZEN / "baseline_inputs_unseen.json").read_text())
    return cells, pred, en, base


def score(timing, energy_rows, out_inputs=None):
    cells, pred, en, base = load_inputs()
    model = en["model"]
    t_meas, t_ok = {}, {}
    for c in timing["cells"]:
        if c.get("per_launch_runtime_s") and c.get("correct"):
            t_meas[c["cell_id"]] = float(c["per_launch_runtime_s"])
    E_meas, t_win = {}, {}
    for r in energy_rows:
        cid = r["cell_id"]
        E_meas[cid] = float(r["board_energy_j_per_launch"])
        t_win[cid] = float(r["counted_launch_interval_s"]) / float(r["launch_count"])
    rep = dict(schema="unseen_score/1", cells_total=len(cells), cells_timed=len(t_meas), cells_energy=len(E_meas), models={}, energy={})
    ids = sorted(cells)
    fam = lambda cid: cells[cid]["operator_id"].replace("unseen_cuda_samples_", "")

    # ---------------- runtime
    def runtime_stats(getp, name):
        err = {}
        for cid in ids:
            p = getp(cid)
            err[cid] = pct(p, t_meas[cid]) if (p and cid in t_meas) else math.inf
        both = [cid for cid in ids if math.isfinite(err[cid])]
        sgn = [(getp(c) / t_meas[c] - 1) * 100 for c in both]
        grp = lambda f: stats([err[c] for c in ids if f(c) and math.isfinite(err[c])])
        return dict(name=name, supported_and_measured=len(both),
                    error=stats([err[c] for c in both]), error_unsupported_as_failure=stats(list(err.values())),
                    signed_median_pct=round(float(np.median(sgn)), 2) if sgn else None,
                    per_kernel={f: grp(lambda c, f=f: fam(c) == f) for f in sorted({fam(c) for c in ids})},
                    per_tier={t: grp(lambda c, t=t: cells[c]["tier"] == t) for t in ("L2", "DRAM")},
                    per_regime={r: grp(lambda c, r=r: cells[c]["regime"] == r) for r in ("small", "medium", "large", "xlarge")},
                    per_candidate={k: grp(lambda c, k=k: cells[c]["candidate_id"] == k) for k in ("c1", "c2")},
                    per_cell={c: dict(measured_us=round(t_meas[c] * 1e6, 3) if c in t_meas else None,
                                      predicted_us=round(getp(c) * 1e6, 3) if getp(c) else None,
                                      signed_error_pct=round((getp(c) / t_meas[c] - 1) * 100, 1) if (getp(c) and c in t_meas) else None) for c in ids}), err
    errs = {}
    for m in ("v2", "v3d", "v3e"):
        rep["models"]["static_runtime_" + m], errs[m] = runtime_stats(lambda c, m=m: pred[m][c].get("primary_s"), "static runtime model " + m)
    rep["models"]["roofline"], errs["roofline"] = runtime_stats(lambda c: base["roofline"][c].get("roofline_s"), "memory-plus-compute roofline")
    shared = [c for c in ids if math.isfinite(errs["v3e"][c]) and math.isfinite(errs["roofline"][c])]
    med = lambda e: float(np.median([e[c] for c in shared])) if shared else None
    r1 = rep["models"]["static_runtime_v3e"]["error"]
    rep["criterion_1_runtime"] = dict(v3e_median_pct=r1 and r1["median_pct"], roofline_median_same_cells=med(errs["roofline"]), v3e_median_same_cells=med(errs["v3e"]),
                                      shared_cells=len(shared),
                                      pass_=bool(r1 and r1["median_pct"] is not None and r1["median_pct"] <= 15 and shared and med(errs["v3e"]) < med(errs["roofline"])))

    # ---------------- energy
    P0, cap = model["P0_w"], model["cap_w"]
    eps = base["runtime_plus_traffic_eps_pJ_per_byte"]
    below = [c for c in ids if c in E_meas and E_meas[c] / t_win[c] < CAP_FRACTION * cap]
    typical = base["typical_power_w"]
    anchors = {}
    for c in ids:
        if cells[c]["regime"] == "medium" and cells[c]["candidate_id"] == "c1" and c in E_meas:
            anchors[fam(c)] = E_meas[c] / t_win[c]
    def e_dec(cid, t):
        return EM.predict(en["cells"][cid]["terms"], t, model)["energy_j"]
    def e_rpt(cid, t):
        nb = cells[cid]["logical_bytes_per_launch"]
        return min(P0 * t + eps[cells[cid]["tier"]] * 1e-12 * nb, cap * t)
    def energy_stats(cset, label):
        out = {}
        ok = [c for c in cset if en["cells"][c].get("status") == "ok" and c in t_meas]
        def S(fn, restrict=None):
            cs = [c for c in (restrict or ok)]
            return stats([pct(fn(c), E_meas[c]) for c in cs])
        out["cells"] = len(ok)
        out["decomposed_at_v3e_runtime"] = S(lambda c: en["cells"][c]["with_v3e_runtime"]["energy_j"])
        out["decomposed_at_measured_runtime"] = S(lambda c: e_dec(c, t_meas[c]))
        out["runtime_plus_traffic_at_v3e_runtime"] = S(lambda c: e_rpt(c, pred["v3e"][c]["primary_s"]))
        out["runtime_plus_traffic_at_measured_runtime"] = S(lambda c: e_rpt(c, t_meas[c]))
        out["baseline_time_the_kernel_x_typical_power"] = S(lambda c: typical * t_meas[c])
        anch = [c for c in ok if fam(c) in anchors and not (cells[c]["regime"] == "medium" and cells[c]["candidate_id"] == "c1")]
        out["baseline_time_the_kernel_x_anchor_power"] = S(lambda c: anchors[fam(c)] * t_meas[c], anch)
        out["per_kernel_decomposed_at_v3e_runtime"] = {f: stats([pct(en["cells"][c]["with_v3e_runtime"]["energy_j"], E_meas[c]) for c in ok if fam(c) == f]) for f in sorted({fam(c) for c in ok})}
        out["per_tier_decomposed_at_v3e_runtime"] = {t: stats([pct(en["cells"][c]["with_v3e_runtime"]["energy_j"], E_meas[c]) for c in ok if cells[c]["tier"] == t]) for t in ("L2", "DRAM")}
        out["per_cell"] = {c: dict(measured_j=E_meas[c], measured_power_w=round(E_meas[c] / t_win[c], 1), predicted_j=en["cells"][c]["with_v3e_runtime"]["energy_j"],
                                   predicted_terms_j={k: en["cells"][c]["with_v3e_runtime"][k] for k in ("runtime_term_j", "memory_term_j", "compute_term_j", "sfu_term_j")},
                                   signed_error_pct=round((en["cells"][c]["with_v3e_runtime"]["energy_j"] / E_meas[c] - 1) * 100, 1)) for c in ok}
        return out
    rep["energy"]["below_cap"] = energy_stats(below, "below_cap")
    rep["energy"]["all_measured_cells"] = energy_stats([c for c in ids if c in E_meas], "all")
    rep["energy"]["capped_cells"] = [c for c in ids if c in E_meas and c not in below]
    rep["energy"]["measured_power_w"] = {c: round(E_meas[c] / t_win[c], 1) for c in ids if c in E_meas}
    rep["runtime_consistency_timing_vs_energy_window_pct"] = stats([pct(t_win[c], t_meas[c]) for c in ids if c in t_win and c in t_meas])
    b = rep["energy"]["below_cap"]
    d_pred, d_meas, rpt = b["decomposed_at_v3e_runtime"], b["decomposed_at_measured_runtime"], b["runtime_plus_traffic_at_v3e_runtime"]
    rep["criterion_2_energy"] = dict(median_at_v3e=d_pred and d_pred["median_pct"], median_at_measured=d_meas and d_meas["median_pct"],
                                     margin_pp=None if not (d_pred and d_meas and d_pred["median_pct"] is not None and d_meas["median_pct"] is not None) else round(d_pred["median_pct"] - d_meas["median_pct"], 2),
                                     pass_=bool(d_pred and d_meas and d_pred["median_pct"] is not None and d_meas["median_pct"] is not None and d_pred["median_pct"] - d_meas["median_pct"] <= 5))
    rep["criterion_3_compute_term"] = dict(decomposed_median=d_pred and d_pred["median_pct"], runtime_plus_traffic_median=rpt and rpt["median_pct"],
                                           pass_=bool(d_pred and rpt and d_pred["median_pct"] is not None and rpt["median_pct"] is not None and d_pred["median_pct"] < rpt["median_pct"]))
    rep["verdict"] = {k: rep[k]["pass_"] for k in ("criterion_1_runtime", "criterion_2_energy", "criterion_3_compute_term")}
    rep["note"] = "Estimation accuracy only. Energy saved by a decision and measurement cost saved are not established by these criteria."
    return rep


def load_energy_csv(path, run_prefix=None):
    rows = []
    for r in csv.DictReader(Path(path).open(newline="")):
        if run_prefix and not r["run_id"].startswith(run_prefix):
            continue
        fam_ = {"matmul": "matrix_multiply", "bs": "black_scholes", "scan": "scan", "conv": "separable_convolution"}
        # parent_id is the cell's operator_id; the cell id is blackwell/<operator_id>/<regime>/<candidate>
        r["cell_id"] = "blackwell/%s/%s/%s" % (r["parent_id"], r["regime"], r["candidate_id"])
        rows.append(r)
    return rows


def selftest():
    cells, pred, en, base = load_inputs()
    t = {c: pred["v3e"][c]["primary_s"] * 1.0 for c in cells}
    timing = dict(cells=[dict(cell_id=c, per_launch_runtime_s=t[c], correct=True) for c in cells])
    rows = []
    for c in cells:
        e = en["cells"][c]["with_v3e_runtime"]["energy_j"]
        rows.append(dict(cell_id=c, board_energy_j_per_launch=e, counted_launch_interval_s=t[c] * 1000, launch_count=1000))
    rep = score(timing, rows)
    assert rep["models"]["static_runtime_v3e"]["error"]["median_pct"] == 0.0
    assert rep["energy"]["below_cap"]["decomposed_at_v3e_runtime"]["median_pct"] == 0.0
    assert rep["criterion_2_energy"]["margin_pp"] == 0.0 and rep["criterion_2_energy"]["pass_"]
    # a 2x slower measured runtime must show up as 50% error and fail criterion 1
    timing2 = dict(cells=[dict(cell_id=c, per_launch_runtime_s=2 * t[c], correct=True) for c in cells])
    rep2 = score(timing2, rows)
    assert abs(rep2["models"]["static_runtime_v3e"]["error"]["median_pct"] - 50.0) < 1e-6 and not rep2["criterion_1_runtime"]["pass_"]
    # missing measurements are failures, not drops
    timing3 = dict(cells=timing["cells"][:16])
    rep3 = score(timing3, rows)
    assert rep3["models"]["static_runtime_v3e"]["error_unsupported_as_failure"]["failures"] == 16
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--energy-csv", type=Path)
    ap.add_argument("--run-prefix", default="UNSEEN-ENERGY")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.timing and a.energy_csv and a.out):
        ap.error("--timing, --energy-csv and --out are required")
    if a.out.exists():
        raise SystemExit("REFUSED: %s exists; the unseen-kernel score is written once" % a.out)
    timing = json.loads(a.timing.read_text())
    rep = score(timing, load_energy_csv(a.energy_csv, a.run_prefix))
    rep["inputs_sha256"] = {"timing": sha(a.timing), "energy_csv": sha(a.energy_csv), "score_unseen.py": sha(__file__),
                            "freeze": sha(FROZEN / "PREDICTION_FREEZE_UNSEEN.json")}
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rep[k] for k in ("verdict", "criterion_1_runtime", "criterion_2_energy", "criterion_3_compute_term")}, indent=1))


if __name__ == "__main__":
    main()
