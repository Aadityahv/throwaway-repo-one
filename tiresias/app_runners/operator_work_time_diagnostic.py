#!/usr/bin/env python3
"""WP-6/WP-7 operator work + time model (H5/H6), per OPERATOR_WORK_TIME_PROTOCOL_2026-09-25.md.

CPU only. Constants are fitted on the Blackwell calibration suite only
(calibration_raw/CALIB-BLACKWELL-20260922T164827Z). Operator data are read only through
hybrid_probe_diagnostic.load_data() (the 7 nonsealed parents, same filters). No sealed path.

    python tiresias/app_runners/operator_work_time_diagnostic.py --fit-only   # calibration only
    python tiresias/app_runners/operator_work_time_diagnostic.py              # full scoring
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CAL_RUN = "CALIB-BLACKWELL-20260922T164827Z"
CAL_DIR = ROOT / "calibration_raw" / CAL_RUN
WORKLOAD_CELLS = ROOT / "workload_cells.csv"
OUT = ROOT / "operator_work_time_diagnostic_results.json"

# HARDWARE_GROUND_TRUTH.md, Blackwell GPU1 (the only authority for these two numbers).
P_CAP_W = 600.0
L2_BYTES = 134_217_728
FLOAT_BYTES = 4
CAP_EXCLUDE_FRAC = 0.95          # calibration windows with mean power >= 0.95 * P_cap leave the primary fit
CAPPED_SPLIT_FRAC = 0.95         # operator cells with measured mean power >= 0.95 * P_cap -> "near cap" group
TIE_2PCT = 2.0

PARENTS = ("train_triton_vector_add", "dev_triton_softmax", "final_cuda_samples_copy",
           "train_cuda_samples_transpose", "train_cuda_samples_reduction",
           "dev_pytorch_rowwise_softmax", "final_triton_layer_norm")
# Source-level global passes per parent (protocol section 3); read off the kernel source before scoring.
READ_PASSES = {"train_triton_vector_add": "A,B once; C once", "dev_triton_softmax": "X once; Y once",
               "final_cuda_samples_copy": "A,B once; C once", "train_cuda_samples_transpose": "in once; out once",
               "train_cuda_samples_reduction": "in once; one partial per block",
               "dev_pytorch_rowwise_softmax": "X once; Y once",
               "final_triton_layer_norm": "X three times; W,B once per row; Y once; mean,rstd once per row"}


def assert_allowed(path: Path) -> None:
    assert "sealed_labels" not in path.parts and not any(p.startswith("FINAL-") for p in path.parts), path


# ---------------------------------------------------------------- calibration
def fixture_bytes(fixture: str, c: dict) -> tuple[int, int]:
    """Identical to iteration4_fit_score.fixture_bytes (the suite's own logical byte rule, W4)."""
    if fixture == "triad":
        e = c["n"] * c["n"]
        return 3 * e * FLOAT_BYTES, e * FLOAT_BYTES
    if fixture == "transpose":
        e = c["n"] * c["n"]
        return e * FLOAT_BYTES, e * FLOAT_BYTES
    if fixture == "store":
        return 0, c["blocks"] * c["threads"] * c["batches"] * c["iterations"] * FLOAT_BYTES
    if fixture == "shared_stage":
        return (c["blocks"] * c["batches"] * c["chunks"] * c["stage_elements"] * FLOAT_BYTES,
                c["blocks"] * c["threads"] * FLOAT_BYTES)
    raise ValueError(fixture)


def load_calibration() -> list[dict]:
    assert_allowed(CAL_DIR)
    manifest = [json.loads(l) for l in (CAL_DIR / "manifest.jsonl").read_text().splitlines() if l.strip()]
    key = lambda d: (int(d["session"]), d["fixture"], d["regime"], d["candidate_id"])  # noqa: E731
    by_key = {key(m): m for m in manifest}
    assert len(by_key) == len(manifest) == 32
    rows = []
    with (CAL_DIR / "raw_records.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            m = by_key[key(r)]
            assert r["correctness_check"] == "True" and r["graph_batch"] == "1000", key(r)
            rb, wb = fixture_bytes(m["fixture"], m["controls"])
            t_s = float(r["counted_launch_interval_s"]) / int(r["launch_count"])
            power = float(r["board_energy_j_per_launch"]) / t_s
            tier = "DRAM" if m["candidate_tier"] == "above_l2" else "L2"   # below_l2 and tiny -> L2
            strided = m["controls"].get("layout") == "strided" and m["fixture"] == "store"
            if strided:
                excl = "strided store: logical bytes do not represent sectors moved"
            elif power >= CAP_EXCLUDE_FRAC * P_CAP_W:
                excl = f"mean power >= {CAP_EXCLUDE_FRAC} x P_cap"
            else:
                excl = None
            rows.append(dict(session=int(r["session"]), fixture=m["fixture"], regime=m["regime"],
                             candidate=m["candidate_id"], role=m["role"], tier=tier,
                             bytes=rb + wb, t_us=t_s * 1e6, power_w=power,
                             rate_gbps=(rb + wb) / t_s / 1e9, strided=strided, excluded=excl))
    assert len(rows) == 32
    return rows


def fit(rows: list[dict]) -> dict:
    X = np.array([[1.0, r["bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "L2" else 0.0,
                   r["bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "DRAM" else 0.0] for r in rows])
    y = np.array([r["power_w"] for r in rows])
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ theta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"P0_w": float(theta[0]), "eps_L2_pj_per_byte": float(theta[1]) * 1e12,
            "eps_DRAM_pj_per_byte": float(theta[2]) * 1e12, "n_windows": len(rows),
            "r2": 1 - ss_res / ss_tot, "rmse_w": (ss_res / len(rows)) ** 0.5}


def work_power(c: dict, tier: str, nbytes: float, t_us: float) -> float:
    eps = c["eps_L2_pj_per_byte"] if tier == "L2" else c["eps_DRAM_pj_per_byte"]
    return eps * 1e-12 * nbytes / (t_us * 1e-6)


def predict_power(c: dict, tier: str, nbytes: float, t_us: float) -> float:
    return min(P_CAP_W, c["P0_w"] + work_power(c, tier, nbytes, t_us))


def calibration_block() -> dict:
    rows = load_calibration()
    fit_rows = [r for r in rows if r["excluded"] is None]
    primary = fit(fit_rows)
    for r in rows:
        r["pred_w"] = predict_power(primary, r["tier"], r["bytes"], r["t_us"])
        r["resid_w"] = r["power_w"] - (primary["P0_w"] + work_power(primary, r["tier"], r["bytes"], r["t_us"]))
    # H6 delta: relative error of the work term on non-tiny primary-fit windows.
    rel = [abs(r["resid_w"]) / work_power(primary, r["tier"], r["bytes"], r["t_us"])
           for r in fit_rows if r["regime"] != "tiny"]
    sens_strided = fit([r for r in rows if not (r["excluded"] or "").startswith("mean power")])
    sens_cap = fit([r for r in rows if not r["strided"]])
    const_power = stats.mean(r["power_w"] for r in fit_rows)
    return {"primary": primary, "sensitivity_incl_strided": sens_strided,
            "sensitivity_incl_near_cap_triad": sens_cap,
            "calibration_mean_power_w": const_power,
            "h6_delta_median": stats.median(rel), "h6_delta_max": max(rel),
            "h6_delta_values": sorted(rel), "windows": rows}


# ---------------------------------------------------------------- operator design (no labels)
def footprint_bytes(parent: str, shape_a: int, shape_b: int, rb: int, wb: int) -> int:
    if parent in ("train_triton_vector_add", "final_cuda_samples_copy"):
        return 3 * shape_a * FLOAT_BYTES
    if parent in ("dev_triton_softmax", "dev_pytorch_rowwise_softmax", "train_cuda_samples_transpose"):
        return 2 * shape_a * shape_b * FLOAT_BYTES
    if parent == "final_triton_layer_norm":
        return (2 * shape_a * shape_b + 2 * shape_b + 2 * shape_a) * FLOAT_BYTES
    if parent == "train_cuda_samples_reduction":
        return rb + wb   # input once + one partial per block, all distinct
    raise ValueError(parent)


def operator_design() -> dict:
    out = {}
    with WORKLOAD_CELLS.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["parent_id"] not in PARENTS:
                continue
            rb, wb = int(r["global_read_bytes"]), int(r["global_write_bytes"])
            fp = footprint_bytes(r["parent_id"], int(r["shape_a"]), int(r["shape_b"]), rb, wb)
            out[(r["parent_id"], r["regime"], r["candidate_id"])] = {
                "bytes": rb + wb, "read_bytes": rb, "write_bytes": wb, "footprint_bytes": fp,
                "l2_ratio": fp / L2_BYTES, "tier": "L2" if fp / L2_BYTES < 1.0 else "DRAM"}
    assert len(out) == 84, len(out)
    return out


# ---------------------------------------------------------------- scoring
def quantile(v, q):
    from hybrid_probe_diagnostic import quantile as qf
    return qf(v, q)


def summarize(apes: list[float]) -> dict:
    return {"n": len(apes), "mdape_pct": stats.median(apes) if apes else None,
            "p90_ape_pct": quantile(apes, 0.9) if apes else None,
            "mean_ape_pct": stats.mean(apes) if apes else None}


def score(cal: dict, design: dict) -> dict:
    import hybrid_probe_diagnostic as hp
    grid, sessions = hp.load_data()
    hyb = hp.evaluate(grid)
    hyb_x = hp.evaluate(grid, sessions)
    c = cal["primary"]
    variants = {
        "work_time": lambda d, t: predict_power(c, d["tier"], d["bytes"], t),
        "no_work_P0": lambda d, t: min(P_CAP_W, c["P0_w"]),
        "no_work_calib_mean": lambda d, t: min(P_CAP_W, cal["calibration_mean_power_w"]),
        "work_time_sens_strided": lambda d, t: predict_power(cal["sensitivity_incl_strided"], d["tier"], d["bytes"], t),
        "work_time_sens_near_cap": lambda d, t: predict_power(cal["sensitivity_incl_near_cap_triad"], d["tier"], d["bytes"], t),
        "work_time_tier_0p5": lambda d, t: predict_power(c, "L2" if d["l2_ratio"] < 0.5 else "DRAM", d["bytes"], t),
    }
    cells = []
    for p in sorted(grid):
        for (regime, cand), v in sorted(grid[p].items()):
            d = design[(p, regime, cand)]
            meas_w = v["e"] / v["t"] * 1e6
            row = {"parent": p, "regime": regime, "candidate": cand, "t_us": v["t"], "e_true_j": v["e"],
                   "measured_power_w": meas_w, "tier": d["tier"], "l2_ratio": d["l2_ratio"],
                   "bytes": d["bytes"], "near_cap_measured": meas_w >= CAPPED_SPLIT_FRAC * P_CAP_W,
                   "pred": {}, "ape_pct": {}, "signed_pct": {}}
            for name, f in variants.items():
                pw = f(d, v["t"])
                e = pw * v["t"] * 1e-6
                row["pred"][name] = {"power_w": pw, "energy_j": e}
                row["ape_pct"][name] = abs(e / v["e"] - 1) * 100
                row["signed_pct"][name] = (e / v["e"] - 1) * 100
            row["predicted_capped"] = row["pred"]["work_time"]["power_w"] >= P_CAP_W - 1e-9
            cells.append(row)
    assert len(cells) == 84
    # Nulls on the same 63 unprobed targets (hybrid evaluate()).
    hyb_pred = {(r["parent"], r["regime"], r["candidate"]): r["ape_pct"] for r in _hyb_predictions(hp, grid)}
    targets = [r for r in cells if r["candidate"] != "c1"]
    assert len(targets) == 63 == len(hyb_pred)
    for r in targets:
        r["ape_pct"]["null_other_parent_mean_power"] = hyb_pred[(r["parent"], r["regime"], r["candidate"])]["training_mean_power"]
        r["ape_pct"]["null_c1_power"] = hyb_pred[(r["parent"], r["regime"], r["candidate"])]["anchor_constant_power"]
    names = list(variants)
    primary = {n: summarize([r["ape_pct"][n] for r in targets]) for n in names + ["null_other_parent_mean_power", "null_c1_power"]}
    primary["work_time"]["median_signed_pct"] = stats.median(r["signed_pct"]["work_time"] for r in targets)
    bar0 = hyb["energy_error"]["training_mean_power"]["mdape_pct"]
    bar1 = hyb["energy_error"]["anchor_constant_power"]["mdape_pct"]
    assert abs(primary["null_other_parent_mean_power"]["mdape_pct"] - bar0) < 1e-9
    assert round(bar0, 2) == 20.37 and round(bar1, 2) == 6.37, (bar0, bar1)
    all84 = {n: summarize([r["ape_pct"][n] for r in cells]) for n in names}
    splits = {}
    for label, pred in (("measured_near_cap", lambda r: r["near_cap_measured"]),
                        ("measured_below_cap", lambda r: not r["near_cap_measured"]),
                        ("predicted_capped", lambda r: r["predicted_capped"]),
                        ("predicted_uncapped", lambda r: not r["predicted_capped"]),
                        ("tier_L2", lambda r: r["tier"] == "L2"), ("tier_DRAM", lambda r: r["tier"] == "DRAM")):
        sel = [r for r in cells if pred(r)]
        splits[label] = {"all84": summarize([r["ape_pct"]["work_time"] for r in sel]),
                         "targets63": summarize([r["ape_pct"]["work_time"] for r in sel if r["candidate"] != "c1"]),
                         "median_signed_pct": stats.median(r["signed_pct"]["work_time"] for r in sel) if sel else None}
    splits["confusion_measured_vs_predicted_cap"] = {
        f"meas_{a}_pred_{b}": sum((r["near_cap_measured"] == a) and (r["predicted_capped"] == b) for r in cells)
        for a in (True, False) for b in (True, False)}
    per_group = []
    for p in sorted(grid):
        for regime in hp.REGIMES:
            sel = [r for r in cells if r["parent"] == p and r["regime"] == regime]
            per_group.append({"parent": p, "regime": regime, "tier": sel[0]["tier"], "l2_ratio": sel[0]["l2_ratio"],
                              "mdape_work_time_pct": stats.median(r["ape_pct"]["work_time"] for r in sel),
                              "max_ape_work_time_pct": max(r["ape_pct"]["work_time"] for r in sel),
                              "median_signed_work_time_pct": stats.median(r["signed_pct"]["work_time"] for r in sel),
                              "mdape_null_mean_power_c2c4_pct": stats.median(r["ape_pct"]["null_other_parent_mean_power"] for r in sel if r["candidate"] != "c1"),
                              "n_measured_near_cap": sum(r["near_cap_measured"] for r in sel),
                              "n_predicted_capped": sum(r["predicted_capped"] for r in sel),
                              "median_measured_power_w": stats.median(r["measured_power_w"] for r in sel),
                              "median_predicted_power_w": stats.median(r["pred"]["work_time"]["power_w"] for r in sel)})
    # Cross-session: session-1 runtime -> session-2 energy (4 parents, 48 cells; c2-c4 = 36).
    xs = []
    for p in sorted(sessions):
        for (regime, cand), v1 in sorted(sessions[p]["1"].items()):
            d = design[(p, regime, cand)]
            e2 = sessions[p]["2"][(regime, cand)]["e"]
            e_hat = predict_power(c, d["tier"], d["bytes"], v1["t"]) * v1["t"] * 1e-6
            xs.append({"parent": p, "regime": regime, "candidate": cand, "ape_pct": abs(e_hat / e2 - 1) * 100})
    cross = {"work_time_all": summarize([x["ape_pct"] for x in xs]),
             "work_time_c2c4": summarize([x["ape_pct"] for x in xs if x["candidate"] != "c1"]),
             "null_other_parent_mean_power_c2c4": {k: hyb_x["energy_error"]["training_mean_power"][k] for k in ("mdape_pct", "p90_ape_pct")},
             "null_c1_power_c2c4": {k: hyb_x["energy_error"]["anchor_constant_power"][k] for k in ("mdape_pct", "p90_ape_pct")}}
    h5 = primary["work_time"]["mdape_pct"]
    verdict = {"h5_pass": h5 < bar0, "h5_stretch": h5 < bar1, "bar_zero_probe_mdape": bar0,
               "bar_zero_probe_p90": hyb["energy_error"]["training_mean_power"]["p90_ape_pct"],
               "bar_one_probe_mdape": bar1, "bar_one_probe_p90": hyb["energy_error"]["anchor_constant_power"]["p90_ape_pct"]}
    return {"grid": grid, "cells": cells, "primary_63": primary, "all_84": all84, "splits": splits,
            "per_group": per_group, "cross_session": cross, "h5_verdict": verdict, "hybrid_groups": hyb["groups"]}


def _hyb_predictions(hp, grid):
    """Re-run hybrid evaluate() internals to obtain per-target APEs (same code path, same numbers)."""
    out = []
    for parent in sorted(grid):
        for regime in hp.REGIMES:
            truth = {c: grid[parent][(regime, c)]["e"] for c in hp.CANDIDATES}
            source = {c: grid[parent][(regime, c)] for c in hp.CANDIDATES}
            mean_power, _ = hp.training_priors(grid, parent, regime)
            anchor = source["c1"]["e"] / source["c1"]["t"]
            for c in hp.CANDIDATES[1:]:
                out.append({"parent": parent, "regime": regime, "candidate": c, "ape_pct": {
                    "training_mean_power": abs(source[c]["t"] * mean_power / truth[c] - 1) * 100,
                    "anchor_constant_power": abs(source[c]["t"] * anchor / truth[c] - 1) * 100}})
    return out


def h6(cal: dict, sc: dict, design: dict) -> dict:
    import hybrid_probe_diagnostic as hp
    c = cal["primary"]
    grid = sc["grid"]
    res = {}
    for dname in ("h6_delta_median", "h6_delta_max"):
        delta = cal[dname]
        groups = []
        for p in sorted(grid):
            for regime in hp.REGIMES:
                cells = {k: grid[p][(regime, k)] for k in hp.CANDIDATES}
                rank = sorted(hp.CANDIDATES, key=lambda k: (cells[k]["t"], k))
                margin = (cells[rank[1]]["t"] / cells[rank[0]]["t"] - 1) * 100
                d = design[(p, regime, rank[0])]
                pw = work_power(c, d["tier"], d["bytes"], cells[rank[0]]["t"])
                tau = delta * min(pw, P_CAP_W - c["P0_w"]) / c["P0_w"] * 100
                oracle = min(hp.CANDIDATES, key=lambda k: (cells[k]["e"], k))
                exc = oracle != rank[0]
                groups.append({"parent": p, "regime": regime, "margin_pct": margin, "tau_pct": tau,
                               "flag_derived": margin <= tau, "flag_2pct": margin <= TIE_2PCT,
                               "exception": exc})
        def rule(key):
            fl = [g for g in groups if g[key]]
            return {"flagged_groups": len(fl), "energy_cells_measured": 4 * len(fl),
                    "exceptions_flagged": sum(g["exception"] for g in fl),
                    "exceptions_total": sum(g["exception"] for g in groups)}
        der, two = rule("flag_derived"), rule("flag_2pct")
        res[dname] = {"delta": delta, "derived": der, "rule_2pct": two, "groups": groups,
                      "h6_pass": der["exceptions_flagged"] == der["exceptions_total"]
                                 and der["energy_cells_measured"] < two["energy_cells_measured"]}
    assert res["h6_delta_median"]["rule_2pct"]["flagged_groups"] == 6
    assert res["h6_delta_median"]["rule_2pct"]["exceptions_total"] == 3
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true", help="calibration fit only; reads no operator label")
    args = ap.parse_args()
    cal = calibration_block()
    pr = cal["primary"]
    print(f"primary fit ({pr['n_windows']} windows): P0={pr['P0_w']:.2f} W  eps_L2={pr['eps_L2_pj_per_byte']:.2f} pJ/B  "
          f"eps_DRAM={pr['eps_DRAM_pj_per_byte']:.2f} pJ/B  R2={pr['r2']:.3f}  RMSE={pr['rmse_w']:.1f} W")
    for k in ("sensitivity_incl_strided", "sensitivity_incl_near_cap_triad"):
        s = cal[k]
        print(f"{k}: P0={s['P0_w']:.2f} eps_L2={s['eps_L2_pj_per_byte']:.2f} eps_DRAM={s['eps_DRAM_pj_per_byte']:.2f} R2={s['r2']:.3f}")
    print(f"calibration mean power {cal['calibration_mean_power_w']:.2f} W; H6 delta median {cal['h6_delta_median']:.4f}, max {cal['h6_delta_max']:.4f}")
    for w in cal["windows"]:
        print(f"  s{w['session']} {w['fixture']:12s} {w['regime']:8s} {w['candidate']:15s} {w['tier']:4s} "
              f"{w['rate_gbps']:8.1f} GB/s  P={w['power_w']:6.1f}  pred={w['pred_w']:6.1f}  {w['excluded'] or ''}")
    if args.fit_only:
        return
    design = operator_design()
    sc = score(cal, design)
    h6r = h6(cal, sc, design)
    result = {"protocol": "OPERATOR_WORK_TIME_PROTOCOL_2026-09-25.md", "calibration": cal,
              "design": {"|".join(k): v for k, v in design.items()}, "read_passes": READ_PASSES,
              "h5": {k: v for k, v in sc.items() if k not in ("grid", "hybrid_groups")}, "h6": h6r}
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
    p = sc["primary_63"]
    print("\nH5 primary (63 c2-c4 targets):")
    for n, s in p.items():
        print(f"  {n:34s} MdAPE {s['mdape_pct']:7.2f}%  P90 {s['p90_ape_pct']:7.2f}%")
    print("verdict:", sc["h5_verdict"])
    print("all 84:", {n: round(s["mdape_pct"], 2) for n, s in sc["all_84"].items()})
    for k, v in sc["splits"].items():
        print(" split", k, v)
    print("cross-session:", sc["cross_session"])
    for g in sc["per_group"]:
        print(f"  {g['parent']:30s} {g['regime']:6s} {g['tier']:4s} MdAPE {g['mdape_work_time_pct']:6.1f} "
              f"signed {g['median_signed_work_time_pct']:6.1f} meas {g['median_measured_power_w']:5.0f}W pred {g['median_predicted_power_w']:5.0f}W "
              f"cap m/p {g['n_measured_near_cap']}/{g['n_predicted_capped']}")
    for k, v in h6r.items():
        print(f"H6 {k}: delta={v['delta']:.4f} derived={v['derived']} 2%={v['rule_2pct']} pass={v['h6_pass']}")
        for g in v["groups"]:
            print(f"    {g['parent']:30s} {g['regime']:6s} margin {g['margin_pct']:7.3f}% tau {g['tau_pct']:7.3f}% "
                  f"flag {g['flag_derived']!s:5s} 2% {g['flag_2pct']!s:5s} exc {g['exception']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
