#!/usr/bin/env python3
"""EXPLORATORY, post hoc: on-chip work term for the M362 operator model.

Plan: EXPLORATORY_ONCHIP_PLAN_2026-09-26.md (committed before this script computed any error).
M362 (OPERATOR_WORK_TIME_RESULT_2026-09-25.md) remains the registered operator result; nothing frozen is changed.

CPU only. Operator data come only from hybrid_probe_diagnostic.load_data() (7 nonsealed Blackwell parents).
Calibration data come from operator_work_time_diagnostic.load_calibration(). No sealed path is opened.

    python tiresias/app_runners/explore_onchip.py
"""
from __future__ import annotations

import csv
import json
import statistics as stats
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import hybrid_probe_diagnostic as hp  # noqa: E402
import operator_work_time_diagnostic as owt  # noqa: E402

OUT = ROOT / "explore_onchip_results.json"
P_CAP = owt.P_CAP_W
CAP_SPLIT = 0.95 * P_CAP
FB = 4
SECTOR = 32

# CALIBRATION_SUITE_SPEC.md, Blackwell row: static sector counts per launch for the strided stores.
STRIDED_STORE_SECTORS_BELOW = {"strided_s7": 132_352, "strided_s33": 192_512}


def assert_allowed(p: Path) -> None:
    assert "sealed_labels" not in p.parts and not any(x.startswith("FINAL-") for x in p.parts), p


# ---------------------------------------------------------------- calibration features
def calib_manifest() -> dict:
    d = owt.CAL_DIR
    assert_allowed(d)
    out = {}
    for line in (d / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            out[(int(m["session"]), m["fixture"], m["regime"], m["candidate_id"])] = m
    return out


def calib_shared_bytes(fixture: str, c: dict) -> int:
    if fixture == "transpose":                    # tiled local transpose: one tile staged per element
        return 2 * c["n"] * c["n"] * FB
    if fixture == "shared_stage":
        w = c["blocks"] * c["batches"] * c["chunks"] * c["stage_elements"] * FB
        r = c["blocks"] * c["batches"] * c["chunks"] * c["threads"] * c["reuse"] * FB
        return w + r
    return 0


def calib_rows() -> list[dict]:
    man = calib_manifest()
    rows = owt.load_calibration()
    for r in rows:
        m = man[(r["session"], r["fixture"], r["regime"], r["candidate"])]
        r["shared_bytes"] = calib_shared_bytes(r["fixture"], m["controls"])
        r["sector_bytes"] = r["bytes"]
        if r["strided"] and r["tier"] == "L2":
            r["sector_bytes"] = STRIDED_STORE_SECTORS_BELOW[r["candidate"]] * SECTOR
    return rows


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ theta
    ss_res = float(((y - pred) ** 2).sum())
    return theta, 1 - ss_res / float(((y - y.mean()) ** 2).sum()), (ss_res / len(y)) ** 0.5


def fit_e1(rows: list[dict]) -> dict:
    fr = [r for r in rows if r["excluded"] is None]
    assert len(fr) == 22
    X = np.array([[1.0,
                   r["bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "L2" else 0.0,
                   r["bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "DRAM" else 0.0,
                   r["shared_bytes"] / (r["t_us"] * 1e-6)] for r in fr])
    y = np.array([r["power_w"] for r in fr])
    th, r2, rmse = ols(X, y)
    return {"P0_w": th[0], "eps_L2_pj_per_byte": th[1] * 1e12, "eps_DRAM_pj_per_byte": th[2] * 1e12,
            "eps_sh_pj_per_byte": th[3] * 1e12, "n_windows": len(fr), "r2": r2, "rmse_w": rmse}


def fit_s2(rows: list[dict]) -> dict:
    fr = [r for r in rows if r["excluded"] is None or r["strided"]]
    assert len(fr) == 30
    X = np.array([[1.0,
                   r["sector_bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "L2" else 0.0,
                   r["sector_bytes"] / (r["t_us"] * 1e-6) if r["tier"] == "DRAM" else 0.0] for r in fr])
    y = np.array([r["power_w"] for r in fr])
    th, r2, rmse = ols(X, y)
    return {"P0_w": th[0], "eps_L2_pj_per_byte": th[1] * 1e12, "eps_DRAM_pj_per_byte": th[2] * 1e12,
            "n_windows": len(fr), "r2": r2, "rmse_w": rmse}


# ---------------------------------------------------------------- operator features (design only, no labels)
def reduction_shared_bytes(which: int, T: int, blocks: int) -> int:
    """static_features.py tree formulas (element counts there), converted to bytes here."""
    log_t = T.bit_length() - 1
    del log_t
    if which in (2, 3):
        rd = 2 * (T - 1) + (1 if which == 2 else 0)
        wr = 2 * T - 1
    elif which == 4:
        stages, s = [], T // 2
        while s > 32:
            stages.append(s)
            s //= 2
        active = sum(stages)
        rd = 2 * active + (32 if T >= 64 else 0)
        wr = T + active
    else:
        active = sum(s for s in (256, 128, 64) if T >= 2 * s)
        rd = active + (32 if T >= 64 else 0)
        wr = T + active
    return (rd + wr) * blocks * FB


def operator_features() -> dict:
    design = owt.operator_design()
    feats = {}
    with owt.WORKLOAD_CELLS.open(newline="") as f:
        for r in csv.DictReader(f):
            p = r["parent_id"]
            if p not in owt.PARENTS:
                continue
            key = (p, r["regime"], r["candidate_id"])
            d = dict(design[key])
            a, b = int(r["shape_a"]), int(r["shape_b"])
            other = json.loads(r["other_traffic_json"]) if r["other_traffic_json"] else {}
            ctl = json.loads(r["native_controls_json"]) if r["native_controls_json"] else {}
            rb, wb = d["read_bytes"], d["write_bytes"]
            sh, op, sec = 0, 0, rb + wb
            if p == "train_cuda_samples_transpose":
                if ctl["kernel_variant"] != 2:
                    sh = 2 * a * b * FB
                else:
                    sec = rb + wb * (SECTOR // FB)      # naive: one 4 B element per 32 B write sector
            elif p == "train_cuda_samples_reduction":
                sh = reduction_shared_bytes(ctl["which_kernel"], ctl["threads"], ctl["blocks"])
                op = other["logical_sum_operations"]
                sec = rb + wb * (SECTOR // FB)          # one-thread partial write per block
            elif p in ("dev_triton_softmax", "dev_pytorch_rowwise_softmax"):
                op = other["row_max_comparisons"] + other["row_sum_additions"] + other["exp_operations"]
                assert op == 2 * a * (b - 1) + a * b
            elif p == "final_triton_layer_norm":
                op = 2 * a * (b - 1) + a
            d.update(shared_bytes=sh, onchip_ops=op, sector_bytes=sec)
            feats[key] = d
    assert len(feats) == 84
    return feats


# ---------------------------------------------------------------- models
def base_power(c: dict, d: dict, t_us: float, nbytes: float | None = None) -> float:
    """Uncapped P0 + tier work power."""
    n = d["bytes"] if nbytes is None else nbytes
    return c["P0_w"] + owt.work_power(c, d["tier"], n, t_us)


def fit_eps_op(cells: list[dict], c: dict, byte_key: str) -> float:
    num = den = 0.0
    for r in cells:
        if r["meas_w"] >= CAP_SPLIT:
            continue
        x = r["d"]["onchip_ops"] / (r["t_us"] * 1e-6)
        if x == 0:
            continue
        resid = r["meas_w"] - base_power(c, r["d"], r["t_us"], r["d"][byte_key])
        num += x * resid
        den += x * x
    assert den > 0, "no training cell with on-chip ops"
    return max(0.0, num / den)


def summarize(v: list[float]) -> dict:
    return {"n": len(v), "mdape_pct": stats.median(v), "p90_ape_pct": hp.quantile(v, 0.9)}


def main() -> None:
    cal = owt.calibration_block()
    M = cal["primary"]
    assert round(M["P0_w"], 2) == 152.53
    crow = calib_rows()
    e1 = fit_e1(crow)
    s2 = fit_s2(crow)
    feats = operator_features()

    grid, _ = hp.load_data()
    hyb = hp.evaluate(grid)
    bar0 = hyb["energy_error"]["training_mean_power"]["mdape_pct"]
    bar1 = hyb["energy_error"]["anchor_constant_power"]["mdape_pct"]
    assert round(bar0, 2) == 20.37 and round(bar1, 2) == 6.37

    cells = []
    for p in sorted(grid):
        for (regime, cand), v in sorted(grid[p].items()):
            d = feats[(p, regime, cand)]
            cells.append({"parent": p, "regime": regime, "candidate": cand, "t_us": v["t"], "e_j": v["e"],
                          "meas_w": v["e"] / v["t"] * 1e6, "d": d})
    assert len(cells) == 84

    def e1_power(r):
        d, t = r["d"], r["t_us"]
        return min(P_CAP, base_power(e1, d, t) + e1["eps_sh_pj_per_byte"] * 1e-12 * d["shared_bytes"] / (t * 1e-6))

    # E2 LOPO and in-sample, on logical bytes (E2) and sector bytes (E2+S1)
    eps_lopo = {"bytes": {}, "sector_bytes": {}}
    for bk in eps_lopo:
        for p in owt.PARENTS:
            eps_lopo[bk][p] = fit_eps_op([r for r in cells if r["parent"] != p], M, bk)
    eps_in = {bk: fit_eps_op(cells, M, bk) for bk in ("bytes", "sector_bytes")}

    def e2_power(r, eps, bk):
        d, t = r["d"], r["t_us"]
        return min(P_CAP, base_power(M, d, t, d[bk]) + eps * d["onchip_ops"] / (t * 1e-6))

    variants = {
        "M362": lambda r: min(P_CAP, base_power(M, r["d"], r["t_us"])),
        "E1_calib_shared": e1_power,
        "E2_onchip_ops_LOPO": lambda r: e2_power(r, eps_lopo["bytes"][r["parent"]], "bytes"),
        "E2_onchip_ops_insample": lambda r: e2_power(r, eps_in["bytes"], "bytes"),
        "S1_sector_bytes_M362_constants": lambda r: min(P_CAP, base_power(M, r["d"], r["t_us"], r["d"]["sector_bytes"])),
        "S2_sector_bytes_calib_refit": lambda r: min(P_CAP, base_power(s2, r["d"], r["t_us"], r["d"]["sector_bytes"])),
        "E2_plus_S1_LOPO": lambda r: e2_power(r, eps_lopo["sector_bytes"][r["parent"]], "sector_bytes"),
    }
    for r in cells:
        r["signed"] = {}
        for name, f in variants.items():
            pw = f(r)
            r["signed"][name] = (pw * r["t_us"] * 1e-6 / r["e_j"] - 1) * 100

    targets = [r for r in cells if r["candidate"] != "c1"]
    below = [r for r in targets if r["meas_w"] < CAP_SPLIT]
    assert len(targets) == 63 and len(below) == 35
    m362 = summarize([abs(r["signed"]["M362"]) for r in targets])
    assert round(m362["mdape_pct"], 2) == 5.69, m362
    assert round(summarize([abs(r["signed"]["M362"]) for r in below])["mdape_pct"], 2) == 10.44

    def block(rows):
        return {n: summarize([abs(r["signed"][n]) for r in rows]) for n in variants}

    summary = {"targets_63": block(targets), "below_cap_35": block(below), "all_84": block(cells),
               "streaming_c2c4": block([r for r in targets if r["parent"] in (
                   "train_triton_vector_add", "final_cuda_samples_copy", "train_cuda_samples_transpose")])}
    per_group = []
    for p in owt.PARENTS:
        for regime in ("small", "medium", "large"):
            g = [r for r in cells if r["parent"] == p and r["regime"] == regime]
            gt = [r for r in g if r["candidate"] != "c1"]
            per_group.append({"parent": p, "regime": regime, "tier": g[0]["d"]["tier"],
                              "meas_w_median": stats.median(r["meas_w"] for r in g),
                              "median_signed_c2c4": {n: stats.median(r["signed"][n] for r in gt) for n in variants},
                              "c1_signed": {n: next(r for r in g if r["candidate"] == "c1")["signed"][n]
                                            for n in variants}})
    out = {
        "label": "EXPLORATORY, post hoc: M362 remains the registered operator result.",
        "plan": "EXPLORATORY_ONCHIP_PLAN_2026-09-26.md",
        "bars": {"zero_probe_mdape": bar0, "one_probe_mdape": bar1},
        "constants": {"M362": {k: M[k] for k in ("P0_w", "eps_L2_pj_per_byte", "eps_DRAM_pj_per_byte")},
                      "E1": e1, "S2": s2,
                      "E2_eps_op_pj_LOPO": {p: v * 1e12 for p, v in eps_lopo["bytes"].items()},
                      "E2_eps_op_pj_insample": eps_in["bytes"] * 1e12,
                      "E2S1_eps_op_pj_LOPO": {p: v * 1e12 for p, v in eps_lopo["sector_bytes"].items()},
                      "E2S1_eps_op_pj_insample": eps_in["sector_bytes"] * 1e12},
        "summary": summary, "per_group": per_group,
        "cells": [{k: r[k] for k in ("parent", "regime", "candidate", "t_us", "e_j", "meas_w", "signed")}
                  | {"shared_bytes": r["d"]["shared_bytes"], "onchip_ops": r["d"]["onchip_ops"],
                     "sector_bytes": r["d"]["sector_bytes"], "bytes": r["d"]["bytes"], "tier": r["d"]["tier"]}
                  for r in cells],
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")

    # console report
    print(out["label"])
    print("E1 constants:", {k: round(v, 3) for k, v in e1.items()})
    print("S2 constants:", {k: round(v, 3) for k, v in s2.items()})
    print("eps_op LOPO (pJ/op):", {p[:18]: round(v, 1) for p, v in out["constants"]["E2_eps_op_pj_LOPO"].items()},
          "in-sample", round(out["constants"]["E2_eps_op_pj_insample"], 1))
    print("eps_op E2+S1 LOPO:", {p[:18]: round(v, 1) for p, v in out["constants"]["E2S1_eps_op_pj_LOPO"].items()})
    for sk in ("targets_63", "below_cap_35", "all_84", "streaming_c2c4"):
        print(f"\n{sk}")
        for n, s in summary[sk].items():
            print(f"  {n:34s} n={s['n']:2d} MdAPE {s['mdape_pct']:6.2f}  P90 {s['p90_ape_pct']:6.2f}")
    print("\nper group median signed c2-c4 (M362 | E1 | E2-LOPO | S1 | S2 | E2+S1) and c1 (M362 | S1 | S2)")
    for g in per_group:
        s, c = g["median_signed_c2c4"], g["c1_signed"]
        print(f"  {g['parent'][:26]:26s} {g['regime']:6s} {g['meas_w_median']:5.0f}W "
              f"{s['M362']:+6.1f} {s['E1_calib_shared']:+6.1f} {s['E2_onchip_ops_LOPO']:+6.1f} "
              f"{s['S1_sector_bytes_M362_constants']:+6.1f} {s['S2_sector_bytes_calib_refit']:+6.1f} "
              f"{s['E2_plus_S1_LOPO']:+6.1f} | c1 {c['M362']:+6.1f} {c['S1_sector_bytes_M362_constants']:+6.1f} "
              f"{c['S2_sector_bytes_calib_refit']:+6.1f} {c['E2_onchip_ops_LOPO']:+6.1f}")


if __name__ == "__main__":
    main()
