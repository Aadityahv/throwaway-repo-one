#!/usr/bin/env python3
"""Scorer for the unseen-operator test. Implements freeze.md exactly; nothing here is tuned.

    python3 score_unseen_operators.py --selftest                 # synthetic data only, no label is read
    python3 score_unseen_operators.py --gpu blackwell --raw-dirs DIR [DIR ...] --run-once
    python3 score_unseen_operators.py --gpu h100 --calib-run RUN --raw-dirs DIR ... --run-once

Only this script may read unseen-operator labels. It never opens `sealed_labels/` or any `FINAL-` directory,
and never reads Blackwell labels of the held-out operators.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESET = ROOT / "tiresias" / "app_runners"
sys.path.insert(0, str(RESET))
import operator_work_time_diagnostic as op  # noqa: E402  (calibration fit and predict_power, unchanged)

ADAPTERS = HERE.parent / "adapters" / "adapters.json"
GROUND_TRUTH = ROOT / "HARDWARE_GROUND_TRUTH.md"
WORKLOAD_CELLS = RESET / "workload_cells.csv"
REGIMES = ("small", "medium", "large")
BELOW_CAP_FRAC = 0.95
NEAR_TIER_BAND = (0.98, 1.02)
BOOT_B, BOOT_SEED = 10_000, 20260927
MIN_BELOW_CAP = 10
THREADS = 256

NEW_PARENTS = ("alt_cuda_samples_reduction", "alt_cuda_samples_copy", "alt_cuda_samples_transposefine")
HELD_PARENTS = ("final_pytorch_layer_norm", "final_pytorch_embedding", "final_xformers_indexed_select")

# freeze.md section 3 (constants) -- asserted against the committed calibration, never refitted.
GPUS = {
    "blackwell": {"cap_w": 600.0, "l2_bytes": 134_217_728, "calib_run": "CALIB-BLACKWELL-20260922T164827Z",
                  "expect": (152.53, 80.64, 179.59, 315.479), "held_out": False},
    "ada": {"cap_w": 250.0, "l2_bytes": 67_108_864, "calib_run": "CALIB-ADA-20260926_134601-89d8540",
            "expect": (112.52, 54.42, 190.91, 176.910), "held_out": True},
    "a100": {"cap_w": 400.0, "l2_bytes": 41_943_040, "calib_run": None, "expect": None, "held_out": True},
    "h100": {"cap_w": 700.0, "l2_bytes": 52_428_800, "calib_run": None, "expect": None, "held_out": True},
}
# freeze.md section 4 tier table for the new-code cells (parent, regime) -> tier per GPU. Small and medium are L2
# except where listed. The script stops if its computed tiers differ.
_DRAM = {
    "blackwell": {("alt_cuda_samples_copy", "large"), ("alt_cuda_samples_transposefine", "large")},
    "ada": {("alt_cuda_samples_reduction", "large"), ("alt_cuda_samples_copy", "large"),
            ("alt_cuda_samples_transposefine", "large")},
    "a100": {("alt_cuda_samples_reduction", "large"), ("alt_cuda_samples_copy", "large"),
             ("alt_cuda_samples_transposefine", "large")},
    "h100": {("alt_cuda_samples_reduction", "large"), ("alt_cuda_samples_copy", "large"),
             ("alt_cuda_samples_transposefine", "large")},
}


def frozen_tier(gpu: str, parent: str, regime: str) -> str:
    return "DRAM" if (parent, regime) in _DRAM[gpu] else "L2"


def assert_allowed(path: Path) -> None:
    assert "sealed_labels" not in path.parts and not any(p.startswith("FINAL-") for p in path.parts), path


def assert_hardware(gpu: str) -> None:
    cfg, text = GPUS[gpu], GROUND_TRUTH.read_text()
    assert f"{cfg['l2_bytes']:,}" in text, f"{gpu}: L2 size {cfg['l2_bytes']:,} not in HARDWARE_GROUND_TRUTH.md"
    cap = f"{cfg['cap_w']:g}"
    assert any("Power limit" in ln and cap in ln for ln in text.splitlines()), f"{gpu}: cap {cap} W not in file"


# ------------------------------------------------------------------ design (no labels)
def new_code_bytes(parent: str, n_or_side: int) -> int:
    if parent == "alt_cuda_samples_reduction":
        return 4 * n_or_side + 4 * math.ceil(n_or_side / THREADS)
    if parent == "alt_cuda_samples_copy":
        return 2 * 4 * n_or_side
    if parent == "alt_cuda_samples_transposefine":
        return 2 * 4 * n_or_side * n_or_side
    raise ValueError(parent)


def design(gpu: str) -> dict:
    """{(parent, regime, candidate): {bytes, footprint, l2_ratio, tier, tier_0p5, near_band}} for this GPU."""
    cfg, l2, out = GPUS[gpu], GPUS[gpu]["l2_bytes"], {}

    def add(parent, regime, cand, nbytes, fp):
        ratio = fp / l2
        out[(parent, regime, cand)] = {
            "bytes": nbytes, "footprint_bytes": fp, "l2_ratio": ratio,
            "tier": "L2" if ratio < 1.0 else "DRAM", "tier_0p5": "L2" if ratio < 0.5 else "DRAM",
            "near_band": NEAR_TIER_BAND[0] <= ratio <= NEAR_TIER_BAND[1]}

    for a in json.loads(ADAPTERS.read_text())["adapters"]:
        if a["parent_id"] not in NEW_PARENTS:
            continue
        for regime, r in a["regimes"].items():
            b = new_code_bytes(a["parent_id"], r["n"])
            for cand in a["candidates"]:
                add(a["parent_id"], regime, cand, b, b)
    if cfg["held_out"]:
        with WORKLOAD_CELLS.open(newline="") as f:
            for r in csv.DictReader(f):
                p = r["parent_id"]
                if p not in HELD_PARENTS:
                    continue
                rb, wb = int(r["global_read_bytes"]), int(r["global_write_bytes"])
                if p == "final_pytorch_layer_norm":   # held-out protocol section 3
                    stride = int(json.loads(r["native_controls_json"])["row_stride"])
                    fp = int(r["shape_a"]) * stride * 4 + 2 * int(r["shape_b"]) * 4 + wb
                else:
                    fp = rb + wb
                add(p, r["regime"], r["candidate_id"], rb + wb, fp)
    expected = 21 if not cfg["held_out"] else 48
    assert len(out) == expected, (gpu, len(out))
    for (p, regime, cand), d in out.items():
        if p in NEW_PARENTS:
            assert d["tier"] == frozen_tier(gpu, p, regime), f"tier differs from freeze.md table: {gpu} {p} {regime}"
    return out


# ------------------------------------------------------------------ labels
def read_labels(dirs, gpu: str, dsg: dict):
    """Returns (cells, excluded, uuids). cells[(p, reg, cand)] = {t, e, t1, e1, t2, e2}, t in microseconds."""
    scored = set(NEW_PARENTS) | (set(HELD_PARENTS) if GPUS[gpu]["held_out"] else set())
    by = defaultdict(lambda: defaultdict(dict))
    bad, uuids = {}, set()
    for d in dirs:
        d = Path(d)
        assert_allowed(d)
        with (d / "application_energy_raw.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                if r["parent_id"] not in scored:
                    continue
                assert r["launch_model"] == "graph" and r["graph_batch"] == "1000", r["run_id"]
                assert r["family"] == "application" and r["session"] in ("1", "2"), r["run_id"]
                key = (r["parent_id"], r["regime"], r["candidate_id"])
                assert key in dsg, f"cell not in the frozen design: {key}"
                assert r["session"] not in by[key] and (key, r["session"]) not in bad, f"duplicate row {key}"
                uuids.add(r["gpu_uuid"])
                if r["correctness_check"] != "True":
                    bad[(key, r["session"])] = "correctness check failed"
                    continue
                t = float(r["counted_launch_interval_s"]) / int(r["launch_count"]) * 1e6
                e = float(r["board_energy_j_per_launch"])
                assert t > 0 and e > 0, r["run_id"]
                by[key][r["session"]] = {"t": t, "e": e}
    cells, excluded = {}, {}
    for key in dsg:
        s = by.get(key, {})
        if set(s) == {"1", "2"}:
            cells[key] = {"t": st.mean((s["1"]["t"], s["2"]["t"])), "e": st.mean((s["1"]["e"], s["2"]["e"])),
                          "t1": s["1"]["t"], "e1": s["1"]["e"], "t2": s["2"]["t"], "e2": s["2"]["e"]}
        else:
            why = [bad[(key, ss)] for ss in ("1", "2") if (key, ss) in bad]
            excluded[key] = "; ".join(why) if why else "missing session " + "/".join(
                ss for ss in ("1", "2") if ss not in s)
    return cells, excluded, uuids


# ------------------------------------------------------------------ constants
def load_constants(gpu: str, calib_run: str | None) -> tuple[dict, dict, set]:
    cfg = GPUS[gpu]
    run = calib_run or cfg["calib_run"]
    assert run, f"{gpu}: no calibration run given (--calib-run)"
    op.P_CAP_W, op.L2_BYTES = cfg["cap_w"], cfg["l2_bytes"]
    op.CAL_DIR = RESET / "calibration_raw" / run
    cal = op.calibration_block()
    c = cal["primary"]
    if cfg["expect"] is not None and calib_run is None:
        got = (c["P0_w"], c["eps_L2_pj_per_byte"], c["eps_DRAM_pj_per_byte"], cal["calibration_mean_power_w"])
        for g, w, nm in zip(got, cfg["expect"], ("P0", "eps_L2", "eps_DRAM", "mean power")):
            assert abs(g - w) < 0.006 * max(1.0, abs(w)) or round(g, 2) == w or round(g, 3) == w, (gpu, nm, g, w)
    with (op.CAL_DIR / "raw_records.csv").open(newline="") as f:
        uuids = {r["gpu_uuid"] for r in csv.DictReader(f)}
    return c, {"calibration_mean_power_w": cal["calibration_mean_power_w"], "n_windows": c["n_windows"]}, uuids


# ------------------------------------------------------------------ scoring
def q(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summ(errs_abs, errs_signed):
    return {"n": len(errs_abs), "median_ape_pct": st.median(errs_abs) if errs_abs else None,
            "p90_ape_pct": q(errs_abs, 0.9), "median_signed_pct": st.median(errs_signed) if errs_signed else None}


def predictors(gpu: str, c: dict, cal_mean_w: float):
    cap = GPUS[gpu]["cap_w"]
    return {"calibration_only_model": lambda d, t, tier: op.predict_power(c, tier, d["bytes"], t),
            "constant_calibration_power": lambda d, t, tier: cal_mean_w,
            "cap_times_runtime": lambda d, t, tier: cap}


def score_cells(gpu, c, cal_mean_w, dsg, cells):
    cap, rows = GPUS[gpu]["cap_w"], []
    preds = predictors(gpu, c, cal_mean_w)
    for key, v in sorted(cells.items()):
        d = dsg[key]
        row = {"parent": key[0], "regime": key[1], "candidate": key[2], "set": "new_code" if key[0] in NEW_PARENTS
               else "held_out", "tier": d["tier"], "l2_ratio": d["l2_ratio"], "bytes": d["bytes"], "t_us": v["t"],
               "e_true_j": v["e"], "measured_power_w": v["e"] / v["t"] * 1e6, "ape_pct": {}, "signed_pct": {}}
        row["below_cap"] = row["measured_power_w"] < BELOW_CAP_FRAC * cap
        for name, f in preds.items():
            e = f(d, v["t"], d["tier"]) * v["t"] * 1e-6
            row["signed_pct"][name], row["ape_pct"][name] = (e / v["e"] - 1) * 100, abs(e / v["e"] - 1) * 100
        rows.append(row)
    for r in rows:   # one-measurement reference: same operator and regime, c1 measured power x t (c2 and above)
        anchor = next((x for x in rows if x["parent"] == r["parent"] and x["regime"] == r["regime"]
                       and x["candidate"] == "c1"), None)
        if r["candidate"] != "c1" and anchor is not None:
            e = anchor["measured_power_w"] * r["t_us"] * 1e-6
            r["signed_pct"]["one_measurement_reference"] = (e / r["e_true_j"] - 1) * 100
            r["ape_pct"]["one_measurement_reference"] = abs(e / r["e_true_j"] - 1) * 100
    return rows


def block(rows):
    names = sorted({n for r in rows for n in r["ape_pct"]})
    return {n: summ([r["ape_pct"][n] for r in rows if n in r["ape_pct"]],
                    [r["signed_pct"][n] for r in rows if n in r["signed_pct"]]) for n in names}


def med_ape(rows, name):
    return st.median(r["ape_pct"][name] for r in rows)


def bootstrap_delta(rows, a="calibration_only_model", b="constant_calibration_power"):
    """Paired bootstrap of median APE(a) - median APE(b) on below-cap cells, resampling operators."""
    by = defaultdict(list)
    for r in rows:
        by[r["parent"]].append(r)
    parents = sorted(by)
    if len(parents) < 2:
        return {"n_operators": len(parents), "interval": None, "note": "fewer than 2 operators: not informative"}
    rng = np.random.default_rng(BOOT_SEED)
    deltas = []
    for _ in range(BOOT_B):
        pick = rng.integers(0, len(parents), len(parents))
        sel = [r for i in pick for r in by[parents[i]]]
        deltas.append(med_ape(sel, a) - med_ape(sel, b))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_operators": len(parents), "point_delta_pct": med_ape(rows, a) - med_ape(rows, b),
            "interval": [float(lo), float(hi)], "spans_zero": bool(lo <= 0 <= hi)}


def score(gpu, c, cal_mean_w, dsg, cells, excluded):
    rows = score_cells(gpu, c, cal_mean_w, dsg, cells)
    below = [r for r in rows if r["below_cap"]]
    at_cap = [r for r in rows if not r["below_cap"]]
    prim_m, prim_c = (med_ape(below, "calibration_only_model"), med_ape(below, "constant_calibration_power")) \
        if below else (None, None)
    underpowered = len(below) < MIN_BELOW_CAP
    boot = bootstrap_delta(below) if below else {"interval": None, "note": "no below-cap cells"}
    verdict = {"n_below_cap": len(below), "model_median_ape_pct": prim_m, "constant_median_ape_pct": prim_c,
               "underpowered": underpowered,
               "result": ("underpowered" if underpowered else "pass" if prim_m < prim_c else "fail"),
               "paired_interval": boot,
               "qualifier": ("pass, unresolved (interval spans zero)"
                             if not underpowered and prim_m < prim_c and boot.get("spans_zero") else None)}
    per_op = {p: block([r for r in rows if r["parent"] == p]) for p in sorted({r["parent"] for r in rows})}
    per_op_reg = {f"{p}|{g}": block([r for r in rows if r["parent"] == p and r["regime"] == g])
                  for p in sorted({r["parent"] for r in rows}) for g in REGIMES if any(
                      r["parent"] == p and r["regime"] == g for r in rows)}
    # tier sensitivities (never change the verdict)
    sens = {}
    for label, sel in (("near_boundary_other_tier", lambda d: d["near_band"]), ):
        flip = [k for k in cells if sel(dsg[k])]
        alt = []
        for k in flip:
            other = "L2" if dsg[k]["tier"] == "DRAM" else "DRAM"
            e = op.predict_power(c, other, dsg[k]["bytes"], cells[k]["t"]) * cells[k]["t"] * 1e-6
            alt.append({"cell": list(k), "primary_tier": dsg[k]["tier"], "ape_primary_pct": next(
                r["ape_pct"]["calibration_only_model"] for r in rows if (r["parent"], r["regime"], r["candidate"]) == k),
                "ape_other_tier_pct": abs(e / cells[k]["e"] - 1) * 100})
        sens[label] = alt
    half = []
    for k, v in cells.items():
        e = op.predict_power(c, dsg[k]["tier_0p5"], dsg[k]["bytes"], v["t"]) * v["t"] * 1e-6
        half.append(abs(e / v["e"] - 1) * 100)
    sens["tier_threshold_0p5"] = {"median_ape_pct_all_cells": st.median(half) if half else None}
    return {"gpu": gpu, "constants": c, "calibration_mean_power_w": cal_mean_w, "cap_w": GPUS[gpu]["cap_w"],
            "l2_bytes": GPUS[gpu]["l2_bytes"], "n_cells_scored": len(rows),
            "excluded_cells": {"|".join(k): v for k, v in excluded.items()},
            "verdict": verdict, "below_cap": block(below), "at_cap": block(at_cap), "all_cells": block(rows),
            "new_code": block([r for r in rows if r["set"] == "new_code"]),
            "held_out": block([r for r in rows if r["set"] == "held_out"]),
            "per_operator": per_op, "per_operator_regime": per_op_reg, "sensitivity": sens,
            "accelwattch": "omitted unless static inputs were committed before the freeze (freeze.md section 5)",
            "cells": rows}


def freeze_constants(gpu, calib_run, out_dir):
    """Calibration only. Commit this artifact before scoring a new platform's labels."""
    import hashlib
    assert_hardware(gpu)
    c, meta, uuids = load_constants(gpu, calib_run)
    assert len(uuids) == 1, "calibration must come from one physical board"
    run_id = calib_run or GPUS[gpu]["calib_run"]
    artifact = {"gpu": gpu, "calib_run": run_id, "constants": c, "metadata": meta,
                "board_uuids": sorted(uuids), "cap_w": GPUS[gpu]["cap_w"],
                "l2_bytes": GPUS[gpu]["l2_bytes"],
                "inputs_sha256": {name: hashlib.sha256((op.CAL_DIR / name).read_bytes()).hexdigest()
                                  for name in ("manifest.jsonl", "raw_records.csv")}}
    path = Path(out_dir) / f"constants_{gpu}.json"
    assert not path.exists(), f"{path} exists: refusing to replace frozen constants"
    path.write_text(json.dumps(artifact, indent=1) + "\n")
    return artifact



def require_committed_constants(path):
    """Verify the exact artifact is in HEAD, without displaying its contents or opening labels."""
    import subprocess
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise AssertionError("constants artifact must be committed inside the repository") from None
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True)
    assert proc.returncode == 0 and proc.stdout == path.read_bytes(), "commit frozen constants before reading operator labels"

def run(gpu, raw_dirs, calib_run, out_dir, run_once=True):
    out = Path(out_dir) / f"results_{gpu}.json"
    assert not (run_once and out.exists()), f"{out} exists: one scoring run per GPU (freeze.md section 7)"
    assert_hardware(gpu)
    dsg = design(gpu)
    c, meta, cal_uuids = load_constants(gpu, calib_run)   # H100 constants are fitted and fixed before any label opens
    if gpu in ("a100", "h100"):
        import hashlib
        frozen_path = Path(out_dir) / f"constants_{gpu}.json"
        assert frozen_path.exists(), "run --freeze-constants and commit it before reading operator labels"
        frozen = json.loads(frozen_path.read_text())
        assert frozen["gpu"] == gpu and frozen["calib_run"] == calib_run
        assert frozen["constants"] == c and frozen["metadata"] == meta, "frozen constants differ"
        assert frozen["board_uuids"] == sorted(cal_uuids), "frozen calibration board differs"
        for name, digest in frozen["inputs_sha256"].items():
            assert hashlib.sha256((op.CAL_DIR / name).read_bytes()).hexdigest() == digest, "calibration changed"
        require_committed_constants(frozen_path)
    cells, excluded, uuids = read_labels(raw_dirs, gpu, dsg)
    assert len(uuids) == 1, f"operator windows span several boards: {uuids}"
    same_board = uuids <= cal_uuids and len(cal_uuids) == 1
    res = score(gpu, c, meta["calibration_mean_power_w"], dsg, cells, excluded)
    res["board_uuid_operators"], res["board_uuid_calibration"] = sorted(uuids), sorted(cal_uuids)
    res["same_board_as_calibration"] = bool(same_board)
    out.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ self-test (synthetic data only)
def _synth(tmp: Path, gpu, dsg, c, cal_mean_w, truth, uuid="GPU-SYNTH", drop=(), badcorr=()):
    cap = GPUS[gpu]["cap_w"]
    hdr = ("run_id,session,family,parent_id,regime,candidate_id,gpu_uuid,launch_count,counted_launch_interval_s,"
           "board_energy_j_per_launch,correctness_check,launch_model,graph_batch").split(",")
    rng = np.random.default_rng(1)
    d = tmp / "raw"
    d.mkdir()
    with (d / "application_energy_raw.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for k, v in sorted(dsg.items()):
            for s in ("1", "2"):
                if (k, s) in drop:
                    continue
                t_s = 1e-5 * (1 + (hash(k) % 7)) * (1 + 0.01 * rng.standard_normal())
                if truth == "model":
                    p = op.predict_power(c, v["tier"], v["bytes"], t_s * 1e6)
                else:
                    p = cal_mean_w * (0.8 + 0.4 * rng.random())
                p = min(p, 0.9 * cap) * (1 + 0.01 * rng.standard_normal())
                w.writerow([f"{k}-{s}", s, "application", *k, uuid, 1000, t_s * 1000, p * t_s,
                            "False" if (k, s) in badcorr else "True", "graph", "1000"])
    return [d]


def selftest():
    import subprocess
    # 1. byte formulas agree with the corrected adapter checker; frozen tiers reproduce for every GPU
    sys.path.insert(0, str(HERE.parent / "adapters"))
    import check_manifest as cm
    man = json.loads(ADAPTERS.read_text())
    rep = {a["parent_id"]: cm.check_bytes_tiers(a, cm.check_geometry(a)) for a in man["adapters"]}
    for a in man["adapters"]:
        for regime, r in a["regimes"].items():
            assert rep[a["parent_id"]][regime]["bytes"] == new_code_bytes(a["parent_id"], r["n"]), (a["parent_id"], regime)
    for g in GPUS:
        assert_hardware(g)
        dd = design(g)
        assert len(dd) == (21 if not GPUS[g]["held_out"] else 48)
    # 2. tier table mismatch is refused
    saved = set(_DRAM["ada"])
    _DRAM["ada"].discard(("alt_cuda_samples_reduction", "large"))
    try:
        design("ada")
        raise AssertionError("tier mismatch accepted")
    except AssertionError as e:
        assert "tier differs" in str(e), e
    finally:
        _DRAM["ada"].clear(); _DRAM["ada"].update(saved)
    # 3. sealed and FINAL- paths refused
    for bad in (RESET / "sealed_labels" / "x", RESET / "application_energy_raw" / "FINAL-X"):
        try:
            assert_allowed(bad)
            raise SystemExit("forbidden path accepted")
        except AssertionError:
            pass
    # 4. Blackwell constants reproduce; predict_power identical to the diagnostic on Blackwell
    c, meta, cal_uuids = load_constants("blackwell", None)
    assert op.predict_power(c, "L2", 1e6, 10.0) == min(600.0, c["P0_w"] + c["eps_L2_pj_per_byte"] * 1e-12 * 1e6 / 1e-5)
    dsg = design("blackwell")
    cal_mean = meta["calibration_mean_power_w"]
    # 5. synthetic truth = the model: model beats constant power, 21 cells scored, run-once enforced
    for truth, expect in (("model", "pass"), ("constant", "fail")):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dirs = _synth(td, "blackwell", dsg, c, cal_mean, truth,
                          uuid=sorted(cal_uuids)[0], drop={(("alt_cuda_samples_copy", "small", "c2"), "2")},
                          badcorr={(("alt_cuda_samples_reduction", "large", "c1"), "1")})
            r = run("blackwell", dirs, None, td)
            assert r["n_cells_scored"] == 19 and len(r["excluded_cells"]) == 2, r["excluded_cells"]
            assert "correctness check failed" in json.dumps(r["excluded_cells"])
            assert r["verdict"]["result"] in (expect, "underpowered"), (truth, r["verdict"])
            assert r["same_board_as_calibration"], "uuid match expected"
            if r["verdict"]["result"] != "underpowered":
                assert r["verdict"]["result"] == expect
            try:
                run("blackwell", dirs, None, td)
                raise AssertionError("second run accepted")
            except AssertionError as e:
                assert "one scoring run per GPU" in str(e)
            # wrong board is reported, not silently accepted
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dirs = _synth(td, "blackwell", dsg, c, cal_mean, "model", uuid="GPU-OTHER")
        r = run("blackwell", dirs, None, td)
        assert r["same_board_as_calibration"] is False
    print("SELFTEST UNSEEN-OPERATOR SCORER PASSED (synthetic data only; no label read)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze-constants", action="store_true")
    ap.add_argument("--gpu", choices=sorted(GPUS))
    ap.add_argument("--raw-dirs", nargs="+")
    ap.add_argument("--calib-run")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--run-once", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.freeze_constants:
        assert a.gpu and a.calib_run and not a.raw_dirs, "--gpu and --calib-run only; no label paths"
        freeze_constants(a.gpu, a.calib_run, a.out_dir)
        print("CALIBRATION CONSTANTS FROZEN: commit the artifact before scoring labels")
        return
    assert a.gpu and a.raw_dirs and a.run_once, "--gpu, --raw-dirs and --run-once are required"
    res = run(a.gpu, a.raw_dirs, a.calib_run, a.out_dir)
    print(json.dumps({k: res[k] for k in ("gpu", "n_cells_scored", "excluded_cells", "verdict",
                                          "same_board_as_calibration")}, indent=1))


if __name__ == "__main__":
    main()
