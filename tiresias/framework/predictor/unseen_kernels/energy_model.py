"""Decomposed energy model for the unseen-kernel test (Blackwell). CPU only.

    E = min( P0*t  +  eps_L2*bytes_L2 + eps_DRAM*bytes_DRAM  +  eps_inst*N_non-memory + eps_sfu*N_special-function ,  cap*t )

runtime term     P0*t, base power times runtime (the only place a predicted runtime enters)
memory term      logical bytes of the launch, charged at the L2 or the DRAM rate of the cell's tier
compute term     lane-level dynamic instruction counts: all instructions except global loads and stores, plus a
                 separate rate for special-function (MUFU) instructions

The four rates are fitted ONCE, by non-negative least squares on relative error, on the below-cap Blackwell development
cells that have static instruction counts (the 73 cells of the exploratory analysis, energy_decomposition/). The fit is
written to energy_model_frozen.json with every digit and is never refitted after any unseen-kernel measurement.
Base power P0 and the power cap come from the development table, as in the frozen runtime-plus-traffic formula.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
MI = HERE.parents[1] / "compile_evidence"
DEV_TABLE = MI / "development/development_cells.csv"
FEATURES = HERE.parent / "features_blackwell.json"
FROZEN = HERE / "energy_model_frozen.json"
TERMS = ("bytes_L2", "bytes_DRAM", "nonmem_lane_instructions", "sfu_lane_instructions")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def terms_from_totals(totals, logical_bytes, tier):
    """Model inputs from a per-launch totals block (adapter.per_launch_totals) and the cell's logical bytes and tier."""
    fam = totals["families"]
    gl = fam.get("global_load", {}).get("lane_instructions", 0) + fam.get("global_store", {}).get("lane_instructions", 0)
    return dict(bytes_L2=logical_bytes if tier == "L2" else 0.0, bytes_DRAM=logical_bytes if tier == "DRAM" else 0.0,
                nonmem_lane_instructions=totals["total_lane_instructions"] - gl,
                sfu_lane_instructions=fam.get("special_function", {}).get("lane_instructions", 0))


def load_development():
    """Blackwell development cells with static counts: [(cell_id, op, runtime_s, energy_j, terms dict, below_cap)], P0, cap."""
    feats = {r["cell_id"]: r for r in json.loads(FEATURES.read_text())["rows"] if r["status"] in ("supported", "supported_with_assumptions")}
    out, p0s, caps = [], set(), set()
    for r in csv.DictReader(DEV_TABLE.open(newline="")):
        if r["gpu"] != "blackwell" or r["cell_id"] not in feats:
            continue
        f = feats[r["cell_id"]]
        w = f["work"]
        t = terms_from_totals(dict(families=w["families"], total_lane_instructions=w["total_lane_instructions"]),
                              float(r["actual_logical_bytes_per_launch"]), r["tier"])
        out.append((r["cell_id"], r["operator_group"].split("/")[-1], float(r["runtime_s"]), float(r["energy_j"]), t, r["below_cap"] == "True"))
        p0s.add(round(float(r["physics_p0_w"]), 9))
        caps.add(float(r["cap_w"]))
    if len(p0s) != 1 or len(caps) != 1:
        raise SystemExit("base power or cap is not constant across Blackwell development cells")
    return out, p0s.pop(), caps.pop()


def fit():
    cells, p0, cap = load_development()
    below = [c for c in cells if c[5]]
    X = np.array([[c[4][k] * 1e-12 for k in TERMS] for c in below])
    t = np.array([c[2] for c in below])
    e = np.array([c[3] for c in below])
    w = 1 / e
    coef, _ = nnls(X * w[:, None], (e - p0 * t) * w)
    return dict(P0_w=p0, cap_w=cap, rates_pJ_per_unit=dict(zip(TERMS, [float(x) for x in coef])), fitted_cells=len(below), development_cells_with_counts=len(cells))


def write_frozen():
    model = fit()
    model.update(schema="decomposed_energy_model_frozen/1", fit="non-negative least squares on relative error, below-cap Blackwell development cells with static counts",
                 inputs_sha256={"development_cells.csv": sha(DEV_TABLE), "features_blackwell.json": sha(FEATURES), "energy_model.py": sha(__file__)},
                 note="Written before any unseen-kernel timing or energy. Never refitted afterwards.")
    FROZEN.write_text(json.dumps(model, indent=1, sort_keys=True) + "\n")
    return model


def predict(terms, runtime_s, model=None):
    """-> dict(energy_j, runtime_term_j, memory_term_j, compute_term_j, sfu_term_j, capped)."""
    m = model or json.loads(FROZEN.read_text())
    r = m["rates_pJ_per_unit"]
    runtime = m["P0_w"] * runtime_s
    memory = (r["bytes_L2"] * terms["bytes_L2"] + r["bytes_DRAM"] * terms["bytes_DRAM"]) * 1e-12
    compute = r["nonmem_lane_instructions"] * terms["nonmem_lane_instructions"] * 1e-12
    sfu = r["sfu_lane_instructions"] * terms["sfu_lane_instructions"] * 1e-12
    uncapped = runtime + memory + compute + sfu
    cap = m["cap_w"] * runtime_s
    return dict(energy_j=min(uncapped, cap), uncapped_j=uncapped, runtime_term_j=runtime, memory_term_j=memory,
                compute_term_j=compute, sfu_term_j=sfu, capped=uncapped > cap)


if __name__ == "__main__":
    print(json.dumps(write_frozen(), indent=1))
