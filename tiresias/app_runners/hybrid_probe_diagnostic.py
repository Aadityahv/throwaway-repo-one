#!/usr/bin/env python3
"""Retrospective one-energy-probe analysis, per HYBRID_PROBE_PROTOCOL_2026-09-24.md.

Only committed train/development Blackwell data are read. No sealed path is used.
"""
from __future__ import annotations

import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "model_dataset/model_dataset_blackwell_trainval.csv"
RAW_SOURCES = {
    "dev_pytorch_rowwise_softmax": [
        ROOT / "application_energy_raw/SOFTMAX-DEV-20260922T185912Z/application_energy_raw.csv",
    ],
    "final_triton_layer_norm": [
        ROOT / "application_energy_raw/LAYERNORMDEV-20260923T000000Z/application_energy_raw.csv",
        ROOT / "application_energy_raw/LAYERNORMDEV-S2-20260923T013000Z/application_energy_raw.csv",
    ],
}
SESSION_SOURCES = {
    "train_triton_vector_add": ROOT / "application_energy_raw_graph/application_energy_raw.csv",
    "dev_triton_softmax": ROOT / "application_energy_raw_graph/application_energy_raw.csv",
    **{k: v[0] for k, v in RAW_SOURCES.items()},
}
REGIMES = ("small", "medium", "large")
CANDIDATES = ("c1", "c2", "c3", "c4")
CANONICAL_PARENTS = {
    "train_triton_vector_add", "dev_triton_softmax", "final_cuda_samples_copy",
    "train_cuda_samples_transpose", "train_cuda_samples_reduction",
}
EXPECTED_PARENTS = CANONICAL_PARENTS | set(RAW_SOURCES)


def read_csv(path: Path) -> list[dict[str, str]]:
    assert "sealed_labels" not in path.parts and not any(p.startswith("FINAL-") for p in path.parts), path
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def raw_rows(parent: str) -> list[dict]:
    rows = []
    for path in RAW_SOURCES[parent]:
        rows += [r for r in read_csv(path) if r["parent_id"] == parent]
    return rows


def check_raw(r: dict) -> None:
    assert r["correctness_check"] == "True", r["run_id"]
    assert r["launch_model"] == "graph" and r["graph_batch"] == "1000", r["run_id"]
    assert r["family"] == "application" and r["session"] in ("1", "2"), r["run_id"]
    assert r["regime"] in REGIMES and r["candidate_id"] in CANDIDATES, r["run_id"]


def labels_from_raw(parent: str, rows: list[dict]) -> tuple[dict, dict]:
    by_session = defaultdict(dict)
    for r in rows:
        check_raw(r)
        key = (r["regime"], r["candidate_id"])
        assert key not in by_session[r["session"]], (parent, r["session"], key)
        t = float(r["counted_launch_interval_s"]) / int(r["launch_count"]) * 1e6
        e = float(r["board_energy_j_per_launch"])
        assert t > 0 and e > 0
        by_session[r["session"]][key] = {"t": t, "e": e}
    expected = {(regime, c) for regime in REGIMES for c in CANDIDATES}
    assert set(by_session) == {"1", "2"}, (parent, set(by_session))
    assert set(by_session["1"]) == set(by_session["2"]) == expected, parent
    med = {key: {v: stats.median((by_session["1"][key][v], by_session["2"][key][v]))
                 for v in ("t", "e")} for key in expected}
    return med, by_session


def load_data() -> tuple[dict, dict]:
    grid = defaultdict(dict)
    for r in read_csv(CANONICAL):
        parent = r["parent_id"]
        if parent not in CANONICAL_PARENTS:
            continue  # Local fixtures have only one candidate per regime.
        assert r["split"] in ("train", "development")
        key = (r["regime"], r["candidate_id"])
        assert key not in grid[parent]
        assert int(r["n_sessions"]) >= 1
        grid[parent][key] = {"t": float(r["runtime_us_per_launch"]),
                             "e": float(r["board_j_per_launch"])}
    session_grid = {}
    for parent in RAW_SOURCES:
        grid[parent], session_grid[parent] = labels_from_raw(parent, raw_rows(parent))
    for parent in ("train_triton_vector_add", "dev_triton_softmax"):
        rows = [r for r in read_csv(SESSION_SOURCES[parent])
                if r["parent_id"] == parent and "-PRE-" not in r["run_id"]]
        _, session_grid[parent] = labels_from_raw(parent, rows)
    assert set(grid) == EXPECTED_PARENTS, set(grid)
    expected = {(regime, c) for regime in REGIMES for c in CANDIDATES}
    assert all(set(cells) == expected for cells in grid.values())
    assert all(v["t"] > 0 and v["e"] > 0 for cells in grid.values() for v in cells.values())
    return grid, session_grid


def training_priors(grid: dict, heldout: str, regime: str) -> tuple[float, dict[str, float]]:
    others = {p: rows for p, rows in grid.items() if p != heldout}
    mean_power = stats.mean(v["e"] / v["t"] for rows in others.values() for v in rows.values())
    ratios = {"c1": 1.0}
    for c in CANDIDATES[1:]:
        ratios[c] = stats.median(
            (rows[(regime, c)]["e"] / rows[(regime, c)]["t"]) /
            (rows[(regime, "c1")]["e"] / rows[(regime, "c1")]["t"])
            for rows in others.values()
        )
    return mean_power, ratios


def choose(values: dict[str, float]) -> str:
    return min(CANDIDATES, key=lambda c: (values[c], c))


def evaluate(grid: dict, source_sessions: dict | None = None) -> dict:
    predictions = []
    groups = []
    for parent in sorted(source_sessions if source_sessions is not None else grid):
        for regime in REGIMES:
            truth = {c: grid[parent][(regime, c)]["e"] for c in CANDIDATES}
            if source_sessions is None:
                source = {c: grid[parent][(regime, c)] for c in CANDIDATES}
            else:
                source = {c: source_sessions[parent]["1"][(regime, c)] for c in CANDIDATES}
                truth = {c: source_sessions[parent]["2"][(regime, c)]["e"] for c in CANDIDATES}
            mean_power, ratios = training_priors(grid, parent, regime)
            anchor_power = source["c1"]["e"] / source["c1"]["t"]
            estimates = {
                "anchor_constant_power": {c: source[c]["t"] * anchor_power for c in CANDIDATES},
                "training_mean_power": {c: source[c]["t"] * mean_power for c in CANDIDATES},
                "hybrid_ratio": {c: source[c]["t"] * anchor_power * ratios[c] for c in CANDIDATES},
            }
            picked = {name: choose(est) for name, est in estimates.items()}
            picked["fastest"] = choose({c: source[c]["t"] for c in CANDIDATES})
            picked["oracle"] = choose(truth)
            e_best = truth[picked["oracle"]]
            regrets = {name: (truth[c] / e_best - 1) * 100 for name, c in picked.items()}
            groups.append({"parent": parent, "regime": regime, "selected": picked,
                           "regret_pct": regrets, "oracle_energy_j": e_best,
                           "source_anchor_energy_j": source["c1"]["e"]})
            for c in CANDIDATES[1:]:
                predictions.append({"parent": parent, "regime": regime, "candidate": c,
                                    "truth_j": truth[c],
                                    "ape_pct": {name: abs(est[c] / truth[c] - 1) * 100
                                                for name, est in estimates.items()}})
    def summary(name: str) -> dict:
        err = [r["ape_pct"][name] for r in predictions]
        per_parent = {p: stats.median(r["ape_pct"][name] for r in predictions if r["parent"] == p)
                      for p in sorted({r["parent"] for r in predictions})}
        return {"mdape_pct": stats.median(err), "p90_ape_pct": quantile(err, 0.9),
                "per_parent_mdape_pct": per_parent}
    less = sum(g["regret_pct"]["hybrid_ratio"] < g["regret_pct"]["fastest"] - 1e-9 for g in groups)
    more = sum(g["regret_pct"]["hybrid_ratio"] > g["regret_pct"]["fastest"] + 1e-9 for g in groups)
    exceptions = [g for g in groups if g["selected"]["fastest"] != g["selected"]["oracle"]]
    return {
        "n_parents": len({g["parent"] for g in groups}), "n_groups": len(groups),
        "n_unprobed_predictions": len(predictions),
        "energy_error": {name: summary(name) for name in
                         ("anchor_constant_power", "training_mean_power", "hybrid_ratio")},
        "selection": {name: {"mean_regret_pct": stats.mean(g["regret_pct"][name] for g in groups),
                             "median_regret_pct": stats.median(g["regret_pct"][name] for g in groups),
                             "max_regret_pct": max(g["regret_pct"][name] for g in groups),
                             "oracle_count": sum(g["selected"][name] == g["selected"]["oracle"]
                                                 for g in groups)}
                      for name in ("fastest", "anchor_constant_power", "training_mean_power", "hybrid_ratio")},
        "hybrid_vs_fastest": {"better": less, "tie": len(groups) - less - more, "worse": more},
        "fastest_oracle_exceptions": exceptions,
        "groups": groups,
    }


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    x = (len(values) - 1) * q
    lo = int(x)
    return values[lo] * (1 - (x - lo)) + values[min(lo + 1, len(values) - 1)] * (x - lo)


def main() -> None:
    grid, sessions = load_data()
    result = {"protocol": "HYBRID_PROBE_PROTOCOL_2026-09-24.md",
              "primary": evaluate(grid), "cross_session": evaluate(grid, sessions)}
    out = ROOT / "hybrid_probe_diagnostic_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for label in ("primary", "cross_session"):
        r = result[label]
        print(f"{label}: {r['n_parents']} parents, {r['n_groups']} groups, "
              f"{r['n_unprobed_predictions']} unprobed predictions")
        print("hybrid vs fastest:", r["hybrid_vs_fastest"])
        for name, score in r["selection"].items():
            print(f"  {name}: mean regret {score['mean_regret_pct']:.3f}%, "
                  f"max {score['max_regret_pct']:.3f}%, oracle {score['oracle_count']}/{r['n_groups']}")
        for name, score in r["energy_error"].items():
            print(f"  {name}: MdAPE {score['mdape_pct']:.2f}%, P90 {score['p90_ape_pct']:.2f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
