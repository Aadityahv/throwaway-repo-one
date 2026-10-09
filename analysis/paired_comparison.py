#!/usr/bin/env python
"""Paired grouped-bootstrap comparison between two models on identical rows (audit B7).

`shared_evaluator.ModelResult.bootstrap_metric_ci` gives each model an independent confidence
interval. That is not enough to compare two models, and the difference is not pedantic: two
independent CIs can overlap substantially while the *paired* difference is consistently one-signed,
because both models see the same rows and therefore share most of their noise. Comparing overlap of
independent CIs systematically under-detects real differences.

G4's criterion is stated in paired terms -- "M4 improves paired held-out MdAPE and regret versus
fair B0/B1 with grouped-bootstrap uncertainty" -- so this is the machinery that criterion needs.

What this does:
  * per-group (per parent-kernel family) paired deltas, so a single family cannot carry a result;
  * a grouped bootstrap that resamples FAMILIES, not rows, and recomputes the difference on each
    resample -- repeats and configurations of one parent are not independent observations;
  * the sign-consistency of the difference across bootstrap replicates, which is what "survives
    uncertainty" actually means here.

The audit's own warning applies: with five or six parent families this is an effect-size and
uncertainty report, not a significance test, and it is labelled as such. Reads committed
predictions only; touches no GPU.
"""
from __future__ import annotations

import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_evaluator import ModelResult, _percentile


def _metric(result: ModelResult, keys: Sequence[tuple], metric: str) -> float:
    """Recompute a metric on an arbitrary (possibly repeated) key list."""
    subset = ModelResult(
        name=result.name,
        predictions={index: result.predictions[key] for index, key in enumerate(keys)},
        actuals={index: result.actuals[key] for index, key in enumerate(keys)},
    )
    value = subset.metric_summary().get(metric)
    if value is None:
        raise ValueError(f"metric {metric!r} undefined on this subset")
    return float(value)


def _group_of(result: ModelResult, key: tuple, group_columns: Sequence[str]) -> tuple:
    metadata = result.metadata.get(key, dict(key))
    return tuple(metadata.get(column, "<absent>") for column in group_columns)


def paired_delta(
    baseline: ModelResult,
    candidate: ModelResult,
    metric: str = "mape",
    group_columns: Sequence[str] = ("family",),
    samples: int = 2000,
    seed: int = 0,
    lower_is_better: bool = True,
) -> dict:
    """Paired grouped bootstrap of (candidate - baseline) on the metric, over shared rows.

    Returns the point estimate, a percentile CI, the share of bootstrap replicates favouring the
    candidate, and the per-group deltas that produced it.
    """
    shared = sorted(set(baseline.scoreable_keys()) & set(candidate.scoreable_keys()))
    if not shared:
        raise ValueError("the two models share no scoreable rows")

    groups: dict[tuple, list[tuple]] = {}
    for key in shared:
        groups.setdefault(_group_of(baseline, key, group_columns), []).append(key)
    if len(groups) < 2:
        raise ValueError("paired bootstrap needs at least two independent groups")

    point = _metric(candidate, shared, metric) - _metric(baseline, shared, metric)

    per_group = {}
    for group, keys in sorted(groups.items()):
        try:
            per_group[group] = (_metric(candidate, keys, metric)
                                - _metric(baseline, keys, metric))
        except (ValueError, ZeroDivisionError):
            per_group[group] = float("nan")

    grouped_keys = list(groups.values())
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        keys = [key for _ in grouped_keys for key in rng.choice(grouped_keys)]
        try:
            deltas.append(_metric(candidate, keys, metric) - _metric(baseline, keys, metric))
        except (ValueError, ZeroDivisionError):
            continue
    if not deltas:
        raise ValueError("every bootstrap replicate was undefined")

    favourable = sum(1 for d in deltas if (d < 0) if lower_is_better) if lower_is_better \
        else sum(1 for d in deltas if d > 0)
    low, high = _percentile(deltas, 0.025), _percentile(deltas, 0.975)
    return {
        "metric": metric,
        "point": point,
        "ci": (low, high),
        "excludes_zero": (low > 0.0) or (high < 0.0),
        "share_favouring_candidate": favourable / len(deltas),
        "per_group": per_group,
        "n_groups": len(groups),
        "n_rows": len(shared),
        "replicates": len(deltas),
    }


def report(baseline: ModelResult, candidate: ModelResult, metrics: Sequence[str] = ("mape",),
           group_columns: Sequence[str] = ("family",), samples: int = 2000) -> None:
    print(f"{candidate.name}  vs  {baseline.name}")
    for metric in metrics:
        outcome = paired_delta(baseline, candidate, metric=metric,
                               group_columns=group_columns, samples=samples)
        low, high = outcome["ci"]
        verdict = "excludes zero" if outcome["excludes_zero"] else "includes zero"
        print(f"\n  {metric}: delta = {outcome['point']:+.4f}  "
              f"95% CI [{low:+.4f}, {high:+.4f}]  ({verdict})")
        print(f"    grouped bootstrap over {outcome['n_groups']} parent families, "
              f"{outcome['n_rows']} shared rows, {outcome['replicates']} replicates")
        print(f"    replicates favouring {candidate.name}: "
              f"{outcome['share_favouring_candidate']*100:.1f}%")
        print("    per-family delta (negative favours the candidate):")
        for group, delta in outcome["per_group"].items():
            label = "/".join(str(part) for part in group)
            print(f"      {label:<12} {delta:+.4f}")
    print("\n  With this few parent families these are effect sizes and uncertainty, not a")
    print("  significance test. Report them as such (audit B7).")


def _self_test() -> None:
    """Mechanics check on synthetic results, including the identical-models degenerate case."""
    keys = [(("family", f), ("candidate", c)) for f in ("a", "b", "c") for c in ("l1", "l2")]
    actuals = {key: 10.0 + index for index, key in enumerate(keys)}
    metadata = {key: dict(key) for key in keys}

    exact = ModelResult("exact", dict(actuals), dict(actuals), metadata)
    # Two identical models must give a zero point estimate and a CI containing zero. This is the
    # degenerate case a paired test has to get right -- an unpaired one would not.
    same = paired_delta(exact, ModelResult("exact copy", dict(actuals), dict(actuals), metadata),
                        samples=200)
    assert abs(same["point"]) < 1e-12, same["point"]
    assert same["ci"][0] <= 0.0 <= same["ci"][1]
    assert not same["excludes_zero"]

    # A uniformly worse model must produce a positive delta on an error metric, with the CI
    # excluding zero and every replicate agreeing.
    worse = ModelResult("worse", {key: value * 2.0 for key, value in actuals.items()},
                        dict(actuals), metadata)
    gap = paired_delta(exact, worse, samples=200)
    assert gap["point"] > 0.0
    assert gap["excludes_zero"], gap["ci"]
    assert gap["share_favouring_candidate"] == 0.0

    # And the reverse direction is symmetric.
    gain = paired_delta(worse, exact, samples=200)
    assert gain["point"] < 0.0 and gain["share_favouring_candidate"] == 1.0
    print("paired_comparison self-test passed")


if __name__ == "__main__":
    _self_test()
