#!/usr/bin/env python
"""Shared, leak-resistant evaluator for the full-paper static-energy comparison.

The deployed task is predicting whole-kernel dynamic energy for one logical kernel launch. Its one
canonical target is dynamic_energy_j_per_launch_mean from analysis/energy_features.csv. The raw
dynamic_energy_j_mean label is energy for an NVML measurement batch whose launch count differs by
candidate; it is provenance, not a prediction target. energy_per_load_pj_mean remains in the feature
tables for mechanism plots and legacy diagnostic scripts only.

This module is intentionally model-agnostic. It provides:
- variant-safe row identity and common-row enforcement;
- family-held-out fitting;
- out-of-fold physical, relative, log-scale, bias, and ranking metrics; and
- deterministic bootstrap confidence intervals over declared independent groups.

It cannot prove a caller's features are inference-legal. Each model must provide its input
provenance separately, and static models must never pass target-run measurements into a predictor.

Metric suite (taxonomy §5.3): MAE/RMSE/nRMSE (range-normalized), MAPE/median/P90/symmetric APE,
log-space R², signed bias, geometric multiplicative error, energy-tercile stratification, Spearman
and Kendall rank correlation, top-1/top-3 selection accuracy, energy regret, prediction-interval
coverage, worst family/architecture, tail P95. Percentage metrics are fractions, not 0–100 values.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = REPO_ROOT / "analysis" / "energy_features.csv"

# Full-paper deployed label. Do not change this in individual model scripts.
LABEL_COLUMN = "dynamic_energy_j_per_launch_mean"
MECHANISM_LABEL_COLUMN = "energy_per_load_pj_mean"

# A row key must distinguish configurations that share a family and cache candidate. The fields are
# all compile-/launch-/problem-level identifiers; label and target-run measurements are excluded.
ROW_ID_COLUMNS = (
    "family",
    "candidate",
    "layout",
    "access_width_bytes",
    "stride",
    "working_set_bytes",
    "temporal_reach_bytes",
    "occupancy_blocks_per_sm",
    "occupancy_multiple",
    "blocks",
    "threads",
    "step_warps",
)

DEFAULT_NUMERIC_COLUMNS = (
    LABEL_COLUMN,
    MECHANISM_LABEL_COLUMN,
    "energy_per_load_pj_sd",
    "dynamic_energy_j_mean",
    "dynamic_energy_j_per_launch_sd",
    "temporal_reach_bytes",
    "indirection_depth",
    "working_set_bytes",
    "predicted_sectors_per_warp",
    "measured_sectors_per_warp",
    "loads_per_repeat",
    "stores_per_repeat",
    "access_width_bytes",
    "stride",
    "occupancy_blocks_per_sm",
    "occupancy_multiple",
    "blocks",
    "threads",
    "step_warps",
    "iterations",
    "batches",
    "sass_global_ld_count",
    "sass_global_st_count",
    "kernel_time_s_mean",
    "kernel_time_s_per_launch_mean",
    "measurement_launches",
    "launches",
    "executed_global_loads",
    "n_repeats",
)


def row_cv_percent(
    row: Mapping[str, object],
    mean_column: str = LABEL_COLUMN,
    sd_column: str = "dynamic_energy_j_per_launch_sd",
) -> float | None:
    """Repeat CV (%) of the deployed label, or None when mean/sd are missing or the mean is 0."""
    try:
        mean = float(row[mean_column])  # type: ignore[arg-type]
        sd = float(row[sd_column])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None
    if mean == 0:
        return None
    return abs(sd / mean) * 100.0


def cv_qualifier(threshold: float = 25.0) -> Callable[[dict], bool]:
    """Row predicate keeping the label-quality-qualified set (taxonomy §5.4).

    Rows whose repeat CV is at or below `threshold` pass. Rows with undefined CV
    fail closed: they stay in the full set but leave the qualified set, since an
    unmeasured quality cannot be asserted.
    """
    return lambda row: (
        (cv := row_cv_percent(row)) is not None and cv <= threshold
    )


def load_rows(
    path: Path = FEATURES_CSV,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
    qualifier: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Read a feature/label table, coercing the available requested numeric columns.

    With `qualifier` (e.g. `cv_qualifier()`), only rows passing the predicate are
    returned — the full/qualified row-set switch. Callers report both sets.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    present = set(rows[0])
    for row in rows:
        for column in numeric_columns:
            if column in present and row[column] != "":
                row[column] = float(row[column])
    if LABEL_COLUMN not in present:
        raise ValueError(f"{path} does not contain the required deployed label {LABEL_COLUMN}")
    if qualifier is not None:
        rows = [row for row in rows if qualifier(row)]
        if not rows:
            raise ValueError(f"{path}: qualifier excluded every row")
    return rows


def partition_full_qualified(
    rows: list[dict], threshold: float = 25.0
) -> tuple[list[dict], list[dict]]:
    """Split loaded rows into (full, qualified) sets per the label-quality policy.

    The full set is every row; the qualified set is rows at or below `threshold`
    % repeat CV. No row is ever dropped from the full set.
    """
    predicate = cv_qualifier(threshold)
    return list(rows), [row for row in rows if predicate(row)]


def _canonical_value(value: object) -> object:
    """Keep equivalent integer-looking numeric values from producing different key values."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def row_key(row: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Stable identity for a family/candidate/configuration row.

    Missing fields are represented explicitly. Thus a table that has no width column cannot
    accidentally compare equal to one that specifies a width, and same family/candidate variants
    cannot overwrite one another in a prediction dictionary.
    """
    return tuple((column, _canonical_value(row.get(column, "<absent>"))) for column in ROW_ID_COLUMNS)


def relative_error(predicted: float, actual: float) -> float:
    if actual <= 0:
        raise ValueError(f"relative error requires positive actual energy, got {actual}")
    return abs(predicted - actual) / actual


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, with rank 1 assigned to the smallest value."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end + 1) / 2.0
        for ordered_index in range(index, end + 1):
            ranks[order[ordered_index]] = rank
        index = end + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return None if denominator == 0 else numerator / denominator


def _kendall(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall tau-b rank correlation (stdlib only). Ties in either input are corrected."""
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            elif dx != 0:
                tied_y += 1
            elif dy != 0:
                tied_x += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


@dataclass
class ModelResult:
    """Predictions and observed whole-kernel energy keyed by variant-safe row identity."""

    name: str
    predictions: dict[tuple, float]
    actuals: dict[tuple, float] = field(default_factory=dict)
    metadata: dict[tuple, dict] = field(default_factory=dict)

    def keys(self) -> set[tuple]:
        return set(self.predictions)

    def scoreable_keys(self) -> set[tuple]:
        return set(self.predictions) & set(self.actuals)

    def _pairs(self) -> list[tuple[tuple, float, float]]:
        pairs = []
        for key in sorted(self.scoreable_keys(), key=repr):
            predicted = self.predictions[key]
            actual = self.actuals[key]
            if predicted <= 0 or actual <= 0:
                raise ValueError(
                    f"{self.name}: positive energy required for key={key}; "
                    f"predicted={predicted}, actual={actual}"
                )
            pairs.append((key, predicted, actual))
        if not pairs:
            raise ValueError(f"{self.name}: no scoreable rows")
        return pairs

    def restricted_to(self, keys: set[tuple]) -> "ModelResult":
        return ModelResult(
            name=self.name,
            predictions={key: value for key, value in self.predictions.items() if key in keys},
            actuals={key: value for key, value in self.actuals.items() if key in keys},
            metadata={key: value for key, value in self.metadata.items() if key in keys},
        )

    def per_row_errors(self) -> dict[tuple, float]:
        return {
            key: relative_error(predicted, actual)
            for key, predicted, actual in self._pairs()
        }

    def metric_summary(self) -> dict[str, float | None]:
        """Out-of-fold metrics. Percentage metrics are returned as fractions, not 0–100 values."""
        pairs = self._pairs()
        predictions = [predicted for _, predicted, _ in pairs]
        actuals = [actual for _, _, actual in pairs]
        residuals = [predicted - actual for predicted, actual in zip(predictions, actuals)]
        absolute_errors = [abs(value) for value in residuals]
        ape = [abs(value) / actual for value, actual in zip(residuals, actuals)]
        signed = [value / actual for value, actual in zip(residuals, actuals)]
        log_absolute = [
            abs(math.log(predicted / actual)) for predicted, actual in zip(predictions, actuals)
        ]
        # Symmetric APE stays bounded for near-floor labels where plain APE explodes.
        smape = [
            2.0 * abs(value) / (predicted + actual)
            for value, predicted, actual in zip(residuals, predictions, actuals)
        ]
        actual_mean = statistics.mean(actuals)
        total_variance = sum((actual - actual_mean) ** 2 for actual in actuals)
        r_squared = (
            None
            if total_variance == 0
            else 1.0 - sum(value**2 for value in residuals) / total_variance
        )
        rmse = math.sqrt(statistics.mean([value**2 for value in residuals]))
        label_range = max(actuals) - min(actuals)
        # Log-space R²: fit quality in multiplicative (order-of-magnitude) terms.
        log_predictions = [math.log(predicted) for predicted in predictions]
        log_actuals = [math.log(actual) for actual in actuals]
        log_mean = statistics.mean(log_actuals)
        log_total = sum((value - log_mean) ** 2 for value in log_actuals)
        log_residuals = [
            predicted - actual for predicted, actual in zip(log_predictions, log_actuals)
        ]
        log_r_squared = (
            None
            if log_total == 0
            else 1.0 - sum(value**2 for value in log_residuals) / log_total
        )
        return {
            "n": float(len(pairs)),
            "mae_j": statistics.mean(absolute_errors),
            "rmse_j": rmse,
            "nrmse_range": None if label_range == 0 else rmse / label_range,
            "mape": statistics.mean(ape),
            "median_ape": statistics.median(ape),
            "p90_ape": _percentile(ape, 0.90),
            "smape": statistics.mean(smape),
            "signed_percentage_bias": statistics.mean(signed),
            "log_mae": statistics.mean(log_absolute),
            "geometric_multiplicative_error": math.exp(statistics.mean(log_absolute)),
            "r_squared": r_squared,
            "log_r_squared": log_r_squared,
        }

    def mean_relative_error(self) -> float:
        """Compatibility alias for legacy diagnostic scripts; equals MAPE as a fraction."""
        return float(self.metric_summary()["mape"])

    def per_family_errors(self) -> dict[str, float]:
        by_family: dict[str, list[float]] = {}
        for key, error in self.per_row_errors().items():
            family = self.metadata.get(key, {}).get("family")
            if family is None:
                family = dict(key).get("family", "<unknown>")
            by_family.setdefault(str(family), []).append(error)
        return {family: statistics.mean(errors) for family, errors in by_family.items()}

    def worst_family(self) -> tuple[str, float]:
        per_family = self.per_family_errors()
        worst = max(per_family, key=per_family.get)
        return worst, per_family[worst]

    def worst_group(self, column: str = "family") -> tuple[str, float]:
        """Worst mean APE over any grouping column (family, architecture, ...).

        `worst_family()` is retained as the named shorthand. Groups whose rows all
        lack the column collapse to "<unknown>" rather than failing.
        """
        by_group: dict[str, list[float]] = {}
        for key, error in self.per_row_errors().items():
            group = self.metadata.get(key, {}).get(column)
            if group is None:
                group = dict(key).get(column, "<unknown>")
            by_group.setdefault(str(group), []).append(error)
        if not by_group:
            raise ValueError(f"{self.name}: no scoreable rows for worst_group")
        means = {group: statistics.mean(errors) for group, errors in by_group.items()}
        worst = max(means, key=means.get)
        return worst, means[worst]

    def tail_ape(self, percentile: float = 0.95) -> float:
        """Tail APE at the given percentile (default P95). Near-floor rows surface here."""
        return _percentile(list(self.per_row_errors().values()), percentile)

    def error_by_tercile(self) -> dict[str, float | None]:
        """Mean APE in low/mid/high thirds of observed energy (magnitude stratification).

        Near-floor labels concentrate in the low tercile by construction; they are
        reported, never dropped. Groups left empty by tiny row counts report None.
        """
        ordered = sorted(self._pairs(), key=lambda pair: pair[2])
        n = len(ordered)
        cuts = (n // 3, 2 * n // 3)
        groups = {
            "low": ordered[: cuts[0]],
            "mid": ordered[cuts[0] : cuts[1]],
            "high": ordered[cuts[1] :],
        }
        result: dict[str, float | None] = {}
        for name, members in groups.items():
            if not members:
                result[name] = None
            else:
                result[name] = statistics.mean(
                    abs(predicted - actual) / actual for _, predicted, actual in members
                )
        return result

    def ranking_metrics(
        self,
        group_columns: Sequence[str] = ("family",),
        top_k: Sequence[int] = (1, 3),
    ) -> dict[str, float | None]:
        """Energy-selection quality across declared candidate sets.

        `group_columns` declares the candidate set (e.g. configurations of one
        kernel). Each group needs at least two configurations. The chosen
        configuration minimizes predicted energy; regret is measured using its
        observed energy. Rank correlations are reported across all scoreable rows
        and are intentionally separate from top-k correctness.
        """
        groups: dict[tuple, list[tuple[tuple, float, float]]] = {}
        for pair in self._pairs():
            key = pair[0]
            metadata = self.metadata.get(key, dict(key))
            group = tuple(metadata.get(column, "<absent>") for column in group_columns)
            groups.setdefault(group, []).append(pair)

        eligible = [pairs for pairs in groups.values() if len(pairs) >= 2]
        base: dict[str, float | None] = {
            "ranking_groups": 0.0,
            "top1_accuracy": None,
            "mean_energy_regret": None,
            "p90_energy_regret": None,
            "spearman_rho": None,
            "kendall_tau": None,
        }
        for k in top_k:
            if k != 1:
                base[f"top{k}_accuracy"] = None
        if not eligible:
            return base

        hits: dict[int, int] = {k: 0 for k in top_k}
        regrets = []
        all_predicted = []
        all_actual = []
        for pairs in eligible:
            ranked = sorted(pairs, key=lambda pair: pair[1])
            best_actual = min(pair[2] for pair in pairs)
            for k in top_k:
                top_predicted = ranked[:k] if k <= len(ranked) else ranked
                if any(
                    math.isclose(pair[2], best_actual, rel_tol=1e-12, abs_tol=0.0)
                    for pair in top_predicted
                ):
                    hits[k] += 1
            chosen_actual = ranked[0][2]
            regrets.append((chosen_actual - best_actual) / best_actual)
            all_predicted.extend(pair[1] for pair in pairs)
            all_actual.extend(pair[2] for pair in pairs)

        result: dict[str, float | None] = {
            "ranking_groups": float(len(eligible)),
            "top1_accuracy": (hits[1] / len(eligible)) if 1 in hits else None,
            "mean_energy_regret": statistics.mean(regrets),
            "p90_energy_regret": _percentile(regrets, 0.90),
            "spearman_rho": _pearson(
                _average_ranks(all_predicted), _average_ranks(all_actual)
            ),
            "kendall_tau": _kendall(all_predicted, all_actual),
        }
        for k in top_k:
            if k != 1:
                result[f"top{k}_accuracy"] = hits[k] / len(eligible)
        return result

    def interval_coverage(
        self, intervals: Mapping[tuple, tuple[float, float, float]]
    ) -> dict[str, float]:
        """Prediction-interval coverage and width (taxonomy §5.3: 80% nominal).

        `intervals` maps each scoreable row key to a (lo, median, hi) triple in J.
        Missing keys and negative widths fail loudly. Coverage is the fraction of
        observed energies inside [lo, hi]; width is reported absolute and relative.
        """
        keys = self.scoreable_keys()
        missing = [key for key in keys if key not in intervals]
        if missing:
            raise ValueError(
                f"{self.name}: interval_coverage missing {len(missing)} rows, "
                f"e.g. {missing[0]!r}"
            )
        covered = 0
        widths: list[float] = []
        relative_widths: list[float] = []
        for key in keys:
            low, _, high = intervals[key]
            if high < low:
                raise ValueError(f"{self.name}: negative interval width for key={key}")
            actual = self.actuals[key]
            if low <= actual <= high:
                covered += 1
            widths.append(high - low)
            relative_widths.append((high - low) / actual)
        return {
            "n": float(len(keys)),
            "coverage": covered / len(keys),
            "mean_width_j": statistics.mean(widths),
            "mean_relative_width": statistics.mean(relative_widths),
        }

    def bootstrap_metric_ci(
        self,
        metric: str,
        group_columns: Sequence[str] = ("family",),
        samples: int = 2000,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Percentile 95% CI by resampling declared independent groups, not raw repeats."""
        if samples < 100:
            raise ValueError("bootstrap samples must be at least 100")
        groups: dict[tuple, list[tuple]] = {}
        for key in self.scoreable_keys():
            metadata = self.metadata.get(key, dict(key))
            group = tuple(metadata.get(column, "<absent>") for column in group_columns)
            groups.setdefault(group, []).append(key)
        if len(groups) < 2:
            raise ValueError("bootstrap needs at least two independent groups")

        grouped_keys = list(groups.values())
        rng = random.Random(seed)
        values = []
        for _ in range(samples):
            keys = [key for _ in grouped_keys for key in rng.choice(grouped_keys)]
            sampled = ModelResult(
                name=self.name,
                predictions={index: self.predictions[key] for index, key in enumerate(keys)},
                actuals={index: self.actuals[key] for index, key in enumerate(keys)},
            )
            result = sampled.metric_summary().get(metric)
            if result is None:
                raise ValueError(f"metric {metric} is undefined for a bootstrap sample")
            values.append(float(result))
        return _percentile(values, 0.025), _percentile(values, 0.975)


def restrict_to_common_keys(*results: ModelResult) -> tuple[list[ModelResult], set[tuple]]:
    """Restrict results to an identical, explicitly stated scoreable row set."""
    if not results:
        raise ValueError("restrict_to_common_keys needs at least one result")
    common = results[0].scoreable_keys()
    for result in results[1:]:
        common &= result.scoreable_keys()
    if not common:
        raise ValueError("no scoreable row is common to every supplied result")
    return [result.restricted_to(common) for result in results], common


def paper_fold_of(row: Mapping[str, object]) -> str:
    """Paper fold definition (taxonomy §2.2): vector rows score inside the tier group.

    The six independent folds are tier (+vector), store, gather, stencil, intensity,
    reduction. Vector shares tier's addressing and parent lineage, so it is never an
    independent held-out family. The intensity (R1-generator) family is first-class.
    """
    family = str(row.get("family", "<unknown>"))
    return "tier" if family == "vector" else family


def leave_one_group_out(
    rows: list[dict],
    fit_fn: Callable[[list[dict]], object],
    predict_fn: Callable[[object, dict], float],
    group_of: Callable[[Mapping[str, object]], str],
    label_column: str = LABEL_COLUMN,
    name: str = "model",
) -> ModelResult:
    """Fit once per held-out group defined by `group_of`. Groups never straddle train/test.

    This is the generic splitter behind `leave_one_family_out`. Callers define
    within-family configuration hold-outs (interpolation vs extrapolation) by passing
    their own `group_of`; the interpolation/extrapolation separation itself is the
    caller's declared split, recorded in metadata.
    """
    groups = sorted({group_of(row) for row in rows})
    if len(groups) < 2:
        raise ValueError("leave_one_group_out needs at least two groups")

    predictions: dict[tuple, float] = {}
    actuals: dict[tuple, float] = {}
    metadata: dict[tuple, dict] = {}
    for held_out in groups:
        train = [row for row in rows if group_of(row) != held_out]
        test = [row for row in rows if group_of(row) == held_out]
        params = fit_fn(train)
        for row in test:
            key = row_key(row)
            if key in predictions:
                raise ValueError(f"duplicate variant-safe row key: {key}")
            predictions[key] = predict_fn(params, row)
            actuals[key] = float(row[label_column])
            metadata[key] = dict(row)
    return ModelResult(name=name, predictions=predictions, actuals=actuals, metadata=metadata)


def leave_one_family_out(
    rows: list[dict],
    fit_fn: Callable[[list[dict]], object],
    predict_fn: Callable[[object, dict], float],
    label_column: str = LABEL_COLUMN,
    name: str = "model",
    fold_of: Callable[[Mapping[str, object]], str] | None = None,
) -> ModelResult:
    """Fit once per held-out family. Family is the independent OOD unit, never a random row.

    With `fold_of` (e.g. `paper_fold_of`), folds follow the paper definition instead
    of the raw family column. Defaults to raw family for backward compatibility.
    """
    return leave_one_group_out(
        rows,
        fit_fn,
        predict_fn,
        fold_of or (lambda row: str(row["family"])),
        label_column,
        name,
    )


def _format_optional(value: float | None, spec: str) -> str:
    """Format a metric that may be undefined (None) without raising."""
    return "undefined" if value is None else format(value, spec)


def print_comparison(
    *results: ModelResult,
    set_label: str = "",
    timings: Mapping[tuple, float] | None = None,
) -> None:
    """Print a common-row, whole-energy comparison with per-family and ranking evidence.

    `set_label` names the row set (e.g. "full" vs "qualified") so callers report
    both label-quality sets. `timings` optionally maps row keys to per-kernel
    static feature-extraction + inference seconds; mean/P90 are reported.
    """
    restricted, common_keys = restrict_to_common_keys(*results)
    families = sorted(
        {
            result.metadata.get(key, dict(key)).get("family", "<unknown>")
            for result in restricted
            for key in common_keys
        }
    )
    title = f"Common row set: {len(common_keys)} configurations; families: {', '.join(families)}"
    if set_label:
        title += f"; set: {set_label}"
    print(title)
    for result in restricted:
        metrics = result.metric_summary()
        print(f"\n{result.name}:")
        print(
            "  MAE={mae_j:.6g} J  RMSE={rmse_j:.6g} J  "
            "MAPE={mape:.1%}  MdAPE={median_ape:.1%}  P90={p90_ape:.1%}  "
            "bias={signed_percentage_bias:+.1%}".format(**metrics)
        )
        print(
            "  sMAPE={smape:.1%}  nRMSE={nrmse}  "
            "log-MAE={log_mae:.4f}  geometric-error={geometric:.3f}x  "
            "R²={r2}  log-R²={logr2}".format(
                smape=metrics["smape"],
                nrmse=_format_optional(metrics["nrmse_range"], ".4f"),
                log_mae=metrics["log_mae"],
                geometric=metrics["geometric_multiplicative_error"],
                r2=_format_optional(metrics["r_squared"], ".4f"),
                logr2=_format_optional(metrics["log_r_squared"], ".4f"),
            )
        )
        worst_family, worst_error = result.worst_family()
        print(f"  worst family: {worst_family} ({worst_error:.1%} MAPE)")
        for family, error in sorted(result.per_family_errors().items()):
            print(f"    {family}: {error:.1%} MAPE")
        terciles = result.error_by_tercile()
        print(
            "  terciles (low/mid/high): {}/{}/{}  tail-P95={:.1%}".format(
                _format_optional(terciles["low"], ".1%"),
                _format_optional(terciles["mid"], ".1%"),
                _format_optional(terciles["high"], ".1%"),
                result.tail_ape(),
            )
        )
        ranking = result.ranking_metrics()
        if ranking["top1_accuracy"] is not None:
            line = (
                "  ranking: groups={ranking_groups:.0f}, top-1={top1_accuracy:.1%}, "
                "mean-regret={mean_energy_regret:.1%}, P90-regret={p90_energy_regret:.1%}, "
                "Spearman={spearman}, Kendall={kendall}".format(
                    ranking_groups=ranking["ranking_groups"],
                    top1_accuracy=ranking["top1_accuracy"],
                    mean_energy_regret=ranking["mean_energy_regret"],
                    p90_energy_regret=ranking["p90_energy_regret"],
                    spearman=_format_optional(ranking["spearman_rho"], ".3f"),
                    kendall=_format_optional(ranking["kendall_tau"], ".3f"),
                )
            )
            top3 = ranking.get("top3_accuracy")
            if top3 is not None:
                line += f", top-3={top3:.1%}"
            print(line)
    if timings is not None:
        timed = [timings[key] for key in common_keys if key in timings]
        if timed:
            print(
                f"\nTiming (static extraction + inference per kernel over {len(timed)} rows): "
                f"mean={statistics.mean(timed):.4g} s  P90={_percentile(timed, 0.90):.4g} s"
            )


def _self_test() -> None:
    """Deterministic synthetic check for variant identity, metrics, ranking, and bootstrap."""
    rows = [
        {"family": "a", "candidate": "l1", "layout": "strided", "stride": 1},
        {"family": "a", "candidate": "l1", "layout": "coalesced", "stride": 1},
        {"family": "a", "candidate": "l2", "layout": "strided", "stride": 1},
        {"family": "b", "candidate": "l1", "layout": "strided", "stride": 1},
        {"family": "b", "candidate": "l2", "layout": "strided", "stride": 1},
    ]
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("variant-safe key collapsed distinct configurations")

    actual = [10.0, 20.0, 30.0, 40.0, 80.0]
    predicted = [10.0, 10.0, 60.0, 80.0, 40.0]
    result = ModelResult(
        name="synthetic",
        predictions=dict(zip(keys, predicted)),
        actuals=dict(zip(keys, actual)),
        metadata=dict(zip(keys, rows)),
    )
    metrics = result.metric_summary()
    if not math.isclose(float(metrics["mae_j"]), 24.0):
        raise AssertionError(f"unexpected MAE: {metrics['mae_j']}")
    if not math.isclose(float(metrics["mape"]), 0.6):
        raise AssertionError(f"unexpected MAPE: {metrics['mape']}")
    # Hand-computed: sMAPE rows are 0, 2/3, 2/3, 2/3, 2/3 -> mean 8/15.
    if not math.isclose(float(metrics["smape"]), 8.0 / 15.0, rel_tol=1e-9):
        raise AssertionError(f"unexpected sMAPE: {metrics['smape']}")
    # Hand-computed: RMSE sqrt(840), label range 70.
    if not math.isclose(
        float(metrics["nrmse_range"]), math.sqrt(840.0) / 70.0, rel_tol=1e-9
    ):
        raise AssertionError(f"unexpected nRMSE: {metrics['nrmse_range']}")
    # Hand-computed on natural logs: SS_res ~= 1.92181, SS_tot ~= 2.40488.
    if not math.isclose(float(metrics["log_r_squared"]), 0.2009, abs_tol=1e-4):
        raise AssertionError(f"unexpected log-R²: {metrics['log_r_squared']}")
    ranking = result.ranking_metrics()
    if ranking["ranking_groups"] != 2.0:
        raise AssertionError(f"unexpected ranking groups: {ranking}")
    # Hand-computed Kendall tau-b: C=7, D=2, ties in x=1 -> 5/sqrt(90).
    if not math.isclose(float(ranking["kendall_tau"]), 5.0 / math.sqrt(90.0), rel_tol=1e-9):
        raise AssertionError(f"unexpected Kendall: {ranking['kendall_tau']}")
    if ranking["top3_accuracy"] != 1.0:
        raise AssertionError(f"unexpected top-3: {ranking['top3_accuracy']}")
    # Terciles by actual (10 | 20,30 | 40,80): APE 0 | 0.5,1.0 | 1.0,0.5.
    terciles = result.error_by_tercile()
    if terciles != {"low": 0.0, "mid": 0.75, "high": 0.75}:
        raise AssertionError(f"unexpected terciles: {terciles}")
    if not math.isclose(result.tail_ape(), 1.0):
        raise AssertionError(f"unexpected tail P95: {result.tail_ape()}")
    if result.worst_group("family") != result.worst_family():
        raise AssertionError("worst_group(family) disagrees with worst_family()")
    # Interval coverage: (0.5a, p, 2a) covers everything; width 1.5a per row.
    full_intervals = {key: (0.5 * act, pred, 2.0 * act) for key, pred, act in zip(keys, predicted, actual)}
    coverage = result.interval_coverage(full_intervals)
    if coverage["coverage"] != 1.0 or not math.isclose(coverage["mean_width_j"], 54.0):
        raise AssertionError(f"unexpected coverage: {coverage}")
    if not math.isclose(coverage["mean_relative_width"], 1.5):
        raise AssertionError(f"unexpected relative width: {coverage}")
    missed = dict(full_intervals)
    missed[keys[4]] = (0.5 * actual[4], predicted[4], 0.6 * actual[4])
    if result.interval_coverage(missed)["coverage"] != 0.8:
        raise AssertionError("miss case should give coverage 0.8")
    try:
        result.interval_coverage({keys[0]: (1.0, 1.0, 0.0)})
        raise AssertionError("negative interval width should fail")
    except ValueError:
        pass
    # Label-quality switch: CVs 0/10/20/30/40% at threshold 25 keep the first three.
    quality_rows = [
        {"dynamic_energy_j_per_launch_mean": 100.0, "dynamic_energy_j_per_launch_sd": cv}
        for cv in (0.0, 10.0, 20.0, 30.0, 40.0)
    ]
    full, qualified = partition_full_qualified(quality_rows, threshold=25.0)
    if len(full) != 5 or len(qualified) != 3:
        raise AssertionError("full/qualified partition wrong")
    if row_cv_percent({"dynamic_energy_j_per_launch_mean": 0.0, "dynamic_energy_j_per_launch_sd": 1.0}) is not None:
        raise AssertionError("zero-mean CV should be None")
    # Paper folds: vector scores inside the tier group; raw-family default unchanged.
    if paper_fold_of({"family": "vector"}) != "tier":
        raise AssertionError("vector must fold into tier")
    if paper_fold_of({"family": "gather"}) != "gather":
        raise AssertionError("non-vector families keep their fold")
    fold_rows = [
        {"family": family, "candidate": candidate, LABEL_COLUMN: float(10 * (index + 1))}
        for index, (family, candidate) in enumerate(
            [("a", "l1"), ("a", "l2"), ("b", "l1"), ("b", "l2")]
        )
    ]
    fit = lambda train: statistics.mean(float(r[LABEL_COLUMN]) for r in train)
    predict = lambda params, row: params
    default_result = leave_one_family_out(fold_rows, fit, predict, name="fold-check")
    paper_result = leave_one_family_out(
        fold_rows, fit, predict, name="fold-check", fold_of=paper_fold_of
    )
    if default_result.predictions != paper_result.predictions:
        raise AssertionError("paper folds should match raw families when no vector rows exist")
    lower, upper = result.bootstrap_metric_ci("mape", samples=100, seed=1)
    if not (0 <= lower <= upper):
        raise AssertionError(f"invalid bootstrap interval: {(lower, upper)}")
    print("shared_evaluator self-test passed")


if __name__ == "__main__":
    _self_test()
