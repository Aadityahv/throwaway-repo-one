#!/usr/bin/env python3
"""Per-window gate of the H100 energy measurements near the power cap (CPU only; reads the files the energy stage wrote, never changes them).

    python window_gate.py --energy-dir <dir with application_energy_raw.csv> [--energy-dir ...] --cap-w 700 --out window_gate_<profile>.json

The power-sensor update test of the H100 (h100_power_sensor_test/runs/h100_j16) validated 40 s unpadded windows within 1.5% of the reference at a steady plateau of about 282 W, i.e. at
about 40% of the 700 W limit (its own field max_plateau_fraction_of_power_limit). It says nothing about a window near the limit, where the board may throttle its clocks and the sensor may
saturate. Tensor-core and attention kernels run near the limit on Blackwell (five of eight cells at 583 to 592 W under a 600 W limit). This gate is how measurability near the cap is
confirmed or refused FOR EACH WINDOW, with rules fixed here before any H100 energy window exists and never changed afterwards:

  record, per window: mean board power (the harness's energy over its counted interval), its fraction of the enforced limit, the graphics clock minimum / median / maximum inside the window and
  the maximum of the whole trace, the longest gap between power samples inside the window, temperature at the start and end, the number of samples.
  rule 1 (cap rule, the calibrator's admission rule and the scorer's exclusion rule): mean power above 98.5% of the enforced limit -> `above_cap_rule`: excluded from the energy scoring of
         every method alike (the raw energy stays in the files, listed by score_h100.py).
  rule 2 (reported, not an exclusion): below cap = mean power under 95% of the limit; between 95% and 98.5% = `near_cap` (scored, reported as its own band).
  rule 3 (reported, not an exclusion): `clock_throttled` = median graphics clock inside the window below 90% of the maximum clock of the whole trace (the calibrator's own check); a gap
         between power samples of 0.1 s or more is flagged `sample_gap`.
Exit code 0 always when the files could be read; the verdicts are data for the scorer and the paper. A window whose trace is missing is listed as `no_trace` (its power fraction is still taken
from the raw row) and is never silently admitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ADMIT_FRACTION_OF_CAP = 0.985
BELOW_CAP_FRACTION = 0.95
CLOCK_MEDIAN_MIN_FRACTION = 0.9
MAX_SAMPLE_GAP_S = 0.1
RULES = dict(above_cap_rule_fraction=ADMIT_FRACTION_OF_CAP, below_cap_fraction=BELOW_CAP_FRACTION, clock_median_min_fraction_of_trace_max=CLOCK_MEDIAN_MIN_FRACTION, max_sample_gap_s=MAX_SAMPLE_GAP_S)


def _median(v):
    v = sorted(v)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def read_trace(trace_dir):
    p = Path(trace_dir)
    samples = list(csv.DictReader((p / "samples.csv").open(newline="")))
    win = next(csv.DictReader((p / "windows.csv").open(newline="")))
    return samples, int(win["host_begin_monotonic_ns"]), int(win["host_end_monotonic_ns"])


def trace_stats(samples, t0, t1):
    inside = [r for r in samples if t0 <= int(r["monotonic_ns"]) <= t1]
    if not inside:
        return None
    clk = [int(r["graphics_clock_mhz"]) for r in inside]
    ts = [int(r["monotonic_ns"]) for r in inside]
    gaps = [(b - a) / 1e9 for a, b in zip(ts, ts[1:])]
    return dict(samples=len(inside), clock_min_mhz=min(clk), clock_median_mhz=_median(clk), clock_max_trace_mhz=max(int(r["graphics_clock_mhz"]) for r in samples),
                max_gap_s=max(gaps) if gaps else None, temperature_start_c=int(inside[0]["temperature_c"]), temperature_end_c=int(inside[-1]["temperature_c"]),
                max_power_w=max(int(r["board_power_mw"]) for r in inside) / 1000.0)


def gate_row(row, cap_w, energy_dir):
    total, interval = float(row["board_energy_j_total"]), float(row["counted_launch_interval_s"])
    mean_w = total / interval
    cid = "h100/%s/%s/%s" % (row["parent_id"], row["regime"], row["candidate_id"])
    out = dict(cell_id=cid, run_id=row["run_id"], window_s=interval, mean_power_w=round(mean_w, 2), fraction_of_limit=round(mean_w / cap_w, 4))
    trace = Path(row["trace_dir"]) if row.get("trace_dir") else None
    if trace is None or not (trace / "samples.csv").is_file():             # a pulled-back copy: the trace lives next to the raw csv
        alt = sorted((Path(energy_dir) / "raw_attempts" / row["run_id"]).glob("*/samples.csv"))
        trace = alt[0].parent if alt else None
    stats = None
    if trace is not None:
        try:
            samples, t0, t1 = read_trace(trace)
            stats = trace_stats(samples, t0, t1)
        except (OSError, KeyError, ValueError, StopIteration):
            stats = None
    out["trace"] = "present" if stats else "no_trace"
    if stats:
        out.update(stats)
        out["clock_throttled"] = bool(stats["clock_median_mhz"] < CLOCK_MEDIAN_MIN_FRACTION * stats["clock_max_trace_mhz"])
        out["sample_gap"] = bool(stats["max_gap_s"] is not None and stats["max_gap_s"] >= MAX_SAMPLE_GAP_S)
    f = mean_w / cap_w
    out["band"] = "above_cap_rule" if f > ADMIT_FRACTION_OF_CAP else ("near_cap" if f >= BELOW_CAP_FRACTION else "below_cap")
    out["excluded_from_energy_scoring"] = out["band"] == "above_cap_rule"
    return out


def gate(energy_dirs, cap_w):
    rows = []
    for d in energy_dirs:
        p = Path(d) / "application_energy_raw.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                rows.append(gate_row(r, cap_w, d))
    ids = [r["cell_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: two energy rows for one cell")
    band = lambda b: sorted(r["cell_id"] for r in rows if r["band"] == b)
    return dict(schema="h100_window_gate/1", cap_w=cap_w, rules=RULES, windows=len(rows), below_cap=len(band("below_cap")), near_cap=band("near_cap"), above_cap_rule=band("above_cap_rule"),
                clock_throttled=sorted(r["cell_id"] for r in rows if r.get("clock_throttled")), sample_gap=sorted(r["cell_id"] for r in rows if r.get("sample_gap")),
                no_trace=sorted(r["cell_id"] for r in rows if r["trace"] == "no_trace"), rows=sorted(rows, key=lambda r: r["cell_id"]),
                note="Rules fixed before any H100 energy window existed; not changed after measuring. Only `above_cap_rule` excludes a window from the energy scoring (every method alike).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy-dir", action="append", required=True, type=Path)
    ap.add_argument("--cap-w", type=float, required=True, help="the enforced power limit of the board in W (nvidia-smi power.limit of the job; 700 W in HARDWARE_GROUND_TRUTH.md)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit("REFUSED: %s exists; never overwritten" % a.out)
    doc = gate(a.energy_dir, a.cap_w)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] if not isinstance(doc[k], list) else len(doc[k]) for k in ("windows", "below_cap", "near_cap", "above_cap_rule", "clock_throttled", "sample_gap", "no_trace")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
