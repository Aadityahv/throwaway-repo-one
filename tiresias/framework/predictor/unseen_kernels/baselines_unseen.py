#!/usr/bin/env python3
"""Baseline inputs for the unseen-kernel test, frozen before any measurement (CPU only, no label of an unseen cell).

Runtime baseline (needs no execution): memory-plus-compute roofline
    t = t0 + max( logical_bytes / BW_tier , fp32_lane_instructions / (SMs * 128 lanes * SM clock) )
  BW_tier is the measured streaming read bandwidth of the tier (constants/stream_constants.json: L2 sectors, DRAM);
  t0 is the same file's fixed launch overhead per kernel; fp32 lane instructions are FFMA + FADD + FMUL lane counts of
  the whole launch (static counts); SM count and clock come from HARDWARE_GROUND_TRUTH.md (the 2,617 MHz boost reading
  the earlier plans used). A launch of k kernels pays t0 k times.
Energy baselines (computed at scoring time from measured labels, defined here so they cannot be chosen after the fact):
  time the kernel x typical power -- typical power = median measured power of the Blackwell development cells below the
      cap (constant recorded below, from exposed development cells, not from any unseen cell);
  time the kernel x anchor power -- power of the same kernel's medium/c1 cell measured in the same run (needs one energy
      measurement per kernel);
  runtime-plus-traffic model -- the frozen E = min(P0 t + eps_tier bytes, cap t) with the eps per tier below.
Published-method baselines are a separate, later step and are not part of this freeze.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SR = HERE.parent
OUT = HERE / "frozen"
DEV = HERE.parents[1] / "compile_evidence/development/development_cells.csv"
SM_CLOCK_HZ = 2.617e9   # HARDWARE_GROUND_TRUTH.md, Blackwell "SM clock (at time of probe)" 2,617,000 kHz
FP32_LANES_PER_SM = 128


def typical_power():
    p = []
    for r in csv.DictReader(DEV.open(newline="")):
        if r["gpu"] == "blackwell" and r["below_cap"] == "True":
            p.append(float(r["energy_j"]) / float(r["runtime_s"]))
    return float(np.median(p)), len(p)


def main():
    feats = json.loads((OUT / "features_unseen.json").read_text())
    cells = {c["cell_id"]: c for c in json.loads((OUT / "cells_unseen.json").read_text())["cells"]}
    S = json.loads((SR / "constants/stream_constants.json").read_text())["constants"]
    sm = feats["hardware_from_ground_truth"]["sm_count"]
    peak = sm * FP32_LANES_PER_SM * SM_CLOCK_HZ
    bw = {"L2": S["L2_read_sector_TBps"] * 1e12, "DRAM": S["DRAM_read_TBps"] * 1e12}
    roof = {}
    for f in feats["rows"]:
        cid = f["cell_id"]
        c = cells[cid]
        t = f.get("per_launch_totals")
        if not t:
            roof[cid] = dict(roofline_s=None, reason="static counts unavailable")
            continue
        oc = t["opcode_lane_counts"]
        fp32 = sum(n for op, n in oc.items() if op.split(".")[0] in ("FFMA", "FADD", "FMUL"))
        mem_s = c["logical_bytes_per_launch"] / bw[c["tier"]]
        comp_s = fp32 / peak
        roof[cid] = dict(roofline_s=S["t0_us"] * 1e-6 * t["kernels_per_launch"] + max(mem_s, comp_s), memory_s=mem_s, compute_s=comp_s,
                         fp32_lane_instructions=fp32, bound="compute" if comp_s > mem_s else "memory")
    tp, n = typical_power()
    doc = dict(schema="unseen_baseline_inputs/1", frozen_before_timing=True,
               roofline_definition="t = t0*kernels + max(logical_bytes/BW_tier, fp32_lane_instructions/(SMs*128*SM_clock)); see module docstring",
               roofline_constants=dict(t0_us=S["t0_us"], L2_read_TBps=S["L2_read_sector_TBps"], DRAM_read_TBps=S["DRAM_read_TBps"], sm_count=sm,
                                       fp32_lanes_per_sm=FP32_LANES_PER_SM, sm_clock_hz=SM_CLOCK_HZ, peak_fp32_lane_ops_per_s=peak),
               typical_power_w=tp, typical_power_cells=n,
               runtime_plus_traffic_eps_pJ_per_byte={"L2": 80.642125766766, "DRAM": 179.59310363418655},
               runtime_plus_traffic_P0_w=152.52913482177462, cap_w=600.0,
               roofline=roof)
    (OUT / "baseline_inputs_unseen.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(typical_power_w=tp, n=n, roofline_cells=sum(r["roofline_s"] is not None for r in roof.values())), indent=1))


if __name__ == "__main__":
    main()
