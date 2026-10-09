#!/usr/bin/env python3
"""Static pipeline for fresh set B (PyTorch row softmax and layer norm, Blackwell, CPU only).

Imports ../fresh/build_fresh.py (never edited) and re-parameterises it: operator ids, regimes (four, incl. xlarge) and
shapes are patched into the imported module, whose define_cells / dispatch_cells / corrected_phase_rows then run
unchanged. The only code that differs is the copy-kernel block-class routine: set A required cols % 32 == 0
("request-row containment"); set B's small regime has cols 112, so that requirement is dropped here. The remaining
guard is unchanged: periodicity of the class pattern is CHECKED (blocks `period`, `period+1`, last must reproduce
their class representative), and test_fresh_b.py compares the totals with an interpreter-free closed form that is
valid for any cols (requests may straddle rows). Corrected IMAD.HI.U32 semantics (../fresh/divider_fix.py) are used
for every padded cell. No measured runtime or energy value is read.

Outputs (next to this file): fresh_cells_b.json, dispatch_trace_fresh_b.json, features_fresh_b.json,
phases_fresh_b.json (corrected divider table; same schema as ../fresh/phases_fresh_divider_corrected.json).
Run: python3 build_fresh_b.py   (background; many minutes)
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRESH = HERE.parent / "fresh"
sys.path.insert(0, str(FRESH))
import build_fresh as B  # noqa: E402  (imported, never edited)

PH, A, C, X, DR, STATIC = B.PH, B.A, B.C, B.X, B.DR, B.STATIC

SOFTMAX_OP = "fresh_b_pytorch_rowwise_softmax"
LAYERNORM_OP = "fresh_b_pytorch_layer_norm"
REGIMES = ("small", "medium", "large", "xlarge")
# (rows, cols): rows multiples of 16, rows*cols multiple of 256; cols from the retained softmax instantiations
# (112: log2 7; 320: log2 9; 640 and 960: log2 10). See FRESH_B.md for the footprint targets and the large-regime note.
SHAPES_B = {"small": (3040, 112), "medium": (7376, 320), "large": (13936, 640), "xlarge": (54608, 960)}
SET_A_SHAPES = {"small": (1368, 96), "medium": (5464, 384), "large": (10920, 768)}

B.SOFTMAX_OP, B.LAYERNORM_OP, B.REGIMES, B.FRESH_SHAPES = SOFTMAX_OP, LAYERNORM_OP, REGIMES, SHAPES_B


def check_dispatch_conditions(cells):
    """Persistent-warp softmax path (SoftMax.cu:1097: dim <= 2048 and dim*4 <= 8192; PersistentSoftmax needs
    log2_ceil(dim) <= 10, i.e. dim <= 1024, for the retained instantiations) and vectorized layer-norm path
    (layer_norm_kernel.cu:1103-1111: N <= 2^24, N % 4 == 0, 16-B alignment of rows)."""
    for c in cells:
        cols = c["cols"]
        if c["operator_id"] == SOFTMAX_OP:
            assert cols <= 1024 and cols * 4 <= 4096 and cols <= 2048, c["cell_id"]
            assert 7 <= math.ceil(math.log2(cols)) <= 10, c["cell_id"]
        else:
            assert cols % 4 == 0 and cols <= (1 << 24) and (cols * 4) % 16 == 0, c["cell_id"]
        assert c["rows"] % 16 == 0 and (c["rows"] * cols) % 256 == 0, c["cell_id"]


def copy_kernel_phases_by_block_class(kern, cell):
    """As B.copy_kernel_phases_by_block_class, without the cols % 32 requirement (periodicity still checked)."""
    constants, grid, block = PH.bind_pytorch("k1", kern, cell)
    rows, cols = cell["input"]["shape"]
    nblocks = math.prod(grid)
    C.require((rows * cols) % 256 == 0, "copy kernel tail block: element count is not a multiple of 256")
    # Address = 4*(e mod cols) + 4*stride*(e div cols). Advancing e by lcm(256, 32*cols) elements keeps e mod cols and
    # shifts the address by a multiple of 128 B, so block classes are b mod period.
    period = (cols * 32) // math.gcd(cols * 32, 256)
    threads = math.prod(block)
    sites = A.P.parse((PH.HERE / "libtorch_sm120" / "k1.isolated.sass").read_text())

    def one(b):
        obs = PH.Observer(threads)
        obs.block_x = block[0]
        interp = A.P.Interp(C.D, ext=True, fchk_fast_path=True, trace=obs.pytorch)

        def coords(l):
            return {k: A.P.V.exact(v) for k, v in {"SR_CTAID.X": b % grid[0], "SR_CTAID.Y": b // grid[0], "SR_CTAID.Z": 0,
                                                   "SR_CgaCtaId": 0, "SR_TID.X": l % block[0], "SR_TID.Y": l // block[0],
                                                   "SR_TID.Z": 0, "SR_LANEID": l % 32}.items()}
        interp.run_block(sites, {k: A.P.V.exact(v) for k, v in constants.items()}, coords, threads)
        return obs.finish(1)

    reps = [one(b) for b in range(min(period, nblocks))]
    for b in sorted({period, period + 1, nblocks - 1}):
        if b < nblocks:
            C.require(one(b) == reps[b % period], "block %d does not reproduce class %d: periodicity not proved" % (b, b % period))
    first = reps[0]
    for rep in reps:
        C.require(rep["barrier_sequence"] == first["barrier_sequence"] and len(rep["phases"]) == len(first["phases"]), "classes differ in barrier structure")
    counts = [nblocks // period + (1 if c < nblocks % period else 0) for c in range(len(reps))]
    phases = []
    for i in range(len(first["phases"])):
        row = dict(index=i, repetitions=1, issue_warp_instructions=collections.Counter(), read_sectors=0, write_sectors=0, lines=0,
                   read_bytes=0, write_bytes=0, critical_path_compute_instructions=0, dependent_global_load_depth=0)
        for rep, n in zip(reps, counts):
            ph = rep["phases"][i]
            for op, v in ph["issue_warp_instructions"].items():
                row["issue_warp_instructions"][op] += v * n
            for f in B.SUM_FIELDS:
                row[f] = None if row[f] is None or ph[f] is None else row[f] + ph[f] * n
            for f in ("critical_path_compute_instructions", "dependent_global_load_depth"):
                row[f] = max(row[f], ph[f])
        row["issue_warp_instructions"] = dict(row["issue_warp_instructions"])
        phases.append(row)
    return dict(phases=phases, barrier_sequence=first["barrier_sequence"],
                unknown_guard_upper_bounds=first["unknown_guard_upper_bounds"],
                block_classes=dict(period_blocks=period, classes_interpreted=len(reps), grid_blocks=nblocks))


B.copy_kernel_phases_by_block_class = copy_kernel_phases_by_block_class


def main():
    hw, l2_bytes = B.l2_bytes_from_ground_truth()
    cells, pads = B.define_cells(l2_bytes)
    check_dispatch_conditions(cells)
    for c in cells:
        assert (c["rows"], c["cols"]) not in set(SET_A_SHAPES.values()) | set(B.DEV_SHAPES.values()), c["cell_id"]
    dcells = B.dispatch_cells(cells)
    disp_path = HERE / "dispatch_trace_fresh_b.json"
    disp_path.write_text(json.dumps(B.dispatch_document(dcells), indent=1) + "\n")

    # ---- features (corrected interpreter is identical on instruction-level features; checked in set A)
    A.DISPATCH_JSON = disp_path
    rows = A.build_rows(X, hw, B.dev_rows_for_adapter(cells))
    frows = sorted(rows.values(), key=lambda r: r["cell_id"])
    fcov = X.coverage(frows)
    features = {
        "schema": "static_runtime_features_blackwell/1",
        "note": ("Label-free static features. No runtime, energy or power column was read. Lane-level counts are not warp-issued counts. "
                 "Derivation outputs are candidates, not profiler-validated or scientifically admitted. "
                 "FRESH SET B (fresh_cells_b.json): built by the committed adapter from the set B dispatch trace."),
        "extract_features_py_sha256": B.sha(STATIC / "extract_features.py"),
        "hardware_from_ground_truth": hw,
        "hardware_constants_not_in_ground_truth": list(X.HW_NOT_IN_GROUND_TRUTH),
        "coverage": fcov,
        "input_sha256": dict(sorted(X.READ_HASHES.items())),
        "rows": frows,
    }
    (HERE / "features_fresh_b.json").write_text(json.dumps(features, indent=1, sort_keys=True) + "\n")

    # ---- corrected-divider phase table
    crow = B.corrected_phase_rows(dcells)
    for cid in crow:
        print("corrected", cid, crow[cid]["status"], crow[cid].get("reason") or "", flush=True)
    paths = [STATIC / "phases.py", STATIC / "extract_features.py", STATIC / "coalescing/coalescing.py",
             STATIC / "pytorch_features/adapter.py", STATIC / "pytorch_features/pt_interp.py", FRESH / "divider_fix.py",
             FRESH / "build_fresh.py", disp_path, STATIC / "pytorch_dispatch/src/aten_src_ATen_cuda_detail_IntegerDivider.cuh", Path(__file__)]
    base = json.loads((STATIC / "phases_blackwell.json").read_text())
    phases = dict(schema=base["schema"], input_sha256={str(p.relative_to(X.REPO)): B.sha(p) for p in paths},
                  assumptions=base["assumptions"] + [
                      "Fresh set B: first and last block sampled for main kernels, as for the development PyTorch cells; for softmax cols < next power of two the column tail predicates are decided by the interpreter from the bound element_count.",
                      "VARIANT: IMAD.HI.U32 evaluated as hi32(Ra*Rb + 64-bit register pair at Rc) (../fresh/divider_fix.py); copy-kernel totals are exact whole-grid sums over block classes (period checked), not boundary-block extrapolation. Set B drops set A's cols % 32 == 0 requirement (cols 112); the interpreter-free closed form in test_fresh_b.py covers requests that straddle rows."],
                  transaction_profiler_validation=False, rows=crow)
    (HERE / "phases_fresh_b.json").write_text(json.dumps(phases, indent=1, sort_keys=True) + "\n")

    feat_by = {r["cell_id"]: r for r in frows}
    dis_by = {d["cell_id"]: d for d in dcells}
    for c in cells:
        f, d, cp = feat_by[c["cell_id"]], dis_by[c["cell_id"]], crow[c["cell_id"]]
        c["shape_a"], c["shape_b"] = c["rows"], c["cols"]
        c["call"] = d["call"]
        c["kernels_per_launch"] = d["kernels_per_call"]
        c["kernels"] = [{k: kk[k] for k in ("order", "role", "grid", "block", "dynamic_smem_bytes")} |
                        {"kernel_id": A.kernel_id(kk), "template_log2_elements": kk.get("template_args", {}).get("log2_elements")}
                        for kk in d["kernels"]]
        c["features_status"] = f["status"]
        c["features_missing"] = f["missing_features"]
        c["phases_status"] = cp["status"]
        c["phases_reason"] = cp.get("reason")
        c["derived_fully"] = f["status"] == "supported_with_assumptions" and cp["status"] == "conditional_static_phases"
    doc = {
        "schema": "fresh_pytorch_cells/1",
        "set": "B",
        "purpose": "Second prospective evaluation set for the static runtime predictor; no measured value of any kind is in this file.",
        "hardware_l2_bytes_from_ground_truth": l2_bytes,
        "padding_rule": {"stride": "cols + pad", "pad_by_candidate": pads, "derived_from": "pytorch_dispatch/dispatch_trace.json development controls"},
        "shapes_rows_cols": {k: list(v) for k, v in SHAPES_B.items()},
        "set_a_shapes_rows_cols": {k: list(v) for k, v in SET_A_SHAPES.items()},
        "development_shapes_rows_cols": {k: list(v) for k, v in B.DEV_SHAPES.items()},
        "logical_bytes_rule": {
            SOFTMAX_OP: "read rows*cols*4 + write rows*cols*4; padded stride and copy kernel traffic are not added, as in the development table",
            LAYERNORM_OP: "read (3*rows*cols + 2*rows)*4 + write (rows*cols + 2*rows)*4; padded stride and copy kernel traffic not added"},
        "footprint_rule": {
            SOFTMAX_OP: "2*rows*cols*4 (operator_work_time_diagnostic.py footprint_bytes), independent of stride",
            LAYERNORM_OP: "rows*row_stride*4 + 2*cols*4 + write bytes (heldout_operator_test.py operator_design)"},
        "tier_rule": "L2 iff footprint / verified L2 capacity < 1 (HARDWARE_GROUND_TRUTH.md Blackwell), else DRAM",
        "cells": cells,
        "coverage": dict(collections.Counter("%s derived_fully=%s" % (c["operator_id"], c["derived_fully"]) for c in cells)),
    }
    (HERE / "fresh_cells_b.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "features": fcov["by_status"], "phases": dict(collections.Counter(r["status"] for r in crow.values()))}), flush=True)


if __name__ == "__main__":
    main()
