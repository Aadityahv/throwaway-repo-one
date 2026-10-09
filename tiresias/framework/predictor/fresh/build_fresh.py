#!/usr/bin/env python3
"""Static pipeline for FRESH PyTorch row-softmax and layer-norm shapes on Blackwell (CPU only).

No GPU, no ssh, no measured runtime or energy value is read. The development table is not opened at all
here (its tier and byte rules are re-derived from the committed scripts and checked against it in
test_fresh.py, which reads only the shape/tier/byte columns).

Outputs, all next to this file:
  fresh_cells.json              cell definitions, controls, tier, logical bytes, kernel list, derivation status
  dispatch_trace_fresh.json     the dispatch trace for the fresh cells, produced by calling
                                pytorch_dispatch/dispatch_rules.py (softmax_cell / layernorm_cell), not hand edited
  features_fresh.json           same schema as ../features_blackwell.json (rows built by pytorch_features/adapter.py)
  phases_fresh.json             same schema as ../phases_blackwell.json (rows built by ../phases.py:pytorch)

Run:  python3 build_fresh.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent
sys.path.insert(0, str(STATIC))
sys.path.insert(0, str(STATIC / "pytorch_dispatch"))
import dispatch_rules as DR  # noqa: E402  (imported, never edited)
import phases as PH  # noqa: E402  (imported, never edited; loads extract_features and the adapter)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import divider_fix  # noqa: E402

X, A, C = PH.X, PH.A, PH.C

SOFTMAX_OP = "fresh_pytorch_rowwise_softmax"
LAYERNORM_OP = "fresh_pytorch_layer_norm"
DEV_SOFTMAX_OP = "dev_pytorch_rowwise_softmax"
DEV_LAYERNORM_OP = "final_pytorch_layer_norm"
REGIMES = ("small", "medium", "large")
CANDS = ("c1", "c2", "c3", "c4")
FLOAT_BYTES = 4

# (rows, cols) per regime. cols is chosen from the softmax instantiations that are retained in
# ../libtorch_sm120 (log2 elements 7, 9, 10 -> cols 96, 384, 768). rows is a multiple of 8 so that
#   * softmax grids have no partial block (batches per block is 8 for log2 7 and 4 otherwise),
#   * rows * cols is a multiple of 256 (the padded-copy kernel has no tail block; the interpreter decides
#     the copy kernel's bounds check for all blocks at once and would refuse a tail block).
# Logical footprint rows*cols*4 is within 1.003x of the development set's 0.5 / 8 / 32 MiB.
FRESH_SHAPES = {"small": (1368, 96), "medium": (5464, 384), "large": (10920, 768)}
DEV_SHAPES = {"small": (1024, 128), "medium": (4096, 512), "large": (8192, 1024)}
DEV_FOOTPRINT_TARGET_BYTES = {"small": 1024 * 128 * 4, "medium": 4096 * 512 * 4, "large": 8192 * 1024 * 4}
ROW_STRIDE_PADS = (0, 1, 3, 7)  # reset/build_workload_catalog.py and reset/pytorch_softmax_adapter.py: stride = cols + pad


# ------------------------------------------------------------------ byte and tier rules (re-derived from committed code)
def softmax_logical_bytes(rows, cols):
    """reset/build_workload_catalog.py work_pytorch_softmax: read rows*cols*4, write rows*cols*4. The padded copy
    kernel's own traffic is NOT included (the development table does not include it either)."""
    return rows * cols * FLOAT_BYTES, rows * cols * FLOAT_BYTES


def softmax_footprint(rows, cols, stride):
    """reset/operator_work_time_diagnostic.py footprint_bytes: 2*rows*cols*4 for dev_pytorch_rowwise_softmax
    (independent of the row stride)."""
    return 2 * rows * cols * FLOAT_BYTES


def layernorm_logical_bytes(rows, cols):
    """reset/build_workload_catalog.py work_pytorch_layer_norm: reads (3*rows*cols + 2*rows)*4, writes
    (rows*cols + 2*rows)*4 (the read term as catalogued; the copy kernel is not included)."""
    return (rows * cols * 3 + rows * 2) * FLOAT_BYTES, (rows * cols + rows * 2) * FLOAT_BYTES


def layernorm_footprint(rows, cols, stride):
    """reset/heldout_operator_test.py operator_design: padded input a*stride*4 + 2*cols*4 + write bytes."""
    _, wb = layernorm_logical_bytes(rows, cols)
    return rows * stride * FLOAT_BYTES + 2 * cols * FLOAT_BYTES + wb


def l2_bytes_from_ground_truth():
    hw = X.load_hardware(X.read_text(X.GROUND_TRUTH))
    return hw, hw["l2_bytes"]


def tier_for(footprint, l2_bytes):
    ratio = footprint / l2_bytes
    return ("L2" if ratio < 1.0 else "DRAM"), ratio


# ------------------------------------------------------------------ cell definitions
def padding_rule_from_dev_trace():
    """Derive stride = cols + pad per candidate from the dev dispatch trace's controls (not hard-coded)."""
    disp = json.loads((STATIC / "pytorch_dispatch/dispatch_trace.json").read_text())
    pads = collections.defaultdict(set)
    for c in disp["cells"]:
        if c["operator_id"] not in (DEV_SOFTMAX_OP, DEV_LAYERNORM_OP):
            continue
        ctl = c["controls"]
        cols = ctl.get("dim_size", ctl.get("cols"))
        pads[c["candidate_id"]].add(ctl["row_stride"] - cols)
    rule = {k: sorted(v) for k, v in pads.items()}
    if any(len(v) != 1 for v in rule.values()) or tuple(rule[c][0] for c in CANDS) != ROW_STRIDE_PADS:
        raise SystemExit("dev padding rule differs from ROW_STRIDE_PADS: %r" % rule)
    dev_pairs = {(c["controls"].get("outer_size", c["controls"].get("rows")), c["controls"].get("dim_size", c["controls"].get("cols")))
                 for c in disp["cells"] if c["operator_id"] in (DEV_SOFTMAX_OP, DEV_LAYERNORM_OP)}
    return {k: v[0] for k, v in rule.items()}, dev_pairs


def define_cells(l2_bytes):
    pads, dev_pairs = padding_rule_from_dev_trace()
    cells = []
    for op in (SOFTMAX_OP, LAYERNORM_OP):
        for regime in REGIMES:
            rows, cols = FRESH_SHAPES[regime]
            if (rows, cols) in dev_pairs:
                raise SystemExit("fresh shape equals a development shape: %r" % ((rows, cols),))
            for cand in CANDS:
                stride = cols + pads[cand]
                if op == SOFTMAX_OP:
                    rb, wb = softmax_logical_bytes(rows, cols)
                    fp = softmax_footprint(rows, cols, stride)
                    ctl = {"dim_size": cols, "inner_size": 1, "outer_size": rows, "row_stride": stride}
                else:
                    rb, wb = layernorm_logical_bytes(rows, cols)
                    fp = layernorm_footprint(rows, cols, stride)
                    ctl = {"affine": True, "cols": cols, "eps": 1e-05, "row_stride": stride, "rows": rows}
                tier, ratio = tier_for(fp, l2_bytes)
                cells.append(dict(cell_id="blackwell/%s/%s/%s" % (op, regime, cand), operator_id=op, regime=regime,
                                  candidate_id=cand, rows=rows, cols=cols, row_stride=stride, pad_elements=pads[cand],
                                  controls=ctl, footprint_bytes=fp, l2_ratio=ratio, tier=tier,
                                  logical_read_bytes=rb, logical_write_bytes=wb, logical_bytes_per_launch=rb + wb))
    return cells, pads


def dispatch_cells(cells):
    out = []
    for c in cells:
        op, ctl = c["operator_id"], c["controls"]
        fn = DR.softmax_cell if op == SOFTMAX_OP else DR.layernorm_cell
        # The rules take the development operator id only as a label; the cell id carries the fresh id.
        d = fn(c["cell_id"], op, c["regime"], c["candidate_id"], ctl, ["fresh_unmeasured"])
        if op == SOFTMAX_OP and c["cols"] & (c["cols"] - 1):
            npot = 1 << DR.log2_ceil(c["cols"])
            for k in d["kernels"]:
                if k["role"] == "softmax":
                    k["extra_control_flow"] = (
                        "rows beyond batch_size masked via local_batches; element_count (%d) < next_power_of_two (%d): "
                        "the kernel instantiated for %d elements masks columns >= element_count (tail predication on the "
                        "last warp iterations); the extended interpreter decides these predicates from the bound "
                        "element_count" % (c["cols"], npot, npot))
        out.append(d)
    return out


def dispatch_document(dcells):
    sums = {}
    for line in (DR.SRC / "SHA256SUMS").read_text().splitlines():
        h, n = line.split()
        sums[n] = h
    return {
        "schema": "pytorch_dispatch_trace_v1",
        "pytorch_commit": DR.REVISION,
        "provenance_note": ("Fresh-shape cells. Produced by calling pytorch_dispatch/dispatch_rules.py softmax_cell / "
                            "layernorm_cell with the fresh controls; the rules and sources are unchanged. Same static-host-code "
                            "provenance and assumptions as pytorch_dispatch/dispatch_trace.json (catalog pin 67faf385 is not the "
                            "source that ran; commit 70d99e99 is)."),
        "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120)",
        "source_files_sha256": sums,
        "cells": dcells,
    }


def dev_rows_for_adapter(cells):
    """The adapter's `dev` argument: rows shaped like extract_features.load_dev_rows() output for the allowed
    columns. The development table has no geometry for these operators, and fresh cells have no table row, so
    geometry fields are empty (as they are for the 27 development PyTorch cells)."""
    dev = {}
    for c in cells:
        dev[c["cell_id"]] = {"gpu": "blackwell", "operator_id": c["operator_id"], "cell_id": c["cell_id"], "regime": c["regime"],
                             "candidate_id": c["candidate_id"], "tier": c["tier"], "bytes_per_launch": str(c["logical_bytes_per_launch"]),
                             "actual_logical_bytes_per_launch": str(c["logical_bytes_per_launch"]), "grid_blocks": "",
                             "block_threads": "", "blocks_per_sm": "", "active_sm_fraction": "", "geometry_source": "",
                             "shape_a": str(c["rows"]), "shape_b": str(c["cols"]), "sm_count": ""}
    return dev


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ corrected-IMAD.HI phase table (see FRESH.md)
SUM_FIELDS = ("read_sectors", "write_sectors", "lines", "read_bytes", "write_bytes")


def copy_kernel_phases_by_block_class(kern, cell):
    """Phase rows of the copy kernel for the whole grid, under the corrected interpreter (caller installs it).

    The warp requests of one block depend on the block only through the alignment of its rows. Every warp request
    is 32 consecutive elements, cols is a multiple of 32, so a request lies in one row and its start byte is
    4*stride*row (mod 128) plus a multiple of 128: the pattern repeats when the block's first element advances by
    32 rows = 32*cols elements. Blocks are classes b mod period, period = 32*cols / gcd(32*cols, 256). One block
    of each class is interpreted (blocks=1) and weighted by the number of grid blocks in the class. Periodicity is
    CHECKED, not assumed: blocks `period`, `period+1` and the last block must reproduce their class representative
    exactly, or the cell is refused."""
    constants, grid, block = PH.bind_pytorch("k1", kern, cell)
    rows, cols = cell["input"]["shape"]
    nblocks = math.prod(grid)
    C.require((rows * cols) % 256 == 0, "copy kernel tail block: element count is not a multiple of 256")
    C.require(cols % 32 == 0, "request-row containment needs cols % 32 == 0")
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
            for f in SUM_FIELDS:
                row[f] = None if row[f] is None or ph[f] is None else row[f] + ph[f] * n
            for f in ("critical_path_compute_instructions", "dependent_global_load_depth"):
                row[f] = max(row[f], ph[f])
        row["issue_warp_instructions"] = dict(row["issue_warp_instructions"])
        phases.append(row)
    return dict(phases=phases, barrier_sequence=first["barrier_sequence"],
                unknown_guard_upper_bounds=first["unknown_guard_upper_bounds"],
                block_classes=dict(period_blocks=period, classes_interpreted=len(reps), grid_blocks=nblocks))


def corrected_phase_rows(dcells, cache=None):
    """Same as the frozen loop, with the corrected interpreter installed; copy kernels by block class."""
    orig = A.P.Interp
    A.P.Interp = divider_fix.corrected_interp_class(A.P)
    cache = {} if cache is None else cache
    out = {}
    try:
        for cell in dcells:
            cid = cell["cell_id"]
            kernels = []
            try:
                for kern in cell["kernels"]:
                    kid = A.kernel_id(kern)
                    key = json.dumps([kid, kern, cell["input"] if kid == "k1" else None], sort_keys=True)
                    if key not in cache:
                        cache[key] = copy_kernel_phases_by_block_class(kern, cell) if kid == "k1" else PH.pytorch(kid, kern, cell)
                    kernels.append(dict(kernel_id=kid, **cache[key]))
                unknown = any(p["read_sectors"] is None or p["write_sectors"] is None for k in kernels for p in k["phases"])
                out[cid] = dict(status="unsupported" if unknown else "conditional_static_phases", kernels=kernels,
                                reason="data-dependent global address; transaction counts remain null" if unknown else None)
            except (C.Refusal, A.P.Refusal) as ex:
                out[cid] = dict(status="unsupported", reason=str(ex), kernels=[])
    finally:
        A.P.Interp = orig
    return out


def main():
    hw, l2_bytes = l2_bytes_from_ground_truth()
    cells, pads = define_cells(l2_bytes)
    dcells = dispatch_cells(cells)
    disp_path = HERE / "dispatch_trace_fresh.json"
    disp_path.write_text(json.dumps(dispatch_document(dcells), indent=1) + "\n")

    # ---- features: the committed adapter, pointed at the fresh dispatch trace
    A.DISPATCH_JSON = disp_path
    rows = A.build_rows(X, hw, dev_rows_for_adapter(cells))
    frows = sorted(rows.values(), key=lambda r: r["cell_id"])
    fcov = X.coverage(frows)
    features = {
        "schema": "static_runtime_features_blackwell/1",
        "note": ("Label-free static features. No runtime, energy or power column was read. Lane-level counts are not warp-issued counts. "
                 "Derivation outputs are candidates, not profiler-validated or scientifically admitted. "
                 "FRESH-SHAPE table (fresh_cells.json): built by the committed adapter from the fresh dispatch trace."),
        "extract_features_py_sha256": sha(STATIC / "extract_features.py"),
        "hardware_from_ground_truth": hw,
        "hardware_constants_not_in_ground_truth": list(X.HW_NOT_IN_GROUND_TRUTH),
        "coverage": fcov,
        "input_sha256": dict(sorted(X.READ_HASHES.items())),
        "rows": frows,
    }
    (HERE / "features_fresh.json").write_text(json.dumps(features, indent=1, sort_keys=True) + "\n")

    # ---- phases: ../phases.py:pytorch on each dispatched kernel (same loop as phases.main for PyTorch cells)
    prow, cache = {}, {}
    for cell in dcells:
        cid = cell["cell_id"]
        kernels = []
        try:
            for kern in cell["kernels"]:
                kid = A.kernel_id(kern)
                key = json.dumps([kid, kern, cell["input"] if kid == "k1" else None], sort_keys=True)
                if key not in cache:
                    cache[key] = PH.pytorch(kid, kern, cell)
                kernels.append(dict(kernel_id=kid, **cache[key]))
            unknown = any(p["read_sectors"] is None or p["write_sectors"] is None for k in kernels for p in k["phases"])
            prow[cid] = dict(status="unsupported" if unknown else "conditional_static_phases", kernels=kernels,
                             reason="data-dependent global address; transaction counts remain null" if unknown else None)
        except (C.Refusal, A.P.Refusal) as ex:
            prow[cid] = dict(status="unsupported", reason=str(ex), kernels=[])
        print(cid, prow[cid]["status"], prow[cid].get("reason") or "", flush=True)
    paths = [STATIC / "phases.py", STATIC / "extract_features.py", STATIC / "coalescing/coalescing.py",
             STATIC / "pytorch_features/adapter.py", STATIC / "pytorch_features/pt_interp.py",
             disp_path, STATIC / "pytorch_dispatch/src/aten_src_ATen_cuda_detail_IntegerDivider.cuh", Path(__file__)]
    base = json.loads((STATIC / "phases_blackwell.json").read_text())
    phases = dict(schema=base["schema"], input_sha256={str(p.relative_to(X.REPO)): sha(p) for p in paths},
                  assumptions=base["assumptions"] + [
                      "Fresh-shape cells: first and last block sampled, as for the development PyTorch cells; for softmax cols < next power of two the column tail predicates are decided by the interpreter from the bound element_count."],
                  transaction_profiler_validation=False, rows=prow)
    (HERE / "phases_fresh.json").write_text(json.dumps(phases, indent=1, sort_keys=True) + "\n")

    # ---- variant with the corrected IMAD.HI.U32 (FRESH.md, "Pre-existing interpreter issue")
    crow = corrected_phase_rows(dcells)
    for cid in crow:
        print("corrected", cid, crow[cid]["status"], crow[cid].get("reason") or "", flush=True)
    cphases = dict(phases, assumptions=phases["assumptions"] + [
        "VARIANT: IMAD.HI.U32 evaluated as hi32(Ra*Rb + 64-bit register pair at Rc) (divider_fix.py); copy-kernel totals are the exact whole-grid sums over block classes (period checked), not boundary-block extrapolation."],
        rows=crow)
    (HERE / "phases_fresh_divider_corrected.json").write_text(json.dumps(cphases, indent=1, sort_keys=True) + "\n")
    orig_interp = A.P.Interp
    A.P.Interp = divider_fix.corrected_interp_class(A.P)
    try:
        rows_fixed = A.build_rows(X, hw, dev_rows_for_adapter(cells))
    finally:
        A.P.Interp = orig_interp

    def strip(r):
        keep = {k: r[k] for k in ("work", "structure", "status", "missing_features", "occupancy")}
        keep["secondary"] = [{k: s_.get(k) for k in ("work", "structure", "status", "occupancy")} for s_ in r["secondary_kernels"]]
        return json.dumps(keep, sort_keys=True, default=str)
    features_identical = all(strip(rows[c]) == strip(rows_fixed[c]) for c in rows)

    # ---- impact of the same issue on the frozen development phase table (padded copy kernel, read side), static only
    dev_impact = []
    frozen_dev = json.loads((STATIC / "phases_blackwell.json").read_text())["rows"]
    dev_trace = json.loads((STATIC / "pytorch_dispatch/dispatch_trace.json").read_text())
    dev_padded = [c for c in dev_trace["cells"] if c["operator_id"] in (DEV_SOFTMAX_OP, DEV_LAYERNORM_OP) and c["kernels_per_call"] == 2]
    dev_fixed = corrected_phase_rows(dev_padded)
    for c in dev_padded:
        cid = c["cell_id"]
        fk = next(k for k in frozen_dev[cid]["kernels"] if k["kernel_id"] == "k1")["phases"][0]
        ck = next(k for k in dev_fixed[cid]["kernels"] if k["kernel_id"] == "k1")["phases"][0] if dev_fixed[cid]["kernels"] else None
        dev_impact.append(dict(cell_id=cid, frozen_read_sectors=fk["read_sectors"], corrected_read_sectors=ck and ck["read_sectors"],
                               frozen_lines=fk["lines"], corrected_lines=ck and ck["lines"],
                               frozen_write_sectors=fk["write_sectors"], corrected_write_sectors=ck and ck["write_sectors"],
                               read_sector_ratio=(ck["read_sectors"] / fk["read_sectors"]) if ck and fk["read_sectors"] else None))
    (HERE / "dev_copy_kernel_divider_impact.json").write_text(json.dumps(
        dict(note="Static only. Copy-kernel read sectors of the 18 padded development cells: frozen phases_blackwell.json vs the corrected IMAD.HI.U32 interpreter. No runtime or energy value involved.",
             rows=dev_impact), indent=1) + "\n")

    # ---- cell definitions with derivation status
    feat_by = {r["cell_id"]: r for r in frows}
    dis_by = {d["cell_id"]: d for d in dcells}
    for c in cells:
        f, p, d = feat_by[c["cell_id"]], prow[c["cell_id"]], dis_by[c["cell_id"]]
        c["shape_a"], c["shape_b"] = c["rows"], c["cols"]
        c["call"] = d["call"]
        c["kernels_per_launch"] = d["kernels_per_call"]
        c["kernels"] = [{k: kk[k] for k in ("order", "role", "grid", "block", "dynamic_smem_bytes")} |
                        {"kernel_id": A.kernel_id(kk), "template_log2_elements": kk.get("template_args", {}).get("log2_elements")}
                        for kk in d["kernels"]]
        c["features_status"] = f["status"]
        c["features_missing"] = f["missing_features"]
        c["phases_status"] = p["status"]
        c["phases_reason"] = p.get("reason")
        c["derived_fully"] = f["status"] == "supported_with_assumptions" and p["status"] == "conditional_static_phases"
        cp = crow[c["cell_id"]]
        c["phases_status_divider_corrected"] = cp["status"]
        c["phases_reason_divider_corrected"] = cp.get("reason")
        c["derived_fully_divider_corrected"] = f["status"] == "supported_with_assumptions" and cp["status"] == "conditional_static_phases"
    doc = {
        "schema": "fresh_pytorch_cells/1",
        "purpose": "Prospective evaluation cells for the static runtime predictor; no measured value of any kind is in this file.",
        "hardware_l2_bytes_from_ground_truth": l2_bytes,
        "padding_rule": {"stride": "cols + pad", "pad_by_candidate": pads, "derived_from": "pytorch_dispatch/dispatch_trace.json development controls"},
        "shapes_rows_cols": {k: list(v) for k, v in FRESH_SHAPES.items()},
        "development_shapes_rows_cols": {k: list(v) for k, v in DEV_SHAPES.items()},
        "logical_bytes_rule": {
            SOFTMAX_OP: "read rows*cols*4 + write rows*cols*4 (reset/build_workload_catalog.py work_pytorch_softmax); padded stride and the contiguous() copy kernel traffic are not added, as in the development table",
            LAYERNORM_OP: "read (3*rows*cols + 2*rows)*4 + write (rows*cols + 2*rows)*4 (work_pytorch_layer_norm); padded stride and copy kernel traffic not added, as in the development table"},
        "footprint_rule": {
            SOFTMAX_OP: "2*rows*cols*4 (operator_work_time_diagnostic.py footprint_bytes), independent of stride",
            LAYERNORM_OP: "rows*row_stride*4 + 2*cols*4 + write bytes (heldout_operator_test.py operator_design)"},
        "tier_rule": "L2 iff footprint / verified L2 capacity < 1 (HARDWARE_GROUND_TRUTH.md Blackwell, 134,217,728 B), else DRAM",
        "cells": cells,
        "features_identical_under_corrected_imad_hi": features_identical,
        "coverage": {"frozen_pipeline": dict(collections.Counter("%s derived_fully=%s" % (c["operator_id"], c["derived_fully"]) for c in cells)),
                     "divider_corrected_variant": dict(collections.Counter("%s derived_fully=%s" % (c["operator_id"], c["derived_fully_divider_corrected"]) for c in cells))},
    }
    (HERE / "fresh_cells.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "features": fcov["by_status"], "phases": dict(collections.Counter(r["status"] for r in prow.values()))}))


if __name__ == "__main__":
    main()
