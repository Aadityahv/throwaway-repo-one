#!/usr/bin/env python3
"""CPU-only descriptor for the pinned xFormers index_select_cat forward kernel.

This module remains CPU-safe and does not import Torch/Triton, compile source,
or touch a GPU. ``xformers_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that runner
exists (builder carve-out, mirroring the embedding M286 state) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_xformers_indexed_select"
# Catalog ladder: BLOCK_SIZE_COL per candidate; BLOCK_SIZE_INDEX stays 1.
CANDIDATE_BLOCK_COLS = {"c1": 128, "c2": 256, "c3": 512, "c4": 1024}


class XFormersConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise XFormersConfigError(f"expected one xformers cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    num_rows, num_cols = int(row["shape_a"]), int(row["shape_b"])
    if num_rows <= 0 or num_cols <= 0:
        raise XFormersConfigError(f"non-positive problem shape: {num_rows}x{num_cols}")
    block_col = CANDIDATE_BLOCK_COLS.get(candidate_id)
    if block_col is None:
        raise XFormersConfigError(f"unknown xformers candidate {candidate_id}")
    num_indices = int(controls.get("num_indices", -1))
    if controls != {"num_rows": num_rows, "num_indices": num_indices, "num_cols": num_cols,
                    "block_size_index": 1, "block_size_col": block_col, "dtype": "float32"}:
        raise XFormersConfigError(f"catalog controls do not match source mapping: {controls}")
    if num_indices <= 0 or num_indices >= num_rows:
        raise XFormersConfigError("source wrapper requires 0 < num_indices < num_rows")
    if int(row["threads_per_block"]) != block_col or int(row["items_per_thread"]) != 1:
        raise XFormersConfigError("threads/items must be the catalog placeholders (BLOCK_SIZE_COL, 1)")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise XFormersConfigError("xformers candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "xformers/ops/_triton/k_index_select_cat.py",
        "source_revision": "6d925b3a94312fb46249d83879cab045b6a0aaae",
        "build": {
            "language": "Python/Triton",
            "source_checkout_required": True,
            "compile_entry": "@triton.jit index_select_cat_fwd_kernel",
            "compile_meta": {"BLOCK_SIZE_INDEX": 1, "BLOCK_SIZE_COL": block_col},
        },
        "launch": {
            "kernel": "index_select_cat_fwd_kernel",
            "grid": "(cdiv(num_indices, 1), cdiv(num_cols, BLOCK_SIZE_COL)), one program per index-row x col-block",
            "runtime_args": [f"num_rows={num_rows}", f"num_indices={num_indices}",
                             f"num_cols={num_cols}", "stride0=num_cols", "stride1=1",
                             "BLOCK_SIZE_INDEX=1", f"BLOCK_SIZE_COL={block_col}"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "index_dtype": "int64",
            "operation": "gather of num_indices rows into a full output matrix",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 rounding of large canonical values; exact index/output shape"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 12
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"implemented_unvalidated_runtime"}
    expected_indices = {"small": 1024, "medium": 4096, "large": 16384}
    for regime in ("small", "medium", "large"):
        descriptors = [cell_descriptor(regime, f"c{i}") for i in range(1, 5)]
        identities = {d["problem_identity"] for d in descriptors}
        assert len(identities) == 1
        assert [d["build"]["compile_meta"]["BLOCK_SIZE_COL"] for d in descriptors] == [128, 256, 512, 1024]
        assert all(d["build"]["compile_meta"]["BLOCK_SIZE_INDEX"] == 1 for d in descriptors)
        assert all(d["launch"]["runtime_args"][1] == f"num_indices={expected_indices[regime]}"
                  for d in descriptors)
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except XFormersConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_XFORMERS_DESCRIPTOR_OK")
