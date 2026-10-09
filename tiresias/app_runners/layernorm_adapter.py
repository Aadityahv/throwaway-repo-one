#!/usr/bin/env python3
"""CPU-only descriptor for the pinned Triton layer-norm tutorial.

This module remains CPU-safe and does not import Triton, compile source, or
touch a GPU. ``layernorm_runner.py`` is the separate source/hash-verified runtime
adapter; cells read ``compiled_gpu_verified`` since coordinator-accepted GPU validation
2026-09-19 (12/12 cells, M273 evidence). Do not revert without coordinator direction.
The CPU descriptor is not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_triton_layer_norm"
CANDIDATE_WARPS = {"c1": 1, "c2": 2, "c3": 4, "c4": 8}


class LayerNormConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise LayerNormConfigError(f"expected one layer-norm cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    rows, cols = int(row["shape_a"]), int(row["shape_b"])
    if rows <= 0 or cols <= 0:
        raise LayerNormConfigError(f"non-positive problem shape: {rows}x{cols}")
    num_warps = CANDIDATE_WARPS.get(candidate_id)
    if num_warps is None:
        raise LayerNormConfigError(f"unknown layer-norm candidate {candidate_id}")
    expected_block = 1
    while expected_block < cols:
        expected_block *= 2
    if controls != {
        "block_size": expected_block,
        "cols": cols,
        "dtype": "float32",
        "eps": 1e-05,
        "num_warps": num_warps,
        "rows": rows,
    }:
        raise LayerNormConfigError(f"catalog controls do not match source mapping: {controls}")
    if int(row["threads_per_block"]) != num_warps * 32:
        raise LayerNormConfigError("threads_per_block must equal num_warps x 32")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise LayerNormConfigError("layer-norm candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "python/tutorials/05-layer-norm.py",
        "source_revision": "7369aa3d9b0fae7134bf47b16d83876b09ef8e9f",
        "build": {
            "language": "Python/Triton",
            "source_checkout_required": True,
            "compile_entry": "@triton.jit _layer_norm_fwd_fused (forward pass only)",
            "compile_meta": {"BLOCK_SIZE": expected_block,
                             "num_warps": num_warps},
        },
        "launch": {
            "kernel": "_layer_norm_fwd_fused",
            "grid": "(rows,), one source-kernel program per logical row",
            "runtime_args": ["x_ptr", "y_ptr", "w_ptr", "b_ptr", "mean_ptr", "rstd_ptr",
                             f"n_rows={rows}", f"n_cols={cols}", f"stride={cols}",
                             "eps=1e-05",
                             f"BLOCK_SIZE={expected_block}", f"num_warps={num_warps}"],
            "num_warps": num_warps,
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "row-wise layer norm y = (x - mean)/sqrt(var+eps) * w + b",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 2e-5, "atol": 2e-5,
                                      "reason": "Triton exp/reduction and FP32 accumulation"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 12
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"compiled_gpu_verified"}
    for regime in ("small", "medium", "large"):
        descriptors = [cell_descriptor(regime, f"c{i}") for i in range(1, 5)]
        identities = {d["problem_identity"] for d in descriptors}
        assert len(identities) == 1
        assert [d["build"]["compile_meta"]["num_warps"] for d in descriptors] == [1, 2, 4, 8]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except LayerNormConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_LAYERNORM_DESCRIPTOR_OK")
