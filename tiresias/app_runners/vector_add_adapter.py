#!/usr/bin/env python3
"""CPU-only descriptor for the pinned Triton vector-add tutorial.

This module remains CPU-safe and does not import Triton, compile source, or
touch a GPU. ``vector_add_runner.py`` is the separate source/hash-verified runtime
adapter; cells read ``compiled_gpu_verified`` since coordinator-accepted GPU validation
2026-09-19 (12/12 cells, M271 evidence). Do not revert without coordinator direction. The
CPU descriptor is not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "train_triton_vector_add"


class VectorAddConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise VectorAddConfigError(f"expected one vector-add cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    n = int(row["shape_a"])
    block_size = int(row["threads_per_block"])
    if controls != {"block_size": block_size, "n": n}:
        raise VectorAddConfigError(f"catalog controls do not match source mapping: {controls}")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise VectorAddConfigError("vector-add candidates must be same-problem and selection eligible")
    if int(row["items_per_thread"]) != 1:
        raise VectorAddConfigError("vector-add items_per_thread must be 1 (BLOCK_SIZE programs)")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "python/tutorials/01-vector-add.py",
        "source_revision": "7369aa3d9b0fae7134bf47b16d83876b09ef8e9f",
        "build": {
            "language": "Python/Triton",
            "source_checkout_required": True,
            "compile_entry": "@triton.jit add_kernel",
            "compile_meta": {"BLOCK_SIZE": block_size},
        },
        "launch": {
            "kernel": "add_kernel",
            "grid": "(cdiv(n_elements, BLOCK_SIZE),), one program per BLOCK_SIZE chunk; masked tail",
            "runtime_args": ["x_ptr", "y_ptr", "output_ptr", f"n_elements={n}",
                             f"BLOCK_SIZE={block_size}"],
            "num_warps": "triton default (not a candidate axis; identical across candidates)",
            "num_stages": "triton default (not a candidate axis; identical across candidates)",
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "elementwise vector add z[i] = x[i] + y[i]",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 add, exact full logical output shape"},
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
        assert [d["build"]["compile_meta"]["BLOCK_SIZE"] for d in descriptors] == [128, 256, 512, 1024]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except VectorAddConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_VECTOR_ADD_DESCRIPTOR_OK")
