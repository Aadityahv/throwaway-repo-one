#!/usr/bin/env python3
"""CPU-only descriptor for the pinned CUB BlockReduce test geometry.

This module remains CPU-safe and does not invoke nvcc, compile source, or
touch a GPU. ``cub_block_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that
runner exists (builder carve-out, mirroring the CUDA-samples M274 state) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "train_cub_block_reduce"
# Catalog ladder: (BlockDimX, ItemsPerThread); tile is always 1024 elements.
CANDIDATE_GEOMETRY = {"c1": (32, 32), "c2": (64, 16), "c3": (128, 8), "c4": (256, 4)}
VALID_ITEMS = 1024


class CubBlockConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise CubBlockConfigError(f"expected one cub-block cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    geometry = CANDIDATE_GEOMETRY.get(candidate_id)
    if geometry is None:
        raise CubBlockConfigError(f"unknown cub-block candidate {candidate_id}")
    block_x, items_per_thread = geometry
    if controls != {"block_dim_x": block_x, "block_dim_y": 1, "block_dim_z": 1,
                    "items_per_thread": items_per_thread, "valid_items": VALID_ITEMS,
                    "dtype": "float32"}:
        raise CubBlockConfigError(f"catalog controls do not match source mapping: {controls}")
    if int(row["threads_per_block"]) != block_x or int(row["items_per_thread"]) != items_per_thread:
        raise CubBlockConfigError("threads/items must equal the catalog BlockDimX/ItemsPerThread")
    if block_x * items_per_thread != VALID_ITEMS:
        raise CubBlockConfigError("geometry tile must cover exactly the 1024 valid items")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise CubBlockConfigError("cub-block candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "cub/test/catch2_test_block_reduce.cu",
        "source_revision": "d670cb5242c2b855d0286bb7f4921e85543c177b",
        "build": {
            "language": "CUDA C++ (CUB headers, compiled with nvcc)",
            "source_checkout_required": True,
            "compile_entry": "cub::BlockReduce<float, BlockDimX, ItemsPerThread>::Sum",
            "compile_meta": {"block_dim_x": block_x, "block_dim_y": 1, "block_dim_z": 1,
                             "items_per_thread": items_per_thread,
                             "valid_items": VALID_ITEMS},
        },
        "launch": {
            "kernel": "harness block_reduce kernel (one block; thread t reduces its ItemsPerThread slice)",
            "grid": "single block of BlockDimX threads (not a catalog axis)",
            "runtime_args": [f"valid_items={VALID_ITEMS}", f"block_dim_x={block_x}",
                             f"items_per_thread={items_per_thread}"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "block-wide sum of 1024 elements into one scalar",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 accumulation order, exact scalar shape"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 12
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"implemented_unvalidated_runtime"}
    for regime in ("small", "medium", "large"):
        descriptors = [cell_descriptor(regime, f"c{i}") for i in range(1, 5)]
        identities = {d["problem_identity"] for d in descriptors}
        assert len(identities) == 1
        assert [(d["build"]["compile_meta"]["block_dim_x"],
                 d["build"]["compile_meta"]["items_per_thread"]) for d in descriptors] == [
            (32, 32), (64, 16), (128, 8), (256, 4)]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except CubBlockConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_CUB_BLOCK_DESCRIPTOR_OK")
