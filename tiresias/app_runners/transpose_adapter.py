#!/usr/bin/env python3
"""CPU-only descriptor for the pinned CUDA Samples transpose tutorial.

This module remains CPU-safe and does not compile source or touch a GPU.
``transpose_runner.py`` is the separate source/hash-verified runtime adapter;
cells read ``compiled_gpu_verified`` since coordinator-accepted GPU validation
2026-09-19 (12/12 cells). Do not revert without coordinator direction. The
CPU descriptor is not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.

Launch geometry note (audited): every full-transpose kernel steps its tile
loops by the hardcoded BLOCK_ROWS=16, so blocks are dim3(32, 16) = 512
threads. A 256-thread (32, 8) block would cover only half of each 32-row tile.
The catalog records the runnable 512-thread configuration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "train_cuda_samples_transpose"
CANDIDATE_VARIANTS = {"c1": 2, "c2": 3, "c3": 4, "c4": 7}
VARIANT_NAMES = {2: "transposeNaive", 3: "transposeCoalesced",
                 4: "transposeNoBankConflicts", 7: "transposeDiagonal"}
TILE_DIM = 32
BLOCK_ROWS = 16


class TransposeConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise TransposeConfigError(f"expected one transpose cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    variant = CANDIDATE_VARIANTS.get(candidate_id)
    if variant is None:
        raise TransposeConfigError(f"unknown transpose candidate {candidate_id}")
    n = int(row["shape_a"])
    if int(row["shape_b"]) != n:
        raise TransposeConfigError("transpose matrices must be square")
    if n % TILE_DIM != 0:
        raise TransposeConfigError(f"dimension {n} not a multiple of TILE_DIM (kernel requirement)")
    if controls != {"block_rows": BLOCK_ROWS, "dim_x": n, "dim_y": n, "dtype": "float32",
                    "kernel_variant": variant, "threads_per_block": TILE_DIM * BLOCK_ROWS,
                    "tile_dim": TILE_DIM}:
        raise TransposeConfigError(f"catalog controls do not match source mapping: {controls}")
    if int(row["threads_per_block"]) != TILE_DIM * BLOCK_ROWS:
        raise TransposeConfigError("threads_per_block must equal TILE_DIM x BLOCK_ROWS (512)")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise TransposeConfigError("transpose candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "cpp/6_Performance/transpose/transpose.cu",
        "source_revision": "5443602d89ed99aede2e4b7bf329daddeadb320e",
        "build": {
            "language": "CUDA C++",
            "source_checkout_required": True,
            "compile_entry": f"__global__ void {VARIANT_NAMES[variant]}(float*, float*, int, int)",
            "compile_meta": {"kernel_variant": variant, "threads": (TILE_DIM, BLOCK_ROWS)},
        },
        "launch": {
            "kernel": VARIANT_NAMES[variant],
            "grid": "(dim_x/32, dim_y/32), one 32x32 tile per block",
            "grid_dims": {"rows": n, "cols": n},
            "runtime_args": [f"dim_x={n}", f"dim_y={n}",
                             f"threads_per_block={TILE_DIM * BLOCK_ROWS}"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "out-of-place full matrix transpose",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 permutation, exact full logical output shape"},
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
        assert [d["build"]["compile_meta"]["kernel_variant"] for d in descriptors] == [2, 3, 4, 7]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except TransposeConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_TRANSPOSE_DESCRIPTOR_OK")
