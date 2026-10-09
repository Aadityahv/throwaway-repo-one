#!/usr/bin/env python3
"""CPU-only descriptor for the pinned Triton fused-softmax tutorial.

This module remains CPU-safe and does not import Triton, compile source, or
touch a GPU. ``softmax_runner.py`` is the separate source/hash-verified runtime
adapter; its status is ``compiled_gpu_verified`` since coordinator-accepted GPU
validation 2026-09-18 (12/12 cells, M267 follow-up). The CPU descriptor is not GPU
correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "dev_triton_softmax"


class SoftmaxConfigError(ValueError):
    pass


def _positive(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise SoftmaxConfigError(f"{name} must be a positive integer")
    return value


def _power_of_two_at_least(value: int) -> int:
    _positive("cols", value)
    return 1 << (value - 1).bit_length()


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise SoftmaxConfigError(f"expected one softmax cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    rows, cols = int(row["shape_a"]), int(row["shape_b"])
    expected_block = _power_of_two_at_least(cols)
    if controls != {
        "block_size": expected_block,
        "cols": cols,
        "num_stages": int(row["items_per_thread"]),
        "num_warps": int(row["threads_per_block"]) // 32,
        "rows": rows,
        "seed": 20260914,
    }:
        raise SoftmaxConfigError(f"catalog controls do not match source mapping: {controls}")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise SoftmaxConfigError("softmax candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "python/tutorials/02-fused-softmax.py",
        "source_revision": "7369aa3d9b0fae7134bf47b16d83876b09ef8e9f",
        "build": {
            "language": "Python/Triton",
            "source_checkout_required": True,
            "compile_entry": "@triton.jit softmax_kernel",
            "compile_meta": {"BLOCK_SIZE": expected_block,
                              "num_warps": controls["num_warps"],
                              "num_stages": controls["num_stages"]},
        },
        "launch": {
            "kernel": "softmax_kernel",
            "grid": "(rows, 1, 1), one source-kernel program per logical row; the kernel's tl.num_programs(0) row stride covers all rows",
            "runtime_args": ["output_ptr", "input_ptr", "input_row_stride=cols",
                             "output_row_stride=cols", f"n_rows={rows}", f"n_cols={cols}",
                             f"BLOCK_SIZE={expected_block}", f"num_stages={controls['num_stages']}"],
            "num_warps": controls["num_warps"],
            "num_stages": controls["num_stages"],
            "seed": controls["seed"],
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "row-wise stable softmax",
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
    assert {r["source_status"] for r in rows} == {"source_audited"}
    assert {r["implementation_status"] for r in rows} == {"compiled_gpu_verified"}
    for regime in ("small", "medium", "large"):
        descriptors = [cell_descriptor(regime, f"c{i}") for i in range(1, 5)]
        identities = {d["problem_identity"] for d in descriptors}
        assert len(identities) == 1
        assert {(d["launch"]["num_warps"], d["launch"]["num_stages"]) for d in descriptors} == {
            (2, 2), (4, 2), (8, 2), (8, 4)
        }
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except SoftmaxConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_SOFTMAX_DESCRIPTOR_OK: runtime adapter implemented; GPU validation accepted 2026-09-18")
