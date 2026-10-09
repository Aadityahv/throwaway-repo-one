#!/usr/bin/env python3
"""CPU-only descriptor for the pinned CUDA Samples vectorAdd tutorial.

This module remains CPU-safe and does not compile source or touch a GPU.
``copy_runner.py`` is the separate source/hash-verified runtime adapter; cells
read ``compiled_gpu_verified`` since coordinator-accepted GPU validation
2026-09-19 (12/12 cells). Do not revert without coordinator direction. The
CPU descriptor is not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_cuda_samples_copy"
CANDIDATE_THREADS = {"c1": 128, "c2": 256, "c3": 512, "c4": 1024}


class CopyConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise CopyConfigError(f"expected one copy cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    threads = CANDIDATE_THREADS.get(candidate_id)
    if threads is None:
        raise CopyConfigError(f"unknown copy candidate {candidate_id}")
    n = int(row["shape_a"])
    expected_blocks = n // threads
    if controls != {"blocks": expected_blocks, "dtype": "float32",
                    "threads": threads, "vector_length": n}:
        raise CopyConfigError(f"catalog controls do not match source mapping: {controls}")
    if n % threads != 0:
        raise CopyConfigError(f"vector length {n} not divisible by threads {threads}")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise CopyConfigError("copy candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "cpp/0_Introduction/vectorAdd/vectorAdd.cu",
        "source_revision": "5443602d89ed99aede2e4b7bf329daddeadb320e",
        "build": {
            "language": "CUDA C++",
            "source_checkout_required": True,
            "compile_entry": "__global__ void vecAdd(float*, float*, float*, int)",
            "compile_meta": {"threads": threads, "blocks": expected_blocks},
        },
        "launch": {
            "kernel": "vecAdd",
            "grid": "(blocks,), one thread per element with workIndex guard",
            "runtime_args": [f"vector_length={n}", f"threads={threads}", f"blocks={expected_blocks}"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "elementwise vector add C[i] = A[i] + B[i]",
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
        assert [d["build"]["compile_meta"]["threads"] for d in descriptors] == [128, 256, 512, 1024]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except CopyConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_COPY_DESCRIPTOR_OK")
