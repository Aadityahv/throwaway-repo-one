#!/usr/bin/env python3
"""CPU-only descriptor for the pinned CUDA Samples reduction tutorial.

This module remains CPU-safe and does not compile source or touch a GPU.
``reduction_runner.py`` is the separate source/hash-verified runtime adapter;
cells read ``compiled_gpu_verified`` since coordinator-accepted GPU validation
2026-09-19 (12/12 cells). Do not revert without coordinator direction. The
CPU descriptor is not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.

Reduction model (pinned `reduce<T>()` dispatcher): one kernel call produces one
partial sum per block; the full sum is completed by a host sum over the partials
(the sample's `--cpufinal` mode). Candidates vary whichKernel (2/3/4/5) on the
same input, launch geometry, and host completion.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "train_cuda_samples_reduction"
CANDIDATE_KERNELS = {"c1": 2, "c2": 3, "c3": 4, "c4": 5}


class ReductionConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ReductionConfigError(f"expected one reduction cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    which_kernel = CANDIDATE_KERNELS.get(candidate_id)
    if which_kernel is None:
        raise ReductionConfigError(f"unknown reduction candidate {candidate_id}")
    size = int(row["shape_a"])
    threads = int(row["threads_per_block"])
    if size % threads != 0:
        raise ReductionConfigError(f"size {size} not divisible by threads {threads}")
    blocks = size // threads
    if controls != {"blocks": blocks, "dtype": "float32", "size": size,
                    "threads": threads, "which_kernel": which_kernel}:
        raise ReductionConfigError(f"catalog controls do not match source mapping: {controls}")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise ReductionConfigError("reduction candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu",
        "source_revision": "5443602d89ed99aede2e4b7bf329daddeadb320e",
        "build": {
            "language": "CUDA C++",
            "source_checkout_required": True,
            "compile_entry": f"reduce<float>(size, threads, blocks, whichKernel={which_kernel}) dispatcher",
            "compile_meta": {"which_kernel": which_kernel, "threads": threads, "blocks": blocks},
        },
        "launch": {
            "kernel": f"reduce{which_kernel}",
            "grid": "(blocks,) 1D, one partial sum per block, host-completed full sum",
            "runtime_args": [f"size={size}", f"threads={threads}", f"blocks={blocks}",
                             f"which_kernel={which_kernel}", "cpufinal=true"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "full-array sum via per-block partials plus host final sum",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 summation-order differences, exact shape"},
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
        assert [d["build"]["compile_meta"]["which_kernel"] for d in descriptors] == [2, 3, 4, 5]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except ReductionConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_REDUCTION_DESCRIPTOR_OK")
