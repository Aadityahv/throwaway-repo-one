#!/usr/bin/env python3
"""CPU-only descriptor for the pinned CUB DeviceReduce::Sum configuration.

This module remains CPU-safe and does not invoke nvcc, compile source, or
touch a GPU. ``cub_device_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that
runner exists (builder carve-out, mirroring the cub-block state below) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_cub_device_reduce"


class CubDeviceConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise CubDeviceConfigError(f"expected one cub-device cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    num_items = int(controls.get("num_items", -1))
    if candidate_id != "c1":
        raise CubDeviceConfigError(f"device reduce exposes a single native configuration, got {candidate_id}")
    if controls != {"num_items": num_items, "api": "DeviceReduce::Sum",
                    "dtype": "float32", "temp_storage_query": True}:
        raise CubDeviceConfigError(f"catalog controls do not match source mapping: {controls}")
    if num_items <= 0 or controls.get("api") != "DeviceReduce::Sum":
        raise CubDeviceConfigError(f"illegal device-reduce configuration: {controls}")
    if int(row["threads_per_block"]) != 256 or int(row["items_per_thread"]) != 1:
        raise CubDeviceConfigError("threads/items must be the catalog placeholders (256, 1)")
    if row["candidate_semantics"] != "single_legal_config" or row["selection_eligible"] != "false":
        raise CubDeviceConfigError("device reduce is a single-configuration prediction fixture, not a selection set")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "cub/cub/device/device_reduce.cuh",
        "source_revision": "d670cb5242c2b855d0286bb7f4921e85543c177b",
        "build": {
            "language": "CUDA C++ (CUB headers, compiled with nvcc)",
            "source_checkout_required": True,
            "compile_entry": "cub::DeviceReduce::Sum with runtime temp-storage query",
            "compile_meta": {"num_items": num_items, "api": "DeviceReduce::Sum",
                             "dtype": "float32", "temp_storage_query": True},
        },
        "launch": {
            "kernel": "CUB device-wide reduction (temp-storage query, then Sum)",
            "grid": "CUB-internal launch configuration (not a catalog axis)",
            "runtime_args": [f"num_items={num_items}", "api=DeviceReduce::Sum",
                             "dtype=float32"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "device-wide sum of num_items elements into one scalar",
            "same_problem_across_candidates": False,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 accumulation order, exact scalar shape"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 3
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"implemented_unvalidated_runtime"}
    expected = {"small": 65536, "medium": 1048576, "large": 16777216}
    for regime, num_items in expected.items():
        d = cell_descriptor(regime, "c1")
        assert d["problem_identity"] == f"{PARENT}:{regime}"
        assert d["build"]["compile_meta"]["num_items"] == num_items
    for bad in (("small", "c2"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except CubDeviceConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_CUB_DEVICE_DESCRIPTOR_OK")
