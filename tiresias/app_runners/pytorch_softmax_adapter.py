#!/usr/bin/env python3
"""CPU-only descriptor for the pinned ATen row-wise softmax.

This module remains CPU-safe and does not import Torch, compile source, or
touch a GPU. ``pytorch_softmax_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that runner
exists (builder carve-out, mirroring the vector-add M269 state) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "dev_pytorch_rowwise_softmax"
# Catalog stride ladder: stride = cols + pad, shared with the builder.
ROW_STRIDE_PADS = (0, 1, 3, 7)


class PyTorchSoftmaxConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise PyTorchSoftmaxConfigError(f"expected one pytorch-softmax cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    rows, cols = int(row["shape_a"]), int(row["shape_b"])
    if rows <= 0 or cols <= 0:
        raise PyTorchSoftmaxConfigError(f"non-positive problem shape: {rows}x{cols}")
    stride = int(controls.get("input_row_stride", -1))
    if stride < cols or stride - cols not in ROW_STRIDE_PADS:
        raise PyTorchSoftmaxConfigError(f"row stride {stride} is not the catalog ladder for cols={cols}")
    if controls != {"outer_size": rows, "dim_size": cols, "inner_size": 1,
                    "input_row_stride": stride, "output_row_stride": stride,
                    "layout": "padded_row_major", "dtype": "float32"}:
        raise PyTorchSoftmaxConfigError(f"catalog controls do not match source mapping: {controls}")
    if int(row["threads_per_block"]) != 256 or int(row["items_per_thread"]) != stride:
        raise PyTorchSoftmaxConfigError("threads/items must be the catalog placeholders (256, stride)")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise PyTorchSoftmaxConfigError("pytorch-softmax candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "aten/src/ATen/native/cuda/SoftMax.cu",
        "source_revision": "67faf385dd4133d3e658f9b1c2a4c87dc9b61218",
        "build": {
            "language": "C++/CUDA (ATen, compiled into libtorch)",
            "source_checkout_required": True,
            "compile_entry": "aten::softmax CUDA dispatch (spatial forward kernels)",
            "compile_meta": {"outer_size": rows, "dim_size": cols, "inner_size": 1,
                             "row_stride": stride},
        },
        "launch": {
            "kernel": "at::native CUDA softmax kernels via aten::softmax",
            "grid": "ATen-internal launch configuration (not a catalog axis)",
            "runtime_args": [f"outer_size={rows}", f"dim_size={cols}", "inner_size=1",
                             f"input_row_stride={stride}", f"output_row_stride={stride}"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "row-wise softmax over dim_size with padded row stride",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"rtol": 2e-5, "atol": 2e-5,
                                      "reason": "FP32 exp/reduction, exact full logical output shape"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 12
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"implemented_unvalidated_runtime"}
    expected_strides = {"small": [128, 129, 131, 135], "medium": [512, 513, 515, 519],
                        "large": [1024, 1025, 1027, 1031]}
    for regime in ("small", "medium", "large"):
        descriptors = [cell_descriptor(regime, f"c{i}") for i in range(1, 5)]
        identities = {d["problem_identity"] for d in descriptors}
        assert len(identities) == 1
        assert [d["build"]["compile_meta"]["row_stride"] for d in descriptors] == expected_strides[regime]
    for bad in (("small", "c9"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except PyTorchSoftmaxConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_PYTORCH_SOFTMAX_DESCRIPTOR_OK")
