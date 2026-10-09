#!/usr/bin/env python3
"""CPU-only descriptor for the pinned ATen LayerNorm forward kernels.

This module remains CPU-safe and does not import Torch, compile source, or
touch a GPU. ``pytorch_layernorm_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that runner
exists (builder carve-out, mirroring the pytorch-softmax M283 state) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_pytorch_layer_norm"
# Catalog stride ladder: stride = cols + pad, shared with the builder.
ROW_STRIDE_PADS = (0, 1, 3, 7)


class PyTorchLayerNormConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise PyTorchLayerNormConfigError(f"expected one pytorch-layernorm cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    rows, cols = int(row["shape_a"]), int(row["shape_b"])
    if rows <= 0 or cols <= 0:
        raise PyTorchLayerNormConfigError(f"non-positive problem shape: {rows}x{cols}")
    stride = int(controls.get("row_stride", -1))
    if stride < cols or stride - cols not in ROW_STRIDE_PADS:
        raise PyTorchLayerNormConfigError(f"row stride {stride} is not the catalog ladder for cols={cols}")
    if controls != {"rows": rows, "cols": cols, "row_stride": stride,
                    "eps": 1e-5, "dtype": "float32", "affine": True}:
        raise PyTorchLayerNormConfigError(f"catalog controls do not match source mapping: {controls}")
    if int(row["threads_per_block"]) != 256 or int(row["items_per_thread"]) != stride:
        raise PyTorchLayerNormConfigError("threads/items must be the catalog placeholders (256, stride)")
    if row["candidate_semantics"] != "same_problem_legal_config" or row["selection_eligible"] != "true":
        raise PyTorchLayerNormConfigError("pytorch-layernorm candidates must be same-problem and selection eligible")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
        "source_revision": "67faf385dd4133d3e658f9b1c2a4c87dc9b61218",
        "build": {
            "language": "C++/CUDA (ATen, compiled into libtorch)",
            "source_checkout_required": True,
            "compile_entry": "aten::layer_norm CUDA dispatch (rowwise forward kernels)",
            "compile_meta": {"rows": rows, "cols": cols, "row_stride": stride,
                             "eps": 1e-5, "affine": True},
        },
        "launch": {
            "kernel": "at::native CUDA layer-norm kernels via aten::layer_norm",
            "grid": "ATen-internal launch configuration (not a catalog axis)",
            "runtime_args": [f"rows={rows}", f"cols={cols}", f"row_stride={stride}",
                             "eps=1e-05", "affine=true"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "row-wise layer norm with padded input stride, identity affine",
            "same_problem_across_candidates": True,
            "correctness_tolerance": {"criterion": "close(abs<=1e-5*max(1,|a|,|b|))",
                                      "reason": "FP32 accumulation, exact full logical output shape"},
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
        except PyTorchLayerNormConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_PYTORCH_LAYERNORM_DESCRIPTOR_OK")
