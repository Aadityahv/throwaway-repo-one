#!/usr/bin/env python3
"""CPU-only descriptor for the pinned ATen index_select gather.

This module remains CPU-safe and does not import Torch, compile source, or
touch a GPU. ``embedding_runner.py`` is the separate source/hash-verified
runtime adapter; cells read ``implemented_unvalidated_runtime`` once that runner
exists (builder carve-out, mirroring the pytorch-layernorm M285 state) and
``compiled_gpu_verified`` only on coordinator acceptance. The CPU descriptor is
not GPU correctness evidence.
The CPU reference in :mod:`correctness_reference` remains a semantic oracle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = "final_pytorch_embedding"


class EmbeddingConfigError(ValueError):
    pass


def _cells() -> list[dict[str, str]]:
    with (ROOT / "workload_cells.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["parent_id"] == PARENT]


def cell_descriptor(regime: str, candidate_id: str) -> dict:
    matches = [r for r in _cells() if r["regime"] == regime and r["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise EmbeddingConfigError(f"expected one embedding cell for {regime}/{candidate_id}")
    row = matches[0]
    controls = json.loads(row["native_controls_json"])
    vocab, dim = int(row["shape_a"]), int(row["shape_b"])
    if vocab <= 0 or dim <= 0:
        raise EmbeddingConfigError(f"non-positive problem shape: {vocab}x{dim}")
    if candidate_id != "c1":
        raise EmbeddingConfigError(f"embedding exposes a single native configuration, got {candidate_id}")
    num_indices = int(controls.get("num_indices", -1))
    if controls != {"num_weights": vocab, "feature_dim": dim, "num_indices": num_indices,
                    "dim": 0, "index_dtype": "int64", "source_layout": "contiguous"}:
        raise EmbeddingConfigError(f"catalog controls do not match source mapping: {controls}")
    if num_indices <= 0 or controls.get("dim") != 0 or controls.get("index_dtype") != "int64":
        raise EmbeddingConfigError(f"illegal gather configuration: {controls}")
    if int(row["threads_per_block"]) != 256 or int(row["items_per_thread"]) != 1:
        raise EmbeddingConfigError("threads/items must be the catalog placeholders (256, 1)")
    if row["candidate_semantics"] != "single_legal_config" or row["selection_eligible"] != "false":
        raise EmbeddingConfigError("embedding is a single-configuration prediction fixture, not a selection set")
    return {
        "parent_id": PARENT,
        "regime": regime,
        "candidate_id": candidate_id,
        "problem_identity": row["problem_identity"],
        "source_status": row["source_status"],
        "implementation_status": row["implementation_status"],
        "runnable": row["implementation_status"] in {"implemented_unvalidated_runtime", "compiled_gpu_verified"},
        "source_path": "aten/src/ATen/native/cuda/Indexing.cu",
        "source_revision": "67faf385dd4133d3e658f9b1c2a4c87dc9b61218",
        "build": {
            "language": "C++/CUDA (ATen, compiled into libtorch)",
            "source_checkout_required": True,
            "compile_entry": "aten::index_select CUDA dispatch (indexSelectSmallIndex/indexSelectLargeIndex)",
            "compile_meta": {"num_weights": vocab, "feature_dim": dim,
                             "num_indices": num_indices, "dim": 0,
                             "index_dtype": "int64"},
        },
        "launch": {
            "kernel": "at::native CUDA index_select kernels via aten::index_select",
            "grid": "ATen-internal launch configuration (not a catalog axis)",
            "runtime_args": [f"num_weights={vocab}", f"feature_dim={dim}",
                             f"num_indices={num_indices}", "dim=0",
                             "index_dtype=int64"],
            "seed": 20260914,
        },
        "semantics": {
            "input_dtype": "float32",
            "output_dtype": "float32",
            "operation": "gather of num_indices rows along dim 0 with int64 indices",
            "same_problem_across_candidates": False,
            "correctness_tolerance": {"rtol": 1e-5, "atol": 1e-6,
                                      "reason": "FP32 rounding of large canonical values; exact index/output shape"},
            "global_read_bytes": int(row["global_read_bytes"]),
            "global_write_bytes": int(row["global_write_bytes"]),
        },
    }


def self_test() -> None:
    rows = _cells()
    assert len(rows) == 3
    assert {r["source_status"] for r in rows} == {"source_verified"}
    assert {r["implementation_status"] for r in rows} == {"implemented_unvalidated_runtime"}
    expected = {"small": (4096, 128, 1024), "medium": (65536, 256, 4096),
                "large": (1048576, 512, 16384)}
    for regime, (vocab, dim, selected) in expected.items():
        d = cell_descriptor(regime, "c1")
        assert d["problem_identity"] == f"{PARENT}:{regime}"
        meta = d["build"]["compile_meta"]
        assert (meta["num_weights"], meta["feature_dim"], meta["num_indices"]) == (vocab, dim, selected)
    for bad in (("small", "c2"), ("bad", "c1")):
        try:
            cell_descriptor(*bad)
        except EmbeddingConfigError:
            pass
        else:
            raise AssertionError(f"malformed control accepted: {bad}")


if __name__ == "__main__":
    self_test()
    print("CPU_ONLY_EMBEDDING_DESCRIPTOR_OK")
