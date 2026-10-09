#!/usr/bin/env python3
"""CPU-only, source-backed static features for supported reset kernels.

The extractor intentionally has a small supported surface.  It reads the C
catalog and the corresponding CUDA source, derives work from the actual launch
controls, and never reads a measurement directory or substitutes a measured
counter for static work.  A ``cell_id`` identifies the operation/configuration
independently of a platform.  A separate ``context_id`` is required for any
prediction/measurement join and binds platform, compiler/build, source and
measurement-protocol provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CELLS = ROOT / "workload_cells.csv"
SCHEMA_VERSION = "reset-static-features-v2"
SUPPORTED = {"train_shared_stage", "dev_local_triad", "dev_local_transpose", "dev_triton_softmax",
             "dev_pytorch_rowwise_softmax", "train_cuda_samples_reduction",
             "train_cuda_samples_transpose", "final_cuda_samples_copy",
             "train_triton_vector_add", "final_triton_layer_norm",
             "final_pytorch_layer_norm", "final_pytorch_embedding",
             "final_xformers_indexed_select", "train_cub_block_reduce",
             "final_cub_device_reduce"}
SOURCE_BY_PARENT = {
    "train_shared_stage": "workloads/predictor_shared_stage_load.cu",
    "dev_local_triad": "workloads/e2e_tile_triad_calibration.cu",
    "dev_local_transpose": "workloads/e2e_transpose.cu",
}
FLOAT_BYTES = 4
SOFTMAX_PROVENANCE = ROOT / "triton_softmax_source_provenance.json"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _hash_optional(value: Any) -> str | None:
    return None if value is None else sha256_bytes(canonical(value).encode())


def state(value: Any = None, *, origin: str = "static", units: str = "n/a",
          source: str = "", status: str = "supported", note: str = "") -> dict[str, Any]:
    out = {"value": value, "origin": origin, "units": units, "source": source, "status": status}
    if note:
        out["note"] = note
    return out


def unsupported(units: str, source: str, note: str) -> dict[str, Any]:
    return state(None, origin="static", units=units, source=source,
                 status="unsupported", note=note)


def _row(parent: str, regime: str, candidate: str) -> dict[str, str]:
    with CELLS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if (r["parent_id"], r["regime"], r["candidate_id"]) ==
               (parent, regime, candidate)]
    if len(matches) != 1:
        raise ValueError(f"expected one catalog cell, found {len(matches)} for {parent}/{regime}/{candidate}")
    return matches[0]


def _source_path(row: dict[str, str]) -> Path | None:
    # Neither the Triton tutorial, the ATen sources, nor the CUDA-samples files
    # are vendored: C's runners must verify the exact upstream checkout before
    # compiling it. Committed provenance plus runtime evidence are the
    # source-binding inputs here.
    if row["parent_id"] in ("dev_triton_softmax", "dev_pytorch_rowwise_softmax",
                            "train_cuda_samples_reduction", "train_cuda_samples_transpose",
                            "final_cuda_samples_copy", "train_triton_vector_add",
                            "final_triton_layer_norm", "final_pytorch_layer_norm",
                            "final_pytorch_embedding", "final_xformers_indexed_select",
                            "train_cub_block_reduce", "final_cub_device_reduce"):
        return None
    path = REPO / SOURCE_BY_PARENT[row["parent_id"]]
    if not path.is_file():
        raise ValueError(f"declared source is unavailable: {path}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _source_checked(parent: str, source_text: str) -> None:
    """Refuse to apply a formula to a source whose defining accesses changed."""
    required = {
        "dev_local_triad": ("a[i]", "b[i]", "c[i]", "out[i]", "__global__ void triad"),
        "dev_local_transpose": ("input[", "output[", "__shared__ float tile", "__syncthreads"),
        "train_shared_stage": ("stage[i]", "sink[", "__shared__ float stage", "__syncthreads",
                                "for (std::uint32_t batch", "for (std::uint32_t chunk"),
    }[parent]
    missing = [token for token in required if token not in source_text]
    if missing:
        raise ValueError(f"source no longer matches the supported {parent} formula; missing {missing}")


def _softmax_source_and_controls(regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    """Bind D's formula to C's exact pinned source and validated launch mapping.

    We do not substitute a local reimplementation for the upstream Triton
    tutorial.  ``softmax_adapter`` verifies the native controls against the
    catalog, while the provenance record is the hash C's GPU runner checked
    before the accepted 12-cell execution.
    """
    try:
        from softmax_adapter import cell_descriptor
    except ImportError as exc:  # pragma: no cover - direct-script import guard
        raise ValueError("pinned softmax adapter is unavailable") from exc
    if not SOFTMAX_PROVENANCE.is_file():
        raise ValueError("pinned softmax provenance record is unavailable")
    provenance = json.loads(SOFTMAX_PROVENANCE.read_text())
    descriptor = cell_descriptor(regime, candidate)
    required = ("source_sha256", "source_path", "revision")
    if any(not isinstance(provenance.get(k), str) or not provenance[k] for k in required):
        raise ValueError("pinned softmax provenance is incomplete")
    if (descriptor["source_path"] != provenance["source_path"] or
            descriptor["source_revision"] != provenance["revision"]):
        raise ValueError("softmax adapter/provenance source identity mismatch")
    return provenance["source_sha256"], descriptor


UPSTREAM_PROVENANCE = ROOT / "upstream_source_provenance_2026-09-16.json"
PYTORCH_SOFTMAX_DISPLAY_PATH = "aten/src/ATen/native/cuda/SoftMax.cu"
PYTORCH_LAYERNORM_DISPLAY_PATH = "aten/src/ATen/native/cuda/layer_norm_kernel.cu"
PYTORCH_EMBEDDING_DISPLAY_PATH = "aten/src/ATen/native/cuda/Indexing.cu"


def _pytorch_softmax_source_and_controls(regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    return _aten_source_and_controls("dev_pytorch_rowwise_softmax", "pytorch_softmax_adapter",
                                     regime, candidate)


def _aten_source_and_controls(parent: str, adapter_module: str,
                              regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    """Bind D's formula to C's exact pinned ATen source and validated controls.

    Same binding shape as the Triton path: the per-parent adapter verifies the
    native controls against the catalog, while the committed upstream
    provenance record carries the hash C's GPU runner checks before execution.
    The ATen file itself is C++ compiled into libtorch, so no local token check
    applies; the revision-plus-hash binding is the source anchor.
    """
    import importlib
    try:
        adapter = importlib.import_module(adapter_module)
    except ImportError as exc:  # pragma: no cover - direct-script import guard
        raise ValueError(f"pinned {parent} adapter is unavailable") from exc
    if not UPSTREAM_PROVENANCE.is_file():
        raise ValueError("pinned upstream provenance record is unavailable")
    sources = json.loads(UPSTREAM_PROVENANCE.read_text()).get("sources", {})
    provenance = sources.get(parent, {})
    required = ("sha256", "path", "revision")
    if any(not isinstance(provenance.get(k), str) or not provenance[k] for k in required):
        raise ValueError(f"pinned {parent} provenance is incomplete")
    descriptor = adapter.cell_descriptor(regime, candidate)
    if (descriptor["source_path"] != provenance["path"] or
            descriptor["source_revision"] != provenance["revision"]):
        raise ValueError(f"{parent} adapter/provenance source identity mismatch")
    return provenance["sha256"], descriptor


PINNED_UPSTREAM_ADAPTERS = {
    "train_cuda_samples_reduction": "reduction_adapter",
    "train_cuda_samples_transpose": "transpose_adapter",
    "final_cuda_samples_copy": "copy_adapter",
    "final_xformers_indexed_select": "xformers_adapter",
    "train_cub_block_reduce": "cub_block_adapter",
    "final_cub_device_reduce": "cub_device_adapter",
}
PINNED_UPSTREAM_DISPLAY_PATHS = {
    "train_cuda_samples_reduction": "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu",
    "train_cuda_samples_transpose": "cpp/6_Performance/transpose/transpose.cu",
    "final_cuda_samples_copy": "cpp/0_Introduction/vectorAdd/vectorAdd.cu",
    "final_xformers_indexed_select": "xformers/ops/_triton/k_index_select_cat.py",
    "train_cub_block_reduce": "cub/test/catch2_test_block_reduce.cu",
    "final_cub_device_reduce": "cub/cub/device/device_reduce.cuh",
}


def _pinned_upstream_source_and_controls(parent: str, regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    """Bind D's formula to C's exact pinned CUDA-samples file and validated controls.

    Same binding shape as the softmax paths: the per-family adapter verifies the
    native controls against the catalog, while the committed upstream provenance
    record carries the hash C's nvcc runner checks before compiling. The sample
    files are never compiled here; per-variant structure (staging, sync sites)
    was read off the pinned bytes (see the branch notes below).
    """
    import importlib
    try:
        adapter = importlib.import_module(PINNED_UPSTREAM_ADAPTERS[parent])
    except ImportError as exc:  # pragma: no cover - direct-script import guard
        raise ValueError(f"pinned {parent} adapter is unavailable") from exc
    if not UPSTREAM_PROVENANCE.is_file():
        raise ValueError("pinned upstream provenance record is unavailable")
    sources = json.loads(UPSTREAM_PROVENANCE.read_text()).get("sources", {})
    provenance = sources.get(parent, {})
    required = ("sha256", "path", "revision")
    if any(not isinstance(provenance.get(k), str) or not provenance[k] for k in required):
        raise ValueError(f"pinned {parent} provenance is incomplete")
    descriptor = adapter.cell_descriptor(regime, candidate)
    if (descriptor["source_path"] != provenance["path"] or
            descriptor["source_revision"] != provenance["revision"]):
        raise ValueError(f"{parent} adapter/provenance source identity mismatch")
    return provenance["sha256"], descriptor


TRITON_VECTOR_ADD_DISPLAY_PATH = "python/tutorials/01-vector-add.py"


# Triton 3.6.0's CUDA default num_warps, used when a launch passes none. Confirmed per
# cell by the M311 compile cache (num_warps=4 in every add_kernel entry, M328 raw
# evidence 03_triton_vector_add.txt).
TRITON_DEFAULT_NUM_WARPS = 4


def _triton_vector_add_source_and_controls(regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    """Bind D's formula to C's exact pinned tutorial and validated BLOCK_SIZE mapping.

    There is no separate provenance JSON for this parent; the binding record is
    the runner's own pinned constants (the values its verify_source checked
    before the accepted M271 12-cell execution), cross-checked against the
    adapter's source identity. Same shape as the other upstream bindings.
    """
    try:
        from vector_add_adapter import cell_descriptor
        import vector_add_runner as runner
    except ImportError as exc:  # pragma: no cover - direct-script import guard
        raise ValueError("pinned vector-add adapter/runner is unavailable") from exc
    descriptor = cell_descriptor(regime, candidate)
    if (descriptor["source_path"] != TRITON_VECTOR_ADD_DISPLAY_PATH or
            descriptor["source_revision"] != runner.PINNED_REVISION):
        raise ValueError("vector-add adapter/runner source identity mismatch")
    if not isinstance(runner.PINNED_SHA256, str) or len(runner.PINNED_SHA256) != 64:
        raise ValueError("vector-add runner pinned hash is malformed")
    return runner.PINNED_SHA256, descriptor


TRITON_LAYERNORM_DISPLAY_PATH = "python/tutorials/05-layer-norm.py"


def _triton_layernorm_source_and_controls(regime: str, candidate: str) -> tuple[str, dict[str, Any]]:
    """Bind D's formula to C's exact pinned tutorial and validated launch mapping.

    No separate provenance JSON exists for this parent; the binding record is
    the runner's own pinned constants (the values its verify_source checked
    before the accepted M273 12-cell execution), cross-checked against the
    adapter's source identity. Same shape as the vector-add binding.
    """
    try:
        from layernorm_adapter import cell_descriptor
        import layernorm_runner as runner
    except ImportError as exc:  # pragma: no cover - direct-script import guard
        raise ValueError("pinned layernorm adapter/runner is unavailable") from exc
    descriptor = cell_descriptor(regime, candidate)
    if (descriptor["source_path"] != TRITON_LAYERNORM_DISPLAY_PATH or
            descriptor["source_revision"] != runner.PINNED_REVISION):
        raise ValueError("layernorm adapter/runner source identity mismatch")
    if not isinstance(runner.PINNED_SHA256, str) or len(runner.PINNED_SHA256) != 64:
        raise ValueError("layernorm runner pinned hash is malformed")
    return runner.PINNED_SHA256, descriptor


def _controls(row: dict[str, str], parent: str) -> dict[str, Any]:
    raw = row["native_controls_json"]
    if not raw:
        raise ValueError(f"{parent} cell has no native controls")
    controls = json.loads(raw)
    if not isinstance(controls, dict):
        raise ValueError("native_controls_json must be an object")
    return controls


def _require_controls(controls: dict[str, Any], names: tuple[str, ...], parent: str) -> None:
    missing = [name for name in names if name not in controls]
    if missing:
        raise ValueError(f"{parent} native controls missing required fields: {', '.join(missing)}")


def _add(fields: dict[str, dict[str, Any]], name: str, value: Any, units: str,
         origin: str, source: str) -> None:
    fields[name] = state(value, units=units, origin=origin, source=source)


def _context(*, source_hash: str, compiler: dict[str, Any] | None,
             platform: dict[str, Any] | None, build: Any,
             measurement_protocol: Any) -> dict[str, Any]:
    """Return explicit provenance; incomplete context is deliberately unjoinable."""
    out = {
        "platform_profile_sha256": _hash_optional(platform),
        "compiler_metadata_sha256": _hash_optional(compiler),
        "build_sha256": _hash_optional(build),
        "source_sha256": source_hash,
        "measurement_protocol_sha256": _hash_optional(measurement_protocol),
    }
    complete = all(out[k] for k in ("platform_profile_sha256", "source_sha256",
                                    "measurement_protocol_sha256")) and bool(out["build_sha256"] or out["compiler_metadata_sha256"])
    out["status"] = "complete" if complete else "incomplete"
    if not complete:
        out["note"] = ("A scientific join requires platform profile, source, build/compiler and "
                        "measurement-protocol identity; CPU extraction without these is unjoinable.")
    return out


def _resource_fields(fields: dict[str, dict[str, Any]], compiler: dict[str, Any] | None,
                     platform: dict[str, Any] | None, *, threads: int, blocks: int,
                     launch_known: bool = True, threads_known: bool = True) -> None:
    """Add compiler resources and bounded occupancy when allocation units are known."""
    resource_names = (("registers_per_thread", "registers/thread"),
                      ("shared_bytes_per_block", "bytes/CTA"))
    if compiler is None:
        for name, units in resource_names:
            fields[name] = unsupported(units, "compiler metadata", f"Provide ptxas metadata for {name}.")
        fields["theoretical_occupancy_blocks_per_sm"] = unsupported("resident blocks/SM", "compiler + platform metadata", "Compiler resources and platform limits are absent.")
        fields["waves_per_sm"] = unsupported("waves/SM", "compiler + platform metadata", "Compiler resources and platform limits are absent.")
        return
    for name, units in resource_names:
        if name not in compiler:
            fields[name] = unsupported(units, "compiler metadata", f"Missing {name}; no fallback is permitted.")
        else:
            (_positive_int if name == "registers_per_thread" else _nonnegative_int)(compiler[name], name)
            _add(fields, name, compiler[name], units, "static", "caller-provided compiler metadata")
    missing_compiler = [name for name, _ in resource_names if name not in compiler]
    if missing_compiler:
        note = ("Missing compiler resource metadata: " + ", ".join(missing_compiler) +
                "; occupancy and waves are unsupported rather than inferred.")
        fields["theoretical_occupancy_blocks_per_sm"] = unsupported(
            "resident blocks/SM", "compiler + platform metadata", note)
        fields["waves_per_sm"] = unsupported("waves/SM", "compiler + platform metadata", note)
        return
    if not launch_known or not threads_known:
        note = ("Block/CTA occupancy needs launch geometry and threads-per-program; "
                "at least one is not recovered statically, so occupancy and waves are unsupported.")
        fields["theoretical_occupancy_blocks_per_sm"] = unsupported(
            "resident blocks/SM", "launch geometry", note)
        fields["waves_per_sm"] = unsupported("waves/SM", "launch geometry", note)
        return
    if platform is None:
        fields["theoretical_occupancy_blocks_per_sm"] = unsupported("resident blocks/SM", "compiler + platform metadata", "Platform limits are absent.")
        fields["waves_per_sm"] = unsupported("waves/SM", "compiler + platform metadata", "Platform limits are absent.")
        return
    required = ("sm_count", "max_blocks_per_sm", "max_threads_per_sm", "registers_per_sm",
                "shared_bytes_per_sm", "warp_size", "max_warps_per_sm",
                "register_allocation_unit", "shared_allocation_unit", "shared_reserved_per_block")
    missing = [k for k in required if k not in platform]
    if missing:
        note = f"Missing bounded occupancy metadata: {', '.join(missing)}."
        fields["theoretical_occupancy_blocks_per_sm"] = unsupported("resident blocks/SM", "compiler + platform metadata", note)
        fields["waves_per_sm"] = unsupported("waves/SM", "compiler + platform metadata", note)
        return
    for name in required:
        (_nonnegative_int if name == "shared_reserved_per_block" else _positive_int)(platform[name], f"platform.{name}")
    if threads > platform["max_threads_per_sm"]:
        raise ValueError("block has more threads than platform max_threads_per_sm")
    warp = platform["warp_size"]
    warps = math.ceil(threads / warp)
    regs_per_warp = math.ceil(compiler["registers_per_thread"] * warp /
                              platform["register_allocation_unit"]) * platform["register_allocation_unit"]
    regs_block = regs_per_warp * warps
    # CUDA's driver reserves a fixed per-block shared-memory carveout on top of the kernel's own
    # declared static+dynamic shared memory before rounding to the allocation granularity -- matches
    # cuda_occupancy.h's staticSmemSize = attributes.sharedSizeBytes + reservedSharedMemPerBlock.
    # compiler["shared_bytes_per_block"] is the kernel's own value only (verified against ptxas -v /
    # Triton's compiled-kernel metadata, M328); the reserve is added here, not baked into that field,
    # so it is never double-counted and stays visible as its own platform constant.
    shared_needed = compiler["shared_bytes_per_block"] + platform["shared_reserved_per_block"]
    shared_block = math.ceil(shared_needed / platform["shared_allocation_unit"]) * platform["shared_allocation_unit"]
    limits = {"blocks": platform["max_blocks_per_sm"], "threads": platform["max_threads_per_sm"] // threads,
              "warps": platform["max_warps_per_sm"] // warps, "registers": platform["registers_per_sm"] // regs_block,
              "shared": (platform["max_blocks_per_sm"] if shared_block == 0 else
                         platform["shared_bytes_per_sm"] // shared_block)}
    occupancy = min(limits.values())
    if occupancy <= 0:
        raise ValueError(f"resource-infeasible launch: occupancy limits={limits}")
    _add(fields, "theoretical_occupancy_blocks_per_sm", occupancy, "resident blocks/SM", "derived", "bounded resource-limit minimum with allocation rounding")
    _add(fields, "waves_per_sm", math.ceil(blocks / platform["sm_count"] / occupancy), "waves/SM", "derived", "grid, SM count and bounded occupancy")


def extract(parent: str, regime: str, candidate: str, *, compiler: dict[str, Any] | None = None,
            platform: dict[str, Any] | None = None, build: Any = None,
            measurement_protocol: Any = None) -> dict[str, Any]:
    if parent not in SUPPORTED:
        return {"schema_version": SCHEMA_VERSION, "status": "unsupported",
                "unsupported_reason": f"no CPU adapter for {parent}"}
    row = _row(parent, regime, candidate)
    source = _source_path(row)
    controls = _controls(row, parent)
    adapter_descriptor: dict[str, Any] | None = None
    launch_known = True
    threads_known = True
    staging_known = True
    if parent == "dev_triton_softmax":
        source_hash, adapter_descriptor = _softmax_source_and_controls(regime, candidate)
    elif parent == "dev_pytorch_rowwise_softmax":
        source_hash, adapter_descriptor = _pytorch_softmax_source_and_controls(regime, candidate)
        launch_known = False
    elif parent == "final_xformers_indexed_select":
        source_hash, adapter_descriptor = _pinned_upstream_source_and_controls(
            parent, regime, candidate)
        threads_known = False
    elif parent == "final_cub_device_reduce":
        source_hash, adapter_descriptor = _pinned_upstream_source_and_controls(
            parent, regime, candidate)
        launch_known = False
    elif parent in PINNED_UPSTREAM_ADAPTERS:
        source_hash, adapter_descriptor = _pinned_upstream_source_and_controls(parent, regime, candidate)
    elif parent == "train_triton_vector_add":
        source_hash, adapter_descriptor = _triton_vector_add_source_and_controls(regime, candidate)
    elif parent == "final_triton_layer_norm":
        source_hash, adapter_descriptor = _triton_layernorm_source_and_controls(regime, candidate)
    elif parent == "final_pytorch_layer_norm":
        source_hash, adapter_descriptor = _aten_source_and_controls(
            parent, "pytorch_layernorm_adapter", regime, candidate)
        launch_known = False
    elif parent == "final_pytorch_embedding":
        source_hash, adapter_descriptor = _aten_source_and_controls(
            parent, "embedding_adapter", regime, candidate)
        launch_known = False
    else:
        assert source is not None
        source_bytes = source.read_bytes()
        # Local fixtures are committed with LF endings (.gitattributes pins them -text).
        # A CRLF copy means a checkout converted them, and its hash would silently
        # diverge from the measured source identity (M328 did exactly this).
        if b"\r\n" in source_bytes:
            raise ValueError(f"{source} has CRLF line endings; its hash would not match the "
                             "committed LF source. Re-checkout it (see .gitattributes).")
        source_text = source_bytes.decode()
        _source_checked(parent, source_text)
        source_hash = sha256_bytes(source_bytes)
    fields: dict[str, dict[str, Any]] = {}
    for key in ("n", "tile_dim", "blocks", "threads", "tile_elements", "stage_elements", "chunks", "reuse", "batches"):
        if key in controls:
            _positive_int(controls[key], key)
    if "launches" in controls:
        _positive_int(controls["launches"], "launches")
    dtype_source = "source: float arrays / sizeof(float)"

    operation_counts: dict[str, int] = {}
    op_source = ""
    summed_override: int | None = None
    transpose_staged = False
    # Sector/coalescing efficiency (M330): useful bytes / bytes actually moved once every global
    # transaction is rounded up to a 32-byte sector. Set per parent below from the source's own
    # addressing arithmetic (warp lane i's address as a function of threadIdx.x), never guessed.
    # None means "not classifiable from source alone into a clean per-warp coalesced/strided
    # pattern" -- left unsupported rather than approximated (see static_feature_contract.md).
    read_efficiency: float | None = None
    write_efficiency: float | None = None
    coalescing_note = ""
    if parent == "dev_local_triad":
        _require_controls(controls, ("n", "tile_dim"), parent)
        n, tile = controls["n"], controls["tile_dim"]
        if n % tile:
            raise ValueError("triad n must be divisible by tile_dim")
        if controls.get("batches", 1) != 1:
            raise ValueError("triad source has no batch loop; batches must be 1")
        grid, block = (n // tile, n // tile, 1), (tile, tile, 1)
        ctas, threads = math.prod(grid), math.prod(block)
        elements = n * n
        global_reads, global_writes = 3 * elements * FLOAT_BYTES, elements * FLOAT_BYTES
        source_flops, global_distinct = 4 * elements, 4 * elements * FLOAT_BYTES
        per_block_global = 4 * tile * tile * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 3, 1
        global_factor = shared_factor = 1
        # a/b/c/out[i] with i = y*n+x; threadIdx.x is the fast-varying warp lane and increments x
        # by 1, so consecutive lanes address consecutive floats: every stream is warp-coalesced.
        read_efficiency = write_efficiency = 1.0
        coalescing_note = "a/b/c[i], out[i] with i=y*n+x; contiguous per warp lane (threadIdx.x)"
    elif parent == "dev_local_transpose":
        _require_controls(controls, ("n", "tile_dim"), parent)
        n, tile = controls["n"], controls["tile_dim"]
        if n % tile:
            raise ValueError("transpose n must be divisible by tile_dim")
        if controls.get("batches", 1) != 1:
            raise ValueError("transpose source has no batch loop; batches must be 1")
        grid, block = (n // tile, n // tile, 1), (tile, tile, 1)
        ctas, threads = math.prod(grid), math.prod(block)
        elements = n * n
        global_reads = global_writes = elements * FLOAT_BYTES
        source_flops, global_distinct = 0, 2 * elements * FLOAT_BYTES
        per_block_global = 2 * tile * tile * FLOAT_BYTES
        shared_reads = shared_writes = elements * FLOAT_BYTES
        barrier_sites, barrier_events = 1, ctas
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        # transpose_tiled: input[y_in*n+x_in] read and output[y_out*n+x_out] write both index with
        # threadIdx.x as the fast-varying term on the innermost dimension (x_in/x_out both add
        # threadIdx.x directly) -- source comment states this explicitly ("both the read and the
        # write stay coalesced"). Staging through shared memory is what makes the write coalesced;
        # it is not assumed, it is the documented purpose of the tile.
        read_efficiency = write_efficiency = 1.0
        coalescing_note = "input[y_in*n+x_in] read, output[y_out*n+x_out] write; both contiguous per warp lane via shared-memory tile staging"
    elif parent == "train_shared_stage":
        _require_controls(controls, ("blocks", "threads", "tile_elements", "stage_elements",
                                     "chunks", "reuse", "batches"), parent)
        blocks, threads = controls["blocks"], controls["threads"]
        tile_elements, stage_elements = controls["tile_elements"], controls["stage_elements"]
        chunks, reuse, batches = controls["chunks"], controls["reuse"], controls["batches"]
        if stage_elements > tile_elements:
            raise ValueError("shared-stage stage_elements must not exceed tile_elements")
        if stage_elements & (stage_elements - 1) or tile_elements & (tile_elements - 1):
            raise ValueError("shared-stage tile/stage elements must be powers of two")
        if threads > 1024:
            raise ValueError("shared-stage threads exceeds CUDA block limit")
        grid, block, ctas = (blocks, 1, 1), (threads, 1, 1), blocks
        touched_input_elements = min(tile_elements, chunks * stage_elements)
        global_reads = blocks * batches * chunks * stage_elements * FLOAT_BYTES
        global_writes = blocks * threads * FLOAT_BYTES
        shared_writes = blocks * batches * chunks * stage_elements * FLOAT_BYTES
        shared_reads = blocks * batches * chunks * threads * reuse * FLOAT_BYTES
        source_flops = 0
        global_distinct = blocks * (touched_input_elements + threads) * FLOAT_BYTES
        per_block_global = (touched_input_elements + threads) * FLOAT_BYTES
        barrier_sites, barrier_events = 2, blocks * batches * chunks * 2
        streams, logical_factor = 1, batches * chunks * reuse
        global_factor, shared_factor = batches * chunks, batches * chunks * reuse
        # STAGE loop: "Cooperative, always coalesced" (source comment) -- logical=threadIdx.x
        # increments the shared/global offset by 1 per lane. sink write: sink[blockIdx.x*blockDim.x
        # + logical], also contiguous per lane. The --layout coalesced|strided control only governs
        # the COMPUTE phase's shared-memory reads (on-chip, not global), so it does not change
        # global sector efficiency either way.
        read_efficiency = write_efficiency = 1.0
        coalescing_note = "STAGE loop and sink write are both contiguous per warp lane (threadIdx.x); --layout only affects on-chip shared reads"
    elif parent == "dev_pytorch_rowwise_softmax":
        # ATen spatial softmax over a padded row-major layout. Logical traffic
        # matches the catalog work formula (one FP32 read/write per logical
        # element; max/subtract/exp/sum/divide per row); the padded stride is a
        # layout control, so it is recorded as explicit stride/allocation fields
        # rather than relabeled as traffic. The masked-lane count is zero: ATen
        # has no power-of-two block constraint, unlike the Triton kernel.
        # Launch geometry, on-chip staging and synchronization are ATen-internal
        # and are not recovered statically (launch_known=False below).
        _require_controls(controls, ("outer_size", "dim_size", "inner_size", "input_row_stride",
                                     "output_row_stride", "layout", "dtype"), parent)
        rows, cols = controls["outer_size"], controls["dim_size"]
        if (rows, cols) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("pytorch-softmax controls do not match the declared logical shape")
        if controls["inner_size"] != 1:
            raise ValueError("pytorch-softmax inner_size must be 1")
        stride = controls["input_row_stride"]
        if stride < cols:
            raise ValueError("pytorch-softmax row stride must cover the logical row")
        if controls["output_row_stride"] != stride:
            raise ValueError("pytorch-softmax mixed input/output strides are not a catalog layout")
        if controls["layout"] != "padded_row_major" or controls["dtype"] != "float32":
            raise ValueError("pytorch-softmax layout/dtype is not the catalog configuration")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["outer_size"], compiled["dim_size"], compiled["row_stride"]) != (rows, cols, stride):
            raise ValueError("pytorch-softmax catalog controls disagree with the pinned adapter")
        op_source = "pinned ATen spatial softmax operation semantics"
        elements = rows * cols
        reductions = rows * (cols - 1)
        grid, block, ctas, threads = [], [], 0, 0
        global_reads = global_writes = elements * FLOAT_BYTES
        source_flops = 2 * elements + reductions
        global_distinct = 2 * elements * FLOAT_BYTES
        per_block_global = 2 * cols * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {
            "row_max_comparisons": reductions,
            "row_sum_additions": reductions,
            "subtract_operations": elements,
            "exp_operations": elements,
            "divide_operations": elements,
            "masked_padding_lanes": 0,
        }
        _add(fields, "input_row_stride_elements", stride, "elements/row", "static", "native_controls_json padded layout")
        _add(fields, "output_row_stride_elements", stride, "elements/row", "static", "native_controls_json padded layout")
        _add(fields, "row_stride_pad_elements", stride - cols, "elements/row", "derived", "stride minus logical columns")
        _add(fields, "padded_row_allocation_bytes", rows * stride * FLOAT_BYTES, "bytes", "derived",
             "declared padded layout; allocation, not traffic")
    elif parent == "train_cuda_samples_reduction":
        # reduce2/3/4/5 single-pass kernels with host completion (--cpufinal).
        # Global traffic mirrors the catalog formula exactly (bounded input reads
        # plus one partial FP32 output per block). Shared staging and sync counts
        # below are read off the pinned reduction_kernel.cu bytes: every variant
        # writes one element per thread to sdata, then reduces with per-variant
        # loop/unroll structure; cg::sync sites and executions are counted, not
        # estimated. The final host sum over partials is a recorded semantic, not
        # device work.
        _require_controls(controls, ("size", "threads", "blocks", "which_kernel", "dtype"), parent)
        n, threads, blocks = controls["size"], controls["threads"], controls["blocks"]
        which_kernel = controls["which_kernel"]
        if n != int(row["shape_a"]):
            raise ValueError("reduction size does not match the declared problem shape")
        if which_kernel not in (2, 3, 4, 5):
            raise ValueError("reduction which_kernel is not an admitted full-sum candidate")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["which_kernel"], compiled["threads"], compiled["blocks"]) != (which_kernel, threads, blocks):
            raise ValueError("reduction catalog controls disagree with the pinned adapter")
        reads_per_thread = 1 if which_kernel == 2 else 2
        read_elements = min(n, blocks * threads * reads_per_thread)
        grid, block, ctas = (blocks, 1, 1), (threads, 1, 1), blocks
        global_reads, global_writes = read_elements * FLOAT_BYTES, blocks * FLOAT_BYTES
        source_flops = max(n - 1, 0)
        global_distinct = (n + blocks) * FLOAT_BYTES
        per_block_global = (threads * reads_per_thread + 1) * FLOAT_BYTES
        # Every block writes exactly one partial; blocks covering input also
        # touch their input share, so the CTA sum is exact, not per_block*ctas.
        summed_override = (read_elements + blocks) * FLOAT_BYTES
        log_threads = threads.bit_length() - 1
        if which_kernel in (2, 3):
            syncs = 1 + log_threads
            shared_reads = (2 * (threads - 1) + (1 if which_kernel == 2 else 0)) * blocks
            shared_writes = (2 * threads - 1) * blocks
        elif which_kernel == 4:
            stages = []
            stage = threads // 2
            while stage > 32:
                stages.append(stage)
                stage //= 2
            active = sum(stages)
            syncs = 1 + len(stages)
            shared_reads = (2 * active + (32 if threads >= 64 else 0)) * blocks
            shared_writes = (threads + active) * blocks
        else:  # which_kernel == 5
            active = sum(s for s in (256, 128, 64) if threads >= 2 * s)
            syncs = 4
            shared_reads = (active + (32 if threads >= 64 else 0)) * blocks
            shared_writes = (threads + active) * blocks
        barrier_sites = 2 if which_kernel in (2, 3, 4) else 4
        barrier_events = syncs * blocks
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {"logical_sum_operations": max(n - 1, 0)}
        op_source = "pinned CUDA-samples reduction source semantics"
        transpose_staged = False
        # Input read: idx = blockIdx.x*blockDim.x + threadIdx.x (and idx+blockDim.x for the
        # 2-reads-per-thread variants), contiguous per warp lane -- coalesced. Output write: one
        # thread (threadIdx.x==0) writes g_odata[blockIdx.x] per CTA; a single 4-byte store with no
        # warp partners still costs one full 32-byte sector, the CUDA memory-controller minimum
        # transaction size, regardless of what any other block/warp is doing concurrently.
        read_efficiency = 1.0
        write_efficiency = FLOAT_BYTES / 32.0
        coalescing_note = "input reads coalesced (contiguous idx); one-thread-per-block partial write costs a full 32B sector for 4B useful"
    elif parent == "train_cuda_samples_transpose":
        # Full-transpose variants 2/3/4/7 over square n x n matrices. Read off
        # the pinned transpose.cu bytes: variant 2 (Naive) copies global to
        # global with no staging and no sync; variants 3/4/7 stage one 32x32
        # tile through __shared__ memory with a single cg::sync per invocation.
        # Variants 4/7 allocate a +1 pad column, which the values-only footprint
        # convention excludes (as for the local transpose).
        _require_controls(controls, ("dim_x", "dim_y", "kernel_variant", "threads_per_block",
                                     "tile_dim", "block_rows", "dtype"), parent)
        n = controls["dim_x"]
        if controls["dim_y"] != n or int(row["shape_a"]) != n or int(row["shape_b"]) != n:
            raise ValueError("transpose matrices must be square and match the declared shape")
        variant = controls["kernel_variant"]
        if variant not in (2, 3, 4, 7):
            raise ValueError("transpose kernel_variant is not an admitted full transpose")
        if controls["tile_dim"] != 32 or controls["block_rows"] != 16:
            raise ValueError("transpose tile/block_rows are not the source-real 32x16 geometry")
        if n % 32 != 0:
            raise ValueError("transpose dimension is not a multiple of TILE_DIM (kernel requirement)")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if compiled["kernel_variant"] != variant:
            raise ValueError("transpose catalog controls disagree with the pinned adapter")
        transpose_staged = variant != 2
        elements = n * n
        tiles = (n // 32) ** 2
        grid, block, ctas, threads = (n // 32, n // 32, 1), (32, 16, 1), tiles, 512
        global_reads = global_writes = elements * FLOAT_BYTES
        source_flops = 0
        global_distinct = 2 * elements * FLOAT_BYTES
        per_block_global = 2 * 32 * 32 * FLOAT_BYTES
        shared_reads = shared_writes = elements * FLOAT_BYTES if transpose_staged else 0
        barrier_sites, barrier_events = (1, ctas) if transpose_staged else (0, 0)
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        op_source = "pinned CUDA-samples transpose source semantics"
        # Read idata[y*width+x] is always contiguous per warp lane (x = ...+threadIdx.x). Variant 2
        # (Naive) writes odata[x*height+y] directly: consecutive lanes (consecutive x) address
        # locations `height` elements = 4*height bytes apart; height=n>=32 (TILE_DIM constraint), so
        # 4*height>=128B, meaning every lane lands in its own distinct 32B sector -- 4 of 32 bytes
        # useful. Variants 3/4/7 stage through shared memory and write with the block indices
        # swapped, so the write address is contiguous per lane too (same argument as the local
        # transpose fixture) -- that staging is the whole point of the optimization these variants
        # exist to demonstrate.
        read_efficiency = 1.0
        write_efficiency = 1.0 if transpose_staged else FLOAT_BYTES / 32.0
        coalescing_note = ("idata read always coalesced; " +
                           ("odata write coalesced via shared-memory tile staging (variant %d)" % variant
                            if transpose_staged else
                            "odata[x*height+y] write strided by height elements per lane (variant 2, Naive): one 32B sector per 4B written"))
    elif parent == "final_cuda_samples_copy":
        # vecAdd: one thread per element with a workIndex guard, no staging, no
        # sync. Two input streams (A, B) plus one output; outputs are excluded
        # from the stream count.
        _require_controls(controls, ("vector_length", "threads", "blocks", "dtype"), parent)
        n, threads, blocks = controls["vector_length"], controls["threads"], controls["blocks"]
        if n != int(row["shape_a"]):
            raise ValueError("copy vector length does not match the declared problem shape")
        if blocks != (n + threads - 1) // threads:
            raise ValueError("copy blocks are not ceil(vector_length/threads)")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["threads"], compiled["blocks"]) != (threads, blocks):
            raise ValueError("copy catalog controls disagree with the pinned adapter")
        grid, block, ctas = (blocks, 1, 1), (threads, 1, 1), blocks
        global_reads, global_writes = 2 * n * FLOAT_BYTES, n * FLOAT_BYTES
        source_flops = n
        global_distinct = 3 * n * FLOAT_BYTES
        per_block_global = 3 * threads * FLOAT_BYTES
        summed_override = 3 * n * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 2, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        op_source = "pinned CUDA-samples vecAdd source semantics"
        transpose_staged = False
        # A[i], B[i] read and C[i] write with i = blockIdx.x*blockDim.x + threadIdx.x: contiguous
        # per warp lane on every stream, guard aside (the guard only disables the tail, never
        # reorders addresses).
        read_efficiency = write_efficiency = 1.0
        coalescing_note = "A[i]/B[i]/C[i] with i=blockIdx.x*blockDim.x+threadIdx.x; contiguous per warp lane"
    elif parent == "train_triton_vector_add":
        # add_kernel: two tl.load (x, y), one add, one tl.store per element, one
        # program per BLOCK_SIZE chunk with a masked tail. No staging, no sync.
        # Two input streams (x, y); the output is excluded from the count.
        _require_controls(controls, ("block_size", "n"), parent)
        n, block_size = controls["n"], controls["block_size"]
        if n != int(row["shape_a"]):
            raise ValueError("vector-add n does not match the declared problem shape")
        if block_size not in (128, 256, 512, 1024):
            raise ValueError("vector-add BLOCK_SIZE is not a catalog candidate")
        # The catalog's threads_per_block for this parent is BLOCK_SIZE, i.e. elements per
        # program, not CUDA threads. The compiled CTA is num_warps*32 threads; the runner
        # never passes num_warps, so Triton's default applies to every candidate (M329).
        if int(row["threads_per_block"]) != block_size or int(row["items_per_thread"]) != 1:
            raise ValueError("vector-add catalog elements/items must be (BLOCK_SIZE, 1)")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if compiled.get("BLOCK_SIZE") != block_size:
            raise ValueError("vector-add catalog controls disagree with the pinned adapter")
        if "num_warps" in compiled or "num_warps" in (compiler or {}):
            warps = (compiler or {}).get("num_warps", compiled.get("num_warps"))
            if warps != TRITON_DEFAULT_NUM_WARPS:
                raise ValueError(f"vector-add compiled with num_warps={warps}, not Triton's default")
        programs = (n + block_size - 1) // block_size
        threads = TRITON_DEFAULT_NUM_WARPS * 32
        grid, block, ctas = (programs, 1, 1), (threads, 1, 1), programs
        global_reads, global_writes = 2 * n * FLOAT_BYTES, n * FLOAT_BYTES
        source_flops = n
        global_distinct = 3 * n * FLOAT_BYTES
        per_block_global = 3 * block_size * FLOAT_BYTES
        summed_override = 3 * n * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 2, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        op_source = "pinned Triton vector-add source semantics"
        transpose_staged = False
        # tl.load/tl.store at offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE): Triton compiles
        # this to a per-lane offset equal to the program's base plus the lane's position in the
        # arange, i.e. contiguous per lane -- the standard vectorized-elementwise pattern; masked
        # tail lanes do not issue a memory transaction at all (existing masked_tail_lanes field).
        read_efficiency = write_efficiency = 1.0
        coalescing_note = "tl.load/tl.store at pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE); contiguous per lane, masked tail lanes issue no transaction"
        _add(fields, "add_operations", n, "operations/invocation", "static", op_source)
        _add(fields, "masked_tail_lanes", (-n) % block_size, "lanes/invocation", "derived",
             "single partial program tail; masked lanes do not access global memory")
    elif parent == "final_triton_layer_norm":
        # _layer_norm_fwd_fused: one program per logical row. X is read in mean,
        # variance, and normalized-output passes; W/B read once (broadcast per
        # element in the catalog convention); Y plus per-row mean/rstd written.
        # Row-local reduction stays on chip; no staging, no sync. Three input
        # streams (X, W, B); outputs are excluded from the count.
        _require_controls(controls, ("rows", "cols", "block_size", "num_warps", "eps", "dtype"), parent)
        rows, cols = controls["rows"], controls["cols"]
        if (rows, cols) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("layernorm controls do not match the declared logical shape")
        expected_block = 1
        while expected_block < cols:
            expected_block *= 2
        if controls["block_size"] != expected_block:
            raise ValueError("layernorm BLOCK_SIZE is not the source-real next power-of-two mapping")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["BLOCK_SIZE"], compiled["num_warps"]) != (controls["block_size"], controls["num_warps"]):
            raise ValueError("layernorm catalog controls disagree with the pinned adapter")
        op_source = "pinned Triton layernorm source semantics"
        elements = rows * cols
        grid, block, ctas, threads = (rows, 1, 1), (controls["num_warps"] * 32, 1, 1), rows, controls["num_warps"] * 32
        global_reads = (3 * elements + 2 * elements) * FLOAT_BYTES
        global_writes = (elements + 2 * rows) * FLOAT_BYTES
        source_flops = 9 * elements + 2 * rows
        global_distinct = (2 * elements + 2 * cols + 2 * rows) * FLOAT_BYTES
        per_block_global = (6 * cols + 2) * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 3, 1
        global_factor = shared_factor = 1
        operation_counts = {"masked_padding_lanes": rows * (expected_block - cols)}
        _add(fields, "x_read_passes", 3, "passes/invocation", "static", op_source)
        _add(fields, "affine_read_elements", 2 * elements, "elements/invocation", "static", op_source)
        _add(fields, "mean_rstd_write_elements", 2 * rows, "elements/invocation", "static", op_source)
        # One program per row; X/W/B/Y offsets are row_start + tl.arange(0, BLOCK_SIZE) (masked
        # beyond cols), the same contiguous-per-lane pattern as the vector-add/softmax kernels.
        # Simplification (documented, not hidden): this treats every row read/write as starting at
        # a sector-aligned offset and ignores possible partial-sector waste at a row boundary that
        # does not fall on a 32-byte multiple, which depends on the runtime tensor base pointer and
        # is not visible to a CPU-only source read.
        read_efficiency = write_efficiency = 1.0
        coalescing_note = ("row_start+tl.arange(0,BLOCK_SIZE) offsets on X/W/B/Y; contiguous per lane. "
                           "Row-boundary sector alignment is runtime base-pointer-dependent and not modeled (simplification, see static_feature_contract.md).")
    elif parent == "final_pytorch_layer_norm":
        # ATen LayerNorm forward over a padded row-major layout with identity
        # affine. Logical traffic matches the catalog formula (X in three
        # passes, W/B reads, Y plus mean/rstd writes); the padded stride is a
        # layout control recorded as explicit stride/allocation fields.
        # Launch geometry, staging and sync are ATen-internal (launch_known=False).
        _require_controls(controls, ("rows", "cols", "row_stride", "eps", "dtype", "affine"), parent)
        rows, cols = controls["rows"], controls["cols"]
        if (rows, cols) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("pytorch-layernorm controls do not match the declared logical shape")
        stride = controls["row_stride"]
        if stride < cols:
            raise ValueError("pytorch-layernorm row stride must cover the logical row")
        if controls["eps"] != 1e-5 or controls["dtype"] != "float32" or controls["affine"] is not True:
            raise ValueError("pytorch-layernorm eps/dtype/affine is not the catalog configuration")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["rows"], compiled["cols"], compiled["row_stride"]) != (rows, cols, stride):
            raise ValueError("pytorch-layernorm catalog controls disagree with the pinned adapter")
        op_source = "pinned ATen layernorm operation semantics"
        elements = rows * cols
        grid, block, ctas, threads = [], [], 0, 0
        global_reads = (3 * elements + 2 * rows) * FLOAT_BYTES
        global_writes = (elements + 2 * rows) * FLOAT_BYTES
        source_flops = 5 * elements
        global_distinct = (2 * elements + 2 * cols + 2 * rows) * FLOAT_BYTES
        per_block_global = (6 * cols + 2) * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 3, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        _add(fields, "input_row_stride_elements", stride, "elements/row", "static", "native_controls_json padded layout")
        _add(fields, "row_stride_pad_elements", stride - cols, "elements/row", "derived", "stride minus logical columns")
        _add(fields, "padded_row_allocation_bytes", rows * stride * FLOAT_BYTES, "bytes", "derived",
             "declared padded layout; allocation, not traffic")
        _add(fields, "mean_rstd_write_elements", 2 * rows, "elements/invocation", "static", op_source)
    elif parent == "final_pytorch_embedding":
        # ATen index_select gather of single rows along dim 0. One int64 index
        # read per selected row, one FP32 gather read and one FP32 output write
        # per feature; no hidden duplicate-load factor. Launch geometry,
        # staging and sync are ATen-internal (launch_known=False). Two input
        # streams: the source matrix and the index tensor.
        _require_controls(controls, ("num_weights", "feature_dim", "num_indices", "dim",
                                     "index_dtype", "source_layout"), parent)
        vocab, dim, selected = controls["num_weights"], controls["feature_dim"], controls["num_indices"]
        if (vocab, dim) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("embedding controls do not match the declared problem shape")
        if controls["dim"] != 0 or controls["index_dtype"] != "int64":
            raise ValueError("embedding gather must be dim-0 with int64 indices")
        if controls["source_layout"] != "contiguous":
            raise ValueError("embedding source layout is not the catalog configuration")
        if selected <= 0:
            raise ValueError("embedding num_indices must be positive")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["num_weights"], compiled["feature_dim"], compiled["num_indices"]) != (vocab, dim, selected):
            raise ValueError("embedding catalog controls disagree with the pinned adapter")
        op_source = "pinned ATen index_select operation semantics"
        elements = selected * dim
        grid, block, ctas, threads = [], [], 0, 0
        global_reads = elements * FLOAT_BYTES + selected * 8
        global_writes = elements * FLOAT_BYTES
        source_flops = 0
        global_distinct = (vocab * dim + elements) * FLOAT_BYTES + selected * 8
        per_block_global = 0
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 2, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        _add(fields, "index_elements", selected, "elements/invocation", "static", op_source)
        _add(fields, "index_bytes", selected * 8, "bytes/invocation", "static", op_source)
        _add(fields, "index_dtype", "int64", "dtype", "static", op_source)
        _add(fields, "gather_elements", elements, "elements/invocation", "static", op_source)
    elif parent == "final_xformers_indexed_select":
        # index_select_cat_fwd_kernel over a 2D grid: one program per
        # (index, col-block) with BLOCK_SIZE_INDEX=1. The grid shape is
        # source-known, but threads-per-program is Triton-chosen, so block
        # geometry and occupancy stay unsupported (threads_known=False). int32
        # index reads; masked col lanes in the tail block do no traffic.
        _require_controls(controls, ("num_rows", "num_indices", "num_cols", "block_size_index",
                                     "block_size_col", "dtype"), parent)
        num_rows, num_indices, num_cols = controls["num_rows"], controls["num_indices"], controls["num_cols"]
        if (num_rows, num_cols) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("xformers controls do not match the declared problem shape")
        block_col = controls["block_size_col"]
        if controls["block_size_index"] != 1 or block_col not in (128, 256, 512, 1024):
            raise ValueError("xformers BLOCK_SIZE_INDEX/BLOCK_SIZE_COL is not a catalog candidate")
        if not 0 < num_indices < num_rows:
            raise ValueError("xformers source wrapper requires 0 < num_indices < num_rows")
        if controls["dtype"] != "float32":
            raise ValueError("xformers dtype is not the catalog configuration")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["BLOCK_SIZE_COL"], compiled["BLOCK_SIZE_INDEX"]) != (block_col, 1):
            raise ValueError("xformers catalog controls disagree with the pinned adapter")
        op_source = "pinned xFormers index_select_cat operation semantics"
        elements = num_indices * num_cols
        col_programs = (num_cols + block_col - 1) // block_col
        grid, block, ctas, threads = (num_indices, col_programs, 1), [], num_indices * col_programs, 0
        global_reads = elements * FLOAT_BYTES + num_indices * 4
        global_writes = elements * FLOAT_BYTES
        source_flops = 0
        global_distinct = (num_rows * num_cols + 2 * elements) * FLOAT_BYTES + num_indices * 4
        per_block_global = 0
        summed_override = 2 * elements * FLOAT_BYTES + num_indices * 4
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 2, 1
        global_factor = shared_factor = 1
        operation_counts = {}
        _add(fields, "index_elements", num_indices, "elements/invocation", "static", op_source)
        _add(fields, "index_bytes", num_indices * 4, "bytes/invocation", "static", op_source)
        _add(fields, "index_dtype", "int32", "dtype", "static", op_source)
        _add(fields, "gather_elements", elements, "elements/invocation", "static", op_source)
        _add(fields, "masked_col_lanes", num_indices * (col_programs * block_col - num_cols),
             "lanes/invocation", "derived", "tail col-block; masked lanes do not access global memory")
    elif parent == "train_cub_block_reduce":
        # Single-block cub::BlockReduce over 1024 valid items with compile-time
        # (BlockDimX, ItemsPerThread) geometry. Global traffic and n-1 logical
        # additions match the catalog formula. BlockReduce staging and sync live
        # inside CUB's TempStorage implementation, whose size is not recovered
        # statically, so shared traffic, shared footprint, and barriers read
        # unsupported rather than invented.
        _require_controls(controls, ("block_dim_x", "block_dim_y", "block_dim_z", "items_per_thread",
                                     "valid_items", "dtype"), parent)
        block_x, items = controls["block_dim_x"], controls["items_per_thread"]
        if (controls["block_dim_y"], controls["block_dim_z"]) != (1, 1):
            raise ValueError("cub block geometry must be 1D")
        if block_x * items != 1024 or controls["valid_items"] != 1024:
            raise ValueError("cub block tile must cover exactly the 1024 valid items")
        if (block_x, items) not in ((32, 32), (64, 16), (128, 8), (256, 4)):
            raise ValueError("cub block geometry is not a catalog candidate")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["block_dim_x"], compiled["items_per_thread"]) != (block_x, items):
            raise ValueError("cub block catalog controls disagree with the pinned adapter")
        op_source = "pinned CUB BlockReduce operation semantics"
        staging_known = False
        grid, block, ctas, threads = (1, 1, 1), (block_x, 1, 1), 1, block_x
        global_reads, global_writes = 1024 * FLOAT_BYTES, FLOAT_BYTES
        source_flops = 1023
        global_distinct = (1024 + 1) * FLOAT_BYTES
        per_block_global = (1024 + 1) * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {"logical_sum_operations": 1023}
    elif parent == "final_cub_device_reduce":
        # DeviceReduce::Sum over num_items with the source API's runtime
        # temp-storage query. Grid/block choice is CUB-internal
        # (launch_known=False); traffic is one input read per item plus one
        # scalar output. Staging/sync inside the implementation are unsupported.
        _require_controls(controls, ("num_items", "api", "dtype", "temp_storage_query"), parent)
        n = controls["num_items"]
        if controls["api"] != "DeviceReduce::Sum" or controls["dtype"] != "float32":
            raise ValueError("cub device configuration is not the catalog Sum/float32 setup")
        if controls["temp_storage_query"] is not True:
            raise ValueError("cub device temp-storage query must be enabled")
        if n <= 0:
            raise ValueError("cub device num_items must be positive")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if compiled["num_items"] != n:
            raise ValueError("cub device catalog controls disagree with the pinned adapter")
        op_source = "pinned CUB DeviceReduce operation semantics"
        grid, block, ctas, threads = [], [], 0, 0
        global_reads, global_writes = n * FLOAT_BYTES, FLOAT_BYTES
        source_flops = max(n - 1, 0)
        global_distinct = (n + 1) * FLOAT_BYTES
        per_block_global = 0
        shared_reads = shared_writes = 0
        barrier_sites, barrier_events = 0, 0
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {"logical_sum_operations": max(n - 1, 0)}
    else:  # dev_triton_softmax
        # The accepted runner executes one Triton program per logical row.  The
        # row reduction stays in that program's on-chip storage; it is not a
        # shared-memory or DRAM spill claim.  Arithmetic deliberately follows
        # C's catalog convention: subtract + reduce-add + divide are FLOPs;
        # max comparisons and exp remain separately named operation classes.
        op_source = "pinned Triton softmax source semantics"
        _require_controls(controls, ("rows", "cols", "block_size", "num_warps", "num_stages", "seed"), parent)
        rows, cols = controls["rows"], controls["cols"]
        if (rows, cols) != (int(row["shape_a"]), int(row["shape_b"])):
            raise ValueError("softmax controls do not match the declared logical shape")
        expected_block = 1 << (cols - 1).bit_length()
        if controls["block_size"] != expected_block:
            raise ValueError("softmax BLOCK_SIZE is not the source-real next power-of-two mapping")
        assert adapter_descriptor is not None
        compiled = adapter_descriptor["build"]["compile_meta"]
        if (compiled["BLOCK_SIZE"], compiled["num_warps"], compiled["num_stages"]) != (
                controls["block_size"], controls["num_warps"], controls["num_stages"]):
            raise ValueError("softmax catalog controls disagree with the pinned adapter")
        elements = rows * cols
        reductions = rows * (cols - 1)
        grid, block, ctas, threads = (rows, 1, 1), (controls["num_warps"] * 32, 1, 1), rows, controls["num_warps"] * 32
        global_reads = global_writes = elements * FLOAT_BYTES
        source_flops = 2 * elements + reductions
        global_distinct = 2 * elements * FLOAT_BYTES
        per_block_global = 2 * cols * FLOAT_BYTES
        shared_reads = shared_writes = 0
        barrier_sites = barrier_events = 0
        streams, logical_factor = 1, 1
        global_factor = shared_factor = 1
        operation_counts = {
            "row_max_comparisons": reductions,
            "row_sum_additions": reductions,
            "subtract_operations": elements,
            "exp_operations": elements,
            "divide_operations": elements,
            "masked_padding_lanes": rows * (expected_block - cols),
        }
        # Same contiguous-per-lane pattern and the same row-boundary-alignment simplification as
        # the Triton layernorm kernel above (both are one-program-per-row Triton tutorials with
        # row_start + tl.arange(0, BLOCK_SIZE) offsets).
        read_efficiency = write_efficiency = 1.0
        coalescing_note = ("row_start+tl.arange(0,BLOCK_SIZE) offsets; contiguous per lane. "
                           "Row-boundary sector alignment is runtime base-pointer-dependent and not modeled (simplification, see static_feature_contract.md).")

    # The catalog is a declared descriptor, not the authority for work.  Keep
    # this check so a stale descriptor fails loudly instead of silently joining
    # source-derived features to contradictory C metadata.
    catalog_expected = {
        "global_read_bytes": int(row["global_read_bytes"]),
        "global_write_bytes": int(row["global_write_bytes"]),
        "source_flops": int(row["source_level_flops"]),
    }
    derived_expected = {"global_read_bytes": global_reads, "global_write_bytes": global_writes,
                        "source_flops": source_flops}
    if catalog_expected != derived_expected:
        raise ValueError(f"catalog/source work mismatch for {parent}/{regime}/{candidate}: "
                         f"catalog={catalog_expected} derived={derived_expected}")

    _add(fields, "source_flops", source_flops, "FLOP/invocation", "static", "source-backed arithmetic formula")
    if "row_max_comparisons" in operation_counts:
        _add(fields, "row_max_comparisons", operation_counts["row_max_comparisons"], "comparisons/invocation", "static", op_source)
        _add(fields, "row_sum_additions", operation_counts["row_sum_additions"], "additions/invocation", "static", op_source)
        _add(fields, "subtract_operations", operation_counts["subtract_operations"], "operations/invocation", "static", op_source)
        _add(fields, "exp_operations", operation_counts["exp_operations"], "SFU exp operations/invocation", "static", op_source)
        _add(fields, "divide_operations", operation_counts["divide_operations"], "operations/invocation", "static", op_source)
    if "masked_padding_lanes" in operation_counts:
        mask_source = ("BLOCK_SIZE mask; masked lanes do not access global memory"
                       if parent in ("dev_triton_softmax", "final_triton_layer_norm")
                       else op_source + "; ATen has no block-padding constraint; padded layout is not masked lanes")
        _add(fields, "masked_padding_lanes", operation_counts["masked_padding_lanes"], "lanes/invocation", "derived",
             mask_source)
    if "logical_sum_operations" in operation_counts:
        _add(fields, "logical_sum_operations", operation_counts["logical_sum_operations"],
             "additions/invocation", "static", op_source)
    _add(fields, "global_read_bytes", global_reads, "bytes/invocation", "derived", dtype_source)
    _add(fields, "global_write_bytes", global_writes, "bytes/invocation", "derived", dtype_source)
    _add(fields, "shared_read_bytes", shared_reads, "bytes/invocation", "derived", dtype_source)
    _add(fields, "shared_write_bytes", shared_writes, "bytes/invocation", "derived", dtype_source)
    _add(fields, "barrier_sites", barrier_sites, "lexical __syncthreads sites/kernel", "static", "source __syncthreads sites")
    _add(fields, "barriers", barrier_events, "CTA synchronization events/invocation", "derived", "CTA count × source loop executions")
    _add(fields, "barrier_events", barrier_events, "CTA synchronization events/invocation", "derived", "CTA count × source loop executions")
    _add(fields, "global_access_repeat_factor", global_factor, "logical global-access factor", "derived", "source loop bounds")
    _add(fields, "shared_access_repeat_factor", shared_factor, "logical shared-access factor", "derived", "source loop bounds")
    _add(fields, "repeated_logical_traffic_factor", logical_factor, "logical repeat factor", "derived", "source loop bounds")
    _add(fields, "global_distinct_footprint_bytes", global_distinct, "bytes", "derived", "union of source-touched global array addresses")
    _add(fields, "distinct_footprint_bytes", global_distinct, "bytes", "derived", "global address union; shared address space reported separately")
    _add(fields, "summed_block_footprint_bytes",
         summed_override if summed_override is not None else per_block_global * ctas,
         "bytes", "derived", "sum of per-CTA distinct global footprints")
    if parent == "train_shared_stage":
        shared_distinct = controls["blocks"] * controls["stage_elements"] * FLOAT_BYTES
    elif parent == "dev_local_transpose":
        shared_distinct = ctas * controls["tile_dim"] * controls["tile_dim"] * FLOAT_BYTES
    elif parent == "train_cuda_samples_transpose":
        shared_distinct = ctas * 32 * 32 * FLOAT_BYTES if transpose_staged else 0
    elif parent == "train_cuda_samples_reduction":
        shared_distinct = threads * FLOAT_BYTES
    else:
        shared_distinct = 0
    _add(fields, "shared_distinct_footprint_bytes", shared_distinct, "bytes", "derived", "union in each CTA's shared address space")
    _add(fields, "predicted_service_level_traffic_bytes", None, "bytes/invocation", "derived", "not inferred by CPU extractor")
    fields["predicted_service_level_traffic_bytes"]["status"] = "unsupported"
    fields["predicted_service_level_traffic_bytes"]["note"] = "Logical bytes and distinct footprints are not physical/service-level traffic."
    _add(fields, "input_shape", [int(row["shape_a"]), int(row["shape_b"])], "elements", "static", "workload_cells.csv shape")
    _add(fields, "grid", list(grid), "blocks per dimension", "static", "native_controls_json/source launch")
    _add(fields, "block", list(block), "threads per dimension", "static", "native_controls_json/source launch")
    _add(fields, "cta_count", ctas, "CTAs/invocation", "derived", "grid product")
    _add(fields, "threads_per_block", threads, "threads/CTA", "derived", "block product")
    if not launch_known:
        # ATen-internal launch geometry, staging and synchronization are not
        # recovered statically. Placeholders above are replaced, never kept.
        launch_note = "ATen-internal launch/staging/sync; not recovered statically, not relabeled as zero."
        for name, units in (("shared_read_bytes", "bytes/invocation"),
                            ("shared_write_bytes", "bytes/invocation"),
                            ("barrier_sites", "lexical __syncthreads sites/kernel"),
                            ("barriers", "CTA synchronization events/invocation"),
                            ("barrier_events", "CTA synchronization events/invocation"),
                            ("summed_block_footprint_bytes", "bytes"),
                            ("shared_distinct_footprint_bytes", "bytes"),
                            ("grid", "blocks per dimension"),
                            ("block", "threads per dimension"),
                            ("cta_count", "CTAs/invocation"),
                            ("threads_per_block", "threads/CTA")):
            fields[name] = unsupported(units, "ATen launch geometry", launch_note)
    elif not threads_known:
        # The launch grid is source-known but threads-per-program is chosen by
        # the runtime, so block geometry and occupancy stay unsupported while
        # grid/CTA counts remain supported.
        thread_note = "Threads-per-program is runtime-chosen; block geometry and occupancy not recovered statically."
        for name, units in (("block", "threads per dimension"),
                            ("threads_per_block", "threads/CTA")):
            fields[name] = unsupported(units, "runtime launch geometry", thread_note)
    if not staging_known:
        # Staging bytes and sync events live inside the library implementation
        # and are not recovered statically; placeholders above are replaced.
        staging_note = "Library-internal staging/sync; not recovered statically, not relabeled as zero."
        for name, units in (("shared_read_bytes", "bytes/invocation"),
                            ("shared_write_bytes", "bytes/invocation"),
                            ("shared_distinct_footprint_bytes", "bytes"),
                            ("barrier_sites", "lexical __syncthreads sites/kernel"),
                            ("barriers", "CTA synchronization events/invocation"),
                            ("barrier_events", "CTA synchronization events/invocation")):
            fields[name] = unsupported(units, "library-internal staging", staging_note)
    _add(fields, "stream_count", streams, "logical input streams", "static", "source operation semantics (outputs excluded)")
    fields["stream_semantics"] = state("input arrays only; output/sink excluded", units="definition", source="source array declarations")
    _add(fields, "launch_repetition_count", controls.get("launches", 1), "measurement launches (not work/invocation)", "static", "native_controls_json")
    fields["dependency_depth"] = unsupported("dependency levels", "source/compiler metadata", "Not inferred from timing or target measurements.")
    for name in ("global_footprint_to_capacity_ratio", "shared_footprint_to_capacity_ratio"):
        fields[name] = unsupported("ratio", "platform capacity metadata", "Capacity is not available in the CPU descriptor.")
    # M330: L2-footprint ratio, for the tier feature set. global_distinct_footprint_bytes is
    # always supported (computed above for every SUPPORTED parent); the only missing input can be
    # the platform's l2_bytes, so that is the only unsupported reason here.
    if platform is not None and "l2_bytes" in platform:
        l2_bytes = platform["l2_bytes"]
        if not isinstance(l2_bytes, int) or l2_bytes <= 0:
            raise ValueError("platform.l2_bytes must be a positive integer")
        _add(fields, "l2_footprint_ratio", global_distinct / l2_bytes, "ratio", "derived",
             "global_distinct_footprint_bytes / platform.l2_bytes (HARDWARE_GROUND_TRUTH.md)")
    else:
        fields["l2_footprint_ratio"] = unsupported(
            "ratio", "platform capacity metadata",
            "platform.l2_bytes is absent; no L2 capacity claim is made without it.")
    # M330: static sector/coalescing efficiency estimate, per-parent derivation in
    # static_feature_contract.md. read/write_efficiency are None (unsupported) unless a branch
    # above set them from the source's own per-lane address arithmetic.
    if read_efficiency is not None and write_efficiency is not None:
        _add(fields, "coalescing_read_efficiency", read_efficiency, "useful/moved bytes ratio",
             "derived", coalescing_note)
        _add(fields, "coalescing_write_efficiency", write_efficiency, "useful/moved bytes ratio",
             "derived", coalescing_note)
        moved_read = global_reads / read_efficiency if read_efficiency > 0 else 0.0
        moved_write = global_writes / write_efficiency if write_efficiency > 0 else 0.0
        effective_bytes = moved_read + moved_write
        overall_efficiency = ((global_reads + global_writes) / effective_bytes
                              if effective_bytes > 0 else 1.0)
        _add(fields, "coalescing_efficiency", overall_efficiency, "useful/moved bytes ratio",
             "derived", "bytes-weighted combination of coalescing_read/write_efficiency")
        _add(fields, "effective_global_bytes", effective_bytes, "bytes/invocation", "derived",
             "logical global bytes / per-direction coalescing efficiency, i.e. bytes actually moved in 32B sectors")
    else:
        note = ("Access pattern is not classifiable from source alone into a clean per-warp "
                "coalesced/strided pattern (e.g. library/compiler-internal per-thread mapping, "
                "runtime-scattered indices, or unresolved on-chip staging); see static_feature_contract.md.")
        for name, units in (("coalescing_read_efficiency", "useful/moved bytes ratio"),
                            ("coalescing_write_efficiency", "useful/moved bytes ratio"),
                            ("coalescing_efficiency", "useful/moved bytes ratio"),
                            ("effective_global_bytes", "bytes/invocation")):
            fields[name] = unsupported(units, "source access-pattern analysis", note)
    _resource_fields(fields, compiler, platform, threads=threads, blocks=ctas,
                     launch_known=launch_known, threads_known=threads_known)

    # Identity excludes measurement-only launch repetition.  The same operation remains one cell across platforms.
    identity_material = {"parent_id": parent, "regime": regime, "candidate_id": candidate,
                         "catalog_row": {k: v for k, v in row.items() if k != "native_controls_json"},
                         "source_sha256": source_hash,
                         "controls": {k: v for k, v in controls.items() if k != "launches"}}
    cell_id = sha256_bytes(canonical(identity_material).encode())
    context = _context(source_hash=source_hash, compiler=compiler, platform=platform,
                       build=build, measurement_protocol=measurement_protocol)
    context_id = sha256_bytes(canonical(context).encode()) if context["status"] == "complete" else None
    return {
        "schema_version": SCHEMA_VERSION, "status": "supported", "cell_id": cell_id,
        "operation_id": cell_id, "context_id": context_id, "context": context,
        "identity": {"parent_id": parent, "regime": regime, "candidate_id": candidate,
                     "source_sha256": source_hash,
                     "source_path": ({"dev_triton_softmax": "python/tutorials/02-fused-softmax.py",
                                      "dev_pytorch_rowwise_softmax": PYTORCH_SOFTMAX_DISPLAY_PATH,
                                      "final_pytorch_layer_norm": PYTORCH_LAYERNORM_DISPLAY_PATH,
                                      "final_pytorch_embedding": PYTORCH_EMBEDDING_DISPLAY_PATH,
                                      "train_triton_vector_add": TRITON_VECTOR_ADD_DISPLAY_PATH,
                                      "final_triton_layer_norm": TRITON_LAYERNORM_DISPLAY_PATH,
                                      **{p: PINNED_UPSTREAM_DISPLAY_PATHS[p] for p in PINNED_UPSTREAM_DISPLAY_PATHS}}[parent]
                                     if source is None else str(source.relative_to(REPO))),
                     "compiler_metadata_sha256": _hash_optional(compiler),
                     "platform_profile_sha256": _hash_optional(platform),
                     "build_sha256": _hash_optional(build),
                     "measurement_protocol_sha256": _hash_optional(measurement_protocol)},
        "input": {"shape_a": int(row["shape_a"]), "shape_b": int(row["shape_b"]),
                  "controls": controls, "origin": "catalog/source descriptor", "units": "declared per field"},
        "work": {k: fields[k] for k in ("source_flops", "global_read_bytes", "global_write_bytes",
                  "shared_read_bytes", "shared_write_bytes", "barriers", "barrier_events",
                  "repeated_logical_traffic_factor", "global_distinct_footprint_bytes",
                  "distinct_footprint_bytes", "summed_block_footprint_bytes", "shared_distinct_footprint_bytes")},
        "features": fields,
        "measurement_inputs": {"S0_HF": True, "HK_anchor": False, "runtime_power_counters": False,
                               "note": "No target measurements, counters, runtime, power, or measurement launch count are admitted."},
        "provenance": {"source_hash_algorithm": "sha256", "source_hash": source_hash,
                       "compiler": compiler, "platform": platform, "build": build,
                       "measurement_protocol": measurement_protocol},
    }


def _join_context(item: dict[str, Any], kind: str) -> tuple[str, str]:
    cell_id = item.get("cell_id")
    if not cell_id or not isinstance(cell_id, str):
        raise ValueError(f"{kind} record is missing cell_id; B must emit D's cell identity before joining")
    context_id = item.get("context_id")
    context = item.get("context")
    if not context_id or not isinstance(context_id, str) or not isinstance(context, dict):
        raise ValueError(f"{kind} record {cell_id} is missing complete context_id/context")
    required = ("platform_profile_sha256", "source_sha256", "measurement_protocol_sha256")
    if any(not context.get(k) for k in required) or not (context.get("build_sha256") or context.get("compiler_metadata_sha256")):
        raise ValueError(f"{kind} record {cell_id} has incomplete platform/build/source/protocol context")
    expected = sha256_bytes(canonical(context).encode())
    if expected != context_id:
        raise ValueError(f"{kind} record {cell_id} has context_id inconsistent with context")
    return cell_id, context_id


def join_predictions_labels(predictions: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join only equal cells under exactly equal, complete measurement contexts."""
    if not isinstance(predictions, list) or not isinstance(labels, list):
        raise ValueError("predictions and labels must be lists")
    p: dict[str, dict[str, Any]] = {}; l: dict[str, dict[str, Any]] = {}
    pctx: dict[str, str] = {}; lctx: dict[str, str] = {}
    for item in predictions:
        cid, ctx = _join_context(item, "prediction")
        if cid in p:
            raise ValueError(f"duplicate prediction cell_id: {cid}")
        p[cid], pctx[cid] = item, ctx
    for item in labels:
        cid, ctx = _join_context(item, "label")
        if cid in l:
            raise ValueError(f"duplicate label cell_id: {cid}")
        l[cid], lctx[cid] = item, ctx
    if set(p) != set(l):
        missing, extra = sorted(set(p) - set(l)), sorted(set(l) - set(p))
        raise ValueError(f"prediction/label cell IDs differ; missing labels={missing}, extra labels={extra}")
    for cid in sorted(p):
        if pctx[cid] != lctx[cid]:
            raise ValueError(f"platform/build/source/protocol context mismatch for cell_id {cid}")
        if p[cid].get("operation_id", cid) != l[cid].get("operation_id", cid):
            raise ValueError(f"operation identity mismatch for cell_id {cid}")
    return [{"cell_id": cid, "context_id": pctx[cid], "prediction": p[cid], "label": l[cid]}
            for cid in sorted(p)]


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU-only source-backed static feature extraction")
    ap.add_argument("--parent", required=True); ap.add_argument("--regime", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--compiler-json", type=Path); ap.add_argument("--platform-json", type=Path)
    ap.add_argument("--build-json", type=Path); ap.add_argument("--measurement-protocol-json", type=Path); ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    compiler = json.loads(args.compiler_json.read_text()) if args.compiler_json else None
    platform = json.loads(args.platform_json.read_text()) if args.platform_json else None
    build = json.loads(args.build_json.read_text()) if args.build_json else None
    protocol = json.loads(args.measurement_protocol_json.read_text()) if args.measurement_protocol_json else None
    result = extract(args.parent, args.regime, args.candidate, compiler=compiler, platform=platform,
                     build=build, measurement_protocol=protocol)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output: args.output.write_text(encoded)
    else: print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
