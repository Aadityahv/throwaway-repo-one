#!/usr/bin/env python3
"""Run the pinned ATen LayerNorm forward path, when its real runtime is present.

The repository deliberately contains no PyTorch/CUDA dependency. This runner
therefore has a strict two-state contract: source and runtime checks either
pass and the actual pinned ATen LayerNorm path is launched, or the command
fails loudly without downloading, installing, or substituting an
implementation. The CPU self-test exercises source hashing, candidate
controls, and the full reference/rejection path without importing Torch or
touching a GPU.

ATen note: like SoftMax.cu, ``aten/src/ATen/native/cuda/layer_norm_kernel.cu``
is C++ compiled into libtorch, not an extractable kernel. The GPU path
therefore verifies the pinned source checkout (exact revision plus file
SHA-256) and then launches the operation through the installed Torch
``aten::layer_norm`` CUDA dispatch on the catalog's exact padded-row-stride
layout with identity affine. The emitted record carries the installed
Torch/CUDA build identity alongside the pinned source identity, so a reviewer
can see exactly what was executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = "aten/src/ATen/native/cuda/layer_norm_kernel.cu"
PINNED_REVISION = "67faf385dd4133d3e658f9b1c2a4c87dc9b61218"
PINNED_SHA256 = "0a214206fd98885a3121e32fa30cf835dce769ed52546412e9969e7bc386bd0f"
# Reference default seed for LayerNorm values (correctness_reference uses
# 20260915 for norm inputs); passed explicitly, so any fixed seed is equivalent.
VALUE_SEED = 20260915


class RunnerBlocked(RuntimeError):
    """A required provenance/runtime check failed; never silently fall back."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_root: Path) -> tuple[Path, str]:
    """Verify exact Git revision and ATen source bytes before any GPU launch."""
    source_root = source_root.expanduser().resolve()
    source_file = source_root / SOURCE_PATH
    if not source_file.is_file():
        raise RunnerBlocked(f"RUNNER_BLOCKED: missing pinned source {source_file}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerBlocked("RUNNER_BLOCKED: source_root is not a readable Git checkout") from exc
    if revision != PINNED_REVISION:
        raise RunnerBlocked(f"RUNNER_BLOCKED: expected revision {PINNED_REVISION}, got {revision}")
    actual_hash = sha256_file(source_file)
    if actual_hash != PINNED_SHA256:
        raise RunnerBlocked(f"RUNNER_BLOCKED: source SHA-256 mismatch: expected {PINNED_SHA256}, got {actual_hash}")
    return source_file, revision


def _runtime():
    """Load the real Torch/CUDA runtime only after source verification."""
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: PyTorch runtime unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RunnerBlocked("RUNNER_BLOCKED: torch.cuda.is_available() is false")
    return torch


def run_candidate(source_root: Path, regime: str, candidate_id: str) -> dict:
    """Launch the verified ATen LayerNorm path and validate every output element."""
    # Local imports avoid making CPU catalog/reference tests depend on Torch.
    from pytorch_layernorm_adapter import cell_descriptor
    from correctness_reference import (
        pytorch_layer_norm_output_matches,
        softmax_input,
    )

    _, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch = _runtime()
    meta = descriptor["build"]["compile_meta"]
    rows, cols, stride = int(meta["rows"]), int(meta["cols"]), int(meta["row_stride"])
    values = softmax_input(rows, stride, seed=VALUE_SEED)
    # Padded row-major buffer on device; the logical problem is the leading
    # ``cols`` entries of each stride row, so the narrowed view is
    # non-contiguous with exactly the catalog's row stride. LayerNorm over the
    # last dim with identity affine dispatches through aten::layer_norm on that
    # strided layout.
    padded = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(rows, stride)
    view = padded[:, :cols]
    assert view.stride(0) == stride and view.stride(1) == 1, "strided view lost the catalog layout"
    weight = torch.ones(cols, dtype=torch.float32, device="cuda")
    bias = torch.zeros(cols, dtype=torch.float32, device="cuda")
    out = torch.nn.functional.layer_norm(view, [cols], weight, bias, eps=1e-5)
    torch.cuda.synchronize()
    output = out.detach().cpu().reshape(-1).tolist()
    if not pytorch_layer_norm_output_matches(rows, cols, stride, output, values):
        raise RuntimeError("CORRUPT_OUTPUT_REJECTED: full pytorch-layernorm output mismatches reference")
    # Status mirrors the catalog row: runner exists and is CPU-tested, GPU
    # validation pending (pytorch-softmax precedent M283). Promote to
    # compiled_gpu_verified only on coordinator acceptance of GPU evidence.
    result = {
        "status": "implemented_unvalidated_runtime",
        "source_revision": revision,
        "source_sha256": PINNED_SHA256,
        "source_path": SOURCE_PATH,
        "regime": regime,
        "candidate_id": candidate_id,
        "rows": rows,
        "cols": cols,
        "controls": descriptor["build"]["compile_meta"],
        "reference": {"elements_checked": rows * cols, "criterion": "close(abs<=1e-5*max(1,|a|,|b|))"},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
    }
    return result


def prepare_energy_context(source_root: Path, regime: str, candidate_id: str):
    """Build once, return an EnergyContext for energy_harness/application_energy_harness.py.
    Mirrors run_candidate()'s setup exactly; does not call or affect it."""
    hpc_dir = Path(__file__).resolve().parents[2] / "energy_harness"
    if str(hpc_dir) not in sys.path:
        sys.path.insert(0, str(hpc_dir))
    from application_energy_harness import EnergyContext
    from torch_graph_capture import make_graph_replayer

    from pytorch_layernorm_adapter import cell_descriptor
    from correctness_reference import pytorch_layer_norm_output_matches, softmax_input

    _, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch = _runtime()
    meta = descriptor["build"]["compile_meta"]
    rows, cols, stride = int(meta["rows"]), int(meta["cols"]), int(meta["row_stride"])
    values = softmax_input(rows, stride, seed=VALUE_SEED)
    padded = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(rows, stride)
    view = padded[:, :cols]
    weight = torch.ones(cols, dtype=torch.float32, device="cuda")
    bias = torch.zeros(cols, dtype=torch.float32, device="cuda")
    holder = {}

    def launch_once():
        holder["out"] = torch.nn.functional.layer_norm(view, [cols], weight, bias, eps=1e-5)

    def sync():
        torch.cuda.synchronize()

    def check():
        output = holder["out"].detach().cpu().reshape(-1).tolist()
        return pytorch_layer_norm_output_matches(rows, cols, stride, output, values)

    def cuda_event_seconds(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1000.0

    return EnergyContext(
        parent_id="final_pytorch_layer_norm", regime=regime, candidate_id=candidate_id,
        source_revision=revision, source_sha256=PINNED_SHA256, source_path=SOURCE_PATH,
        controls=meta, runtime_info={
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        },
        launch_once=launch_once, sync=sync, check=check, cuda_event_seconds=cuda_event_seconds,
        capture_graph=lambda batch: make_graph_replayer(torch, launch_once, batch),
    )


def self_test() -> None:
    from correctness_reference import (
        pytorch_layer_norm_output_matches,
        ref_pytorch_layer_norm,
        softmax_input,
    )
    from pytorch_layernorm_adapter import cell_descriptor

    with tempfile.TemporaryDirectory() as temp:
        fixture = Path(temp) / "source.cu"
        fixture.write_bytes(b"pinned fixture")
        expected = sha256_file(fixture)
        assert expected != PINNED_SHA256
        assert sha256_file(fixture) == expected
        fixture.write_bytes(b"corrupt fixture")
        assert sha256_file(fixture) != expected
    for regime in ("small", "medium", "large"):
        for candidate in ("c1", "c2", "c3", "c4"):
            d = cell_descriptor(regime, candidate)
            assert d["source_status"] == "source_verified"
            assert d["implementation_status"] == "implemented_unvalidated_runtime"
            assert d["semantics"]["same_problem_across_candidates"]
    rows, cols, stride = 5, 7, 8
    values = softmax_input(rows, stride, seed=VALUE_SEED)
    expected = ref_pytorch_layer_norm(rows, cols, stride, values)
    assert pytorch_layer_norm_output_matches(rows, cols, stride, expected, values)
    corrupt = list(expected); corrupt[11] += 0.25
    assert not pytorch_layer_norm_output_matches(rows, cols, stride, corrupt, values)
    assert not pytorch_layer_norm_output_matches(rows, cols, stride, expected[:-1], values)
    assert not pytorch_layer_norm_output_matches(rows, cols, stride, [float("nan")] * rows * cols, values)
    try:
        ref_pytorch_layer_norm(rows, cols, cols - 1, values)
        raise AssertionError("short row stride was accepted")
    except ValueError:
        pass
    print("CPU_ONLY_PYTORCH_LAYERNORM_RUNNER_OK: hash/reference/candidate/corruption checks passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--regime", choices=("small", "medium", "large"))
    parser.add_argument("--candidate", choices=("c1", "c2", "c3", "c4"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.source_root is None or args.regime is None or args.candidate is None:
        parser.error("--source-root, --regime, and --candidate are required unless --self-test")
    try:
        print(json.dumps(run_candidate(args.source_root, args.regime, args.candidate), sort_keys=True))
    except RunnerBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
