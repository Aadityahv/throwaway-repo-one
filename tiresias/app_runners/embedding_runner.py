#!/usr/bin/env python3
"""Run the pinned ATen index_select gather, when its real runtime is present.

The repository deliberately contains no PyTorch/CUDA dependency. This runner
therefore has a strict two-state contract: source and runtime checks either
pass and the actual pinned ATen gather path is launched, or the command fails
loudly without downloading, installing, or substituting an implementation. The
CPU self-test exercises source hashing, candidate controls, and the full
reference/rejection path without importing Torch or touching a GPU.

ATen note: like SoftMax.cu, ``aten/src/ATen/native/cuda/Indexing.cu`` is C++
compiled into libtorch, not an extractable kernel. The GPU path therefore
verifies the pinned source checkout (exact revision plus file SHA-256) and
then launches the gather through the installed Torch ``aten::index_select``
CUDA dispatch on the catalog's exact contiguous layout with int64 indices.
The emitted record carries the installed Torch/CUDA build identity alongside
the pinned source identity, so a reviewer can see exactly what was executed.

Tolerance note: the CPU oracle builds canonical source values as exact
``float(row * dim + col)``. A float32 gather returns those values rounded to
float32, which differs from the float64 oracle above 2**24 (the large regime
reaches ~5.4e8). The runtime check therefore compares every output element
under the cell's declared tolerance (rtol=1e-5, atol=1e-6) instead of claiming
bit-exactness the hardware cannot provide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = "aten/src/ATen/native/cuda/Indexing.cu"
PINNED_REVISION = "67faf385dd4133d3e658f9b1c2a4c87dc9b61218"
PINNED_SHA256 = "4f871517560f51d022951e867f4bca9b26697a4af93ee7c4821aeda4d8f26fc9"
SEED = 20260914
RTOL = 1e-5
ATOL = 1e-6


class RunnerBlocked(RuntimeError):
    """A required provenance/runtime check failed; never silently fall back."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_indices(num_weights: int, num_indices: int, seed: int = SEED) -> list[int]:
    """Fixed pseudorandom in-range int64 indices; identical on every host."""
    if num_weights <= 0 or num_indices <= 0:
        raise RunnerBlocked("RUNNER_BLOCKED: non-positive gather shape")
    return [(seed * (i + 1) * 2654435761) % num_weights for i in range(num_indices)]


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


def embedding_output_matches(num_rows: int, feature_dim: int, indices: list[int],
                             output: list[float]) -> bool:
    """Full-output check under the cell tolerance (see tolerance note above)."""
    from correctness_reference import ref_pytorch_index_select

    if len(output) != len(indices) * feature_dim:
        return False
    if not all(math.isfinite(v) for v in output):
        return False
    expected = ref_pytorch_index_select(num_rows, feature_dim, indices)
    return all(abs(got - want) <= ATOL + RTOL * abs(want)
               for got, want in zip(output, expected))


def run_candidate(source_root: Path, regime: str, candidate_id: str) -> dict:
    """Launch the verified ATen gather and validate every output element."""
    # Local imports avoid making CPU catalog/reference tests depend on Torch.
    from embedding_adapter import cell_descriptor

    _, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch = _runtime()
    meta = descriptor["build"]["compile_meta"]
    vocab, dim, selected = int(meta["num_weights"]), int(meta["feature_dim"]), int(meta["num_indices"])
    indices = deterministic_indices(vocab, selected)
    # Canonical contiguous source with the reference's exact values; float32
    # rounding is absorbed by the declared tolerance in the output check.
    source = torch.arange(vocab * dim, dtype=torch.float32, device="cuda").reshape(vocab, dim)
    index = torch.tensor(indices, dtype=torch.int64, device="cuda")
    out = torch.index_select(source, 0, index)
    torch.cuda.synchronize()
    output = out.detach().cpu().reshape(-1).tolist()
    if not embedding_output_matches(vocab, dim, indices, output):
        raise RuntimeError("CORRUPT_OUTPUT_REJECTED: full embedding output mismatches reference")
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
        "num_weights": vocab,
        "feature_dim": dim,
        "num_indices": selected,
        "controls": descriptor["build"]["compile_meta"],
        "reference": {"elements_checked": selected * dim, "rtol": RTOL, "atol": ATOL},
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

    from embedding_adapter import cell_descriptor

    _, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch = _runtime()
    meta = descriptor["build"]["compile_meta"]
    vocab, dim, selected = int(meta["num_weights"]), int(meta["feature_dim"]), int(meta["num_indices"])
    indices = deterministic_indices(vocab, selected)
    source = torch.arange(vocab * dim, dtype=torch.float32, device="cuda").reshape(vocab, dim)
    index = torch.tensor(indices, dtype=torch.int64, device="cuda")
    holder = {}

    def launch_once():
        holder["out"] = torch.index_select(source, 0, index)

    def sync():
        torch.cuda.synchronize()

    def check():
        output = holder["out"].detach().cpu().reshape(-1).tolist()
        return embedding_output_matches(vocab, dim, indices, output)

    def cuda_event_seconds(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1000.0

    return EnergyContext(
        parent_id="final_pytorch_embedding", regime=regime, candidate_id=candidate_id,
        source_revision=revision, source_sha256=PINNED_SHA256, source_path=SOURCE_PATH,
        controls=meta, runtime_info={
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        },
        launch_once=launch_once, sync=sync, check=check, cuda_event_seconds=cuda_event_seconds,
        capture_graph=lambda batch: make_graph_replayer(torch, launch_once, batch),
    )


def self_test() -> None:
    from correctness_reference import ref_pytorch_index_select
    from embedding_adapter import cell_descriptor

    with tempfile.TemporaryDirectory() as temp:
        fixture = Path(temp) / "source.cu"
        fixture.write_bytes(b"pinned fixture")
        expected = sha256_file(fixture)
        assert expected != PINNED_SHA256
        assert sha256_file(fixture) == expected
        fixture.write_bytes(b"corrupt fixture")
        assert sha256_file(fixture) != expected
    assert deterministic_indices(4096, 1024) == deterministic_indices(4096, 1024)
    assert len(set(deterministic_indices(4096, 1024))) > 1
    assert all(0 <= i < 4096 for i in deterministic_indices(4096, 1024))
    for regime in ("small", "medium", "large"):
        d = cell_descriptor(regime, "c1")
        assert d["source_status"] == "source_verified"
        assert d["implementation_status"] == "implemented_unvalidated_runtime"
        assert not d["semantics"]["same_problem_across_candidates"]
    vocab, dim, selected = 8, 4, 5
    indices = deterministic_indices(vocab, selected)
    expected = ref_pytorch_index_select(vocab, dim, indices)
    assert embedding_output_matches(vocab, dim, indices, expected)
    corrupt = list(expected); corrupt[3] += 0.25
    assert not embedding_output_matches(vocab, dim, indices, corrupt)
    assert not embedding_output_matches(vocab, dim, indices, expected[:-1])
    assert not embedding_output_matches(vocab, dim, indices, [float("nan")] * selected * dim)
    try:
        ref_pytorch_index_select(vocab, dim, [])
        raise AssertionError("empty index list was accepted")
    except ValueError:
        pass
    try:
        ref_pytorch_index_select(vocab, dim, [vocab])
        raise AssertionError("out-of-range index was accepted")
    except ValueError:
        pass
    print("CPU_ONLY_EMBEDDING_RUNNER_OK: hash/reference/candidate/corruption checks passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--regime", choices=("small", "medium", "large"))
    parser.add_argument("--candidate", choices=("c1",))
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
