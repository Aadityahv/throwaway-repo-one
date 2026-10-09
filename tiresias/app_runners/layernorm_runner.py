#!/usr/bin/env python3
"""Run the pinned Triton tutorial layer-norm forward pass, when its real runtime is present.

The repository deliberately contains no Triton/PyTorch/CUDA dependency.  This
runner therefore has a strict two-state contract: source and runtime checks
either pass and the actual pinned tutorial kernel is launched, or the command
fails loudly without downloading, installing, or substituting an implementation.
The CPU self-test exercises source hashing, candidate controls, and the full
reference/rejection path without importing Triton or touching a GPU.

Scope is the FORWARD pass only (`_layer_norm_fwd_fused`). The tutorial's
backward kernels (`_layer_norm_bwd_dx_fused`, `_layer_norm_bwd_dwdb`), autograd
`LayerNorm` wrapper, `test_layer_norm` (which asserts backward agreement and
needs apex-optional paths), demo prints, and `@triton.testing.perf_report`
benchmark block must never execute: this runner AST-extracts only the decorated
forward kernel. Only the forward pass is in the catalog cells; backward coverage
would be a separately scoped addition, not an implied extension.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import linecache
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = "python/tutorials/05-layer-norm.py"
PINNED_REVISION = "7369aa3d9b0fae7134bf47b16d83876b09ef8e9f"
PINNED_SHA256 = "241f9f74e2f18567b26f3b4cf9b9873ae859150ccd10976d65d38fe24896d1d2"
SEED = 20260914
EPS = 1e-5


class RunnerBlocked(RuntimeError):
    """A required provenance/runtime check failed; never silently fall back."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_root: Path) -> tuple[Path, str]:
    """Verify exact Git revision and tutorial bytes before importing source."""
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


def extract_pinned_kernel_source(source_file: Path) -> str:
    """Return exactly the forward kernel, never executing the tutorial module."""
    source = source_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_file))
    except SyntaxError as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: pinned source is not valid Python: {exc}") from exc
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == "_layer_norm_fwd_fused"]
    if len(matches) != 1:
        raise RunnerBlocked(f"RUNNER_BLOCKED: expected exactly one _layer_norm_fwd_fused, found {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
    end = node.end_lineno
    kernel_source = "".join(lines[start:end])
    if not kernel_source or not node.decorator_list:
        raise RunnerBlocked("RUNNER_BLOCKED: extracted kernel is empty or undecorated")
    return kernel_source


def import_pinned_kernel(source_file: Path):
    """Compile only the extracted forward kernel against real Triton bindings."""
    try:
        import triton  # type: ignore
        import triton.language as tl  # type: ignore
    except ImportError as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: Triton runtime unavailable: {exc}") from exc
    extracted = extract_pinned_kernel_source(source_file)
    filename = str(source_file) + ":_layer_norm_fwd_fused_extracted"
    # Triton's JIT introspects decorated Python functions.  Registering the exact extracted text
    # lets inspect/linecache see the same code that exec compiles, without exposing tutorial setup.
    linecache.cache[filename] = (len(extracted), None, extracted.splitlines(keepends=True), filename)
    namespace = {"triton": triton, "tl": tl, "__name__": "_pinned_layernorm_kernel_only"}
    try:
        exec(compile(extracted, filename, "exec"), namespace, namespace)
    except Exception as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: extracted kernel compilation failed: {exc}") from exc
    kernel = namespace.get("_layer_norm_fwd_fused")
    if kernel is None:
        raise RunnerBlocked("RUNNER_BLOCKED: extracted source has no _layer_norm_fwd_fused symbol")
    return kernel


def _runtime():
    """Load the real runtime only after provenance verification."""
    try:
        import torch  # type: ignore
        import triton  # type: ignore
    except ImportError as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: Triton/PyTorch runtime unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RunnerBlocked("RUNNER_BLOCKED: torch.cuda.is_available() is false")
    return torch, triton


def run_candidate(source_root: Path, regime: str, candidate_id: str) -> dict:
    """Launch the verified forward kernel and validate every output element."""
    # Local imports avoid making CPU catalog/reference tests depend on Triton.
    from layernorm_adapter import cell_descriptor
    from correctness_reference import (
        layer_norm_output_matches,
        ref_layer_norm,
        softmax_input,
    )

    source_file, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch, triton = _runtime()
    kernel = import_pinned_kernel(source_file)
    meta = descriptor["build"]["compile_meta"]
    rows = int(descriptor["launch"]["runtime_args"][6].split("=")[1])
    cols = int(descriptor["launch"]["runtime_args"][7].split("=")[1])
    block_size = int(meta["BLOCK_SIZE"])
    num_warps = int(meta["num_warps"])
    # Cell-declared deterministic inputs (seed 20260914), affine identity (w=1, b=0):
    # matches ref_layer_norm's contract with explicit arguments, no defaults relied on.
    values = softmax_input(rows, cols, seed=SEED)
    w_vals = [1.0] * cols
    b_vals = [0.0] * cols
    x = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(rows, cols)
    w = torch.tensor(w_vals, dtype=torch.float32, device="cuda")
    b = torch.tensor(b_vals, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    mean = torch.empty((rows,), dtype=torch.float32, device="cuda")
    rstd = torch.empty((rows,), dtype=torch.float32, device="cuda")
    # One program per source-kernel row (tutorial's own launch shape); the tutorial's
    # BLOCK_SIZE//256 warp heuristic is not reimplemented -- num_warps comes from the
    # catalog candidate, identical procedure to the softmax runner.
    kernel[(rows,)](
        x, y, w, b, mean, rstd,
        cols, cols, EPS,
        BLOCK_SIZE=block_size, num_warps=num_warps, num_ctas=1,
    )
    torch.cuda.synchronize()
    output = y.detach().cpu().reshape(-1).tolist()
    if not layer_norm_output_matches(rows, cols, output, values, w_vals, b_vals):
        raise RuntimeError("CORRUPT_OUTPUT_REJECTED: full layer-norm output mismatches reference")
    # Status reflects coordinator-accepted GPU validation 2026-09-19 (12/12 cells).
    # Do not revert without coordinator direction.
    result = {
        "status": "compiled_gpu_verified",
        "source_revision": revision,
        "source_sha256": PINNED_SHA256,
        "source_path": SOURCE_PATH,
        "regime": regime,
        "candidate_id": candidate_id,
        "rows": rows,
        "cols": cols,
        "controls": descriptor["build"]["compile_meta"],
        "reference": {"elements_checked": rows * cols, "rtol": 2e-5, "atol": 2e-5},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", "unknown"),
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

    from layernorm_adapter import cell_descriptor
    from correctness_reference import layer_norm_output_matches, softmax_input

    source_file, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch, triton = _runtime()
    kernel = import_pinned_kernel(source_file)
    meta = descriptor["build"]["compile_meta"]
    rows = int(descriptor["launch"]["runtime_args"][6].split("=")[1])
    cols = int(descriptor["launch"]["runtime_args"][7].split("=")[1])
    block_size = int(meta["BLOCK_SIZE"])
    num_warps = int(meta["num_warps"])
    values = softmax_input(rows, cols, seed=SEED)
    w_vals = [1.0] * cols
    b_vals = [0.0] * cols
    x = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(rows, cols)
    w = torch.tensor(w_vals, dtype=torch.float32, device="cuda")
    b = torch.tensor(b_vals, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    mean = torch.empty((rows,), dtype=torch.float32, device="cuda")
    rstd = torch.empty((rows,), dtype=torch.float32, device="cuda")

    def launch_once():
        kernel[(rows,)](
            x, y, w, b, mean, rstd,
            cols, cols, EPS,
            BLOCK_SIZE=block_size, num_warps=num_warps, num_ctas=1,
        )

    def sync():
        torch.cuda.synchronize()

    def check():
        output = y.detach().cpu().reshape(-1).tolist()
        return layer_norm_output_matches(rows, cols, output, values, w_vals, b_vals)

    def cuda_event_seconds(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1000.0

    return EnergyContext(
        parent_id="final_triton_layer_norm", regime=regime, candidate_id=candidate_id,
        source_revision=revision, source_sha256=PINNED_SHA256, source_path=SOURCE_PATH,
        controls=meta, runtime_info={
            "python": platform.python_version(), "torch": torch.__version__,
            "triton": getattr(triton, "__version__", "unknown"),
            "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        },
        launch_once=launch_once, sync=sync, check=check, cuda_event_seconds=cuda_event_seconds,
        capture_graph=lambda batch: make_graph_replayer(torch, launch_once, batch),
    )


def self_test() -> None:
    from correctness_reference import (
        layer_norm_output_matches,
        ref_layer_norm,
        softmax_input,
    )
    from layernorm_adapter import cell_descriptor

    with tempfile.TemporaryDirectory() as temp:
        fixture = Path(temp) / "source.py"
        fixture.write_bytes(b"pinned fixture")
        expected = sha256_file(fixture)
        assert expected != PINNED_SHA256
        assert sha256_file(fixture) == expected
        fixture.write_bytes(b"corrupt fixture")
        assert sha256_file(fixture) != expected
        guarded = Path(temp) / "guarded.py"
        guarded.write_text(
            "import apex\n"
            "DEVICE = 'cuda'\n"
            "@triton.jit\n"
            "def _layer_norm_fwd_fused(X, Y):\n    return tl.load(X)\n"
            "@triton.jit\n"
            "def _layer_norm_bwd_dx_fused(DX):\n    return DX\n"
            "test_layer_norm(4, 8, 'float32')\n",
            encoding="utf-8")
        extracted = extract_pinned_kernel_source(guarded)
        assert "import apex" not in extracted
        assert "test_layer_norm" not in extracted
        assert "_layer_norm_bwd_dx_fused" not in extracted
        assert "def _layer_norm_fwd_fused" in extracted
        ambiguous = Path(temp) / "ambiguous.py"
        ambiguous.write_text("def _layer_norm_fwd_fused(): pass\ndef _layer_norm_fwd_fused(): pass\n",
                             encoding="utf-8")
        try:
            extract_pinned_kernel_source(ambiguous)
            raise AssertionError("ambiguous kernel source was accepted")
        except RunnerBlocked:
            pass
    for regime in ("small", "medium", "large"):
        for candidate in ("c1", "c2", "c3", "c4"):
            d = cell_descriptor(regime, candidate)
            assert d["source_status"] == "source_verified"
            assert d["implementation_status"] == "compiled_gpu_verified"
            assert d["semantics"]["same_problem_across_candidates"]
    values = softmax_input(5, 7, seed=SEED)
    expected = ref_layer_norm(5, 7, values, [1.0] * 7, [0.0] * 7)
    assert layer_norm_output_matches(5, 7, expected, values, [1.0] * 7, [0.0] * 7)
    corrupt = list(expected); corrupt[11] += 0.25
    assert not layer_norm_output_matches(5, 7, corrupt, values, [1.0] * 7, [0.0] * 7)
    assert not layer_norm_output_matches(5, 7, expected[:-1], values, [1.0] * 7, [0.0] * 7)
    assert not layer_norm_output_matches(5, 7, [float("nan")] * 35, values, [1.0] * 7, [0.0] * 7)
    print("CPU_ONLY_LAYERNORM_RUNNER_OK: hash/reference/candidate/corruption checks passed")


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
    main()
