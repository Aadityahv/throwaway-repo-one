#!/usr/bin/env python3
"""Run the pinned xFormers index_select_cat forward kernel, when its real runtime is present.

The repository deliberately contains no Triton/PyTorch/CUDA dependency. This
runner therefore has a strict two-state contract: source and runtime checks
either pass and the actual pinned forward kernel is launched, or the command
fails loudly without downloading, installing, or substituting an
implementation. The CPU self-test exercises source hashing, candidate
controls, and the full reference/rejection path without importing Triton or
touching a GPU.

Relative to the file's own ``index_select_cat_fwd()`` wrapper (fixed
``BLOCK_SIZE_COL=512`` plus shape checks that must never execute here): this
runner AST-extracts only the decorated ``index_select_cat_fwd_kernel`` and
launches it with the catalog candidate's native ``BLOCK_SIZE_COL``.
``BLOCK_SIZE_INDEX`` stays 1 as in the source wrapper. The backward kernel,
the wrapper validation, and every other top-level statement are never imported
or executed.

Tolerance note: the CPU oracle builds canonical source values as exact
``float(row * dim + col)``. A float32 gather returns those values rounded to
float32, which differs from the float64 oracle above 2**24 (the large regime
reaches ~5.4e8). The runtime check therefore compares every output element
under the cell's declared tolerance (rtol=1e-5, atol=1e-6) instead of claiming
bit-exactness the hardware cannot provide.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import linecache
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = "xformers/ops/_triton/k_index_select_cat.py"
PINNED_REVISION = "6d925b3a94312fb46249d83879cab045b6a0aaae"
PINNED_SHA256 = "fe0140bae51b29dfc21f403091cfecb29c44d2599de18fa02e774bbb71b76eb0"
KERNEL_NAME = "index_select_cat_fwd_kernel"
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


def deterministic_indices(num_rows: int, num_indices: int, seed: int = SEED) -> list[int]:
    """Fixed pseudorandom in-range int64 indices; identical on every host."""
    if num_rows <= 0 or num_indices <= 0 or num_indices >= num_rows:
        raise RunnerBlocked("RUNNER_BLOCKED: source wrapper requires 0 < num_indices < num_rows")
    return [(seed * (i + 1) * 2654435761) % num_rows for i in range(num_indices)]


def verify_source(source_root: Path) -> tuple[Path, str]:
    """Verify exact Git revision and source bytes before importing anything."""
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
    """Return exactly the decorated forward kernel, never executing the module."""
    source = source_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_file))
    except SyntaxError as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: pinned source is not valid Python: {exc}") from exc
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == KERNEL_NAME]
    if len(matches) != 1:
        raise RunnerBlocked(f"RUNNER_BLOCKED: expected exactly one {KERNEL_NAME}, found {len(matches)}")
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
    filename = str(source_file) + ":index_select_cat_fwd_kernel_extracted"
    linecache.cache[filename] = (len(extracted), None, extracted.splitlines(keepends=True), filename)
    namespace = {"triton": triton, "tl": tl, "__name__": "_pinned_xformers_kernel_only"}
    try:
        exec(compile(extracted, filename, "exec"), namespace, namespace)
    except Exception as exc:
        raise RunnerBlocked(f"RUNNER_BLOCKED: extracted kernel compilation failed: {exc}") from exc
    kernel = namespace.get(KERNEL_NAME)
    if kernel is None:
        raise RunnerBlocked(f"RUNNER_BLOCKED: extracted source has no {KERNEL_NAME} symbol")
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


def xformers_output_matches(num_rows: int, num_cols: int, indices: list[int],
                            output: list[float]) -> bool:
    """Full-output check under the cell tolerance (see tolerance note above)."""
    from correctness_reference import ref_xformers_index_select

    if len(output) != len(indices) * num_cols:
        return False
    if not all(math.isfinite(v) for v in output):
        return False
    expected = ref_xformers_index_select(num_rows, num_cols, indices)
    return all(abs(got - want) <= ATOL + RTOL * abs(want)
               for got, want in zip(output, expected))


def run_candidate(source_root: Path, regime: str, candidate_id: str) -> dict:
    """Launch the verified source kernel and validate every output element."""
    # Local imports avoid making CPU catalog/reference tests depend on Triton.
    from xformers_adapter import cell_descriptor

    source_file, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch, triton = _runtime()
    kernel = import_pinned_kernel(source_file)
    meta = descriptor["build"]["compile_meta"]
    args = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in descriptor["launch"]["runtime_args"]
            if "=" in kv and kv.split("=")[1].lstrip("-").isdigit()}
    num_rows, num_indices, num_cols = args["num_rows"], args["num_indices"], args["num_cols"]
    block_col = int(meta["BLOCK_SIZE_COL"])
    assert int(meta["BLOCK_SIZE_INDEX"]) == 1, "catalog fixes BLOCK_SIZE_INDEX=1"
    indices = deterministic_indices(num_rows, num_indices)
    # Contiguous float32 source with the reference's canonical values (float32
    # rounding is absorbed by the declared tolerance in the output check);
    # int64 index tensor; output gathered into a plain num_indices x num_cols
    # buffer, matching the kernel's source-strided store for this layout.
    source = torch.arange(num_rows * num_cols, dtype=torch.float32, device="cuda").reshape(num_rows, num_cols)
    index = torch.tensor(indices, dtype=torch.int64, device="cuda")
    output = torch.empty(num_indices, num_cols, dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(num_indices, 1), triton.cdiv(num_cols, block_col))
    kernel[grid](output, source, index, num_indices, num_cols, num_cols, 1,
                 BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=block_col)
    torch.cuda.synchronize()
    flat = output.detach().cpu().reshape(-1).tolist()
    if not xformers_output_matches(num_rows, num_cols, indices, flat):
        raise RuntimeError("CORRUPT_OUTPUT_REJECTED: full xformers output mismatches reference")
    # Status mirrors the catalog row: runner exists and is CPU-tested, GPU
    # validation pending (vector-add precedent M269/M271). Promote to
    # compiled_gpu_verified only on coordinator acceptance of GPU evidence.
    result = {
        "status": "implemented_unvalidated_runtime",
        "source_revision": revision,
        "source_sha256": PINNED_SHA256,
        "source_path": SOURCE_PATH,
        "regime": regime,
        "candidate_id": candidate_id,
        "num_rows": num_rows,
        "num_indices": num_indices,
        "num_cols": num_cols,
        "controls": descriptor["build"]["compile_meta"],
        "reference": {"elements_checked": num_indices * num_cols, "rtol": RTOL, "atol": ATOL},
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

    from xformers_adapter import cell_descriptor

    source_file, revision = verify_source(source_root)
    descriptor = cell_descriptor(regime, candidate_id)
    torch, triton = _runtime()
    kernel = import_pinned_kernel(source_file)
    meta = descriptor["build"]["compile_meta"]
    args = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in descriptor["launch"]["runtime_args"]
            if "=" in kv and kv.split("=")[1].lstrip("-").isdigit()}
    num_rows, num_indices, num_cols = args["num_rows"], args["num_indices"], args["num_cols"]
    block_col = int(meta["BLOCK_SIZE_COL"])
    indices = deterministic_indices(num_rows, num_indices)
    source = torch.arange(num_rows * num_cols, dtype=torch.float32, device="cuda").reshape(num_rows, num_cols)
    index = torch.tensor(indices, dtype=torch.int64, device="cuda")
    output = torch.empty(num_indices, num_cols, dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(num_indices, 1), triton.cdiv(num_cols, block_col))

    def launch_once():
        kernel[grid](output, source, index, num_indices, num_cols, num_cols, 1,
                     BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=block_col)

    def sync():
        torch.cuda.synchronize()

    def check():
        flat = output.detach().cpu().reshape(-1).tolist()
        return xformers_output_matches(num_rows, num_cols, indices, flat)

    def cuda_event_seconds(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1000.0

    return EnergyContext(
        parent_id="final_xformers_indexed_select", regime=regime, candidate_id=candidate_id,
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
    from correctness_reference import ref_xformers_index_select
    from xformers_adapter import cell_descriptor

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
            "import torch\n"
            "torch.manual_seed(0)\n"
            "@triton.jit\n"
            "def index_select_cat_fwd_kernel(o_ptr, s_ptr, i_ptr, n, c, s0, s1, B0: tl.constexpr, B1: tl.constexpr):\n    return tl.load(s_ptr)\n"
            "@triton.jit\n"
            "def index_select_cat_bwd_kernel(g):\n    return g\n"
            "def index_select_cat_fwd(o, s, i):\n    raise RuntimeError('WRAPPER_MUST_NOT_RUN')\n",
            encoding="utf-8")
        extracted = extract_pinned_kernel_source(guarded)
        assert "def index_select_cat_fwd_kernel" in extracted
        assert "index_select_cat_bwd_kernel" not in extracted
        assert "WRAPPER_MUST_NOT_RUN" not in extracted
        assert "torch.manual_seed" not in extracted
        ambiguous = Path(temp) / "ambiguous.py"
        ambiguous.write_text(
            "def index_select_cat_fwd_kernel(): pass\ndef index_select_cat_fwd_kernel(): pass\n",
            encoding="utf-8")
        try:
            extract_pinned_kernel_source(ambiguous)
            raise AssertionError("ambiguous kernel source was accepted")
        except RunnerBlocked:
            pass
    assert deterministic_indices(4096, 1024) == deterministic_indices(4096, 1024)
    assert all(0 <= i < 4096 for i in deterministic_indices(4096, 1024))
    for regime in ("small", "medium", "large"):
        for candidate in ("c1", "c2", "c3", "c4"):
            d = cell_descriptor(regime, candidate)
            assert d["source_status"] == "source_verified"
            assert d["implementation_status"] == "implemented_unvalidated_runtime"
            assert d["semantics"]["same_problem_across_candidates"]
    num_rows, num_cols, selected = 11, 6, 5
    indices = deterministic_indices(num_rows, selected)
    expected = ref_xformers_index_select(num_rows, num_cols, indices)
    assert xformers_output_matches(num_rows, num_cols, indices, expected)
    corrupt = list(expected); corrupt[3] += 0.25
    assert not xformers_output_matches(num_rows, num_cols, indices, corrupt)
    assert not xformers_output_matches(num_rows, num_cols, indices, expected[:-1])
    assert not xformers_output_matches(num_rows, num_cols, indices, [float("nan")] * selected * num_cols)
    try:
        deterministic_indices(8, 8)
        raise AssertionError("num_indices == num_rows was accepted")
    except RunnerBlocked:
        pass
    print("CPU_ONLY_XFORMERS_RUNNER_OK: hash/reference/candidate/corruption checks passed")


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
