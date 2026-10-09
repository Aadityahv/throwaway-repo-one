#!/usr/bin/env python3
"""Runner for the unseen-operator parent ``alt_cuda_samples_reduction``: pinned CUDA Samples reduce0, reduce1, reduce6.

Candidates c1 = reduce0, c2 = reduce1, c3 = reduce6 (sample whichKernel 0, 1, 6) from the same pinned
``reduction_kernel.cu`` the existing reduction runner uses for whichKernel 2..5.  Everything mechanical is reused,
not re-derived: the GPU-validated driver (``reduction_runner.DRIVER_TEMPLATE``: single-pass ``reduce<float>()``
dispatch, host completion of the per-block partials, native cudaGraph capture on the per-thread default stream),
the seeded input, the oracle (sum of the seed-20260914 float32 buffer, accumulated in double) and the tolerances
(RTOL 1e-5, ATOL 1e-6).  Only the candidate table, the launch geometry and the result fields are new.

Launch geometry (256 threads per block, all cells):
  reduce0, reduce1: one element per thread, blocks = n / 256 (256 / 4096 / 65536).
  reduce6: grid capped at the sample's default 64 blocks (REDUCE6_MAX_BLOCKS), so 64 in every regime.
The byte model of the freeze, 4n + 4*blocks, therefore uses the actual block count; ``controls`` carries it.

Accumulation check: pass/fail is the existing tolerance; the observed absolute error, relative error and error
relative to the L1 norm are additionally recorded per cell (controls_json in the energy path, the result JSON in
the correctness-only path).

No silent fallback: a missing checkout, hash mismatch, missing kernel symbol, missing nvcc or wrong-arch build
raises ``RunnerBlocked`` (exit code 3).  Nothing here touches NVML or measures energy.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import unseen_common as uc
from unseen_common import RunnerBlocked, UnseenOperatorConfigError  # noqa: F401  (RunnerBlocked is looked up by the CLI)

import reduction_runner as rr  # tiresias/app_runners, unchanged: driver, seed, tolerances, scalar_matches

PARENT = "alt_cuda_samples_reduction"
RUNNER = uc.PARENTS[PARENT]["runner"]
SOURCE_RELS = (uc.REDUCTION_KERNEL_REL, uc.REDUCTION_UTIL_REL)
KERNEL_NAMES = ("reduce0", "reduce1", "reduce6")
assert (rr.RTOL, rr.ATOL, rr.SEED) == (uc.RTOL, uc.ATOL, uc.SEED), "reduction oracle constants drifted from the existing runner"
assert rr.KERNEL_REL == uc.REDUCTION_KERNEL_REL and rr.UTIL_REL == uc.REDUCTION_UTIL_REL
assert rr.PINNED_KERNEL_SHA256 == uc.PINNED_SHA256[uc.REDUCTION_KERNEL_REL]
assert rr.PINNED_UTIL_SHA256 == uc.PINNED_SHA256[uc.REDUCTION_UTIL_REL]

_BUILD_CACHE: dict = {}


def cell_plan(regime: str, candidate_id: str) -> dict:
    """Manifest cell plus the launch geometry and byte model.  Raises UnseenOperatorConfigError for a non-cell."""
    cell = uc.cell_config(PARENT, regime, candidate_id)
    n, which = cell["n"], cell["selector"]
    blocks = uc.reduction_blocks(which, n)
    return {**cell, "threads": uc.REDUCTION_THREADS, "blocks": blocks, "which_kernel": which,
            "bytes_model": uc.reduction_bytes(n, blocks),
            "bytes_model_formula": "4*n + 4*blocks (blocks = actual grid size launched)"}


def error_metrics(got: float, want: float, values) -> dict:
    """Observed fp32 accumulation error against the double-precision sum (reported per cell, never used to relax a gate)."""
    abs_err = abs(got - want)
    l1 = math.fsum(abs(v) for v in values)
    return {"got_sum": got, "expected_sum": want, "abs_error": abs_err,
            "rel_error": abs_err / abs(want) if want != 0 else float("inf"),
            "err_over_l1": abs_err / l1 if l1 > 0 else float("inf"), "l1_norm": l1,
            "rtol": uc.RTOL, "atol": uc.ATOL,
            "tolerance_abs": uc.ATOL + uc.RTOL * abs(want)}


def build_binary(source_root: Path, workdir: Path, nvcc: str) -> Path:
    """Same three nvcc steps as reduction_runner.build_binary (pinned kernel + util objects, sample main renamed at
    compile time only, per-thread default stream so the pinned dispatcher's launches are capturable), with this
    module's RunnerBlocked.  Built once per (source, arch, workdir) per process."""
    key = (str(source_root), uc.build_arch(), str(workdir))
    if key in _BUILD_CACHE and _BUILD_CACHE[key].is_file():
        return _BUILD_CACHE[key]
    kernel_src, util_src = source_root / rr.KERNEL_REL, source_root / rr.UTIL_REL
    header_dir, common_inc = kernel_src.parent, source_root / "Common"
    if not (source_root / rr.HEADER_REL).is_file():
        raise RunnerBlocked(f"RUNNER_BLOCKED: missing pinned header {source_root / rr.HEADER_REL}")
    if not common_inc.is_dir():
        raise RunnerBlocked(f"RUNNER_BLOCKED: missing helper headers at {common_inc}")
    (workdir / "reduction_driver.cu").write_text(rr.DRIVER_TEMPLATE, encoding="utf-8")
    kernel_obj, util_obj, binary = workdir / "reduction_kernel.o", workdir / "reduction_util.o", workdir / "reduction_harness"
    arch, flags = f"-arch={uc.build_arch()}", rr.DEFAULT_STREAM_FLAGS
    steps = [
        [nvcc, "-O2", arch, *flags, "-I", str(common_inc), "-I", str(header_dir), "-c", str(kernel_src), "-o", str(kernel_obj)],
        [nvcc, "-O2", arch, *flags, "-I", str(common_inc), "-I", str(header_dir), "-Dmain=reduction_sample_main_disabled",
         "-c", str(util_src), "-o", str(util_obj)],
        [nvcc, "-O2", arch, *flags, "-I", str(header_dir), str(workdir / "reduction_driver.cu"), str(kernel_obj),
         str(util_obj), "-o", str(binary)],
    ]
    _BUILD_CACHE[key] = uc.run_nvcc_steps(steps, workdir, binary)
    return binary


def _prepare(source_root: Path, regime: str, candidate_id: str, workdir: Path) -> dict:
    """Shared setup: cell plan, source checks, build, input file, oracle value.  Does not run the binary."""
    plan = cell_plan(regime, candidate_id)
    uc.check_manifest_consistency()
    root, revision, hashes = uc.verify_source(source_root, SOURCE_RELS)
    uc.check_reduce_kernels(uc.read_source_text(root, rr.KERNEL_REL), KERNEL_NAMES)
    nvcc = uc.find_nvcc()
    binary = build_binary(root, workdir, nvcc)
    values = rr.deterministic_inputs(plan["n"])
    in_path, out_path = workdir / "in.bin", workdir / "out.bin"
    with open(in_path, "wb") as f:
        values.tofile(f)
    expected = sum(values)  # the same double-precision oracle as correctness_reference.ref_sum
    return {"plan": plan, "root": root, "revision": revision, "hashes": hashes, "nvcc": nvcc, "binary": binary,
            "values": values, "in_path": in_path, "out_path": out_path, "expected": expected}


def _read_scalar(path: Path) -> float | None:
    out = array.array("f")
    try:
        with open(path, "rb") as f:
            out.fromfile(f, 1)
    except (OSError, EOFError):
        return None
    return out[0] if len(out) == 1 else None


def _argv_prefix(ctx: dict) -> list[str]:
    p = ctx["plan"]
    return [str(p["n"]), str(p["threads"]), str(p["blocks"]), str(p["which_kernel"]), str(ctx["in_path"]), str(ctx["out_path"])]


def _controls(plan: dict) -> dict:
    return {"kernel_name": plan["kernel_name"], "which_kernel": plan["which_kernel"], "threads": plan["threads"],
            "blocks": plan["blocks"], "n": plan["n"], "bytes_model": plan["bytes_model"],
            "bytes_model_formula": plan["bytes_model_formula"], "dtype": "float32"}


def prepare_binary_energy_context(source_root: Path, regime: str, candidate_id: str, workdir: Path):
    """Build once, return a BinaryEnergyContext for energy_harness/application_energy_harness.py (does not invoke the binary)."""
    ctx = _prepare(source_root, regime, candidate_id, workdir)
    plan = ctx["plan"]
    controls = _controls(plan)

    def check() -> bool:
        got = _read_scalar(ctx["out_path"])
        if got is None or not math.isfinite(got):
            return False
        # Record the observed error in controls_json: the harness serialises ctx.controls after check() returns.
        controls["accumulation_check"] = error_metrics(got, ctx["expected"], ctx["values"])
        return rr.scalar_matches(got, ctx["expected"])

    return uc.import_binary_energy_context()(
        parent_id=PARENT, regime=regime, candidate_id=candidate_id,
        source_revision=ctx["revision"], source_sha256=ctx["hashes"][rr.KERNEL_REL], source_path=rr.KERNEL_REL,
        controls=controls, runtime_info={"python": platform.python_version(), "nvcc": ctx["nvcc"], "arch": uc.build_arch()},
        binary=ctx["binary"], argv_prefix=_argv_prefix(ctx), check=check)


def run_candidate(source_root: Path, regime: str, candidate_id: str, workdir: Path) -> dict:
    """Correctness-only: one single-shot launch (no graph, no NVML, no energy window) checked against the oracle."""
    ctx = _prepare(source_root, regime, candidate_id, workdir)
    plan = ctx["plan"]
    proc = subprocess.run([str(ctx["binary"]), *_argv_prefix(ctx)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RunnerBlocked(f"RUNNER_BLOCKED: harness run failed: {proc.stderr[-2000:]}")
    got = _read_scalar(ctx["out_path"])
    finite = got is not None and math.isfinite(got)
    metrics = error_metrics(got, ctx["expected"], ctx["values"]) if finite else {"got_sum": None}
    passed = bool(finite and rr.scalar_matches(got, ctx["expected"]))
    return {
        "status": "correctness_check_passed" if passed else "correctness_check_failed",
        "correctness_check": passed, "parent_id": PARENT, "runner": RUNNER, "regime": regime,
        "candidate_id": candidate_id, "source_revision": ctx["revision"], "source_sha256": ctx["hashes"][rr.KERNEL_REL],
        "source_path": rr.KERNEL_REL, "util_sha256": ctx["hashes"][rr.UTIL_REL], "controls": _controls(plan),
        "accumulation_check": metrics,
        "reference": {"elements_checked": plan["n"], "rtol": uc.RTOL, "atol": uc.ATOL,
                      "completion": "host double sum over per-block partials (sample --cpufinal mode)",
                      "input": f"seed {uc.SEED} uniform(-1,1) float32 (same as reduction_runner)"},
        "build": {"nvcc": ctx["nvcc"], "arch": uc.build_arch(), "flags": rr.DEFAULT_STREAM_FLAGS,
                  "defines": ["main=reduction_sample_main_disabled (util object only)"]},
        "runtime": {"python": platform.python_version(), "driver_stdout": proc.stdout.strip()},
    }


def self_test() -> None:
    """CPU-only: tables, geometry, source-check refusals, driver reuse.  Never invokes nvcc or touches a GPU."""
    uc.check_manifest_consistency()
    cells = [c for c in uc.enumerate_cells() if c["parent_id"] == PARENT]
    assert len(cells) == uc.EXPECTED_CELLS[PARENT] == 9
    expected_blocks = {("small", "c1"): 256, ("small", "c2"): 256, ("small", "c3"): 64,
                       ("medium", "c1"): 4096, ("medium", "c2"): 4096, ("medium", "c3"): 64,
                       ("large", "c1"): 65536, ("large", "c2"): 65536, ("large", "c3"): 64}
    for regime, cand in expected_blocks:
        plan = cell_plan(regime, cand)
        assert plan["blocks"] == expected_blocks[(regime, cand)], (regime, cand, plan["blocks"])
        assert plan["bytes_model"] == 4 * plan["n"] + 4 * plan["blocks"]
        assert plan["bytes_model"] <= 4 * plan["n"] + 4 * math.ceil(plan["n"] / uc.REDUCTION_THREADS)
    for bad in (("small", "c4"), ("tiny", "c1")):
        try:
            cell_plan(*bad)
        except UnseenOperatorConfigError:
            pass
        else:
            raise AssertionError(f"non-existent cell accepted: {bad}")
    assert "reduce<float>" in rr.DRIVER_TEMPLATE and "cudaStreamBeginCapture(cudaStreamPerThread" in rr.DRIVER_TEMPLATE
    assert rr.DEFAULT_STREAM_FLAGS == ["--default-stream", "per-thread"]
    vals = rr.deterministic_inputs(64)
    assert list(vals) == list(rr.deterministic_inputs(64))
    m = error_metrics(sum(vals), sum(vals), vals)
    assert m["abs_error"] == 0 and m["rel_error"] == 0
    print("CPU_ONLY_ALT_REDUCTION_RUNNER_OK: manifest/geometry/oracle/driver-reuse checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Correctness-only check of the pinned reduce0/reduce1/reduce6 candidates "
                                                 "(one single-shot launch per cell; no energy window).")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--regime", choices=uc.REGIME_ORDER)
    parser.add_argument("--candidate", choices=("c1", "c2", "c3"))
    parser.add_argument("--all-cells", action="store_true", help="run all 9 cells of this parent, one JSON line each")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.source_root is None or (not args.all_cells and (args.regime is None or args.candidate is None)):
        parser.error("--source-root and (--regime and --candidate, or --all-cells) are required unless --self-test")
    cells = ([(c["regime"], c["candidate_id"]) for c in uc.enumerate_cells() if c["parent_id"] == PARENT]
             if args.all_cells else [(args.regime, args.candidate)])
    worst = 0
    with tempfile.TemporaryDirectory(prefix="alt_reduction_harness_") as temp:
        workdir = args.workdir or Path(temp)
        workdir.mkdir(parents=True, exist_ok=True)
        for regime, cand in cells:
            try:
                result = run_candidate(args.source_root, regime, cand, workdir)
            except RunnerBlocked as exc:
                print(str(exc), file=sys.stderr)
                worst = 3
                if not args.all_cells:
                    return 3
                continue
            print(json.dumps(result, sort_keys=True))
            if not result["correctness_check"]:
                worst = 3
    return worst


if __name__ == "__main__":
    sys.exit(main())
