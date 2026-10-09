#!/usr/bin/env python3
"""CLI entry point for Task C's application-energy harness.

Dispatches to one of the twelve application runners (tiresias/app_runners/*_runner.py):
the seven Python-callable ones via prepare_energy_context()/run_python_callable_energy(), or
the five CUDA-driver ones (native cudaGraph_t capture, no host-loop mode -- see
APPLICATION_ENERGY_HARNESS_RUNBOOK.md section 4) via prepare_binary_energy_context()/
run_binary_energy(). Both run through energy_harness/application_energy_harness.py's B-derived
cooldown/precondition/window-gate protocol.

Usage:
  run_application_energy.py --dry-run --runner vector_add --source-root <triton clone> \
      --regime small --candidate c1 --window-target-seconds 15 --session 1
  run_application_energy.py --runner vector_add --source-root <triton clone> \
      --regime small --candidate c1 --window-target-seconds 15 --session 1 \
      --run-id C-ENERGY-R1 --out-dir tiresias/app_runners/application_energy_raw \
      --gpu-index 1 --platform blackwell

--dry-run skips every hardware GATE (cooldown wait, idle/unshared check,
CUDA_VISIBLE_DEVICES enforcement) and the timed/NVML-sampled measurement loop,
so it never requires GPU1 idle and never blocks on cooldown. It is NOT a
zero-GPU preview, unlike energy_harness/measurement_runner.py's --dry-run for B's separate
binaries: prepare_energy_context() still imports the real Triton/PyTorch
runtime and allocates the candidate's input tensors on a CUDA device to build
the EnergyContext at all (there is no binary to inspect argv for instead). A
small, real device allocation happens even in --dry-run; a fully hardware-free
preview would need a further design change, not implemented here.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESET_DIR = REPO / "tiresias" / "app_runners"

RUNNER_MODULES = {
    "vector_add": "vector_add_runner",
    "triton_layernorm": "layernorm_runner",
    "softmax": "softmax_runner",
    "pytorch_softmax": "pytorch_softmax_runner",
    "pytorch_layernorm": "pytorch_layernorm_runner",
    "embedding": "embedding_runner",
    "xformers": "xformers_runner",
}

# CUDA-driver runners: native cudaGraph_t capture only (APPLICATION_ENERGY_HARNESS_RUNBOOK.md
# section 4), never dry-run-verified for real correctness/output on a GPU yet.
BINARY_RUNNER_MODULES = {
    "cub_block": "cub_block_runner",
    "cub_device": "cub_device_runner",
    "transpose": "transpose_runner",
    "reduction": "reduction_runner",
    "copy": "copy_runner",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, choices=sorted(set(RUNNER_MODULES) | set(BINARY_RUNNER_MODULES)))
    ap.add_argument("--source-root", type=Path, required=True,
                     help="Pinned upstream checkout (Triton/PyTorch/xFormers/CCCL/CUDA-Samples clone) "
                          "the chosen runner's own verify_source() checks against a specific revision/hash.")
    ap.add_argument("--regime", required=True, choices=("small", "medium", "large"))
    ap.add_argument("--candidate", required=True, choices=("c1", "c2", "c3", "c4"))
    ap.add_argument("--window-target-seconds", type=float, required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out-dir", type=Path, default=RESET_DIR / "application_energy_raw")
    ap.add_argument("--launch-model", choices=("graph", "host"), default="graph",
                     help="graph (default): replay a CUDA graph of --graph-batch launches, as B's fixtures "
                          "do (M299/M311). host: plain Python loop, the declared cross-check only -- its "
                          "rows are dominated by Python submission (M310) and must not be pooled. Ignored "
                          "for the five CUDA-driver runners, which only implement graph capture in C++.")
    ap.add_argument("--graph-batch", type=int, default=1000)
    ap.add_argument("--workdir", type=Path, default=None,
                     help="Build/scratch directory for the five CUDA-driver runners (build_binary() output, "
                          "input/output files, calibration probes). Defaults to a fresh temp dir. Ignored "
                          "for the seven Python-callable runners.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--platform", default="blackwell", choices=["blackwell", "ada", "a100", "h100"])
    ap.add_argument("--gpu-index", type=int, default=None,
                     help="Defaults to 1 only for --platform blackwell (GPU0 there is never a "
                          "valid target while another lab user's session may be on it -- verify "
                          "live via nvidia-smi first, not assumed). No default for other platforms.")
    args = ap.parse_args()

    is_binary = args.runner in BINARY_RUNNER_MODULES
    if is_binary and args.launch_model == "host":
        print(f"FATAL: --runner {args.runner} only implements native CUDA-graph capture "
              "(APPLICATION_ENERGY_HARNESS_RUNBOOK.md section 4); there is no host-loop mode "
              "to select. Omit --launch-model or pass --launch-model graph.", file=sys.stderr)
        return 1

    gpu_index = args.gpu_index
    if gpu_index is None and args.platform == "blackwell":
        gpu_index = 1
    if gpu_index is None:
        print(f"FATAL: --gpu-index is required for --platform {args.platform} "
              "(no default assumed; verify live first).", file=sys.stderr)
        return 1
    if args.platform == "blackwell" and gpu_index != 1:
        print("FATAL: --platform blackwell only ever supports --gpu-index 1. Refusing.", file=sys.stderr)
        return 1

    if str(RESET_DIR) not in sys.path:
        sys.path.insert(0, str(RESET_DIR))
    # The CUDA-driver runners build for HARNESS_CUDA_ARCH (cuda_build_target.py). Derive it from
    # --platform so the arch can never silently disagree with the platform being measured.
    from cuda_build_target import PLATFORM_ARCH
    platform_arch = PLATFORM_ARCH[args.platform]
    preset_arch = os.environ.get("HARNESS_CUDA_ARCH", "").strip()
    if preset_arch and preset_arch != platform_arch:
        print(f"FATAL: HARNESS_CUDA_ARCH={preset_arch} contradicts --platform {args.platform} "
              f"({platform_arch}). Refusing.", file=sys.stderr)
        return 1
    os.environ["HARNESS_CUDA_ARCH"] = platform_arch
    module = importlib.import_module((BINARY_RUNNER_MODULES if is_binary else RUNNER_MODULES)[args.runner])

    if str(REPO / "energy_harness") not in sys.path:
        sys.path.insert(0, str(REPO / "energy_harness"))
    from application_energy_harness import (
        run_python_callable_energy, run_binary_energy, append_raw_or_rejected, preflight_before_context,
    )

    run_id = args.run_id or f"C-ENERGY-{args.runner}-{args.regime}-{args.candidate}"

    # preflight_before_context() MUST run before prepare_energy_context()/prepare_binary_energy_context():
    # the Python-callable path allocates real device tensors in THIS process, and the idle/unshared check
    # it reuses from B (energy_harness/measurement_runner.py) has no way to tell "my own about-to-exist context" apart
    # from another user's process -- it would reject every run on its own memory otherwise. The binary path
    # doesn't touch the GPU here either (only build_binary(), a CPU-only compile), so the same ordering is
    # just as safe and keeps one code path. Skipped entirely in --dry-run (see its own --help text).
    gpu_uuid = "" if args.dry_run else preflight_before_context(gpu_index, args.platform)

    import tempfile
    workdir_cm = None
    try:
        if is_binary:
            workdir = args.workdir
            if workdir is None:
                workdir_cm = tempfile.TemporaryDirectory(prefix=f"{args.runner}_energy_")
                workdir = Path(workdir_cm.name)
            workdir.mkdir(parents=True, exist_ok=True)
            ctx = module.prepare_binary_energy_context(args.source_root, args.regime, args.candidate, workdir)
        else:
            ctx = module.prepare_energy_context(args.source_root, args.regime, args.candidate)
    except module.RunnerBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 -- narrowed below; anything else is a real bug and re-raised
        # Each adapter raises its own *ConfigError when the catalog has no such (regime, candidate) cell,
        # e.g. cub_device only defines c1 per regime. That is a non-existent cell, not a failed measurement.
        if type(exc).__name__.endswith("ConfigError"):
            print(f"NO_SUCH_CELL: {args.runner} {args.regime}/{args.candidate}: {exc}", file=sys.stderr)
            return 4
        raise

    if is_binary:
        result = run_binary_energy(
            ctx, window_target_seconds=args.window_target_seconds, session=args.session,
            run_id=run_id, out_dir=args.out_dir, gpu_index=gpu_index, platform=args.platform,
            runner_name=args.runner, gpu_uuid=gpu_uuid, dry_run=args.dry_run,
            graph_batch=args.graph_batch,
        )
    else:
        result = run_python_callable_energy(
            ctx, window_target_seconds=args.window_target_seconds, session=args.session,
            run_id=run_id, out_dir=args.out_dir, gpu_index=gpu_index, platform=args.platform,
            runner_name=args.runner, gpu_uuid=gpu_uuid, dry_run=args.dry_run,
            launch_model=args.launch_model, graph_batch=args.graph_batch,
        )

    if workdir_cm is not None:
        workdir_cm.cleanup()

    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
        return 0

    append_raw_or_rejected(args.out_dir, result)
    if result["status"] == "raw":
        print(f"RAW {json.dumps(result['row'], default=str)}")
        return 0
    print(f"REJECTED {json.dumps(result['row'], default=str)}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
