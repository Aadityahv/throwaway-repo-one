#!/usr/bin/env python3
"""Energy-harness entry point for the three unseen-operator parents.

``energy_harness/run_application_energy.py`` dispatches CUDA-driver runners through a fixed registry
(``BINARY_RUNNER_MODULES``).  This wrapper adds the three unseen-operator runners to that registry in-process and
then calls the harness's own ``main()`` unchanged, so every gate, tolerance, cooldown, precondition and retry rule
is exactly the committed harness's, and no committed file (harness, catalog, ``workload_cells.csv``) is edited.

Runner names: ``alt_reduction``, ``alt_copy``, ``alt_transposefine`` (the ``runner`` column of the raw CSV; the
``parent_id`` column carries ``alt_cuda_samples_reduction`` etc.).

Differences from calling the harness directly, both deliberate:
  * ``--out-dir`` is REQUIRED.  The harness default is ``tiresias/app_runners/application_energy_raw``, the
    development dataset; unseen-operator rows must never land there.  A directory under ``sealed_labels`` or named
    ``FINAL-*`` is refused as well.
  * ``--print-plan`` prints the exact commands for one session of the 21 cells and exits without touching anything.

Usage:
  run_unseen_operator_energy.py --print-plan --source-root DIR --out-dir DIR --platform blackwell --session 1
  run_unseen_operator_energy.py --runner alt_copy --source-root DIR --regime small --candidate c1 \\
      --window-target-seconds 15 --session 1 --run-id UNSEEN-... --out-dir DIR --platform blackwell --gpu-index 1
(all other flags are the harness's own; see energy_harness/run_application_energy.py).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import unseen_common as uc


def _refuse_bad_out_dir(out_dir: Path) -> None:
    resolved = out_dir.expanduser().resolve()
    dev_default = (uc.RESET_DIR / "application_energy_raw").resolve()
    if resolved == dev_default or dev_default in resolved.parents:
        raise SystemExit(f"FATAL: --out-dir {out_dir} is inside the development dataset directory {dev_default}. "
                         "Use a fresh directory for the unseen-operator campaign.")
    if "sealed_labels" in resolved.parts or any(p.startswith("FINAL-") for p in resolved.parts):
        raise SystemExit(f"FATAL: --out-dir {out_dir} is a sealed/final-label path. Refusing.")


def _flag_value(argv: list[str], flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def print_plan(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_unseen_operator_energy.py --print-plan")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--platform", required=True, choices=("blackwell", "ada", "h100", "a100"))
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--session", type=int, required=True, choices=(1, 2))
    ap.add_argument("--window-target-seconds", type=float, default=15.0)
    ap.add_argument("--run-prefix", default="UNSEEN")
    ap.add_argument("--reverse", action="store_true", help="walk the cells in reverse (counterbalanced second session)")
    args = ap.parse_args(argv)
    _refuse_bad_out_dir(args.out_dir)
    gpu = args.gpu_index if args.gpu_index is not None else (1 if args.platform == "blackwell" else None)
    if gpu is None:
        raise SystemExit("FATAL: --gpu-index is required for this platform (verified live, never assumed).")
    cells = uc.enumerate_cells()
    if args.reverse:
        cells = cells[::-1]
    here = Path(__file__).resolve()
    for c in cells:
        run_id = f"{args.run_prefix}-{c['runner']}-{c['regime']}-{c['candidate_id']}-s{args.session}"
        cmd = ["env", f"CUDA_VISIBLE_DEVICES={gpu}", "python3", str(here), "--runner", c["runner"], "--source-root",
               args.source_root, "--regime", c["regime"], "--candidate", c["candidate_id"],
               "--window-target-seconds", f"{args.window_target_seconds:g}", "--session", str(args.session),
               "--run-id", run_id, "--out-dir", str(args.out_dir), "--gpu-index", str(gpu), "--platform", args.platform]
        print(" ".join(shlex.quote(str(x)) for x in cmd))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print-plan" in argv:
        return print_plan(argv)
    out_dir = _flag_value(argv, "--out-dir")
    if not out_dir:
        raise SystemExit("FATAL: --out-dir is required (the harness default is the development dataset directory).")
    _refuse_bad_out_dir(Path(out_dir))

    import run_application_energy as harness_cli  # energy_harness/, on sys.path via unseen_common

    for runner, module in uc.RUNNER_MODULES.items():
        existing = harness_cli.BINARY_RUNNER_MODULES.get(runner)
        if existing is not None and existing != module:
            raise SystemExit(f"FATAL: harness registry already maps {runner!r} to {existing!r}")
        harness_cli.BINARY_RUNNER_MODULES[runner] = module
    if str(uc.HERE) not in sys.path:
        sys.path.insert(0, str(uc.HERE))
    sys.argv = [sys.argv[0]] + argv
    return harness_cli.main()


if __name__ == "__main__":
    sys.exit(main())
