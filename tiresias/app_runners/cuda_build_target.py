"""Build target (GPU arch + nvcc) for the CUDA-driver application runners.

The copy/transpose/reduction runners were written for Blackwell only and hard-coded
``-arch=sm_120`` with a CUDA 12.8 nvcc search. Running them on A100 (sm_80) or Ada (sm_89)
needs the arch and compiler to come from the platform instead. Blackwell stays the default,
so a Blackwell run with neither variable set builds exactly what it built before.

- ``HARNESS_CUDA_ARCH``: target arch, e.g. ``sm_80``. Unset -> ``sm_120`` (Blackwell).
  ``energy_harness/run_application_energy.py`` sets it from ``--platform``.
- ``HARNESS_NVCC``: absolute nvcc path. Unset -> the original search order. If set it must
  exist; there is no fallback to another compiler.

``assert_live_arch`` refuses to build for an arch other than the target GPU's own compute
capability, read from nvidia-smi for the GPU in ``CUDA_VISIBLE_DEVICES``. A wrong-arch binary
either fails at launch or runs through a JIT path this project never measured.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_ARCH = "sm_120"
PLATFORM_ARCH = {"blackwell": "sm_120", "a100": "sm_80", "h100": "sm_90", "ada": "sm_89"}
LEGACY_NVCC_CANDIDATES = ("/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda/bin/nvcc")


def target_arch(blocked: type[Exception]) -> str:
    arch = os.environ.get("HARNESS_CUDA_ARCH", "").strip() or DEFAULT_ARCH
    if not re.fullmatch(r"sm_\d{2,3}", arch):
        raise blocked(f"RUNNER_BLOCKED: HARNESS_CUDA_ARCH={arch!r} is not an sm_XX arch")
    return arch


def find_nvcc(blocked: type[Exception]) -> str:
    """Locate nvcc without modifying PATH."""
    explicit = os.environ.get("HARNESS_NVCC", "").strip()
    if explicit:
        if not Path(explicit).is_file():
            raise blocked(f"RUNNER_BLOCKED: HARNESS_NVCC={explicit} does not exist")
        return explicit
    for candidate in LEGACY_NVCC_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("nvcc")
    if found:
        return found
    raise blocked("RUNNER_BLOCKED: no nvcc found (set HARNESS_NVCC, or install CUDA 12.8+ for sm_120)")


def live_compute_arch(blocked: type[Exception]) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"\d+", cvd):
        raise blocked(f"RUNNER_BLOCKED: CUDA_VISIBLE_DEVICES must be one GPU index, got {cvd!r}")
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", cvd, "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise blocked("RUNNER_BLOCKED: could not read the target GPU's compute capability") from exc
    if not re.fullmatch(r"\d+\.\d+", out):
        raise blocked(f"RUNNER_BLOCKED: unexpected compute_cap {out!r} from nvidia-smi")
    major, minor = out.split(".")
    return f"sm_{major}{minor}"


def assert_live_arch(arch: str, blocked: type[Exception]) -> None:
    live = live_compute_arch(blocked)
    if live != arch:
        raise blocked(f"RUNNER_BLOCKED: build arch {arch} does not match the target GPU ({live})")
