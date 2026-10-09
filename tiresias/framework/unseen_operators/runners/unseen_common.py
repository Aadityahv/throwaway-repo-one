#!/usr/bin/env python3
"""Shared pieces for the three unseen-operator runners (no GPU, no energy, no measurement).

The unseen-operator test (../freeze.md) measures three new-code parents built from pinned CUDA Samples
sources.  Their shapes, candidates, source hashes and oracles are declared in
``tiresias/framework/adapters/adapters.json``; nothing here edits that file or the committed
``workload_cells.csv``.  The runners are self-describing: ``enumerate_cells()`` reads the manifest, and
every constant that also lives in the manifest is cross-checked against it (``check_manifest_consistency``).

Contract, same as the existing CUDA-driver runners: source, toolchain and runtime checks either pass and the
actual pinned sample kernels launch, or the command fails loudly (``RunnerBlocked``).  Nothing is downloaded,
installed or substituted, and the sample's own benchmark ``main`` is never executed.

This module is import-safe on a machine without CUDA.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESET_DIR = REPO / "tiresias" / "app_runners"
HPC_DIR = REPO / "energy_harness"
ADAPTERS_JSON = REPO / "tiresias" / "framework" / "adapters" / "adapters.json"

for _p in (HPC_DIR, RESET_DIR, HERE):  # HERE ends up first so these modules win any name clash
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cuda_build_target  # noqa: E402  (tiresias/app_runners, unchanged)

PINNED_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
REDUCTION_KERNEL_REL = "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu"
REDUCTION_UTIL_REL = "cpp/2_Concepts_and_Techniques/reduction/reduction.cpp"
REDUCTION_HEADER_REL = "cpp/2_Concepts_and_Techniques/reduction/reduction.h"
TRANSPOSE_REL = "cpp/6_Performance/transpose/transpose.cu"

# Hard-coded on purpose (second source of truth, cross-checked against adapters.json by
# check_manifest_consistency).  The reduction.cpp hash is the one the existing reduction runner pins; it is
# needed to build (isPow2 helper) but is not part of the manifest.
PINNED_SHA256 = {
    REDUCTION_KERNEL_REL: "c9ac9a9a424726522dd616fa24d58e6f0919ac95dd746a233d5fb857bffffde9",
    REDUCTION_UTIL_REL: "a416ae8aa45a9c521125ecb74343f426f1fa264479192076817e4ca536afbdba",
    TRANSPOSE_REL: "ba300b6b5f4dbc17ec72bfd3d1ee082508632ec431fe46ba54e6e3ce05292a95",
}

REGIME_ORDER = ("small", "medium", "large")
SEED = 20260914
RTOL = 1e-5   # same tolerances as reduction_runner.py
ATOL = 1e-6

REDUCTION_THREADS = 256
# The sample's own launch policy for whichKernel 6 caps the grid at its default --maxblocks (64); kernels 0 and 1
# use one element per thread.  Written from the sample's documented behaviour and NOT verifiable on the Mac:
# the authors confirms against the pinned reduction.cpp before the freeze commit (see HANDOFF.md).
REDUCE6_MAX_BLOCKS = 64

# parent -> runner name (the ``runner`` column of application_energy_raw.csv) and candidate -> (kernel, selector).
# The selector is the sample's whichKernel for the reduction and the sample's kernel-variant number for the
# tile-family driver (0 copy, 1 copySharedMem, 5 coarse-grained, 6 fine-grained).
PARENTS = {
    "alt_cuda_samples_reduction": {
        "runner": "alt_reduction", "module": "alt_reduction_runner",
        "kernels": {"c1": ("reduce0", 0), "c2": ("reduce1", 1), "c3": ("reduce6", 6)},
    },
    "alt_cuda_samples_copy": {
        "runner": "alt_copy", "module": "alt_copy_runner",
        "kernels": {"c1": ("copy", 0), "c2": ("copySharedMem", 1)},
    },
    "alt_cuda_samples_transposefine": {
        "runner": "alt_transposefine", "module": "alt_transposefine_runner",
        "kernels": {"c1": ("transposeFineGrained", 6), "c2": ("transposeCoarseGrained", 5)},
    },
}
RUNNER_MODULES = {v["runner"]: v["module"] for v in PARENTS.values()}
EXPECTED_CELLS = {"alt_cuda_samples_reduction": 9, "alt_cuda_samples_copy": 6, "alt_cuda_samples_transposefine": 6}


class RunnerBlocked(RuntimeError):
    """A required provenance/toolchain/runtime check failed; never silently fall back."""


class UnseenOperatorConfigError(ValueError):
    """The manifest has no such (parent, regime, candidate) cell.  The CLI reports exit code 4 for it."""


# ---------------------------------------------------------------------------- manifest
def load_manifest() -> dict:
    return json.loads(ADAPTERS_JSON.read_text())


def manifest_adapter(parent_id: str, manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    found = [a for a in manifest["adapters"] if a["parent_id"] == parent_id]
    if len(found) != 1:
        raise UnseenOperatorConfigError(f"adapters.json has {len(found)} entries for parent {parent_id!r}")
    return found[0]


def cell_config(parent_id: str, regime: str, candidate_id: str) -> dict:
    """Manifest-derived description of one cell; raises UnseenOperatorConfigError for a non-existent cell."""
    if parent_id not in PARENTS:
        raise UnseenOperatorConfigError(f"unknown parent {parent_id!r}")
    adapter = manifest_adapter(parent_id)
    if regime not in adapter["regimes"]:
        raise UnseenOperatorConfigError(f"{parent_id} has no regime {regime!r}")
    if candidate_id not in adapter["candidates"]:
        raise UnseenOperatorConfigError(f"{parent_id} has no candidate {candidate_id!r}")
    kernel_name, selector = PARENTS[parent_id]["kernels"][candidate_id]
    if adapter["candidates"][candidate_id]["kernel"] != kernel_name:
        raise UnseenOperatorConfigError(
            f"{parent_id}/{candidate_id}: manifest kernel {adapter['candidates'][candidate_id]['kernel']!r} "
            f"differs from this runner's table ({kernel_name!r})")
    return {"parent_id": parent_id, "runner": PARENTS[parent_id]["runner"], "regime": regime,
            "candidate_id": candidate_id, "n": int(adapter["regimes"][regime]["n"]),
            "kernel_name": kernel_name, "selector": selector, "family": adapter["family"],
            "manifest_threads_per_block": adapter.get("threads_per_block"),
            "source_file": adapter["source"]["file"], "source_sha256": adapter["source"]["sha256"]}


def enumerate_cells() -> list[dict]:
    """Every cell of the three parents, in manifest order: parent, regime small..large, candidate c1..."""
    manifest = load_manifest()
    out = []
    for adapter in manifest["adapters"]:
        pid = adapter["parent_id"]
        if pid not in PARENTS:
            continue
        for regime in REGIME_ORDER:
            for cand in sorted(adapter["candidates"]):
                out.append(cell_config(pid, regime, cand))
    return out


def check_manifest_consistency(manifest: dict | None = None) -> None:
    """Fail loudly if adapters.json and this runner's hard-coded tables disagree."""
    manifest = manifest or load_manifest()
    seen = set()
    for adapter in manifest["adapters"]:
        pid = adapter["parent_id"]
        if pid not in PARENTS:
            continue
        seen.add(pid)
        src = adapter["source"]
        if src["revision"] != PINNED_REVISION:
            raise RunnerBlocked(f"RUNNER_BLOCKED: {pid}: manifest revision {src['revision']} != {PINNED_REVISION}")
        if PINNED_SHA256.get(src["file"]) != src["sha256"]:
            raise RunnerBlocked(f"RUNNER_BLOCKED: {pid}: manifest sha256 for {src['file']} differs from the runner's pin")
        cands = {c: v["kernel"] for c, v in adapter["candidates"].items()}
        if cands != {c: k[0] for c, k in PARENTS[pid]["kernels"].items()}:
            raise RunnerBlocked(f"RUNNER_BLOCKED: {pid}: manifest candidates {cands} differ from the runner's kernel table")
        if set(adapter["regimes"]) != set(REGIME_ORDER):
            raise RunnerBlocked(f"RUNNER_BLOCKED: {pid}: unexpected regimes {sorted(adapter['regimes'])}")
    if seen != set(PARENTS):
        raise RunnerBlocked(f"RUNNER_BLOCKED: manifest is missing parents {sorted(set(PARENTS) - seen)}")


# ---------------------------------------------------------------------------- geometry and bytes
def reduction_blocks(which_kernel: int, n: int) -> int:
    """Grid size launched for the reduction candidates (1D grid of REDUCTION_THREADS-thread blocks)."""
    if n <= 0 or n % REDUCTION_THREADS != 0:
        raise UnseenOperatorConfigError(f"reduction size {n} must be a positive multiple of {REDUCTION_THREADS}")
    if which_kernel in (0, 1):     # one element per thread
        return n // REDUCTION_THREADS
    if which_kernel == 6:          # two elements per thread per grid-stride step, grid capped at the sample default
        return min(REDUCE6_MAX_BLOCKS, math.ceil(n / (2 * REDUCTION_THREADS)))
    raise UnseenOperatorConfigError(f"whichKernel {which_kernel} is not an admitted unseen-operator reduction kernel")


def reduction_bytes(n: int, blocks: int) -> int:
    """freeze.md section 4: input read once (4n) plus one float32 partial sum written per block."""
    return 4 * n + 4 * blocks


def tile_bytes(dim_x: int, dim_y: int) -> int:
    """Copy and (partial) transpose: every element read once and written once, float32."""
    return 2 * 4 * dim_x * dim_y


# ---------------------------------------------------------------------------- source verification
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_revision(source_root: Path) -> str:
    try:
        revision = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerBlocked("RUNNER_BLOCKED: source_root is not a readable Git checkout") from exc
    if revision != PINNED_REVISION:
        raise RunnerBlocked(f"RUNNER_BLOCKED: expected revision {PINNED_REVISION}, got {revision}")
    return revision


def check_hashes(source_root: Path, rels: tuple[str, ...]) -> dict[str, str]:
    """SHA-256 of every named file must equal the pin; missing or modified files refuse the run."""
    actual = {}
    for rel in rels:
        path = source_root / rel
        if not path.is_file():
            raise RunnerBlocked(f"RUNNER_BLOCKED: missing pinned source {path}")
        digest = sha256_file(path)
        if digest != PINNED_SHA256[rel]:
            raise RunnerBlocked(f"RUNNER_BLOCKED: SHA-256 mismatch for {rel}: expected {PINNED_SHA256[rel]}, got {digest}")
        actual[rel] = digest
    return actual


def verify_source(source_root: Path, rels: tuple[str, ...]) -> tuple[Path, str, dict[str, str]]:
    """Revision and byte hashes of the exact pinned files, before anything is compiled."""
    source_root = Path(source_root).expanduser().resolve()
    for rel in rels:
        if not (source_root / rel).is_file():
            raise RunnerBlocked(f"RUNNER_BLOCKED: missing pinned source {source_root / rel}")
    revision = check_revision(source_root)
    return source_root, revision, check_hashes(source_root, rels)


_FOUR_ARG_SIGNATURE = (r"__global__\s+void\s+{name}\s*\(\s*float\s*\*\s*\w+\s*,\s*float\s*\*\s*\w+\s*,"
                       r"\s*int\s+\w+\s*,\s*int\s+\w+\s*\)")


def read_source_text(root: Path, rel: str) -> str:
    return (Path(root) / rel).read_text(encoding="utf-8", errors="replace")


def check_four_arg_kernels(source_text: str, names: tuple[str, ...]) -> None:
    """The tile-family driver declares each kernel as (float*, float*, int, int).  The hash already pins the bytes,
    so this only turns a would-be link error (or a wrong-signature launch) into a clear message."""
    for name in names:
        if not re.search(_FOUR_ARG_SIGNATURE.format(name=re.escape(name)), source_text):
            raise RunnerBlocked(
                f"RUNNER_BLOCKED: pinned source has no `__global__ void {name}(float*, float*, int, int)`; the "
                "driver's prototype would not link. Do not edit the driver to fit: report this as a manifest revision.")


def check_reduce_kernels(source_text: str, names: tuple[str, ...]) -> None:
    for name in names:
        if not re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(", source_text):
            raise RunnerBlocked(f"RUNNER_BLOCKED: pinned reduction source has no `__global__ void {name}(`")


# ---------------------------------------------------------------------------- toolchain
def build_arch() -> str:
    return cuda_build_target.target_arch(RunnerBlocked)


def find_nvcc() -> str:
    """nvcc plus the live-GPU arch check (cuda_build_target); raises RunnerBlocked, never falls back."""
    nvcc = cuda_build_target.find_nvcc(RunnerBlocked)
    cuda_build_target.assert_live_arch(build_arch(), RunnerBlocked)
    return nvcc


def run_nvcc_steps(steps: list[list[str]], workdir: Path, binary: Path) -> Path:
    for step in steps:
        proc = subprocess.run(step, capture_output=True, text=True, cwd=str(workdir))
        if proc.returncode != 0:
            raise RunnerBlocked(f"RUNNER_BLOCKED: nvcc failed: {' '.join(step)}\n{proc.stderr[-2000:]}")
    if not binary.is_file():
        raise RunnerBlocked("RUNNER_BLOCKED: harness binary missing after build")
    return binary


def import_binary_energy_context():
    """energy_harness/application_energy_harness.BinaryEnergyContext (imports measurement_runner; no GPU touched)."""
    from application_energy_harness import BinaryEnergyContext
    return BinaryEnergyContext
