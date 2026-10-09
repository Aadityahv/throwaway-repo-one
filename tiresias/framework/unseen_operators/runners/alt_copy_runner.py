#!/usr/bin/env python3
"""Runner for the unseen-operator parent ``alt_cuda_samples_copy``: pinned CUDA Samples ``copy`` and ``copySharedMem``.

Candidates c1 = copy (direct global copy), c2 = copySharedMem (shared-memory staged copy), both kernels of the pinned
``cpp/6_Performance/transpose/transpose.cu`` (not the vectorAdd kernel the existing copy runner uses).  Regimes:
n = 1,048,576 / 8,388,608 / 67,108,864 float32 elements.  All the mechanics live in ``tile_family.py``; see its
docstring for the build model, the fixed (32, 16) = 512-thread block, the NaN output sentinel and the oracle.

Oracle (freeze): bitwise identity, out == in.  A kernel that does not write every element fails it and the cell is
excluded under freeze section 2; the correctness-only mode reports the fraction written so the reason is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tile_family as tf
import unseen_common as uc
from unseen_common import RunnerBlocked, UnseenOperatorConfigError  # noqa: F401  (RunnerBlocked is looked up by the CLI)

PARENT = tf.COPY_PARENT
RUNNER = uc.PARENTS[PARENT]["runner"]


def cell_plan(regime: str, candidate_id: str) -> dict:
    return tf.cell_plan(PARENT, regime, candidate_id)


def prepare_binary_energy_context(source_root: Path, regime: str, candidate_id: str, workdir: Path):
    return tf.prepare_binary_energy_context(PARENT, source_root, regime, candidate_id, workdir)


def run_candidate(source_root: Path, regime: str, candidate_id: str, workdir: Path) -> dict:
    return tf.run_candidate(PARENT, source_root, regime, candidate_id, workdir)


def self_test() -> None:
    """CPU-only; never invokes nvcc or touches a GPU."""
    uc.check_manifest_consistency()
    cells = [c for c in uc.enumerate_cells() if c["parent_id"] == PARENT]
    assert len(cells) == uc.EXPECTED_CELLS[PARENT] == 6
    expected = {"small": (1024, 1024), "medium": (2048, 2048), "large": (8192, 8192)}
    for regime, dims in expected.items():
        for cand in ("c1", "c2"):
            plan = cell_plan(regime, cand)
            assert (plan["dim_x"], plan["dim_y"]) == dims and plan["elements"] == plan["n"]
            assert plan["blocks"] == plan["n"] // (tf.TILE_DIM * tf.TILE_DIM)
            assert plan["bytes_model"] == 2 * 4 * plan["n"] and plan["oracle"] == "identity"
            assert plan["threads_per_block"] == 512
    for bad in (("small", "c3"), ("tiny", "c1")):
        try:
            cell_plan(*bad)
        except UnseenOperatorConfigError:
            pass
        else:
            raise AssertionError(f"non-existent cell accepted: {bad}")
    x = tf.copy_input(256)
    assert list(x) == list(tf.copy_input(256))
    assert tf.matches_identity(x, x)
    y = type(x)("f", x)
    y[7] += 1.0
    assert not tf.matches_identity(x, y)
    print("CPU_ONLY_ALT_COPY_RUNNER_OK: manifest/geometry/oracle checks passed")


def main() -> int:
    return tf.run_main(PARENT, "Correctness-only check of the pinned copy and copySharedMem candidates "
                               "(one single-shot launch per cell; no energy window).", self_test)


if __name__ == "__main__":
    sys.exit(main())
