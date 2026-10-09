#!/usr/bin/env python3
"""Runner for the unseen-operator parent ``alt_cuda_samples_transposefine``: pinned CUDA Samples
``transposeFineGrained`` and ``transposeCoarseGrained``.

Candidates c1 = transposeFineGrained, c2 = transposeCoarseGrained, both kernels of the pinned
``cpp/6_Performance/transpose/transpose.cu``; regimes n x n float32 with n = 1024 / 2048 / 8192.  All the mechanics
live in ``tile_family.py``.

Oracle: each kernel's own partial permutation (default, chosen by the freeze owner after the Blackwell correctness
run, CHANGELOG M411).  The pinned kernels are PARTIAL transposes (fine-grained: transpose inside each tile, tile left
in place; coarse-grained: tile moved, no transpose inside it), so a full transpose cannot pass; it remains available
as a diagnostic via UNSEEN_TRANSPOSEFINE_ORACLE=full_transpose.  The oracle in force is written into every result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tile_family as tf
import unseen_common as uc
from unseen_common import RunnerBlocked, UnseenOperatorConfigError  # noqa: F401  (RunnerBlocked is looked up by the CLI)

PARENT = tf.TRANSPOSEFINE_PARENT
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
    for regime, side in (("small", 1024), ("medium", 2048), ("large", 8192)):
        for cand in ("c1", "c2"):
            plan = cell_plan(regime, cand)
            assert (plan["dim_x"], plan["dim_y"]) == (side, side) and plan["elements"] == side * side
            assert plan["blocks"] == (side // 32) ** 2 and plan["bytes_model"] == 2 * 4 * side * side
            assert plan["oracle"] == "kernel_semantics" and plan["threads_per_block"] == 512
    assert cell_plan("small", "c1")["variant"] == 6 and cell_plan("small", "c2")["variant"] == 5
    for bad in (("small", "c3"), ("tiny", "c1")):
        try:
            cell_plan(*bad)
        except UnseenOperatorConfigError:
            pass
        else:
            raise AssertionError(f"non-existent cell accepted: {bad}")
    n = 64
    x = tf.transpose_input(n * n)
    full = type(x)("f", [x[c * n + r] for r in range(n) for c in range(n)])
    assert tf.matches_full_transpose(x, full, n) and not tf.matches_full_transpose(x, x, n)
    assert not tf.matches_fine_grained_partial(x, full, n) and not tf.matches_coarse_grained_partial(x, full, n)
    print("CPU_ONLY_ALT_TRANSPOSEFINE_RUNNER_OK: manifest/geometry/oracle checks passed")


def main() -> int:
    return tf.run_main(PARENT, "Correctness-only check of the pinned transposeFineGrained and transposeCoarseGrained "
                               "candidates (one single-shot launch per cell; no energy window).", self_test)


if __name__ == "__main__":
    sys.exit(main())
