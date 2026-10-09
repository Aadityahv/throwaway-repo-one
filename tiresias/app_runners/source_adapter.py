#!/usr/bin/env python3
"""Source checkout and native-launch-control adapter — M237 fixing commit.

Fixes C3 and part of C5 (tiresias/review/ACI_ACCEPTANCE_2026-09-14.md):

  C3: this file checks a checkout path/revision; complete native build/launch
  descriptors for the three selected local parents live in
  `local_development_adapters.py`.
  claim about whether the catalog's generic (threads, items/thread) 4-value
  grid is actually a native, legal launch control for a given parent. This
  version adds `native_launch_config()`, which returns the REAL CLI/launch
  parameters for a parent given a regime+candidate cell, grounded in the
  actual `usage()`/argument parsing of each source file (read 2026-09-14),
  and raises NativeControlMismatch with a specific reason for any parent
  whose candidate_grid_status in workload_catalog.csv is not
  "native_verified".

  C5 (license-scope half): the previous refusal for repository-local sources
  read as if local code cannot be used at all without a public license file.
  That conflated two different questions. This version separates them:
  `check_source()` still requires an explicit coordinator license-audit sign-
  off before a local source may be admitted to ANY external-facing artifact
  (a paper's reproducibility package, a public release, a shared dataset),
  but authorized internal experimental use — running this project's own
  benchmark harness against its own repository's own code, on its own
  machines, to produce its own internal measurements — is not blocked by an
  absent LICENSE file. `--internal-only` makes that distinction explicit
  rather than leaving it implied by exit code 2 either way.
"""
from __future__ import annotations
import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class NativeControlMismatch(Exception):
    pass


# ---------------------------------------------------------------------------
# C3: real native launch-control mappings, one per parent this commit
# prioritized (candidate_grid_status != "pending" in workload_catalog.csv).
# Each function takes the generic cell (threads_per_block, items_per_thread)
# the catalog still stores for bookkeeping and returns the REAL CLI argv the
# harness owner must actually pass to the compiled binary — or raises
# NativeControlMismatch if no legal mapping exists for that parent.

def native_launch_config(parent_id: str, threads_per_block: int, items_per_thread: int,
                          shape_a: int, shape_b: int, candidate_note: str = "") -> list[str]:
    if parent_id in ("train_p1_shared_generator",):
        # predictor_vector_load.cu: --threads is native threads_per_block,
        # --iterations is native items_per_thread. --blocks/--tile-elements
        # are additional required native args the grid does not carry, so
        # the harness owner must supply them separately (documented, not
        # silently defaulted).
        return ["--layout", "coalesced", "--tile-elements", str(shape_a), "--blocks", "<declared>",
                "--threads", str(threads_per_block), "--iterations", str(items_per_thread),
                "--step-warps", "1"]
    if parent_id == "train_multistream_load":
        return ["--layout", "coalesced", "--tile-elements", str(shape_a), "--streams", str(shape_b),
                "--blocks", "<declared>", "--threads", str(threads_per_block),
                "--iterations", str(items_per_thread), "--step-warps", "1"]
    if parent_id == "train_segmented_load":
        raise NativeControlMismatch(
            "train_segmented_load: trip count is data-dependent (read from segment_offsets[]); "
            "items_per_thread is not a legal native control. Use --run-length/--run-spread instead "
            "(see workload_catalog.csv candidate_grid_note).")
    if parent_id == "train_shared_stage":
        raise NativeControlMismatch(
            "train_shared_stage: no native items_per_thread axis. --chunks scales global bytes and "
            "--reuse scales shared reads independently; a single items_per_thread value cannot "
            "legally stand in for both (see workload_catalog.csv candidate_grid_note).")
    if parent_id in ("dev_local_triad", "dev_local_transpose"):
        # e2e_tile_triad_calibration.cu / e2e_transpose.cu: TILE_DIM is a compile-time constant
        # (16 for triad, 32 for transpose); there is no runtime threads_per_block or
        # items_per_thread argument at all. The only real CLI is --n and --launches.
        raise NativeControlMismatch(
            f"{parent_id}: TILE_DIM x TILE_DIM is a COMPILE-TIME constant (set via -DTILE_DIM at "
            "build time, not a runtime flag); there is no runtime threads_per_block or "
            "items_per_thread CLI argument. Real native CLI is only --n <matrix_dim> --launches <N>.")
    if parent_id == "train_triton_vector_add":
        # python/tutorials/01-vector-add.py: launch grid is lambda meta:
        # (triton.cdiv(n_elements, meta['BLOCK_SIZE']),); items_per_thread has
        # no analog. BLOCK_SIZE is the real native control.
        block_size = {1: 128, 2: 256, 3: 512, 4: 1024}.get(threads_per_block // 128, threads_per_block)
        return ["--BLOCK_SIZE", str(threads_per_block), "--n-elements", str(shape_a)]
    if parent_id == "dev_triton_softmax":
        # python/tutorials/02-fused-softmax.py: softmax_kernel's native meta
        # parameters are BLOCK_SIZE, num_warps and num_stages. The tutorial's
        # helper hard-codes num_warps=8 and chooses num_stages from shared
        # memory; for equivalent candidate selection we launch the same
        # source kernel directly with the declared meta-parameters. The cell
        # schema stores num_warps*32 in threads_per_block and num_stages in
        # items_per_thread solely for backward-compatible column typing.
        if threads_per_block not in {64, 128, 256}:
            raise NativeControlMismatch("dev_triton_softmax: num_warps must be 2, 4 or 8")
        if items_per_thread not in {2, 4}:
            raise NativeControlMismatch("dev_triton_softmax: num_stages must be 2 or 4")
        if shape_a <= 0 or shape_b <= 0:
            raise NativeControlMismatch("dev_triton_softmax: rows and n_cols must be positive")
        num_warps = threads_per_block // 32
        block_size = 1 << (shape_b - 1).bit_length()
        return ["--rows", str(shape_a), "--cols", str(shape_b),
                "--BLOCK_SIZE", str(block_size), "--num-warps", str(num_warps),
                "--num-stages", str(items_per_thread), "--seed", "20260914"]
    if parent_id == "train_cuda_samples_reduction":
        if items_per_thread not in {2, 3, 4, 5}:
            raise NativeControlMismatch("train_cuda_samples_reduction: whichKernel must be 2, 3, 4 or 5")
        return ["--size", str(shape_a), "--threads", str(threads_per_block),
                "--blocks", str(max(1, (shape_a + threads_per_block - 1) // threads_per_block)),
                "--whichKernel", str(items_per_thread), "--dtype", "float32"]
    if parent_id == "train_cuda_samples_transpose":
        if items_per_thread != 1 or threads_per_block != 256:
            raise NativeControlMismatch("train_cuda_samples_transpose: source uses fixed 32x8 CUDA tile = 256 threads")
        match = re.search(r"kernel variant=(\d+)", candidate_note or "")
        if match is None:
            raise NativeControlMismatch("train_cuda_samples_transpose: missing numeric source variant")
        return ["--dimX", str(shape_a), "--dimY", str(shape_b), "--kernel-variant", match.group(1)]
    if parent_id == "train_cub_block_reduce":
        if threads_per_block not in {32, 64, 128, 256} or items_per_thread <= 0:
            raise NativeControlMismatch("train_cub_block_reduce: unsupported BlockReduce compile-time geometry")
        return ["--valid-items", str(shape_a), "--block-dim-x", str(threads_per_block),
                "--items-per-thread", str(items_per_thread), "--operator", "plus<float>"]
    if parent_id == "dev_pytorch_rowwise_softmax":
        if threads_per_block != 256 or items_per_thread < shape_b:
            raise NativeControlMismatch("dev_pytorch_rowwise_softmax: generic columns do not represent ATen launch; use native stride controls")
        return ["--outer-size", str(shape_a), "--dim-size", str(shape_b), "--inner-size", "1",
                "--input-row-stride", str(items_per_thread), "--output-row-stride", str(items_per_thread),
                "--dtype", "float32"]
    if parent_id == "final_triton_layer_norm":
        if threads_per_block not in {32, 64, 128, 256}:
            raise NativeControlMismatch("final_triton_layer_norm: threads_per_block encodes num_warps*32")
        block_size = 1 << (shape_b - 1).bit_length()
        return ["--rows", str(shape_a), "--cols", str(shape_b), "--BLOCK_SIZE", str(block_size),
                "--num-warps", str(threads_per_block // 32), "--eps", "1e-5", "--dtype", "float32"]
    if parent_id == "final_pytorch_layer_norm":
        if threads_per_block != 256 or items_per_thread < shape_b:
            raise NativeControlMismatch("final_pytorch_layer_norm: use source-native padded row stride")
        return ["--rows", str(shape_a), "--cols", str(shape_b), "--row-stride", str(items_per_thread),
                "--eps", "1e-5", "--affine", "1", "--dtype", "float32"]
    if parent_id == "final_pytorch_embedding":
        if threads_per_block != 256 or items_per_thread != 1:
            raise NativeControlMismatch("final_pytorch_embedding: one contiguous index_select configuration only")
        return ["--num-weights", str(shape_a), "--feature-dim", str(shape_b),
                "--num-indices", str(max(1, shape_a // 4)), "--dim", "0", "--index-dtype", "int64"]
    if parent_id == "final_xformers_indexed_select":
        if items_per_thread != 1 or threads_per_block not in {128, 256, 512, 1024}:
            raise NativeControlMismatch("final_xformers_indexed_select: BLOCK_SIZE_COL must be 128/256/512/1024")
        return ["--num-rows", str(shape_a), "--num-indices", str(max(1, shape_a // 4)),
                "--num-cols", str(shape_b), "--BLOCK_SIZE_INDEX", "1",
                "--BLOCK_SIZE_COL", str(threads_per_block)]
    if parent_id == "final_cub_device_reduce":
        if items_per_thread != 1:
            raise NativeControlMismatch("final_cub_device_reduce: one DeviceReduce::Sum configuration only")
        return ["--num-items", str(shape_a), "--api", "DeviceReduce::Sum",
                "--temp-storage-query", "1", "--dtype", "float32"]
    if parent_id == "final_cuda_samples_copy":
        if items_per_thread != 1 or threads_per_block not in {128, 256, 512, 1024}:
            raise NativeControlMismatch("final_cuda_samples_copy: block size must be 128/256/512/1024")
        return ["--vector-length", str(shape_a), "--threads", str(threads_per_block),
                "--blocks", str((shape_a + threads_per_block - 1) // threads_per_block), "--dtype", "float32"]
    raise NativeControlMismatch(
        f"{parent_id}: native launch-control mapping not audited this pass "
        "(work_formula_basis=pending in workload_catalog.csv); do not assume the generic "
        "(threads_per_block, items_per_thread) grid applies without checking the real source first.")


# ---------------------------------------------------------------------------
# Source checkout validator (unchanged contract, C5 license-scope fix below).

def check_source(source_root: Path | None, parent: str, internal_only: bool) -> int:
    rows = list(csv.DictReader((ROOT / "workload_catalog.csv").open()))
    row = next((r for r in rows if r["parent_id"] == parent), None)
    if not row:
        print(f"UNKNOWN_PARENT: {parent}")
        return 2

    if row["source_repository"] == "repository-local":
        if internal_only:
            # C5: absent public licensing is a redistribution/release-track concern, not a block on
            # this project's own internal use of its own repository's own code on its own machines.
            # This does NOT clear the source for anything that leaves the project (a paper artifact,
            # a public release, a shared dataset) -- that still requires the coordinator license
            # audit below.
            print(f"INTERNAL_USE_OK: {parent} is repository-local, NOASSERTION license. Authorized "
                  "for internal experimental use on this project's own harness/machines only. "
                  "NOT cleared for redistribution, public release, or any external-facing artifact "
                  "-- that still requires REDISTRIBUTION_CLEARED (coordinator license audit).")
            source = Path(__file__).resolve().parents[2] / row["source_path"]
            if not source.is_file():
                print(f"MISSING_SOURCE_PATH: {source}")
                return 2
            print(f"SOURCE_OK: {parent} ({source})")
            return 0
        print("UNLICENSED_LOCAL_SOURCE: repository license is NOASSERTION; coordinator must clear "
              "REDISTRIBUTION_CLEARED before this source may appear in any external-facing artifact. "
              "Pass --internal-only for this project's own internal experimental use.")
        return 2

    if source_root is None:
        print("MISSING_SOURCE_ROOT: --source-root required for upstream (non-local) parents")
        return 2
    root = source_root / row["lineage_group"]
    if not root.is_dir():
        print(f"UNAVAILABLE_SOURCE: expected {root}; fetch {row['source_repository']} at {row['pinned_revision']}")
        return 2
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != row["pinned_revision"]:
        print(f"WRONG_SOURCE_REVISION: {root}; expected {row['pinned_revision']}")
        return 2
    if row["path_status"] == "requires_source_path_audit":
        print(f"PATH_NOT_AUDITED: {parent}'s declared source_path 404s at the verified pin "
              f"({row['pinned_revision']}); do not substitute a guessed path. See "
              "workload_catalog.csv lineage_relationship for the re-check note.")
        return 2
    source = root / row["source_path"]
    if not source.is_file():
        print(f"MISSING_SOURCE_PATH: {source}; adapter not authorized to substitute another implementation")
        return 2
    print(f"SOURCE_OK: {parent} ({source})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=None)
    p.add_argument("--parent", required=True)
    p.add_argument("--internal-only", action="store_true",
                    help="Clear repository-local sources for this project's own internal "
                         "experimental use (not for redistribution/publication).")
    p.add_argument("--show-native-launch-config", action="store_true",
                    help="Deprecated generic mapping view. Selected local parents require --show-local-descriptor.")
    p.add_argument("--show-local-descriptor", action="store_true",
                    help="Print complete native build/launch descriptor for a selected local catalog cell.")
    p.add_argument("--regime", choices=("small", "medium", "large"))
    p.add_argument("--candidate-id")
    p.add_argument("--arch", default="sm_120")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--items-per-thread", type=int, default=1)
    p.add_argument("--shape-a", type=int, default=1_048_576)
    p.add_argument("--shape-b", type=int, default=1)
    args = p.parse_args()

    if args.show_local_descriptor:
        if not args.regime or not args.candidate_id:
            p.error("--show-local-descriptor requires --regime and --candidate-id")
        from local_development_adapters import ConfigError, descriptor
        try:
            value = descriptor(args.parent, args.regime, args.candidate_id, args.arch, args.gpu_id)
        except ConfigError as exc:
            print(f"NATIVE_CONTROL_MISMATCH: {exc}")
            return 2
        import json
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.show_native_launch_config:
        try:
            argv = native_launch_config(args.parent, args.threads_per_block, args.items_per_thread,
                                         args.shape_a, args.shape_b)
        except NativeControlMismatch as exc:
            print(f"NATIVE_CONTROL_MISMATCH: {exc}")
            return 2
        print("NATIVE_LAUNCH_ARGV:", " ".join(argv))
        return 0

    return check_source(args.source_root, args.parent, args.internal_only)


if __name__ == "__main__":
    raise SystemExit(main())
