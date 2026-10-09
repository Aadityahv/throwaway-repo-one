#!/usr/bin/env python3
"""REMOTE. Extract Alavani-style SASS instruction-mix features (compile-only, never launched)
for the reset catalog's operator families, on Blackwell (sm_120) or Ada (sm_89).

Standalone by design (stdlib plus the target env's own Triton; no numpy, no repo checkout):
this must run from a tiny staging directory on Ada, whose disk is ~100% full (see STAGING.md).
Everything it needs -- the classifier (`sass_mix_classifier.py`, staged alongside), per-cell
compile geometry (`--cells-manifest`, a small JSON built locally by `build_cell_manifest.py` from
`workload_cells.csv`), and every other path -- is a CLI argument. All output (including the
Triton cache) is written under `--out-dir`.

Per family:
  - Triton (`train_triton_vector_add`/`dev_triton_softmax`/`final_triton_layer_norm`): the pinned
    `@triton.jit` kernel is extracted from `--triton-source-root` exactly as
    `tiresias/app_runners/*_runner.py`'s `extract_pinned_kernel_source()`/`import_pinned_kernel()`
    do (AST-extract the one decorated function, exec it in an isolated namespace -- the tutorial
    module itself is never imported/run), then compiled WITHOUT LAUNCHING via
    `kernel.warmup(*args, grid=grid, **kwargs)` with pointer arguments passed as their torch dtype
    (Triton's `MockTensor.wrap_dtype`, applied internally by `JITFunction.warmup`, builds a
    `MockTensor` with no GPU allocation) and every scalar argument passed at its REAL value for
    that (regime, candidate) cell, exactly matching the positional argument list, grid, constexprs
    and `num_warps`/`num_stages` the matching `tiresias/app_runners/*_runner.py` actually launches
    with.
    **This replaced an earlier `triton.compile(ASTSource(...))` path** (still used by
    `measured/flipflop/emit_ptx.py`) after the 2026-09-28 Blackwell run showed it reproduces
    the committed CUDA Samples features exactly but NOT the Triton ones: `ASTSource` with no
    `attrs` skips the argument specialization (`tt.divisibility=16` on aligned pointers/integers
    divisible by 16, equal-to-1) a real launch applies, so unaligned/unvectorized code was
    compiled for vector-add and softmax, undercounting or overcounting instructions relative to
    the September real-launch dumps (see STAGING.md's "2026-09-28 fix" section for the exact
    before/after numbers). `warmup()` needs a live CUDA device (the target arch is inferred from
    whichever GPU `CUDA_VISIBLE_DEVICES` selects, exactly as a real launch infers it -- there is no
    explicit `target=` any more); `check_device_arch()` verifies that live device's compute
    capability matches `--arch` before any Triton family is compiled, refusing loudly on mismatch.
    The compiled artifact's `asm["cubin"]` bytes (not `asm["ptx"]`: this script needs real SASS,
    which only `cuobjdump --dump-sass` on a cubin gives) are written to a temp `.cubin` file under
    `--out-dir`.
  - CUDA Samples (`final_cuda_samples_copy`/`train_cuda_samples_reduction`/
    `train_cuda_samples_transpose`): the pinned `.cu` translation unit is compiled directly with
    `nvcc -cubin -arch=<arch>` (plus `-I` include paths from the manifest) -- no execution, no
    linking, no need for the source's own `main()` (transpose.cu/vectorAdd.cu have one;
    reduction_kernel.cu does not -- `-cubin` only ever emits device code either way, so this is
    unaffected by the `flipflop_predict.py` link-step issue that made `-c` [object] necessary
    there).

Either way, `cuobjdump --dump-sass <cubin>` is then run on the resulting cubin, and
`sass_mix_classifier.isolate_function_section()` picks out exactly the one function section named
by the manifest's `entry_substring` (refusing loudly, listing every function name seen, on zero or
several matches -- e.g. reduction's `reduce4`/`reduce5` are templated on blockSize too, so a bare
"reduce4" substring is always ambiguous; the manifest's anchor is
"reduce4IfLj{threads}E", not the bare name).

Refuses (`EmitBlocked`) if `CUDA_VISIBLE_DEVICES` is not exactly `--expect-visible-devices`, per
`AGENTS.md`'s shared-machine rule -- even though nothing here launches a kernel, both nvcc and a
live Triton install can touch a CUDA context / query the default device, so the same guard other
project scripts use is applied here too, rather than assuming compile-only means device-safe.

Output: `sass_features_<arch>.json` (same shape as the committed `sass_features_blackwell.json`:
parent_id -> "regime/candidate_id" -> 10 features) and `manifest_<arch>.json` (source revision +
sha256 per family, compiler versions, kernel names, cubin sha256 per cell).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sass_mix_classifier as smc  # noqa: E402


class EmitBlocked(RuntimeError):
    """A required provenance/runtime/API-shape/device check failed; never silently fall back."""


TRITON_KERNEL_SIGNATURES = {
    "add_kernel": {
        "arg_names": ["x_ptr", "y_ptr", "output_ptr", "n_elements", "BLOCK_SIZE"],
        "signature_types": {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        "constexpr_names": ["BLOCK_SIZE"],
    },
    "softmax_kernel": {
        "arg_names": ["output_ptr", "input_ptr", "input_row_stride", "output_row_stride",
                       "n_rows", "n_cols", "BLOCK_SIZE", "num_stages"],
        "signature_types": {"output_ptr": "*fp32", "input_ptr": "*fp32", "input_row_stride": "i32",
                             "output_row_stride": "i32", "n_rows": "i32", "n_cols": "i32"},
        "constexpr_names": ["BLOCK_SIZE", "num_stages"],
    },
    "_layer_norm_fwd_fused": {
        "arg_names": ["X", "Y", "W", "B", "Mean", "Rstd", "stride", "N", "eps", "BLOCK_SIZE"],
        "signature_types": {"X": "*fp32", "Y": "*fp32", "W": "*fp32", "B": "*fp32", "Mean": "*fp32",
                             "Rstd": "*fp32", "stride": "i32", "N": "i32", "eps": "fp32"},
        "constexpr_names": ["BLOCK_SIZE"],
    },
}

ARCH_CC = {"sm_120": 120, "sm_89": 89}

# Kernel function name (used for both the AST extraction inside the pinned tutorial file and the
# TRITON_KERNEL_SIGNATURES lookup) per parent_id -- redundant with the manifest's own
# "kernel_name" field but kept here too as a hardcoded cross-check (refuses on mismatch below).
TRITON_KERNEL_NAME = {
    "train_triton_vector_add": "add_kernel",
    "dev_triton_softmax": "softmax_kernel",
    "final_triton_layer_norm": "_layer_norm_fwd_fused",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_device_arch(expected_arch: str) -> dict:
    """Query (never launch anything on) the live CUDA device selected by CUDA_VISIBLE_DEVICES and
    refuse if its compute capability does not match `--arch`. The warmup()-based Triton compile
    path has no explicit `target=` any more (see `compile_triton_cubin`'s docstring) -- it infers
    the target arch from whichever device is live, exactly as a real launch would -- so this is
    the only place left that guards against compiling for the wrong architecture."""
    try:
        import torch
    except ImportError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: PyTorch unavailable (needed to verify the live device "
                           f"arch before any Triton warmup compile): {exc}") from exc
    if not torch.cuda.is_available():
        raise EmitBlocked("EMIT_BLOCKED: torch.cuda.is_available() is False -- the warmup()-based "
                           "Triton compile path needs a live CUDA device to infer its target arch from")
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    expected_cc = ARCH_CC[expected_arch]
    if cc != expected_cc:
        raise EmitBlocked(
            f"EMIT_BLOCKED: live device compute capability sm_{cc} does not match --arch "
            f"{expected_arch!r} (expected sm_{expected_cc}); refusing rather than let Triton "
            f"silently compile for whatever device is actually visible."
        )
    return {"device_name": torch.cuda.get_device_name(0), "compute_capability": f"{major}.{minor}"}


def check_visible_devices(expected: str) -> None:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if actual != expected:
        raise EmitBlocked(
            f"EMIT_BLOCKED: CUDA_VISIBLE_DEVICES must be exactly {expected!r}, got {actual!r}. "
            f"Refusing per AGENTS.md's shared-machine rule (never touch a GPU this project did "
            f"not explicitly target)."
        )


def dump_sass(cuobjdump: str, cubin_path: Path) -> str:
    result = subprocess.run([cuobjdump, "--dump-sass", str(cubin_path)],
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise EmitBlocked(f"EMIT_BLOCKED: cuobjdump --dump-sass failed on {cubin_path}: {result.stderr[-4000:]}")
    return result.stdout


def cuobjdump_version(cuobjdump: str) -> str:
    result = subprocess.run([cuobjdump, "--version"], capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def nvcc_version(nvcc: str) -> str:
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


# --- Triton path -----------------------------------------------------------------------------

def import_pinned_triton_kernel(source_file: Path, kernel_name: str):
    """AST-extract exactly one decorated kernel function and exec it in an isolated namespace,
    mirroring `tiresias/app_runners/*_runner.py`'s `extract_pinned_kernel_source()`/
    `import_pinned_kernel()` -- the tutorial module itself is never imported or run."""
    import ast
    import linecache

    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == kernel_name]
    if len(matches) != 1:
        raise EmitBlocked(f"EMIT_BLOCKED: expected exactly one {kernel_name}, found {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    end = node.end_lineno
    kernel_source = "".join(lines[start:end])
    if not kernel_source or not node.decorator_list:
        raise EmitBlocked("EMIT_BLOCKED: extracted kernel is empty or undecorated")

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: Triton runtime unavailable: {exc}") from exc

    filename = str(source_file) + f":{kernel_name}_extracted"
    linecache.cache[filename] = (len(kernel_source), None, kernel_source.splitlines(keepends=True), filename)
    namespace = {"triton": triton, "tl": tl, "__name__": "_pinned_kernel_only"}
    try:
        exec(compile(kernel_source, filename, "exec"), namespace, namespace)
    except Exception as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: extracted kernel compilation failed: {exc}") from exc
    kernel = namespace.get(kernel_name)
    if kernel is None:
        raise EmitBlocked(f"EMIT_BLOCKED: extracted source has no {kernel_name} symbol")
    return kernel


def _triton_positional_args(kernel_name: str, cell: dict, dtype_marker) -> list:
    """Build the exact positional argument list `kernel.warmup(...)` needs to reproduce a real
    launch's specialization -- pointer arguments become `dtype_marker` (a torch dtype, which
    `JITFunction.warmup`'s internal `map(MockTensor.wrap_dtype, args)` turns into a MockTensor
    with the SAME pointer-alignment/divisibility specialization a real tensor argument would get;
    non-dtype values pass through `MockTensor.wrap_dtype` unchanged), and scalar arguments use the
    REAL value the matching `*_runner.py` launch passes for this cell (read from the manifest,
    which `build_cell_manifest.py` populated from `workload_cells.csv`'s own shapes) -- e.g.
    `tt.divisibility=16` on an integer argument depends on its real value being divisible by 16,
    which a placeholder value would get wrong.

    Positional order and which slots are pointers is fixed per kernel, matching each runner's own
    `kernel[grid](...)` call exactly (see module docstring / STAGING.md for the source lines):
      add_kernel:            x_ptr, y_ptr, output_ptr, n_elements
      softmax_kernel:        output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols
      _layer_norm_fwd_fused: X, Y, W, B, Mean, Rstd, stride, N, eps
    (BLOCK_SIZE/num_stages are constexprs, passed as kwargs, never positionally.)
    """
    if kernel_name == "add_kernel":
        return [dtype_marker, dtype_marker, dtype_marker, cell["n"]]
    if kernel_name == "softmax_kernel":
        rows, cols = cell["rows"], cell["cols"]
        return [dtype_marker, dtype_marker, cols, cols, rows, cols]
    if kernel_name == "_layer_norm_fwd_fused":
        rows, cols = cell["rows"], cell["cols"]
        eps = cell.get("eps", 1e-5)
        return [dtype_marker, dtype_marker, dtype_marker, dtype_marker, dtype_marker, dtype_marker,
                cols, cols, eps]
    raise EmitBlocked(f"EMIT_BLOCKED: no positional-argument builder for kernel {kernel_name!r}")


def _triton_grid(kernel_name: str, cell: dict) -> tuple[int]:
    if kernel_name == "add_kernel":
        block_size = cell["constants"]["BLOCK_SIZE"]
        n = cell["n"]
        return ((n + block_size - 1) // block_size,)
    if kernel_name in ("softmax_kernel", "_layer_norm_fwd_fused"):
        return (cell["rows"],)
    raise EmitBlocked(f"EMIT_BLOCKED: no grid builder for kernel {kernel_name!r}")


def compile_triton_cubin(kernel, kernel_name: str, cell: dict) -> tuple[bytes, dict]:
    """Compile a Triton @jit kernel to a cubin WITHOUT LAUNCHING, via `JITFunction.warmup(...)`
    with `MockTensor` pointer specialization -- fixing the 2026-09-28 Blackwell finding that the
    prior `triton.compile(ASTSource(...))` path skipped the argument specialization
    (`tt.divisibility=16` on aligned pointers/divisible integers, equal-to-1) a real launch
    applies, which is exactly why vector-add/softmax's SASS did not reproduce the September
    (real-launch) dumps: `JITFunction.warmup(self, *args, grid, **kwargs)` is implemented (Triton
    3.6.0, confirmed live on Blackwell) as `self.run(grid=grid, warmup=True,
    *map(MockTensor.wrap_dtype, args), **kwargs)` -- passing a torch dtype for each pointer
    argument builds a `MockTensor` (no GPU allocation) that gets the same specialization a real
    tensor argument would, with no kernel launch. This needs a live CUDA device (the target arch
    comes from whichever GPU `CUDA_VISIBLE_DEVICES` selects, exactly as a real launch would infer
    it -- there is no explicit `target=` here any more, unlike the prior ASTSource path)."""
    try:
        import torch
    except ImportError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: PyTorch unavailable (needed for warmup's dtype marker "
                           f"and MockTensor): {exc}") from exc
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: Triton runtime unavailable: {exc}") from exc
    try:
        from triton.runtime.jit import MockTensor  # noqa: F401  -- existence check only; warmup
        # uses this internally via `map(MockTensor.wrap_dtype, args)`, so this import is not
        # strictly required for the call below to work, but its absence means the installed
        # Triton's warmup() does not have the shape this fix assumes -- refuse loudly rather than
        # let a deep AttributeError from inside `run()` stand in for a clear error message.
    except ImportError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: triton.runtime.jit.MockTensor not found on the "
                           f"installed Triton; the warmup()-based compile fix assumes it exists "
                           f"(Triton 3.6.0, confirmed live on Blackwell 2026-09-28): {exc}") from exc

    spec = TRITON_KERNEL_SIGNATURES[kernel_name]
    actual_arg_names = list(getattr(kernel, "arg_names", []) or [])
    if not actual_arg_names:
        raise EmitBlocked(
            f"EMIT_BLOCKED: kernel object for {kernel_name} has no (non-empty) 'arg_names' -- it "
            f"does not look like a @triton.jit JITFunction (got {type(kernel)!r})."
        )
    if actual_arg_names != spec["arg_names"]:
        raise EmitBlocked(
            f"EMIT_BLOCKED: pinned kernel {kernel_name} arg_names changed: expected "
            f"{spec['arg_names']}, got {actual_arg_names}."
        )
    warmup = getattr(kernel, "warmup", None)
    if warmup is None or not callable(warmup):
        raise EmitBlocked(
            f"EMIT_BLOCKED: kernel object for {kernel_name} has no callable 'warmup' method -- "
            f"the installed Triton's JITFunction does not have the shape this fix assumes."
        )

    args = _triton_positional_args(kernel_name, cell, torch.float32)
    grid = _triton_grid(kernel_name, cell)
    kwargs = dict(cell["constants"])
    kwargs["num_warps"] = cell["num_warps"]
    if kernel_name == "_layer_norm_fwd_fused":
        kwargs["num_ctas"] = 1  # matches layernorm_runner.py's own launch kwarg exactly

    compiled = warmup(*args, grid=grid, **kwargs)
    if compiled is None:
        raise EmitBlocked(f"EMIT_BLOCKED: kernel.warmup(...) for {kernel_name} returned None "
                           f"(expected a compiled-kernel object with an 'asm' dict)")
    asm = getattr(compiled, "asm", None)
    if not isinstance(asm, dict) or "cubin" not in asm or not asm["cubin"]:
        raise EmitBlocked(
            f"EMIT_BLOCKED: warmup-compiled Triton kernel has no cubin at compiled.asm['cubin']; "
            f"compiled.asm keys were {list(asm or {})}."
        )
    cubin = asm["cubin"]
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")  # defensive: some Triton versions may hand back str
    return cubin, {"grid": list(grid), "num_warps": kwargs["num_warps"],
                   "compile_method": "JITFunction.warmup_with_MockTensor_specialization"}


# --- CUDA Samples path ------------------------------------------------------------------------

def compile_cuda_samples_cubin(source_file: Path, arch_sm: str, nvcc: str, include_dirs: list[Path],
                                out_path: Path) -> None:
    extra_args = []
    for d in include_dirs:
        extra_args += ["-I", str(d)]
    cmd = [nvcc, "-O3", "-std=c++17", f"-arch={arch_sm}", *extra_args,
           "-cubin", str(source_file), "-o", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise EmitBlocked(f"EMIT_BLOCKED: nvcc -cubin failed: {' '.join(cmd)}\n{result.stderr[-4000:]}")


# --- Driver ------------------------------------------------------------------------------------

def process_family(parent_id: str, family: dict, args, manifest_entries: list[dict]) -> dict:
    out: dict[str, dict] = {}
    kind = family["kind"]
    if kind == "triton":
        kernel_name = family["kernel_name"]
        if TRITON_KERNEL_NAME.get(parent_id) != kernel_name:
            raise EmitBlocked(f"EMIT_BLOCKED: manifest kernel_name {kernel_name!r} for {parent_id} "
                               f"does not match the hardcoded TRITON_KERNEL_NAME mapping")
        source_file = args.triton_source_root / family["source_rel"]
        if not source_file.is_file():
            raise EmitBlocked(f"EMIT_BLOCKED: missing pinned Triton source {source_file}")
        source_sha256 = sha256_file(source_file)
        expected_sha256 = family["source_sha256"]
        if source_sha256 != expected_sha256:
            raise EmitBlocked(f"EMIT_BLOCKED: pinned Triton source sha256 mismatch for {parent_id}: "
                               f"expected {expected_sha256}, got {source_sha256}")
        kernel = import_pinned_triton_kernel(source_file, kernel_name)
        for key, cell in family["cells"].items():
            cubin, compile_info = compile_triton_cubin(kernel, kernel_name, cell)
            cubin_path = Path(tempfile.mkstemp(dir=args.out_dir, prefix=f"{parent_id}_{key.replace('/', '_')}_",
                                                suffix=".cubin")[1])
            cubin_path.write_bytes(cubin)
            sass_text = dump_sass(args.cuobjdump, cubin_path)
            fn_name, fn_sass = smc.isolate_function_section(sass_text, kernel_name)
            out[key] = smc.sass_instruction_mix(fn_sass)
            manifest_entries.append({
                "parent_id": parent_id, "cell": key, "kernel_function_name": fn_name,
                "cubin_sha256": sha256_bytes(cubin), "source_revision": family["source_revision"],
                "source_sha256": source_sha256, "compiler": "triton", "compiler_version": _triton_version(),
                **compile_info,
            })
    elif kind == "cuda_samples":
        source_file = args.cuda_samples_root / family["source_rel"]
        if not source_file.is_file():
            raise EmitBlocked(f"EMIT_BLOCKED: missing pinned CUDA Samples source {source_file}")
        source_sha256 = sha256_file(source_file)
        expected_sha256 = family["source_sha256"]
        if source_sha256 != expected_sha256:
            raise EmitBlocked(f"EMIT_BLOCKED: pinned CUDA Samples source sha256 mismatch for {parent_id}: "
                               f"expected {expected_sha256}, got {source_sha256}")
        include_dirs = [args.cuda_samples_root / rel for rel in family.get("include_rels", [])]
        for d in include_dirs:
            if not d.is_dir():
                raise EmitBlocked(f"EMIT_BLOCKED: missing include dir {d}")
        for key, cell in family["cells"].items():
            cubin_path = Path(tempfile.mkstemp(dir=args.out_dir, prefix=f"{parent_id}_{key.replace('/', '_')}_",
                                                suffix=".cubin")[1])
            compile_cuda_samples_cubin(source_file, args.arch, args.nvcc, include_dirs, cubin_path)
            sass_text = dump_sass(args.cuobjdump, cubin_path)
            fn_name, fn_sass = smc.isolate_function_section(sass_text, cell["entry_substring"])
            out[key] = smc.sass_instruction_mix(fn_sass)
            manifest_entries.append({
                "parent_id": parent_id, "cell": key, "kernel_function_name": fn_name,
                "cubin_sha256": sha256_file(cubin_path), "source_revision": family["source_revision"],
                "source_sha256": source_sha256, "compiler": "nvcc", "compiler_version": nvcc_version(args.nvcc),
            })
    else:
        raise EmitBlocked(f"EMIT_BLOCKED: unknown family kind {kind!r} for {parent_id}")
    return out


def _triton_version() -> str:
    try:
        import triton
        return getattr(triton, "__version__", "unknown")
    except ImportError:
        return "unavailable"


def run(args) -> int:
    check_visible_devices(args.expect_visible_devices)
    manifest = json.loads(args.cells_manifest.read_text())
    requested = args.families.split(",") if args.families else list(manifest)
    unknown = [f for f in requested if f not in manifest]
    if unknown:
        raise EmitBlocked(f"EMIT_BLOCKED: --families named unknown parents not in the manifest: {unknown}")

    device_info = {}
    if any(manifest[p]["kind"] == "triton" for p in requested):
        device_info = check_device_arch(args.arch)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features: dict[str, dict] = {}
    manifest_entries: list[dict] = []
    for parent_id in requested:
        features[parent_id] = process_family(parent_id, manifest[parent_id], args, manifest_entries)

    features_path = args.out_dir / f"sass_features_{args.arch}.json"
    features_path.write_text(json.dumps(features, indent=2, sort_keys=True) + "\n")
    manifest_path = args.out_dir / f"manifest_{args.arch}.json"
    manifest_path.write_text(json.dumps({
        "arch": args.arch, "expect_visible_devices": args.expect_visible_devices,
        "nvcc_version": nvcc_version(args.nvcc) if args.nvcc else None,
        "cuobjdump_version": cuobjdump_version(args.cuobjdump),
        "triton_version": _triton_version(),
        "live_device": device_info or None,
        "cells": manifest_entries,
    }, indent=2, sort_keys=True) + "\n")

    n_cells = sum(len(v) for v in features.values())
    print(f"wrote {features_path} ({len(features)} families, {n_cells} cells)")
    print(f"wrote {manifest_path} ({len(manifest_entries)} cell records)")
    return 0


def self_test() -> None:
    """CPU-only, no Triton/CUDA/nvcc/cuobjdump import. Exercises:
      - the classifier reproduces a committed Blackwell feature vector from synthetic SASS text
        matching sass_features_blackwell.json's dev_triton_softmax/small/c1 cell shape
      - function-section isolation, including refusal on 0 and on >1 matches
      - the manifest builder's family/cell shapes (via the committed cell_manifest_*.json files,
        if present next to this script)
    """
    # 1) Classifier reproduces a hand-built SASS sample with known counts.
    sample = """\
Function : add_kernel
/*0000*/                   LDG.E R2, [R4] ;
/*0010*/                   LDG.E R3, [R6] ;
/*0020*/                   FADD R5, R2, R3 ;
/*0030*/                   IMAD R6, R0, R1, R2 ;
/*0040*/                   STG.E [R8], R5 ;
/*0050*/                   BRA 0x60 ;
/*0060*/                   EXIT ;
"""
    mix = smc.sass_instruction_mix(sample)
    assert mix["static_total_instructions"] == 7, mix
    assert mix["static_load_count"] == 2, mix
    assert mix["static_store_count"] == 1, mix
    assert mix["static_branch_count"] == 2, mix  # BRA + EXIT
    assert mix["static_arith_count"] == 2, mix  # FADD + IMAD
    assert abs(mix["static_load_density"] - 2 / 7) < 1e-12, mix

    # 2) Function-section isolation: exactly one match succeeds.
    name, body = smc.isolate_function_section(sample, "add_kernel")
    assert name == "add_kernel" and "FADD" in body

    # Refuses on zero matches.
    try:
        smc.isolate_function_section(sample, "softmax_kernel")
        raise AssertionError("zero-match isolation was accepted")
    except ValueError as exc:
        assert "found 0" in str(exc), exc

    # Refuses on ambiguous (>1) matches -- two function sections both containing "reduce4".
    multi = """\
Function : _Z7reduce4IfLj512EEvPT_S1_j
/*0000*/                   LDG.E R2, [R4] ;
/*0010*/                   EXIT ;
Function : _Z7reduce4IfLj256EEvPT_S1_j
/*0000*/                   LDG.E R2, [R4] ;
/*0010*/                   STG.E [R6], R2 ;
/*0020*/                   EXIT ;
"""
    try:
        smc.isolate_function_section(multi, "reduce4")
        raise AssertionError("ambiguous isolation was accepted")
    except ValueError as exc:
        assert "found 2" in str(exc), exc
    # The blockSize-anchored substring resolves exactly one.
    name256, body256 = smc.isolate_function_section(multi, "reduce4IfLj256E")
    assert name256 == "_Z7reduce4IfLj256EEvPT_S1_j"
    mix256 = smc.sass_instruction_mix(body256)
    assert mix256["static_total_instructions"] == 3 and mix256["static_store_count"] == 1, mix256

    # 3) Manifest shape check, if the committed manifests are present next to this script.
    for arch, fname in (("blackwell", "cell_manifest_blackwell.json"), ("ada", "cell_manifest_ada.json")):
        path = HERE / fname
        if not path.is_file():
            continue
        m = json.loads(path.read_text())
        for parent_id, family in m.items():
            assert family["kind"] in ("triton", "cuda_samples"), (arch, parent_id, family["kind"])
            assert len(family["cells"]) == 12, (arch, parent_id, len(family["cells"]))
            for key, cell in family["cells"].items():
                assert "/" in key
                if family["kind"] == "triton":
                    assert "constants" in cell and "num_warps" in cell
                else:
                    assert "entry_substring" in cell or "grid_x" in cell

    # 4) Regression test for the 2026-09-28 Blackwell finding: kernel.warmup(...) must be called
    # with pointer args as torch dtypes and REAL scalar values from the cell manifest (not
    # placeholder/zero values), plus the right grid/num_warps/num_stages -- runs entirely against
    # fake torch/triton/MockTensor modules (no real install needed).
    _test_compile_triton_cubin_uses_warmup_with_real_args()

    print("EMIT_SASS_SELFTEST_OK: classifier, isolation (unique/zero/ambiguous), manifest shape, "
          "warmup-based Triton compile args")


def _install_fake_torch_and_triton_for_warmup_test():
    """Install minimal fake `torch`/`triton`/`triton.runtime.jit` modules into sys.modules,
    returning a restore() callback -- lets `compile_triton_cubin()`'s real `import torch`/
    `import triton`/`from triton.runtime.jit import MockTensor` statements succeed on a machine
    with neither installed, so the warmup-argument regression test can run CPU-only."""
    import types

    class FakeDtype:
        def __repr__(self):
            return "torch.float32"

    fake_float32 = FakeDtype()

    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = fake_float32

    class FakeMockTensor:
        @staticmethod
        def wrap_dtype(x):
            return x

    triton_mod = types.ModuleType("triton")
    triton_mod.__version__ = "3.6.0-fake"
    runtime_mod = types.ModuleType("triton.runtime")
    jit_mod = types.ModuleType("triton.runtime.jit")
    jit_mod.MockTensor = FakeMockTensor
    triton_mod.runtime = runtime_mod
    runtime_mod.jit = jit_mod

    import sys as _sys
    saved = {k: _sys.modules.get(k) for k in ("torch", "triton", "triton.runtime", "triton.runtime.jit")}
    _sys.modules["torch"] = torch_mod
    _sys.modules["triton"] = triton_mod
    _sys.modules["triton.runtime"] = runtime_mod
    _sys.modules["triton.runtime.jit"] = jit_mod

    def restore():
        for k, v in saved.items():
            if v is None:
                _sys.modules.pop(k, None)
            else:
                _sys.modules[k] = v

    return fake_float32, restore


def _test_compile_triton_cubin_uses_warmup_with_real_args() -> None:
    fake_float32, restore = _install_fake_torch_and_triton_for_warmup_test()
    try:
        recorded = {}

        class FakeCompiled:
            def __init__(self):
                self.asm = {"cubin": b"FAKE_CUBIN_BYTES"}

        class FakeJITFunction:
            def __init__(self, name, arg_names):
                self.__name__ = name
                self.arg_names = list(arg_names)

            def warmup(self, *args, grid, **kwargs):
                recorded["args"] = args
                recorded["grid"] = grid
                recorded["kwargs"] = kwargs
                return FakeCompiled()

        # vector-add, large/c4: n=67108864, BLOCK_SIZE=1024 (from cell_manifest_ada.json /
        # cell_manifest_blackwell.json -- identical geometry on both, read from workload_cells.csv).
        cell = {"constants": {"BLOCK_SIZE": 1024}, "num_warps": 4, "n": 67108864}
        kernel = FakeJITFunction("add_kernel", TRITON_KERNEL_SIGNATURES["add_kernel"]["arg_names"])
        cubin, info = compile_triton_cubin(kernel, "add_kernel", cell)
        assert cubin == b"FAKE_CUBIN_BYTES", cubin
        args = recorded["args"]
        assert len(args) == 4, args
        assert args[0] is fake_float32 and args[1] is fake_float32 and args[2] is fake_float32, args
        assert args[3] == 67108864, args  # the REAL n for large/c4, not a placeholder
        assert recorded["grid"] == ((67108864 + 1024 - 1) // 1024,), recorded["grid"]
        assert recorded["kwargs"]["num_warps"] == 4, recorded["kwargs"]
        assert recorded["kwargs"]["BLOCK_SIZE"] == 1024, recorded["kwargs"]
        assert info["compile_method"] == "JITFunction.warmup_with_MockTensor_specialization"

        # softmax_kernel: rows/cols/num_stages/num_warps must all be the cell's real values, and
        # positional order must match softmax_runner.py's own kernel[(rows,)](y, x, cols, cols,
        # rows, cols, ...) call -- output_ptr, input_ptr, input_row_stride, output_row_stride,
        # n_rows, n_cols.
        recorded.clear()
        cell_sm = {"constants": {"BLOCK_SIZE": 1024, "num_stages": 2}, "num_warps": 2,
                   "rows": 8192, "cols": 1024}
        kernel_sm = FakeJITFunction("softmax_kernel", TRITON_KERNEL_SIGNATURES["softmax_kernel"]["arg_names"])
        compile_triton_cubin(kernel_sm, "softmax_kernel", cell_sm)
        args_sm = recorded["args"]
        assert args_sm[0] is fake_float32 and args_sm[1] is fake_float32, args_sm
        assert args_sm[2:] == (1024, 1024, 8192, 1024), args_sm  # stride, stride, n_rows, n_cols
        assert recorded["grid"] == (8192,), recorded["grid"]
        assert recorded["kwargs"]["num_stages"] == 2 and recorded["kwargs"]["num_warps"] == 2, recorded["kwargs"]

        # Refuses loudly if the kernel object has no callable warmup (installed API mismatch).
        class NoWarmup:
            arg_names = TRITON_KERNEL_SIGNATURES["add_kernel"]["arg_names"]

        try:
            compile_triton_cubin(NoWarmup(), "add_kernel", cell)
            raise AssertionError("kernel with no warmup method was accepted")
        except EmitBlocked as exc:
            assert "warmup" in str(exc), exc
    finally:
        restore()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCH_CC), help="sm_120 (Blackwell) or sm_89 (Ada)")
    parser.add_argument("--families", default="", help="comma-separated parent_ids; default: all in the manifest")
    parser.add_argument("--cells-manifest", type=Path)
    parser.add_argument("--triton-source-root", type=Path)
    parser.add_argument("--cuda-samples-root", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expect-visible-devices", default="")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        self_test()
        return 0

    for required in ("arch", "cells_manifest", "out_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (unless --selftest)")

    try:
        return run(args)
    except EmitBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
