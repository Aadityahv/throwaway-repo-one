#!/usr/bin/env python3
"""REMOTE (compile-only, never launches a kernel). Extends the static-feature pipeline of
`measured/cell_manifests/emit_sass.py` with the SASS facts that pipeline does not keep:
load/store WIDTHS per memory space, the MUFU / SHFL / BAR / shared-memory / atomic split, and the
widest global load, for the same catalog cells on the same compile paths, plus two more sets of
compile targets (the three unseen-operator parents and the calibration fixtures).

What is reused, not reimplemented
  * `sass_mix_classifier.sass_instruction_mix` -> the ten `static_*` fields, unchanged.
  * `sass_mix_classifier.isolate_function_section` -> the exact function section the existing
    pipeline isolates (catalog cells use the manifest's `entry_substring`, exactly as emit_sass does).
  * `emit_sass` -> Triton compile via `kernel.warmup(...)` with MockTensor specialization, the
    CUDA-Samples `nvcc -O3 -std=c++17 -arch=<arch> -cubin` compile, `dump_sass`, the
    `CUDA_VISIBLE_DEVICES` guard, the live-device-arch check, source sha256 pinning, version probes.

Files that must sit together (flat staging dir, or in-repo where the imports resolve by relative path):
    emit_widths.py  emit_sass.py  sass_mix_classifier.py
plus the JSON manifest `cell_manifest_<gpu>.json` (catalog families) and the pinned sources named
in EMIT_WIDTHS_HANDOFF.md. In-repo, emit_sass.py / sass_mix_classifier.py are found at
`../evaluation_data/measured/cell_manifests/` without copying.

Guarantees
  * Nothing is launched. Triton families need a live CUDA device only to infer the target arch and
    to query its capability (identical to emit_sass); nvcc/cuobjdump never run a kernel.
  * Refuses (exit 2) if CUDA_VISIBLE_DEVICES is not exactly --expect-visible-devices.
  * Writes only under --out-dir.
  * Never guesses a width: an unrecognised width or cache modifier lands in an explicit
    "unknown_width" bucket and flags the cell (`flags`, and the run summary lists every unknown
    token). No plausible-but-unmeasured number is ever substituted.
  * `--reproduce-against <committed sass_features_sm_XXX.json>` recomputes the existing `static_*`
    fields and exits 3 (naming cell and field) on any mismatch; the output JSON is still written so
    the mismatch can be inspected, with `widths_usable: false`.

SASS syntax facts and the byte-width mapping are documented at WIDTH_BYTES / NON_WIDTH_MODIFIERS
below and in EMIT_WIDTHS_HANDOFF.md, including which entries were seen in committed real SASS and
which are inferred.

Output: `<out-dir>/emit_widths_<arch>.json` (schema `emit_widths/1`) and
`<out-dir>/emit_widths_static_features_<arch>.json` (catalog families only, same shape as
emit_sass's `sass_features_<arch>.json`, for direct diffing). Cubins are kept in
`<out-dir>/cubins/`.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ALAVANI_DIR = HERE.parent / "evaluation_data" / "measured" / "cell_manifests"
for _p in (str(HERE), str(_ALAVANI_DIR)):  # flat staging wins; in-repo path is the fallback
    if _p not in sys.path:
        sys.path.append(_p)

import sass_mix_classifier as smc  # noqa: E402
import emit_sass as es  # noqa: E402

SCHEMA_VERSION = "emit_widths/1"
PARSER_VERSION = "1"
EmitBlocked = es.EmitBlocked
EXIT_BLOCKED = 2
EXIT_REPRO_FAILED = 3
EXIT_FAMILY_ERRORS = 4

# --------------------------------------------------------------------------------------------
# SASS parsing
# --------------------------------------------------------------------------------------------

# Superset of smc.SASS_LINE: accepts 4+ hex digits of address (kernels beyond 0xFFFF bytes have
# 5-digit addresses that smc.SASS_LINE, which needs exactly 4, silently skips) and the always/never
# predicates @PT / @!PT (also skipped by smc.SASS_LINE). The parser counts both ways and flags the
# cell if the two totals differ, so the difference is visible instead of silent.
INSTR_LINE = re.compile(
    r"/\*(?P<addr>[0-9a-fA-F]{4,})\*/\s+(?P<pred>@!?U?P(?:\d+|T)\s+)?"
    r"(?P<op>[A-Z][A-Z0-9_]*)(?P<mods>(?:\.[A-Za-z0-9_]+)*)"
)

# Width modifier token -> bytes per thread-level access.
#   Seen in committed real Blackwell (sm_120) SASS (calibration/predictor/blackwell_eps_ident_*,
#   blackwell_gather_width_*): "64" (LDG.E.64, STG.E.64, LDC.64, LDCU.64), "128" (LDG.E.128,
#   STG.E.128, LDCU.128), "U8" (LDC.U8), "S8" (ULDC.S8).
#   Inferred from memory of SASS (NOT seen in a committed real dump here): "U16", "S16", "32".
WIDTH_BYTES = {"U8": 1, "S8": 1, "U16": 2, "S16": 2, "32": 4, "64": 8, "128": 16}
# Bare signedness markers ("LDS.U", "LDS.U.128"): carry no width information by themselves.
# Inferred spelling (task brief lists 'LDS.U'); treated as neutral, never as a width.
NEUTRAL_MODIFIERS = {"U", "S"}
# Modifiers that describe cache policy / consistency / scope / address handling, not width.
#   Seen in real committed SASS: E, STRONG, SM, SYS.
#   Inferred from memory (not seen here): EL, LU, EU, EF, EN, NA, CI, CONSTANT, WEAK, CTA, GPU, MMIO,
#   PRIVATE, ENL2, LTC64B, LTC128B, LTC256B, BYPASS, ACCESS, CG, CA, CS, CV.
# Anything NOT in WIDTH_BYTES, NEUTRAL_MODIFIERS or this set makes the instruction "unknown_width".
NON_WIDTH_MODIFIERS = {
    "E", "STRONG", "SM", "SYS", "GPU", "CTA", "WEAK", "CONSTANT", "MMIO", "PRIVATE",
    "EL", "LU", "EU", "EF", "EN", "NA", "CI", "ENL2", "LTC64B", "LTC128B", "LTC256B",
    "BYPASS", "ACCESS", "CG", "CA", "CS", "CV",
}
NON_WIDTH_SEEN_IN_REAL_SASS = {"E", "STRONG", "SM", "SYS"}
WIDTH_SEEN_IN_REAL_SASS = {"U8", "S8", "64", "128"}

# Memory classes by base opcode. "default_bytes" applies only when NO width modifier is present:
# LDG.E / STG.E / LDC / LDS / STS with no width token are 32-bit (confirmed by the real width
# microbenchmark: the 4-byte kernel's float loads are bare `LDG.E.STRONG.SM`, the uint32 sink store
# is bare `STG.E`; bare LDC/LDS/STS also appear in real dumps). LD/ST/LDL/STL/LDGSTS follow the same
# convention by inference.
MEMORY_CLASS_BY_OPCODE = {
    "LDG": "global_load", "STG": "global_store",
    "LD": "generic_load", "ST": "generic_store",
    "LDS": "shared_load", "STS": "shared_store",
    "LDL": "local_load", "STL": "local_store",
    "LDC": "constant_load", "LDCU": "constant_load", "ULDC": "constant_load",
    "LDGSTS": "async_global_to_shared",
    "LDSM": "shared_matrix_load", "STSM": "shared_matrix_store",
}
# Classes whose bytes are not a per-thread scalar width (no width token exists / is meaningful).
NO_WIDTH_CLASSES = {"shared_matrix_load", "shared_matrix_store"}
# Memory-ish opcodes recognised but not width-classified (counted by name, no flag).
OTHER_MEMORY_OPCODES = {
    "LDGDEPBAR", "LDSLK", "STSCUL", "LDTRAM", "CCTL", "CCTLL", "CCTLT", "TLD", "TLD4", "TEX", "TXQ",
    "SULD", "SUST", "SURED", "SUATOM", "PREFETCH", "MEMSET", "MEMCPY",
}
ATOMIC_GLOBAL_OPCODES = {"ATOM", "ATOMG", "RED", "REDG"}
ATOMIC_SHARED_OPCODES = {"ATOMS"}
BARRIER_OPCODES = {"BAR"}
FENCE_OPCODES = {"MEMBAR", "FENCE", "ERRBAR"}
# Base opcodes that start with LD/ST but are not in any table above: flagged, never silently ignored.
CLASS_ORDER = [
    "global_load", "global_store", "generic_load", "generic_store", "shared_load", "shared_store",
    "local_load", "local_store", "constant_load", "async_global_to_shared", "shared_matrix_load",
    "shared_matrix_store",
]
BYTE_KEYS = ("1", "2", "4", "8", "16", "unknown_width")


def _new_class() -> dict:
    return {"count": 0, "bytes_hist": collections.Counter(), "known_bytes_sum": 0,
            "default_width_used": 0}


def resolve_width(mods: list[str], cls: str) -> tuple[int | None, str, list[str]]:
    """Return (bytes or None, how, offending_tokens).

    how is one of "explicit", "default", "not_applicable", "unknown". None bytes means the width
    could not be determined from the documented mapping (goes to the unknown_width bucket)."""
    if cls in NO_WIDTH_CLASSES:
        return None, "not_applicable", []
    width_tokens = [m for m in mods if m in WIDTH_BYTES]
    other = [m for m in mods if m not in WIDTH_BYTES and m not in NEUTRAL_MODIFIERS
             and m not in NON_WIDTH_MODIFIERS]
    if other:
        return None, "unknown", other
    if len(set(width_tokens)) > 1:
        return None, "unknown", width_tokens
    if width_tokens:
        return WIDTH_BYTES[width_tokens[0]], "explicit", []
    return 4, "default", []


def parse_sass_widths(text: str) -> dict:
    """Parse the SASS text of ONE function section. Pure function, no I/O."""
    classes = {name: _new_class() for name in CLASS_ORDER}
    opcode_hist: collections.Counter = collections.Counter()
    full_hist: collections.Counter = collections.Counter()
    mufu: collections.Counter = collections.Counter()
    shfl: collections.Counter = collections.Counter()
    bar: collections.Counter = collections.Counter()
    fence: collections.Counter = collections.Counter()
    atom_global: collections.Counter = collections.Counter()
    atom_shared: collections.Counter = collections.Counter()
    other_mem: collections.Counter = collections.Counter()
    unclassified_ld_st: collections.Counter = collections.Counter()
    unknown_mods: collections.Counter = collections.Counter()
    conflicting: collections.Counter = collections.Counter()
    n_instr = 0

    for m in INSTR_LINE.finditer(text):
        op, mods_str = m.group("op"), m.group("mods")
        mods = [t for t in mods_str.split(".") if t]
        n_instr += 1
        opcode_hist[op] += 1
        full_hist[op + mods_str] += 1
        cls = MEMORY_CLASS_BY_OPCODE.get(op)
        if cls is not None:
            rec = classes[cls]
            rec["count"] += 1
            nbytes, how, bad = resolve_width(mods, cls)
            if how == "not_applicable":
                rec["bytes_hist"]["unknown_width"] += 0  # keep key absent unless real unknowns
            elif nbytes is None:
                rec["bytes_hist"]["unknown_width"] += 1
                for tok in bad:
                    (conflicting if tok in WIDTH_BYTES else unknown_mods)[f"{op}.{tok}"] += 1
            else:
                rec["bytes_hist"][str(nbytes)] += 1
                rec["known_bytes_sum"] += nbytes
                if how == "default":
                    rec["default_width_used"] += 1
        elif op == "MUFU":
            mufu[".".join(mods) or "(none)"] += 1
        elif op == "SHFL":
            shfl[".".join(mods) or "(none)"] += 1
        elif op in BARRIER_OPCODES:
            bar[".".join(mods) or "(none)"] += 1
        elif op in FENCE_OPCODES:
            fence[op + mods_str] += 1
        elif op in ATOMIC_GLOBAL_OPCODES:
            atom_global[op + mods_str] += 1
        elif op in ATOMIC_SHARED_OPCODES:
            atom_shared[op + mods_str] += 1
        elif op in OTHER_MEMORY_OPCODES:
            other_mem[op] += 1
        elif op.startswith(("LD", "ST", "ULD")):
            # Any other LD*/ST* opcode (e.g. an instruction introduced by a newer arch) is not
            # silently dropped.
            unclassified_ld_st[op] += 1

    flags: list[str] = []
    if n_instr == 0:
        raise ValueError("no SASS instruction lines matched")
    classifier_lines = len(smc.SASS_LINE.findall(text))
    if classifier_lines != n_instr:
        flags.append("classifier_line_count_mismatch")
    if unknown_mods:
        flags.append("unknown_width_modifier")
    if conflicting:
        flags.append("conflicting_width_modifiers")
    if unclassified_ld_st:
        flags.append("unclassified_memory_opcode")

    def finish(rec: dict) -> dict:
        hist = {k: rec["bytes_hist"][k] for k in BYTE_KEYS if rec["bytes_hist"].get(k)}
        return {"count": rec["count"], "bytes_hist": hist, "known_bytes_sum": rec["known_bytes_sum"],
                "default_width_used": rec["default_width_used"]}

    out_classes = {name: finish(rec) for name, rec in classes.items()}

    def widest(names: tuple[str, ...]) -> tuple[int | None, bool]:
        """(max known width in bytes over the named classes or None, has_unknown_width)."""
        widths, unknown = [], False
        for n in names:
            for k, v in out_classes[n]["bytes_hist"].items():
                if k == "unknown_width":
                    unknown = unknown or v > 0
                elif v:
                    widths.append(int(k))
        return (max(widths) if widths else None), unknown

    w_ldg, u_ldg = widest(("global_load",))
    w_glob, u_glob = widest(("global_load", "generic_load"))
    w_stg, u_stg = widest(("global_store",))
    w_glob_st, u_glob_st = widest(("global_store", "generic_store"))
    w_lds, u_lds = widest(("shared_load",))
    w_sts, u_sts = widest(("shared_store",))

    # Combined histogram (LDG plus generic LD), the "global load" definition PROTOCOL.md uses.
    combo = collections.Counter()
    for n in ("global_load", "generic_load"):
        combo.update(out_classes[n]["bytes_hist"])
    combined_global_load = {
        "count": out_classes["global_load"]["count"] + out_classes["generic_load"]["count"],
        "bytes_hist": {k: combo[k] for k in BYTE_KEYS if combo.get(k)},
    }

    atom_total = sum(atom_global.values()) + sum(atom_shared.values())
    return {
        "parser_version": PARSER_VERSION,
        "total_instructions": n_instr,
        "classifier_instruction_lines": classifier_lines,
        "memory": out_classes,
        "global_or_generic_load": combined_global_load,
        "widest_global_load_bytes": w_glob,
        "widest_global_load_is_lower_bound": u_glob,
        "widest_ldg_only_bytes": w_ldg,
        "widest_ldg_only_is_lower_bound": u_ldg,
        "widest_global_store_bytes": w_glob_st,
        "widest_global_store_is_lower_bound": u_glob_st,
        "widest_stg_only_bytes": w_stg,
        "widest_shared_load_bytes": w_lds,
        "widest_shared_load_is_lower_bound": u_lds,
        "widest_shared_store_bytes": w_sts,
        "widest_shared_store_is_lower_bound": u_sts,
        "has_unknown_width_global_load": bool(u_glob),
        "mufu": {"count": sum(mufu.values()), "by_subop": dict(sorted(mufu.items()))},
        "shfl": {"count": sum(shfl.values()), "by_variant": dict(sorted(shfl.items()))},
        "bar": {"count": sum(bar.values()), "by_variant": dict(sorted(bar.items()))},
        "fence": {"count": sum(fence.values()), "by_opcode": dict(sorted(fence.items()))},
        "atomics": {"count": atom_total,
                    "global_count": sum(atom_global.values()), "shared_count": sum(atom_shared.values()),
                    "by_opcode": dict(sorted({**atom_global, **atom_shared}.items()))},
        "ffma": opcode_hist.get("FFMA", 0), "fmul": opcode_hist.get("FMUL", 0),
        "fadd": opcode_hist.get("FADD", 0),
        "packed_fp32_ffma2_fmul2_fadd2": [opcode_hist.get("FFMA2", 0), opcode_hist.get("FMUL2", 0),
                                          opcode_hist.get("FADD2", 0)],
        "other_memory_opcodes": dict(sorted(other_mem.items())),
        "unclassified_ld_st_opcodes": dict(sorted(unclassified_ld_st.items())),
        "unknown_width_modifiers": dict(sorted(unknown_mods.items())),
        "conflicting_width_modifiers": dict(sorted(conflicting.items())),
        "opcode_histogram": dict(sorted(opcode_hist.items())),
        "full_opcode_histogram": dict(sorted(full_hist.items())),
        "flags": flags,
    }


# --------------------------------------------------------------------------------------------
# Compile targets
# --------------------------------------------------------------------------------------------

# Unseen-operator parents (tiresias/framework/adapters/adapters.json; pinned CUDA Samples rev).
# Hard-coded second copy, cross-checked against adapters.json in --selftest when the repo is present.
PINNED_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
REDUCTION_KERNEL_REL = "cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu"
TRANSPOSE_REL = "cpp/6_Performance/transpose/transpose.cu"
UNSEEN_PARENTS = {
    "alt_cuda_samples_reduction": {
        "source_rel": REDUCTION_KERNEL_REL,
        "source_sha256": "c9ac9a9a424726522dd616fa24d58e6f0919ac95dd746a233d5fb857bffffde9",
        "include_rels": ["Common", "cpp/2_Concepts_and_Techniques/reduction"],
        # Runner recipe (alt_reduction_runner.build_binary, first nvcc step), with -c/-o replaced by -cubin.
        "nvcc_flags": ["-O2", "--default-stream", "per-thread"], "defines": [],
        "regimes": {"small": 65536, "medium": 1048576, "large": 16777216},   # n elements
        "candidates": {"c1": "reduce0", "c2": "reduce1", "c3": "reduce6"},
        "threads_per_block": 256,
    },
    "alt_cuda_samples_copy": {
        "source_rel": TRANSPOSE_REL,
        "source_sha256": "ba300b6b5f4dbc17ec72bfd3d1ee082508632ec431fe46ba54e6e3ce05292a95",
        "include_rels": ["Common"],
        # Runner recipe (tile_family.build_binary, first nvcc step), with -c/-o replaced by -cubin.
        "nvcc_flags": ["-O2"], "defines": ["-Dmain=transpose_sample_main_disabled"],
        "regimes": {"small": 1048576, "medium": 4194304, "large": 67108864},   # n elements
        "candidates": {"c1": "copy", "c2": "copySharedMem"},
    },
    "alt_cuda_samples_transposefine": {
        "source_rel": TRANSPOSE_REL,
        "source_sha256": "ba300b6b5f4dbc17ec72bfd3d1ee082508632ec431fe46ba54e6e3ce05292a95",
        "include_rels": ["Common"],
        "nvcc_flags": ["-O2"], "defines": ["-Dmain=transpose_sample_main_disabled"],
        "regimes": {"small": 1024, "medium": 2048, "large": 8192},   # matrix edge n
        "candidates": {"c1": "transposeFineGrained", "c2": "transposeCoarseGrained"},
    },
}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def unseen_anchors(kernel_name: str, n: int, threads: int | None) -> list[str]:
    """Function-name substrings to isolate one kernel, in preference order (first one that matches
    exactly one function wins; a first-matching anchor with several matches refuses).

    reduce0/reduce1: templated on <T> only -> "reduce0If" (same rule build_cell_manifest.py uses for
    reduce2/reduce3). reduce6: templated on <T, blockSize, nIsPow2> -> anchor on blockSize AND the
    nIsPow2 flag the pinned dispatcher would pass for this n (all three regimes have n a power of two
    -> "Lb1E"). Itanium mangling of `bool true` is "Lb1E". The sibling (other nIsPow2) instantiation, if
    present, is reported as a sibling so a wrong isPow2 assumption is visible, not hidden.
    Non-templated __global__ functions (copy, copySharedMem, transpose*): "_Z<len><name>"."""
    if kernel_name in ("reduce0", "reduce1"):
        return [f"{kernel_name}If"]
    if kernel_name == "reduce6":
        return [f"reduce6IfLj{threads}ELb{1 if _is_pow2(n) else 0}E"]
    return [f"_Z{len(kernel_name)}{kernel_name}"]


def fixture_anchors(kernel_name: str) -> list[str]:
    """Global-scope __global__ -> "_Z<len><name>"; inside an anonymous namespace (the
    predictor_*.cu fixtures) -> "<len><name>E" within a `_ZN..._GLOBAL__N__...` name."""
    return [f"_Z{len(kernel_name)}{kernel_name}", f"{len(kernel_name)}{kernel_name}E"]


# Calibration fixtures. Recipe origin: energy_harness/measurement_runner.py FIXTURES + build_argv()
# (`nvcc -O3 -std=c++17 -arch=<arch> [-DPREDICTOR_ENERGY] -DGPU_ID=<gpu> <src> -o <bin> -lnvidia-ml
# [-Xcompiler -pthread]`) with `-o <bin> -lnvidia-ml -Xcompiler -pthread` replaced by `-cubin -o <cubin>`
# (no host compile or link happens for -cubin, so those host-only flags cannot change device code:
# inferred, not checked). vector_width uses energy_harness/run_blackwell_eps_identification.sh's recipe
# (`-DPREDICTOR_ENERGY -DGPU_ID=0 -DNVML_GPU_ID=<gpu>`), the build whose SASS is committed.
# sha256 values are the working-tree hashes at authoring time (2026-09-29, repo HEAD 20ec7438).
FIXTURE_FAMILIES = {
    "calib_triad_stabilization": {
        "source_rel": "energy_harness/triad_stabilization.cu",
        "source_sha256": "5042c8ef061d26f8f94a1406f803464ad42d744c5b7e8741fd6914a3e3703a60",
        "defines": ["-DGPU_ID={gpu}"], "kernels": {"triad": None},
        "role": "the source the calibration suite's triad fixture builds (measurement_runner FIXTURES['triad'])",
    },
    "calib_triad_workloads_source": {
        "source_rel": "workloads/e2e_tile_triad_calibration.cu",
        "source_sha256": "acb59dee00f8a51ed21c2f99657cbe9d284bbb619e20b060c7988fb380ab2ad0",
        "defines": ["-DGPU_ID={gpu}"], "kernels": {"triad": None},
        "role": "the workloads/ triad named in the request; NOT what run_calibration_suite builds "
                "(recipe assumed equal to the triad recipe in build_argv; flagged in handoff)",
    },
    "calib_transpose": {
        "source_rel": "workloads/e2e_transpose.cu",
        "source_sha256": "380f3980996c4e5eb46f1c18ee180bfa829a4cad16241518a0d5a728f9b68455",
        "defines": ["-DPREDICTOR_ENERGY", "-DGPU_ID={gpu}"], "kernels": {"transpose_tiled": None},
        "role": "calibration suite fixture 'transpose' (TILE_DIM default 32)",
    },
    "calib_shared_stage": {
        "source_rel": "workloads/predictor_shared_stage_load.cu",
        "source_sha256": "a91f540b00bc2e1f4543d23934c2b41f2b6457020d843c49a89be68a7ec776c5",
        "defines": ["-DPREDICTOR_ENERGY", "-DGPU_ID={gpu}"], "kernels": {"shared_stage_kernel": None},
        "role": "calibration suite fixture 'shared_stage'",
    },
    "calib_store": {
        "source_rel": "workloads/predictor_store.cu",
        "source_sha256": "aac0fc23836798ad0cb1a28fbea1eb64e86cb184b9bbc336d23b80757e2e5b09",
        "defines": ["-DPREDICTOR_ENERGY", "-DGPU_ID={gpu}"], "kernels": {"store_kernel": None},
        "role": "calibration suite fixture 'store'",
    },
    "calib_vector_width": {
        "source_rel": "workloads/predictor_vector_width.cu",
        "source_sha256": "470e26653543fa8de626ca6217716f1d135ca9b770e51f7cdf989816c10e6db3",
        "defines": ["-DPREDICTOR_ENERGY", "-DGPU_ID=0", "-DNVML_GPU_ID={gpu}"],
        # kernel name -> load width (bytes) the SOURCE declares by element type (float / float2 / float4)
        "kernels": {"vector_width_kernel4": 4, "vector_width_kernel8": 8, "vector_width_kernel16": 16},
        "role": "load-width microbenchmark: the energy-identification kernels (4/8/16-byte loads)",
    },
}
FIXTURE_SUPPORT_FILES = {  # headers the fixtures include by relative path; staged next to the sources
    "energy_harness/raw_trace_recorder.h": "93f781e520376f020abb30bb788bde11ad94bd9838e6b4f7d9e3e780627a95c3",
}

CATALOG_KINDS = ("triton", "cuda_samples")

PYTORCH_PARENTS = {
    "dev_pytorch_rowwise_softmax": "PyTorch row-wise softmax",
    "final_pytorch_layer_norm": "PyTorch layer norm",
    "final_pytorch_embedding": "PyTorch embedding",
}
OTHER_CATALOG_PARENTS_NOT_IN_SCOPE = [
    "train_p1_shared_generator", "train_segmented_load", "train_multistream_load", "train_shared_stage",
    "train_cub_block_reduce", "final_cub_device_reduce", "final_xformers_indexed_select",
    "dev_local_triad", "dev_local_transpose",
]


def static_not_covered() -> list[dict]:
    out = []
    for pid, label in PYTORCH_PARENTS.items():
        out.append({"family": pid, "label": label, "status": "not_covered",
                    "reason": "needs cuobjdump of libtorch_cuda symbols: ATen kernels live in the torch "
                              "library's fatbins, and which symbol / template instantiation runs for a "
                              "given shape needs the installed torch and a symbol-level trace; not done here"})
    for pid in OTHER_CATALOG_PARENTS_NOT_IN_SCOPE:
        out.append({"family": pid, "status": "not_covered",
                    "reason": "outside this script's requested scope (emit_sass.py does not cover it "
                              "either); no compile recipe was decided here"})
    out.append({"family": "calibration:reduction (workloads/predictor_reduction.cu)", "status": "not_covered",
                "reason": "excluded from the calibration suite (energy_harness/run_calibration_suite.py "
                          "FIXTURES_WITHOUT_GRAPH_BATCH refusal set) pending a team decision; not requested"})
    return out


# --------------------------------------------------------------------------------------------
# Isolation, compile, per-cell record
# --------------------------------------------------------------------------------------------

def isolate_with_anchors(sass_text: str, anchors: list[str]) -> tuple[str, str, str, list[str]]:
    """(function_name, function_sass, anchor_used, all_function_names). The first anchor with any
    match must match exactly one function (else ValueError via smc.isolate_function_section)."""
    names = sorted(smc.function_sections(sass_text))
    for anchor in anchors:
        if any(anchor in n for n in names):
            fn_name, fn_sass = smc.isolate_function_section(sass_text, anchor)
            return fn_name, fn_sass, anchor, names
    raise EmitBlocked(f"EMIT_BLOCKED: no SASS function matched any anchor {anchors!r}. "
                      f"All function names seen: {names}")


def sha256_file(path: Path) -> str:
    return es.sha256_file(path)


def make_cell_record(*, sass_text: str, anchors: list[str], extra: dict) -> tuple[dict, dict]:
    """Return (cell_record, static_features_or_empty). `static_*` fields come from
    sass_mix_classifier.sass_instruction_mix over the SAME isolated function text the existing
    pipeline uses."""
    try:
        fn_name, fn_sass, anchor_used, names = isolate_with_anchors(sass_text, anchors)
    except ValueError as exc:
        raise EmitBlocked(f"EMIT_BLOCKED: function isolation failed: {exc}") from exc
    static = smc.sass_instruction_mix(fn_sass)
    widths = parse_sass_widths(fn_sass)
    flags = list(widths["flags"])
    if widths["has_unknown_width_global_load"]:
        flags.append("unknown_width_global_load")
    # Same template family, different template arguments (e.g. the other nIsPow2 instantiation of
    # reduce6): reported so a wrong instantiation assumption is visible rather than hidden.
    prefix = fn_name.split("Lb")[0] if "Lb" in fn_name else None
    sibling = [n for n in names if prefix and n != fn_name and n.startswith(prefix)]
    record = {
        "isolated_function_section": fn_name,
        "isolation_anchor": anchor_used,
        "sibling_instantiations_same_template_prefix": sibling,
        "function_names_in_cubin": names,
        "static_mix": static,
        "widths": widths,
        "flags": sorted(set(flags)),
        **extra,
    }
    return record, static


def cuobjdump_sass(cuobjdump: str, cubin: Path) -> str:
    return es.dump_sass(cuobjdump, cubin)


def _run_nvcc(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise EmitBlocked(f"EMIT_BLOCKED: nvcc -cubin failed: {' '.join(cmd)}\n{result.stderr[-4000:]}")


def process_catalog_family(parent_id: str, family: dict, args, tool: dict) -> tuple[dict, dict]:
    """Catalog cells on emit_sass's own compile paths. Returns (family_record, static_features)."""
    cells_out: dict[str, dict] = {}
    static_out: dict[str, dict] = {}
    kind = family["kind"]
    cubin_dir = args.out_dir / "cubins"
    if kind == "triton":
        kernel_name = family["kernel_name"]
        if es.TRITON_KERNEL_NAME.get(parent_id) != kernel_name:
            raise EmitBlocked(f"EMIT_BLOCKED: manifest kernel_name {kernel_name!r} for {parent_id} does not "
                              f"match emit_sass's hardcoded TRITON_KERNEL_NAME mapping")
        source_file = args.triton_source_root / family["source_rel"]
        if not source_file.is_file():
            raise EmitBlocked(f"EMIT_BLOCKED: missing pinned Triton source {source_file}")
        source_sha256 = sha256_file(source_file)
        if source_sha256 != family["source_sha256"]:
            raise EmitBlocked(f"EMIT_BLOCKED: pinned Triton source sha256 mismatch for {parent_id}: expected "
                              f"{family['source_sha256']}, got {source_sha256}")
        kernel = es.import_pinned_triton_kernel(source_file, kernel_name)
        for key, cell in family["cells"].items():
            cubin, info = es.compile_triton_cubin(kernel, kernel_name, cell)
            cubin_path = cubin_dir / f"{parent_id}_{key.replace('/', '_')}.cubin"
            cubin_path.write_bytes(cubin)
            sass = cuobjdump_sass(args.cuobjdump, cubin_path)
            rec, static = make_cell_record(
                sass_text=sass, anchors=[kernel_name],
                extra={"compiler": "triton", "compiler_version": tool["triton_version"],
                       "cubin_sha256": es.sha256_bytes(cubin), "source_revision": family["source_revision"],
                       "source_sha256": source_sha256, **info})
            cells_out[key] = rec
            static_out[key] = static
        source_info = {"source_rel": family["source_rel"], "source_sha256": source_sha256,
                       "source_revision": family["source_revision"]}
    elif kind == "cuda_samples":
        source_file = args.cuda_samples_root / family["source_rel"]
        if not source_file.is_file():
            raise EmitBlocked(f"EMIT_BLOCKED: missing pinned CUDA Samples source {source_file}")
        source_sha256 = sha256_file(source_file)
        if source_sha256 != family["source_sha256"]:
            raise EmitBlocked(f"EMIT_BLOCKED: pinned CUDA Samples source sha256 mismatch for {parent_id}: "
                              f"expected {family['source_sha256']}, got {source_sha256}")
        include_dirs = [args.cuda_samples_root / rel for rel in family.get("include_rels", [])]
        for d in include_dirs:
            if not d.is_dir():
                raise EmitBlocked(f"EMIT_BLOCKED: missing include dir {d}")
        # emit_sass compiles this same translation unit once per cell with identical flags; the cubin
        # does not depend on the cell, so compile once and reuse (cubin sha256 is recorded per cell).
        cubin_path = cubin_dir / f"{parent_id}.cubin"
        es.compile_cuda_samples_cubin(source_file, args.arch, args.nvcc, include_dirs, cubin_path)
        sass = cuobjdump_sass(args.cuobjdump, cubin_path)
        cubin_sha = sha256_file(cubin_path)
        for key, cell in family["cells"].items():
            rec, static = make_cell_record(
                sass_text=sass, anchors=[cell["entry_substring"]],
                extra={"compiler": "nvcc", "compiler_version": tool["nvcc_version"], "cubin_sha256": cubin_sha,
                       "source_revision": family["source_revision"], "source_sha256": source_sha256,
                       "compile_flags": ["-O3", "-std=c++17", f"-arch={args.arch}", "-cubin"]})
            cells_out[key] = rec
            static_out[key] = static
        source_info = {"source_rel": family["source_rel"], "source_sha256": source_sha256,
                       "source_revision": family["source_revision"]}
    else:
        raise EmitBlocked(f"EMIT_BLOCKED: unknown family kind {kind!r} for {parent_id}")
    return {"group": "catalog", "kind": kind, "source": source_info, "cells": cells_out}, static_out


def process_unseen_family(parent_id: str, spec: dict, args, tool: dict, cache: dict) -> dict:
    source_file = args.cuda_samples_root / spec["source_rel"]
    if not source_file.is_file():
        raise EmitBlocked(f"EMIT_BLOCKED: missing pinned CUDA Samples source {source_file}")
    source_sha256 = sha256_file(source_file)
    if source_sha256 != spec["source_sha256"]:
        raise EmitBlocked(f"EMIT_BLOCKED: pinned CUDA Samples source sha256 mismatch for {parent_id}: expected "
                          f"{spec['source_sha256']}, got {source_sha256}")
    include_dirs = [args.cuda_samples_root / rel for rel in spec["include_rels"]]
    for d in include_dirs:
        if not d.is_dir():
            raise EmitBlocked(f"EMIT_BLOCKED: missing include dir {d}")
    cmd_prefix = [args.nvcc, *spec["nvcc_flags"], f"-arch={args.arch}", *spec["defines"]]
    for d in include_dirs:
        cmd_prefix += ["-I", str(d)]
    cache_key = (spec["source_rel"], tuple(cmd_prefix))
    if cache_key not in cache:
        cubin_path = args.out_dir / "cubins" / f"unseen_{Path(spec['source_rel']).stem}_{len(cache)}.cubin"
        _run_nvcc([*cmd_prefix, "-cubin", str(source_file), "-o", str(cubin_path)])
        cache[cache_key] = (cubin_path, cuobjdump_sass(args.cuobjdump, cubin_path), sha256_file(cubin_path))
    cubin_path, sass, cubin_sha = cache[cache_key]
    cells: dict[str, dict] = {}
    for regime, n in spec["regimes"].items():
        for cand, kernel_name in spec["candidates"].items():
            anchors = unseen_anchors(kernel_name, n, spec.get("threads_per_block"))
            rec, _ = make_cell_record(
                sass_text=sass, anchors=anchors,
                extra={"kernel_symbol_requested": kernel_name, "regime_size_n": n,
                       "compiler": "nvcc", "compiler_version": tool["nvcc_version"], "cubin_sha256": cubin_sha,
                       "source_revision": PINNED_REVISION, "source_sha256": source_sha256,
                       "compile_argv_prefix": cmd_prefix[1:]})
            cells[f"{regime}/{cand}"] = rec
    return {"group": "unseen_operator", "kind": "cuda_samples_unseen",
            "source": {"source_rel": spec["source_rel"], "source_sha256": source_sha256,
                       "source_revision": PINNED_REVISION},
            "note": "compile recipe copied from the runner that produces the measured binary "
                    "(-O2, sample main renamed / per-thread default stream), -c/-o replaced by -cubin",
            "cells": cells}


def process_fixture_family(family_id: str, spec: dict, args, tool: dict) -> dict:
    repo = args.repo_root
    if repo is None:
        raise EmitBlocked(f"EMIT_BLOCKED: --repo-root is required for calibration fixture {family_id}")
    source_file = repo / spec["source_rel"]
    if not source_file.is_file():
        raise EmitBlocked(f"EMIT_BLOCKED: missing fixture source {source_file}")
    source_sha256 = sha256_file(source_file)
    drift = []
    if source_sha256 != spec["source_sha256"]:
        if not args.allow_fixture_hash_drift:
            raise EmitBlocked(f"EMIT_BLOCKED: fixture source sha256 differs from the authoring snapshot for "
                              f"{family_id}: expected {spec['source_sha256']}, got {source_sha256}. Pass "
                              f"--allow-fixture-hash-drift only if the change is understood; it is recorded.")
        drift.append(spec["source_rel"])
    for rel, expected in FIXTURE_SUPPORT_FILES.items():
        f = repo / rel
        if not f.is_file():
            raise EmitBlocked(f"EMIT_BLOCKED: missing fixture support header {f}")
        if sha256_file(f) != expected:
            if not args.allow_fixture_hash_drift:
                raise EmitBlocked(f"EMIT_BLOCKED: support header {rel} differs from the authoring snapshot")
            drift.append(rel)
    gpu = args.expect_visible_devices
    if not re.fullmatch(r"\d+", gpu or ""):
        raise EmitBlocked(f"EMIT_BLOCKED: --expect-visible-devices must be one GPU index to fill -DGPU_ID, got {gpu!r}")
    defines = [d.format(gpu=gpu) for d in spec["defines"]]
    cmd_prefix = [args.nvcc, "-O3", "-std=c++17", f"-arch={args.arch}", *defines]
    cubin_path = args.out_dir / "cubins" / f"{family_id}.cubin"
    _run_nvcc([*cmd_prefix, "-cubin", str(source_file), "-o", str(cubin_path)])
    sass = cuobjdump_sass(args.cuobjdump, cubin_path)
    cubin_sha = sha256_file(cubin_path)
    cells: dict[str, dict] = {}
    for kernel_name, designed_width in spec["kernels"].items():
        rec, _ = make_cell_record(
            sass_text=sass, anchors=fixture_anchors(kernel_name),
            extra={"kernel_symbol_requested": kernel_name, "compiler": "nvcc",
                   "compiler_version": tool["nvcc_version"], "cubin_sha256": cubin_sha,
                   "source_sha256": source_sha256, "compile_argv_prefix": cmd_prefix[1:],
                   "designed_load_width_bytes": designed_width})
        if designed_width is not None:
            got = rec["widths"]["widest_ldg_only_bytes"]
            if got != designed_width or rec["widths"]["widest_ldg_only_is_lower_bound"]:
                rec["flags"] = sorted(set(rec["flags"]) | {"designed_width_mismatch"})
        cells[f"fixture/{kernel_name}"] = rec
    return {"group": "calibration_fixture", "kind": "nvcc_cubin", "role": spec["role"],
            "source": {"source_rel": spec["source_rel"], "source_sha256": source_sha256,
                       "authoring_snapshot_sha256": spec["source_sha256"],
                       "hash_drift_files": drift},
            "cells": cells}


# --------------------------------------------------------------------------------------------
# Reproduction gate and committed-SASS cross-check
# --------------------------------------------------------------------------------------------

STATIC_FIELDS = ("static_total_instructions", "static_unique_opcodes", "static_load_count", "static_store_count",
                 "static_branch_count", "static_arith_count", "static_branch_density", "static_arith_density",
                 "static_load_density", "static_store_density")


def reproduction_gate(computed: dict, reference: dict, requested: list[str], reference_path: str) -> dict:
    """Compare recomputed static_* against a committed sass_features_sm_XXX.json. Ints must be equal;
    floats must agree to rel 1e-9 (the tolerance build_and_score.py's gate uses). A reference cell of a
    requested catalog family that was not recomputed is a failure, so a partial run cannot pass."""
    mismatches: list[dict] = []
    compared = 0
    checked_cells = 0
    for parent, ref_cells in reference.items():
        if parent not in requested:
            continue
        for cell_key, ref_feat in ref_cells.items():
            got_feat = computed.get(parent, {}).get(cell_key)
            if got_feat is None:
                mismatches.append({"parent": parent, "cell": cell_key, "field": "*",
                                   "reference": "present", "recomputed": "missing"})
                continue
            checked_cells += 1
            for field in sorted(set(ref_feat) | set(got_feat)):
                if field not in ref_feat or field not in got_feat:
                    mismatches.append({"parent": parent, "cell": cell_key, "field": field,
                                       "reference": ref_feat.get(field, "absent"),
                                       "recomputed": got_feat.get(field, "absent")})
                    continue
                a, b = ref_feat[field], got_feat[field]
                compared += 1
                if isinstance(a, float) or isinstance(b, float):
                    same = math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
                else:
                    same = a == b
                if not same:
                    mismatches.append({"parent": parent, "cell": cell_key, "field": field,
                                       "reference": a, "recomputed": b})
    parents_not_in_ref = [p for p in requested if p not in reference]
    return {
        "reference_path": reference_path, "status": "FAILED" if mismatches else "PASSED",
        "cells_compared": checked_cells, "field_comparisons": compared,
        "mismatches": mismatches, "requested_catalog_parents_absent_from_reference": parents_not_in_ref,
        "reference_parents_not_requested": sorted(p for p in reference if p not in requested),
    }


def crosscheck_committed_sass(fresh_cells: dict, committed_text: str, path: str) -> dict:
    """Informational (non-fatal): re-parse a committed real `cuobjdump --dump-sass` dump of the
    vector-width executable and compare each kernel's parsed widths with the fresh cubin's. A
    difference is expected if the nvcc release differs from the committed build's."""
    results = {}
    for cell_key, rec in fresh_cells.items():
        kernel = rec.get("kernel_symbol_requested")
        try:
            _, sec, _, _ = isolate_with_anchors(committed_text, fixture_anchors(kernel))
            committed = parse_sass_widths(sec)
        except (EmitBlocked, ValueError) as exc:
            results[cell_key] = {"match": False, "reason": f"could not isolate in committed dump: {exc}"}
            continue
        diff = [k for k in ("memory", "mufu", "shfl", "bar", "atomics", "total_instructions",
                            "widest_global_load_bytes")
                if committed[k] != rec["widths"][k]]
        results[cell_key] = {"match": not diff, "differing_fields": diff,
                             "committed_total_instructions": committed["total_instructions"],
                             "fresh_total_instructions": rec["widths"]["total_instructions"]}
    return {"committed_sass_path": path, "results": results,
            "all_match": all(r["match"] for r in results.values())}


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def all_family_ids(catalog_manifest: dict | None) -> list[str]:
    ids = list(catalog_manifest or {}) + list(UNSEEN_PARENTS) + list(FIXTURE_FAMILIES)
    return ids


def summarize(families: dict) -> dict:
    flagged, unknown_tokens = [], collections.Counter()
    n_cells = 0
    for fam_id, fam in families.items():
        for key, rec in fam["cells"].items():
            n_cells += 1
            if rec["flags"]:
                flagged.append({"family": fam_id, "cell": key, "flags": rec["flags"]})
            unknown_tokens.update(rec["widths"]["unknown_width_modifiers"])
    return {"cells": n_cells, "flagged_cells": flagged, "unknown_modifier_tokens": dict(unknown_tokens)}


def _probe(fn, tool: str) -> str:
    """Version probe that records 'unavailable' instead of crashing (a Triton-only run may not have
    nvcc on PATH); any family that really needs the tool still fails loudly at compile time."""
    try:
        return fn(tool)
    except OSError as exc:
        return f"unavailable: {exc}"


def toolchain(args) -> dict:
    return {
        "nvcc_version": _probe(es.nvcc_version, args.nvcc) if args.nvcc else None,
        "cuobjdump_version": _probe(es.cuobjdump_version, args.cuobjdump),
        "triton_version": es._triton_version(),
        "python": platform.python_version(),
        "emit_widths_sha256": sha256_file(Path(__file__).resolve()),
        "emit_sass_sha256": sha256_file(Path(es.__file__).resolve()),
        "sass_mix_classifier_sha256": sha256_file(Path(smc.__file__).resolve()),
    }


def run(args) -> int:
    es.check_visible_devices(args.expect_visible_devices)
    catalog_manifest = json.loads(args.cells_manifest.read_text()) if args.cells_manifest else None
    if args.families:
        requested = [f for f in args.families.split(",") if f]
    else:
        requested = all_family_ids(catalog_manifest)
    known = set(all_family_ids(catalog_manifest))
    unknown = [f for f in requested if f not in known]
    if unknown:
        raise EmitBlocked(f"EMIT_BLOCKED: --families named unknown families: {unknown}")
    requested_catalog = [f for f in requested if catalog_manifest and f in catalog_manifest]
    if args.reproduce_against and not requested_catalog:
        raise EmitBlocked("EMIT_BLOCKED: --reproduce-against needs at least one catalog family in --families")
    if requested_catalog and args.cells_manifest is None:
        raise EmitBlocked("EMIT_BLOCKED: catalog families need --cells-manifest")

    device_info = {}
    if any(catalog_manifest[p]["kind"] == "triton" for p in requested_catalog):
        device_info = es.check_device_arch(args.arch)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cubins").mkdir(exist_ok=True)
    tool = toolchain(args)
    families: dict[str, dict] = {}
    static_features: dict[str, dict] = {}
    family_errors: dict[str, str] = {}
    unseen_cache: dict = {}
    for fam_id in requested:
        try:
            if catalog_manifest and fam_id in catalog_manifest:
                families[fam_id], static_features[fam_id] = process_catalog_family(
                    fam_id, catalog_manifest[fam_id], args, tool)
            elif fam_id in UNSEEN_PARENTS:
                if args.cuda_samples_root is None:
                    raise EmitBlocked(f"EMIT_BLOCKED: {fam_id} needs --cuda-samples-root")
                families[fam_id] = process_unseen_family(fam_id, UNSEEN_PARENTS[fam_id], args, tool, unseen_cache)
            else:
                families[fam_id] = process_fixture_family(fam_id, FIXTURE_FAMILIES[fam_id], args, tool)
        except EmitBlocked as exc:
            if not args.continue_on_family_error:
                raise
            family_errors[fam_id] = str(exc)

    repro = None
    if args.reproduce_against:
        gates = []
        for ref_path in args.reproduce_against:
            gate = reproduction_gate(static_features, json.loads(Path(ref_path).read_text()),
                                     [f for f in requested_catalog if f in static_features], str(ref_path))
            gates.append(gate)
        errored = sorted(f for f in family_errors if f in requested_catalog)
        repro = {"status": "PASSED" if all(g["status"] == "PASSED" for g in gates) and not errored else "FAILED",
                 "family_errors_among_requested": errored, "references": gates}

    crosscheck = None
    if args.committed_vector_width_sass and "calib_vector_width" in families:
        crosscheck = crosscheck_committed_sass(
            families["calib_vector_width"]["cells"],
            args.committed_vector_width_sass.read_text(errors="replace"), str(args.committed_vector_width_sass))

    not_covered = static_not_covered()
    covered = set(families) | set(family_errors)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "arch": args.arch,
        "expect_visible_devices": args.expect_visible_devices,
        "live_device": device_info or None,
        "toolchain": tool,
        "parser": {
            "parser_version": PARSER_VERSION,
            "width_bytes": WIDTH_BYTES, "neutral_modifiers": sorted(NEUTRAL_MODIFIERS),
            "non_width_modifiers": sorted(NON_WIDTH_MODIFIERS),
            "width_tokens_seen_in_committed_real_sass": sorted(WIDTH_SEEN_IN_REAL_SASS),
            "non_width_tokens_seen_in_committed_real_sass": sorted(NON_WIDTH_SEEN_IN_REAL_SASS),
            "default_bytes_when_no_width_modifier": 4,
            "memory_class_by_opcode": MEMORY_CLASS_BY_OPCODE,
        },
        "requested_families": requested,
        "families": families,
        "family_errors": family_errors,
        "not_covered": [n for n in not_covered if n["family"] not in covered],
        "reproduction": repro,
        "committed_sass_crosscheck": crosscheck,
        "summary": summarize(families),
        "widths_usable": (repro is None or repro["status"] == "PASSED") and not family_errors,
        "widths_usable_note": ("false when the reproduction gate failed or any requested family errored; "
                               "when no --reproduce-against was given, usability is unchecked"),
    }
    out_path = args.out_dir / f"emit_widths_{args.arch}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    static_path = args.out_dir / f"emit_widths_static_features_{args.arch}.json"
    static_path.write_text(json.dumps(static_features, indent=2, sort_keys=True) + "\n")
    n_cells = result["summary"]["cells"]
    print(f"wrote {out_path} ({len(families)} families, {n_cells} cells, "
          f"{len(result['summary']['flagged_cells'])} flagged)")
    print(f"wrote {static_path} ({len(static_features)} catalog families)")
    if result["summary"]["unknown_modifier_tokens"]:
        print(f"UNKNOWN WIDTH MODIFIERS SEEN: {result['summary']['unknown_modifier_tokens']}", file=sys.stderr)
    if repro is not None:
        for g in repro["references"]:
            print(f"reproduction gate vs {g['reference_path']}: {g['status']} ({g['cells_compared']} cells, "
                  f"{g['field_comparisons']} fields)")
            for mm in g["mismatches"][:50]:
                print(f"REPRODUCTION_MISMATCH cell={mm['parent']}:{mm['cell']} field={mm['field']} "
                      f"reference={mm['reference']!r} recomputed={mm['recomputed']!r}", file=sys.stderr)
        for f in repro["family_errors_among_requested"]:
            print(f"REPRODUCTION_FAMILY_ERROR {f}: {family_errors[f][:300]}", file=sys.stderr)
        if repro["status"] != "PASSED":
            return EXIT_REPRO_FAILED
    if family_errors:
        for f, msg in family_errors.items():
            print(f"FAMILY_ERROR {f}: {msg[:500]}", file=sys.stderr)
        return EXIT_FAMILY_ERRORS
    return 0


# --------------------------------------------------------------------------------------------
# Self-test (CPU only; no GPU, nvcc, cuobjdump, Triton or torch)
# --------------------------------------------------------------------------------------------

def _find_repo_root() -> Path | None:
    for p in [HERE, *HERE.parents]:
        if (p / "calibration" / "predictor").is_dir() and (p / "AGENTS.md").is_file():
            return p
    return None


def _line(addr: int, body: str) -> str:
    return f"        /*{addr:04x}*/                   {body}   /* 0x0000000000000000 */"


def selftest(verbose: bool = True) -> None:
    def note(msg: str) -> None:
        if verbose:
            print("  ok:", msg)

    def sass(*bodies: str) -> str:
        return "\n".join(_line(0x10 * i, b) for i, b in enumerate(bodies))

    def p(*bodies: str) -> dict:
        return parse_sass_widths(sass(*bodies))

    # ---- 1. width mapping on synthetic lines (spellings from the task brief) -------------------
    w = p("LDG.E.128 R4, desc[UR6][R2.64] ;", "LDG.E R0, desc[UR6][R2.64] ;", "LDG.E.64 R2, [R4.64] ;",
          "@P0 LDG.E.128 R8, desc[UR6][R2.64] ;", "LDG.E.U8 R0, [R2.64] ;", "LDG.E.S8 R0, [R2.64] ;",
          "LDG.E.U16 R0, [R2.64] ;", "LDG.E.S16 R0, [R2.64] ;", "LDG.E.32 R0, [R2.64] ;",
          "@!P1 LDG.E.128.STRONG.SM R8, desc[UR6][R2.64] ;", "@UP0 LDG.E.EL.128 R8, [R2.64] ;")
    assert w["memory"]["global_load"]["count"] == 11
    assert w["memory"]["global_load"]["bytes_hist"] == {"1": 2, "2": 2, "4": 2, "8": 1, "16": 4}, w["memory"]["global_load"]
    assert w["memory"]["global_load"]["default_width_used"] == 1  # only the bare LDG.E
    assert w["widest_global_load_bytes"] == 16 and w["flags"] == [] and not w["has_unknown_width_global_load"]
    note("LDG widths .E/.E.64/.E.128/.U8/.S8/.U16/.S16/.32, predicated forms, cache modifiers")

    w = p("STG.E.128 [R2.64], R4 ;", "STG.E [R2.64], R4 ;", "STG.E.64.STRONG.SYS [R2.64], R4 ;",
          "STG.E.U8 [R2.64], R4 ;", "STG.E.U16 [R2.64], R4 ;", "@P0 STG.E.EF.128 [R2.64], R4 ;")
    assert w["memory"]["global_store"]["bytes_hist"] == {"1": 1, "2": 1, "4": 1, "8": 1, "16": 2}, w["memory"]["global_store"]
    assert w["widest_global_store_bytes"] == 16
    note("STG widths")

    w = p("LDS R9, [R9] ;", "LDS.U R9, [R9] ;", "LDS.U.64 R8, [R9] ;", "LDS.128 R8, [R9] ;", "LDS.U8 R0, [R9] ;",
          "LDS.S16 R0, [R9] ;", "STS [R3], R0 ;", "STS.64 [R3], R0 ;", "STS.128 [R3], R4 ;", "STS.U8 [R3], R0 ;")
    assert w["memory"]["shared_load"]["bytes_hist"] == {"1": 1, "2": 1, "4": 2, "8": 1, "16": 1}, w["memory"]["shared_load"]
    assert w["memory"]["shared_store"]["bytes_hist"] == {"1": 1, "4": 1, "8": 1, "16": 1}, w["memory"]["shared_store"]
    assert w["widest_shared_load_bytes"] == 16 and w["widest_shared_store_bytes"] == 16
    note("LDS/STS widths, bare 'U' marker treated as neutral")

    w = p("LDC R1, c[0x0][0x37c] ;", "LDC.64 R6, c[0x0][0x390] ;", "LDC.U8 R0, c[0x0][0x3a8] ;",
          "LDCU UR5, c[0x0][0x3a0] ;", "LDCU.64 UR6, c[0x0][0x358] ;", "LDCU.128 UR8, c[0x0][0x370] ;",
          "ULDC UR4, c[0x0][0x28] ;", "ULDC.64 UR4, c[0x0][0x28] ;", "ULDC.S8 UR4, c[0x0][0x28] ;",
          "LDG.E R0, [R2.64] ;")
    assert w["memory"]["constant_load"]["count"] == 9 and w["memory"]["global_load"]["count"] == 1
    assert w["memory"]["constant_load"]["bytes_hist"] == {"1": 2, "4": 3, "8": 3, "16": 1}, w["memory"]["constant_load"]
    note("constant loads counted separately from global loads")

    w = p("LD.E.64 R2, [R4] ;", "LD.E R2, [R4] ;", "ST.E.128 [R4], R8 ;", "LDL R2, [R4] ;", "STL.64 [R4], R2 ;",
          "LDG.E.128 R4, [R2.64] ;")
    assert w["memory"]["generic_load"]["bytes_hist"] == {"4": 1, "8": 1}
    assert w["memory"]["generic_store"]["bytes_hist"] == {"16": 1}
    assert w["memory"]["local_load"]["count"] == 1 and w["memory"]["local_store"]["bytes_hist"] == {"8": 1}
    assert w["global_or_generic_load"] == {"count": 3, "bytes_hist": {"4": 1, "8": 1, "16": 1}}
    assert w["widest_global_load_bytes"] == 16 and w["widest_ldg_only_bytes"] == 16
    w2 = p("LD.E.64 R2, [R4] ;", "LDG.E R2, [R4.64] ;")
    assert w2["widest_global_load_bytes"] == 8 and w2["widest_ldg_only_bytes"] == 4
    note("generic LD/ST, local LDL/STL; widest counts LDG or LD; LDG-only widest kept separately")

    w = p("MUFU.EX2 R3, R2 ;", "MUFU.RSQ R3, R2 ;", "MUFU.RCP R0, R0 ;", "MUFU.EX2 R4, R2 ;",
          "SHFL.BFLY PT, R5, R4, 0x10, 0x1f ;", "SHFL.DOWN PT, R5, R4, 0x1, 0x1f ;", "SHFL.IDX PT, R5, R4, RZ, 0x1f ;",
          "SHFL.UP PT, R5, R4, 0x1, RZ ;", "BAR.SYNC.DEFER_BLOCKING 0x0 ;", "BAR.SYNC.DEFER_BLOCKING 0x0 ;",
          "BAR.ARV 0x1, R2 ;", "MEMBAR.SC.GPU ;", "ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU PT, R2, [R4.64], R5 ;",
          "ATOMG.E.ADD.STRONG.GPU PT, R2, [R4.64], R5 ;", "RED.E.ADD.F32.FTZ.RN.STRONG.GPU [R4.64], R5 ;",
          "ATOMS.ADD R2, [R4], R5 ;", "FFMA R1, R2, R3, R4 ;", "FFMA R1, R2, R3, R4 ;", "FMUL R1, R2, R3 ;",
          "FADD R1, R2, R3 ;", "FADD R1, R2, R3 ;", "FADD R1, R2, R3 ;", "EXIT ;")
    assert w["mufu"] == {"count": 4, "by_subop": {"EX2": 2, "RCP": 1, "RSQ": 1}}, w["mufu"]
    assert w["shfl"]["count"] == 4 and w["shfl"]["by_variant"] == {"BFLY": 1, "DOWN": 1, "IDX": 1, "UP": 1}
    assert w["bar"] == {"count": 3, "by_variant": {"ARV": 1, "SYNC.DEFER_BLOCKING": 2}}, w["bar"]
    assert w["fence"]["count"] == 1
    assert w["atomics"]["count"] == 4 and w["atomics"]["global_count"] == 3 and w["atomics"]["shared_count"] == 1
    assert (w["ffma"], w["fmul"], w["fadd"]) == (2, 1, 3)
    assert w["total_instructions"] == 23 and w["flags"] == []
    note("MUFU sub-ops, SHFL variants, BAR variants, MEMBAR, ATOM/ATOMG/RED/ATOMS, FFMA/FMUL/FADD, total")

    # ---- 2. unknown width modifiers land in the flagged bucket -------------------------------
    w = p("LDG.E.128 R4, [R2.64] ;", "LDG.E.FOO.64 R6, [R2.64] ;", "LDG.E.BAR R8, [R2.64] ;", "STG.E.WEIRD [R2.64], R4 ;")
    assert w["memory"]["global_load"]["bytes_hist"] == {"16": 1, "unknown_width": 2}, w["memory"]["global_load"]
    assert w["memory"]["global_store"]["bytes_hist"] == {"unknown_width": 1}
    assert "unknown_width_modifier" in w["flags"]
    assert w["unknown_width_modifiers"] == {"LDG.BAR": 1, "LDG.FOO": 1, "STG.WEIRD": 1}, w["unknown_width_modifiers"]
    assert w["has_unknown_width_global_load"] and w["widest_global_load_is_lower_bound"]
    assert w["widest_global_load_bytes"] == 16  # lower bound over the known widths only
    w = p("LDG.E.64.128 R4, [R2.64] ;")
    assert w["memory"]["global_load"]["bytes_hist"] == {"unknown_width": 1} and "conflicting_width_modifiers" in w["flags"]
    w = p("LDG.E.256 R4, [R2.64] ;")  # 256-bit spelling is deliberately NOT mapped: flagged, never guessed
    assert w["memory"]["global_load"]["bytes_hist"] == {"unknown_width": 1} and "unknown_width_modifier" in w["flags"]
    w = p("LDSM.16.M88.4 R4, [R2] ;", "LDGSTS.E.BYPASS.LTC128B.128 [R2], desc[UR6][R4.64] ;", "STSM.16.M88.4 [R2], R4 ;",
          "LDNEW R1, [R2] ;")
    assert w["memory"]["shared_matrix_load"]["count"] == 1 and w["memory"]["async_global_to_shared"]["bytes_hist"] == {"16": 1}
    assert w["unclassified_ld_st_opcodes"] == {"LDNEW": 1} and "unclassified_memory_opcode" in w["flags"]
    note("unknown / conflicting modifiers -> unknown_width bucket, flags set, widest is a lower bound; "
         "unclassified LD*/ST* opcode flagged")

    # 5-digit address and @PT predicate lines the existing classifier skips are counted and flagged
    text = "\n".join([_line(0x10, "LDG.E R0, [R2.64] ;"), "        /*10000*/                   FADD R1, R2, R3 ;",
                      _line(0x20, "@PT EXIT ;")])
    w = parse_sass_widths(text)
    assert w["total_instructions"] == 3 and w["classifier_instruction_lines"] == 1
    assert "classifier_line_count_mismatch" in w["flags"]
    note("lines the existing classifier skips (5-digit address, @PT) are counted and flagged")
    try:
        parse_sass_widths("no instructions here")
        raise AssertionError("empty SASS accepted")
    except ValueError:
        pass

    # ---- 3. static_mix equals the existing classifier's output (same isolated text) ----------
    fn = "Function : _Z3fooPf\n" + sass("LDG.E R2, [R4.64] ;", "FADD R5, R2, R2 ;", "STG.E [R8.64], R5 ;", "EXIT ;")
    rec, static = make_cell_record(sass_text=fn, anchors=["foo"], extra={})
    assert static == smc.sass_instruction_mix(fn.split("\n", 1)[1]) and rec["static_mix"] == static
    assert rec["isolated_function_section"] == "_Z3fooPf" and rec["isolation_anchor"] == "foo"
    note("static_mix is smc.sass_instruction_mix over the isolated section, unchanged")

    # ---- 4. isolation anchors (unseen parents, fixtures) ------------------------------------------
    multi = "\n".join([
        "Function : _Z7reduce0IfEvPT_S1_j", sass("LDG.E R2, [R4.64] ;", "EXIT ;"),
        "Function : _Z7reduce1IfEvPT_S1_j", sass("LDS R2, [R4] ;", "EXIT ;"),
        "Function : _Z7reduce6IfLj256ELb1EEvPT_S1_j", sass("LDG.E.64 R2, [R4.64] ;", "EXIT ;"),
        "Function : _Z7reduce6IfLj256ELb0EEvPT_S1_j", sass("LDG.E R2, [R4.64] ;", "LDG.E R3, [R6.64] ;", "EXIT ;"),
        "Function : _Z7reduce6IfLj512ELb1EEvPT_S1_j", sass("LDG.E R2, [R4.64] ;", "EXIT ;"),
        "Function : _Z4copyPfS_ii", sass("LDG.E R2, [R4.64] ;", "EXIT ;"),
        "Function : _Z13copySharedMemPfS_ii", sass("LDG.E R2, [R4.64] ;", "STS [R1], R2 ;", "EXIT ;"),
        "Function : _ZN54_GLOBAL__N__2a8f16e1_25_predictor_vector_width_cu_main20vector_width_kernel8EPK6float2Pjjjjjjfb",
        sass("LDG.E.64.STRONG.SM R2, desc[UR6][R4.64] ;", "EXIT ;"),
        "Function : _ZN54_GLOBAL__N__2a8f16e1_25_predictor_vector_width_cu_main21vector_width_kernel16EPK6float4Pjjjjjjfb",
        sass("LDG.E.128.STRONG.SM R4, desc[UR6][R4.64] ;", "EXIT ;"),
    ])
    assert unseen_anchors("reduce6", 65536, 256) == ["reduce6IfLj256ELb1E"]
    assert unseen_anchors("reduce6", 65535, 256) == ["reduce6IfLj256ELb0E"]
    assert fixture_anchors("vector_width_kernel16") == ["_Z21vector_width_kernel16", "21vector_width_kernel16E"]
    r6, _ = make_cell_record(sass_text=multi, anchors=unseen_anchors("reduce6", 65536, 256), extra={})
    assert r6["isolated_function_section"] == "_Z7reduce6IfLj256ELb1EEvPT_S1_j"
    assert r6["widths"]["widest_global_load_bytes"] == 8
    assert r6["sibling_instantiations_same_template_prefix"] == ["_Z7reduce6IfLj256ELb0EEvPT_S1_j"], r6["sibling_instantiations_same_template_prefix"]
    for k, want in (("reduce0", "_Z7reduce0IfEvPT_S1_j"), ("reduce1", "_Z7reduce1IfEvPT_S1_j")):
        assert make_cell_record(sass_text=multi, anchors=unseen_anchors(k, 65536, 256), extra={})[0][
            "isolated_function_section"] == want
    assert make_cell_record(sass_text=multi, anchors=unseen_anchors("copy", 1, None), extra={})[0][
        "isolated_function_section"] == "_Z4copyPfS_ii"
    assert make_cell_record(sass_text=multi, anchors=unseen_anchors("copySharedMem", 1, None), extra={})[0][
        "isolated_function_section"] == "_Z13copySharedMemPfS_ii"
    v16, _ = make_cell_record(sass_text=multi, anchors=fixture_anchors("vector_width_kernel16"), extra={})
    v8, _ = make_cell_record(sass_text=multi, anchors=fixture_anchors("vector_width_kernel8"), extra={})
    assert v16["widths"]["widest_global_load_bytes"] == 16 and v8["widths"]["widest_global_load_bytes"] == 8
    try:  # ambiguous anchor refuses
        make_cell_record(sass_text=multi, anchors=["reduce6IfLj256E"], extra={})
        raise AssertionError("ambiguous anchor accepted")
    except EmitBlocked as exc:
        assert "found 2" in str(exc)
    try:  # zero matches refuses
        make_cell_record(sass_text=multi, anchors=["nonexistent_kernel"], extra={})
        raise AssertionError("zero-match anchor accepted")
    except EmitBlocked as exc:
        assert "no SASS function matched" in str(exc)
    note("anchors: reduce0/1/6 (incl. nIsPow2 sibling reported), copy vs copySharedMem, namespaced fixtures; "
         "ambiguous and zero-match refuse")

    # ---- 5. reproduction gate: passes on identical input, fails loudly on a planted mismatch ----
    ref = {"parentA": {"small/c1": dict(smc.sass_instruction_mix(fn.split("\n", 1)[1]))}}
    same = json.loads(json.dumps(ref))
    r = reproduction_gate(same, ref, ["parentA"], "ref.json")
    assert r["status"] == "PASSED" and r["cells_compared"] == 1 and r["field_comparisons"] == 10 and not r["mismatches"]
    bad = json.loads(json.dumps(ref))
    bad["parentA"]["small/c1"]["static_load_count"] += 1
    r = reproduction_gate(bad, ref, ["parentA"], "ref.json")
    assert r["status"] == "FAILED" and r["mismatches"][0] == {
        "parent": "parentA", "cell": "small/c1", "field": "static_load_count",
        "reference": 1, "recomputed": 2}, r["mismatches"]
    bad = json.loads(json.dumps(ref))
    bad["parentA"]["small/c1"]["static_load_density"] += 1e-6
    assert reproduction_gate(bad, ref, ["parentA"], "ref.json")["status"] == "FAILED"
    near = json.loads(json.dumps(ref))
    near["parentA"]["small/c1"]["static_load_density"] *= 1 + 1e-12
    assert reproduction_gate(near, ref, ["parentA"], "ref.json")["status"] == "PASSED"
    r = reproduction_gate({"parentA": {}}, ref, ["parentA"], "ref.json")  # cell missing from recompute
    assert r["status"] == "FAILED" and r["mismatches"][0]["recomputed"] == "missing"
    r = reproduction_gate({"parentA": {"small/c1": {k: v for k, v in same["parentA"]["small/c1"].items()
                                                    if k != "static_store_count"}}}, ref, ["parentA"], "ref.json")
    assert r["status"] == "FAILED" and r["mismatches"][0]["field"] == "static_store_count"
    note("reproduction gate: identical passes; changed int, changed float, missing cell, missing field all fail "
         "and name cell and field")

    _selftest_pipeline(note)
    _selftest_embedded_tables(note)
    _selftest_real_sass(note)
    print("EMIT_WIDTHS_SELFTEST_OK: synthetic parser tests, unknown-width flagging, isolation anchors, "
          "reproduction gate, end-to-end pipeline with faked toolchain, embedded-table cross-check, "
          "real committed SASS fixtures (where present). NOTE: nothing here proves the parser matches real "
          "Ada / Blackwell SASS from the compilers used on the target; the first real check is the "
          "reproduction gate on a GPU machine.")


def _selftest_pipeline(note) -> None:
    """Whole run() with nvcc/cuobjdump/compile faked: catalog CUDA-Samples family (gate pass and planted
    failure through the real exit codes), an unseen parent, a fixture, the outputs on disk."""
    import tempfile
    import types
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "samples" / "Common").mkdir(parents=True)
        src = tmp / "samples" / "cpp" / "0_Introduction" / "vectorAdd"
        src.mkdir(parents=True)
        (src / "vectorAdd.cu").write_text("// fake\n")
        sha = sha256_file(src / "vectorAdd.cu")
        (tmp / "samples" / "cpp" / "2_Concepts_and_Techniques" / "reduction").mkdir(parents=True)
        red = tmp / "samples" / "cpp" / "2_Concepts_and_Techniques" / "reduction" / "reduction_kernel.cu"
        red.write_text("// fake reduction\n")
        tr = tmp / "samples" / "cpp" / "6_Performance" / "transpose"
        tr.mkdir(parents=True)
        (tr / "transpose.cu").write_text("// fake transpose\n")
        repo = tmp / "repo"
        (repo / "workloads").mkdir(parents=True)
        (repo / "energy_harness").mkdir()
        (repo / "workloads" / "predictor_vector_width.cu").write_text("// fake vw\n")
        (repo / "energy_harness" / "raw_trace_recorder.h").write_text("// fake header\n")
        manifest = {"final_cuda_samples_copy": {
            "kind": "cuda_samples", "include_rels": [], "source_rel": "cpp/0_Introduction/vectorAdd/vectorAdd.cu",
            "source_revision": "rev", "source_sha256": sha,
            "cells": {"small/c1": {"entry_substring": "vecAdd"}, "small/c2": {"entry_substring": "vecAdd"}}}}
        mpath = tmp / "cells.json"
        mpath.write_text(json.dumps(manifest))
        fake_sass = "\n".join([
            "Function : _Z6vecAddPKfS0_Pfi", "\n".join(
                _line(0x10 * i, b) for i, b in enumerate(
                    ["LDG.E R2, desc[UR6][R4.64] ;", "LDG.E R3, desc[UR6][R6.64] ;", "FADD R5, R2, R3 ;",
                     "STG.E desc[UR6][R8.64], R5 ;", "EXIT ;"])),
            "Function : _Z7reduce0IfEvPT_S1_j", "\n".join(_line(0x10 * i, b) for i, b in enumerate(
                ["LDG.E R2, [R4.64] ;", "LDS R3, [R5] ;", "STS [R5], R3 ;", "BAR.SYNC.DEFER_BLOCKING 0x0 ;", "EXIT ;"])),
            "Function : _Z7reduce1IfEvPT_S1_j", _line(0, "LDG.E.64 R2, [R4.64] ;"),
            "Function : _Z7reduce6IfLj256ELb1EEvPT_S1_j", _line(0, "LDG.E.128 R2, [R4.64] ;"),
            "Function : _Z4copyPfS_ii", _line(0, "LDG.E R2, [R4.64] ;"),
            "Function : _Z13copySharedMemPfS_ii", _line(0, "LDG.E R2, [R4.64] ;"),
            "Function : _Z20transposeFineGrainedPfS_ii", _line(0, "LDG.E.64 R2, [R4.64] ;"),
            "Function : _Z22transposeCoarseGrainedPfS_ii", _line(0, "LDG.E R2, [R4.64] ;"),
            "Function : _ZN54_GLOBAL__N__2a8f16e1_25_predictor_vector_width_cu_main20vector_width_kernel4EPKfPjjjjjjfb",
            _line(0, "LDG.E.STRONG.SM R2, desc[UR6][R4.64] ;"),
            "Function : _ZN54_GLOBAL__N__2a8f16e1_25_predictor_vector_width_cu_main20vector_width_kernel8EPK6float2Pjjjjjjfb",
            _line(0, "LDG.E.64.STRONG.SM R2, desc[UR6][R4.64] ;"),
            "Function : _ZN54_GLOBAL__N__2a8f16e1_25_predictor_vector_width_cu_main21vector_width_kernel16EPK6float4Pjjjjjjfb",
            _line(0, "LDG.E.128.STRONG.SM R2, desc[UR6][R4.64] ;"),
        ])
        # Fake pinned hashes for the unseen parents and the fixture (restored afterwards).
        saved_unseen = json.loads(json.dumps(UNSEEN_PARENTS))
        saved_fix = json.loads(json.dumps(FIXTURE_FAMILIES))
        saved_support = dict(FIXTURE_SUPPORT_FILES)
        saved_funcs = (es.compile_cuda_samples_cubin, es.dump_sass, es.nvcc_version, es.cuobjdump_version,
                       globals()["_run_nvcc"], es._triton_version)
        try:
            UNSEEN_PARENTS["alt_cuda_samples_reduction"]["source_sha256"] = sha256_file(red)
            for pid in ("alt_cuda_samples_copy", "alt_cuda_samples_transposefine"):
                UNSEEN_PARENTS[pid]["source_sha256"] = sha256_file(tr / "transpose.cu")
            for pid in [k for k in FIXTURE_FAMILIES if k != "calib_vector_width"]:
                del FIXTURE_FAMILIES[pid]
            FIXTURE_FAMILIES["calib_vector_width"]["source_sha256"] = sha256_file(repo / "workloads" / "predictor_vector_width.cu")
            FIXTURE_SUPPORT_FILES["energy_harness/raw_trace_recorder.h"] = sha256_file(repo / "energy_harness" / "raw_trace_recorder.h")

            def fake_compile(source_file, arch, nvcc, include_dirs, out_path):
                Path(out_path).write_bytes(b"CUBIN")

            def fake_run_nvcc(cmd):
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"CUBIN" + str(cmd).encode())

            es.compile_cuda_samples_cubin = fake_compile
            es.dump_sass = lambda cuobjdump, cubin: fake_sass
            es.nvcc_version = lambda nvcc: "nvcc: fake 0.0"
            es.cuobjdump_version = lambda c: "cuobjdump: fake 0.0"
            es._triton_version = lambda: "unavailable"
            globals()["_run_nvcc"] = fake_run_nvcc

            def make_args(out: Path, **kw):
                ns = dict(arch="sm_120", families="", cells_manifest=mpath, triton_source_root=None,
                          cuda_samples_root=tmp / "samples", repo_root=repo, nvcc="nvcc", cuobjdump="cuobjdump",
                          out_dir=out, expect_visible_devices="1", reproduce_against=None,
                          committed_vector_width_sass=None, allow_fixture_hash_drift=False,
                          continue_on_family_error=False)
                ns.update(kw)
                return types.SimpleNamespace(**ns)

            old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            try:
                out = tmp / "out1"
                assert run(make_args(out)) == 0
                res = json.loads((out / "emit_widths_sm_120.json").read_text())
                assert res["schema_version"] == SCHEMA_VERSION and res["arch"] == "sm_120"
                assert set(res["families"]) == {"final_cuda_samples_copy", *UNSEEN_PARENTS, "calib_vector_width"}
                assert len(res["families"]["alt_cuda_samples_reduction"]["cells"]) == 9
                assert len(res["families"]["alt_cuda_samples_copy"]["cells"]) == 6
                assert len(res["families"]["alt_cuda_samples_transposefine"]["cells"]) == 6
                vw = res["families"]["calib_vector_width"]["cells"]
                assert [vw[f"fixture/vector_width_kernel{k}"]["widths"]["widest_global_load_bytes"] for k in (4, 8, 16)] == [4, 8, 16]
                assert all(not c["flags"] for c in vw.values())
                assert res["families"]["alt_cuda_samples_reduction"]["cells"]["large/c3"]["widths"]["widest_global_load_bytes"] == 16
                assert res["summary"]["flagged_cells"] == [], res["summary"]["flagged_cells"]
                assert any(n["family"] == "final_pytorch_embedding" for n in res["not_covered"])
                assert res["reproduction"] is None and (out / "emit_widths_static_features_sm_120.json").is_file()
                static = json.loads((out / "emit_widths_static_features_sm_120.json").read_text())
                # gate: reference identical to the recomputed static features -> pass
                refp = tmp / "ref_ok.json"
                refp.write_text(json.dumps(static))
                assert run(make_args(tmp / "out2", families="final_cuda_samples_copy", reproduce_against=[refp])) == 0
                r2 = json.loads((tmp / "out2" / "emit_widths_sm_120.json").read_text())
                assert r2["reproduction"]["status"] == "PASSED" and r2["reproduction"]["references"][0]["cells_compared"] == 2
                assert r2["widths_usable"] is True
                # planted mismatch: exit 3, output still written, widths_usable false
                planted = json.loads(json.dumps(static))
                planted["final_cuda_samples_copy"]["small/c2"]["static_store_count"] += 1
                badp = tmp / "ref_bad.json"
                badp.write_text(json.dumps(planted))
                assert run(make_args(tmp / "out3", families="final_cuda_samples_copy", reproduce_against=[badp])) == EXIT_REPRO_FAILED
                r3 = json.loads((tmp / "out3" / "emit_widths_sm_120.json").read_text())
                assert r3["reproduction"]["status"] == "FAILED" and r3["widths_usable"] is False
                mm = r3["reproduction"]["references"][0]["mismatches"][0]
                assert mm["cell"] == "small/c2" and mm["field"] == "static_store_count", mm
                # two references, one good and one bad: overall FAILED, and the good one still reports PASSED
                assert run(make_args(tmp / "out3b", families="final_cuda_samples_copy",
                                     reproduce_against=[refp, badp])) == EXIT_REPRO_FAILED
                r3b = json.loads((tmp / "out3b" / "emit_widths_sm_120.json").read_text())
                assert [g["status"] for g in r3b["reproduction"]["references"]] == ["PASSED", "FAILED"]
                # wrong visible device refuses
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                try:
                    run(make_args(tmp / "out4"))
                    raise AssertionError("wrong CUDA_VISIBLE_DEVICES accepted")
                except EmitBlocked as exc:
                    assert "CUDA_VISIBLE_DEVICES must be exactly" in str(exc)
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                # source hash mismatch refuses
                src.joinpath("vectorAdd.cu").write_text("// tampered\n")
                try:
                    run(make_args(tmp / "out5", families="final_cuda_samples_copy"))
                    raise AssertionError("hash mismatch accepted")
                except EmitBlocked as exc:
                    assert "sha256 mismatch" in str(exc)
            finally:
                if old_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd
        finally:
            UNSEEN_PARENTS.clear(); UNSEEN_PARENTS.update(saved_unseen)
            FIXTURE_FAMILIES.clear(); FIXTURE_FAMILIES.update(saved_fix)
            FIXTURE_SUPPORT_FILES.clear(); FIXTURE_SUPPORT_FILES.update(saved_support)
            (es.compile_cuda_samples_cubin, es.dump_sass, es.nvcc_version, es.cuobjdump_version,
             globals()["_run_nvcc"], es._triton_version) = saved_funcs
    note("end-to-end run() with faked toolchain: families, cell counts, outputs, gate exit codes 0/3, "
         "wrong-device and source-hash refusals")


def _selftest_embedded_tables(note) -> None:
    root = _find_repo_root()
    if root is None:
        print("  skip: repo not present, embedded-table cross-check (adapters.json / fixture hashes)")
        return
    adapters = json.loads((root / "tiresias" / "framework" / "adapters" / "adapters.json").read_text())["adapters"]
    seen = set()
    for a in adapters:
        spec = UNSEEN_PARENTS[a["parent_id"]]
        seen.add(a["parent_id"])
        assert spec["source_rel"] == a["source"]["file"] and spec["source_sha256"] == a["source"]["sha256"], a["parent_id"]
        assert PINNED_REVISION == a["source"]["revision"]
        assert spec["regimes"] == {k: v["n"] for k, v in a["regimes"].items()}, a["parent_id"]
        assert spec["candidates"] == {k: v["kernel"] for k, v in a["candidates"].items()}, a["parent_id"]
        if "threads_per_block" in spec:
            assert spec["threads_per_block"] == a["threads_per_block"]
    assert seen == set(UNSEEN_PARENTS)
    for fam, spec in FIXTURE_FAMILIES.items():
        f = root / spec["source_rel"]
        if f.is_file():
            got = sha256_file(f)
            # Informational drift check: the hard-coded snapshot must match the checkout the test runs in.
            assert got == spec["source_sha256"], f"{fam}: snapshot hash {spec['source_sha256']} != checkout {got}"
            text = f.read_text()
            for k in spec["kernels"]:
                assert re.search(r"__global__\s+void\s+" + re.escape(k) + r"\s*\(", text), (fam, k)
    for rel, expected in FIXTURE_SUPPORT_FILES.items():
        assert sha256_file(root / rel) == expected
    cm = _ALAVANI_DIR / "cell_manifest_blackwell.json"
    if cm.is_file():
        m = json.loads(cm.read_text())
        assert all(v["kind"] in CATALOG_KINDS for v in m.values())
    note("embedded unseen-parent table equals adapters.json; fixture kernels exist in their sources; "
         "fixture/header sha256 snapshots equal this checkout")


def _selftest_real_sass(note) -> None:
    root = _find_repo_root()
    if root is None:
        print("  skip: repo not present, real committed SASS fixtures")
        return
    base = root / "calibration" / "predictor"
    # (a) Blackwell load-width microbenchmark, executable dump: three kernels, widths 4/8/16.
    vw = base / "blackwell_eps_ident_20260927_022819" / "raw" / "sass_vector_width_energy.txt"
    text = vw.read_text(errors="replace")
    per = {}
    for k, want in ((4, 4), (8, 8), (16, 16)):
        name, sec, anchor, names = isolate_with_anchors(text, fixture_anchors(f"vector_width_kernel{k}"))
        w = parse_sass_widths(sec)
        per[k] = w
        hist = w["memory"]["global_load"]["bytes_hist"]
        assert w["widest_global_load_bytes"] == want and w["widest_ldg_only_bytes"] == want, (k, hist)
        assert set(hist) == {str(want)}, (k, hist)               # every LDG in this kernel has exactly the designed width
        assert not w["has_unknown_width_global_load"] and not w["flags"], (k, w["flags"], w["unknown_width_modifiers"])
        assert w["memory"]["global_store"]["bytes_hist"] == {"4": 1}, w["memory"]["global_store"]
        # regex parity with the existing classifier on real SASS: same instruction count
        assert w["total_instructions"] == smc.sass_instruction_mix(sec)["static_total_instructions"]
        assert w["classifier_instruction_lines"] == w["total_instructions"]
        # the existing static_load_count (LDG+LDS+LDC+LDCU+LD) is reproduced from the width record
        m = w["memory"]
        static = smc.sass_instruction_mix(sec)
        assert static["static_load_count"] == (m["global_load"]["count"] + m["shared_load"]["count"] +
                                                 m["generic_load"]["count"] + sum(
            1 for o in re.findall(smc.SASS_LINE, sec) if o in ("LDC", "LDCU"))), k
        assert static["static_store_count"] == (m["global_store"]["count"] + m["shared_store"]["count"] +
                                                  m["generic_store"]["count"]), k
    assert per[4]["memory"]["global_load"]["count"] > 0 and per[8]["memory"]["global_load"]["count"] > 0
    assert per[16]["memory"]["global_load"]["count"] > 0
    # the fresh-vs-committed cross-check function agrees with itself on the committed dump
    fake_cells = {}
    for k in (4, 8, 16):
        _, sec, _, _ = isolate_with_anchors(text, fixture_anchors(f"vector_width_kernel{k}"))
        fake_cells[f"fixture/vector_width_kernel{k}"] = {"kernel_symbol_requested": f"vector_width_kernel{k}",
                                                         "widths": parse_sass_widths(sec)}
    cc = crosscheck_committed_sass(fake_cells, text, str(vw))
    assert cc["all_match"], cc
    perturbed = json.loads(json.dumps(fake_cells))
    perturbed["fixture/vector_width_kernel16"]["widths"]["memory"]["global_load"]["count"] += 1
    assert not crosscheck_committed_sass(perturbed, text, str(vw))["all_match"]
    note("real Blackwell vector-width SASS: kernel4/8/16 have only 4/8/16-byte LDG loads (widest 4/8/16), "
         "0 unknown, classifier parity, static_load/store_count recomposed from width record")

    # (b) Blackwell gather-width SASS (three more kernels, also 4/8/16 wide data loads + 4-byte index loads).
    gw = base / "blackwell_gather_width_20260905_022248" / "raw" / "sass.txt"
    gtext = gw.read_text(errors="replace")
    for k in (4, 8, 16):
        _, sec, _, _ = isolate_with_anchors(gtext, fixture_anchors(f"gather_width_kernel{k}"))
        w = parse_sass_widths(sec)
        assert w["widest_global_load_bytes"] == k and not w["flags"], (k, w["memory"]["global_load"], w["flags"])
        assert w["classifier_instruction_lines"] == w["total_instructions"]
    note("real Blackwell gather-width SASS: widest LDG = 4/8/16 for kernels 4/8/16, no flags")

    # (c) Ada (sm_89) SASS: different operand syntax ([R6.64], no desc[UR]); bare 32-bit loads.
    ada = base / "ada_tier_20260903_183259" / "raw" / "sass.txt"
    atext = ada.read_text(errors="replace")
    _, sec, _, _ = isolate_with_anchors(atext, fixture_anchors("tier_load_kernel"))
    w = parse_sass_widths(sec)
    assert w["widest_global_load_bytes"] == 4 and set(w["memory"]["global_load"]["bytes_hist"]) == {"4"}, w["memory"]["global_load"]
    assert not w["flags"] and w["classifier_instruction_lines"] == w["total_instructions"]
    note("real Ada sm_89 SASS (tier-load kernel): all LDG are 4-byte, no flags, classifier parity")

    # (d) Every real committed dump: parity with the classifier and no unclassified LD/ST opcode.
    n_files = n_funcs = 0
    unknown_seen = collections.Counter()
    for f in sorted(list(base.glob("*/raw/sass*.txt")) + list((root / "calibration").glob("b1_blackwell_*/*/sass.txt"))):
        t = f.read_text(errors="replace")
        for fname, sec in smc.function_sections(t).items():
            if not INSTR_LINE.search(sec):
                continue
            w = parse_sass_widths(sec)
            n_funcs += 1
            assert w["classifier_instruction_lines"] == w["total_instructions"], (f, fname)
            assert not w["unclassified_ld_st_opcodes"], (f, fname, w["unclassified_ld_st_opcodes"])
            unknown_seen.update(w["unknown_width_modifiers"])
        n_files += 1
    assert not unknown_seen, f"modifiers present in committed real SASS but not in the mapping: {dict(unknown_seen)}"
    note(f"all {n_funcs} function sections in {n_files} committed real SASS dumps parse with zero unknown "
         f"width modifiers, zero unclassified LD/ST opcodes, and exact classifier line-count parity")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(es.ARCH_CC), help="sm_120 (Blackwell) or sm_89 (Ada)")
    parser.add_argument("--families", default="",
                        help="comma-separated family ids (catalog parents, alt_cuda_samples_*, calib_*); default all")
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--cells-manifest", type=Path, help="cell_manifest_<gpu>.json (catalog families)")
    parser.add_argument("--triton-source-root", type=Path)
    parser.add_argument("--cuda-samples-root", type=Path)
    parser.add_argument("--repo-root", type=Path,
                        help="dir containing workloads/*.cu and energy_harness/triad_stabilization.cu, energy_harness/raw_trace_recorder.h")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expect-visible-devices", default="")
    parser.add_argument("--reproduce-against", type=Path, action="append",
                        help="committed sass_features_sm_XXX.json (repeatable); recompute static_* and exit 3 on "
                             "any mismatch, naming cell and field")
    parser.add_argument("--committed-vector-width-sass", type=Path,
                        help="committed real cuobjdump dump of the vector-width executable (informational cross-check)")
    parser.add_argument("--allow-fixture-hash-drift", action="store_true",
                        help="accept calibration-fixture sources whose sha256 differs from the authoring snapshot "
                             "(recorded in the output)")
    parser.add_argument("--continue-on-family-error", action="store_true",
                        help="record a failing family in the output and keep going (exit 4 at the end)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.list_families:
        cm = json.loads(args.cells_manifest.read_text()) if args.cells_manifest else {}
        for f in all_family_ids(cm):
            print(f)
        return 0
    for required in ("arch", "out_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (unless --selftest / --list-families)")
    try:
        return run(args)
    except EmitBlocked as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
