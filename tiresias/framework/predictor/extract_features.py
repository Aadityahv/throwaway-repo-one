#!/usr/bin/env python3
"""Label-free static feature table for the Blackwell operator cells (CPU only).

Inputs are the retained compilations (cubin + isolated SASS), the existing static
derivation outputs (per-class lane counts), the development table (cell ids, launch
geometry, shapes, tier, logical bytes ONLY -- no runtime/energy column is read) and
HARDWARE_GROUND_TRUTH.md (the only source of occupancy constants).

Nothing here executes a kernel, touches a GPU or opens a network connection.
Unknown values are emitted as null with a reason, never zero-filled.

Usage (from the repository root):
    python3 tiresias/framework/predictor/extract_features.py
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MI = REPO / "tiresias/framework/compile_evidence"
DERIV_DIR = MI / "operator_derivation"
GROUND_TRUTH = REPO / "HARDWARE_GROUND_TRUTH.md"
DEV_TABLE = MI / "development/development_cells.csv"
CANDIDATE_COUNTS = DERIV_DIR / "candidate_counts.json"
COLLECTIVE_COUNTS = DERIV_DIR / "collective_counts.json"
CACHE_RECORD_SOFTMAX = REPO / "tiresias/app_runners/compiler_resources_raw/blackwell_2026-09-22/04_triton_softmax.txt"
OUT_JSON = HERE / "features_blackwell.json"

# Development-table columns this script is allowed to read (no runtime, energy,
# power or any other measured column is ever looked at).
DEV_COLUMNS_ALLOWED = (
    "gpu", "operator_id", "cell_id", "regime", "candidate_id", "tier", "bytes_per_launch",
    "actual_logical_bytes_per_launch", "grid_blocks", "block_threads", "blocks_per_sm",
    "active_sm_fraction", "geometry_source", "shape_a", "shape_b", "sm_count",
)

READ_HASHES: dict[str, str] = {}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_bytes(path: Path) -> bytes:
    path = Path(path)
    b = path.read_bytes()
    try:
        key = str(path.resolve().relative_to(REPO))
    except ValueError:
        key = str(path.resolve())
    READ_HASHES[key] = sha_bytes(b)
    return b


def read_text(path: Path) -> str:
    return read_bytes(path).decode()


# --------------------------------------------------------------------------- #
# Hardware ground truth                                                       #
# --------------------------------------------------------------------------- #

HW_FIELDS = {
    # key: (row-label prefix, how many leading integers, names for them)
    "sm_count": ("SM count", ("sm_count",)),
    "l2_bytes": ("L2 cache size", ("l2_bytes",)),
    "warp_size": ("Warp size", ("warp_size",)),
    "threads": ("Max threads per block / per SM", ("max_threads_per_block", "max_threads_per_sm")),
    "regs": ("Registers per block / per SM", ("regs_per_block", "regs_per_sm")),
    "smem_block": ("Shared mem per block", ("shared_per_block_bytes",)),
    "smem_sm": ("Shared mem per SM", ("shared_per_sm_bytes",)),
    "max_blocks": ("Max resident blocks per SM", ("max_blocks_per_sm",)),
    "smem_gran": ("Shared-memory allocation granularity", ("shared_alloc_granularity_bytes",)),
    "reg_gran": ("Register allocation granularity", ("register_alloc_granularity_regs_per_warp",)),
    "max_regs": ("Max registers per thread", ("max_registers_per_thread",)),
    "smem_reserved": ("Reserved shared memory per block", ("reserved_shared_per_block_bytes",)),
    "max_warps": ("Max warps per SM", ("max_warps_per_sm",)),
    "subparts": ("Register sub-partitions per SM", ("register_sub_partitions_per_sm",)),
}
HW_KEYS = [n for _, names in HW_FIELDS.values() for n in names]
# Constants that the occupancy model would also depend on but that
# HARDWARE_GROUND_TRUTH.md does not list for Blackwell.
HW_NOT_IN_GROUND_TRUTH = (
    "max_dynamic_shared_per_block_opt_in (only needed if a block asks for more than shared_per_block_bytes)",
    "L1 data-cache capacity (not needed by any feature; ground truth says none is verified)",
)


def _ints(text: str) -> list[int]:
    return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text)]


def load_hardware(text: str, section_prefix: str = "## Blackwell") -> dict:
    """Parse the Blackwell table of HARDWARE_GROUND_TRUTH.md. Missing rows -> None."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(section_prefix)), None)
    hw = {k: None for k in HW_KEYS}
    if start is None:
        return hw
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    rows = []
    for l in lines[start:end]:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if len(cells) >= 2:
                rows.append((cells[0], cells[1]))
    for prefix, names in HW_FIELDS.values():
        match = [v for k, v in rows if k.startswith(prefix)]
        if len(match) == 1:
            nums = _ints(match[0])
            if len(nums) >= len(names):
                for n, v in zip(names, nums):
                    hw[n] = v
    return hw


# --------------------------------------------------------------------------- #
# Cubin ELF parsing (pure Python)                                             #
# --------------------------------------------------------------------------- #

EIATTR_MAX_THREADS = 0x05
EIATTR_REQNTID = 0x10
EIATTR_FRAME_SIZE = 0x11
EIATTR_MIN_STACK_SIZE = 0x12
EIATTR_MAXREG_COUNT = 0x1B
EIATTR_REGCOUNT = 0x2F


class ElfError(ValueError):
    pass


def _sections(data: bytes):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ElfError("not a little-endian ELF64")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    entsize, num, strndx = struct.unpack_from("<HHH", data, 0x3A)
    if entsize != 64 or not (0 < strndx < num) or shoff + entsize * num > len(data):
        raise ElfError("bad section table")
    secs = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(num)]
    ss = secs[strndx]
    names = data[ss[4]:ss[4] + ss[5]]
    out = []
    for s in secs:
        out.append({"name": names[s[0]:].split(b"\0", 1)[0].decode(), "type": s[1], "flags": s[2],
                    "offset": s[4], "size": s[5], "link": s[6], "info": s[7]})
    return out


def _info_records(raw: bytes):
    """Yield (fmt, tag, payload|value) from an .nv.info style blob."""
    pos = 0
    while pos < len(raw):
        if pos + 4 > len(raw):
            raise ElfError("truncated info header")
        fmt, tag, size = struct.unpack_from("<BBH", raw, pos)
        pos += 4
        if fmt == 4:
            if pos + size > len(raw):
                raise ElfError("truncated info payload")
            yield fmt, tag, raw[pos:pos + size]
            pos += size
        elif fmt in (1, 2, 3):
            yield fmt, tag, size
        else:
            raise ElfError("unsupported info format %d" % fmt)


def parse_cubin_resources(data: bytes, symbol: str, reserved_shared_bytes: int | None) -> dict:
    """Per-kernel resource metadata from the cubin ELF.

    registers: EIATTR_REGCOUNT (0x2f) record for the kernel's symbol in the global .nv.info.
    frame/stack: EIATTR_FRAME_SIZE (0x11) and EIATTR_MIN_STACK_SIZE (0x12), same place.
    shared: size of section .nv.shared.<symbol>; this architecture's toolchain adds the
            driver reserve to it (HARDWARE_GROUND_TRUTH.md cross-checked finding), so the
            declared static size is section_size - reserve; absent section -> 0.
    local: size of .nv.local.<symbol> if present.
    """
    secs = _sections(data)
    by_name = collections.defaultdict(list)
    for s in secs:
        by_name[s["name"]].append(s)

    def one(name):
        got = by_name.get(name, [])
        if len(got) > 1:
            raise ElfError("ambiguous section " + name)
        return got[0] if got else None

    symtab = next((s for s in secs if s["type"] == 2), None)
    if symtab is None:
        raise ElfError("no symbol table")
    strtab = secs[symtab["link"]]
    strs = data[strtab["offset"]:strtab["offset"] + strtab["size"]]
    sym_index = None
    for i in range(symtab["size"] // 24):
        n = struct.unpack_from("<I", data, symtab["offset"] + i * 24)[0]
        if strs[n:].split(b"\0", 1)[0].decode() == symbol:
            if sym_index is not None:
                raise ElfError("ambiguous symbol " + symbol)
            sym_index = i
    if sym_index is None:
        raise ElfError("symbol not found: " + symbol)
    if one(".text." + symbol) is None:
        raise ElfError("no .text section for " + symbol)

    glob = one(".nv.info")
    attrs: dict[int, int] = {}
    if glob is not None:
        raw = data[glob["offset"]:glob["offset"] + glob["size"]]
        for fmt, tag, payload in _info_records(raw):
            if fmt == 4 and tag in (EIATTR_REGCOUNT, EIATTR_FRAME_SIZE, EIATTR_MIN_STACK_SIZE) and len(payload) == 8:
                idx, val = struct.unpack("<II", payload)
                if idx == sym_index:
                    if tag in attrs:
                        raise ElfError("duplicate attribute %#x" % tag)
                    attrs[tag] = val
    per = one(".nv.info." + symbol)
    per_tags: dict[str, object] = {}
    reqntid = None
    max_threads = None
    maxreg = None
    if per is not None:
        raw = data[per["offset"]:per["offset"] + per["size"]]
        for fmt, tag, payload in _info_records(raw):
            if fmt == 4 and tag == EIATTR_REQNTID and len(payload) == 12:
                reqntid = struct.unpack("<III", payload)
            elif fmt == 4 and tag == EIATTR_MAX_THREADS and len(payload) == 12:
                max_threads = struct.unpack("<III", payload)
            elif fmt == 3 and tag == EIATTR_MAXREG_COUNT:
                maxreg = payload
            elif fmt == 4 and tag == EIATTR_REGCOUNT and len(payload) == 8 and EIATTR_REGCOUNT not in attrs:
                attrs[EIATTR_REGCOUNT] = struct.unpack("<II", payload)[1]
    if EIATTR_REGCOUNT not in attrs:
        raise ElfError("no REGCOUNT attribute for " + symbol)
    shared = one(".nv.shared." + symbol)
    local = one(".nv.local." + symbol)
    const0 = one(".nv.constant0." + symbol)
    shared_section = shared["size"] if shared else None
    if shared_section is None:
        static_shared = 0
    elif reserved_shared_bytes is None:
        static_shared = None
    else:
        static_shared = max(0, shared_section - reserved_shared_bytes)
    return {
        "registers_per_thread": attrs[EIATTR_REGCOUNT],
        "frame_size_bytes": attrs.get(EIATTR_FRAME_SIZE),
        "min_stack_size_bytes": attrs.get(EIATTR_MIN_STACK_SIZE),
        "local_memory_bytes_per_thread": local["size"] if local else 0,
        "shared_section_bytes_raw": shared_section,
        "static_shared_bytes_per_block": static_shared,
        "reqntid": list(reqntid) if reqntid else None,
        "max_threads_attr": list(max_threads) if max_threads else None,
        "maxreg_count_attr": maxreg,
        "constant_bank0_bytes": const0["size"] if const0 else None,
    }


# --------------------------------------------------------------------------- #
# Occupancy arithmetic                                                         #
# --------------------------------------------------------------------------- #

WARP_ALLOC_MULTIPLES_CONSIDERED = (1, 2, 4)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def occupancy(hw: dict, regs: int | None, threads_per_block: int | None, grid_blocks: int | None,
              static_shared: int | None, dynamic_shared: int | None) -> dict:
    """Resident blocks per SM and waves. Null (with reasons) if any needed constant/input is missing.

    The register limit uses per-warp allocation with the ground-truth granularity. The
    warp-allocation multiple is NOT in the ground truth, so the register limit is evaluated
    for multiples 1, 2 and 4; blocks_per_sm is reported only if the result is identical for
    all of them (the answer then does not depend on the unlisted constant).
    """
    need_hw = ("sm_count", "regs_per_sm", "max_threads_per_sm", "max_warps_per_sm", "max_blocks_per_sm",
               "shared_per_sm_bytes", "shared_per_block_bytes", "shared_alloc_granularity_bytes",
               "register_alloc_granularity_regs_per_warp", "reserved_shared_per_block_bytes", "warp_size",
               "max_threads_per_block", "max_registers_per_thread",
               "register_sub_partitions_per_sm", "regs_per_block")
    missing = [k for k in need_hw if hw.get(k) is None]
    if regs is None:
        missing.append("registers_per_thread")
    if threads_per_block is None:
        missing.append("threads_per_block")
    if grid_blocks is None:
        missing.append("grid_blocks")
    if static_shared is None:
        missing.append("static_shared_bytes_per_block")
    res: dict = {"blocks_per_sm": None, "limiter": None, "reasons": []}
    # Partial arithmetic that does not need the dynamic shared size.
    if missing:
        res["reasons"].append("missing inputs/constants: " + ", ".join(missing))
        return res
    ws = hw["warp_size"]
    warps = _ceil_div(threads_per_block, ws)
    if threads_per_block > hw["max_threads_per_block"] or regs > hw["max_registers_per_thread"]:
        res["reasons"].append("launch exceeds a per-block/per-thread hardware limit")
        return res
    by_threads = hw["max_threads_per_sm"] // threads_per_block
    by_warps = hw["max_warps_per_sm"] // warps
    by_blocks = hw["max_blocks_per_sm"]
    regs_per_warp = _ceil_div(regs * ws, hw["register_alloc_granularity_regs_per_warp"]) * hw["register_alloc_granularity_regs_per_warp"]
    # cuda_occupancy.h cudaOccMaxBlocksPerSMRegsLimit (CUDA 13.2, cc 12): registers are
    # allocated per warp within each sub-partition; the per-block hardware check rounds
    # the block's warps up to the sub-partition count. Recorded in HARDWARE_GROUND_TRUTH.md.
    parts = hw["register_sub_partitions_per_sm"]
    if regs_per_warp * _ceil_div(warps, parts) * parts > hw["regs_per_block"]:
        by_regs = 0
    else:
        by_regs = ((hw["regs_per_sm"] // parts) // regs_per_warp) * parts // warps
    by_regs_by_mult = {m: by_regs for m in WARP_ALLOC_MULTIPLES_CONSIDERED}
    base_other = min(by_threads, by_warps, by_blocks)
    res.update({
        "warps_per_block": warps, "registers_per_warp_allocated": regs_per_warp,
        "limit_by_threads": by_threads, "limit_by_warps": by_warps, "limit_by_max_blocks": by_blocks,
        "limit_by_registers_by_warp_alloc_multiple": {str(k): v for k, v in by_regs_by_mult.items()},
        "limit_by_shared": None, "limit_by_registers_if_multiple_1": by_regs_by_mult[1],
    })
    shared_needed_fn = lambda dyn: _ceil_div(static_shared + dyn + hw["reserved_shared_per_block_bytes"], hw["shared_alloc_granularity_bytes"]) * hw["shared_alloc_granularity_bytes"]
    # Occupancy ignoring shared memory, and the shared size beyond which shared would bind.
    reg_vals = set(by_regs_by_mult.values())
    ignoring_shared = {m: min(base_other, v) for m, v in by_regs_by_mult.items()}
    res["blocks_per_sm_ignoring_shared_by_warp_alloc_multiple"] = {str(k): v for k, v in ignoring_shared.items()}
    lo_ignoring = min(ignoring_shared.values())
    # shared binds only if floor(shared_per_sm / alloc) < blocks_ignoring_shared
    # -> largest per-block shared (alloc) that is non-binding for blocks=lo_ignoring:
    if lo_ignoring > 0:
        max_alloc = hw["shared_per_sm_bytes"] // lo_ignoring
        max_alloc = (max_alloc // hw["shared_alloc_granularity_bytes"]) * hw["shared_alloc_granularity_bytes"]
        res["shared_nonbinding_up_to_dynamic_bytes"] = max(0, max_alloc - hw["reserved_shared_per_block_bytes"] - static_shared)
    if dynamic_shared is None:
        res["reasons"].append("dynamic shared bytes per launch unknown; occupancy not reported "
                              "(see blocks_per_sm_ignoring_shared_by_warp_alloc_multiple and shared_nonbinding_up_to_dynamic_bytes)")
        return res
    if static_shared + dynamic_shared > hw["shared_per_block_bytes"]:
        res["reasons"].append("static+dynamic shared exceeds the per-block limit in the ground truth (opt-in limit not listed)")
        return res
    alloc = shared_needed_fn(dynamic_shared)
    by_shared = hw["shared_per_sm_bytes"] // alloc
    res["shared_allocated_bytes_per_block"] = alloc
    res["limit_by_shared"] = by_shared
    per_mult = {m: min(base_other, v, by_shared) for m, v in by_regs_by_mult.items()}
    vals = set(per_mult.values())
    if len(vals) != 1:
        res["reasons"].append("blocks_per_sm depends on the warp-allocation multiple, which is not in HARDWARE_GROUND_TRUTH.md: "
                              + json.dumps({str(k): v for k, v in per_mult.items()}))
        res["blocks_per_sm_by_warp_alloc_multiple"] = {str(k): v for k, v in per_mult.items()}
        return res
    bps = vals.pop()
    if bps < 1:
        res["reasons"].append("zero resident blocks")
        return res
    limits = {"threads": by_threads, "warps": by_warps, "max_blocks": by_blocks,
              "registers": by_regs_by_mult[1] if len(reg_vals) == 1 else None, "shared": by_shared}
    res["limiter"] = sorted(k for k, v in limits.items() if v == bps)
    res["blocks_per_sm"] = bps
    res["max_resident_warps_per_sm"] = bps * warps
    res["occupancy_fraction_of_max_warps"] = bps * warps / hw["max_warps_per_sm"]
    slots = hw["sm_count"] * bps
    waves = _ceil_div(grid_blocks, slots)
    res["resident_block_slots"] = slots
    res["waves"] = waves
    res["waves_fractional"] = grid_blocks / slots
    res["last_wave_fill_fraction"] = (grid_blocks - slots * (waves - 1)) / slots
    res["tail_idle_fraction_of_last_wave"] = 1 - res["last_wave_fill_fraction"]
    res["first_wave_resident_blocks_per_sm_avg"] = min(bps, grid_blocks / hw["sm_count"])
    res["first_wave_active_warps_per_sm_avg"] = min(bps, grid_blocks / hw["sm_count"]) * warps
    res["active_sm_fraction"] = min(1.0, grid_blocks / hw["sm_count"])
    res["reasons"] = []
    return res


# --------------------------------------------------------------------------- #
# SASS helpers                                                                 #
# --------------------------------------------------------------------------- #

def load_derivation():
    """Import derive.py / collective.py exactly as their authors require (module registered
    in sys.modules before exec, because they use dataclasses)."""
    sys.path.insert(0, str(DERIV_DIR))

    def imp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    D = imp("derive", DERIV_DIR / "derive.py")
    C = imp("collective", DERIV_DIR / "collective.py")
    return D, C


class TracedPred(str):
    """Guard text that remembers which instruction it belongs to."""
    pc = -1


class TSite:
    """derive.Site look-alike whose guard remembers its pc (so unknown guards can be located)."""
    def __init__(self, site):
        self.pc, self.op, self.a = site.pc, site.op, site.a
        if site.pred is None:
            self.pred = None
        else:
            tp = TracedPred(site.pred)
            tp.pc = site.pc
            self.pred = tp


@contextlib.contextmanager
def trace_unknown_guards(modules):
    unknown: set[int] = set()
    saved = []
    for mod in modules:
        orig = mod.pred
        saved.append((mod, orig))

        def make(orig):
            def traced(t, p):
                v = orig(t, p)
                if v is None and isinstance(t, TracedPred):
                    unknown.add(t.pc)
                return v
            return traced
        mod.pred = make(orig)
    try:
        yield unknown
    finally:
        for mod, orig in saved:
            mod.pred = orig


def op_family(op: str):
    """Return (family, width_bits|None, mode|None). Width is the access width where defined."""
    def w_of(o, default=32):
        if ".128" in o:
            return 128
        if ".64" in o:
            return 64
        if ".U8" in o or ".S8" in o:
            return 8
        if ".U16" in o or ".S16" in o:
            return 16
        return default
    if op.startswith("LDGSTS"):
        return "async_copy_global_to_shared", w_of(op), None
    if op.startswith("LDGDEPBAR") or op.startswith("DEPBAR"):
        return "dependency_barrier", None, None
    if op.startswith("LDG"):
        return "global_load", w_of(op), None
    if op.startswith("STG"):
        return "global_store", w_of(op), None
    if op.startswith("LDSM"):
        return "shared_matrix_load", None, None
    if op.startswith("LDS"):
        return "shared_load", w_of(op), None
    if op.startswith("STS"):
        return "shared_store", w_of(op), None
    if op.startswith("SHFL"):
        return "shuffle", 32, op.split(".", 1)[1] if "." in op else None
    if op.startswith("MUFU"):
        return "special_function", None, op.split(".", 1)[1] if "." in op else None
    if op.startswith("BAR"):
        return "barrier", None, None
    if op == "FFMA" or op == "UFFMA":
        return "fp32_fma", None, None
    if op in ("FADD", "UFADD"):
        return "fp32_add", None, None
    if op in ("FMUL", "UFMUL"):
        return "fp32_mul", None, None
    if op.startswith(("FMNMX", "FSEL", "FSETP", "UFSEL", "UFSETP", "HFMA2", "F2I", "I2F", "I2FP", "UI2FP")):
        return "fp_other", None, None
    if op.startswith(("IADD", "UIADD", "IMAD", "UIMAD", "LEA", "ULEA", "LOP3", "ULOP3", "SHF", "USHF", "ISETP", "UISETP",
                      "SEL", "SGXT", "IMNMX", "POPC", "FLO", "PRMT")):
        return "integer_alu", None, None
    if op.startswith(("BRA", "BSSY", "BSYNC", "WARPSYNC", "ENDCOLLECTIVE", "EXIT", "RET", "CALL", "JMP")):
        return "control", None, None
    if op.startswith(("LDC", "LDCU", "S2R", "S2UR", "CS2R", "R2UR", "MOV", "UMOV", "P2R", "R2P", "PLOP3", "UPLOP3", "NOP")):
        return "move_const_special", None, None
    return "other", None, None


REG_RE = re.compile(r"\b(U?R)(\d+)(\.64|\.128)?\b")
PRED_RE = re.compile(r"\b(U?P)(\d)\b")


def _regs_in(token: str, wide: int = 1):
    out = []
    for m in REG_RE.finditer(token):
        base, idx, suf = m.group(1), int(m.group(2)), m.group(3)
        width = 4 if suf == ".128" else 2 if suf == ".64" else wide
        out.extend("%s%d" % (base, idx + k) for k in range(width))
    for m in PRED_RE.finditer(token):
        out.append(m.group(1) + m.group(2))
    return out


def sass_def_use(site):
    """(dests, srcs) register/predicate name sets for one instruction (heuristic, see FEATURES.md)."""
    op, a = site.op, [x.replace(".reuse", "") for x in site.a]
    dest_w = 4 if ".128" in op else 2 if (".64" in op or ".WIDE" in op) else 1
    srcs, dests = [], []
    if site.pred:
        srcs += _regs_in(str(site.pred).lstrip("!@"))
    no_dest = op.startswith(("STG", "STS", "BAR", "BRA", "EXIT", "NOP", "DEPBAR", "LDGDEPBAR", "BSSY", "BSYNC",
                             "WARPSYNC", "ENDCOLLECTIVE", "LDGSTS", "RET"))
    if no_dest:
        for k, tok in enumerate(a):
            wide = dest_w if (op.startswith(("STG", "STS")) and k == len(a) - 1) else 1
            srcs += _regs_in(tok, wide)
        return set(dests), set(srcs)
    nd = 2 if op.startswith(("ISETP", "FSETP", "UISETP", "UFSETP", "PLOP3", "SHFL")) else 1
    for k, tok in enumerate(a):
        if k < nd and a:
            dests += _regs_in(tok, dest_w if k == nd - 1 else 1)
        else:
            srcs += _regs_in(tok, 1)
    return set(dests), set(srcs)


def chain_metrics(sites_in_order):
    """Longest register def-use chain, global-load -> use edges and load-chain depth over a
    program-order sequence of sites (heuristic: last writer wins; no memory or loop-carried deps)."""
    last_def: dict[str, int] = {}
    depth, ldepth = [], []
    used_by: dict[int, set] = collections.defaultdict(set)
    ops = []
    for i, s in enumerate(sites_in_order):
        d, u = sass_def_use(s)
        prods = {last_def[r] for r in u if r in last_def}
        for p in prods:
            used_by[p].add(i)
        depth.append(1 + max((depth[p] for p in prods), default=0))
        is_ldg = s.op.startswith("LDG") and not s.op.startswith(("LDGSTS", "LDGDEPBAR"))
        ldepth.append((1 if is_ldg else 0) + max((ldepth[p] for p in prods), default=0))
        for r in d:
            last_def[r] = i
        ops.append(s.op)
    ldg = [i for i, o in enumerate(ops) if o.startswith("LDG") and not o.startswith(("LDGSTS", "LDGDEPBAR"))]
    lds = [i for i, o in enumerate(ops) if o.startswith("LDS") and not o.startswith("LDSM")]
    return {
        "instructions": len(ops),
        "critical_path_instructions": max(depth, default=0),
        "global_loads": len(ldg),
        "global_loads_with_a_consumer": sum(1 for i in ldg if used_by[i]),
        "global_load_to_use_edges": sum(len(used_by[i]) for i in ldg),
        "dependent_global_load_depth": max(ldepth, default=0),
        "shared_loads": len(lds),
        "shared_loads_with_a_consumer": sum(1 for i in lds if used_by[i]),
        "async_copies": sum(1 for o in ops if o.startswith("LDGSTS")),
        "dependency_waits": sum(1 for o in ops if o.startswith(("LDGDEPBAR", "DEPBAR"))),
    }


def max_register_index(sites):
    hi = -1
    for s in sites:
        d, u = sass_def_use(s)
        for r in d | u:
            m = re.fullmatch(r"R(\d+)", r)
            if m:
                hi = max(hi, int(m.group(1)))
    return hi


# --------------------------------------------------------------------------- #
# Per-cell work from interpreted per-lane visits                               #
# --------------------------------------------------------------------------- #

def interpret_cell(D, C, corpus, row, root, collective: bool):
    """Interpret every lane of one block with the derivation's own interpreter and return
    (sites, per-lane visits, per-class events, unknown-guard pcs, binding)."""
    constants, coords, threads, blocks, launch = D.binding(corpus, row, root)
    text = read_text(root / row["disassembly_path"])
    sites = D.parse(text)
    tsites = [TSite(s) for s in sites]
    lane_visits = []
    total_events = collections.Counter()
    arrivals_by_lane = []
    mods = [D, C] if collective else [D]
    with trace_unknown_guards(mods) as unknown:
        for lane in range(threads):
            c = {**coords, "SR_TID.X": D.V.exact(lane % launch["block"][0]), "SR_TID.Y": D.V.exact(lane // launch["block"][0]),
                 "SR_TID.Z": D.V.exact(0), "SR_LANEID": D.V.exact(lane % 32)}
            if collective:
                arr = collections.Counter()
                e, v, _u = C.run_lane_div(tsites, constants, c, arrivals=arr)
                arrivals_by_lane.append(arr)
            else:
                e, v, _u = D.run_lane(tsites, constants, c)
            total_events.update(e)
            lane_visits.append(v)
    if collective:
        C.check_warp_convergence(arrivals_by_lane, threads)
    return sites, lane_visits, total_events, set(unknown), (threads, blocks, launch)


def work_features(sites, lane_visits, unknown_pcs, threads, blocks):
    op_at = {s.pc: s.op for s in sites}
    pc_visits = collections.Counter()
    for v in lane_visits:
        pc_visits.update(v)
    per_op = collections.Counter()
    per_op_upper = set()
    fam_tot = collections.Counter()
    fam_width = collections.defaultdict(collections.Counter)
    fam_mode = collections.defaultdict(collections.Counter)
    fam_upper = set()
    static_width = collections.defaultdict(collections.Counter)
    for s in sites:
        fam, w, mode = op_family(s.op)
        if w is not None:
            static_width[fam][str(w)] += 1
    for pc, n in pc_visits.items():
        op = op_at[pc]
        fam, w, mode = op_family(op)
        n_all = n * blocks
        per_op[op] += n_all
        fam_tot[fam] += n_all
        if w is not None:
            fam_width[fam][str(w)] += n_all
        if mode is not None:
            fam_mode[fam][mode] += n_all
        if pc in unknown_pcs:
            per_op_upper.add(op)
            fam_upper.add(fam)
    families = {}
    for fam in sorted(fam_tot):
        families[fam] = {"lane_instructions": fam_tot[fam], "exact": fam not in fam_upper}
        if fam in fam_width:
            families[fam]["width_bits_histogram"] = dict(sorted(fam_width[fam].items(), key=lambda kv: int(kv[0])))
        if fam in fam_mode:
            families[fam]["mode_histogram"] = dict(sorted(fam_mode[fam].items()))
    # LDGSTS carries a zero-fill/source-size predicate that the interpreter does not model, so a
    # masked-off prefetch still counts as an issued lane instruction: always an upper bound.
    if "async_copy_global_to_shared" in fam_tot:
        fam_upper.add("async_copy_global_to_shared")
        families["async_copy_global_to_shared"]["exact"] = False
        families["async_copy_global_to_shared"]["note"] = "issued lane count; zero-fill/masked copies are not modelled, so bytes are an upper bound"
    gl = fam_width.get("global_load", {})
    gs = fam_width.get("global_store", {})
    ga = fam_width.get("async_copy_global_to_shared", {})
    gl_bytes = sum(int(w) // 8 * n for w, n in gl.items())
    gs_bytes = sum(int(w) // 8 * n for w, n in gs.items())
    ga_bytes = sum(int(w) // 8 * n for w, n in ga.items())
    total = sum(per_op.values())
    nthreads = threads * blocks
    return {
        "total_lane_instructions": total,
        "total_lane_instructions_per_thread": total / nthreads,
        "families": families,
        "opcode_lane_counts": dict(sorted(per_op.items())),
        "opcodes_with_unknown_guard_upper_bound": sorted(per_op_upper),
        "all_counts_exact": not per_op_upper,
        "static_global_access_width_sites": {k: dict(v) for k, v in static_width.items() if k in ("global_load", "global_store", "async_copy_global_to_shared")},
        "executed_global_load_bytes": gl_bytes,
        "executed_async_copy_global_to_shared_bytes": ga_bytes,
        "executed_global_store_bytes": gs_bytes,
        "executed_global_bytes_per_thread": (gl_bytes + ga_bytes + gs_bytes) / nthreads,
        "executed_global_bytes_exact": not ({"global_load", "global_store", "async_copy_global_to_shared"} & fam_upper),
    }, pc_visits


def barrier_features(sites, lane_visits, threads, blocks):
    bar_pcs = [s.pc for s in sites if s.op.startswith("BAR")]
    releases, uniform = 0, True
    per_pc = {}
    for pc in bar_pcs:
        counts = [v.get(pc, 0) for v in lane_visits]
        mx = max(counts)
        if mx and min(counts) != mx:
            uniform = False
        per_pc[hex(pc)] = {"arrivals_per_block": sum(counts), "max_lane_arrivals": mx}
        releases += mx
    return {
        "static_barrier_sites": len(bar_pcs),
        "barrier_releases_per_block_estimate": releases,
        "barrier_releases_per_launch_estimate": releases * blocks,
        "barrier_participation_uniform": uniform,
        "barrier_arrivals_per_block": sum(v["arrivals_per_block"] for v in per_pc.values()),
        "barrier_participants_per_release_assumed": threads,
        "note": "releases = max per-lane visit count summed over barrier sites; assumes every barrier is block-wide (BAR.SYNC with the default thread count)",
    }


def loop_features(sites, lane_visits, threads):
    by = {s.pc: s for s in sites}
    pc_tot = collections.Counter()
    for v in lane_visits:
        pc_tot.update(v)
    loops = []
    for s in sites:
        if s.op in ("BRA", "BRA.U") and s.a:
            try:
                t = int(s.a[-1], 0)
            except ValueError:
                continue
            if t <= s.pc and pc_tot.get(t, 0) > 0 and pc_tot.get(s.pc, 0) > 0:
                loops.append((t, s.pc))
    loops = sorted(set(loops))

    def contains(o, i):
        return o[0] <= i[0] and i[1] <= o[1] and o != i
    inner = [l for l in loops if not any(contains(l, m) for m in loops)]
    best = None
    for (t, b) in inner:
        region = sum(pc_tot.get(pc, 0) for pc in range(t, b + 16, 16))
        if best is None or region > best[0]:
            best = (region, t, b)
    out = {"static_backward_branches_executed": len(loops), "innermost_loops": len(inner)}
    executed_sites = sorted(pc for pc, n in pc_tot.items() if n > 0)
    if best is None:
        body = [by[pc] for pc in executed_sites]
        out.update({"hot_region": "whole_kernel_no_loop", "loop_header_pc": None, "loop_back_edge_pc": None,
                    "loop_iterations_per_thread_mean": 1.0, "loop_iterations_per_thread_min": 1,
                    "loop_iterations_per_thread_max": 1, "loop_body_static_instructions": len(body),
                    "loop_body_dynamic_instructions_per_iteration_per_thread_mean": sum(pc_tot.values()) / threads})
        out["hot_region_chain"] = chain_metrics(body)
    else:
        _, t, b = best
        its = [v.get(t, 0) for v in lane_visits]
        active = [(i, v) for i, v in zip(its, lane_visits) if i > 0]
        body_pcs = [pc for pc in range(t, b + 16, 16) if pc_tot.get(pc, 0) > 0]
        dyn = [sum(v.get(pc, 0) for pc in range(t, b + 16, 16)) / i for i, v in active]
        out.update({"hot_region": "innermost_hottest_loop", "loop_header_pc": hex(t), "loop_back_edge_pc": hex(b),
                    "loop_iterations_per_thread_mean": sum(its) / len(its),
                    "loop_iterations_per_thread_min": min(its), "loop_iterations_per_thread_max": max(its),
                    "loop_body_static_instructions": (b - t) // 16 + 1,
                    "loop_body_executed_static_instructions": len(body_pcs),
                    "loop_body_dynamic_instructions_per_iteration_per_thread_mean": sum(dyn) / len(dyn)})
        out["hot_region_chain"] = chain_metrics([by[pc] for pc in body_pcs])
    out["whole_kernel_chain"] = chain_metrics([by[pc] for pc in executed_sites])
    out["iterations_note"] = ("iterations = visits of the loop header per thread; equals the trip count if the loop is entered once")
    return out


# --------------------------------------------------------------------------- #
# Dynamic shared memory                                                        #
# --------------------------------------------------------------------------- #

def parse_softmax_cache_record(text: str):
    recs = {}
    for m in re.finditer(r"dir=(\S+) warps=(\d+) stages=(\d+) shared=(\d+)\s+REG=(\d+) BLOCK_SIZE=(\d+)", text):
        key = (int(m.group(6)), int(m.group(2)), int(m.group(3)))
        recs[key] = {"shared": int(m.group(4)), "reg": int(m.group(5)), "dir": m.group(1)}
    return recs


TRITON_CACHE = REPO / "tiresias/framework/compile_evidence/acquisition_runs/operator_compile_blackwell_20261001_76cdbada/remote_triton_cache"
_TRITON_CACHE_SHARED = None
def triton_cache_shared() -> dict:
    """cubin sha256 -> Triton metadata 'shared' bytes, from the capture's own cache (pulled read-only)."""
    global _TRITON_CACHE_SHARED
    if _TRITON_CACHE_SHARED is None:
        _TRITON_CACHE_SHARED = {}
        for meta in sorted(TRITON_CACHE.rglob("*.json")):
            cubin = meta.with_suffix(".cubin")
            if meta.name.startswith("__grp__") or not cubin.exists():
                continue
            m = json.loads(read_text(meta))
            if isinstance(m.get("shared"), int):
                _TRITON_CACHE_SHARED[sha_bytes(read_bytes(cubin))] = m["shared"]
    return _TRITON_CACHE_SHARED


def dynamic_shared(corpus, row, root, launch, threads, resources, softmax_records):
    """Return (bytes|None, source, reason|None)."""
    oid = row["operator_id"]
    if corpus == "triton":
        ptx_ref = [a for a in json.loads(read_text(root / row["retained_compiler_events"][0]["path"])).get("asm", {}).values()
                   if a["path"].endswith(".ptx")]
        ptx = read_text(root / ptx_ref[0]["path"]) if ptx_ref else None
        if ptx is None:
            return None, None, "no retained PTX to decide whether the kernel uses shared memory"
        cached = triton_cache_shared().get(row["cubin_sha256"])
        if cached is not None:
            # Exact: the original capture's own Triton cache, matched by byte-identical cubin.
            return cached, "Triton metadata 'shared' from the original capture's compile cache, matched by cubin sha256 (remote_triton_cache/)", None
        if ".shared" not in ptx:
            return 0, "retained PTX declares and uses no shared memory", None
        if oid == "dev_triton_softmax":
            ce = launch["constexpr"]
            key = (ce["BLOCK_SIZE"]["value"], ce["num_warps"]["value"], ce["num_stages"]["value"])
            rec = softmax_records.get(key)
            if rec is None:
                return None, None, "no prior compile-cache record for (BLOCK_SIZE,num_warps,num_stages)=%s" % (key,)
            if rec["reg"] != resources["registers_per_thread"]:
                return None, None, ("prior cache record %s reports REG=%d but the retained cubin has %d: not the same compilation"
                                    % (key, rec["reg"], resources["registers_per_thread"]))
            return rec["shared"], ("Triton 'shared' field from a prior compile-cache record (%s), matched on (BLOCK_SIZE,num_warps,num_stages) "
                                   "and confirmed by equal register count" % CACHE_RECORD_SOFTMAX.name), None
        return None, None, ("Triton dynamic shared size is not recorded in the retained compile events or PTX/IR, and no "
                            "compile-cache record exists for this operator")
    # CUDA
    src = read_text(root / row["retained_source_path"])
    if "extern __shared__" not in src and "extern  __shared__" not in src:
        return 0, "retained source has no extern __shared__ declaration", None
    if "reduction" in oid:
        needle = "int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);"
        if needle not in src:
            return None, None, "sample's smemSize formula not found in retained source"
        b = 2 * threads * 4 if threads <= 32 else threads * 4
        return b, "launch formula in retained reduction_kernel.cu (smemSize, T=float) with the derivation's launch threads", None
    return None, None, "kernel declares extern __shared__ but no launch-size formula is known"


# --------------------------------------------------------------------------- #
# Table assembly                                                               #
# --------------------------------------------------------------------------- #

def num(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else f


def load_dev_rows():
    text = read_text(DEV_TABLE)
    rows = {}
    for r in csv.DictReader(text.splitlines()):
        if r["gpu"] != "blackwell":
            continue
        rows[r["cell_id"]] = {k: r[k] for k in DEV_COLUMNS_ALLOWED}
    return rows


def dev_geometry(d, hw):
    grid, bt = num(d["grid_blocks"]), num(d["block_threads"])
    sm = hw.get("sm_count")
    out = {"grid_blocks": grid, "block_threads": bt, "geometry_source": d["geometry_source"] or None,
           "dev_table_blocks_per_sm_is_grid_over_sm_count": None,
           "dev_table_blocks_per_sm": num(d["blocks_per_sm"]), "dev_table_active_sm_fraction": num(d["active_sm_fraction"])}
    if grid is not None and sm and out["dev_table_blocks_per_sm"] is not None:
        out["dev_table_blocks_per_sm_is_grid_over_sm_count"] = abs(grid / sm - out["dev_table_blocks_per_sm"]) < 1e-9
    return out


def dev_memory(d):
    return {"logical_bytes_per_launch": num(d["actual_logical_bytes_per_launch"]), "tier": d["tier"] or None,
            "shape_a": num(d["shape_a"]), "shape_b": num(d["shape_b"])}


def build_pytorch_rows(hw, dev):
    """Rows for the 27 PyTorch cells, from the sm_120 libtorch SASS and the dispatch trace
    (pytorch_features/adapter.py). Imported lazily so the retained-cell path is unaffected."""
    sys.path.insert(0, str(HERE / "pytorch_features"))
    import adapter
    return adapter.build_rows(sys.modules[__name__], hw, dev)


def build_rows(hw):
    D, C = load_derivation()
    cand = json.loads(read_text(CANDIDATE_COUNTS))
    coll = json.loads(read_text(COLLECTIVE_COUNTS))
    cand_by = {(r["operator_id"], r["cell"]): r for r in cand["rows"]}
    coll_by = {(r["operator_id"], r["cell"]): r for r in coll["rows"]}
    dev = load_dev_rows()
    softmax_records = parse_softmax_cache_record(read_text(CACHE_RECORD_SOFTMAX))
    read_text(GROUND_TRUTH)
    for p in (DERIV_DIR / "derive.py", DERIV_DIR / "collective.py", MI / "retained_count/derive.py", D.CONFIG,
              MI.parent / "adapters/adapters.json"):
        read_bytes(p)
    rows_out = []
    seen_dev = set()
    for corpus, root in D.CORPORA.items():
        manifest = json.loads(read_text(root / "retention_manifest.json"))
        for row in manifest["rows"]:
            cell_id = "blackwell/%s/%s" % (row["operator_id"], row["cell"])
            missing = []
            rec = {"cell_id": cell_id, "operator_id": row["operator_id"], "cell": row["cell"], "corpus": corpus}
            if cell_id not in dev:
                raise SystemExit("retained cell not in development table: " + cell_id)
            seen_dev.add(cell_id)
            # hash-verify the two binaries against the manifest
            cubin_b = read_bytes(root / row["cubin_path"])
            sass_b = read_bytes(root / row["disassembly_path"])
            if sha_bytes(cubin_b) != row["cubin_sha256"] or sha_bytes(sass_b) != row["disassembly_sha256"]:
                raise SystemExit("hash mismatch against retention manifest: " + cell_id)
            rec["inputs"] = {"cubin_sha256": row["cubin_sha256"], "isolated_sass_sha256": row["disassembly_sha256"],
                             "kernel_symbol": row["isolated_function_section"]}
            read_bytes(root / "retention_manifest.json")
            for ref in row["retained_compiler_events"]:
                read_bytes(root / ref["path"])
            read_bytes(root / row["retained_source_path"])
            collective = (row["operator_id"], row["cell"]) in coll_by and cand_by[(row["operator_id"], row["cell"])]["status"] != "candidate_class_counts"
            # --- resources
            res = parse_cubin_resources(cubin_b, row["isolated_function_section"], hw["reserved_shared_per_block_bytes"])
            # --- interpretation
            sites, lane_visits, events, unknown_pcs, (threads, blocks, launch) = interpret_cell(D, C, corpus, row, root, collective)
            d = dev[cell_id]
            geom = dev_geometry(d, hw)
            grid_total = launch["grid"][0] * launch["grid"][1] * launch["grid"][2]
            geom.update({"grid_blocks_derivation": grid_total, "block_threads_derivation": threads,
                         "geometry_matches_dev_table": (geom["grid_blocks"] == grid_total and geom["block_threads"] == threads)
                         if geom["grid_blocks"] is not None and geom["block_threads"] is not None else None})
            if collective:
                ref_row = coll_by[(row["operator_id"], row["cell"])]
            else:
                ref_row = cand_by[(row["operator_id"], row["cell"])]
            # --- dynamic shared
            dyn, dyn_src, dyn_reason = dynamic_shared(corpus, row, root, launch, threads, res, softmax_records)
            res["dynamic_shared_bytes_per_launch"] = dyn
            res["dynamic_shared_source"] = dyn_src
            if dyn_reason:
                res["dynamic_shared_missing_reason"] = dyn_reason
                missing.append({"feature": "dynamic_shared_bytes_per_launch", "reason": dyn_reason})
            hi = max_register_index(sites)
            res["max_register_index_in_isolated_sass"] = hi
            res["register_count_consistent_with_sass"] = hi + 1 <= res["registers_per_thread"]
            if res["reqntid"] is not None:
                res["reqntid_matches_launch_threads"] = (res["reqntid"][0] * res["reqntid"][1] * res["reqntid"][2] == threads)
            rec["geometry"] = geom
            rec["resources"] = res
            # --- occupancy
            occ = occupancy(hw, res["registers_per_thread"], threads, grid_total, res["static_shared_bytes_per_block"], dyn)
            rec["occupancy"] = occ
            if occ["blocks_per_sm"] is None:
                for r_ in occ["reasons"]:
                    missing.append({"feature": "occupancy (blocks_per_sm, waves, tail)", "reason": r_})
            # --- work
            work, pc_visits = work_features(sites, lane_visits, unknown_pcs, threads, blocks)
            # cross-check six classes against the frozen derivation outputs
            cls_map = {"shuffle": "shuffle", "shared_load": "shared_load", "shared_store": "shared_store",
                       "exponential": "special_function", "fma": "fp32_fma", "barrier": "barrier"}
            op_at = {s.pc: s.op for s in sites}
            check = {}
            for cname, ref_c in ref_row["classes"].items():
                mine = 0
                for pc, n in pc_visits.items():
                    if cname == "exponential":
                        hit = op_at[pc] == "MUFU.EX2"
                    elif cname == "fma":
                        hit = op_at[pc] == "FFMA"
                    elif cname == "shuffle":
                        hit = op_at[pc].startswith("SHFL.")
                    elif cname == "barrier":
                        hit = op_at[pc].startswith("BAR.SYNC")
                    elif cname == "shared_load":
                        hit = op_at[pc].split(".")[0] == "LDS"
                    else:
                        hit = op_at[pc].split(".")[0] == "STS"
                    if hit:
                        mine += n * blocks
                check[cname] = {"recomputed": mine, "frozen_derivation": ref_c["predicate_true_thread_instruction"],
                                "equal": mine == ref_c["predicate_true_thread_instruction"]}
            work["six_class_counts_from_frozen_derivation"] = {k: v["predicate_true_thread_instruction"] for k, v in ref_row["classes"].items()}
            work["six_class_width_histograms_from_frozen_derivation"] = {k: v["width_bits_histogram"] for k, v in ref_row["classes"].items()}
            work["six_class_cross_check"] = check
            work["six_class_cross_check_all_equal"] = all(v["equal"] for v in check.values())
            work["source_of_six_class_counts"] = "collective_counts.json" if collective else "candidate_counts.json"
            work["missing_instruction_families_in_frozen_derivation"] = sorted(ref_row.get("missing_instruction_families", {}))
            work["note"] = ("lane-level predicate-true instruction counts per launch; the frozen derivation does not infer warp-issued counts. "
                            "Opcodes listed in opcodes_with_unknown_guard_upper_bound had at least one instance whose guard predicate "
                            "was not decidable; such totals are upper bounds.")
            rec["work"] = work
            if not work["six_class_cross_check_all_equal"]:
                missing.append({"feature": "work.six_class_counts", "reason": "recomputed class counts differ from the frozen derivation"})
            # --- structure
            rec["structure"] = loop_features(sites, lane_visits, threads)
            rec["structure"].update(barrier_features(sites, lane_visits, threads, blocks))
            # --- memory
            mem = dev_memory(d)
            mem["executed_global_load_bytes_lane_level"] = work["executed_global_load_bytes"]
            mem["executed_async_copy_global_to_shared_bytes_lane_level"] = work["executed_async_copy_global_to_shared_bytes"]
            mem["executed_global_store_bytes_lane_level"] = work["executed_global_store_bytes"]
            mem["executed_global_bytes_per_thread"] = work["executed_global_bytes_per_thread"]
            mem["executed_global_bytes_exact"] = work["executed_global_bytes_exact"]
            mem["logical_bytes_per_thread"] = (mem["logical_bytes_per_launch"] / (threads * blocks)) if mem["logical_bytes_per_launch"] is not None else None
            mem["global_load_width_bits_histogram"] = work["families"].get("global_load", {}).get("width_bits_histogram", {})
            mem["global_store_width_bits_histogram"] = work["families"].get("global_store", {}).get("width_bits_histogram", {})
            mem["async_copy_width_bits_histogram"] = work["families"].get("async_copy_global_to_shared", {}).get("width_bits_histogram", {})
            mem["tier_source_note"] = "tier is the development table's analytical footprint-over-L2 rule, not a counter"
            rec["memory"] = mem
            if mem["logical_bytes_per_launch"] is None or mem["tier"] is None:
                missing.append({"feature": "memory.logical_bytes/tier", "reason": "absent from development table"})
            rec["status"] = "supported" if not missing else "missing_features"
            rec["missing_features"] = missing
            rows_out.append(rec)
    # Cells without retained binaries.
    for cell_id in sorted(set(dev) - seen_dev):
        d = dev[cell_id]
        miss = [{"feature": f, "reason": "no retained binary for this cell"} for f in
                ("resources", "occupancy", "work", "structure")]
        rows_out.append({"cell_id": cell_id, "operator_id": d["operator_id"], "cell": "%s/%s" % (d["regime"], d["candidate_id"]),
                         "corpus": None, "status": "no_retained_binary", "missing_features": miss,
                         "geometry": dev_geometry(d, hw), "memory": dev_memory(d)})
    # PyTorch cells (no retained cubin): replace the placeholder rows by features from the libtorch sm_120 SASS.
    pt_rows = build_pytorch_rows(hw, dev)
    rows_out = [pt_rows.get(r["cell_id"], r) if r["status"] == "no_retained_binary" else r for r in rows_out]
    rows_out.sort(key=lambda r: r["cell_id"])
    return rows_out


def coverage(rows):
    out = collections.Counter(r["status"] for r in rows)
    by_op = collections.defaultdict(collections.Counter)
    for r in rows:
        by_op[r["operator_id"]][r["status"]] += 1
    return {"total_rows": len(rows), "by_status": dict(out), "by_operator": {k: dict(v) for k, v in sorted(by_op.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    hw_text = read_text(GROUND_TRUTH)
    hw = load_hardware(hw_text)
    rows = build_rows(hw)
    self_sha = sha_bytes(Path(__file__).read_bytes())
    doc = {
        "schema": "static_runtime_features_blackwell/1",
        "note": ("Label-free static features. No runtime, energy or power column was read. Lane-level counts are not warp-issued counts. "
                 "Derivation outputs are candidates, not profiler-validated or scientifically admitted."),
        "extract_features_py_sha256": self_sha,
        "hardware_from_ground_truth": hw,
        "hardware_constants_not_in_ground_truth": list(HW_NOT_IN_GROUND_TRUTH),
        "coverage": coverage(rows),
        "input_sha256": dict(sorted(READ_HASHES.items())),
        "rows": rows,
    }
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc["coverage"], indent=1))


if __name__ == "__main__":
    main()
