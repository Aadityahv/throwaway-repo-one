#!/usr/bin/env python3
"""Static features for the 27 Blackwell PyTorch cells (CPU only; no GPU, ssh or measured column).

Inputs: the sm_120 SASS and resource lines extracted from libtorch_cuda.so (../libtorch_sm120/), the
dispatch trace (../pytorch_dispatch/dispatch_trace.json) and the extended interpreter (pt_interp.py).
`build_rows(X, hw, dev)` returns {cell_id: row}; `X` is the already-imported extract_features module (so
its READ_HASHES records every input file). Documentation and evidence: ADAPTER.md in this directory.

Nothing is zero-filled: a kernel the interpreter refuses gets null work/structure features with the exact
refusal text, and the row is marked missing_features.
"""
from __future__ import annotations

import collections
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt_interp as P  # noqa: E402

STATIC = HERE.parent
SASS_DIR = STATIC / "libtorch_sm120"
DISPATCH_JSON = STATIC / "pytorch_dispatch/dispatch_trace.json"

KERNEL_REGEX = [
    ("k1", re.compile(r"elementwise_kernel<128, 2, .*direct_copy_kernel_cuda")),
    ("k2", re.compile(r"vectorized_gather_kernel<16, long>")),
    ("k3", re.compile(r"vectorized_layer_norm_kernel<float, float, false>")),
    ("k4", re.compile(r"softmax_warp_forward<float, float, float, 10, false, false>")),
    ("k5", re.compile(r"softmax_warp_forward<float, float, float, 7, false, false>")),
    ("k6", re.compile(r"softmax_warp_forward<float, float, float, 9, false, false>")),
]

# Non-zero, 4096-aligned placeholder for pointer parameters whose only observable effect is a null check.
POINTER_PLACEHOLDERS = (0x7F0000001000, 0x7F0000101000)

# --------------------------------------------------------------------------- parameter ABI
# sm_120 kernel parameters start at c[0x0][0x380] and are packed with natural alignment (ptr/long 8,
# int/float 4, bool 1). Implicit launch constants: blockDim x,y,z at 0x360/0x364/0x368, gridDim x,y,z at
# 0x370/0x374/0x378 (the layout the frozen derivation already uses for CUDA kernels).
PARAM_BASE = 0x380
IMPLICIT = {"blockDim.x": 0x360, "blockDim.y": 0x364, "blockDim.z": 0x368,
            "gridDim.x": 0x370, "gridDim.y": 0x374, "gridDim.z": 0x378}
IMPLICIT_ALSO_ACCESSED = {0x358: "global memory descriptor base (desc[UR][...] operands)", 0x37C: "initial stack pointer"}

SIZES = {"ptr": 8, "i64": 8, "i32": 4, "f32": 4, "bool": 1}


def layout(fields):
    """[(name, type)] -> [(name, offset, size)] with natural alignment from PARAM_BASE."""
    out, off = [], PARAM_BASE
    for name, ty in fields:
        sz = SIZES[ty]
        off = (off + sz - 1) // sz * sz
        out.append((name, off, sz))
        off += sz
    return out


def functor_fields():
    """The elementwise_kernel's by-value lambda: OffsetCalculator<2> (dims, 25 IntDivider{divisor,m1,shift},
    25x2 strides) followed by data[2] pointers. Source: aten/src/ATen/cuda/detail/OffsetCalculator.cuh
    (MAX_DIMS = 25) and CUDALoops.cuh gpu_kernel_impl_nocast, saved in ../pytorch_dispatch/src/."""
    f = [("dims", "i32")]
    for i in range(25):
        f += [("sizes_%d.divisor" % i, "i32"), ("sizes_%d.m1" % i, "i32"), ("sizes_%d.shift" % i, "i32")]
    for i in range(25):
        f += [("strides_%d_0" % i, "i32"), ("strides_%d_1" % i, "i32")]
    f += [("data_0", "ptr"), ("data_1", "ptr")]
    return f


PARAM_FIELDS = {
    "k1": [("N", "i32"), ("_functor_alignment_pad", "i32")] + functor_fields(),
    "k2": [("out", "ptr"), ("inp", "ptr"), ("idx", "ptr"), ("num_ind", "i32"), ("slice_size", "i64"),
           ("ind_dim_size", "i64"), ("inp_stride", "i64"), ("out_stride", "i64"), ("allow_neg_indices", "bool")],
    "k3": [("N", "i32"), ("eps", "f32"), ("X", "ptr"), ("gamma", "ptr"), ("beta", "ptr"), ("mean", "ptr"),
           ("rstd", "ptr"), ("Y", "ptr")],
    "softmax": [("dst", "ptr"), ("src", "ptr"), ("batch_size", "i32"), ("stride", "i32"), ("element_count", "i32"),
                ("mask", "ptr"), ("head_chunk_size", "i32"), ("is_transformer_mask", "bool")],
}
for _k in ("k4", "k5", "k6"):
    PARAM_FIELDS[_k] = PARAM_FIELDS["softmax"]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def abi_layout(kid):
    return layout(PARAM_FIELDS[kid])


def sass_constant_accesses(sites):
    """[(offset, nbytes)] of every c[0x0][...] operand in the SASS (width from the opcode)."""
    out = set()
    for s in sites:
        for tok in s.a:
            m = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-f]+)\]", tok)
            if not m:
                continue
            off = int(m.group(1), 0)
            n = 16 if ".128" in s.op else 8 if ".64" in s.op else 1 if ".U8" in s.op else 4
            out.add((off, n))
    return sorted(out)


def abi_check(kid, sites):
    """Every constant-bank access must be an implicit launch constant or land on declared fields: the access
    must start at a field start and cover whole fields. Returns (ok, problems)."""
    fields = abi_layout(kid)
    starts = {o: (n, sz) for n, o, sz in fields}
    covered = set()
    for _, o, sz in fields:
        covered.update(range(o, o + sz))
    implicit = set(IMPLICIT.values()) | set(IMPLICIT_ALSO_ACCESSED)
    problems = []
    for off, n in sass_constant_accesses(sites):
        if off in implicit:
            continue
        if off not in starts:
            problems.append("c[0x0][%#x] (%d B) does not start at a declared field" % (off, n))
            continue
        if n >= 4 and not all(b in covered for b in range(off, off + n)):
            problems.append("c[0x0][%#x] (%d B) extends past the declared fields" % (off, n))
            continue
        if n == 1 and starts[off][1] != 1:
            problems.append("c[0x0][%#x] byte access to a %d-byte field" % (off, starts[off][1]))
    return not problems, problems


# --------------------------------------------------------------------------- assumptions
ASSUMPTION_TEXT = {
    "division_fast_path": (
        "Division fast path. Each FCHK-guarded division (`FCHK Pd,a,b` then `@!Pd BRA fast`) is assumed to take the fast "
        "sequence on every lane, i.e. the slow-path subroutine (reached only for operands outside the normal range) is "
        "never entered. The slow-path subroutine contains no shuffle, shared-memory, MUFU.EX2 or barrier instruction, so "
        "only FFMA and the integer/float opcode counts can be too low, and only for lanes whose operands leave the normal "
        "range. For the measured inputs (softmax_input: values in [-3.35, 3.35]) every softmax numerator is in "
        "[exp(-6.7), 1] and every denominator in [1, cols], and layer-norm variances are positive and finite, so no lane "
        "should leave the fast path; this is an argument about input values, not an observation."),
    "bounds_assert_not_taken": (
        "Gather bounds check. The data-dependent `0 <= index < ind_dim_size` device assert (a `@!P0 BRA P1` over a "
        "CALL to the assert routine) is assumed not to fire. The measured runs completed (an assert would abort the "
        "process) and the runner rejects out-of-range indices, so the assert path is not executed."),
}


def sass_shape_checks(kid, sites):
    """Verify the instruction patterns the assumptions rely on. Returns a list of problems (empty = ok)."""
    problems = []
    by = {s.pc: s for s in sites}
    for s in sites:
        if s.op == "FCHK":
            pd = s.a[0]
            nxt = [t for t in sites if s.pc < t.pc <= s.pc + 16 * 12 and t.op == "BRA" and t.pred == "!" + pd]
            if not nxt:
                problems.append("FCHK at %#x is not followed by `@!%s BRA` within 12 instructions" % (s.pc, pd))
                continue
            br = nxt[0]
            between = [t for t in sites if s.pc < t.pc < int(br.a[-1], 0)]
            if not any(t.op.startswith("CALL") for t in between):
                problems.append("FCHK at %#x: slow path (CALL) not between the check and the fast-path target" % s.pc)
    return problems


def slow_path_ops(sites):
    """Opcode set of the division slow-path subroutine(s): from each CALL.REL target to its RET."""
    ops = collections.Counter()
    for s in sites:
        if s.op.startswith("CALL.REL"):
            tgt = int(s.a[-1], 0)
            pc = tgt
            while pc in {t.pc for t in sites}:
                t = next(x for x in sites if x.pc == pc)
                ops[t.op] += 1
                if t.op.startswith("RET"):
                    break
                pc += 16
    return ops


# --------------------------------------------------------------------------- bindings
def kernel_id(kern):
    name = kern["demangled_name_regex"]
    hits = [kid for kid, rx in KERNEL_REGEX if rx.search(name)]
    if len(hits) != 1:
        raise SystemExit("cannot identify kernel from dispatch regex %r -> %r" % (name, hits))
    return hits[0]


def binding(kid, kern):
    """(constants {offset: int}, grid, block, pointer_params, provenance) for one dispatched kernel."""
    a = kern["args"]
    grid, block = list(kern["grid"]), list(kern["block"])
    c = {IMPLICIT["blockDim.x"]: block[0], IMPLICIT["blockDim.y"]: block[1], IMPLICIT["blockDim.z"]: block[2],
         IMPLICIT["gridDim.x"]: grid[0], IMPLICIT["gridDim.y"]: grid[1], IMPLICIT["gridDim.z"]: grid[2]}
    off = {n: o for n, o, _ in abi_layout(kid)}
    prov = {}

    def put(name, value, source):
        o = off[name]
        c[o] = value & 0xFFFFFFFF
        prov[name] = {"offset": hex(o), "value": value, "source": source}
        if dict((n, s) for n, _, s in abi_layout(kid))[name] == 8:
            c[o + 4] = (value >> 32) & 0xFFFFFFFF
            prov[name]["high_word_offset"] = hex(o + 4)

    ptrs = []
    if kid in ("k4", "k5", "k6"):
        put("batch_size", a["batch_size"], "dispatch args.batch_size")
        put("stride", a["stride"], "dispatch args.stride")
        put("element_count", a["element_count"], "dispatch args.element_count")
        put("mask", 0, "dispatch args.mask = null")
        put("head_chunk_size", a["head_chunk_size"], "dispatch args.head_chunk_size")
        put("is_transformer_mask", int(a["is_transformer_mask"]), "dispatch args.is_transformer_mask")
        ptrs = ["dst", "src"]
    elif kid == "k3":
        put("N", a["N"], "dispatch args.N")
        put("eps", f32_bits(a["eps"]), "dispatch args.eps rounded to binary32")
        require(a["gamma_defined"] and a["beta_defined"], "layer-norm binding needs gamma and beta defined")
        put("gamma", POINTER_PLACEHOLDERS[0], "placeholder non-null pointer (gamma_defined)")
        put("beta", POINTER_PLACEHOLDERS[1], "placeholder non-null pointer (beta_defined)")
        ptrs = ["X", "mean", "rstd", "Y"]
    elif kid == "k2":
        put("num_ind", a["num_ind"], "dispatch args.num_ind")
        put("slice_size", a["slice_size_bytes"], "dispatch args.slice_size_bytes")
        put("ind_dim_size", a["ind_dim_size"], "dispatch args.ind_dim_size")
        put("inp_stride", a["inp_stride_bytes"], "dispatch args.inp_stride_bytes")
        put("out_stride", a["out_stride_bytes"], "dispatch args.out_stride_bytes")
        put("allow_neg_indices", int(a["allow_neg_indices"]), "dispatch args.allow_neg_indices")
        ptrs = ["out", "inp", "idx"]
    elif kid == "k1":
        put("N", a["N"], "dispatch args.N")
        dims = len(re.search(r"\[([^\]]*)\]", a["offset_calculator"]).group(1).split(","))
        put("dims", dims, "dispatch args.offset_calculator: number of dims in 'over dims [..]'")
        ptrs = ["data_0", "data_1"]
    else:
        raise SystemExit("no binding for " + kid)
    return c, grid, block, ptrs, prov


def require(ok, msg):
    if not ok:
        raise SystemExit(msg)


# --------------------------------------------------------------------------- running
def run_kernel(D, kid, sites, consts, grid, block, extra=None, assumptions=True):
    """Extended interpreter on one block (all lanes, lock-step collectives). Returns a dict."""
    forced = {}
    if kid == "k2" and assumptions:
        forced = {0x150: True}
        s = {t.pc: t for t in sites}
        sh = s[0x150]
        require(sh.op == "BRA" and sh.pred == "!P0" and sh.a == ("P1", "0x260"), "k2 bounds-check branch shape changed")
        require(s[0x250].op.startswith("CALL"), "k2 assert call shape changed")
    I = P.Interp(D, ext=True, fchk_fast_path=assumptions, forced_branches=forced, rcp_ulp_delta=(extra or {}).get("rcp", 0))
    cm = {o: P.V.exact(v) for o, v in consts.items()}
    coords = {"SR_CTAID.X": P.V(0, grid[0] - 1), "SR_CTAID.Y": P.V(0, grid[1] - 1), "SR_CTAID.Z": P.V.exact(0),
              "SR_CgaCtaId": P.V.exact(0)}
    threads = block[0] * block[1] * block[2]

    def lane(l):
        return {**coords, "SR_TID.X": P.V.exact(l % block[0]), "SR_TID.Y": P.V.exact(l // block[0]),
                "SR_TID.Z": P.V.exact(0), "SR_LANEID": P.V.exact(l % 32)}
    try:
        res = I.run_block(sites, cm, lane, threads)
    except P.Refusal as ex:
        return {"status": "refused", "reason": str(ex)}
    visits = [v for _, v, _ in res]
    events = collections.Counter()
    for e, _, _ in res:
        events.update(e)
    return {"status": "ok", "visits": visits, "events": events, "unknown_pcs": set(I.unknown_pcs),
            "assumption_log": {"%s@%#x" % k: v for k, v in sorted(I.assumption_log.items())},
            "shared_addresses": (min(I.shared_touched), max(I.shared_touched)) if I.shared_touched else None,
            "threads": threads}


def strict_outcome(D, text, consts, grid, block):
    """Outcome of the UNMODIFIED frozen derive.run_lane on this kernel with the same bindings: first lane that refuses."""
    cm = {o: D.V.exact(v) for o, v in consts.items()}
    coords = {"SR_CTAID.X": D.V(0, grid[0] - 1), "SR_CTAID.Y": D.V(0, grid[1] - 1), "SR_CTAID.Z": D.V.exact(0),
              "SR_CgaCtaId": D.V.exact(0)}
    dsites = D.parse(text)
    threads = block[0] * block[1] * block[2]
    for l in range(threads):
        c = {**coords, "SR_TID.X": D.V.exact(l % block[0]), "SR_TID.Y": D.V.exact(l // block[0]), "SR_TID.Z": D.V.exact(0),
             "SR_LANEID": D.V.exact(l % 32)}
        try:
            D.run_lane(dsites, cm, c)
        except D.Refusal as ex:
            return {"status": "refused", "reason": str(ex), "first_refusing_lane": l}
    return {"status": "ok"}


SIX_CLASSES = ("shuffle", "shared_load", "shared_store", "exponential", "fma", "barrier")


def six_class(events, blocks):
    out = {c: {"predicate_true_thread_instruction": 0, "width_bits_histogram": {}} for c in SIX_CLASSES}
    for (pc, name, width), n in events.items():
        q = out[name]
        q["predicate_true_thread_instruction"] += n * blocks
        if width is not None:
            q["width_bits_histogram"][str(width)] = q["width_bits_histogram"].get(str(width), 0) + n * blocks
    return out


def recompute_six_class(sites, pc_visits, blocks):
    op_at = {s.pc: s.op for s in sites}
    out = {}
    for cname in SIX_CLASSES:
        tot = 0
        for pc, n in pc_visits.items():
            op = op_at[pc]
            hit = (op == "MUFU.EX2" if cname == "exponential" else op == "FFMA" if cname == "fma"
                   else op.startswith("SHFL.") if cname == "shuffle" else op.startswith("BAR.SYNC") if cname == "barrier"
                   else op.split(".")[0] == "LDS" if cname == "shared_load" else op.split(".")[0] == "STS")
            if hit:
                tot += n * blocks
        out[cname] = tot
    return out


RES_RE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+) CONSTANT\[0\]:(\d+)")


def kernel_features(X, D, hw, kid, role, kern, idx_rec, sites, run, strict, consts, prov, abi_ok, abi_problems,
                    sass_sha, dev_memory, dynamic_smem):
    """One kernel's feature block in the single-kernel schema of the retained rows."""
    grid_total = kern["grid"][0] * kern["grid"][1] * kern["grid"][2]
    threads = kern["block"][0] * kern["block"][1] * kern["block"][2]
    m = RES_RE.search(idx_rec["res_usage_line"])
    require(m is not None, "unparsed resource line for " + kid)
    reg, stack, shared_raw, local, c0 = (int(x) for x in m.groups())
    require(reg == idx_rec["registers_per_thread"] and shared_raw == idx_rec["static_shared_bytes"], "index fields disagree")
    reserved = hw["reserved_shared_per_block_bytes"]
    # SHARED decision: see ADAPTER.md. cuobjdump's SHARED = reserve (+ static bytes). Static = SHARED - reserve if any
    # shared-memory instruction addresses below reserve + dynamic, else the same formula; both give 0 for these kernels.
    static_shared = max(0, shared_raw - reserved)
    hi = X.max_register_index(sites)
    block = {
        "kernel_id": kid, "role": role, "kernel_symbol": idx_rec["mangled"], "isolated_sass_sha256": sass_sha,
        "geometry": {"grid_blocks": grid_total, "block_threads": threads, "grid": kern["grid"], "block": kern["block"],
                     "geometry_source": "pytorch_dispatch/dispatch_trace.json (static host-code analysis, not observed)"},
    }
    res = {"registers_per_thread": reg, "frame_size_bytes": stack, "min_stack_size_bytes": None,
           "local_memory_bytes_per_thread": local, "shared_section_bytes_raw": shared_raw,
           "static_shared_bytes_per_block": static_shared, "reqntid": None, "max_threads_attr": None,
           "maxreg_count_attr": None, "constant_bank0_bytes": c0,
           "dynamic_shared_bytes_per_launch": dynamic_smem,
           "dynamic_shared_source": "dispatch_trace.json dynamic_smem_bytes (host launch rule: threads.y*3/2*4 for layer norm)" if dynamic_smem else
                                    "dispatch_trace.json dynamic_smem_bytes = 0 (no extern __shared__ in the kernel source)",
           "max_register_index_in_isolated_sass": hi, "register_count_consistent_with_sass": hi + 1 <= reg,
           "source_of_resources": "cuobjdump --dump-resource-usage line in libtorch_sm120/isolated_index.json (no cubin ELF retained for libtorch kernels); STACK is cuobjdump's STACK field",
           "shared_decision": "SHARED:%d is the %d-byte driver reserve, not declared static shared memory (ADAPTER.md)" % (shared_raw, reserved) if shared_raw else "no shared section"}
    block["resources"] = res
    occ = X.occupancy(hw, reg, threads, grid_total, static_shared, dynamic_smem)
    # sensitivity: occupancy if the whole 1,024 B had been static shared memory
    occ_alt = X.occupancy(hw, reg, threads, grid_total, shared_raw, dynamic_smem)
    occ["blocks_per_sm_if_shared_were_static"] = occ_alt.get("blocks_per_sm")
    occ["shared_decision_changes_occupancy"] = occ_alt.get("blocks_per_sm") != occ.get("blocks_per_sm")
    block["occupancy"] = occ
    missing = []
    if occ["blocks_per_sm"] is None:
        for r_ in occ["reasons"]:
            missing.append({"feature": "occupancy (blocks_per_sm, waves, tail)", "reason": r_})
    block["abi"] = {"parameter_fields": [{"name": n, "offset": hex(o), "bytes": s} for n, o, s in abi_layout(kid) if kid != "k1" or not n.startswith(("sizes_", "strides_"))],
                    "all_constant_bank_accesses_on_declared_fields": abi_ok, "problems": abi_problems,
                    "bound_values": prov}
    block["interpreter_outcomes"] = {"frozen_derive_run_lane_unmodified": strict}
    if run["status"] != "ok":
        block["interpreter_outcomes"]["extended_interpreter"] = {"status": "refused", "reason": run["reason"]}
        for f in ("work", "structure"):
            block[f] = None
            missing.append({"feature": f, "reason": "extended interpreter refused: " + run["reason"]})
        block["missing_features"] = missing
        return block
    blocks = grid_total
    work, pc_visits = X.work_features(sites, run["visits"], run["unknown_pcs"], threads, blocks)
    sc = six_class(run["events"], blocks)
    work["six_class_counts"] = {k: v["predicate_true_thread_instruction"] for k, v in sc.items()}
    work["six_class_width_histograms"] = {k: v["width_bits_histogram"] for k, v in sc.items()}
    chk = recompute_six_class(sites, pc_visits, blocks)
    work["six_class_cross_check"] = {k: {"from_events": work["six_class_counts"][k], "from_visits": chk[k],
                                         "equal": work["six_class_counts"][k] == chk[k]} for k in SIX_CLASSES}
    work["six_class_cross_check_all_equal"] = all(v["equal"] for v in work["six_class_cross_check"].values())
    work["source_of_six_class_counts"] = ("pytorch_features/pt_interp.py (extended interpreter); the frozen derivation refuses these "
                                          "kernels, so there is no frozen reference")
    work["six_class_counts_from_frozen_derivation"] = None
    work["float_variant_lane_counts"] = {o: n for o, n in work["opcode_lane_counts"].items()
                                         if o.startswith(("FFMA", "FADD", "FMUL", "FSEL", "FSETP", "MUFU"))}
    work["note"] = ("lane-level predicate-true instruction counts per launch of this kernel; counts are CONDITIONAL on "
                    "`assumptions_applied` and are not warp-issued counts. FFMA modifier variants (.SAT/.RM/.RZ/.RP) are in "
                    "opcode_lane_counts and float_variant_lane_counts, not in the six-class fma count (FFMA only), and "
                    "families.other.")
    work["assumptions_applied"] = sorted({{"fchk_fast_path": "division_fast_path", "forced_branch": "bounds_assert_not_taken"}[k.split("@")[0]]
                                          for k in run["assumption_log"]})
    work["assumption_log_lane_visits"] = run["assumption_log"]
    work["upper_bound_note"] = ("opcodes_with_unknown_guard_upper_bound lists opcodes with a guard predicate the interpreter could "
                                "not decide; they are counted as executed")
    block["work"] = work
    if not work["six_class_cross_check_all_equal"]:
        missing.append({"feature": "work.six_class_counts", "reason": "event counts differ from visit-based recomputation"})
    block["structure"] = X.loop_features(sites, run["visits"], threads)
    block["structure"].update(X.barrier_features(sites, run["visits"], threads, blocks))
    mem = dict(dev_memory)
    mem.update({"executed_global_load_bytes_lane_level": work["executed_global_load_bytes"],
                "executed_async_copy_global_to_shared_bytes_lane_level": work["executed_async_copy_global_to_shared_bytes"],
                "executed_global_store_bytes_lane_level": work["executed_global_store_bytes"],
                "executed_global_bytes_per_thread": work["executed_global_bytes_per_thread"],
                "executed_global_bytes_exact": work["executed_global_bytes_exact"],
                "global_load_width_bits_histogram": work["families"].get("global_load", {}).get("width_bits_histogram", {}),
                "global_store_width_bits_histogram": work["families"].get("global_store", {}).get("width_bits_histogram", {})})
    block["memory_executed"] = mem
    block["interpreter_outcomes"]["extended_interpreter"] = {"status": "ok", "assumptions_applied": work["assumptions_applied"],
                                                             "shared_addresses_touched_min_max": run["shared_addresses"]}
    block["missing_features"] = missing
    return block


def sum_counter_dicts(ds):
    out = collections.Counter()
    for d in ds:
        out.update(d)
    return dict(sorted(out.items()))


def per_launch_totals(blocks):
    """Sum of work over the kernels of one launch; occupancy/waves stay per kernel."""
    live = [b for b in blocks if b.get("work")]
    if len(live) != len(blocks):
        return None
    fam: dict = {}
    for b in live:
        for k, v in b["work"]["families"].items():
            f = fam.setdefault(k, {"lane_instructions": 0, "exact": True})
            f["lane_instructions"] += v["lane_instructions"]
            f["exact"] = f["exact"] and v["exact"]
            for h in ("width_bits_histogram", "mode_histogram"):
                if h in v:
                    t = f.setdefault(h, {})
                    for kk, vv in v[h].items():
                        t[kk] = t.get(kk, 0) + vv
    return {
        "kernels_per_launch": len(blocks),
        "total_lane_instructions": sum(b["work"]["total_lane_instructions"] for b in live),
        "six_class_counts": {c: sum(b["work"]["six_class_counts"][c] for b in live) for c in SIX_CLASSES},
        "opcode_lane_counts": sum_counter_dicts(b["work"]["opcode_lane_counts"] for b in live),
        "families": dict(sorted(fam.items())),
        "executed_global_load_bytes_lane_level": sum(b["work"]["executed_global_load_bytes"] for b in live),
        "executed_global_store_bytes_lane_level": sum(b["work"]["executed_global_store_bytes"] for b in live),
        "all_counts_exact": all(b["work"]["all_counts_exact"] for b in live),
        "barrier_releases_per_launch_estimate": sum(b["structure"]["barrier_releases_per_launch_estimate"] for b in live),
        "grid_blocks_sum": sum(b["geometry"]["grid_blocks"] for b in live),
        "assumptions_applied": sorted({a for b in live for a in b["work"]["assumptions_applied"]}),
        "note": "sum over the kernels of one measured launch; occupancy, waves and tail stay per kernel in each kernel block",
    }


def build_rows(X, hw, dev):
    """{cell_id: row} for the PyTorch cells. `dev` is X.load_dev_rows()."""
    D, _ = X.load_derivation()
    disp = json.loads(X.read_text(DISPATCH_JSON))
    idx = json.loads(X.read_text(SASS_DIR / "isolated_index.json"))
    for p in ("README.md", "cuobjdump_version.txt", "libtorch_cuda.sha256", "index.txt"):
        X.read_bytes(SASS_DIR / p)
    X.read_bytes(STATIC / "pytorch_dispatch/DISPATCH.md")
    libtorch_sha = (SASS_DIR / "libtorch_cuda.sha256").read_text().split()[0]
    sass, sass_text = {}, {}
    for kid, rec in idx.items():
        b = X.read_bytes(SASS_DIR / (kid + ".isolated.sass"))
        if X.sha_bytes(b) != rec["isolated_sha256"]:
            raise SystemExit("isolated SASS hash mismatch for " + kid)
        sass[kid] = P.parse(b.decode())
        sass_text[kid] = b.decode()
    for p in (HERE / "pt_interp.py", HERE / "adapter.py"):
        X.read_bytes(p)
    problems = {kid: sass_shape_checks(kid, s) for kid, s in sass.items()}
    abi = {kid: abi_check(kid, s) for kid, s in sass.items()}
    cache = {}
    rows = {}
    for cell in disp["cells"]:
        cid = cell["cell_id"]
        if cid not in dev:
            raise SystemExit("dispatch cell not in development table: " + cid)
        d = dev[cid]
        kblocks = []
        for kern in cell["kernels"]:
            kid = kernel_id(kern)
            consts, grid, block, ptrs, prov = binding(kid, kern)
            key = (kid, tuple(sorted(consts.items())), tuple(grid), tuple(block))
            if key not in cache:
                cache[key] = (run_kernel(D, kid, sass[kid], consts, grid, block),
                              strict_outcome(D, sass_text[kid], consts, grid, block),
                              run_kernel(D, kid, sass[kid], consts, grid, block, assumptions=False))
            run, strict = cache[key][:2]
            role = "copy" if kid == "k1" else "main"
            blk = kernel_features(X, D, hw, kid, role, kern, idx[kid], sass[kid], run, strict, consts, prov,
                                  abi[kid][0], abi[kid][1], idx[kid]["isolated_sha256"], X.dev_memory(d),
                                  kern["dynamic_smem_bytes"])
            na = cache[key][2]
            blk["interpreter_outcomes"]["extended_interpreter_without_assumptions"] = (
                {"status": "ok"} if na["status"] == "ok" else {"status": "refused", "reason": na["reason"]})
            blk["shape_problems"] = problems[kid]
            blk["dispatch_role_text"] = kern["role"]
            kblocks.append(blk)
        main = kblocks[-1]
        secondary = kblocks[:-1]
        rec = {"cell_id": cid, "operator_id": d["operator_id"], "cell": "%s/%s" % (d["regime"], d["candidate_id"]),
               "corpus": "libtorch_sm120", "kernels_per_launch": len(kblocks)}
        rec["inputs"] = {"isolated_sass_sha256": main["isolated_sass_sha256"], "kernel_symbol": main["kernel_symbol"],
                         "libtorch_cuda_so_sha256": libtorch_sha, "pytorch_commit": disp["pytorch_commit"],
                         "dispatch_trace_confidence": cell.get("confidence"),
                         "secondary_kernel_symbols": [s["kernel_symbol"] for s in secondary]}
        geom = X.dev_geometry(d, hw)
        geom.update({"grid_blocks_dispatch": main["geometry"]["grid_blocks"], "block_threads_dispatch": main["geometry"]["block_threads"],
                     "geometry_source_dispatch": main["geometry"]["geometry_source"]})
        rec["geometry"] = geom
        rec["resources"] = main["resources"]
        rec["occupancy"] = main["occupancy"]
        rec["work"] = main["work"]
        rec["structure"] = main["structure"]
        mem = X.dev_memory(d)
        if main.get("memory_executed"):
            mem.update({k: v for k, v in main["memory_executed"].items() if k.startswith(("executed_", "global_"))})
            mem["logical_bytes_per_thread"] = (mem["logical_bytes_per_launch"] / (main["geometry"]["grid_blocks"] * main["geometry"]["block_threads"])
                                               if mem["logical_bytes_per_launch"] is not None else None)
        mem["tier_source_note"] = "tier is the development table's analytical footprint-over-L2 rule, not a counter"
        mem["executed_bytes_scope"] = "main kernel only; per-launch sums are in per_launch_totals"
        rec["memory"] = mem
        rec["main_kernel"] = {k: main[k] for k in ("kernel_id", "role", "kernel_symbol", "abi", "interpreter_outcomes", "shape_problems", "dispatch_role_text")}
        rec["secondary_kernels"] = secondary
        rec["per_launch_totals"] = per_launch_totals(kblocks)
        rec["assumptions"] = [{"id": k, "text": ASSUMPTION_TEXT[k]} for k in sorted({a for b in kblocks if b.get("work")
                                                                                     for a in b["work"]["assumptions_applied"]})]
        rec["assumptions"].append({"id": "dispatch_trace", "text": "Kernel identity, grid, block, dynamic shared size and scalar arguments come from the static dispatch trace "
                                   "(host-code analysis of ATen at commit 70d99e99), not from a profiler. Binary identity with that source is taken from "
                                   "CHANGELOG M443 and the presence of every demangled name in the sm_120 code."})
        if cell.get("confidence") != "high":
            rec["assumptions"].append({"id": "embedding_fast_path_eligibility",
                                       "text": "Dispatch confidence for this cell is '%s': whether the vectorized gather kernel (rather than the generic "
                                               "scatter/gather elementwise kernel) is launched rests on a hand derivation of TensorIterator strides." % cell["confidence"]})
        miss = []
        for b in kblocks:
            for m_ in b["missing_features"]:
                miss.append(dict(m_, kernel=b["kernel_id"]))
        for kid_, pb in ((b["kernel_id"], b["shape_problems"]) for b in kblocks):
            for p_ in pb:
                miss.append({"feature": "assumption shape check", "reason": p_, "kernel": kid_})
        for b in kblocks:
            if not b["abi"]["all_constant_bank_accesses_on_declared_fields"]:
                miss.append({"feature": "parameter ABI", "reason": "; ".join(b["abi"]["problems"]), "kernel": b["kernel_id"]})
        rec["missing_features"] = miss
        rec["status"] = "supported_with_assumptions" if not miss else "missing_features"
        rows[cid] = rec
    return rows
