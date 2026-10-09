"""Static pipeline for fresh set D (CUDA Samples kernels at new geometries, Blackwell), CPU only.

Binds the frozen scripts (operator_derivation/derive.py and collective.py, coalescing/coalescing.py, phases.py,
reuse/phases_unique.py, extract_features.py) to NEW cells WITHOUT editing them:

* Evidence (cubin, isolated SASS, source, build record, compile events) is the retained one of a development cell that
  launches the same kernel. `derive.verify()` therefore still checks every hash against that development row, which keeps
  its own identity (operator_id and cell). The new cell id lives only in this package's tables.
* `derive.binding()` is the one function that is hard-wired to the development grid (geometry read from the development
  manifests, the copy-family `n` taken from the regime name). It is replaced, in every loaded copy of the derivation, by
  `bind()` below: rows carrying a `_fresh` marker are bound from explicit geometry (constant-bank offsets identical to the
  frozen function), every other row falls through to the frozen function unchanged.
* The cross-check of `phases.retained()` against the frozen full-launch coalescing totals is satisfied with the totals of
  `coalescing.analyse_cell()` run on the same new cell (the frozen file has no such row).
* Features reuse the loop body of `extract_features.build_rows()` (copied, with the development-table values replaced by
  the fresh cell table and the frozen derivation counts replaced by the derivation run on the new cell).

No measured runtime, energy or power value is read anywhere. The development table is read through
`extract_features.load_dev_rows()`, whose column whitelist excludes every measured column.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR))
sys.path.insert(0, str(SR / "coalescing"))
import phases as PH  # noqa: E402  (imported, never edited; loads coalescing, extract_features, pytorch adapter)

C = PH.C
X = PH.X
D = C.D                      # derive.py as loaded by coalescing.py
COL = C.COL                  # collective.py as loaded by coalescing.py
D_X, COL_X = X.load_derivation()   # the copies extract_features.interpret_cell() uses (re-registers 'derive')
Refusal = D.Refusal
REPO = X.REPO
OPD = X.DERIV_DIR
PU_PATH = SR / "reuse" / "phases_unique.py"
CUDA_ROOT = D.CORPORA["cuda"]

SOURCE_REVISION = "5443602d89ed99aede2e4b7bf329daddeadb320e"
TILE = 32

# --------------------------------------------------------------------------------------------- cell table
# kernel -> (family, development operator whose retained evidence is reused, development candidate id of that kernel)
KERNELS = {
    "vecAdd": ("vecadd", "final_cuda_samples_copy", "c1"),
    "transposeNaive": ("tile", "train_cuda_samples_transpose", "c1"),
    "transposeCoalesced": ("tile", "train_cuda_samples_transpose", "c2"),
    "transposeNoBankConflicts": ("tile", "train_cuda_samples_transpose", "c3"),
    "transposeDiagonal": ("tile", "train_cuda_samples_transpose", "c4"),
    "copy": ("tile", "alt_cuda_samples_copy", "c1"),
    "copySharedMem": ("tile", "alt_cuda_samples_copy", "c2"),
    "transposeFineGrained": ("tile", "alt_cuda_samples_transposefine", "c1"),
    "transposeCoarseGrained": ("tile", "alt_cuda_samples_transposefine", "c2"),
    "reduce0": ("reduction", "alt_cuda_samples_reduction", "c1"),
    "reduce1": ("reduction", "alt_cuda_samples_reduction", "c2"),
    "reduce6": ("reduction", "alt_cuda_samples_reduction", "c3"),
    "reduce2": ("reduction", "train_cuda_samples_reduction", "c1"),
}
# which_kernel number of the pinned reduce<T>() dispatcher; tile variants of the harness drivers
REDUCE_WHICH = {"reduce0": 0, "reduce1": 1, "reduce2": 2, "reduce6": 6}
TILE_VARIANT = {"transposeNaive": 2, "transposeCoalesced": 3, "transposeNoBankConflicts": 4, "transposeDiagonal": 7,
                "copy": 0, "copySharedMem": 1, "transposeCoarseGrained": 5, "transposeFineGrained": 6}

L2_DIMS = 1536      # 18 MiB logical (read + write), ratio 0.14
DRAM_DIMS = 5120    # 200 MiB logical, ratio 1.5625
VEC_N = {"l2": 3_145_728, "dram": 25_165_824}
VEC_THREADS = (128, 1024)
RED_N = {"l2": 6_291_456, "dram": 50_331_648}
# reduce6 is retained only as the nIsPow2 = true instantiation reduce6<float, 256, true> (symbol _Z7reduce6IfLj256ELb1EEvPT_S1_j);
# the pinned dispatcher selects the nIsPow2 = false instantiation (not retained) for any other n, so reduce6 needs powers of two.
RED6_N = {"l2": 8_388_608, "dram": 67_108_864}

# operator id -> (kernels in candidate order, geometry rule)
OPERATORS = {
    "fresh_d_cuda_samples_vector_add": ["vecAdd", "vecAdd"],
    "fresh_d_cuda_samples_transpose": ["transposeNaive", "transposeCoalesced", "transposeNoBankConflicts", "transposeDiagonal"],
    "fresh_d_cuda_samples_copy": ["copy", "copySharedMem"],
    "fresh_d_cuda_samples_transposefine": ["transposeFineGrained", "transposeCoarseGrained"],
    "fresh_d_cuda_samples_reduction": ["reduce0", "reduce1", "reduce6"],
    "fresh_d_cuda_samples_reduce2": ["reduce2"],
}
REGIMES = ("l2", "dram")


def l2_bytes_from_ground_truth():
    hw = X.load_hardware(X.read_text(X.GROUND_TRUTH))
    return hw, hw["l2_bytes"]


def define_cells(l2_bytes):
    """The 28 fresh cells. Geometry is explicit; footprint, bytes and tier are recomputed (never copied)."""
    cells = []

    def add(op, regime, cand, kernel, **g):
        family, origin_op, origin_cand = KERNELS[kernel]
        if family == "vecadd":
            n, threads = g["n"], g["threads"]
            assert n % threads == 0
            blocks = n // threads
            geometry = dict(n=n, threads=threads, blocks=blocks, grid=[blocks, 1, 1], block=[threads, 1, 1])
            logical = 3 * n * 4
            shape_a, shape_b = n, 1
        elif family == "tile":
            dim = g["dim"]
            assert dim % TILE == 0
            geometry = dict(dim_x=dim, dim_y=dim, grid=[dim // TILE, dim // TILE, 1], block=[TILE, 16, 1],
                            blocks=(dim // TILE) ** 2, threads=TILE * 16)
            logical = 2 * dim * dim * 4
            shape_a, shape_b = (dim, dim) if op.endswith("transpose") else ((dim * dim, 1) if op.endswith("copy") else (dim, 1))
        else:
            n = g["n"]
            threads = 256
            blocks = 64 if kernel == "reduce6" else n // threads
            assert n % threads == 0
            geometry = dict(n=n, threads=threads, blocks=blocks, grid=[blocks, 1, 1], block=[threads, 1, 1])
            logical = 4 * n + 4 * blocks
            shape_a, shape_b = n, 1
        footprint = logical
        cells.append(dict(
            cell_id="blackwell/%s/%s/%s" % (op, regime, cand), operator_id=op, regime=regime, candidate_id=cand,
            kernel=kernel, kernel_family=family, origin_operator_id=origin_op, origin_cell="small/" + origin_cand,
            geometry=geometry, shape_a=shape_a, shape_b=shape_b,
            logical_bytes_per_launch=logical, footprint_bytes=footprint,
            footprint_over_l2=footprint / l2_bytes, tier="L2" if footprint / l2_bytes < 1 else "DRAM",
            grid_blocks=geometry["blocks"], block_threads=geometry["threads"]))

    for regime in REGIMES:
        for cand, threads in zip(("c1", "c2"), VEC_THREADS):
            add("fresh_d_cuda_samples_vector_add", regime, cand, "vecAdd", n=VEC_N[regime], threads=threads)
    dims = {"l2": L2_DIMS, "dram": DRAM_DIMS}
    for op in ("fresh_d_cuda_samples_transpose", "fresh_d_cuda_samples_copy", "fresh_d_cuda_samples_transposefine"):
        for regime in REGIMES:
            for i, kernel in enumerate(OPERATORS[op]):
                add(op, regime, "c%d" % (i + 1), kernel, dim=dims[regime])
    for op in ("fresh_d_cuda_samples_reduction", "fresh_d_cuda_samples_reduce2"):
        for regime in REGIMES:
            for i, kernel in enumerate(OPERATORS[op]):
                add(op, regime, "c%d" % (i + 1), kernel, n=(RED6_N if kernel == "reduce6" else RED_N)[regime])
    return cells


# --------------------------------------------------------------------------------------------- binding
def bind_fresh(family, geometry):
    """Same return value and constant-bank offsets as derive.binding() for the CUDA branches, with every scalar explicit."""
    V = D.V
    if family == "tile":
        gx, gy = geometry["grid"][0], geometry["grid"][1]
        threads = geometry["block"][0] * geometry["block"][1]
        blocks = gx * gy
        const = {0x390: gx * 32, 0x394: gy * 32}
        launch = {"grid": [gx, gy, 1], "block": [geometry["block"][0], geometry["block"][1], 1]}
    else:
        threads, blocks, n = geometry["threads"], geometry["blocks"], geometry["n"]
        n_offset = 0x390 if family == "reduction" else 0x398
        const = {0x360: threads, 0x364: 1, 0x368: 1, 0x370: blocks, n_offset: n}
        launch = {"grid": [blocks, 1, 1], "block": [threads, 1, 1], "n": n}
    const = {k: V.exact(v) for k, v in const.items()}
    coords = {"SR_CTAID.X": V(0, launch["grid"][0] - 1), "SR_CTAID.Y": V(0, launch["grid"][1] - 1),
              "SR_CTAID.Z": V.exact(0), "SR_CgaCtaId": V.exact(0)}
    return const, coords, threads, blocks, launch


def _wrap(orig):
    def binding(corpus, row, root):
        fresh = row.get("_fresh")
        if fresh is None:
            return orig(corpus, row, root)
        if corpus != "cuda":
            raise Refusal("fresh binding implemented for CUDA corpus only")
        return bind_fresh(fresh["family"], fresh["geometry"])
    binding._fresh_wrapper = True
    return binding


def install_binding():
    """Replace `binding` in every loaded copy of the derivation (idempotent)."""
    mods = [D, COL, D_X, COL_X, sys.modules["derive"]]
    for m in mods:
        if not getattr(m.binding, "_fresh_wrapper", False):
            m.binding = _wrap(m.binding)


install_binding()


def retained_rows():
    m = json.loads((CUDA_ROOT / "retention_manifest.json").read_text())
    return {(r["operator_id"], r["cell"]): r for r in m["rows"]}


def marked_row(origin_row, family, geometry):
    row = dict(origin_row)
    row["_fresh"] = {"family": family, "geometry": geometry}
    return row


# --------------------------------------------------------------------------------------------- derivation
def derive_classes(row):
    """Plain derivation first; the collective (BRA.DIV) extension only if the plain one refuses, as the frozen tables did.
    Returns (collective_used, result dict). A refusal of both is returned with both exact reasons."""
    try:
        return False, D.derive("cuda", row, CUDA_ROOT)
    except Refusal as plain:
        try:
            return True, COL.derive_collective("cuda", row, CUDA_ROOT)
        except Refusal as coll:
            return True, {"status": "refused", "reason": "plain derivation: %s; collective derivation: %s" % (plain, coll)}


# --------------------------------------------------------------------------------------------- features (copy of build_rows body)
def feature_row(hw, row, cell_id, operator_id, cell_name, d, ref_row, collective):
    """Loop body of extract_features.build_rows() for one retained-corpus cell. `d` has the allowed development-table keys."""
    root, corpus = CUDA_ROOT, "cuda"
    missing = []
    rec = {"cell_id": cell_id, "operator_id": operator_id, "cell": cell_name, "corpus": corpus}
    cubin_b = X.read_bytes(root / row["cubin_path"])
    sass_b = X.read_bytes(root / row["disassembly_path"])
    if X.sha_bytes(cubin_b) != row["cubin_sha256"] or X.sha_bytes(sass_b) != row["disassembly_sha256"]:
        raise Refusal("hash mismatch against retention manifest: " + cell_id)
    rec["inputs"] = {"cubin_sha256": row["cubin_sha256"], "isolated_sass_sha256": row["disassembly_sha256"],
                     "kernel_symbol": row["isolated_function_section"]}
    X.read_bytes(root / "retention_manifest.json")
    for ref in row["retained_compiler_events"]:
        X.read_bytes(root / ref["path"])
    X.read_bytes(root / row["retained_source_path"])
    res = X.parse_cubin_resources(cubin_b, row["isolated_function_section"], hw["reserved_shared_per_block_bytes"])
    sites, lane_visits, events, unknown_pcs, (threads, blocks, launch) = X.interpret_cell(D_X, COL_X, corpus, row, root, collective)
    geom = X.dev_geometry(d, hw)
    grid_total = launch["grid"][0] * launch["grid"][1] * launch["grid"][2]
    geom.update({"grid_blocks_derivation": grid_total, "block_threads_derivation": threads,
                 "geometry_matches_dev_table": (geom["grid_blocks"] == grid_total and geom["block_threads"] == threads)
                 if geom["grid_blocks"] is not None and geom["block_threads"] is not None else None})
    dyn, dyn_src, dyn_reason = X.dynamic_shared(corpus, row, root, launch, threads, res, {})
    res["dynamic_shared_bytes_per_launch"] = dyn
    res["dynamic_shared_source"] = dyn_src
    if dyn_reason:
        res["dynamic_shared_missing_reason"] = dyn_reason
        missing.append({"feature": "dynamic_shared_bytes_per_launch", "reason": dyn_reason})
    hi = X.max_register_index(sites)
    res["max_register_index_in_isolated_sass"] = hi
    res["register_count_consistent_with_sass"] = hi + 1 <= res["registers_per_thread"]
    if res["reqntid"] is not None:
        res["reqntid_matches_launch_threads"] = (res["reqntid"][0] * res["reqntid"][1] * res["reqntid"][2] == threads)
    rec["geometry"] = geom
    rec["resources"] = res
    occ = X.occupancy(hw, res["registers_per_thread"], threads, grid_total, res["static_shared_bytes_per_block"], dyn)
    rec["occupancy"] = occ
    if occ["blocks_per_sm"] is None:
        for r_ in occ["reasons"]:
            missing.append({"feature": "occupancy (blocks_per_sm, waves, tail)", "reason": r_})
    work, pc_visits = X.work_features(sites, lane_visits, unknown_pcs, threads, blocks)
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
        check[cname] = {"recomputed": mine, "frozen_derivation": ref_c["predicate_true_thread_instruction"], "equal": mine == ref_c["predicate_true_thread_instruction"]}
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
    rec["structure"] = X.loop_features(sites, lane_visits, threads)
    rec["structure"].update(X.barrier_features(sites, lane_visits, threads, blocks))
    mem = X.dev_memory(d)
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
    return rec


def fresh_dev_like(cell, sm_count):
    """Row with the keys of load_dev_rows() (allowed development-table columns), filled from the fresh cell table."""
    return {"gpu": "blackwell", "operator_id": cell["operator_id"], "cell_id": cell["cell_id"], "regime": cell["regime"],
            "candidate_id": cell["candidate_id"], "tier": cell["tier"], "bytes_per_launch": str(cell["logical_bytes_per_launch"]),
            "actual_logical_bytes_per_launch": str(cell["logical_bytes_per_launch"]), "grid_blocks": str(cell["grid_blocks"]),
            "block_threads": str(cell["block_threads"]), "blocks_per_sm": "", "active_sm_fraction": "",
            "geometry_source": "tiresias/framework/predictor/fresh_d/fresh_cells_d.json",
            "shape_a": str(cell["shape_a"]), "shape_b": str(cell["shape_b"]), "sm_count": str(sm_count)}


# --------------------------------------------------------------------------------------------- per-cell analysis
def _jsonable(x):
    return json.loads(json.dumps(x, sort_keys=True))


def analyse(spec):
    """spec: dict(cell_id, operator_id, cell_name, family, origin=(oid, cell), geometry, dev_like, derive_ref=None).
    Returns dict(derivation, features, coalescing, phases, unique_entries[, error]). Every table is JSON-roundtripped."""
    PU = _pu()
    hw, _ = l2_bytes_from_ground_truth()
    origin = retained_rows()[tuple(spec["origin"])]
    row = marked_row(origin, spec["family"], spec["geometry"])
    out = {"cell_id": spec["cell_id"]}
    collective, deriv = derive_classes(row)
    out["derivation_path"] = "collective" if collective else "plain"
    out["derivation"] = _jsonable(deriv)
    if deriv["status"] != "candidate_class_counts":
        reason = deriv.get("reason")
        out["features"] = _jsonable({"cell_id": spec["cell_id"], "operator_id": spec["operator_id"], "cell": spec["cell_name"], "corpus": "cuda",
                                     "status": "missing_features", "missing_features": [{"feature": "derivation", "reason": reason}],
                                     "geometry": X.dev_geometry(spec["dev_like"], hw), "memory": X.dev_memory(spec["dev_like"])})
        out["coalescing"] = {"status": "refused", "reason": reason}
        out["phases"] = {"status": "unsupported", "reason": reason, "kernels": []}
        out["unique"] = {"status": "unsupported", "reason": reason, "kernels": []}
        return out
    try:
        frozen_like = C.analyse_cell("cuda", row, CUDA_ROOT)
    except Refusal as ex:
        frozen_like = None
        out["coalescing"] = {"status": "refused", "reason": str(ex)}
    if frozen_like is not None:
        out["coalescing"] = _jsonable(frozen_like)
    try:
        out["features"] = _jsonable(feature_row(hw, row, spec["cell_id"], spec["operator_id"], spec["cell_name"], spec["dev_like"], deriv, collective))
    except Refusal as ex:
        out["features"] = _jsonable({"cell_id": spec["cell_id"], "operator_id": spec["operator_id"], "cell": spec["cell_name"], "corpus": "cuda",
                                     "status": "missing_features", "missing_features": [{"feature": "interpretation", "reason": str(ex)}],
                                     "geometry": X.dev_geometry(spec["dev_like"], hw), "memory": X.dev_memory(spec["dev_like"])})
    if frozen_like is None:
        reason = out["coalescing"]["reason"]
        out["phases"] = {"status": "unsupported", "reason": reason, "kernels": []}
        out["unique"] = {"status": "unsupported", "reason": reason, "kernels": []}
        return out
    try:
        q = PH.retained("cuda", row, CUDA_ROOT, frozen_like)
        out["phases"] = _jsonable({"status": "conditional_static_phases", "kernels": [q]})
    except (C.Refusal, PH.P.Refusal) as ex:
        out["phases"] = {"status": "unsupported", "reason": str(ex), "kernels": []}
        out["unique"] = {"status": "unsupported", "reason": "frozen phase table unsupported: " + str(ex), "kernels": []}
        return out
    # first touch: wraps the same phases.py (second instance inside phases_unique.py), capturing the exact addresses
    del PU.CAPTURED[:]
    try:
        PU.PH.retained("cuda", row, CUDA_ROOT, frozen_like)
        bps = out["features"].get("occupancy", {}).get("blocks_per_sm")
        e = PU.kernel_entry(list(PU.CAPTURED), out["phases"]["kernels"][0], bps)
        ok = e["status"] == "ok"
        out["unique"] = _jsonable({"status": "ok" if ok else e["status"], "reason": None if ok else e["reason"],
                                   "kernels_per_launch": out["features"].get("kernels_per_launch"),
                                   "blocks_per_sm_note": "features occupancy", "kernels": [e] if ok else []})
    except (C.Refusal, PH.P.Refusal) as ex:
        out["unique"] = {"status": "unsupported", "reason": str(ex), "kernels": []}
    return out


_PU = None


def _pu():
    global _PU
    if _PU is None:
        _PU = C._load("phases_unique_for_fresh_d", PU_PATH)
    return _PU


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- regression gate helpers
def dev_spec(cid, dev_rows):
    """Spec that binds a DEVELOPMENT cell through the fresh binding (geometry from the allowed development-table columns)."""
    d = dev_rows[cid]
    oid, cell = d["operator_id"], "%s/%s" % (d["regime"], d["candidate_id"])
    origin = retained_rows()[(oid, cell)]
    kernel = origin["isolated_function_section"]
    sa = X.num(d["shape_a"])
    if oid == "final_cuda_samples_copy":
        family, geom = "vecadd", dict(n=sa, threads=X.num(d["block_threads"]), blocks=X.num(d["grid_blocks"]))
    elif oid in ("train_cuda_samples_reduction", "alt_cuda_samples_reduction"):
        family, geom = "reduction", dict(n=sa, threads=X.num(d["block_threads"]), blocks=X.num(d["grid_blocks"]))
    else:
        dim = math.isqrt(sa) if oid == "alt_cuda_samples_copy" else sa
        family, geom = "tile", dict(grid=[dim // TILE, dim // TILE, 1], block=[TILE, 16, 1])
    if family != "tile":
        geom.update(grid=[geom["blocks"], 1, 1], block=[geom["threads"], 1, 1])
    return dict(cell_id=cid, operator_id=oid, cell_name=cell, family=family, origin=[oid, cell], geometry=geom,
                dev_like={k: d[k] for k in d}, kernel_symbol=kernel)


def load_dev():
    return X.load_dev_rows()


# --------------------------------------------------------------------------------------------- timing description
def timing_spec(cell):
    """How the existing C++ drivers are invoked for this cell (read by the timing script of this package). Pure data."""
    k, g = cell["kernel"], cell["geometry"]
    if cell["kernel_family"] == "vecadd":
        return dict(runner="copy_runner", numeric_args=[g["n"], g["threads"]], input_kind="vecadd_uniform_pair", elements=g["n"],
                    oracle="vector_add", file_args=["x.bin", "y.bin", "out.bin"])
    if cell["kernel_family"] == "reduction":
        return dict(runner="reduction_runner", numeric_args=[g["n"], g["threads"], g["blocks"], REDUCE_WHICH[k]],
                    input_kind="reduction_uniform", elements=g["n"], oracle="sum_double", file_args=["in.bin", "out.bin"])
    n = g["dim_x"]
    if cell["operator_id"].endswith("_transpose"):
        return dict(runner="transpose_runner", numeric_args=[n, n, TILE_VARIANT[k]], input_kind="transpose_mod8192", elements=n * n,
                    oracle="full_transpose", file_args=["in.bin", "out.bin"])
    if cell["operator_id"].endswith("_copy"):
        return dict(runner="tile_family", numeric_args=[n, n, TILE_VARIANT[k]], input_kind="copy_uniform", elements=n * n,
                    oracle="identity", file_args=["in.bin", "out.bin"])
    return dict(runner="tile_family", numeric_args=[n, n, TILE_VARIANT[k]], input_kind="transpose_mod2p24", elements=n * n,
                oracle="kernel_semantics:" + k, file_args=["in.bin", "out.bin"])
