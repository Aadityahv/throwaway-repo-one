"""Tests of the cross-GPU port extension (CPU only). Run: python test_port_ext.py [--quick]

1. Unit tests of each new opcode and operand form against hand-computed values (from the audited SASS forms).
2. Blackwell unchanged: with the extension installed for sm_120, committed unseen-kernel cells rebuild to exactly the committed
   feature, phase and first-touch rows (build time excluded).
3. Cross-architecture invariants on the provisional A100/H100/Ada builds: for the same kernel and launch, the memory behaviour does
   not depend on the compiler's instruction selection, so per block the global sectors read and written, the global bytes, the
   dynamic barrier count and the set of shared-memory addresses touched must equal the Blackwell build's. Instruction counts may
   differ and are not compared.
No GPU, no measured value.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(HERE))
import port_ext as PX  # noqa: E402

UP, P, A = PX.UP, PX.P, PX.A
# Which compiled set the cross-architecture checks read: the provisional local CUDA 13.0 build (default) or e.g.
# compiled_cluster_cuda12.1 (A100/H100 only), with PORT_ARCHS=sm_90,sm_80 to restrict the architectures.
SASS_SUBDIR = os.environ.get("PORT_SASS_SUBDIR", "compiled_provisional")
_ARCH_FOLDERS = {"sm_90": "port_h100", "sm_80": "port_a100", "sm_89": "port_ada"}
PORT_ARCHS = [(a, _ARCH_FOLDERS[a]) for a in os.environ.get("PORT_ARCHS", "sm_90,sm_80,sm_89").split(",")]
V = P.V
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------- 1. unit tests
def unit():
    PX.install("sm_90")
    I = PX.PortInterp(UP.C.D, ext=True, fchk_fast_path=True)
    I._consts = {0x168: V.exact(0x1000), 0x16c: V.exact(0x7f), 0x170: V.exact(0xFFFFFFFE)}
    r = {"R2": V.exact(0xFFFFFFF0), "R3": V.exact(5), "R0": V.exact(3), "UR4": V.exact(0x100)}
    p = {}
    check("constant operand", I.val("c[0x0][0x168]", r) == V.exact(0x1000))
    check("negated constant operand", I.val("-c[0x0][0x168]", r) == V.exact(-0x1000))
    check("64-bit constant operand", I.val64("c[0x0][0x168]", r) == (0x7f << 32) | 0x1000)
    v, c = I.port_add_carry(["R2", "0x20", "RZ"], r)
    check("IADD3 carry-out", v == V.exact(0x10) and c is True, (v, c))
    v, c = I.port_add_carry(["R3", "0x20", "RZ"], r)
    check("IADD3 no carry", v == V.exact(0x25) and c is False, (v, c))
    p["P0"] = True
    check("IADD3.X carry-in", I.port_add_x(["RZ", "R3", "RZ"], ["P0", "!PT"], r, p) == V.exact(6))
    check("IMAD.X carry-in", I.port_imad_x(["RZ", "RZ", "RZ"], "P0", r, p) == V.exact(1))
    # LEA.HI.X.SX32 R4, R2, RZ, 0x1, P0 with R2 = -16: high word of sext(-16) << 1 = 0xFFFFFFFF, + 0 + carry 1 = 0
    check("LEA.HI.X.SX32 sign extension", I.port_lea_hi_x_sx32(("R4", "R2", "RZ", "0x1", "P0"), r, p) == V.exact(0))
    check("HFMA2.MMA constant 31", I.port_hfma2_const(("R7", "-RZ", "RZ", "0", "1.847743988037109375e-06")) == V.exact(0x1f))
    check("HFMA2.MMA constant 16", I.port_hfma2_const(("R5", "-RZ", "RZ", "0", "9.5367431640625e-07")) == V.exact(0x10))
    check("HFMA2.MMA zero", I.port_hfma2_const(("R5", "-RZ", "RZ", "0", "0")) == V.exact(0))
    check("HFMA2.MMA high half", I.port_hfma2_const(("R5", "-RZ", "RZ", "1", "0")) == V.exact(0x3C000000))
    check("scaled shared address", I.smem_addr("[R0.X4+0x1000]", r) == 0x100c)
    check("scaled + uniform shared address", I.smem_addr("[R3.X8+UR4+0x10]", r) == 5 * 8 + 0x100 + 0x10)
    check("register + uniform shared address stays as committed (unknown)", I.smem_addr("[R3+UR4]", r) is None)
    check("plain shared address unchanged", I.smem_addr("[R3+0x10]", r) == 0x15)
    check("global operand without descriptor", PX.MEMRE_PORT.match("[R4.64+0x80]").groups() == (None, "R4", "0x80"))
    check("global operand with descriptor", PX.MEMRE_PORT.match("desc[UR4][R2.64]").groups() == ("UR4", "R2", None))
    import bank_conflicts as B
    check("bank evaluator scaled term", B.eval_terms("R0.X4+0x10", lambda t: {"R0": 3}.get(t)) == 0x1c)
    check("bank evaluator plain term unchanged", B.eval_terms("R0+0x10", lambda t: {"R0": 3}.get(t)) == 0x13)


# --------------------------------------------------------------------------- 2. Blackwell unchanged
def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ("build_seconds",)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def blackwell_unchanged(names):
    PX.install("sm_120")
    import cells as CELLS
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    cells = {c["cell_id"]: c for c in CELLS.define_cells(hw)}
    for name in names:
        ref = json.loads((SR / "unseen_kernels/build/cells" / name).read_text())
        cell = cells[ref["cell"]["cell_id"]]
        rec, prow, urow = UP.build_cell(cell, hw)
        got = json.loads(json.dumps(dict(features=rec, phases=prow, unique=urow), sort_keys=True, default=str))
        same = all(strip(got[k]) == strip(ref[k]) for k in ("features", "phases", "unique"))
        diff = [k for k in ("features", "phases", "unique") if strip(got[k]) != strip(ref[k])]
        check("Blackwell unchanged: " + name, same, diff)


# --------------------------------------------------------------------------- 3. cross-architecture invariants
def block_signature(kid, kern, consts, grid, block):
    run = UP.run_exact(kid, kern, consts, grid, block)
    if run["status"] != "ok":
        return dict(refused=run["reason"])
    sigs, _ = UP.trace_kernel(kid, kern, consts, grid, block)
    s = sigs[0]
    tot = lambda k: None if any(ph[k] is None for ph in s["phases"]) else sum(ph[k] for ph in s["phases"])
    return dict(read_sectors=tot("read_sectors"), write_sectors=tot("write_sectors"), read_bytes=tot("read_bytes"),
                write_bytes=tot("write_bytes"), barriers=len(s["barrier_sequence"]),
                # span, not absolute: sm_90/sm_120 place user shared memory 1,024 B above a reserved window, sm_80/89 at 0
                shared_span=(run["shared_addresses"][1] - run["shared_addresses"][0]) if run.get("shared_addresses") else None,
                assumptions=sorted({k.split("@")[0] for k in run.get("assumption_log", {})}))


# Differences that come from the committed Blackwell reference, not from the port, each with its reason.
KNOWN_DIFFERENCES = {
    (arch, "scan_bottom"): dict(override=dict(shared_span=2044),
                                why="the committed interpreter leaves the Blackwell operand [R11+UR5] unknown, so its "
                                    "reference span stops 4 B short; this build uses only resolved forms")
    for arch in ("sm_90", "sm_80", "sm_89")
}


def kernels_of_cell(cell):
    for kern in cell["kernels"]:
        kern = dict(kern, sample_blocks=[0])          # one block suffices for the invariant; cells pick their own otherwise
        yield kern["kid"], kern


def cross_arch(archs, families):
    import cells as CELLS
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    cells = [c for c in CELLS.define_cells(hw) if c["family"] in families and c["candidate_id"] == "c1" and c["regime"] == "small"]
    try:
        import cells_e as CE
        cells += [c for c in CE.define_cells(hw) if c["candidate_id"] == "c1" and c["regime"] == "small"]
    except Exception as ex:  # noqa: BLE001
        print("note: set-E cells not loaded:", ex)
    ref = {}
    PX.install("sm_120")
    for c in cells:
        for kid, kern in kernels_of_cell(c):
            consts, grid, block, _ = UP.binding(kid, kern)
            ref[(c["cell_id"], kid)] = block_signature(kid, kern, consts, grid, block)
    for arch, folder in archs:
        PX.install(arch, SR / folder / SASS_SUBDIR)
        for c in cells:
            for kid, kern in kernels_of_cell(c):
                consts, grid, block, _ = UP.binding(kid, kern)
                a = UP.kernel_assets(kid)
                got = block_signature(kid, kern, consts, grid, block)
                r = ref[(c["cell_id"], kid)]
                known = KNOWN_DIFFERENCES.get((arch, kid))
                cmp_got = {k: v for k, v in got.items() if k != "assumptions"}
                cmp_ref = {k: v for k, v in r.items() if k != "assumptions"}
                if known and cmp_got == dict(cmp_ref, **known["override"]):
                    check("%s %s %s (known reference difference: %s)" % (arch, c["cell_id"].split("/")[1], kid, known["why"]), a["abi_ok"])
                    continue
                ok = a["abi_ok"] and cmp_got == cmp_ref
                if ok and got.get("assumptions"):
                    print("     logged assumptions:", got["assumptions"])
                check("%s %s %s" % (arch, c["cell_id"].split("/")[1], kid), ok,
                      dict(abi=a["abi_problems"], got=got, blackwell=r) if not ok else "")


# --------------------------------------------------------------------------- 4. bank conflicts across architectures
def bank_summary(kid, kern):
    """Per-kernel totals of the bank-conflict analysis (bank/bank_unseen.py, imported, never edited) at one block, gated
    against a phase table built on the same architecture."""
    import bank_unseen as BU
    consts, grid, block, _ = UP.binding(kid, kern)
    sigs, _ = UP.trace_kernel(kid, kern, consts, grid, block)
    res = BU.analyse_kernel(kid, kern, dict(phases=sigs[0]["phases"]))
    ph = [p["shared"] for p in res["phases"]]
    hist = {}
    for p in ph:
        for k, v in p["shared_conflict_degree_histogram"].items():
            hist[k] = hist.get(k, 0) + v
    cost = None if any(p["shared_cost_cycles"] is None for p in ph) else round(sum(p["shared_cost_cycles"] for p in ph), 6)
    return dict(requests=sum(p["shared_requests"] for p in ph), unknown=sum(p["shared_requests_unknown"] for p in ph),
                cost_cycles=cost, degree_histogram=dict(sorted(hist.items())))


# Scalar product: the local CUDA 13.0 sm_120 build reproduces the committed CUDA 13.2 Blackwell totals exactly (41,216 requests,
# 71,680 cycles), while sm_80, sm_89 and sm_90 (same toolchain) agree with each other: the reduction tail is laid out differently
# per architecture (fewer fully predicated-off requests, 256 more active ones per sampled launch). Every degree is 1 on all builds.
KNOWN_BANK_DIFFERENCES = {
    "sp": dict(expected=dict(requests=37888, unknown=0, cost_cycles=72192.0, degree_histogram={"1": 36096}),
               why="architecture-specific reduction layout; all conflict degrees 1; cost +0.7%"),
}


def bank_cross_arch(archs):
    import cells as CELLS
    import cells_e as CE
    hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH))
    cells = [c for c in CELLS.define_cells(hw) + CE.define_cells(hw) if c["candidate_id"] == "c1" and c["regime"] == "small"]
    targets = [(c["cell_id"], kid, kern) for c in cells for kid, kern in kernels_of_cell(c)
               if UP.KERNELS[kid]["sass"] != "BlackScholes"]          # no shared memory
    PX.install("sm_120")
    ref = {(cid, kid): bank_summary(kid, kern) for cid, kid, kern in targets}
    for arch, folder in archs:
        PX.install(arch, SR / folder / SASS_SUBDIR)
        for cid, kid, kern in targets:
            try:
                got = bank_summary(kid, kern)
            except (UP.C.Refusal, P.Refusal) as ex:
                got = dict(refused=str(ex))
            r = ref[(cid, kid)]
            known = KNOWN_BANK_DIFFERENCES.get(kid)
            if known and got == known["expected"]:
                check("bank %s %s %s (known codegen difference: %s)" % (arch, cid.split("/")[1], kid, known["why"]), True)
                continue
            same_cost = got.get("cost_cycles") == r["cost_cycles"] and got.get("unknown") == 0
            check("bank %s %s %s" % (arch, cid.split("/")[1], kid), same_cost, dict(got=got, blackwell=r))


# --------------------------------------------------------------------------- 5. full cell builds with each GPU's occupancy rows
GT_SECTION = {"sm_90": "## H100", "sm_80": "## A100", "sm_89": "## Ada"}


def occupancy_end_to_end(archs):
    """build_cell (features with occupancy, phases, first-touch) for the small first-candidate cell of every family, with the
    GPU's own HARDWARE_GROUND_TRUTH.md rows. The launches are Blackwell's (cells.py sizes cells from Blackwell's L2; the tier
    label is therefore not valid for the other GPU and is not checked). Static shared bytes must equal Blackwell's."""
    import cells as CELLS
    import cells_e as CE
    text = UP.X.read_text(UP.X.GROUND_TRUTH)
    bw = UP.X.load_hardware(text)
    cells = [c for c in CELLS.define_cells(bw) + CE.define_cells(bw) if c["regime"] == "small" and c["candidate_id"] == "c1"]

    def summary(rec, cell):
        blocks = rec["secondary_kernels"] + [rec]
        return [(k["kid"], b["resources"]["static_shared_bytes_per_block"], (b.get("occupancy") or {}).get("blocks_per_sm"))
                for k, b in zip(cell["kernels"], blocks)]
    PX.install("sm_120")
    ref = {c["cell_id"]: summary(UP.build_cell(c, bw)[0], c) for c in cells}
    for arch, folder in archs:
        hw = UP.X.load_hardware(text, GT_SECTION[arch])
        missing = [k for k, v in hw.items() if v is None]
        if missing:
            check("occupancy rows %s" % arch, False, "missing in HARDWARE_GROUND_TRUTH.md: " + ", ".join(missing))
            continue
        PX.install(arch, SR / folder / SASS_SUBDIR)
        for c in cells:
            try:
                rec, prow, urow = UP.build_cell(c, hw)
            except (UP.C.Refusal, P.Refusal, KeyError) as ex:
                check("cell build %s %s" % (arch, c["cell_id"].split("/")[1]), False, repr(ex))
                continue
            got = summary(rec, c)
            ok = (rec["status"].startswith("supported") and prow["status"] == "conditional_static_phases" and urow["status"] == "ok"
                  and all(o is not None for _, _, o in got)
                  and [(k, s) for k, s, _ in got] == [(k, s) for k, s, _ in ref[c["cell_id"]]])
            check("cell build %s %s (blocks/SM %s)" % (arch, c["cell_id"].split("/")[1], [o for _, _, o in got]), ok,
                  dict(status=(rec["status"], prow["status"], urow["status"]), got=got, blackwell=ref[c["cell_id"]],
                       missing=rec["missing_features"][:3]))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    unit()
    blackwell_unchanged(["blackwell__unseen_cuda_samples_black_scholes__small__c1.json",
                         "blackwell__unseen_cuda_samples_scan__small__c1.json"])
    if not quick:
        archs = PORT_ARCHS
        print("cross-architecture checks on", SASS_SUBDIR, [a for a, _ in archs])
        cross_arch(archs, {"bs", "scan", "matmul", "conv"})
        bank_cross_arch(archs)
        occupancy_end_to_end([(a, f) for a, f in archs if a != "sm_89"])   # Ada has no complete occupancy rows yet
    print("\n%d failure(s)" % len(FAILS))
    sys.exit(1 if FAILS else 0)
