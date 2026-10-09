"""Set D through the cross-GPU port (CPU only). Run: python test_set_d_port.py

1. Blackwell equality: for all 28 set D cells, the port route (set_d_port.py, retained Blackwell cubins) gives exactly the committed
   phase table (fresh_d/phases_fresh_d.json): barrier sequence, phase count, and per phase the read/write sectors, lines, bytes,
   per-opcode warp-instruction counts and the two dependency-chain lengths.
2. Cross-architecture invariants on the Cluster CUDA 12.1 builds (PORT_SASS_SUBDIR, default compiled_cluster_cuda12.1; A100 and
   H100): per kernel at its first sampled block, global sectors read/written, global bytes, dynamic barrier count and shared span
   equal the Blackwell route's, and the bank-conflict cost equals Blackwell's. Vector add has no CUDA 12.1 build (it does not compile
   with 12.1) and is skipped there, reported.
No GPU, no measured value.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import set_d_port as SD  # noqa: E402
import test_port_ext as T  # noqa: E402  (reuses its check(), block_signature() and bank_summary())

PX, UP = SD.PX, SD.UP
SUB = os.environ.get("PORT_SASS_SUBDIR", "compiled_cluster_cuda12.1")
ARCHS = [("sm_90", "port_h100"), ("sm_80", "port_a100")]
FIELDS = ("read_sectors", "write_sectors", "lines", "read_bytes", "write_bytes", "issue_warp_instructions",
          "critical_path_compute_instructions", "dependent_global_load_depth")


def blackwell_equality(cells):
    PX.install("sm_120", SD.BLACKWELL_DIR)
    SD.register(blackwell=True)
    for c in cells:
        kern, kid = c["kern"], c["kern"]["kid"]
        consts, grid, block, _ = UP.binding(kid, kern)
        a = UP.kernel_assets(kid)
        sigs, _ = UP.trace_kernel(kid, kern, consts, grid, block)
        got, ref = sigs[0], c["committed_phases"]["kernels"][0]
        diffs = []
        if got["barrier_sequence"] != ref["barrier_sequence"] or len(got["phases"]) != len(ref["phases"]):
            diffs.append("barrier sequence or phase count")
        for i, (g, r) in enumerate(zip(got["phases"], ref["phases"])):
            diffs += ["phase %d %s" % (i, f) for f in FIELDS if g.get(f) != r.get(f)]
        T.check("set D Blackwell equals committed: " + c["cell_id"].split("/", 1)[1], a["abi_ok"] and not diffs, diffs[:4])


def cross_arch(cells):
    one = {}
    for c in cells:                      # one cell per kernel (the L2 cell, first candidate using that kernel)
        if c["cell_id"].split("/")[2] == "l2":
            one.setdefault(c["kernel"], c)
    PX.install("sm_120", SD.BLACKWELL_DIR)
    SD.register(blackwell=True)
    ref = {}
    for k, c in one.items():
        kern = dict(c["kern"], sample_blocks=c["kern"].get("sample_blocks", [0])[:1])
        consts, grid, block, _ = UP.binding(kern["kid"], kern)
        ref[k] = (T.block_signature(kern["kid"], kern, consts, grid, block), T.bank_summary(kern["kid"], kern))
    for arch, folder in ARCHS:
        PX.install(arch, PX.SR / folder / SUB)
        SD.register(blackwell=False)
        for k, c in one.items():
            if k == "vecAdd":
                print("SKIP %s set D vecAdd: no CUDA 12.1 build (cuda/cmath missing)" % arch)
                continue
            kern = dict(c["kern"], sample_blocks=c["kern"].get("sample_blocks", [0])[:1])
            try:
                consts, grid, block, _ = UP.binding(kern["kid"], kern)
                a = UP.kernel_assets(kern["kid"])
                sig, bank = T.block_signature(kern["kid"], kern, consts, grid, block), T.bank_summary(kern["kid"], kern)
            except (UP.C.Refusal, PX.P.Refusal) as ex:
                T.check("set D %s %s" % (arch, k), False, repr(ex))
                continue
            rs, rb = ref[k]
            same = ({x: v for x, v in sig.items() if x != "assumptions"} == {x: v for x, v in rs.items() if x != "assumptions"}
                    and bank["cost_cycles"] == rb["cost_cycles"] and bank["unknown"] == 0)
            T.check("set D %s %s (assumptions %s)" % (arch, k, sig.get("assumptions")), a["abi_ok"] and same,
                    dict(got=sig, bank=bank, blackwell=rs, blackwell_bank=rb))


if __name__ == "__main__":
    SD.prepare_blackwell()
    cells = SD.cells()
    blackwell_equality(cells)
    cross_arch(cells)
    print("\n%d failure(s)" % len(T.FAILS))
    sys.exit(1 if T.FAILS else 0)
