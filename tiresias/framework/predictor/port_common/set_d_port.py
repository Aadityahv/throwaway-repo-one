"""Set D (CUDA-samples vector add, copy, transposes, reductions) through the cross-GPU port pipeline, CPU only.

Set D was built for Blackwell through the frozen derivation (operator_derivation/derive.py via fresh_d/fresh_d_lib.py), which
binds kernels to retained development evidence and refuses any parameter base other than 0x380. Instead of porting that path,
set D's kernels are registered with the same unseen-kernel pipeline the port already extends (port_ext.py): parameter layouts from
the samples' kernel signatures, launches and dynamic shared sizes from the committed set D cell and feature tables. Whether this
route reproduces set D's committed Blackwell phase tables is checked by test_set_d_port.py before it is used on another GPU.

No measured value is read. Nothing in fresh_d/ is edited.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import port_ext as PX

UP, A = PX.UP, PX.A
SR = PX.SR
FD = SR / "fresh_d"
RETAINED = (SR.parent / "compile_evidence/acquisition_runs/cuda_compile_blackwell_20261001_51afb2f3/output/nvcc/artifacts")
BLACKWELL_DIR = PX.HERE / "set_d_blackwell"          # sass/, log/, cubin/ dumped from the retained Blackwell cubins

PTR2_INT2 = [("odata", "ptr"), ("idata", "ptr"), ("width", "i32"), ("height", "i32")]
RED = [("g_idata", "ptr"), ("g_odata", "ptr"), ("n", "i32")]
# kernel -> (symbol, source family for the SASS file name, parameter list)
SET_D_KERNELS = {
    "vecAdd": ("_Z6vecAddPfS_S_i", "vectorAdd", [("A", "ptr"), ("B", "ptr"), ("C", "ptr"), ("vectorLength", "i32")]),
    "transposeNaive": ("_Z14transposeNaivePfS_ii", "transpose", PTR2_INT2),
    "transposeCoalesced": ("_Z18transposeCoalescedPfS_ii", "transpose", PTR2_INT2),
    "transposeNoBankConflicts": ("_Z24transposeNoBankConflictsPfS_ii", "transpose", PTR2_INT2),
    "transposeDiagonal": ("_Z17transposeDiagonalPfS_ii", "transpose", PTR2_INT2),
    "copy": ("_Z4copyPfS_ii", "transpose", PTR2_INT2),
    "copySharedMem": ("_Z13copySharedMemPfS_ii", "transpose", PTR2_INT2),
    "transposeFineGrained": ("_Z20transposeFineGrainedPfS_ii", "transpose", PTR2_INT2),
    "transposeCoarseGrained": ("_Z22transposeCoarseGrainedPfS_ii", "transpose", PTR2_INT2),
    "reduce0": ("_Z7reduce0IfEvPT_S1_j", "reduction", RED),
    "reduce1": ("_Z7reduce1IfEvPT_S1_j", "reduction", RED),
    "reduce2": ("_Z7reduce2IfEvPT_S1_j", "reduction", RED),
    "reduce6": ("_Z7reduce6IfLj256ELb1EEvPT_S1_j", "reduction", RED),
}
# Blackwell: the retained cubin each set D cell's evidence came from (fresh_cells_d.json retained_cubin_sha256). reduce2 comes from a
# different build of the same source than reduce0/1/6, so on Blackwell it gets its own SASS file name.
BLACKWELL_CUBIN = {"vectorAdd": "16a4a73c1d8a0115711b0f1a2973fc9e9f7bd8e48c6a58529cb5117040ae5fef",
                   "transpose": "a6de435837aafb6da14e2789671407b65ab7d1f727742d09203d7bb9f7b0553e",
                   "reduction": "686c4f1f5e69774fefceab0dc068b888525fd09824c5e6211c544a8c8387b2fe",
                   "reduction_train": "b0dfb32a2f6f38c813f6de067a1405a2b03fa4afbfe694ebae81acdc348b253c"}


def prepare_blackwell():
    """Disassemble the retained Blackwell cubins (local cuobjdump; SASS text is toolkit-independent) into BLACKWELL_DIR."""
    for d in ("sass", "log", "cubin"):
        (BLACKWELL_DIR / d).mkdir(parents=True, exist_ok=True)
    for name, h in BLACKWELL_CUBIN.items():
        src = RETAINED / h / "kernel.cubin"
        dst = BLACKWELL_DIR / "cubin" / (name + ".cubin")
        long_src = "\\\\?\\" + str(src.resolve()) if sys.platform == "win32" else str(src)   # path exceeds MAX_PATH on Windows
        shutil.copyfile(long_src, dst)
        (BLACKWELL_DIR / "sass" / (name + ".sass")).write_text(
            subprocess.run(["cuobjdump", "--dump-sass", str(dst)], capture_output=True, text=True, check=True).stdout, encoding="utf-8")
        (BLACKWELL_DIR / "log" / (name + ".res")).write_text(
            subprocess.run(["cuobjdump", "--dump-resource-usage", str(dst)], capture_output=True, text=True, check=True).stdout,
            encoding="utf-8")


def register(blackwell):
    for kid, (sym, fam, params) in SET_D_KERNELS.items():
        sass = "reduction_train" if (blackwell and kid == "reduce2") else fam
        UP.KERNELS["d_" + kid] = dict(sass=sass, needle=sym, role="main", params=params)
        A.PARAM_FIELDS["d_" + kid] = params


def cells():
    """Set D cells in the pipeline's kernel-entry form, from the committed tables (launch, arguments, dynamic shared size and the
    blocks the committed phase table sampled)."""
    table = json.loads((FD / "fresh_cells_d.json").read_text(encoding="utf-8"))["cells"]
    feats = {r["cell_id"]: r for r in json.loads((FD / "features_fresh_d.json").read_text(encoding="utf-8"))["rows"]}
    phases = json.loads((FD / "phases_fresh_d.json").read_text(encoding="utf-8"))["rows"]
    out = []
    for c in table:
        g = c["geometry"]
        k = c["kernel"]
        if k == "vecAdd":
            args = dict(vectorLength=g["n"])
        elif k.startswith("reduce"):
            args = dict(n=g["n"])
        else:
            args = dict(width=g["dim_x"], height=g["dim_y"])
        dyn = feats[c["cell_id"]]["resources"]["dynamic_shared_bytes_per_launch"]
        ph = phases[c["cell_id"]]
        sampled = ph["kernels"][0].get("sampled_blocks") if ph.get("kernels") else None
        out.append(dict(cell_id=c["cell_id"], kernel=k, committed_phases=ph,
                        kern=dict(kid="d_" + k, grid=g["grid"], block=g["block"], args=args, dynamic_smem=dyn,
                                  **({"sample_blocks": sampled} if sampled else {}))))
    return out
