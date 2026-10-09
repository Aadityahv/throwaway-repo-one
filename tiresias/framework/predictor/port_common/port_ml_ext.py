"""Machine-learning, tensor-core and attention kernels (fresh sets F, G, H: GELU, SwiGLU gate, RMSNorm, rotary embedding, FP32 matrix multiply, bf16 tensor-core matrix multiply,
fused attention) through the cross-GPU static port, for A100 (sm_80) and H100 (sm_90). CPU only; no measured value is read.

This module only adds, in memory and on top of port_ext.py, what the Blackwell implementations of these sets (fresh_f_lib.py, fresh_g_lib.py, fresh_h_lib.py) added on top of the unseen-kernel
pipeline, expressed for the port's interpreter:
  * the 14 kernels of the combined driver (fresh_f/gpu/drivers/driver_ml.cu: one translation unit with the three kernel headers) registered with their parameter layouts;
  * the interpreter extensions of those libraries, applied to the SAME source text the port patches (anchors verified exactly as in port_ext.py): HFMA2 with -RZ, RZ literal operands and
    64-bit uniform add (UIADD3.64), the tensor-core and shared/bf16 opcodes as data-only instructions, the per-thread step limit 600,000;
  * the tensor_core instruction family of extract_features.op_family;
  * the cubin hash of an architecture folder that has no cubin (the CUDA 12.1 builds of the combined driver stay on Cluster; their SASS dump and resource usage are committed): the hash then
    names the committed SASS dump instead, labelled as such, and a Blackwell folder with its cubin is unaffected.
Nothing here edits port_ext.py or any frozen file, so test_port_ext.py ("Blackwell unchanged") and test_set_d_port.py are unaffected. Any instruction the port cannot interpret makes the
cell refuse with the exact opcode (`static_support_*.json`); nothing is approximated.

    import port_ml_ext as PM
    PM.install("sm_90", SR / "port_h100" / "compiled_cluster_cuda12.1_ml")      # flat folder: driver_ml.sass and driver_ml.res
"""
from __future__ import annotations

import inspect
import textwrap
from fractions import Fraction
from pathlib import Path

import port_ext as PX

UP, A, P = PX.UP, PX.A, PX.P
SR = PX.SR

PTR2 = [("x", "ptr"), ("y", "ptr")]
PTR3 = [("a", "ptr"), ("b", "ptr"), ("y", "ptr")]
PTR4 = [("x", "ptr"), ("cs", "ptr"), ("sn", "ptr"), ("y", "ptr")]
RMS = [("x", "ptr"), ("w", "ptr"), ("y", "ptr"), ("cols", "i32"), ("inv_cols", "f32"), ("eps", "f32")]
RMS4 = [("x", "ptr"), ("w", "ptr"), ("y", "ptr"), ("cols4", "i32"), ("inv_cols", "f32"), ("eps", "f32")]
GEMM = [("A", "ptr"), ("B", "ptr"), ("C", "ptr"), ("N", "i32"), ("K", "i32")]
ATTN = [("Q", "ptr"), ("K", "ptr"), ("V", "ptr"), ("O", "ptr"), ("S", "i32")]
SRC = "driver_ml"            # file stem of the combined driver's SASS dump and resource usage in the architecture folder
KERNELS = {
    "gelu_s": ("gelu_s", PTR2), "gelu_v4": ("gelu_v4", PTR2), "swiglu_s": ("swiglu_s", PTR3), "swiglu_v4": ("swiglu_v4", PTR3),
    "rms_s": ("rmsnorm_sILi256E", RMS), "rms_v4": ("rmsnorm_v4ILi128E", RMS4), "rope_all": ("rope_allILi8E", PTR4), "rope_one": ("rope_oneILi8E", PTR4),
    "sg64": ("sgemmILi64E", GEMM), "sg128": ("sgemmILi128E", GEMM), "tc128": ("tc_gemmILi128E", GEMM), "tc64": ("tc_gemmILi64E", GEMM),
    "at4": ("attn_fwdILi4E", ATTN), "at8": ("attn_fwdILi8E", ATTN),
}
# Data-only or no-op forms the Blackwell libraries accept (EXT_OPS of fresh_f_lib / fresh_g_lib / fresh_h_lib); PX.PORT_OPS already holds the port's own additions.
ML_OPS = frozenset(["LDG.E.CONSTANT", "UIADD3.64", "HMMA.16816.F32.BF16", "STS.128", "STS.U16", "F2FP.BF16.F32.PACK_AB", "F2FP.BF16.PACK_AB"])
MAX_STEPS = 600_000          # per-thread walk guard of the register-tiled matrix multiply (fresh_f_lib.py); raising it cannot change a walk that already finished
HFMA2_ANCHOR = "            elif o == 'P2R' and a[1:3] == ('PR', 'RZ'):\n"
HFMA2_NEW = (
    "            elif o == 'HFMA2' and len(a) == 5 and a[1:3] == ('-RZ', 'RZ'):   # literal-operand form: (half hi << 16) | half lo\n"
    "                try:\n"
    "                    _np = __import__('numpy'); result = V.exact((int(_np.float16(float(a[3])).view(_np.uint16)) << 16) | int(_np.float16(float(a[4])).view(_np.uint16)))\n"
    "                except ValueError:\n"
    "                    result = None\n" + HFMA2_ANCHOR)
UIADD3_ANCHOR = "            elif ext and o in ('IADD.64',):\n"
UIADD3_NEW = (
    "            elif ext and o == 'UIADD3.64':   # URd, UPT, UPT, URa, URb, URz: 64-bit sum of three register pairs; the carry predicates are not used\n"
    "                x, y, z = self.val64(a[3], r), self.val64(a[4], r), self.val64(a[5], r)\n"
    "                pair_result = ('pair', (x + y + z) & M64 if x is not None and y is not None and z is not None else None)\n" + UIADD3_ANCHOR)
# sm_80 only (CUDA 12.1 loops of the register-tiled matrix multiply, the tensor-core matrix multiply and attention end with `@P0 CALL.REL.NOINC <next but one>; BRA <loop head>; <target>: ...`): a call
# whose target is the instruction after the following branch (a BRA or NOP), in a kernel that contains no RET at all, never returns; it is the exit jump of the loop. The port already accepts a
# call to an EXIT; this accepts the same no-return call to a BRA or NOP, under the same logged assumption key (its text is extended below). Any other CALL still refuses.
CALL_ANCHOR = "            if o == 'CALL.REL.NOINC' and len(a) == 1 and int(a[0], 0) in by and by[int(a[0], 0)].op == 'EXIT':\n"
CALL_NEW = (
    "            if o == 'CALL.REL.NOINC' and len(a) == 1 and int(a[0], 0) in by and by[int(a[0], 0)].op in ('BRA', 'NOP') and not any(v_.op.startswith('RET') for v_ in by.values()):\n"
    "                require(do is not None, 'unknown call-as-jump predicate at ' + hex(pc))\n"
    "                self.assumption_log[('call_to_exit', pc)] += 1\n"
    "                pc = int(a[0], 0)\n                continue\n" + CALL_ANCHOR)
EXTRA_ANCHORS = [(HFMA2_ANCHOR, HFMA2_NEW), (UIADD3_ANCHOR, UIADD3_NEW), (CALL_ANCHOR, CALL_NEW)]
CALL_TEXT = (" Extension for the machine-learning kernels (sm_80): the same no-return call whose target is a BRA or NOP, in a kernel with no RET, is the loop-exit jump and is taken as a branch; "
             "any other CALL still refuses.")


def _interp_class():
    src = inspect.getsource(P.Interp._lane_gen)
    for needle, repl in list(PX._ANCHORS) + EXTRA_ANCHORS:
        if src.count(needle) != 1:
            raise SystemExit("port ML extension: interpreter anchor changed or ambiguous: " + needle.strip()[:70])
        src = src.replace(needle, repl)
    env = dict(P.Interp._lane_gen.__globals__)
    env["Fraction"] = Fraction
    exec(compile(textwrap.dedent(src), "<cross-gpu-port-ml-ext>", "exec"), env)

    class PortMLInterp(PX.PortInterp):
        _lane_gen = env["_lane_gen"]

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.KNOWN_OPS = set(self.KNOWN_OPS) | ML_OPS

        def run_block(self, sites, constants, coords_for_lane, threads, max_steps=MAX_STEPS):
            return super().run_block(sites, constants, coords_for_lane, threads, max(max_steps, MAX_STEPS))

    return PortMLInterp


_orig_op_family = UP.X.op_family


def _op_family_with_tensor(op):
    if op.startswith(("HMMA", "IMMA", "QMMA", "DMMA")):
        return "tensor_core", None, op.split(".", 1)[1] if "." in op else None
    return _orig_op_family(op)


_orig_assets_from = PX._assets_from


def _assets_from_with_cubin_fallback(kid, s):
    """As port_ext._assets_from; when the folder has no cubin the cubin hash names the committed SASS dump (labelled), instead of failing."""
    cubin = PX._cubin_dir[0] / (s["sass"] + ".cubin")
    if cubin.exists():
        return _orig_assets_from(kid, s)
    stub = PX._cubin_dir[0]
    fake = Path(stub)
    sass = (UP.SASS_DIR / (s["sass"] + ".sass"))
    # run the original with a temporary cubin hash substitution: the original reads (dir / <sass>.cubin).read_bytes(); give it the SASS dump bytes through a shim path object
    class _Shim:
        def __truediv__(self, name):
            return self
        def read_bytes(self):
            return b"NO-CUBIN-IN-REPOSITORY; SASS DUMP SHA256 FOLLOWS:" + sass.read_bytes()
        def exists(self):
            return True
    PX._cubin_dir[0] = _Shim()
    try:
        out = _orig_assets_from(kid, s)
    finally:
        PX._cubin_dir[0] = fake
    out["cubin_sha_is_sass_dump_hash"] = True
    return out


def install(arch, sass_root):
    """Port installed for `arch` on the folder with driver_ml.sass / driver_ml.res (flat), the 14 kernels registered, the ML interpreter and the tensor family in place."""
    if arch == "sm_120":
        raise SystemExit("port_ml_ext is for the CUDA 12.1 builds of A100 and H100; Blackwell uses fresh_f/g/h directly")
    PX.install(arch, sass_root)
    UP.SASS_DIR = UP.LOG_DIR = Path(sass_root)       # flat folder: <stem>.sass and <stem>.res side by side
    UP.INTERP = _interp_class()
    for kid, (needle, params) in KERNELS.items():
        UP.KERNELS[kid] = dict(sass=SRC, needle=needle, role="main", params=params)
        A.PARAM_FIELDS[kid] = params
    UP._KERNEL_CACHE.clear()
    UP.X.op_family = _op_family_with_tensor
    if CALL_TEXT not in A.ASSUMPTION_TEXT["call_to_exit"]:
        A.ASSUMPTION_TEXT["call_to_exit"] += CALL_TEXT
    PX._assets_from = _assets_from_with_cubin_fallback
    PX.STATE.update(ml=True)
