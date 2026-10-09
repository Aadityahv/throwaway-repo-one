"""Registers the fused attention kernels (src/attn_kernels.cuh, compiled with CUDA 13.2.78 for sm_120) with the unseen-kernel static pipeline (imported, never edited) by pointing its
SASS/resource directories at fresh_h/compiled and extending its kernel registry in memory. Extensions, all in memory: the interpreter accepts HMMA.16816.F32.BF16 (a warp-level multiply-accumulate
whose results are data, never addresses: unknown result like the float instructions) and STS.128 (a 128-bit shared store, a store like the accepted STS forms), plus the control-flow and
64-bit helpers of fresh_f_lib; and the family map of extract_features.op_family gets a `tensor_core` family for the MMA opcodes (HMMA, IMMA, QMMA, DMMA) instead of the catch-all `other`."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR / 'unseen_kernels')); sys.path.insert(0, str(SR / 'fresh_f'))
import unseen_pipeline as UP  # noqa: E402

UP.HERE = HERE; UP.SASS_DIR = HERE / 'compiled/sass'; UP.LOG_DIR = HERE / 'compiled/log'; UP.ISO_DIR = HERE / 'isolated'
UP.KERNELS.clear()
UP.KERNELS.update({
    'at4': dict(sass='atk', needle='attn_fwdILi4E', role='main', params=[('Q', 'ptr'), ('K', 'ptr'), ('V', 'ptr'), ('O', 'ptr'), ('S', 'i32')]),
    'at8': dict(sass='atk', needle='attn_fwdILi8E', role='main', params=[('Q', 'ptr'), ('K', 'ptr'), ('V', 'ptr'), ('O', 'ptr'), ('S', 'i32')]),
})
for _kid, _spec in UP.KERNELS.items():
    UP.A.PARAM_FIELDS[_kid] = _spec['params']
UP._KERNEL_CACHE.clear()


# ---- private interpreter extension: control-flow bookkeeping opcodes of block reductions (BSSY.RELIABLE, BSYNC.RELIABLE, BREAK.RELIABLE, WARPSYNC.ALL) manage warp
# convergence only and are no-ops for the single-lane walk (same extension as fresh_e_lib.py; the on-disk interpreter is never edited), plus LDG.E.CONSTANT, UISETP.NE.U32.AND (a
# sibling of the accepted ISETP forms) and UIADD3.64 (uniform 64-bit three-operand add, evaluated with the interpreter's own register-pair helper).
EXT_OPS = frozenset(['BSSY.RELIABLE', 'BSYNC.RELIABLE', 'BREAK.RELIABLE', 'WARPSYNC.ALL', 'LDG.E.CONSTANT', 'UISETP.NE.U32.AND', 'UIADD3.64', 'HMMA.16816.F32.BF16', 'STS.128', 'STS.U16', 'F2FP.BF16.F32.PACK_AB'])


def fresh_f_interp_class():
    import inspect
    import textwrap
    P = UP.P
    src = inspect.getsource(P.Interp._lane_gen)
    anchors = {
        UP._NEEDLE: UP._REPL,
        "            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n":
            "            if o in ('BSSY.RELIABLE', 'BREAK.RELIABLE', 'BSYNC.RELIABLE', 'WARPSYNC.ALL'):\n                pc += 16\n                continue\n"
            "            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n",
    }
    anchors["            elif o == 'P2R' and a[1:3] == ('PR', 'RZ'):\n"] = (
        "            elif o == 'HFMA2' and len(a) == 5 and a[1:3] == ('-RZ', 'RZ'):   # HFMA2 Rd, -RZ, RZ, hi, lo with literal operands: the compiler's way of materialising a 32-bit constant, (half hi << 16) | half lo\n"
        "                try:\n"
        "                    _np = __import__('numpy'); result = V.exact((int(_np.float16(float(a[3])).view(_np.uint16)) << 16) | int(_np.float16(float(a[4])).view(_np.uint16)))\n"
        "                except ValueError:\n"
        "                    result = None\n"
        "            elif o == 'P2R' and a[1:3] == ('PR', 'RZ'):\n")
    anchors["            elif ext and o in ('IADD.64',):\n"] = (
        "            elif ext and o == 'UIADD3.64':   # UIADD3.64 URd, UPT, UPT, URa, URb, URz: 64-bit sum of three register pairs; the carry predicates are not used\n"
        "                x, y, z = self.val64(a[3], r), self.val64(a[4], r), self.val64(a[5], r)\n"
        "                pair_result = ('pair', (x + y + z) & M64 if x is not None and y is not None and z is not None else None)\n"
        "            elif ext and o in ('IADD.64',):\n")
    for k, v in anchors.items():
        if src.count(k) != 1:
            raise SystemExit('interpreter anchor changed: ' + k[:60])
        src = src.replace(k, v)
    env = dict(P.Interp._lane_gen.__globals__)
    exec(compile(textwrap.dedent(src), '<fresh-f-ext>', 'exec'), env)

    class FreshFInterp(P.Interp):
        _lane_gen = env['_lane_gen']

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.KNOWN_OPS = set(self.KNOWN_OPS) | UP.UNSEEN_EXT_OPS | EXT_OPS

    return FreshFInterp


UP.INTERP = fresh_f_interp_class()

_orig_op_family = UP.X.op_family


def _op_family_with_tensor(op):
    if op.startswith(('HMMA', 'IMMA', 'QMMA', 'DMMA')):
        return 'tensor_core', None, op.split('.', 1)[1] if '.' in op else None
    return _orig_op_family(op)


UP.X.op_family = _op_family_with_tensor

