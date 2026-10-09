"""Registers the unseen machine-learning kernels (src/ml_kernels.cuh, compiled with CUDA 13.2.78 for sm_120) with the unseen-kernel static pipeline
(imported, never edited) by pointing its SASS/resource directories at fresh_f/compiled and extending its kernel registry in memory. The interpreter gets one
extra known opcode, LDG.E.CONSTANT (the non-coherent global load the compiler emits for read-only restrict pointers), handled exactly like the already-accepted
LDG.E.64.CONSTANT: a global read with an unknown data result."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR / 'unseen_kernels'))
import unseen_pipeline as UP  # noqa: E402

UP.HERE = HERE; UP.SASS_DIR = HERE / 'compiled/sass'; UP.LOG_DIR = HERE / 'compiled/log'; UP.ISO_DIR = HERE / 'isolated'
UP.KERNELS.clear()
PTR2, PTR3, PTR4 = [('x', 'ptr'), ('y', 'ptr')], [('a', 'ptr'), ('b', 'ptr'), ('y', 'ptr')], [('x', 'ptr'), ('cs', 'ptr'), ('sn', 'ptr'), ('y', 'ptr')]
UP.KERNELS.update({
    'gelu_s': dict(sass='mlk', needle='gelu_s', role='main', params=PTR2),
    'gelu_v4': dict(sass='mlk', needle='gelu_v4', role='main', params=PTR2),
    'swiglu_s': dict(sass='mlk', needle='swiglu_s', role='main', params=PTR3),
    'swiglu_v4': dict(sass='mlk', needle='swiglu_v4', role='main', params=PTR3),
    'rms_s': dict(sass='mlk', needle='rmsnorm_sILi256E', role='main', params=[('x', 'ptr'), ('w', 'ptr'), ('y', 'ptr'), ('cols', 'i32'), ('inv_cols', 'f32'), ('eps', 'f32')]),
    'rms_v4': dict(sass='mlk', needle='rmsnorm_v4ILi128E', role='main', params=[('x', 'ptr'), ('w', 'ptr'), ('y', 'ptr'), ('cols4', 'i32'), ('inv_cols', 'f32'), ('eps', 'f32')]),
    'rope_all': dict(sass='mlk', needle='rope_allILi8E', role='main', params=PTR4),
    'rope_one': dict(sass='mlk', needle='rope_oneILi8E', role='main', params=PTR4),
    'sg64': dict(sass='mlk', needle='sgemmILi64E', role='main', params=[('A', 'ptr'), ('B', 'ptr'), ('C', 'ptr'), ('N', 'i32'), ('K', 'i32')]),
    'sg128': dict(sass='mlk', needle='sgemmILi128E', role='main', params=[('A', 'ptr'), ('B', 'ptr'), ('C', 'ptr'), ('N', 'i32'), ('K', 'i32')]),
})
for _kid, _spec in UP.KERNELS.items():
    UP.A.PARAM_FIELDS[_kid] = _spec['params']
UP._KERNEL_CACHE.clear()


# ---- private interpreter extension: control-flow bookkeeping opcodes of block reductions (BSSY.RELIABLE, BSYNC.RELIABLE, BREAK.RELIABLE, WARPSYNC.ALL) manage warp
# convergence only and are no-ops for the single-lane walk (same extension as fresh_e_lib.py; the on-disk interpreter is never edited), plus LDG.E.CONSTANT, UISETP.NE.U32.AND (a
# sibling of the accepted ISETP forms) and UIADD3.64 (uniform 64-bit three-operand add, evaluated with the interpreter's own register-pair helper).
MAX_STEPS = 600_000
EXT_OPS = frozenset(['BSSY.RELIABLE', 'BSYNC.RELIABLE', 'BREAK.RELIABLE', 'WARPSYNC.ALL', 'LDG.E.CONSTANT', 'UISETP.NE.U32.AND', 'UIADD3.64'])


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

        def run_block(self, sites, constants, coords_for_lane, threads, max_steps=MAX_STEPS):
            # the committed per-thread step limit (100,000) is a guard against non-terminating walks; the register-tiled matrix multiply at K = 2048 needs about 190,000 steps per thread.
            # Raising the guard cannot change the result of any walk that already finished under the old limit.
            return super().run_block(sites, constants, coords_for_lane, threads, max_steps)

    return FreshFInterp


UP.INTERP = fresh_f_interp_class()
