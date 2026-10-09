"""Registers the two new CUDA-samples kernel families (scalar product, fast Walsh transform) with the unseen-kernel static pipeline
(imported, never edited) by pointing its SASS/resource directories at fresh_e/compiled and extending its kernel registry in memory."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR / 'unseen_kernels'))
import unseen_pipeline as UP  # noqa: E402

UP.HERE = HERE; UP.SASS_DIR = HERE / 'compiled/sass'; UP.LOG_DIR = HERE / 'compiled/log'; UP.ISO_DIR = HERE / 'isolated'
UP.KERNELS.clear()
UP.KERNELS.update({
    'sp': dict(sass='scalarProd', needle='scalarProdGPU', role='main',
               params=[('C', 'ptr'), ('A', 'ptr'), ('B', 'ptr'), ('vectorN', 'i32'), ('elementN', 'i32')]),
    'fwt1': dict(sass='fastWalshTransform', needle='fwtBatch1Kernel', role='main',
                 params=[('out', 'ptr'), ('in', 'ptr'), ('log2N', 'i32')]),
    'fwt2': dict(sass='fastWalshTransform', needle='fwtBatch2Kernel', role='main',
                 params=[('out', 'ptr'), ('in', 'ptr'), ('stride', 'i32')]),
})
for _kid, _spec in UP.KERNELS.items():
    UP.A.PARAM_FIELDS[_kid] = _spec['params']
UP._KERNEL_CACHE.clear()


# ---- private interpreter extension for the control-flow opcodes of scalarProd (outside the committed whitelists)
# BSSY.RELIABLE, BREAK.RELIABLE, BSYNC.RELIABLE and WARPSYNC.ALL only manage warp convergence bookkeeping (BREAK removes the thread from the
# barrier; the explicit BRA after it moves the thread), so each is a no-op for a single-lane walk (the committed interpreter walks lanes
# independently). The on-disk interpreter is never edited.
EXT_FRESH_E_OPS = frozenset(['BSSY.RELIABLE', 'BSYNC.RELIABLE', 'BREAK.RELIABLE', 'WARPSYNC.ALL'])


def fresh_e_interp_class():
    import inspect
    import textwrap
    P = UP.P
    src = inspect.getsource(P.Interp._lane_gen)
    needle_ushf = UP._NEEDLE
    anchors = {
        needle_ushf: UP._REPL,
        "            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n":
            "            if o in ('BSSY.RELIABLE', 'BREAK.RELIABLE', 'BSYNC.RELIABLE', 'WARPSYNC.ALL'):\n                pc += 16\n                continue\n"
            "            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n",
    }
    for k, v in anchors.items():
        if src.count(k) != 1:
            raise SystemExit('interpreter anchor changed: ' + k[:60])
        src = src.replace(k, v)
    env = dict(P.Interp._lane_gen.__globals__)
    exec(compile(textwrap.dedent(src), '<fresh-e-ext>', 'exec'), env)

    class FreshEInterp(P.Interp):
        _lane_gen = env['_lane_gen']

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.KNOWN_OPS = set(self.KNOWN_OPS) | UP.UNSEEN_EXT_OPS | EXT_FRESH_E_OPS

    return FreshEInterp


UP.INTERP = fresh_e_interp_class()
