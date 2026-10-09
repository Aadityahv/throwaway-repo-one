"""Read-only corrected variant of the extended interpreter's IMAD.HI.U32 (CPU only; the on-disk interpreter is
never edited).

Finding (see FRESH.md, "Pre-existing interpreter issue"): pytorch_features/pt_interp.py evaluates
`IMAD.HI.U32 Rd, Ra, Rb, Rc` as hi32(Ra*Rb) + Rc. The SASS idiom is hi32(Ra*Rb + {R(c+1):R(c)}): the addend
is the 64-bit register pair starting at Rc. Evidence in the retained SASS:
  * every use in the copy kernel is `IMAD.HI.U32 Rd, Rn, <magic>, Rz` immediately after `MOV Rz, RZ` with
    Rz+1 == Rn, so the pair is (n << 32) and the instruction is (umulhi(n, m1) + n), the `(t + n) >> shift`
    of IntDivider<unsigned int>::div in the saved ATen source;
  * in the layer-norm kernel, `IMAD.HI.U32 R13, R5, R15, R4` follows `F2I ... R5` and `IMAD R15, R15, R5, RZ`
    (the Newton step x + umulhi(x, -d*x)); the pair R4:R5 adds x.
`corrected_interp_class()` returns a subclass of pytorch_features.pt_interp.Interp whose `_lane_gen` differs only
in that one instruction. test_fresh.py checks that, with the source IntDivider constants, it reproduces the
closed-form OffsetCalculator address for every lane of several blocks.
"""
from __future__ import annotations

import inspect
import textwrap

NEEDLE = """            elif ext and o == 'IMAD.HI.U32':
                vs = [val(x, r) for x in a[1:4]]
                if all(v is not None and v.exact_value is not None for v in vs):
                    result = V.exact(((vs[0].lo * vs[1].lo) >> 32) + vs[2].lo)
"""
REPLACEMENT = """            elif ext and o == 'IMAD.HI.U32':
                vs = [val(x, r) for x in a[1:3]]
                c64 = self.val64(a[3], r)
                if all(v is not None and v.exact_value is not None for v in vs) and c64 is not None:
                    result = V.exact((((vs[0].lo * vs[1].lo) + c64) >> 32) & 0xFFFFFFFF)
"""


def corrected_interp_class(P):
    src = inspect.getsource(P.Interp._lane_gen)
    if src.count(NEEDLE) != 1:
        raise ValueError("IMAD.HI.U32 insertion point changed in pt_interp.py")
    env = dict(P.Interp._lane_gen.__globals__)
    exec(compile(textwrap.dedent(src.replace(NEEDLE, REPLACEMENT)), "<corrected-imad-hi>", "exec"), env)

    class CorrectedInterp(P.Interp):
        _lane_gen = env["_lane_gen"]

    CorrectedInterp.__name__ = "CorrectedImadHiInterp"
    return CorrectedInterp
