"""Cross-GPU port of the static runtime pipeline (A100 sm_80, H100 sm_90, Ada sm_89): in-memory extensions only, CPU only.

Installs, in the already imported copies of the unseen-kernel pipeline (and the modules it imports), the four things the
Phase A audit (port_audit/FINDINGS.md) found missing for non-Blackwell SASS. No file of the frozen pipeline is edited; every
change is an in-memory replacement whose anchor text is checked against the file on disk, as the unseen-kernel and set-E
extensions already do.

1. Constant-bank layout per architecture: kernel-parameter base and the implicit launch constants (adapter.PARAM_BASE,
   adapter.IMPLICIT, adapter.IMPLICIT_ALSO_ACCESSED). Values are read off the audited SASS; the adapter's ABI check (every
   constant-bank access lands on a declared parameter or a listed implicit offset) checks them independently per kernel.
2. Constant-bank operands inside arithmetic instructions (`IADD3 R2, P0, R10, c[0x0][0x168], RZ`): Blackwell SASS loads every
   constant with LDC/LDCU first, the other architectures also read them directly. Resolved from the launch constants.
3. Operand forms: global addresses without a descriptor (`[R4.64]`, A100/Ada) and scaled shared-address terms
   (`[R0.X4+0x1000]`, A100/Ada), in the interpreter, the phase observer and the bank-conflict evaluator.
4. Opcode semantics: uniform constant loads (ULDC, ULDC.64), integer multiply-add used as move or add (IMAD.MOV, IMAD.MOV.U32,
   IMAD.IADD, IMAD.U32), VIADD, add with carry-out (IADD3/UIADD3 with a predicate carry) and carry-in (IADD3.X, UIADD3.X,
   IMAD.X, LEA.HI.X.SX32), HFMA2.MMA constant materialisation, I2F (signed 32-bit to binary32), BAR.SYNC, UISETP.GT.AND and a
   bare WARPSYNC (logged no-op, like the set-E extension's WARPSYNC.ALL).

Not done here (needs Phase B or a decision): issue-class mapping of HFMA2.MMA and the IMAD aliases for the runtime model (left to
the existing classifier, which puts them in the integer class; see FINDINGS.md), hardware rows for the occupancy model (A1), and
the frozen derivation's own `parameter_base == 0x380` check, which only the development-set path (set D) uses.

Usage:
    import port_ext as PX
    PX.install('sm_90', sass_root=<dir with compiled_provisional/{sass,log,cubin}>)
    UP = PX.UP   # the patched unseen-kernel pipeline; UP.run_exact / UP.trace_kernel / UP.binding work as before
"""
from __future__ import annotations

import collections
import inspect
import re
import struct
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / "unseen_kernels"))
sys.path.insert(0, str(SR / "fresh_e"))
sys.path.insert(0, str(SR / "bank"))

import unseen_pipeline as UP  # noqa: E402  (imported, never edited)

_UNSEEN_KERNELS = dict(UP.KERNELS)
import fresh_e_lib as FE  # noqa: E402  (imported, never edited; it points UP at set E and replaces UP.INTERP)

UP.KERNELS.update(_UNSEEN_KERNELS)          # one registry: unseen-kernel set + set E
for _kid, _spec in UP.KERNELS.items():
    UP.A.PARAM_FIELDS[_kid] = _spec["params"]

A, C, P = UP.A, UP.C, UP.P

# --------------------------------------------------------------------------- 1. constant-bank layout per architecture
# From the audited SASS (port_audit/opcode_audit_result.json): sm_90 and sm_80/sm_89 read blockDim at 0x0/0x4/0x8, gridDim at
# 0xc/0x10/0x14, the initial stack pointer at 0x28 and the global-memory descriptor at 0x208 (sm_90) or 0x118 (sm_80, sm_89).
ABI = {
    "sm_120": dict(param_base=0x380, implicit={"blockDim.x": 0x360, "blockDim.y": 0x364, "blockDim.z": 0x368,
                                               "gridDim.x": 0x370, "gridDim.y": 0x374, "gridDim.z": 0x378},
                   also={0x358: "global memory descriptor base (desc[UR][...] operands)", 0x37C: "initial stack pointer"}),
    "sm_90": dict(param_base=0x210, implicit={"blockDim.x": 0x0, "blockDim.y": 0x4, "blockDim.z": 0x8,
                                              "gridDim.x": 0xc, "gridDim.y": 0x10, "gridDim.z": 0x14},
                  also={0x208: "global memory descriptor base (desc[UR][...] operands)", 0x28: "initial stack pointer"}),
    "sm_80": dict(param_base=0x160, implicit={"blockDim.x": 0x0, "blockDim.y": 0x4, "blockDim.z": 0x8,
                                              "gridDim.x": 0xc, "gridDim.y": 0x10, "gridDim.z": 0x14},
                  also={0x118: "global memory descriptor base (unused by [R.64] operands; loaded into a uniform register)",
                        0x28: "initial stack pointer"}),
}
ABI["sm_89"] = ABI["sm_80"]
# cuobjdump's SHARED field: on sm_120 and sm_90 it is the declared static bytes + the 1,024-byte per-block reserve (Blackwell:
# HARDWARE_GROUND_TRUTH.md M328); on sm_80 and sm_89 it is the declared bytes only, although the driver reserves 1,024 B per block
# there too (A100 live query, job j06). Evidence: all eleven audited functions of the same source differ by exactly 1,024
# between the sm_80/sm_89 and sm_90/sm_120 builds (e.g. matrix multiply 32x32: 8,192 vs 9,216; dynamic-only kernels 0 vs 1,024).
# The adapter computes static = SHARED - reserve, so for sm_80/89 the resource line is normalised to SHARED + reserve.
SHARED_INCLUDES_RESERVE = {"sm_120": True, "sm_90": True, "sm_80": False, "sm_89": False}
RESERVE_BYTES = 1024

# --------------------------------------------------------------------------- 3. operand forms
# Global: group 1 descriptor register (absent on sm_80/sm_89), group 2 address pair register, group 3 offset (same groups as
# coalescing.MEMRE, which it replaces in memory).
MEMRE_PORT = re.compile(r'(?:desc\[(UR\d+)\])?\[(R\d+)\.64(?:\+(-?0x[0-9a-f]+))?\]$')
_SCALED = re.compile(r'(U?R\d+)\.X(\d+)$')


def shared_terms(inner):
    """[(register or None, scale, immediate)] for an operand body like R0.X4+UR4+0x1000; None if a term is not understood."""
    out = []
    for term in inner.replace('.reuse', '').split('+'):
        term = term.strip()
        if term == '':
            continue
        m = _SCALED.fullmatch(term)
        if m:
            out.append((m[1], int(m[2]), 0))
        elif re.fullmatch(r'-?0x[0-9a-f]+', term):
            out.append((None, 0, int(term, 0)))
        elif term in ('RZ', 'URZ'):
            out.append((None, 0, 0))
        elif re.fullmatch(r'U?R\d+', term):
            out.append((term, 1, 0))
        else:
            return None
    return out


def shared_address(inner, get):
    """Exact shared byte address from operand terms and a register getter (exact int or None)."""
    terms = shared_terms(inner)
    if terms is None:
        return None
    total = 0
    for reg, scale, imm in terms:
        if reg is None:
            total += imm
            continue
        v = get(reg)
        if v is None:
            return None
        total += scale * v
    return total & 0xFFFFFFFF


# --------------------------------------------------------------------------- 4. interpreter
PORT_OPS = frozenset("ULDC ULDC.64 IMAD.MOV IMAD.MOV.U32 IMAD.IADD IMAD.U32 VIADD IADD3.X UIADD3.X IMAD.X LEA.HI.X.SX32 "
                     "HFMA2.MMA I2F BAR.SYNC UISETP.GT.AND UISETP.NE.U32.AND WARPSYNC UPRMT BRA.DIV".split())
_ANCHORS = [
    # unseen-kernel extension (unseen_pipeline.py), replicated
    (UP._NEEDLE, UP._REPL),
    # set-E extension (fresh_e_lib.py), replicated; bare WARPSYNC added as a logged no-op
    ("            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n",
     "            if o in ('BSSY.RELIABLE', 'BREAK.RELIABLE', 'BSYNC.RELIABLE', 'WARPSYNC.ALL'):\n                pc += 16\n                continue\n"
     "            if o == 'WARPSYNC':\n                self.assumption_log[('warpsync_noop', pc)] += 1\n                pc += 16\n                continue\n"
     "            if o == 'BRA.DIV':\n"
     "                require(do is not None, 'unknown BRA.DIV predicate at ' + hex(pc))\n"
     "                m_ = V.exact(0xFFFFFFFF) if a[0] in ('~URZ', '~RZ') else val(a[0], r)\n"
     "                require(len(a) == 2 and m_ is not None and m_.exact_value is not None and 'PORT_LANE' in coords, 'unknown BRA.DIV mask at ' + hex(pc))\n"
     "                self.port_div[(coords['PORT_LANE'].lo, pc, m_.lo)] += 1\n"
     "                self.assumption_log[('bra_div_converged', pc)] += 1\n"
     "                pc += 16\n                continue\n"
     "            if o == 'CALL.REL.NOINC' and len(a) == 1 and int(a[0], 0) in by and by[int(a[0], 0)].op == 'EXIT':\n"
     "                require(do is not None, 'unknown call-to-exit predicate at ' + hex(pc))\n"
     "                self.assumption_log[('call_to_exit', pc)] += 1\n"
     "                pc = int(a[0], 0)\n                continue\n"
     "            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)\n"),
    # add with one predicate carry-out (IADD3 Rd, Pc, a, b, c: five operands) is allowed; two carry-outs still refuse
    ("            if o in ('IADD3', 'UIADD3'):\n                require(a[1:3] in (('PT', 'PT'), ('UPT', 'UPT')), 'unsupported add carry outputs')\n",
     "            if o in ('IADD3', 'UIADD3'):\n                require(a[1:3] in (('PT', 'PT'), ('UPT', 'UPT')) or len(a) == 4 or (len(a) == 5 and re.fullmatch(r'U?P\\d+', a[1])), 'unsupported add carry outputs')\n"),
    # new opcode branches, placed before the constant-load branch; ULDC joins the constant-load branch
    ("            elif o.startswith(('LDC', 'LDCU')):\n",
     "            elif o in ('IMAD.MOV', 'IMAD.MOV.U32', 'IMAD.IADD', 'IMAD.U32'):\n"
     "                result = self.imad([val(x, r) for x in a[1:4]])\n"
     "            elif o == 'VIADD':\n"
     "                cv = [self.val_coef(x, r) for x in a[1:3]]\n"
     "                result = self.affine(cv) if all(v is not None for _, v in cv) else None\n"
     "            elif o in ('IADD3', 'UIADD3') and len(a) == 4:\n"
     "                cv = [self.val_coef(x, r) for x in a[1:4]]\n"
     "                result = self.affine(cv) if all(v is not None for _, v in cv) else None\n"
     "            elif o in ('IADD3', 'UIADD3') and len(a) == 5:\n"
     "                r[a[0]], p[a[1]] = self.port_add_carry(a[2:5], r)\n"
     "                pc += 16\n"
     "                continue\n"
     "            elif o in ('IADD3.X', 'UIADD3.X'):\n"
     "                result = self.port_add_x(a[1:4], a[4:6], r, p)\n"
     "            elif o == 'IMAD.X':\n"
     "                result = self.port_imad_x(a[1:4], a[4], r, p)\n"
     "            elif o == 'LEA.HI.X.SX32':\n"
     "                result = self.port_lea_hi_x_sx32(a, r, p)\n"
     "            elif o == 'HFMA2.MMA':\n"
     "                result = self.port_hfma2_const(a)\n"
     "            elif o == 'R2UR' and self.port_r2ur:\n"
     "                result = val(a[1], r)\n"
     "            elif o == 'I2F':\n"
     "                x = val(a[1], r)\n"
     "                result = self.fres(Fraction(x.lo - (1 << 32) if x.lo >> 31 else x.lo)) if x is not None and x.exact_value is not None else None\n"
     "            elif o.startswith(('LDC', 'LDCU', 'ULDC')):\n"),
    ("                    if not o.startswith(('LDC', 'LDCU')) or do is None:\n",
     "                    if not o.startswith(('LDC', 'LDCU', 'ULDC')) or do is None:\n"),
    # IMAD.HI.U32 addend is the 64-bit pair {R(c+1):R(c)} (fresh/divider_fix.py, FRESH.md "Pre-existing interpreter issue").
    # The committed interpreter adds only R(c). The corrected form is applied when port_divider_fix is set.
    ("            elif ext and o == 'IMAD.HI.U32':\n"
     "                vs = [val(x, r) for x in a[1:4]]\n"
     "                if all(v is not None and v.exact_value is not None for v in vs):\n"
     "                    result = V.exact(((vs[0].lo * vs[1].lo) >> 32) + vs[2].lo)\n",
     "            elif ext and o == 'IMAD.HI.U32' and self.port_divider_fix:\n"
     "                vs = [val(x, r) for x in a[1:3]]\n"
     "                c64 = self.val64(a[3], r)\n"
     "                if all(v is not None and v.exact_value is not None for v in vs) and c64 is not None:\n"
     "                    result = V.exact((((vs[0].lo * vs[1].lo) + c64) >> 32) & 0xFFFFFFFF)\n"
     "            elif ext and o == 'IMAD.HI.U32':\n"
     "                vs = [val(x, r) for x in a[1:4]]\n"
     "                if all(v is not None and v.exact_value is not None for v in vs):\n"
     "                    result = V.exact(((vs[0].lo * vs[1].lo) >> 32) + vs[2].lo)\n"),
    # UPRMT (uniform byte permute; CUDA 12.1 sm_90 builds the shared-window base with it) uses the committed PRMT semantics
    ("            elif ext and o == 'PRMT':\n", "            elif ext and o in ('PRMT', 'UPRMT'):\n"),
]


def _port_lane_gen():
    src = inspect.getsource(P.Interp._lane_gen)
    for needle, repl in _ANCHORS:
        if src.count(needle) != 1:
            raise SystemExit("port extension: interpreter anchor changed: " + needle.strip()[:70])
        src = src.replace(needle, repl)
    env = dict(P.Interp._lane_gen.__globals__)
    env["Fraction"] = Fraction
    exec(compile(textwrap.dedent(src), "<cross-gpu-port-ext>", "exec"), env)
    return env["_lane_gen"]


def _half_bits(tok):
    x = float(tok)
    b = struct.unpack('<H', struct.pack('<e', x))[0]
    if struct.unpack('<e', struct.pack('<H', b))[0] != x:
        return None                     # not exactly representable: do not guess
    return b


class PortInterp(FE.UP.INTERP):
    """Set-E interpreter (which includes the unseen-kernel extension) plus the cross-GPU extensions."""
    _lane_gen = _port_lane_gen()
    # R2UR (register to uniform register) is whitelisted but valueless in the committed interpreter. Its copy semantics are
    # enabled only for the new architectures so that Blackwell results stay exactly as committed.
    port_r2ur = False
    # Corrected IMAD.HI.U32 (64-bit addend); off by default so committed results are reproduced; see FINDINGS / CHANGELOG.
    port_divider_fix = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.KNOWN_OPS = set(self.KNOWN_OPS) | PORT_OPS
        self._consts = {}

    # ---- constants passed to run_block / run_lane are kept so that c[0x0][off] operands resolve
    def run_block(self, sites, constants, coords_for_lane, threads, max_steps=100000):
        self._consts = constants
        self.port_div = collections.Counter()      # (lane, pc, mask) -> arrivals at BRA.DIV

        def coords(l):
            c = dict(coords_for_lane(l))
            c["PORT_LANE"] = P.V.exact(l)          # lane index for the BRA.DIV convergence check; read by nothing else
            return c
        out = super().run_block(sites, constants, coords, threads, max_steps)
        self._check_div_convergence(threads)
        return out

    def _check_div_convergence(self, threads):
        """Same rule as the committed collective extension (operator_derivation/collective.py, check_warp_convergence): every
        warp reaching a BRA.DIV site arrives as one full warp on mask 0xffffffff with equal per-lane counts; else refuse."""
        by_site = collections.defaultdict(dict)
        for (lane, pc, mask), n in self.port_div.items():
            by_site[(lane // 32, pc, mask)][lane] = n
        for (w, pc, mask), lanes in by_site.items():
            P.require(mask == 0xffffffff, 'BRA.DIV mask %#x is not the full-warp mask at %s' % (mask, hex(pc)))
            P.require(len(lanes) == 32, 'partial warp %d reaches BRA.DIV at %s' % (w, hex(pc)))
            P.require(len(set(lanes.values())) == 1, 'warp %d diverges at BRA.DIV %s' % (w, hex(pc)))

    def run_lane(self, sites, constants, coords, max_steps=100000):
        self._consts = constants
        return super().run_lane(sites, constants, coords, max_steps)

    @staticmethod
    def _cbank(t):
        m = re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]', t)
        return int(m[1], 0) if m else None

    def val(self, t, r):
        tt = t.replace('.reuse', '')
        neg = tt.startswith('-')
        off = self._cbank(tt[1:] if neg else tt)
        if off is None:
            return super().val(t, r)
        v = self._consts.get(off)
        if v is None or not neg:
            return v
        return P.V.exact(-v.lo) if v.exact_value is not None else self.affine([(-1, v)])

    def val64(self, t, r):
        tt = t.replace('.reuse', '')
        neg = tt.startswith('-')
        off = self._cbank(tt[1:] if neg else tt)
        if off is None:
            return super().val64(t, r)
        lo, hi = self._consts.get(off), self._consts.get(off + 4)
        if lo is None or hi is None or lo.exact_value is None or hi.exact_value is None:
            return None
        x = (hi.lo << 32) | lo.lo
        return (-x) & P.M64 if neg else x

    def fval(self, t, r):
        tt = t.replace('.reuse', '').strip()
        neg = tt.startswith('-')
        body = tt[1:] if neg else tt
        absf = body.startswith('|') and body.endswith('|')
        body = body[1:-1] if absf else body
        off = self._cbank(body)
        if off is None:
            return super().fval(t, r)
        v = self._consts.get(off)
        if v is None or v.exact_value is None:
            return None
        x = P.f32_value(v.lo)
        if x is None:
            return None
        x = abs(x) if absf else x
        return -x if neg else x

    def smem_addr(self, tok, r):
        # Only operands with a scaled term are new; every other form keeps the committed behaviour exactly (which, for
        # example, leaves [R11+UR5] unknown although the bank-conflict evaluator resolves it).
        t = tok.replace('.reuse', '')
        if '.X' not in t:
            return super().smem_addr(tok, r)
        if not (t.startswith('[') and t.endswith(']')):
            return None

        def get(name):
            v = r.get(name)
            return None if v is None or v.exact_value is None else v.lo
        return shared_address(t[1:-1], get)

    # ---- carry arithmetic (exact operands only; an unknown operand gives an unknown result and an unknown carry)
    def _plain(self, toks, r):
        if any(t.replace('.reuse', '').startswith(('-', '~')) for t in toks):
            return None                 # negated operands in a carry chain: not modelled
        vs = [self.val(t, r) for t in toks]
        if any(v is None or v.exact_value is None for v in vs):
            return None
        return [v.lo for v in vs]

    def port_add_carry(self, toks, r):
        xs = self._plain(toks, r)
        if xs is None:
            return None, None
        s = sum(xs)
        if s >> 32 > 1:
            return P.V.exact(s), None   # two carries: the second carry output is PT, so the bit is not representable
        return P.V.exact(s), bool(s >> 32)

    def port_add_x(self, toks, ptoks, r, p):
        xs = self._plain(toks, r)
        cs = [self.pred(t, p) for t in ptoks]
        if xs is None or any(c is None for c in cs):
            return None
        return P.V.exact(sum(xs) + sum(int(c) for c in cs))

    def port_imad_x(self, toks, ptok, r, p):
        xs = self._plain(toks, r)
        c = self.pred(ptok, p)
        if xs is None or c is None:
            return None
        return P.V.exact(xs[0] * xs[1] + xs[2] + int(c))

    def port_lea_hi_x_sx32(self, a, r, p):
        # LEA.HI.X.SX32 Rd, Ra, Rb, n, Pc: high word of (sign-extended Ra << n) + Rb + carry
        xs = self._plain([a[1], a[2], a[3]], r)
        c = self.pred(a[4], p) if len(a) > 4 else False
        if xs is None or c is None:
            return None
        x, b, n = xs
        sx = x - (1 << 32) if x >> 31 else x
        return P.V.exact((((sx << n) >> 32) + b + int(c)))

    def port_hfma2_const(self, a):
        # HFMA2.MMA Rd, -RZ, RZ, hi, lo: (-0 * 0) + (hi, lo) = the two half-precision immediates; last operand = low half
        if tuple(a[1:3]) != ('-RZ', 'RZ') or len(a) != 5:
            return None
        hi, lo = _half_bits(a[3]), _half_bits(a[4])
        if hi is None or lo is None:
            return None
        return P.V.exact((hi << 16) | lo)


# --------------------------------------------------------------------------- logged assumptions in the feature rows
A.ASSUMPTION_TEXT.update({
    "call_to_exit": (
        "Call to exit. On sm_80/sm_89 builds, a grid-stride loop ends with `@P CALL.REL.NOINC <target>` whose target "
        "instruction is EXIT (no RET anywhere on that path), so a taken call is treated as a branch to EXIT. Accepted only "
        "when the target is EXIT and the predicate is known; any other CALL still refuses."),
    "bra_div_converged": (
        "Converged warp at BRA.DIV. `BRA.DIV Ux, target` enters a warp-collective fallback only when the warp is not converged "
        "on mask Ux; the fall-through (converged fast path) is taken and, as in the committed collective extension, every warp "
        "is verified to arrive with all 32 lanes, the full mask and equal per-lane counts, else the kernel is refused."),
    "warpsync_noop": (
        "Bare WARPSYNC. `WARPSYNC Rn` (sm_80/sm_89 reductions) only orders the warp's lanes; the interpreter walks lanes "
        "independently and exchanges shuffle values in lock-step, so it is treated as a no-op, as WARPSYNC.ALL is in set E."),
})
_KF_NEEDLE = '{"fchk_fast_path": "division_fast_path", "forced_branch": "bounds_assert_not_taken"}'


def _port_kernel_features():
    src = inspect.getsource(A.kernel_features)
    if src.count(_KF_NEEDLE) != 1:
        raise SystemExit("port extension: adapter.kernel_features anchor changed")
    src = src.replace(_KF_NEEDLE, '{"fchk_fast_path": "division_fast_path", "forced_branch": "bounds_assert_not_taken", '
                                  '"call_to_exit": "call_to_exit", "warpsync_noop": "warpsync_noop", '
                                  '"bra_div_converged": "bra_div_converged"}')
    env = dict(A.kernel_features.__globals__)
    exec(compile(textwrap.dedent(src), "<cross-gpu-port-features>", "exec"), env)
    return env["kernel_features"]


A.kernel_features = _port_kernel_features()

# --------------------------------------------------------------------------- install
STATE = {}
_orig_abi_check = A.abi_check


def _abi_check_halves(kid, sites):
    """The adapter's ABI check, plus one accepted form: on sm_80/89/90 a 64-bit pointer parameter's upper word is often read
    by a separate 4-byte access (c[0x0][base+4]), where Blackwell reads the pair with LDC.64. Only that exact case is
    accepted: a 4-byte access starting 4 bytes into a declared 8-byte field. Every other problem is kept."""
    ok, problems = _orig_abi_check(kid, sites)
    upper = {o + 4 for _, o, sz in A.abi_layout(kid) if sz == 8}
    kept = []
    for msg in problems:
        m = re.match(r'c\[0x0\]\[(0x[0-9a-f]+)\] \(4 B\) does not start at a declared field$', msg)
        if m and int(m[1], 0) in upper:
            continue
        kept.append(msg)
    return not kept, kept


def install(arch, sass_root=None):
    """Point the pipeline at one architecture. `sass_root` holds compiled/{sass,log,cubin}-like folders
    (sass/, log/, cubin/); default: the committed Blackwell directories of the two CUDA-samples sets."""
    if arch not in ABI:
        raise SystemExit("unknown architecture " + arch)
    abi = ABI[arch]
    A.PARAM_BASE = abi["param_base"]
    A.IMPLICIT.clear(); A.IMPLICIT.update(abi["implicit"])
    A.IMPLICIT_ALSO_ACCESSED.clear(); A.IMPLICIT_ALSO_ACCESSED.update(abi["also"])
    C.MEMRE = MEMRE_PORT
    A.abi_check = _orig_abi_check if arch == "sm_120" else _abi_check_halves
    UP.INTERP = PortInterp
    PortInterp.port_r2ur = arch != "sm_120"
    # Corrected IMAD.HI.U32 on the new architectures. On Blackwell it changes none of the committed cells that use the
    # instruction (set E fast Walsh fwt1, all 8 cells; set D transposeDiagonal, both cells: identical with and without),
    # but it is kept off there so the committed interpreter is reproduced by construction.
    PortInterp.port_divider_fix = arch != "sm_120"
    UP._KERNEL_CACHE.clear()
    _cubin_dir[0] = None            # default: the committed Blackwell folders (chosen per kernel in _kernel_assets)
    # The pipeline writes each kernel's isolated SASS as a side effect; keep that out of the repository.
    import tempfile
    UP.ISO_DIR = Path(tempfile.gettempdir()) / ("port_isolated_" + arch)
    if sass_root is not None:
        root = Path(sass_root)
        UP.HERE = root.parent if root.name == "compiled" else root
        UP.SASS_DIR, UP.LOG_DIR = root / "sass", root / "log"
        _cubin_dir[0] = root / "cubin"
    STATE.update(arch=arch, sass_root=str(sass_root))
    _install_bank()


_cubin_dir = [None]
_orig_kernel_assets = UP.kernel_assets


def _kernel_assets(kid):
    """As the pipeline's kernel_assets, but the cubin hash is read from the selected architecture's folder and the per-kernel
    SASS is looked up per directory (the committed function looks for both sets' SASS in one folder)."""
    if kid in UP._KERNEL_CACHE:
        return UP._KERNEL_CACHE[kid]
    s = UP.KERNELS[kid]
    if _cubin_dir[0] is None:
        # committed Blackwell layout: unseen-kernel SASS and set-E SASS live in two folders
        d = SR / ("fresh_e" if s["sass"] in ("scalarProd", "fastWalshTransform") else "unseen_kernels") / "compiled"
        UP.SASS_DIR, UP.LOG_DIR, UP.HERE = d / "sass", d / "log", d.parent
    return _orig_kernel_assets(kid) if _cubin_dir[0] is None else _assets_from(kid, s)


def _assets_from(kid, s):
    text = UP.isolated_text(s["sass"], s["needle"])
    mangled, res_line, reg, shared = UP.resource_record(s["sass"], s["needle"])
    if not SHARED_INCLUDES_RESERVE[STATE["arch"]]:
        res_line = re.sub(r"SHARED:(\d+)", lambda m: "SHARED:%d" % (int(m[1]) + RESERVE_BYTES), res_line)
        shared += RESERVE_BYTES
    # `c[0x0][RZ]` is a constant read indexed by the zero register, i.e. offset 0 (blockDim.x on sm_80/89/90); written out as
    # the immediate form so the constant-load branch, the operand resolver and the ABI check all see it.
    sites = [P.Site(s.pc, s.pred, s.op, tuple('c[0x0][0x0]' if x == 'c[0x0][RZ]' else x for x in s.a)) for s in P.parse(text)]
    ok, problems = A.abi_check(kid, sites)
    rec = dict(mangled=mangled, res_usage_line=res_line, registers_per_thread=reg, static_shared_bytes=shared)
    UP._KERNEL_CACHE[kid] = dict(text=text, sites=sites, idx=rec, abi_ok=ok, abi_problems=problems,
                                 sha=UP.sha(text.encode()), cubin_sha=UP.sha((_cubin_dir[0] / (s["sass"] + ".cubin")).read_bytes()))
    return UP._KERNEL_CACHE[kid]


UP.kernel_assets = _kernel_assets


def _install_bank():
    """Scaled shared-address terms in the bank-conflict evaluator (bank/bank_conflicts.py, imported, never edited)."""
    try:
        import bank_conflicts as B
    except Exception:            # the bank module is optional for callers that only need phases
        return
    if getattr(B.eval_terms, "_port", False):
        return
    orig = B.eval_terms

    def eval_terms(inner, getter):
        if '.X' not in inner:
            return orig(inner, getter)
        return shared_address(inner, getter)
    eval_terms._port = True
    B.eval_terms = eval_terms
