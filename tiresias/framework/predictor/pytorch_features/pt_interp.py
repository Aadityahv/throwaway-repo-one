"""Extended per-lane SASS interpreter for the traced PyTorch kernels (CPU only).

This is a copy of the frozen derivation's `derive.run_lane` (tiresias/framework/compile_evidence/
operator_derivation/derive.py, left untouched) with explicitly listed extensions that the PyTorch
kernels need. With `ext=False` it reproduces `derive.run_lane` exactly (checked by
test_pytorch_features.py on the retained CUDA/Triton cells). With `ext=True` it adds:

  1. Interval arithmetic with modular wrap-around and a trailing-zero-bit tag (so that
     `tid | (blockIdx << 8)` and `batch - first_batch` are decided over the block-index interval).
  2. 64-bit register pairs for the ops these kernels use (IADD.64, IMAD.WIDE, SEL.64, ISETP.*.S64).
  3. Exact IEEE-754 binary32 evaluation of FADD, FMUL, FFMA (default rounding), I2FP, I2F.RP,
     MUFU.RCP (modelled as correctly rounded; sensitivity-tested), F2I and FSETP, but only when every
     operand is a known value. Any operand loaded from memory is unknown and poisons the result.
  4. A few integer ops (PRMT, LEA.HI, SHF.R.S32.HI, IMAD.HI.U32, VIMNMX, LDC.U8, LDCU.128).
  5. Two explicitly logged assumption mechanisms (never silent):
       - FCHK (division fast-path check): predicate forced False (the division slow path is not
         entered). Logged under "fchk_fast_path".
       - A per-binary list of branch PCs forced taken/not taken (used for a device-side bounds
         assert). Logged under "forced_branch".
Every undecidable guard on a counted instruction, on a branch, or on EXIT still refuses.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from fractions import Fraction

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1


class Refusal(ValueError):
    pass


def require(ok, msg):
    if not ok:
        raise Refusal(msg)


def _tz(n: int) -> int:
    n &= M32
    return 32 if n == 0 else (n & -n).bit_length() - 1


@dataclass(frozen=True)
class V:
    lo: int
    hi: int
    tz: int = field(default=0, compare=False)

    @classmethod
    def exact(cls, n):
        n &= M32
        return cls(n, n, _tz(n))

    @property
    def exact_value(self):
        return self.lo if self.lo == self.hi else None

    def tzeros(self):
        return _tz(self.lo) if self.lo == self.hi else self.tz


# --------------------------------------------------------------------------- parsing
LINE = re.compile(r'\s*/\*([0-9a-f]+)\*/\s+(?:@(!?(?:U?P\d+|U?PT))\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;')


@dataclass(frozen=True)
class Site:
    pc: int
    pred: str | None
    op: str
    a: tuple


def parse(text):
    sites = []
    for line in text.splitlines():
        if not re.match(r'\s*/\*[0-9a-f]+\*/\s+[A-Z@]', line):
            continue
        m = LINE.match(line)
        require(m is not None, 'unparsed instruction')
        pc, p, o, a = m.groups()
        sites.append(Site(int(pc, 16), p, o, tuple(x.strip() for x in a.split(','))))
    require(sites and [s.pc for s in sites] == list(range(0, len(sites) * 16, 16)), 'PC gap')
    return sites


# --------------------------------------------------------------------------- binary32 helpers
def f32_value(bits: int):
    """Fraction value of a finite normal or zero binary32, else None (nan/inf/denormal/-0 unknown)."""
    bits &= M32
    s, e, m = bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if e == 0xFF:
        return None
    if e == 0:
        return Fraction(0) if (m == 0 and s == 0) else None
    v = Fraction((1 << 23) | m, 1 << 23) * (Fraction(2) ** (e - 127))
    return -v if s else v


def f32_round(x: Fraction, mode: str = 'rn'):
    """Round a Fraction to binary32 bits. None when the result is not a normal number or zero."""
    if x == 0:
        return 0
    s = 1 if x < 0 else 0
    a = -x if s else x
    e = 0
    # find e with 2^e <= a < 2^(e+1)
    n, d = a.numerator, a.denominator
    e = n.bit_length() - d.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126 or e > 127:
        return None
    scaled = a / (Fraction(2) ** (e - 23))  # in [2^23, 2^24)
    fl = scaled.numerator // scaled.denominator
    rem = scaled - fl
    if mode == 'rn':
        up = rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1)
    elif mode == 'rp':       # toward +inf: magnitude rounds up only for positive values
        up = rem > 0 and not s
    elif mode == 'rz':
        up = False
    else:
        raise Refusal('unsupported rounding mode ' + mode)
    m = fl + (1 if up else 0)
    if m == 1 << 24:
        m >>= 1
        e += 1
        if e > 127:
            return None
    return (s << 31) | ((e + 127) << 23) | (m & 0x7FFFFF)


class Interp:
    """Holds options; `run_lane` mirrors derive.run_lane's signature and return value."""

    def __init__(self, derive_module, ext=True, fchk_fast_path=False, forced_branches=None, rcp_ulp_delta=0, trace=None):
        self.D = derive_module
        self.ext = ext
        self.fchk_fast_path = fchk_fast_path
        self.forced_branches = dict(forced_branches or {})
        self.rcp_ulp_delta = rcp_ulp_delta
        self.KNOWN_OPS = set(derive_module.KNOWN_OPS)
        if ext:
            self.KNOWN_OPS |= EXT_KNOWN_OPS
        self.assumption_log = collections.Counter()
        self.unknown_pcs = set()
        self.shared_touched = set()
        self.trace = trace  # optional read-only pre-instruction observer; never changes interpreter state

    # ------------------------------------------------------------------ value helpers
    def val(self, t, r):
        t = t.replace('.reuse', '')
        neg = t.startswith('-')
        t = t[1:] if neg else t
        if t in ('RZ', 'URZ'):
            v = V.exact(0)
        elif re.fullmatch(r'U?R\d+', t):
            v = r.get(t)
        else:
            try:
                v = V.exact(int(t, 0))
            except ValueError:
                return None
        if neg:
            if v is None:
                return None
            if v.exact_value is not None:
                return V.exact(-v.lo)
            if self.ext:
                return self.affine([(-1, v)])
            return None
        return v

    def val_coef(self, t, r):
        """(+1|-1, value) with the sign kept as a coefficient (so negated intervals stay representable)."""
        t = t.replace('.reuse', '')
        if t.startswith('-') and re.fullmatch(r'-U?R\d+', t):
            return -1, self.val(t[1:], r)
        return 1, self.val(t, r)

    @staticmethod
    def _signed(v):
        if v.hi < 0x80000000:
            return v.lo, v.hi
        if v.lo >= 0x80000000:
            return v.lo - (1 << 32), v.hi - (1 << 32)
        return None

    def affine(self, terms, const=0):
        """sum(coef * v) + const over intervals with modular wrap; None if the image straddles a wrap."""
        L = H = const
        tz = 32
        for coef, v in terms:
            if v is None:
                return None
            sv = self._signed(v)
            if sv is None:
                return None
            lo, hi = sv
            a, b = coef * lo, coef * hi
            L += min(a, b)
            H += max(a, b)
            tz = min(tz, v.tzeros() + (_tz(coef) if coef else 32))
        if L == H:
            return V.exact(L)
        k = L >> 32
        if (H >> 32) != k:
            return None
        return V(L - (k << 32), H - (k << 32), min(tz, _tz(const) if const else 32, 31))

    @staticmethod
    def _signed_const(v):
        x = v.exact_value
        return x if x < 0x80000000 else x - (1 << 32)

    def calc_derive(self, vs, fun):
        if any(v is None for v in vs):
            return None
        if all(v.exact_value is not None for v in vs):
            return V.exact(fun(*(v.lo for v in vs)))
        lo = fun(*(v.lo for v in vs))
        hi = fun(*(v.hi for v in vs))
        return V(lo, hi) if 0 <= lo <= hi <= 0xFFFFFFFF else None

    def cmp(self, x, y, rel, unsigned):
        if x is None or y is None:
            return None

        def signed(v):
            if unsigned:
                return v
            if v.hi < 0x80000000:
                return v
            if v.lo >= 0x80000000:
                return V(v.lo - 2 ** 32, v.hi - 2 ** 32)
            return None
        x, y = signed(x), signed(y)
        if x is None or y is None:
            return None
        if rel == 'EQ':
            ne = self.cmp(x, y, 'NE', True)
            return not ne if ne is not None else None
        if rel == 'NE':
            if x.hi < y.lo or y.hi < x.lo:
                return True
            if x.lo == x.hi and y.lo == y.hi:
                return x.lo != y.lo
        if rel == 'GE':
            if x.lo >= y.hi:
                return True
            if x.hi < y.lo:
                return False
        if rel == 'GT':
            if x.lo > y.hi:
                return True
            if x.hi <= y.lo:
                return False
        if rel == 'LT':
            return self.cmp(y, x, 'GT', True)
        if rel == 'LE':
            return self.cmp(y, x, 'GE', True)
        return None

    @staticmethod
    def pred(t, p):
        inv = t.startswith('!')
        t = t.lstrip('!')
        v = True if t in ('PT', 'UPT') else p.get(t)
        return not v if inv and v is not None else v

    # ---- pairs (64-bit)
    @staticmethod
    def pair_names(t):
        t = t.replace('.reuse', '').replace('.64', '')
        m = re.fullmatch(r'(U?R)(\d+)', t)
        if not m:
            return None
        return m[1] + m[2], m[1] + str(int(m[2]) + 1)

    def val64(self, t, r):
        t = t.replace('.reuse', '')
        neg = t.startswith('-')
        t = t[1:] if neg else t
        if t in ('RZ', 'URZ'):
            x = 0
        elif re.fullmatch(r'U?R\d+', t):
            lo_n, hi_n = self.pair_names(t)
            lo, hi = r.get(lo_n), r.get(hi_n)
            if lo is None or hi is None or lo.exact_value is None or hi.exact_value is None:
                return None
            x = (hi.lo << 32) | lo.lo
        else:
            try:
                x = int(t, 0) & M64
            except ValueError:
                return None
        return (-x) & M64 if neg else x

    @staticmethod
    def sx64(x):
        return x - (1 << 64) if x >> 63 else x

    # ---- float operand
    def fval(self, t, r):
        """Fraction | None for a float operand token (with -, |.| modifiers)."""
        t = t.replace('.reuse', '').strip()
        neg = absf = False
        if t.startswith('-'):
            neg, t = True, t[1:]
        if t.startswith('|') and t.endswith('|'):
            absf, t = True, t[1:-1]
        if t in ('RZ', 'URZ'):
            x = Fraction(0)
        elif re.fullmatch(r'U?R\d+', t):
            v = r.get(t)
            if v is None or v.exact_value is None:
                return None
            x = f32_value(v.lo)
        elif t.startswith('0x'):
            x = f32_value(int(t, 16))
        else:
            try:
                fr = Fraction(t)
            except (ValueError, ZeroDivisionError):
                return None
            if fr == 0:
                x = Fraction(0)
            else:
                b = f32_round(fr)
                x = None if b is None else f32_value(b)
        if x is None:
            return None
        if absf:
            x = abs(x)
        if neg:
            x = -x
        return x

    def fres(self, x, mode='rn'):
        if x is None:
            return None
        b = f32_round(x, mode)
        return None if b is None else V.exact(b)

    # ------------------------------------------------------------------ main loop
    def run_lane(self, sites, constants, coords, max_steps=100000):
        """Single independent lane (no cross-lane value exchange: shuffle results and shared memory are unknown).
        Same return value as derive.run_lane."""
        gen = self._lane_gen(sites, constants, coords, {}, max_steps)
        try:
            req = next(gen)
            while True:
                req = gen.send(None)
        except StopIteration as stop:
            return stop.value

    def _lane_gen(self, sites, constants, coords, shm, max_steps=100000):
        D, ext = self.D, self.ext
        val, pred, cmp = self.val, self.pred, self.cmp
        r = {}
        p = {}
        events = collections.Counter()
        visits = collections.Counter()
        unknown_unpriced = collections.Counter()
        by = {s.pc: s for s in sites}
        pc = 0
        self.debug_state = (r, p)
        target = D.target
        for step in range(max_steps):
            require(pc in by, 'control outside function')
            s = by[pc]
            a = s.a
            o = s.op
            t = target(o)
            require(o in self.KNOWN_OPS, 'unsupported opcode ' + o)
            if o in ('IADD3', 'UIADD3'):
                require(a[1:3] in (('PT', 'PT'), ('UPT', 'UPT')), 'unsupported add carry outputs')
            if o.startswith(('ISETP.', 'UISETP.')) or o == 'PLOP3.LUT':
                require(a[1] in ('PT', 'UPT'), 'unsupported second predicate output')
            if o in ('LOP3.LUT', 'ULOP3.LUT'):
                require(a[-1] in ('!PT', '!UPT'), 'unsupported LOP3 predicate combiner')
                if a[0].startswith(('P', 'UP')):
                    require(a[5] in ('0xc0', '0xfc'), 'unsupported LOP3 predicate LUT')
            # ---- forced branch assumption (ext only, logged)
            if ext and o in ('BRA', 'BRA.U') and pc in self.forced_branches:
                take = self.forced_branches[pc]
                self.assumption_log[('forced_branch', pc)] += 1
                if self.trace is not None:
                    self.trace(coords, s, True, r, p)
                visits[pc] += 1
                pc = int(a[-1], 0) if take else pc + 16
                continue
            do = pred(s.pred, p) if s.pred else True
            if self.trace is not None:
                self.trace(coords, s, do, r, p)
            if t:
                require(do is not None, 'unknown target predicate at ' + hex(pc))
            if o in ('EXIT', 'BRA', 'BRA.U'):
                require(do is not None, 'unknown control at ' + hex(pc) + ' ' + str(s.pred))
            if do is False:
                pc += 16
                continue
            if t:
                events[(s.pc, t[0], t[1])] += 1
            visits[s.pc] += 1
            if o == 'EXIT':
                return events, visits, unknown_unpriced
            if o in ('BRA', 'BRA.U'):
                if ext and o == 'BRA' and len(a) == 2:
                    q = pred(a[0], p)
                    require(q is not None, 'unknown branch predicate operand at ' + hex(pc))
                    take = q
                else:
                    take = True if o == 'BRA' else pred(a[0], p)
                require(take is not None, 'unknown uniform branch')
                pc = int(a[-1], 0) if take else pc + 16
                continue
            require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.', 'LEPC')), 'unsupported control ' + o)
            if o.startswith('BAR'):
                yield ('bar', pc)
                pc += 16
                continue
            if ext and o.startswith('STS'):
                addr = self.smem_addr(a[0], r)
                nw = 4 if '.128' in o else 2 if '.64' in o else 1
                srcn = a[1].replace('.reuse', '')
                if addr is None:
                    shm['poison'] = True      # a store to an unknown address may overwrite anything: later LDS are unknown
                for j in range(nw if addr is not None else 0):
                    self.shared_touched.add(addr + 4 * j)
                    if do is True:
                        shm[addr + 4 * j] = self.reg_plus(srcn, j, r)
                    else:
                        shm[addr + 4 * j] = None
                pc += 16
                continue
            if o.startswith(('STG', 'STS', 'BSSY', 'BSYNC', 'DEPBAR', 'LDGDEPBAR', 'LDGSTS', 'NOP')):
                pc += 16
                continue
            dest = a[1] if o.startswith('SHFL') or (o == 'LOP3.LUT' and a[0].startswith(('P', 'UP'))) else a[0] if a else None
            result = None
            pair_result = None  # (lo, hi) for 64-bit results
            if do is None:
                unknown_unpriced[o] += 1
                self.unknown_pcs.add(pc)
                if o == 'LOP3.LUT' and a[0].startswith(('P', 'UP')):
                    p[a[0]] = None
            elif o.startswith(('LDC', 'LDCU')):
                m = re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]', a[1])
                off = int(m[1], 0) if m else None
                result = constants.get(off)
                if ext and o == 'LDC.U8':
                    result = V.exact(result.lo & 0xFF) if result is not None and result.exact_value is not None else None
                nwords = 4 if '.128' in o else 2 if '.64' in o else 1
                if nwords > 1 and re.fullmatch(r'U?R\d+', dest):
                    pref = 'UR' if dest.startswith('UR') else 'R'
                    base = int(dest[len(pref):])
                    for j in range(1, nwords):
                        r[pref + str(base + j)] = constants.get(off + 4 * j) if off else None
            elif o in ('S2R', 'S2UR'):
                result = coords.get(a[1])
            elif o == 'CS2R' and a[1] == 'SRZ':
                result = V.exact(0)
            elif o == 'HFMA2' and a[1:3] == ('-RZ', 'RZ') and a[3:] == ('0', '0'):
                result = V.exact(0)
            elif o == 'P2R' and a[1:3] == ('PR', 'RZ'):
                mask = int(a[3], 0)
                bits = [i for i in range(7) if mask >> i & 1]
                if all(p.get('P' + str(i)) is not None for i in bits):
                    result = V.exact(sum(int(p['P' + str(i)]) << i for i in bits))
            elif o.startswith(('MOV', 'UMOV')):
                result = val(a[1], r)
            elif o in ('IMAD', 'UIMAD', 'IMAD.SHL.U32'):
                vs = [val(x, r) for x in a[1:4]]
                result = self.imad(vs) if ext else self.calc_derive(vs, lambda x, y, z: x * y + z)
            elif o == 'IADD':
                vs = [val(x, r) for x in a[1:3]]
                if ext:
                    cv = [self.val_coef(x, r) for x in a[1:3]]
                    result = self.affine(cv) if all(v is not None for _, v in cv) else None
                else:
                    result = self.calc_derive(vs, lambda x, y: x + y)
            elif o in ('IADD3', 'UIADD3'):
                require(a[1:3] in (('PT', 'PT'), ('UPT', 'UPT')), 'unsupported add carry outputs')
                vs = [val(x, r) for x in a[3:6]]
                if ext:
                    cv = [self.val_coef(x, r) for x in a[3:6]]
                    result = self.affine(cv) if all(v is not None for _, v in cv) else None
                else:
                    result = self.calc_derive(vs, lambda x, y, z: x + y + z)
            elif o in ('LEA', 'ULEA') and len(a) == 4:
                vs = [val(a[1], r), val(a[2], r), val(a[3], r)]
                if ext:
                    if all(v is not None for v in vs) and vs[2].exact_value is not None:
                        result = self.affine([(1 << vs[2].lo, vs[0]), (1, vs[1])])
                else:
                    result = self.calc_derive(vs, lambda x, y, z: (x << z) + y)
            elif ext and o == 'SHF.L.U64.HI':
                vs = [val(x, r) for x in a[1:4]]
                if all(v is not None and v.exact_value is not None for v in vs):
                    low, shift, high = (v.lo for v in vs)
                    result = V.exact((((high << 32) | low) << (shift & 63)) >> 32)
            elif ext and o in ('LEA', 'ULEA') and len(a) == 5:
                vs = [val(x, r) for x in a[2:5]]
                carry = None
                if all(v is not None and v.exact_value is not None for v in vs):
                    x, b, n = (v.lo for v in vs)
                    low_sum = ((x << n) & M32) + b
                    result = V.exact(low_sum)
                    carry = bool(low_sum >> 32)
                p[a[1]] = carry
            elif ext and o == 'LEA.HI.X' and len(a) == 6:
                vs = [val(x, r) for x in a[1:5]]
                carry = pred(a[5], p)
                if all(v is not None and v.exact_value is not None for v in vs) and carry is not None:
                    x, b, c, n = (v.lo for v in vs)
                    result = V.exact((((c << 32 | x) << n) >> 32) + b + int(carry))
            elif ext and o in ('LEA.HI', 'ULEA.HI') and len(a) == 5:
                vs = [val(x, r) for x in a[1:5]]
                if all(v is not None and v.exact_value is not None for v in vs):
                    x, b, c, n = (v.lo for v in vs)
                    hi32 = (((c << 32 | x) << n) >> 32) & M32
                    result = V.exact(hi32 + b)
            elif o.startswith(('ISETP.', 'UISETP.')):
                require(a[1] in ('PT', 'UPT'), 'unsupported second predicate output')
                rel = o.split('.')[1]
                if ext and 'S64' in o.split('.'):
                    x, y = self.val64(a[2], r), self.val64(a[3], r)
                    if x is None or y is None:
                        c = None
                    else:
                        sx, sy = self.sx64(x), self.sx64(y)
                        c = {'EQ': sx == sy, 'NE': sx != sy, 'GE': sx >= sy, 'GT': sx > sy, 'LT': sx < sy, 'LE': sx <= sy}[rel]
                else:
                    require(not (set(o.split('.')) & {'S64', 'U64'}), 'unsupported 64-bit compare')
                    c = cmp(val(a[2], r), val(a[3], r), rel, 'U32' in o)
                q = pred(a[4], p)
                comb = o.split('.')[-1]
                out = ((False if c is False or q is False else True if c is True and q is True else None) if comb == 'AND'
                       else (True if c is True or q is True else False if c is False and q is False else None) if comb == 'OR' else None)
                p[a[0]] = out if do is True else None
                pc += 16
                continue
            elif ext and o.startswith('FSETP.'):
                require(a[1] in ('PT', 'UPT'), 'unsupported second predicate output')
                rel = o.split('.')[1]
                x, y = self.fval(a[2], r), self.fval(a[3], r)
                relmap = {'GT': lambda u, v: u > v, 'GE': lambda u, v: u >= v, 'LT': lambda u, v: u < v, 'LE': lambda u, v: u <= v,
                          'EQ': lambda u, v: u == v, 'NE': lambda u, v: u != v}
                base = rel[:-1] if rel.endswith('U') and rel[:-1] in relmap else rel
                require(base in relmap, 'unsupported FSETP relation ' + rel)
                c = None if x is None or y is None else relmap[base](x, y)
                q = pred(a[4], p)
                comb = o.split('.')[-1]
                out = ((False if c is False or q is False else True if c is True and q is True else None) if comb == 'AND'
                       else (True if c is True or q is True else False if c is False and q is False else None) if comb == 'OR' else None)
                p[a[0]] = out if do is True else None
                pc += 16
                continue
            elif ext and o == 'FCHK':
                require(self.fchk_fast_path, 'FCHK present and the fast-path assumption is not enabled')
                require(a[0].startswith('P'), 'unexpected FCHK form')
                p[a[0]] = False
                self.assumption_log[('fchk_fast_path', pc)] += 1
                pc += 16
                continue
            elif o in ('LOP3.LUT', 'ULOP3.LUT'):
                has_pred = a[0].startswith(('P', 'UP'))
                start = 2 if has_pred else 1
                require(a[-1] in ('!PT', '!UPT'), 'unsupported LOP3 predicate combiner')
                if has_pred:
                    require(a[start + 3] in ('0xc0', '0xfc'), 'unsupported LOP3 predicate LUT')
                vs = [val(x, r) for x in a[start:start + 3]]
                lut = int(a[start + 3], 0)
                if all(v and v.exact_value is not None for v in vs):
                    x, y, z = [v.lo for v in vs]
                    n = 0
                    for bit in range(32):
                        n |= ((lut >> (((x >> bit & 1) << 2) | ((y >> bit & 1) << 1) | (z >> bit & 1))) & 1) << bit
                    result = V.exact(n)
                elif ext and not has_pred and lut == 0xFC and all(v is not None for v in vs):
                    # a | b | c with exactly one interval operand whose known zero low bits cover the others
                    iv = [v for v in vs if v.exact_value is None]
                    orx = 0
                    for v in vs:
                        if v.exact_value is not None:
                            orx |= v.lo
                    if len(iv) == 1 and orx < (1 << iv[0].tzeros()) and iv[0].hi + orx <= M32:
                        result = V(iv[0].lo + orx, iv[0].hi + orx, iv[0].tzeros())
                if has_pred:
                    p[a[0]] = bool(result.lo) if result else None
                    dest = a[1]
            elif o == 'PLOP3.LUT':
                require(a[1] in ('PT', 'UPT'), 'unsupported second predicate output')
                vs = [pred(x, p) for x in a[2:5]]
                if all(v is not None for v in vs):
                    idx = (int(vs[0]) << 2) | (int(vs[1]) << 1) | int(vs[2])
                    p[a[0]] = bool(int(a[5], 0) >> idx & 1)
                else:
                    p[a[0]] = None
                pc += 16
                continue
            elif o in ('SHF.L.U32', 'USHF.L.U32'):
                vs = [val(x, r) for x in a[1:4]]
                if all(v and v.exact_value is not None for v in vs) and vs[2].lo == 0:
                    result = V.exact(vs[0].lo << (vs[1].lo & 31))
                elif ext and all(v is not None for v in vs) and vs[1].exact_value is not None and vs[2].exact_value == 0:
                    n = vs[1].lo & 31
                    if vs[0].hi << n <= M32:
                        result = V(vs[0].lo << n, vs[0].hi << n, min(31, vs[0].tzeros() + n))
            elif o == 'SHF.R.U32.HI':
                vs = [val(x, r) for x in a[1:4]]
                require(vs[0] == V.exact(0), 'unsupported nonzero-low funnel shift')
                if all(v and v.exact_value is not None for v in vs):
                    result = V.exact(((vs[2].lo << 32) | vs[0].lo) >> (vs[1].lo & 31) >> 32)
            elif ext and o in ('SHF.R.S32.HI', 'USHF.R.S32.HI'):
                vs = [val(x, r) for x in a[1:4]]
                if all(v is not None and v.exact_value is not None for v in vs) and vs[1].lo < 32:
                    hi = vs[2].lo if vs[2].lo < 0x80000000 else vs[2].lo - (1 << 32)
                    result = V.exact(hi >> vs[1].lo)
            elif o == 'SEL':
                q = pred(a[3], p)
                x, y = val(a[1], r), val(a[2], r)
                result = x if q is True else y if q is False else x if x == y else None
            elif ext and o == 'SEL.64':
                q = pred(a[3], p)
                x, y = self.val64(a[1], r), self.val64(a[2], r)
                if q is True and x is not None:
                    pair_result = x
                elif q is False and y is not None:
                    pair_result = y
                elif x is not None and x == y:
                    pair_result = x
                pair_result = ('pair', pair_result)
            elif ext and o in ('IADD.64',):
                x, y = self.val64(a[1], r), self.val64(a[2], r)
                pair_result = ('pair', (x + y) & M64 if x is not None and y is not None else None)
            elif ext and o in ('IMAD.WIDE', 'IMAD.WIDE.U32'):
                x, y, z = val(a[1], r), val(a[2], r), self.val64(a[3], r)
                if x is not None and y is not None and z is not None and x.exact_value is not None and y.exact_value is not None:
                    if o == 'IMAD.WIDE':
                        xs = x.lo - (1 << 32) if x.lo >> 31 else x.lo
                        ys = y.lo - (1 << 32) if y.lo >> 31 else y.lo
                        pair_result = ('pair', (xs * ys + z) & M64)
                    else:
                        pair_result = ('pair', (x.lo * y.lo + z) & M64)
                else:
                    pair_result = ('pair', None)
            elif ext and o == 'IMAD.HI.U32':
                vs = [val(x, r) for x in a[1:4]]
                if all(v is not None and v.exact_value is not None for v in vs):
                    result = V.exact(((vs[0].lo * vs[1].lo) >> 32) + vs[2].lo)
            elif ext and o in ('VIMNMX.U32', 'VIMNMX.S32'):
                x, y = val(a[1], r), val(a[2], r)
                q = pred(a[3], p)
                if x is not None and y is not None and x.exact_value is not None and y.exact_value is not None and q is not None:
                    xv, yv = x.lo, y.lo
                    if o.endswith('S32'):
                        xv = xv - (1 << 32) if xv >> 31 else xv
                        yv = yv - (1 << 32) if yv >> 31 else yv
                    result = V.exact(min(xv, yv) if q else max(xv, yv))
            elif ext and o == 'PRMT':
                x, sel, y = val(a[1], r), val(a[2], r), val(a[3], r)
                if all(v is not None and v.exact_value is not None for v in (x, sel, y)):
                    src = [(x.lo >> (8 * i)) & 0xFF for i in range(4)] + [(y.lo >> (8 * i)) & 0xFF for i in range(4)]
                    out = 0
                    for i in range(4):
                        nib = (sel.lo >> (4 * i)) & 0xF
                        b = src[nib & 7]
                        if nib & 8:
                            b = 0xFF if b & 0x80 else 0x00
                        out |= b << (8 * i)
                    result = V.exact(out)
            elif ext and o in ('FADD', 'FADD.FTZ'):
                x, y = self.fval(a[1], r), self.fval(a[2], r)
                result = self.fres(x + y) if x is not None and y is not None else None
            elif ext and o in ('FMUL',):
                x, y = self.fval(a[1], r), self.fval(a[2], r)
                result = self.fres(x * y) if x is not None and y is not None else None
            elif ext and o == 'FFMA':
                x, y, z = self.fval(a[1], r), self.fval(a[2], r), self.fval(a[3], r)
                result = self.fres(x * y + z) if None not in (x, y, z) else None
            elif ext and o == 'I2FP.F32.S32':
                x = val(a[1], r)
                if x is not None and x.exact_value is not None:
                    result = self.fres(Fraction(x.lo - (1 << 32) if x.lo >> 31 else x.lo))
            elif ext and o == 'I2F.U32.RP':
                x = val(a[1], r)
                if x is not None and x.exact_value is not None:
                    result = self.fres(Fraction(x.lo), 'rp')
            elif ext and o == 'MUFU.RCP':
                x = self.fval(a[1], r)
                if x is not None and x != 0:
                    b = f32_round(1 / x)
                    if b is not None:
                        b += self.rcp_ulp_delta
                        result = V.exact(b)
            elif ext and o == 'F2I.FTZ.U32.TRUNC.NTZ':
                x = self.fval(a[1], r)
                if x is not None:
                    result = V.exact(0 if x <= 0 else min(M32, int(x)))
            elif ext and o == 'FSEL':
                x, y = val(a[1], r), val(a[2], r)
                q = pred(a[3], p)
                result = x if q is True else y if q is False else x if x == y else None
            elif ext and o.startswith('LDS'):
                addr = self.smem_addr(a[1], r)
                nw = 4 if '.128' in o else 2 if '.64' in o else 1
                if addr is not None:
                    self.shared_touched.update(addr + 4 * j for j in range(nw))
                words = [shm.get(addr + 4 * j) if addr is not None and not shm.get('poison') else None for j in range(nw)]
                pair_result = ('words', words)
            elif o.startswith('SHFL'):
                if a[0] not in ('PT', 'UPT'):
                    p[a[0]] = None
                req = ('shfl', pc, o, self.reg_plus(a[2].replace('.reuse', ''), 0, r), val(a[3], r), val(a[4], r))
                result = yield req
            elif o.startswith(('FSETP', 'P2R', 'R2P')):
                if a and a[0].startswith(('P', 'UP')):
                    p[a[0]] = None
            if dest and re.fullmatch(r'U?R\d+', dest):
                if pair_result is not None and pair_result[0] == 'words':
                    pref = 'UR' if dest.startswith('UR') else 'R'
                    idx = int(dest[len(pref):])
                    for j, w in enumerate(pair_result[1]):
                        r[pref + str(idx + j)] = w
                elif pair_result is not None:
                    pref = 'UR' if dest.startswith('UR') else 'R'
                    idx = int(dest[len(pref):])
                    x = pair_result[1]
                    r[dest] = V.exact(x & M32) if x is not None else None
                    r[pref + str(idx + 1)] = V.exact(x >> 32) if x is not None else None
                else:
                    r[dest] = result
                    width = 4 if '.128' in o else 2 if '.64' in o or '.WIDE' in o or o == 'CS2R' else 1
                    if not o.startswith(('LDC', 'LDCU')) or do is None:
                        pref = 'UR' if dest.startswith('UR') else 'R'
                        idx = int(dest[len(pref):])
                        for j in range(1, width):
                            r[pref + str(idx + j)] = V.exact(0) if o == 'CS2R' and a[1] == 'SRZ' else None
            elif dest and re.fullmatch(r'U?P\d+', dest):
                p[dest] = None
            if o.startswith(('IMAD', 'IADD3')) or (o.startswith('LEA') and len(a) < 5):
                for output in a[:2]:
                    if re.fullmatch(r'U?P\d+', output):
                        p[output] = None
            pc += 16
        raise Refusal('step limit')

    @staticmethod
    def reg_plus(name, j, r):
        """Value of register `name` advanced by j (RZ stays zero)."""
        if name in ('RZ', 'URZ'):
            return V.exact(0)
        m = re.fullmatch(r'(U?R)(\d+)', name)
        if not m:
            return None
        return r.get(m[1] + str(int(m[2]) + j))

    def smem_addr(self, tok, r):
        """Exact shared-memory byte address of an operand like [R5], [UR4], [R13+0x10]; else None."""
        m = re.fullmatch(r'\[(U?R\d+|RZ|URZ)?(?:\+(0x[0-9a-f]+))?\]', tok.replace('.reuse', ''))
        if not m:
            return None
        base = V.exact(0) if m[1] in (None, 'RZ', 'URZ') else r.get(m[1])
        if base is None or base.exact_value is None:
            return None
        return base.lo + (int(m[2], 0) if m[2] else 0)

    def run_block(self, sites, constants, coords_for_lane, threads, max_steps=100000):
        """Run all lanes of one block in lock-step phases so that shuffles exchange values within a warp and
        shared-memory words written before a barrier are visible after it. A warp whose unfinished lanes are
        not converged on the same shuffle, or a barrier not reached by all unfinished lanes at one pc, refuses.
        Returns per-lane results (events, visits, unknown_unpriced)."""
        shm = {}
        gens = [self._lane_gen(sites, constants, coords_for_lane(l), shm, max_steps) for l in range(threads)]
        st = ['new'] * threads            # new | ready | shfl | bar | done
        req = [None] * threads
        send = [None] * threads
        results = [None] * threads
        nwarps = (threads + 31) // 32
        while any(x != 'done' for x in st):
            progressed = False
            for l in range(threads):
                if st[l] in ('new', 'ready'):
                    try:
                        q = next(gens[l]) if st[l] == 'new' else gens[l].send(send[l])
                        req[l] = q
                        st[l] = q[0]
                    except StopIteration as stop:
                        results[l] = stop.value
                        st[l] = 'done'
                    progressed = True
            for w in range(nwarps):
                lanes = [l for l in range(32 * w, min(threads, 32 * w + 32)) if st[l] != 'done']
                at = [l for l in lanes if st[l] == 'shfl']
                if not at or len(at) != len(lanes):
                    continue
                q0 = req[at[0]]
                require(all(req[l][1] == q0[1] and req[l][2] == q0[2] for l in at), 'divergent shuffle in a warp at ' + hex(q0[1]))
                for l in at:
                    _, pcq, opq, src, dl, cl = req[l]
                    require(cl is not None and cl.exact_value == 0x1f, 'unsupported shuffle clamp at ' + hex(pcq))
                    if dl is None or dl.exact_value is None:
                        send[l] = None
                    else:
                        d = dl.lo
                        i = l - 32 * w
                        mode = opq.split('.')[1]
                        j = {'DOWN': i + d, 'UP': i - d, 'BFLY': i ^ d, 'IDX': d}.get(mode)
                        require(j is not None, 'unsupported shuffle mode ' + opq)
                        send[l] = src if (j < 0 or j > 31) else req[32 * w + j][3] if 32 * w + j < threads and st[32 * w + j] == 'shfl' else None
                for l in at:
                    st[l] = 'ready'
                progressed = True
            live = [l for l in range(threads) if st[l] != 'done']
            if live and all(st[l] == 'bar' for l in live):
                pcs = {req[l][1] for l in live}
                require(len(pcs) == 1, 'lanes at different barriers: ' + ','.join(hex(x) for x in sorted(pcs)))
                for l in live:
                    send[l] = None
                    st[l] = 'ready'
                progressed = True
            require(progressed, 'deadlock: lanes wait at mixed collectives')
        return results

    def imad(self, vs):
        """x*y+z with at most one non-exact factor (interval), modular."""
        x, y, z = vs
        if x is None or y is None or z is None:
            return None
        if x.exact_value is not None and y.exact_value is not None:
            return V.exact(x.lo * y.lo + z.lo) if z.exact_value is not None else self.affine([(1, z)], const=x.lo * y.lo)
        if x.exact_value is not None:
            return self.affine([(self._signed_const(x), y), (1, z)])
        if y.exact_value is not None:
            return self.affine([(self._signed_const(y), x), (1, z)])
        return None


EXT_KNOWN_OPS = frozenset(
    ('LDCU.128 LDC.U8 PRMT SHF.R.S32.HI SEL.64 ISETP.GE.S64.AND ISETP.GT.S64.AND ISETP.NE.S64.AND ISETP.EQ.S64.OR '
     'VIMNMX.U32 VIMNMX.S32 ULEA.HI LEA.HI LDG.E.128.CONSTANT FFMA.SAT FFMA.RM FFMA.RZ FFMA.RP FADD.FTZ FSETP.NEU.AND '
     'FSETP.NEU.FTZ.AND FSETP.GTU.FTZ.AND FSETP.GEU.AND FSETP.GT.AND FCHK MUFU.RSQ SHFL.IDX STS.64 '
     'ISETP.LT.U32.OR ISETP.NE.OR ISETP.GE.U32.OR ISETP.GT.U32.OR ISETP.EQ.OR ISETP.LT.OR ISETP.GE.OR '
     'I2FP.F32.S32 I2F.U32.RP F2I.FTZ.U32.TRUNC.NTZ IMAD.WIDE IMAD.WIDE.U32 IMAD.HI.U32 FMUL FADD FFMA FSEL '
     'LEA.HI.X IADD.64 CALL.REL.NOINC CALL.ABS.NOINC RET.REL.NODEC LEPC').split())
