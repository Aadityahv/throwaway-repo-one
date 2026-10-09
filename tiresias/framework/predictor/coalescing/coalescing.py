"""Static per-launch global-memory transaction counts (32 B sectors, 128 B lines).

CPU only. Reuses the retained-corpus loaders, hash verification, cubin ABI check
and launch binding of operator_derivation/derive.py (imported, never edited) and
the BRA.DIV convergence check of operator_derivation/collective.py.

Method: every thread of every analysed block is interpreted concretely with the
derive.py conventions (loaded data is unknown and poisons its outputs; unknown
control refuses the cell). Pointer parameters get concrete, 256-byte-aligned
base addresses (assumption, see COALESCING.md), so each lane's global address is
an exact integer when its offset depends only on threadIdx/blockIdx, scalar
arguments and loop counters. A warp request is the k-th dynamic arrival of the
32 lanes of a warp at a memory site; its sectors/lines are the distinct 32 B /
128 B granules touched by the active lanes. Block coordinates are exact per
block; large grids are sampled and checked for uniformity (see analyse_cell).
"""
from __future__ import annotations
import collections, hashlib, importlib.util, json, math, random, re, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPD = HERE.parents[1] / 'compile_evidence' / 'operator_derivation'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # needed before exec (dataclasses)
    spec.loader.exec_module(mod)
    return mod


D = _load('coalescing_operator_derive', OPD / 'derive.py')
if str(OPD) not in sys.path:
    sys.path.append(str(OPD))
COL = _load('coalescing_operator_collective', OPD / 'collective.py')  # imports derive as 'derive' from OPD

Refusal, require, parse, target, KNOWN_OPS = D.Refusal, D.require, D.parse, D.target, D.KNOWN_OPS
M32, M64 = 0xffffffff, (1 << 64) - 1
SECTOR, LINE = 32, 128
PTR_BASE0 = 0x7f00_0000_0000   # concrete pointer bases: 4 GiB apart, 256 B (in fact 4 GiB) aligned
PTR_STRIDE = 0x1_0000_0000

MEMRE = re.compile(r'desc\[(UR\d+)\]\[(R\d+)\.64(?:\+(-?0x[0-9a-f]+))?\]$')


PAIR_OPS = frozenset(['IADD.64', 'UIADD3.64', 'IMAD.WIDE', 'IMAD.WIDE.U32', 'UIMAD.WIDE', 'UIMAD.WIDE.U32', 'MOV.64', 'CS2R'])


def f32_from_bits(b): return struct.unpack('<f', struct.pack('<I', b & M32))[0]
def f32_bits(f): return struct.unpack('<I', struct.pack('<f', f))[0]


def i2f_u32_rp(x):
    """u32 -> binary32, rounding toward +infinity (I2F.U32.RP)."""
    b = f32_bits(float(x))
    if f32_from_bits(b) < x: b += 1
    return b


def mufu_rcp(b):
    f = f32_from_bits(b)
    if f == 0: return 0x7f800000
    return f32_bits(1.0 / f)


def f2i_u32_trunc(b):
    f = f32_from_bits(b)
    if f != f or f <= 0: return 0
    return M32 if f >= 2.0 ** 32 else int(f)


def half_bits(x):
    try: h = struct.pack('<e', float(x))
    except (ValueError, OverflowError): return None
    return struct.unpack('<H', h)[0] if struct.unpack('<e', h)[0] == float(x) else None


def s32(x): return x - (1 << 32) if x >> 31 else x


def mem_kind(op):
    """(kind, width_bytes) for global memory sites, else None. Unknown width suffix refuses."""
    base = op.split('.')[0]
    if base not in ('LDG', 'STG', 'LDGSTS'):
        return None
    mods = op.split('.')[1:]
    width = 4
    for t in mods:
        if t.isdigit():
            width = int(t) // 8
        elif t not in ('E', 'BYPASS'):
            raise Refusal('unsupported global memory modifier ' + op)
    require(width in (4, 8, 16), 'unsupported access width ' + op)
    return {'LDG': 'load', 'STG': 'store', 'LDGSTS': 'async_load'}[base], width


def parse_mem_operand(s, a):
    """Return (pair_register, immediate, src_pred) of the global operand."""
    if s.op.startswith('LDG') and not s.op.startswith('LDGSTS') and not s.op.startswith('LDGDEP'):
        txt, sp = a[1], None
    elif s.op.startswith('STG'):
        txt, sp = a[0], None
    else:  # LDGSTS [smem], desc[..][R.64+K], Ppred
        txt, sp = a[1], a[2] if len(a) > 2 else None
    m = MEMRE.match(txt)
    require(m is not None, 'unsupported global address operand ' + txt)
    return m[2], int(m[3], 0) if m[3] else 0, sp


class Mach:
    """Concrete register machine for one thread. Values are ints in [0,2^32) or None (unknown)."""

    def __init__(self, consts, coords):
        self.r, self.p, self.c, self.coords = {}, {}, consts, coords

    # ---- operands
    def rd(self, t):
        t = t.replace('.reuse', '')
        if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', t):
            return int(t, 0) & M32
        neg = t.startswith('-')
        t = t[1:] if neg else t
        if t in ('RZ', 'URZ'): v = 0
        elif re.fullmatch(r'U?R\d+', t): v = self.r.get(t)
        else:
            m = re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]', t)
            v = self.c.get(int(m[1], 0)) if m else None
        return None if v is None else ((-v) & M32 if neg else v)

    def rd64(self, t):
        t = t.replace('.reuse', '')
        if t in ('RZ', 'URZ'): return 0
        if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', t): return int(t, 0) & M64
        m = re.fullmatch(r'(U?R)(\d+)', t)
        if not m: return None
        lo, hi = self.r.get(t), self.r.get(m[1] + str(int(m[2]) + 1))
        return None if lo is None or hi is None else (hi << 32) | lo

    def pr(self, t):
        inv = t.startswith('!'); t = t.lstrip('!')
        v = True if t in ('PT', 'UPT') else self.p.get(t)
        return (not v) if inv and v is not None else v

    def set(self, dest, v, width=1):
        m = re.fullmatch(r'(U?R)(\d+)', dest or '')
        if not m: return
        self.r[dest] = v
        for j in range(1, width): self.r[m[1] + str(int(m[2]) + j)] = None

    def set64(self, dest, v):
        m = re.fullmatch(r'(U?R)(\d+)', dest or '')
        if not m: return
        self.r[dest] = None if v is None else v & M32
        self.r[m[1] + str(int(m[2]) + 1)] = None if v is None else (v >> 32) & M32


def icmp(x, y, rel, unsigned):
    if x is None or y is None: return None
    if not unsigned: x, y = s32(x), s32(y)
    return {'EQ': x == y, 'NE': x != y, 'GE': x >= y, 'GT': x > y, 'LT': x < y, 'LE': x <= y}[rel]


def run_thread(sites, consts, coords, max_steps=200000, probe=None):
    """Interpret one thread. Returns (target_events, mem_log, arrivals, reach).

    mem_log: list of (pc, dynamic_index, guard(True/False/None), src_pred(True/False/None), address|None).
    """
    m = Mach(consts, coords); by = {s.pc: s for s in sites}
    events, reach, mem_log, arrivals = collections.Counter(), collections.Counter(), [], collections.Counter()
    pc = 0
    for _ in range(max_steps):
        require(pc in by, 'control outside function'); s = by[pc]; a = s.a; o = s.op; t = target(o)
        require(o in KNOWN_OPS, 'unsupported opcode ' + o)
        if o in ('IADD3', 'UIADD3'): require(a[1:3] in (('PT', 'PT'), ('UPT', 'UPT')), 'unsupported add carry outputs')
        if o.startswith(('ISETP.', 'UISETP.')) or o == 'PLOP3.LUT': require(a[1] in ('PT', 'UPT'), 'unsupported second predicate output')
        if o in ('LOP3.LUT', 'ULOP3.LUT'):
            require(a[-1] in ('!PT', '!UPT'), 'unsupported LOP3 predicate combiner')
            if a[0].startswith(('P', 'UP')): require(a[5] in ('0xc0', '0xfc'), 'unsupported LOP3 predicate LUT')
        do = m.pr(s.pred) if s.pred else True
        if t: require(do is not None, 'unknown target predicate at ' + hex(pc))
        if o in ('EXIT', 'BRA', 'BRA.U', 'BRA.DIV'): require(do is not None, 'unknown control at %s %s' % (hex(pc), s.pred))
        mk = mem_kind(o)
        if mk:
            reg, imm, sp = parse_mem_operand(s, a)
            base = m.rd64(reg)
            addr = None if base is None else (base + imm) & M64
            srcp = m.pr(sp) if sp else True
            mem_log.append((pc, reach[pc], do, srcp, addr))
            if probe is not None: probe.setdefault(pc, []).append((do, reg, dict(m.r)))
        reach[pc] += 1
        if do is False: pc += 16; continue
        if t: events[(pc, t[0], t[1])] += 1
        if o == 'EXIT': return events, mem_log, arrivals, reach
        if o in ('BRA', 'BRA.U'):
            take = True if o == 'BRA' else m.pr(a[0]); require(take is not None, 'unknown uniform branch')
            pc = int(a[-1], 0) if take else pc + 16; continue
        if o == 'BRA.DIV':   # same convergence treatment as collective.run_lane_div
            mask = m.rd(a[0]); require(mask is not None and len(a) == 2, 'unknown BRA.DIV mask at ' + hex(pc))
            arrivals[(pc, mask)] += 1; pc += 16; continue
        require(not o.startswith(('CALL', 'RET', 'JMP', 'BRX', 'WARPSYNC', 'BRA.')), 'unsupported control ' + o)
        if o.startswith(('STG', 'STS', 'BAR', 'BSSY', 'BSYNC', 'DEPBAR', 'LDGDEPBAR', 'LDGSTS', 'NOP')): pc += 16; continue
        dest = a[1] if o.startswith('SHFL') or (o == 'LOP3.LUT' and a[0].startswith(('P', 'UP'))) else a[0] if a else None
        width = 4 if '.128' in o else 2 if ('.64' in o or '.WIDE' in o or o == 'CS2R') else 1
        res, res64 = None, None; known = do is True
        if not known:
            if o == 'LOP3.LUT' and a[0].startswith(('P', 'UP')): m.p[a[0]] = None
        elif o.startswith(('LDC', 'LDCU')):
            mm = re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]', a[1]); off = int(mm[1], 0) if mm else None
            if '.64' in o:
                lo, hi = (m.c.get(off), m.c.get(off + 4)) if off is not None else (None, None)
                res64 = None if lo is None or hi is None else (hi << 32) | lo
            else: res = m.c.get(off) if off is not None else None
        elif o in ('S2R', 'S2UR'):
            v = coords.get(a[1]); res = None if v is None else v & M32
        elif o == 'R2UR': res = m.rd(a[1])
        elif o == 'CS2R' and a[1] == 'SRZ': res64 = 0
        elif o == 'HFMA2' and a[1:3] == ('-RZ', 'RZ') and len(a) == 5:
            hi, lo = half_bits(a[3]), half_bits(a[4])      # -RZ*RZ + {hi,lo} half immediates: a constant register
            if hi is not None and lo is not None: res = (hi << 16) | lo
        elif o == 'I2F.U32.RP':
            x = m.rd(a[1]); res = None if x is None else i2f_u32_rp(x)
        elif o == 'MUFU.RCP':
            x = m.rd(a[1]); res = None if x is None else mufu_rcp(x)
        elif o == 'F2I.FTZ.U32.TRUNC.NTZ':
            x = m.rd(a[1]); res = None if x is None else f2i_u32_trunc(x)
        elif o == 'P2R' and a[1:3] == ('PR', 'RZ'):
            bits = [i for i in range(7) if int(a[3], 0) >> i & 1]
            if all(m.p.get('P' + str(i)) is not None for i in bits): res = sum(int(m.p['P' + str(i)]) << i for i in bits)
        elif o in ('MOV.64',): res64 = m.rd64(a[1])
        elif o.startswith(('MOV', 'UMOV')): res = m.rd(a[1])
        elif o in ('IMAD', 'UIMAD', 'IMAD.SHL.U32'):
            vs = [m.rd(x) for x in a[1:4]]; res = None if None in vs else (vs[0] * vs[1] + vs[2]) & M32
        elif o in ('IMAD.WIDE', 'IMAD.WIDE.U32', 'UIMAD.WIDE', 'UIMAD.WIDE.U32'):
            x, y, z = m.rd(a[1]), m.rd(a[2]), m.rd64(a[3])
            if None not in (x, y, z):
                if o.endswith('.U32'): prod = x * y
                else: prod = s32(x) * s32(y)
                res64 = (prod + z) & M64
        elif o in ('IMAD.HI', 'IMAD.HI.U32'):
            # high word of (Ra*Rb + 64-bit pair {Rc+1:Rc}); the pair form is what makes the compiler's
            # reciprocal-refinement idiom exact (checked against divmod in test_coalescing.py)
            x, y, z = m.rd(a[1]), m.rd(a[2]), m.rd64(a[3])
            if None not in (x, y, z):
                prod = x * y if o.endswith('.U32') else s32(x) * s32(y)
                res = (((prod + z) & M64) >> 32) & M32
        elif o == 'IADD':
            vs = [m.rd(x) for x in a[1:3]]; res = None if None in vs else sum(vs) & M32
        elif o in ('IADD3', 'UIADD3'):
            vs = [m.rd(x) for x in a[3:6]]; res = None if None in vs else sum(vs) & M32
        elif o in ('IADD.64', 'UIADD3.64'):
            srcs = a[1:3] if o == 'IADD.64' else a[3:6]
            if o == 'UIADD3.64': require(a[1:3] == ('UPT', 'UPT'), 'unsupported add carry outputs')
            vs = [m.rd64(x) for x in srcs]; res64 = None if None in vs else sum(vs) & M64
        elif o in ('LEA', 'ULEA') and len(a) == 4:
            vs = [m.rd(a[1]), m.rd(a[2]), m.rd(a[3])]
            res = None if None in vs else (((vs[0] << (vs[2] & 31)) & M32) + vs[1]) & M32
        elif o in ('LEA', 'ULEA') and len(a) == 5 and a[1].startswith(('P', 'UP')):
            vs = [m.rd(a[2]), m.rd(a[3]), m.rd(a[4])]
            if None in vs: m.p[a[1]] = None
            else:
                tot = ((vs[0] << (vs[2] & 31)) & M32) + vs[1]; res = tot & M32; m.p[a[1]] = bool(tot >> 32)
        elif o == 'LEA.HI.X' and len(a) == 6:
            x, y, z, sh, cy = m.rd(a[1]), m.rd(a[2]), m.rd(a[3]), m.rd(a[4]), m.pr(a[5])
            if None not in (x, y, z, sh, cy):
                res = ((((z << 32) | x) << (sh & 63) >> 32) + y + int(cy)) & M32
        elif o.startswith(('ISETP.', 'UISETP.')):
            rel = o.split('.')[1]; c = icmp(m.rd(a[2]), m.rd(a[3]), rel, 'U32' in o); q = m.pr(a[4]); comb = o.split('.')[-1]
            out = ((False if c is False or q is False else True if c is True and q is True else None) if comb == 'AND'
                   else (True if c is True or q is True else False if c is False and q is False else None) if comb == 'OR' else None)
            m.p[a[0]] = out; pc += 16; continue
        elif o in ('LOP3.LUT', 'ULOP3.LUT'):
            has_pred = a[0].startswith(('P', 'UP')); st = 2 if has_pred else 1
            vs = [m.rd(x) for x in a[st:st + 3]]; lut = int(a[st + 3], 0)
            if None not in vs:
                x, y, z = vs; n = 0
                for bit in range(32): n |= ((lut >> (((x >> bit & 1) << 2) | ((y >> bit & 1) << 1) | (z >> bit & 1))) & 1) << bit
                res = n
            if has_pred: m.p[a[0]] = bool(res) if res is not None else None; dest = a[1]
        elif o == 'PLOP3.LUT':
            vs = [m.pr(x) for x in a[2:5]]
            m.p[a[0]] = None if None in vs else bool(int(a[5], 0) >> ((int(vs[0]) << 2) | (int(vs[1]) << 1) | int(vs[2])) & 1)
            pc += 16; continue
        elif o in ('SHF.L.U32', 'USHF.L.U32'):
            vs = [m.rd(x) for x in a[1:4]]
            if None not in vs and vs[2] == 0: res = (vs[0] << (vs[1] & 31)) & M32
        elif o == 'SHF.L.U64.HI':
            vs = [m.rd(x) for x in a[1:4]]
            if None not in vs: res = ((((vs[2] << 32) | vs[0]) << (vs[1] & 63)) >> 32) & M32
        elif o == 'SHF.R.U32.HI':
            vs = [m.rd(x) for x in a[1:4]]; require(vs[0] == 0, 'unsupported nonzero-low funnel shift')
            if None not in vs: res = vs[2] >> (vs[1] & 31)
        elif o == 'USHF.R.S32.HI':
            vs = [m.rd(x) for x in a[1:4]]; require(vs[0] == 0, 'unsupported nonzero-low funnel shift')
            if None not in vs: res = (s32(vs[2]) >> (vs[1] & 31)) & M32
        elif o == 'SEL':
            q = m.pr(a[3]); x, y = m.rd(a[1]), m.rd(a[2]); res = x if q is True else y if q is False else x if x == y else None
        elif o.startswith('SHFL'):
            if a[0] not in ('PT', 'UPT'): m.p[a[0]] = None
        elif o.startswith(('FSETP', 'P2R', 'R2P')):
            if a and a[0].startswith(('P', 'UP')): m.p[a[0]] = None
        # ---- write back
        if dest and re.fullmatch(r'U?R\d+', dest):
            if known and (o in PAIR_OPS or (o.startswith(('LDC', 'LDCU')) and '.64' in o)): m.set64(dest, res64)
            else: m.set(dest, res, width)
        elif dest and re.fullmatch(r'U?P\d+', dest): m.p[dest] = None
        # carry/predicate outputs of address arithmetic are never kept stale
        if o.startswith(('IMAD', 'IADD3')) or (o.startswith('LEA') and len(a) < 5):
            for out in a[:2]:
                if re.fullmatch(r'U?P\d+', out): m.p[out] = None
        pc += 16
    raise Refusal('step limit')


# ----------------------------------------------------------------------------- warp requests
def granules(addr, size, g):
    return range(addr // g, (addr + size - 1) // g + 1)


def block_requests(sites, consts, ctaid, bdim, threads):
    """Interpret every thread of one block; return per-site request lists and target-event counts.

    Returns (reqs, events, info). reqs[pc] = list of request dicts, one per (warp, dynamic arrival)
    with at least one active lane; each has either exact (sectors, lines) or None when unknowable.
    """
    bx = bdim[0]; logs, evt, arrs, reaches = [], collections.Counter(), [], []
    for lane in range(threads):
        c = {'SR_CTAID.X': ctaid[0], 'SR_CTAID.Y': ctaid[1], 'SR_CTAID.Z': 0, 'SR_CgaCtaId': 0,
             'SR_TID.X': lane % bx, 'SR_TID.Y': (lane // bx) % bdim[1], 'SR_TID.Z': lane // (bx * bdim[1]), 'SR_LANEID': lane % 32}
        e, log, arr, reach = run_thread(sites, consts, c)
        evt.update(e); logs.append(log); arrs.append(arr); reaches.append(reach)
    COL.check_warp_convergence([{(k[0], k[1]): v for k, v in a.items()} for a in arrs], threads)
    memsites = {s.pc: s for s in sites if mem_kind(s.op)}
    reqs = collections.defaultdict(list); uneven = collections.Counter()
    for w in range(math.ceil(threads / 32)):
        lanes = range(w * 32, min(threads, (w + 1) * 32)); per = collections.defaultdict(lambda: collections.defaultdict(list))
        for l in lanes:
            for (pc, k, guard, srcp, addr) in logs[l]: per[pc][k].append((guard, srcp, addr))
        for pc in per:
            if len({reaches[l][pc] for l in lanes}) > 1: uneven[pc] += 1
            kind, width = mem_kind(memsites[pc].op)
            for k, ents in sorted(per[pc].items()):
                # lanes that never reached arrival k are treated as inactive (lockstep approximation)
                def act(e, ignore_src):
                    g, sp, _ = e
                    if g is False: return False
                    if g is None: return None
                    if kind == 'async_load' and not ignore_src: return sp
                    return True
                for ignore_src in ((False, True) if kind == 'async_load' else (False,)):
                    st = [act(e, ignore_src) for e in ents]
                    if any(x is True for x in st) or any(x is None for x in st):
                        pass
                    else: continue            # warp-wide predicated off: no request
                    if any(x is None for x in st):
                        r = {'status': 'data_dependent_predicate', 'active': None, 'bytes': None, 'sectors': None, 'lines': None}
                    else:
                        addrs = [e[2] for e, x in zip(ents, st) if x]
                        n = len(addrs); r = {'active': n, 'bytes': n * width}
                        if any(x is None for x in addrs):
                            r.update(status='data_dependent_address', sectors=None, lines=None)
                        else:
                            r.update(status='known', sectors=len({g for x in addrs for g in granules(x, width, SECTOR)}),
                                     lines=len({g for x in addrs for g in granules(x, width, LINE)}))
                    r['ignore_src_pred'] = ignore_src
                    reqs[pc].append(r)
    return dict(reqs), evt, {'uneven_reach_warps': dict(uneven)}


def site_signature(reqs):
    """Aggregate request lists per site and (src-pred interpretation) into comparable tuples."""
    out = {}
    for pc, lst in reqs.items():
        for ig in {r['ignore_src_pred'] for r in lst}:
            rr = [r for r in lst if r['ignore_src_pred'] == ig]
            stat = {r['status'] for r in rr}
            hist = collections.Counter(r['sectors'] for r in rr)
            known_active = all(r['active'] is not None for r in rr)
            out[(pc, ig)] = (len(rr), sum(r['active'] for r in rr) if known_active else None, sum(r['bytes'] for r in rr) if known_active else None,
                             sum(r['sectors'] for r in rr) if all(r['sectors'] is not None for r in rr) else None,
                             sum(r['lines'] for r in rr) if all(r['lines'] is not None for r in rr) else None,
                             tuple(sorted(stat)), tuple(sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
    return out


def sample_blocks(grid, n_random=56, seed=20261001, exhaustive_limit=96):
    gx, gy = grid[0], grid[1]; total = gx * gy
    if total <= exhaustive_limit: return list(range(total)), True
    s = {0, 1, 2, 3, total - 2, total - 1, gx - 1, total // 2, gx, gx + 1, min(total - 1, gx * (gy // 2))}
    rnd = random.Random(seed)
    while len(s) < n_random + 11: s.add(rnd.randrange(total))
    return sorted(x for x in s if 0 <= x < total), False


MAX_REFINE_BLOCKS = 600     # cap on extra blocks interpreted while locating signature change points


def _key(sig, evt): return json.dumps([sorted(map(str, sig.items())), sorted(map(str, evt.items()))])


def analyse_blocks(sites, consts, grid, bdim, threads, **kw):
    """Interpret sampled blocks, then bisect between neighbouring samples whose signatures differ.

    Assumption (documented): the per-block signature is piecewise constant in the linear block index
    (b = x + y*gridDim.x); two samples with equal signatures are taken to bound a constant run.
    Returns (blocks, exhaustive, total, sigs, evts, info) for every interpreted block.
    """
    gx, gy = grid[0], grid[1]; total = gx * gy
    blocks, exhaustive = sample_blocks(grid, **kw)
    sigs, evts, info = {}, {}, {}

    def ev(b):
        if b not in sigs:
            reqs, evt, inf = block_requests(sites, consts, (b % gx, b // gx), bdim, threads)
            sigs[b] = site_signature(reqs); evts[b] = evt; info[b] = inf
        return _key(sigs[b], evts[b])
    for b in blocks: ev(b)
    extra = {'refined': 0, 'capped': False}
    if not exhaustive:
        stack = [(x, y) for x, y in zip(sorted(sigs), sorted(sigs)[1:]) if ev(x) != ev(y)]
        while stack:
            lo, hi = stack.pop()
            if hi - lo <= 1: continue
            if len(sigs) - len(blocks) >= MAX_REFINE_BLOCKS: extra['capped'] = True; break
            mid = (lo + hi) // 2; ev(mid); extra['refined'] += 1
            for x, y in ((lo, mid), (mid, hi)):
                if ev(x) != ev(y): stack.append((x, y))
    for b in sigs: info[b]['_refine'] = extra
    return sorted(sigs), exhaustive, total, sigs, evts, info


def combine(blocks, exhaustive, total, sigs, evts):
    """Weights every interpreted block by the run of blocks it stands for and sums the signatures.

    Block b_i stands for [b_i, b_{i+1}) when the next interpreted block has an identical signature
    (constant run), otherwise only for itself (adjacent change point, or the last block).
    If some adjacent unequal pair is still separated by unseen blocks (refinement cap), totals are bounded.
    """
    blocks = sorted(blocks); keys = {b: _key(sigs[b], evts[b]) for b in blocks}
    weights, unresolved = {}, 0
    for i, b in enumerate(blocks):
        nxt = blocks[i + 1] if i + 1 < len(blocks) else total
        if i + 1 == len(blocks): weights[b] = nxt - b
        elif keys[b] == keys[nxt]: weights[b] = nxt - b
        else:
            weights[b] = 1
            if nxt - b > 1: unresolved += nxt - b - 1
    classes = collections.Counter(keys.values())
    cov = {'blocks_total': total, 'blocks_interpreted': len(blocks), 'exhaustive': exhaustive, 'distinct_block_signatures': len(classes),
           'change_points_located_by_bisection': sum(1 for i, b in enumerate(blocks[:-1]) if keys[b] != keys[blocks[i + 1]]),
           'blocks_in_unresolved_gaps': unresolved}
    if exhaustive: cov['method'] = 'every block interpreted'
    elif len(classes) == 1: cov['method'] = 'all %d interpreted blocks have identical per-site signatures; scaled to the grid' % len(blocks)
    else: cov['method'] = 'signature changes at %d located boundary(ies); each constant run scaled by its length' % cov['change_points_located_by_bisection']
    if unresolved:
        cov['method'] += '; UNRESOLVED gaps -> bounds'
        cov['bounded'] = True
        return _sum_bounded(blocks, sigs, evts, total), cov
    return _sum(sigs, evts, weights), cov


def _new():
    return {'requests': 0, 'active': 0, 'bytes': 0, 'sectors': 0, 'lines': 0, 'status': set(), 'hist': collections.Counter()}


def _sum(sigs, evts, weights):
    agg, eagg = {}, collections.Counter()
    for b, w in weights.items():
        for k, v in sigs[b].items():
            n, act, by, sec, lin, stat, hist = v; cur = agg.setdefault(k, _new())
            cur['requests'] += n * w
            for f, x in (('active', act), ('bytes', by), ('sectors', sec), ('lines', lin)):
                cur[f] = None if cur[f] is None or x is None else cur[f] + x * w
            cur['status'] |= set(stat)
            for sc, cnt in hist: cur['hist'][sc] += cnt * w
        for k, v in evts[b].items(): eagg[k] += v * w
    return agg, eagg


def _sum_bounded(blocks, sigs, evts, total):
    agg = {}
    for k in set().union(*[set(sigs[b]) for b in blocks]):
        vs = [sigs[b].get(k) for b in blocks]; vs = [v if v else (0, 0, 0, 0, 0, (), ()) for v in vs]
        cur = agg.setdefault(k, _new()); cur['status'].add('block_nonuniform_bounded')
        for f, i in (('requests', 0), ('active', 1), ('bytes', 2), ('sectors', 3), ('lines', 4)):
            xs = [v[i] for v in vs]; cur[f] = None if None in xs else (min(xs) * total, max(xs) * total)
    return (agg, None)


# ----------------------------------------------------------------------------- cell level
def pointer_consts(meta, misalign=0):
    """Concrete aligned base for every 8-byte kernel parameter, as constant-bank words."""
    cons, bases = {}, {}
    flags = meta['opaque_parameter_flags']
    for ordinal, off in meta['ordinal_offsets']:
        size = (int(flags[str(ordinal)], 16) >> 18) & 0xff
        if size == 8:
            base = PTR_BASE0 + ordinal * PTR_STRIDE + misalign
            cons[meta['parameter_base'] + off], cons[meta['parameter_base'] + off + 4] = base & M32, base >> 32
            bases[ordinal] = base
    return cons, bases


def _class_totals(events):
    out = collections.Counter()
    for (_, cls, _w), n in events.items(): out[cls] += n
    return dict(out)


def _fold(agg, sites, sectors_ref):
    """Per-site rows and per-direction totals from the combined aggregate."""
    op = {s.pc: s.op for s in sites}; rows = []
    for (pc, ig), v in sorted(agg.items()):
        kind, width = mem_kind(op[pc])
        rows.append({'pc': hex(pc), 'op': op[pc], 'kind': kind, 'width_bytes': width,
                     'trailing_predicate_ignored_variant': bool(ig) if kind == 'async_load' else None,
                     'warp_requests': v['requests'], 'active_lane_accesses': v['active'], 'logical_bytes': v['bytes'],
                     'sectors': v['sectors'], 'lines': v['lines'], 'status': sorted(v['status']),
                     'sectors_per_request_histogram': {str(k): n for k, n in sorted(v['hist'].items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}})
    return rows


def _num(x): return x if isinstance(x, int) else None


def totals(rows, kinds, variant_ignore):
    """Direction totals over sites whose kind is in kinds; async sites use the chosen predicate interpretation."""
    sel = [r for r in rows if r['kind'] in kinds and (r['kind'] != 'async_load' or r['trailing_predicate_ignored_variant'] == variant_ignore)]
    out = {'sites': len(sel)}
    unknown = [r['pc'] for r in sel if r['status'] != ['known']]
    for f in ('warp_requests', 'active_lane_accesses', 'logical_bytes', 'sectors', 'lines'):
        vals = [r[f] for r in sel]
        out[f] = sum(vals) if all(isinstance(v, int) for v in vals) else None
        if any(isinstance(v, tuple) for v in vals): out[f] = None
    out['sector_efficiency'] = (out['logical_bytes'] / (out['sectors'] * SECTOR)) if out['sectors'] and out['logical_bytes'] is not None else None
    out['line_efficiency'] = (out['logical_bytes'] / (out['lines'] * LINE)) if out['lines'] and out['logical_bytes'] is not None else None
    out['sectors_per_request'] = out['sectors'] / out['warp_requests'] if out['sectors'] is not None and out['warp_requests'] else None
    out['unknown_sites'] = unknown
    out['status'] = 'full_static_sector_count' if not unknown and out['sectors'] is not None else ('no_sites' if not sel else 'partial_or_data_dependent')
    out['sectors_known_sites_only'] = sum(r['sectors'] for r in sel if isinstance(r['sectors'], int))
    return out


def analyse_cell(corpus, row, root, misalign=0, **kw):
    proof = D.verify(root, row)
    meta = D.parameter_layout(root, row)
    consts, coords, threads, blocks, launch = D.binding(corpus, row, root)
    cb = {k: v.exact_value for k, v in consts.items()}
    grid_, bdim_ = launch['grid'], launch['block']
    pc_, bases = pointer_consts(meta, misalign)
    require(not (set(pc_) & set(cb)), 'pointer/scalar parameter offset overlap')
    cb.update(pc_)
    for off, v in zip((0x360, 0x364, 0x368), (bdim_[0], bdim_[1], 1)): cb.setdefault(off, v)
    for off, v in zip((0x370, 0x374, 0x378), (grid_[0], grid_[1], 1)): cb.setdefault(off, v)
    sites = parse((root / row['disassembly_path']).read_text())
    grid, bdim = launch['grid'], launch['block']
    blocks_l, exh, total, sigs, evts, info = analyse_blocks(sites, cb, grid, bdim, threads, **kw)
    agg, eagg, cov = None, None, None
    (agg, eagg), cov = combine(blocks_l, exh, total, sigs, evts)
    rows = _fold(agg, sites, None)
    uneven = sorted({hex(pc) for b in info.values() for pc in b['uneven_reach_warps']})
    res = {'status': 'ok', 'launch': {'grid': grid, 'block': bdim, 'threads_per_block': threads, 'blocks': total},
           'pointer_parameter_ordinals': sorted(bases), 'block_coverage': cov, 'sites': rows,
           'sites_with_unequal_per_lane_arrival_counts': uneven,
           'loads': {'ldg_only': totals(rows, ('load',), False), 'ldg_plus_ldgsts': totals(rows, ('load', 'async_load'), False),
                     'ldg_plus_ldgsts_if_trailing_predicate_ignored': totals(rows, ('load', 'async_load'), True)},
           'stores': totals(rows, ('store',), False),
           'target_class_events': _class_totals(eagg) if eagg is not None else None,
           'evidence': {'hashes_verified_by': 'operator_derivation/derive.py verify()', 'retention_manifest_sha256': proof['retention_manifest_sha256'],
                        'sass_sha256': proof['sass_sha256'], 'cubin_sha256': proof['cubin_sha256'], 'source_sha256': proof['source_sha256']}}
    return res


def consistency_with_derivation(row_result, oid, cell):
    """Compare per-class thread-instruction counts with the committed operator_derivation JSON (read-only artifacts)."""
    ref = _DERIVED.get((oid, cell))
    if ref is None or row_result.get('target_class_events') is None: return {'compared': False}
    names = {'shuffle': 'shuffle', 'shared_load': 'shared_load', 'shared_store': 'shared_store', 'exponential': 'exponential', 'fma': 'fma', 'barrier': 'barrier'}
    mine = row_result['target_class_events']; ok = True; diffs = {}
    for k, nm in names.items():
        a, b = mine.get(k, 0), ref[nm]['predicate_true_thread_instruction']
        if a != b: ok = False; diffs[k] = [a, b]
    return {'compared': True, 'identical': ok, 'differences_mine_vs_derivation': diffs}


_DERIVED = {}
def _load_derived():
    for f in ('candidate_counts.json', 'collective_counts.json'):
        for r in json.loads((OPD / f).read_text())['rows']:
            if r['status'] == 'candidate_class_counts': _DERIVED[(r['operator_id'], r['cell'])] = r['classes']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out=HERE / 'static_sectors.json', misalign=0):
    _load_derived(); rows = []
    for corpus, root in D.CORPORA.items():
        for row in json.loads((root / 'retention_manifest.json').read_text())['rows']:
            cid = 'blackwell/%s/%s' % (row['operator_id'], row['cell'])
            try:
                r = analyse_cell(corpus, row, root, misalign=misalign); r['interpreter_consistency'] = consistency_with_derivation(r, row['operator_id'], row['cell'])
            except Refusal as e:
                r = {'status': 'refused', 'reason': str(e)}
            rows.append({'cell_id': cid, 'operator_id': row['operator_id'], 'cell': row['cell'], 'corpus': corpus, 'description_of_scope': 'static candidate; not profiler-validated', **r})
            print(cid, r['status'], r.get('reason', ''), flush=True)
    doc = {'schema': 'static_global_memory_sectors/1', 'coalescing_py_sha256': sha(__file__), 'derive_py_sha256': sha(OPD / 'derive.py'),
           'collective_py_sha256': sha(OPD / 'collective.py'),
           'assumptions': {'pointer_base_alignment_bytes': 256, 'pointer_base_misalign_bytes_applied': misalign, 'sector_bytes': SECTOR, 'line_bytes': LINE,
                           'request_definition': 'k-th dynamic arrival of the warp lanes at a site; lanes that did not arrive are inactive; requests with no active lane are not counted',
                           'ldgsts_trailing_predicate': 'treated as source-valid predicate (false: no global read); the ignored-predicate variant is reported as an upper alternative',
                           'atomics': 'no global atomic/reduction opcode occurs in any of the 93 retained isolated SASS bodies'},
           'frozen_before_profiler_comparison': True, 'total_cells': len(rows), 'rows': rows}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print(collections.Counter(r['status'] for r in rows))


if __name__ == '__main__':
    main()
