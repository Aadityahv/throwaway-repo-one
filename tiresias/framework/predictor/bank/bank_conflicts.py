"""Static shared-memory bank-conflict analysis (CPU only; no profiler, runtime or energy value is read).

Reuses, by import and without editing, the retained interpreters and the barrier-phase machinery:

* retained CUDA / Triton cells: phases.TRACE_THREAD (the frozen coalescing interpreter with a pre-instruction observer),
  phases.Observer (barrier phases, k-th dynamic arrival of a warp at a site = one warp request, global sectors/lines);
* PyTorch kernels: pytorch_features/pt_interp.py through phases.Observer.pytorch.

`SharedObserver` subclasses phases.Observer. It records, for every executed shared-memory instruction, the lane's guard and
the lane's shared byte address (evaluated by the same register machines), keyed exactly like the frozen warp-instruction
count, then folds them into per-phase conflict statistics with the SAME block sampling and extrapolation as phases.py
(first, middle, last block interpreted; sampled blocks must agree; scaled by the grid).

Conflict model (see BANK.md):
  bank = (byte_address / 4) % 32; a request is split into groups (32-bit and narrower: one group of 32 lanes; 64-bit: two
  half-warps of 16 lanes; 128-bit and each LDSM matrix: quarter-warps of 8 lanes); the degree of a group is the largest
  number of DISTINCT 32-bit words that map to one bank (the same word is one broadcast access). Atomics (ATOMS) count every
  lane access, so same-word lanes serialise.
  request wavefronts   W = sum over non-empty groups of the group degree
  shared_cost_cycles   = max(2.0, W) per request   (primary; equals the measured max(2.0, degree) for 32-bit accesses)
Alternatives reported next to it: per-group floor sum(max(2.0, degree_g)) and its lane-scaled form sum(max(2.0, d_g)*lanes_g/32).

Regression gate: the instrumented run must reproduce the frozen phase table (issue counts, read/write sectors, lines, bytes,
chain depths, barrier sequence) byte-exactly, and the shared request counts must equal the frozen warp-instruction counts of
the shared opcodes. A cell that fails either is refused with the reason.
"""
from __future__ import annotations

import collections
import functools
import json
import math
import multiprocessing
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent
sys.path.insert(0, str(STATIC))
sys.path.insert(0, str(STATIC / 'coalescing'))
sys.path.insert(0, str(STATIC / 'fresh'))
import phases as PH  # noqa: E402  (imported, never edited)
import divider_fix  # noqa: E402  (imported, never edited)

C, X, A = PH.C, PH.X, PH.A
P = A.P
D = C.D
M32 = 0xFFFFFFFF
FLOOR = 2.0
NBANKS = 32
SCHEMA = 'static_shared_bank_conflicts/1'

# ------------------------------------------------------------------------------------------------ opcode classification
SHARED_BASE = ('LDS', 'STS', 'ATOMS', 'LDSM', 'STSM')
ASYNC_BASE = ('LDGSTS',)
WIDTH_MODS = {'U8': 1, 'S8': 1, 'U16': 2, 'S16': 2, '32': 4, '64': 8, '128': 16, 'U': 4, 'S': 4}
ATOMS_MODS = {'ADD', 'MIN', 'MAX', 'AND', 'OR', 'XOR', 'EXCH', 'CAS', 'POPC', 'INC', 'DEC', 'S32', 'U32', 'S64', 'U64',
              'CAST', 'SPIN', 'F32', 'FTZ', 'RN', 'E'}


@functools.lru_cache(maxsize=None)
def shared_info(op):
    """None, or dict(kind, width_bytes, atomic, lanes_used, group_lanes) for an opcode that touches shared memory.

    kind: load | store | atomic | ldsm | async_write (LDGSTS destination). Unknown modifiers refuse, never guess."""
    base, *mods = op.split('.')
    if base in ASYNC_BASE:
        width = 4
        for t in mods:
            if t.isdigit():
                width = int(t) // 8
            elif t not in ('E', 'BYPASS'):
                raise C.Refusal('unsupported LDGSTS modifier ' + op)
        info = dict(kind='async_write', width=width, atomic=False)
    elif base in ('LDS', 'STS'):
        width = 4
        for t in mods:
            if t not in WIDTH_MODS:
                raise C.Refusal('unsupported shared modifier ' + op)
            width = WIDTH_MODS[t]
        info = dict(kind='load' if base == 'LDS' else 'store', width=width, atomic=False)
    elif base == 'ATOMS':
        width = 8 if '64' in mods else 4
        for t in mods:
            if t not in ATOMS_MODS and not t.isdigit():
                raise C.Refusal('unsupported ATOMS modifier ' + op)
        info = dict(kind='atomic', width=width, atomic=True)
    elif base in ('LDSM', 'STSM'):
        matrices = 1
        for t in mods:
            if t in ('16', 'M88', 'MT88', 'T'):
                continue
            if t in ('1', '2', '4'):
                matrices = int(t)
            else:
                raise C.Refusal('unsupported LDSM modifier ' + op)
        info = dict(kind='ldsm', width=16, atomic=False, matrices=matrices)
    else:
        return None
    w = info['width']
    if w not in (1, 2, 4, 8, 16):
        raise C.Refusal('unsupported shared width ' + op)
    info['group_lanes'] = 8 if w == 16 else 16 if w == 8 else 32
    info['lanes_used'] = 8 * info['matrices'] if info['kind'] == 'ldsm' else 32
    return info


def has_shared_ops(sass_text):
    for line in sass_text.splitlines():
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)', line)
        if m and m[1].split('.')[0] in SHARED_BASE + ASYNC_BASE:
            return True
    return False


# ------------------------------------------------------------------------------------------------ conflict arithmetic
def group_degree(addresses, width, atomic=False):
    """Degree of one group: max over banks of the number of distinct 32-bit words (lane accesses when atomic)."""
    banks = collections.defaultdict(set) if not atomic else collections.defaultdict(list)
    for lane_i, a in enumerate(addresses):
        for w in range(a // 4, (a + width - 1) // 4 + 1):
            if atomic:
                banks[w % NBANKS].append((lane_i, w))
            else:
                banks[w % NBANKS].add(w)
    return max((len(v) for v in banks.values()), default=0)


def request_degrees(lane_addresses, width, atomic=False, group_lanes=None):
    """lane_addresses: {lane: byte_address} of the participating lanes. Returns tuple of group degrees (non-empty groups)."""
    g = group_lanes or (8 if width == 16 else 16 if width == 8 else 32)
    groups = collections.defaultdict(list)
    for lane, a in sorted(lane_addresses.items()):
        groups[lane // g].append(a)
    return tuple(group_degree(groups[k], width, atomic) for k in sorted(groups))


def request_costs(degrees, group_lanes):
    """(primary, per_group_floor, lane_scaled) cost in cycles for one request."""
    wave = sum(degrees)
    return (max(FLOOR, float(wave)),
            sum(max(FLOOR, float(d)) for d in degrees),
            sum(max(FLOOR, float(d)) * group_lanes / 32.0 for d in degrees))


# ------------------------------------------------------------------------------------------------ address evaluation
def operand_inner(s):
    for t in s.a:
        if t.startswith('[') and t.endswith(']'):
            return t[1:-1]
    raise C.Refusal('no shared-memory operand in ' + s.op + ' ' + ','.join(s.a))


def eval_terms(inner, getter):
    """Shared byte address of an operand body like R5+UR4+0x20; None when a register is unknown."""
    total = 0
    for term in inner.split('+'):
        term = term.strip().replace('.reuse', '')
        if term == '':
            continue
        if re.fullmatch(r'-?0x[0-9a-f]+', term):
            v = int(term, 0)
        elif term in ('RZ', 'URZ'):
            v = 0
        elif re.fullmatch(r'U?R\d+', term):
            v = getter(term)
        else:
            raise C.Refusal('unsupported shared address term ' + term)
        if v is None:
            return None
        total += v
    return total & M32


def _pt_getter(r):
    def get(t):
        v = r.get(t)
        return None if v is None or v.exact_value is None else v.lo
    return get


# ------------------------------------------------------------------------------------------------ observer
class SharedObserver(PH.Observer):
    def __init__(self, threads):
        super().__init__(threads)
        self.shared = collections.defaultdict(list)   # (warp, phase, pc, k, op) -> [(lane, guard, address)]
        self.info = {}

    def _record(self, lane, s, guard, address):
        info = self.info.get(s.op)
        if info is None:
            info = self.info[s.op] = shared_info(s.op)
        if info is None:
            return
        ph = self.phase[lane]
        k = self.arrivals[(lane, ph, s.pc)]      # event() increments this afterwards: same k as the frozen warp count
        self.shared[(lane // 32, ph, s.pc, k, s.op)].append((lane % 32, guard, address))

    def concrete(self, lane):
        base = super().concrete(lane)

        def callback(s, guard, m):
            if shared_info(s.op) is not None:
                self._record(lane, s, guard, eval_terms(operand_inner(s), m.rd))
            base(s, guard, m)
        return callback

    def pytorch(self, coords, s, guard, r, p):
        if shared_info(s.op) is not None:
            lane = coords['SR_TID.X'].exact_value + coords['SR_TID.Y'].exact_value * self.block_x
            self._record(lane, s, guard, eval_terms(operand_inner(s), _pt_getter(r)))
        super().pytorch(coords, s, guard, r, p)

    # ---- fold the records of one block into counters (multiplied by `blocks` later)
    def classify(self):
        """Counter over (op, status, degrees, reason) for one block. status: known | off | unknown."""
        out = collections.Counter()
        for (warp, ph, pc, k, op), ents in self.shared.items():
            info = self.info[op]
            used = [e for e in ents if e[0] < info['lanes_used']]
            guards = [e[1] for e in used]
            if any(g is None for g in guards):
                out[(ph, op, 'unknown', (), 'data-dependent predicate')] += 1
            elif not any(g is True for g in guards):
                out[(ph, op, 'off', (), '')] += 1
            else:
                act = [e for e in used if e[1] is True]
                if any(e[2] is None for e in act):
                    out[(ph, op, 'unknown', (), 'data-dependent or unevaluated shared address')] += 1
                else:
                    lanes = {e[0]: e[2] for e in act}
                    deg = request_degrees(lanes, info['width'], info['atomic'], info['group_lanes'])
                    out[(ph, op, 'known', deg, '')] += 1
        return out


def fold_phase(counter, phase_index, blocks, infos):
    """Per-phase output dict from the classify() Counter (one block) scaled by `blocks`."""
    row = collections.OrderedDict()
    req = act = off = unk = 0
    cost = cost_floor = cost_scaled = 0.0
    a_req = a_act = a_unk = 0
    a_cost = 0.0
    hist, ghist, whist = collections.Counter(), collections.Counter(), collections.Counter()
    a_hist = collections.Counter()
    by_op = collections.Counter()
    detail = collections.defaultdict(lambda: dict(requests=0, active_requests=0, unknown_requests=0, cost_cycles=0.0,
                                                  degree_sum=0, kind=None, width_bytes=None))
    reasons = collections.Counter()
    gt2 = 0
    deg_sum = 0
    for (ph, op, status, deg, reason), n in counter.items():
        if ph != phase_index:
            continue
        n *= blocks
        info = infos[op]
        is_async = info['kind'] == 'async_write'
        d = detail[op]
        d['kind'], d['width_bytes'] = info['kind'], info['width']
        d['requests'] += n
        if is_async:
            a_req += n
        else:
            by_op[op] += n
            req += n
        if status == 'off':
            off += 0 if is_async else n
            continue
        if status == 'unknown':
            d['unknown_requests'] += n
            if is_async:
                a_unk += n
            else:
                unk += n
            reasons[reason] += n
            continue
        d['active_requests'] += n
        c0, c1, c2 = request_costs(deg, info['group_lanes'])
        d['cost_cycles'] += c0 * n
        mx = max(deg)
        if is_async:
            a_act += n; a_cost += c0 * n; a_hist[mx] += n
            continue
        act += n
        cost += c0 * n; cost_floor += c1 * n; cost_scaled += c2 * n
        hist[mx] += n
        whist[sum(deg)] += n
        for g in deg:
            ghist[g] += n
        gt2 += n if mx > 2 else 0
        deg_sum += mx * n
        d['degree_sum'] += mx * n
    off_all = sum(n * blocks for (ph, op, st, deg, r), n in counter.items() if ph == phase_index and st == 'off' and infos[op]['kind'] != 'async_write')
    exact = unk == 0
    srt = lambda c: {str(k): v for k, v in sorted(c.items())}
    row['shared_requests'] = req
    row['shared_requests_active'] = act
    row['shared_requests_all_lanes_predicated_off'] = off_all
    row['shared_requests_unknown'] = unk
    row['shared_unknown_reasons'] = dict(reasons)
    row['shared_cost_cycles'] = cost if exact else None
    row['shared_cost_cycles_known_requests_only'] = cost
    row['shared_cost_cycles_per_group_floor'] = cost_floor if exact else None
    row['shared_cost_cycles_lane_scaled'] = cost_scaled if exact else None
    row['shared_conflict_degree_histogram'] = srt(hist)
    row['shared_group_degree_histogram'] = srt(ghist)
    row['shared_request_wavefront_histogram'] = srt(whist)
    row['shared_requests_degree_gt2'] = gt2
    row['shared_degree_sum_active'] = deg_sum
    row['shared_requests_by_opcode'] = dict(sorted(by_op.items()))
    row['shared_by_opcode_detail'] = {op: v for op, v in sorted(detail.items()) if infos[op]['kind'] != 'async_write'}
    row['async_shared_write_requests'] = a_req
    row['async_shared_write_requests_active'] = a_act
    row['async_shared_write_requests_unknown'] = a_unk
    row['async_shared_write_cost_cycles_known_only'] = a_cost
    row['async_shared_write_degree_histogram'] = srt(a_hist)
    row['async_shared_write_by_opcode_detail'] = {op: v for op, v in sorted(detail.items()) if infos[op]['kind'] == 'async_write'}
    return row


def shared_phases(obs, blocks, nphases):
    counter = obs.classify()
    return [fold_phase(counter, i, blocks, obs.info) for i in range(nphases)]


def scale_free_signature(obs, nphases):
    """One-block conflict statistics, used to compare sampled blocks."""
    return json.dumps(shared_phases(obs, 1, nphases), sort_keys=True)


# ------------------------------------------------------------------------------------------------ gate
GATE_KEYS = ('phases', 'barrier_sequence', 'unknown_guard_upper_bounds')


def canon(x):
    return json.dumps(x, sort_keys=True)


def gate_kernel(mine, frozen, shared_rows):
    """Raise Refusal unless the instrumented run reproduces the frozen kernel record and shared counts."""
    diffs = []
    for k in GATE_KEYS:
        if canon(mine[k]) != canon(frozen[k]):
            diffs.append(k)
    if diffs:
        detail = []
        for i, (a, b) in enumerate(zip(mine['phases'], frozen['phases'])):
            for f in a:
                if canon(a[f]) != canon(b.get(f)):
                    detail.append('phase %d %s' % (i, f))
        raise C.Refusal('regression gate: instrumented run differs from frozen phase table in %s (%s)' % (diffs, '; '.join(detail[:6])))
    for i, (sh, fp) in enumerate(zip(shared_rows, frozen['phases'])):
        want = {op: n for op, n in fp['issue_warp_instructions'].items() if _is_shared_opcode(op)}
        have = dict(sh['shared_requests_by_opcode'])
        for op, v in sh['async_shared_write_by_opcode_detail'].items():
            have[op] = v['requests']
        if want != have:
            raise C.Refusal('regression gate: shared request counts %s differ from frozen warp instruction counts %s in phase %d' % (have, want, i))


def _is_shared_opcode(op):
    try:
        return shared_info(op) is not None
    except C.Refusal:
        return True


# ------------------------------------------------------------------------------------------------ per-kernel runners
def run_retained(corpus, row, root):
    """Same binding and block sampling as phases.retained; returns (base_kernel_record, shared_rows, diagnostics)."""
    D.verify(root, row)
    meta = D.parameter_layout(root, row)
    constants, coords, threads, blocks, launch = D.binding(corpus, row, root)
    cb = {k: v.exact_value for k, v in constants.items()}
    cb.update(C.pointer_consts(meta)[0])
    grid, block = launch['grid'], launch['block']
    for off, v in zip((0x360, 0x364, 0x368), block):
        cb.setdefault(off, v)
    for off, v in zip((0x370, 0x374, 0x378), grid):
        cb.setdefault(off, v)
    sites = D.parse((root / row['disassembly_path']).read_text())
    samples = sorted({0, blocks // 2, blocks - 1})
    bases, shareds, sigs, diag = [], [], [], []
    for b in samples:
        obs = SharedObserver(threads)
        arrivals = []
        for l in range(threads):
            xyz = {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': b // grid[0], 'SR_CTAID.Z': 0,
                   'SR_CgaCtaId': 0, 'SR_TID.X': l % block[0], 'SR_TID.Y': (l // block[0]) % block[1],
                   'SR_TID.Z': l // (block[0] * block[1]), 'SR_LANEID': l % 32}
            _, _, arr, _ = PH.TRACE_THREAD(sites, cb, xyz, observer=obs.concrete(l))
            arrivals.append(arr)
        C.COL.check_warp_convergence(arrivals, threads)
        base = obs.finish(blocks)
        n = len(base['phases'])
        bases.append(base)
        shareds.append(shared_phases(obs, blocks, n))
        sigs.append(scale_free_signature(obs, n))
        diag.append(dict(block=b, shared_phases_unscaled=shared_phases(obs, 1, n)))
    if any(x != bases[0] for x in bases[1:]):
        err = C.Refusal('sampled blocks have different phase signatures; grid phase split unproved')
        err.diagnostic = diag
        raise err
    if any(x != sigs[0] for x in sigs[1:]):
        err = C.Refusal('sampled blocks have different shared-memory signatures; grid extrapolation unproved')
        err.diagnostic = diag
        raise err
    base = bases[0]
    base['sampled_blocks'] = samples
    return base, shareds[0], diag


def run_pytorch(kid, kern, cell, corrected):
    constants, grid, block = PH.bind_pytorch(kid, kern, cell)
    threads = math.prod(block)
    sites = P.parse((STATIC / 'libtorch_sm120' / (kid + '.isolated.sass')).read_text())
    Interp = divider_fix.corrected_interp_class(P) if corrected else P.Interp
    bases, shareds, sigs = [], [], []
    for b in sorted({0, math.prod(grid) - 1}):
        obs = SharedObserver(threads)
        obs.block_x = block[0]
        interp = Interp(C.D, ext=True, fchk_fast_path=True, forced_branches={0x150: True} if kid == 'k2' else {}, trace=obs.pytorch)

        def coords(l, b=b):
            return {k: P.V.exact(v) for k, v in {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': b // grid[0], 'SR_CTAID.Z': 0,
                                                 'SR_CgaCtaId': 0, 'SR_TID.X': l % block[0], 'SR_TID.Y': l // block[0],
                                                 'SR_TID.Z': 0, 'SR_LANEID': l % 32}.items()}
        interp.run_block(sites, {k: P.V.exact(v) for k, v in constants.items()}, coords, threads)
        base = obs.finish(math.prod(grid))
        bases.append(base)
        n = len(base['phases'])
        shareds.append(shared_phases(obs, math.prod(grid), n))
        sigs.append(scale_free_signature(obs, n))
    C.require(bases[0] == bases[-1], 'PyTorch boundary block phase signatures differ')
    C.require(sigs[0] == sigs[-1], 'PyTorch boundary block shared-memory signatures differ')
    return bases[0], shareds[0]


# ------------------------------------------------------------------------------------------------ summaries
def kernel_summary(phase_rows):
    tot = collections.Counter()
    unk = 0
    cost = cost_floor = cost_scaled = 0.0
    for r in phase_rows:
        for f in ('shared_requests', 'shared_requests_active', 'shared_requests_all_lanes_predicated_off', 'shared_requests_unknown',
                  'shared_requests_degree_gt2', 'shared_degree_sum_active', 'async_shared_write_requests',
                  'async_shared_write_requests_active'):
            tot[f] += r[f]
        cost += r['shared_cost_cycles_known_requests_only']
        cost_floor += r['shared_cost_cycles_per_group_floor'] if r['shared_cost_cycles_per_group_floor'] is not None else 0.0
        cost_scaled += r['shared_cost_cycles_lane_scaled'] if r['shared_cost_cycles_lane_scaled'] is not None else 0.0
        unk += r['shared_requests_unknown']
    act = tot['shared_requests_active']
    out = dict(tot)
    out['shared_cost_cycles'] = cost if unk == 0 else None
    out['shared_cost_cycles_known_requests_only'] = cost
    out['shared_cost_cycles_per_group_floor'] = cost_floor if unk == 0 else None
    out['shared_cost_cycles_lane_scaled'] = cost_scaled if unk == 0 else None
    out['share_requests_degree_gt2'] = (tot['shared_requests_degree_gt2'] / act) if act and unk == 0 else (None if unk else 0.0)
    out['mean_request_degree'] = (tot['shared_degree_sum_active'] / act) if act and unk == 0 else (None if unk else 0.0)
    out['requests_by_opcode'] = dict(sum((collections.Counter(r['shared_requests_by_opcode']) for r in phase_rows), collections.Counter()))
    return out


def cell_summary(kernel_summaries):
    tot = collections.Counter()
    unknown = False
    for k in kernel_summaries:
        for f in ('shared_requests', 'shared_requests_active', 'shared_requests_unknown', 'shared_requests_degree_gt2',
                  'shared_degree_sum_active'):
            tot[f] += k[f]
        unknown |= k['shared_cost_cycles'] is None
    out = dict(tot)
    out['shared_cost_cycles'] = None if unknown else sum(k['shared_cost_cycles'] for k in kernel_summaries)
    act = tot['shared_requests_active']
    out['share_requests_degree_gt2'] = None if unknown else (tot['shared_requests_degree_gt2'] / act if act else 0.0)
    out['mean_request_degree'] = None if unknown else (tot['shared_degree_sum_active'] / act if act else 0.0)
    return out


# ------------------------------------------------------------------------------------------------ set-level driver
def _dev_lookup():
    """cell_id -> (corpus, manifest row, root) for the retained development corpus."""
    out = {}
    for corpus, root in D.CORPORA.items():
        for row in json.loads((root / 'retention_manifest.json').read_text())['rows']:
            out['blackwell/' + row['operator_id'] + '/' + row['cell']] = (corpus, row, root)
    return out


def fresh_d_lookup(cells_path=None):
    """cell_id -> (corpus, marked manifest row, root) for the CUDA Samples set D (installs its binding patch by importing its library)."""
    sys.path.insert(0, str(STATIC / 'fresh_d'))
    import fresh_d_lib as L  # noqa: E402  (imported, never edited; replaces derive.binding in every loaded copy)
    cells = json.loads(Path(cells_path or STATIC / 'fresh_d' / 'fresh_cells_d.json').read_text())['cells']
    rows = L.retained_rows()
    return {c['cell_id']: ('cuda', L.marked_row(rows[(c['origin_operator_id'], c['origin_cell'])], c['kernel_family'], c['geometry']), L.CUDA_ROOT)
            for c in cells}


def _find_dispatch(features_path):
    fp = Path(features_path)
    if fp.name == 'features_blackwell.json':
        return STATIC / 'pytorch_dispatch' / 'dispatch_trace.json'
    hits = sorted(fp.parent.glob('dispatch_trace*.json'))
    return hits[0] if hits else None


# worker tasks (module level so that a forked pool can run them)
def _task(args):
    kind = args[0]
    try:
        if kind == 'retained':
            _, cid, corpus, row, root = args
            base, shared, diag = run_retained(corpus, row, root)
            return (kind, cid, 'ok', base, shared, diag)
        _, key, kid, kern, cell, corrected = args
        base, shared = run_pytorch(kid, kern, cell, corrected)
        return (kind, key, 'ok', base, shared, None)
    except (C.Refusal, P.Refusal) as ex:
        return (kind, args[1], 'refused', str(ex), None, getattr(ex, 'diagnostic', None))
    except Exception as ex:  # unexpected: surface loudly but keep the pool alive
        return (kind, args[1], 'error', ''.join(traceback.format_exception_only(type(ex), ex)).strip() + ' @ ' + traceback.format_exc()[-400:], None, None)


def _pt_key(kid, kern, cell):
    return json.dumps([kid, kern, cell['input'] if kid == 'k1' else None], sort_keys=True)


ASSUMPTIONS = [
    'Bank of a 32-bit word = (byte_address / 4) % 32; shared byte addresses are the evaluated register sums of the operand (R+UR+imm); the dynamic-shared base is whatever the interpreter resolves (no extra offset is added).',
    'A warp request is the k-th dynamic arrival of the lanes of a warp at a site (same as the frozen warp-instruction count); lanes that did not arrive or are predicated off do not participate.',
    'Groups: <=32-bit one group of 32 lanes; 64-bit two half-warp groups; 128-bit and every LDSM matrix quarter-warp groups of 8 lanes; degree = max distinct words in one bank per group (same word = broadcast, counted once); ATOMS counts every lane access.',
    'Primary cost per request = max(2.0, sum of group degrees): one wavefront per cycle with a 2.0-cycle issue floor. This equals the microbenchmark rule max(2.0, degree) for 32-bit accesses and is an UNMEASURED extension for 64/128-bit accesses; the per-group floor and lane-scaled alternatives are reported as well.',
    'A warp instruction whose lanes are all predicated off is issued (counted in shared_requests, matching the frozen warp counts) but charged 0 cycles.',
    'Cross-lane/loop-carried data-dependent addresses or predicates stay null with the reason; any phase containing one has shared_cost_cycles = null (the known part is reported separately).',
    'LDGSTS (cp.async) shared writes are analysed separately (async_shared_write_*); they are not in shared_requests / shared_cost_cycles. The source-size predicate is ignored for the shared destination (zero fill still writes).',
    'Grid extrapolation: same sampled blocks as phases.py (first, middle, last; PyTorch first and last); the shared signature of the sampled blocks must agree, otherwise the cell is refused.',
]


def analyse_set(features_path, phases_path, name, *, dispatch_path=None, corrected_divider=None, lookup=None, workers=4):
    """Shared-memory bank-conflict table for one set with the schemas of features_blackwell.json / phases_blackwell.json.

    features_path / phases_path: the set's feature and phase tables (the phase table is the regression reference).
    lookup: optional {cell_id: (corpus, manifest_row, root)} for retained-binary cells outside the development corpus
            (e.g. a CUDA Samples set whose binding patch has been installed by importing its library first).
    dispatch_path: dispatch trace of the PyTorch cells (auto-detected next to the feature table otherwise).
    corrected_divider: use the IMAD.HI.U32-corrected interpreter (the fresh PyTorch sets); auto: True unless the
            feature table is the development table.
    """
    features = json.loads(Path(features_path).read_text())
    frozen = json.loads(Path(phases_path).read_text())
    if corrected_divider is None:
        corrected_divider = Path(features_path).name != 'features_blackwell.json'
    lookup = _dev_lookup() if lookup is None else lookup
    dpath = dispatch_path or _find_dispatch(features_path)
    dcells = {c['cell_id']: c for c in json.loads(Path(dpath).read_text())['cells']} if dpath and Path(dpath).exists() else {}
    tasks, pt_tasks = [], {}
    plan = {}
    for cid, frow in sorted(frozen['rows'].items()):
        if cid in dcells:
            kinds = []
            for kern in dcells[cid]['kernels']:
                kid = A.kernel_id(kern)
                key = _pt_key(kid, kern, dcells[cid])
                text = (STATIC / 'libtorch_sm120' / (kid + '.isolated.sass')).read_text()
                if kid == 'k1' or not text:
                    if has_shared_ops(text):
                        raise C.Refusal('copy kernel unexpectedly has shared opcodes')
                    kinds.append((kid, key, 'no_shared_opcodes'))
                else:
                    kinds.append((kid, key, 'simulate'))
                    if key not in pt_tasks:
                        pt_tasks[key] = ('pytorch', key, kid, kern, dcells[cid], corrected_divider)
            plan[cid] = ('pytorch', kinds)
        elif cid in lookup and frow['kernels']:
            corpus, row, root = lookup[cid]
            tasks.append(('retained', cid, corpus, row, root))
            plan[cid] = ('retained',)
        elif cid in lookup:
            corpus, row, root = lookup[cid]
            tasks.append(('retained', cid, corpus, row, root))     # frozen refused it: still run for the refusal diagnostic
            plan[cid] = ('retained',)
        else:
            plan[cid] = ('none',)
    all_tasks = list(pt_tasks.values()) + tasks
    results = {}
    ctx = multiprocessing.get_context('fork')
    if workers and workers > 1 and len(all_tasks) > 1:
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(_task, all_tasks):
                results[(res[0], res[1])] = res
                print(name, res[0], str(res[1])[:90], res[2], flush=True)
    else:
        for t in all_tasks:
            res = _task(t)
            results[(res[0], res[1])] = res
            print(name, res[0], str(res[1])[:90], res[2], flush=True)

    rows = {}
    gate = collections.Counter()
    for cid, frow in sorted(frozen['rows'].items()):
        st = frow['status']
        out = dict(status=None, reason=frow.get('reason'), phases_status=st, kernels=[])
        try:
            kind = plan[cid][0]
            if kind == 'none':
                out.update(status='unsupported', reason=frow.get('reason') or 'no source binding for this cell')
                gate['not_analysed_unsupported_in_phase_table'] += 1
            elif kind == 'pytorch':
                krows = []
                for (kid, key, how), fk in zip(plan[cid][1], frow['kernels']):
                    if how == 'no_shared_opcodes':
                        n = len(fk['phases'])
                        sh = [fold_phase(collections.Counter(), i, 1, {}) for i in range(n)]
                        for r in sh:
                            r['shared_cost_cycles'] = 0.0
                        for ph in fk['phases']:
                            assert not any(_is_shared_opcode(op) for op in ph['issue_warp_instructions']), 'frozen counts show shared ops in a kernel without any'
                        krow = dict(kernel_id=fk.get('kernel_id', kid), barrier_sequence=fk['barrier_sequence'], phases=_merge_phase_ids(fk['phases'], sh),
                                    gate='no shared-memory opcode in the isolated SASS and none in the frozen counts; global sectors not re-simulated (kernel not instrumented)')
                    else:
                        res = results[('pytorch', key)]
                        if res[2] != 'ok':
                            raise C.Refusal(res[3])
                        base, sh = res[3], res[4]
                        gate_kernel(base, fk, sh)
                        krow = dict(kernel_id=fk.get('kernel_id', kid), barrier_sequence=fk['barrier_sequence'], phases=_merge_phase_ids(fk['phases'], sh),
                                    gate='passed: phases/barriers/sectors/lines byte-identical to the frozen table; shared counts equal frozen warp counts')
                    krow['shared_summary'] = kernel_summary([p['shared'] for p in krow['phases']])
                    krows.append(krow)
                out.update(kernels=krows)
                if not krows:
                    gate['not_analysed_unsupported_in_phase_table'] += 1
                out['status'] = 'unsupported' if not krows else 'static_shared_conflicts' if st == 'conditional_static_phases' else 'static_shared_conflicts_phase_table_unsupported'
                if st != 'conditional_static_phases':
                    out['reason'] = frow.get('reason')
            else:
                res = results[('retained', cid)]
                if res[2] != 'ok':
                    out.update(status='unsupported', reason=res[3])
                    if res[5]:
                        out['sampled_block_shared_diagnostic_not_extrapolated'] = res[5]
                    if frow['kernels']:
                        # the frozen table had phases but the instrumented run refused: this is a gate failure
                        gate['GATE_FAILED'] += 1
                        out['reason'] = 'regression gate: instrumented run refused where the frozen table has phases: ' + res[3]
                    else:
                        gate['refused_as_in_phase_table'] += 1
                else:
                    base, sh, diag = res[3], res[4], res[5]
                    if not frow['kernels']:
                        raise C.Refusal('regression gate: frozen table refused this cell but the instrumented run did not')
                    fk = frow['kernels'][0]
                    gate_kernel(base, fk, sh)
                    krow = dict(barrier_sequence=fk['barrier_sequence'], phases=_merge_phase_ids(fk['phases'], sh), sampled_blocks=base['sampled_blocks'],
                                gate='passed: phases/barriers/sectors/lines byte-identical to the frozen table; shared counts equal frozen warp counts')
                    krow['shared_summary'] = kernel_summary([p['shared'] for p in krow['phases']])
                    out.update(kernels=[krow], status='static_shared_conflicts')
            if out['kernels']:
                out['shared_summary'] = cell_summary([k['shared_summary'] for k in out['kernels']])
                gate['passed'] += 1
        except C.Refusal as ex:
            out.update(status='refused', reason=str(ex), kernels=[])
            gate['GATE_FAILED' if str(ex).startswith('regression gate') else 'refused_other'] += 1
        rows[cid] = out
    doc = dict(schema=SCHEMA, set=name, features_path=str(features_path), phases_path=str(phases_path),
               cost_model=dict(floor_cycles=FLOOR, banks=NBANKS, primary='max(2.0, sum of group degrees) per request',
                               alternatives=['shared_cost_cycles_per_group_floor', 'shared_cost_cycles_lane_scaled']),
               corrected_divider_interpreter=bool(corrected_divider), assumptions=ASSUMPTIONS,
               gate_summary=dict(gate), transaction_profiler_validation=False, shared_microbenchmark_validation=False,
               rows=rows)
    return doc


def _merge_phase_ids(frozen_phases, shared_rows):
    out = []
    for fp, sh in zip(frozen_phases, shared_rows):
        out.append(dict(index=fp['index'], repetitions=fp['repetitions'], shared=sh))
    return out


SETS = {
    'dev': ('features_blackwell.json', 'phases_blackwell.json', 'bank_conflicts_dev.json'),
    'fresh_a': ('fresh/features_fresh.json', 'fresh/phases_fresh_divider_corrected.json', 'bank_conflicts_fresh_a.json'),
    'fresh_b': ('fresh_b/features_fresh_b.json', 'fresh_b/phases_fresh_b.json', 'bank_conflicts_fresh_b.json'),
    'fresh_d': ('fresh_d/features_fresh_d.json', 'fresh_d/phases_fresh_d.json', 'bank_conflicts_fresh_d.json'),
    'fresh_c': ('fresh_c/features_fresh_c.json', 'fresh_c/phases_fresh_c.json', 'bank_conflicts_fresh_c.json'),
}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='dev,fresh_a,fresh_b,fresh_c')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args(argv)
    for s in args.sets.split(','):
        f, p, o = SETS[s]
        doc = analyse_set(STATIC / f, STATIC / p, s, workers=args.workers, lookup=fresh_d_lookup() if s == 'fresh_d' else None,
                          corrected_divider=False if s == 'fresh_d' else None)
        (HERE / o).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
        print(s, 'gate', doc['gate_summary'], flush=True)


if __name__ == '__main__':
    main()
