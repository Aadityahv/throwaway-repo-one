"""Label-free dynamic barrier phases and memory requests from retained SASS.

No profiler or operator runtime is opened. The retained interpreter is instrumented
with a pre-instruction observer in memory; its file and numerical semantics remain
unchanged. Grid extrapolation is conditional on sampled block signatures, as in
the frozen coalescing analysis. Refused paths and unknown addresses stay null.
"""
import collections
import hashlib
import inspect
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'coalescing'))
import coalescing as C
X = C._load('runtime_phase_features', HERE / 'extract_features.py')
A = C._load('runtime_phase_pytorch_adapter', HERE / 'pytorch_features/adapter.py')
P = A.P


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def instrument(fn, needle, replacement):
    source = inspect.getsource(fn)
    if source.count(needle) != 1:
        raise ValueError('observer insertion point changed')
    env = dict(fn.__globals__)
    exec(compile(source.replace(needle, replacement), '<read-only-phase-observer>', 'exec'), env)
    return env[fn.__name__]


# Add the observer argument without touching the frozen interpreter on disk.
_src = inspect.getsource(C.run_thread).replace('probe=None):', 'probe=None, observer=None):')
_src = _src.replace('        do = m.pr(s.pred) if s.pred else True\n',
    '        do = m.pr(s.pred) if s.pred else True\n        if observer is not None: observer(s, do, m)\n')
_env = dict(C.run_thread.__globals__)
exec(compile(_src, '<read-only-phase-observer>', 'exec'), _env)
TRACE_THREAD = _env['run_thread']


def memory_kind(op):
    base = op.split('.')[0]
    if base not in ('LDG', 'STG', 'LDGSTS'):
        return None
    width = next((int(s) // 8 for s in op.split('.')[1:] if s.isdigit()), 4)
    C.require(width in (4, 8, 16), 'unknown global width')
    return ('write' if base == 'STG' else 'read'), width


class Observer:
    def __init__(self, threads):
        self.threads = threads
        self.phase = [0] * threads
        self.barriers = [[] for _ in range(threads)]
        self.regs = [{} for _ in range(threads)]
        self.chains = collections.defaultdict(lambda: [0, 0])
        self.completed_chains = collections.defaultdict(lambda: [0, 0])
        self.last_pc = [None] * threads
        self.op_visits = collections.Counter()
        self.arrivals = collections.Counter()
        self.memory = collections.defaultdict(list)
        self.unknown_guards = set()

    def event(self, lane, s, guard, address=None, memory_guard=True):
        ph = self.phase[lane]
        if self.last_pc[lane] is not None and s.pc < self.last_pc[lane]:
            # The preregistered latency term charges each executed loop trip.
            old = self.chains[(lane, ph)]
            total = self.completed_chains[(lane, ph)]
            total[0] += old[0]; total[1] += old[1]
            self.chains[(lane, ph)] = [0, 0]
            self.regs[lane] = {}
        self.last_pc[lane] = s.pc
        key = (lane, ph, s.pc)
        k = self.arrivals[key]
        self.arrivals[key] += 1
        self.op_visits[(lane // 32, ph, s.pc, k, s.op)] += 1
        if guard is None:
            self.unknown_guards.add(hex(s.pc))
        if guard is not False:
            defs, uses = X.sass_def_use(s)
            predecessors = [self.regs[lane].get(r, (0, 0)) for r in uses]
            is_load = s.op.split('.')[0] in ('LDG', 'LDGSTS')
            # Compute path counts exclude global loads, which have their own latency.
            cp = max((x[0] for x in predecessors), default=0) + (0 if is_load else 1)
            ld = max((x[1] for x in predecessors), default=0) + int(is_load)
            for r in defs:
                self.regs[lane][r] = (cp, ld)
            ch = self.chains[(lane, ph)]
            ch[0] = max(ch[0], cp)
            ch[1] = max(ch[1], ld)
            mk = memory_kind(s.op)
            if mk and memory_guard is not False:
                self.memory[(lane // 32, ph, s.pc, k, *mk)].append(
                    address if guard is True and memory_guard is True else None)
            if s.op.startswith('BAR.SYNC'):
                C.require(guard is True, 'unknown barrier guard')
                self.barriers[lane].append(s.pc)
                self.phase[lane] += 1
                self.regs[lane] = {}  # phases serialize; cross-phase chains do not add twice
                self.last_pc[lane] = None

    def concrete(self, lane):
        def callback(s, guard, m):
            address, src = None, True
            if memory_kind(s.op):
                reg, imm, sp = C.parse_mem_operand(s, s.a)
                v = m.rd64(reg)
                address = None if v is None else (v + imm) & C.M64
                src = m.pr(sp) if sp else True
            self.event(lane, s, guard, address, src)
        return callback

    def pytorch(self, coords, s, guard, r, p):
        lane = coords['SR_TID.X'].exact_value + coords['SR_TID.Y'].exact_value * self.block_x
        address = None
        if memory_kind(s.op):
            token = s.a[0] if s.op.startswith('STG') else s.a[1]
            m = C.MEMRE.match(token)
            C.require(m is not None, 'unknown PyTorch memory operand')
            lo = r.get(m[2]); rn = re.match(r'(U?R)(\d+)', m[2])
            hi = r.get(rn[1] + str(int(rn[2]) + 1))
            if lo is not None and hi is not None and lo.exact_value is not None and hi.exact_value is not None:
                address = ((hi.lo << 32) | lo.lo) + (int(m[3], 0) if m[3] else 0)
        self.event(lane, s, guard, address)

    def finish(self, blocks):
        C.require(all(b == self.barriers[0] for b in self.barriers), 'nonuniform block barrier sequence')
        result = []
        for ph in range(max(self.phase) + 1):
            ops = collections.Counter()
            for (warp, phase, pc, k, op) in self.op_visits:
                if phase == ph:
                    ops[op] += blocks
            row = dict(index=ph, repetitions=1, issue_warp_instructions=dict(ops),
                       read_sectors=0, write_sectors=0, lines=0, read_bytes=0, write_bytes=0,
                       critical_path_compute_instructions=max((v[0] + self.completed_chains[(l, p)][0] for (l, p), v in self.chains.items() if p == ph), default=0),
                       dependent_global_load_depth=max((v[1] + self.completed_chains[(l, p)][1] for (l, p), v in self.chains.items() if p == ph), default=0))
            for (warp, phase, pc, k, direction, width), addresses in self.memory.items():
                if phase != ph:
                    continue
                row[direction + '_bytes'] += len(addresses) * width * blocks
                if any(a is None for a in addresses):
                    row[direction + '_sectors'] = None
                    row['lines'] = None
                else:
                    if row[direction + '_sectors'] is not None:
                        row[direction + '_sectors'] += len({g for a in addresses for g in C.granules(a, width, 32)}) * blocks
                    if row['lines'] is not None:
                        row['lines'] += len({g for a in addresses for g in C.granules(a, width, 128)}) * blocks
            result.append(row)
        return dict(phases=result, barrier_sequence=[hex(pc) for pc in self.barriers[0]],
                    unknown_guard_upper_bounds=sorted(self.unknown_guards))


def retained(corpus, row, root, frozen):
    D = C.D
    D.verify(root, row)
    meta = D.parameter_layout(root, row)
    constants, coords, threads, blocks, launch = D.binding(corpus, row, root)
    cb = {k: v.exact_value for k, v in constants.items()}
    cb.update(C.pointer_consts(meta)[0])
    grid, block = launch['grid'], launch['block']
    for off, v in zip((0x360, 0x364, 0x368), block): cb.setdefault(off, v)
    for off, v in zip((0x370, 0x374, 0x378), grid): cb.setdefault(off, v)
    sites = D.parse((root / row['disassembly_path']).read_text())
    samples = sorted({0, blocks // 2, blocks - 1})
    signatures = []
    for b in samples:
        obs = Observer(threads)
        arrivals = []
        for l in range(threads):
            xyz = {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': b // grid[0], 'SR_CTAID.Z': 0,
                   'SR_CgaCtaId': 0, 'SR_TID.X': l % block[0], 'SR_TID.Y': (l // block[0]) % block[1],
                   'SR_TID.Z': l // (block[0] * block[1]), 'SR_LANEID': l % 32}
            _, _, arr, _ = TRACE_THREAD(sites, cb, xyz, observer=obs.concrete(l))
            arrivals.append(arr)
        C.COL.check_warp_convergence(arrivals, threads)
        signatures.append(obs.finish(blocks))
    # Full transaction totals from the separately frozen analysis must agree.
    base = signatures[0]
    if any(x != base for x in signatures[1:]):
        raise C.Refusal('sampled blocks have different phase signatures; grid phase split unproved')
    for field, expected in [('read_sectors', frozen['loads']['ldg_plus_ldgsts']['sectors']),
                            ('write_sectors', frozen['stores']['sectors']),
                            ('lines', frozen['loads']['ldg_plus_ldgsts']['lines'] + frozen['stores']['lines'])]:
        C.require(sum(p[field] for p in base['phases']) == expected, 'phase totals differ from frozen ' + field)
    base['sampled_blocks'] = samples
    base['grid_extrapolation'] = 'conditional on uniform sampled signatures, checked against frozen full-launch memory totals'
    return base


def bind_pytorch(kid, kern, cell):
    constants, grid, block, ptrs, _ = A.binding(kid, kern)
    off = {n: o for n, o, _ in A.abi_layout(kid)}
    # Include gamma/beta, which the prior adapter binds only for non-nullness.
    ptrs = ptrs + (['gamma', 'beta'] if kid == 'k3' else [])
    for i, name in enumerate(ptrs):
        value = C.PTR_BASE0 + i * C.PTR_STRIDE
        constants[off[name]], constants[off[name] + 4] = value & C.M32, value >> 32
    if kid == 'k1':
        rows, cols = cell['input']['shape']
        strides = cell['input']['strides']
        for i, divisor in enumerate([cols, rows]):
            shift = (divisor - 1).bit_length()
            magic = ((1 << 32) * ((1 << shift) - divisor)) // divisor + 1
            for field, value in [('divisor', divisor), ('m1', magic), ('shift', shift)]:
                constants[off[f'sizes_{i}.{field}']] = value
        # TensorIterator argument 0 is output, 1 input; byte strides in [cols,rows] order.
        for i, values in enumerate([(4, strides[1] * 4), (cols * 4, strides[0] * 4)]):
            for j, value in enumerate(values): constants[off[f'strides_{i}_{j}']] = value
    return constants, grid, block


def pytorch(kid, kern, cell):
    constants, grid, block = bind_pytorch(kid, kern, cell)
    threads = math.prod(block)
    sites = P.parse((HERE / 'libtorch_sm120' / (kid + '.isolated.sass')).read_text())
    signatures = []
    for b in sorted({0, math.prod(grid) - 1}):
        obs = Observer(threads); obs.block_x = block[0]
        interp = P.Interp(C.D, ext=True, fchk_fast_path=True,
            forced_branches={0x150: True} if kid == 'k2' else {}, trace=obs.pytorch)
        def coords(l):
            return {k: P.V.exact(v) for k, v in {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': b // grid[0],
                'SR_CTAID.Z': 0, 'SR_CgaCtaId': 0, 'SR_TID.X': l % block[0],
                'SR_TID.Y': l // block[0], 'SR_TID.Z': 0, 'SR_LANEID': l % 32}.items()}
        interp.run_block(sites, {k: P.V.exact(v) for k, v in constants.items()}, coords, threads)
        signatures.append(obs.finish(math.prod(grid)))
    C.require(signatures[0] == signatures[-1], 'PyTorch boundary block phase signatures differ')
    return signatures[0]


def main():
    features = json.loads((HERE / 'features_blackwell.json').read_text())
    frozen = {r['cell_id']: r for r in json.loads((HERE / 'coalescing/static_sectors_frozen.json').read_text())['rows']}
    manifest = json.loads((HERE / 'coalescing/FREEZE.json').read_text())
    for path, digest in manifest['hashes'].items():
        C.require(sha(X.REPO / path) == digest, 'frozen coalescing input changed: ' + path)
    rows = {}
    for corpus, root in C.D.CORPORA.items():
        for row in json.loads((root / 'retention_manifest.json').read_text())['rows']:
            cid = 'blackwell/' + row['operator_id'] + '/' + row['cell']
            try:
                q = retained(corpus, row, root, frozen[cid])
                rows[cid] = dict(status='conditional_static_phases', kernels=[q])
            except (C.Refusal, P.Refusal) as ex:
                rows[cid] = dict(status='unsupported', reason=str(ex), kernels=[])
            print(cid, rows[cid]['status'], rows[cid].get('reason', ''), flush=True)
    dispatch = json.loads((HERE / 'pytorch_dispatch/dispatch_trace.json').read_text())
    cache = {}
    for cell in dispatch['cells']:
        cid = cell['cell_id']; kernels = []
        try:
            for kern in cell['kernels']:
                kid = A.kernel_id(kern)
                key = json.dumps([kid, kern, cell['input'] if kid == 'k1' else None], sort_keys=True)
                if key not in cache:
                    cache[key] = pytorch(kid, kern, cell)
                kernels.append(dict(kernel_id=kid, **cache[key]))
            unknown = any(p['read_sectors'] is None or p['write_sectors'] is None for k in kernels for p in k['phases'])
            rows[cid] = dict(status='unsupported' if unknown else 'conditional_static_phases', kernels=kernels,
                reason='data-dependent global address; transaction counts remain null' if unknown else None)
        except (C.Refusal, P.Refusal) as ex:
            rows[cid] = dict(status='unsupported', reason=str(ex), kernels=[])
        print(cid, rows[cid]['status'], rows[cid].get('reason'), flush=True)
    for feature in features['rows']:
        rows.setdefault(feature['cell_id'], dict(status='unsupported', reason='no retained binary', kernels=[]))
    paths = [Path(__file__), HERE/'features_blackwell.json', HERE/'coalescing/static_sectors_frozen.json',
        HERE/'coalescing/coalescing.py', HERE/'extract_features.py', HERE/'pytorch_features/adapter.py',
        HERE/'pytorch_features/pt_interp.py', HERE/'pytorch_dispatch/dispatch_trace.json',
        HERE/'pytorch_dispatch/src/aten_src_ATen_cuda_detail_IntegerDivider.cuh']
    output = dict(schema='static_dynamic_barrier_phases/1', input_sha256={str(p.relative_to(X.REPO)): sha(p) for p in paths},
        assumptions=['Aligned synthetic pointer bases.', 'Uniform phase signatures at sampled blocks, checked against separately frozen aggregate sectors for retained kernels.',
                     'PyTorch dispatch and fast-path assumptions from the committed adapter.',
                     'Copy TensorIterator keeps the traced two dimensions and argument order.',
                     'Last-writer register chains omit shared-memory and cross-lane def-use edges; executed loop trips charge separate chain segments.',
                     'Unknown non-control instruction guards contribute an explicit upper bound.'],
        transaction_profiler_validation=False, rows=rows)
    (HERE / 'phases_blackwell.json').write_text(json.dumps(output, indent=1, sort_keys=True) + '\n')
    print(collections.Counter(r['status'] for r in rows.values()))


if __name__ == '__main__': main()
