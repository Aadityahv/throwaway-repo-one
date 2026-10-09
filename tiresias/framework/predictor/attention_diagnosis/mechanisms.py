"""Mechanism tests on all 143 cells (131 evaluation + 12 validation; 4 refused count as failures): each candidate changes ONE composition rule of the current model and uses only constants
already measured by a microbenchmark (list below). No constant is fitted on any cell. python3 mechanisms.py > mechanisms.txt   (also writes mechanisms.json)
Constants used:
  beta_hmma(w), beta_ffma(w): raw/dep_frag.jsonl (generic dependent fragment-load microbenchmark, run 3), median over shapes at each resident-warps-per-SM value
  service latency (loaded) L2 0.3054 us, DRAM 0.6506 us (constants/v3c_constants.json), pointer-chase latency L2 134 ns, DRAM 311 ns (predict_runtime_v2.LATENCY_NS, already in the model)."""
import json, statistics, math
import numpy as np
from mech_lib import *

def tables():
    rows = [json.loads(l) for l in open(HERE / 'raw/dep_frag.jsonl')]
    out = {}
    for cons, shapes in (('hmma', None), ('ffma', [(8, 8)])):
        by = {}
        for r in rows:
            if r['consumer'] != cons: continue
            if shapes and (r['TM'], r['TN']) not in shapes: continue
            by.setdefault(r['warps_per_sm'], []).append(r['beta'])
        out[cons] = {w: statistics.median(v) for w, v in sorted(by.items())}
    return out
BETA = tables()
def beta(cons, w):
    xs = sorted(BETA[cons]); ys = [BETA[cons][x] for x in xs]
    return float(np.interp(w, xs, ys))
SERVICE = {'L2': 0.3054185e-6, 'DRAM': 0.6505897e-6}
LAT = V2.LATENCY_NS

def issue_with_beta(k, p, ffma=False):
    P = dict(p['pipes_s']); sh = P.get('shared_load', 0.0); tc = P.get('tensor_mma', 0.0); ff = P.get('fp32_fma', 0.0)
    w = k['resident_warps']; others = {n: v for n, v in P.items()}
    if tc > 0 and sh > 0:
        x = max(sh, tc) + beta('hmma', w) * min(sh, tc); others.pop('shared_load'); others.pop('tensor_mma'); others['pair'] = x
    elif ffma and ff > 0 and sh > 0:
        x = max(sh, ff) + beta('ffma', w) * min(sh, ff); others.pop('shared_load'); others.pop('fp32_fma'); others['pair'] = x
    return max(p['disp_s'], max(others.values(), default=0.0)), others

class Mech:
    def __init__(self, beta_tc=False, beta_ffma=False, stage=False, lat_add=False, lat_service=False, store_add=False, tail=False):
        self.__dict__.update(locals())
    def tailf(self, k):
        # per-SM pipe work is set by the most loaded SM: ceil(blocks / SMs) blocks, not the average blocks / SMs (blocks > SMs only; fewer blocks than SMs are already handled by the active-SM fraction)
        if not self.tail: return 1.0
        b = k['grid_blocks']; sm = 188
        return 1.0 if b <= sm else math.ceil(b / sm) / (b / sm)
    def issue(self, k, p):
        if self.beta_tc or self.beta_ffma:
            iss, others = issue_with_beta(k, p, ffma=self.beta_ffma)
            if self.stage and p['kind'] == 'compute_tensor' and 'pair' in others:
                rest = max([v for n, v in others.items() if n != 'pair'] + [0.0])
                iss = max(p['disp_s'], others['pair'] + rest)
            return iss * self.tailf(k)
        return p['t']['issue_s'] * self.tailf(k)
    def phase_s(self, k, p):
        t = p['t']; iss = self.issue(k, p)
        if p['kind'] == 'load' and (self.lat_add or self.lat_service or self.store_add) and k['barrier_releases'] > 0:
            if self.store_add: base = max(t['stream_s'], t['latency_s']) + iss
            else: base = max(t['stream_s'], iss)
            if self.lat_add: base += t['latency_s']
            if self.lat_service: base += k['waves'] * p['depth'] * SERVICE[k['tier']]
            return base
        return max(t['stream_s'], t['latency_s'], iss)
    def full_terms(self, k, p):
        t = p['t']; iss = self.issue(k, p)
        return {'stream': t['stream_s'], 'latency': t['latency_s'], 'issue': iss}

def run(cells, mech):
    return [compose(c, mech) if c['pred_primary_s'] is not None else None for c in cells]
ape = lambda p, m: abs(p / m - 1) * 100 if p else float('inf')

def summarize(cells, preds, label, base=None):
    out = {}
    def st(sel):
        v = [ape(p, c['measured_s']) for c, p in zip(cells, preds) if sel(c)]
        return (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if v else None
    out['all131'] = st(lambda c: not c['validation']); out['val12'] = st(lambda c: c['validation'])
    out['predicted72'] = None
    kernels = sorted(set(c['kernel'] for c in cells))
    out['kernels'] = {k: st(lambda c, k=k: c['kernel'] == k and not c['validation']) for k in kernels}
    out['kernels_val'] = {k: st(lambda c, k=k: c['kernel'] == k and c['validation']) for k in kernels}
    return out

if __name__ == '__main__':
    cells = load_cells()
    print('beta tables', json.dumps(BETA))
    CAND = {
        'baseline': Mech(),
        'pair overlap coefficient (shared load and MMA)': Mech(beta_tc=True),
        'pair overlap coefficient, also FP32 FMA pair': Mech(beta_tc=True, beta_ffma=True),
        'pair overlap + stage serialisation (stage rule)': Mech(beta_tc=True, stage=True),
        'load latency added to load phases': Mech(lat_add=True),
        'loaded service latency added to load phases': Mech(lat_service=True),
        'load phases: transfer and store issue additive': Mech(store_add=True),
        'pair overlap + load latency': Mech(beta_tc=True, lat_add=True),
        'stage rule + load latency': Mech(beta_tc=True, stage=True, lat_add=True),
        'stage rule + additive store issue': Mech(beta_tc=True, stage=True, store_add=True),
        'stage rule + additive store issue + load latency': Mech(beta_tc=True, stage=True, store_add=True, lat_add=True),
        'busiest-SM rule (issue terms)': Mech(tail=True),
        'stage rule + busiest-SM rule': Mech(beta_tc=True, stage=True, tail=True),
    }
    res = {}; base = None
    for name, m in CAND.items():
        preds = run(cells, m); res[name] = (preds, summarize(cells, preds, name))
    json.dump({n: dict(summary=r[1]) for n, r in res.items()}, open(HERE / 'mechanisms.json', 'w'), indent=1)
    b = res['baseline'][1]
    print('\n%-62s %17s %17s' % ('mechanism (runtime error, median/p90)', 'all 131', '12 validation'))
    for n, (preds, s) in res.items():
        print('%-62s %6.1f/%6.1f (%d) %6.1f/%6.1f' % (n, s['all131'][0], s['all131'][1], s['all131'][2], s['val12'][0], s['val12'][1]))
    print('\nper-kernel median runtime error, evaluation cells (baseline | mechanism); * = rises more than 2 points')
    keys = [k for k in b['kernels'] if b['kernels'][k]]
    print('%-40s %6s' % ('kernel', 'n') + ''.join('%14s' % n[:13] for n in res))
    for k in keys:
        print('%-40s %6d' % (k, b['kernels'][k][2]) + ''.join('%13.1f%s' % (r[1]['kernels'][k][0], '*' if r[1]['kernels'][k][0] > b['kernels'][k][0] + 2 else ' ') for n, r in res.items()))
    print('\nper-cell ratio predicted/measured, tensor-core matmul and attention cells (evaluation + validation)')
    sel = [i for i, c in enumerate(cells) if c['kernel'] in ('Tensor-core matrix multiply', 'Fused attention', 'tensor_matmul', 'fused_attention') or 'tensor_matmul' in c['cid'] or 'fused_attention' in c['cid']]
    print('%-52s %-5s' % ('cell', 'tier') + ''.join('%10s' % n[:9] for n in res))
    for i in sel:
        c = cells[i]; print('%-52s %-5s' % (c['cid'].split('/', 1)[1], c['tier']) + ''.join('%10s' % ('refused' if r[0][i] is None else '%.2f' % (r[0][i] / c['measured_s'])) for n, r in res.items()))
