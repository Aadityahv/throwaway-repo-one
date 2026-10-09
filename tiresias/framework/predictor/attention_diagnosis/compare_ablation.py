"""Compare the ablation measurements (raw/ablation.jsonl) with the model's own per-phase-category terms. python3 compare_ablation.py > ablation_vs_model.txt
Categories of the model's phase list: load phase (has global loads), compute phase (has HMMA), epilogue (the last phase). Model category time = sum over its phases of max(stream, latency, issue)
(the serial leg of the model; the overlap leg only reduces it). Marginal cost from the ablation: compute = full - nocomp (what removing compute saves), load = full - noload."""
import json
from common import *
A = load_all()
abl = {}
for l in open(HERE / 'raw/ablation.jsonl'):
    r = json.loads(l); abl.setdefault(r['cell'], {})[r['variant']] = r['us_median']
def cat(x):
    cats = {'load': 0.0, 'compute': 0.0, 'epilogue': 0.0, 'lat_load': 0.0}
    k = x['pred']['kernels'][0]; phs = x['ph']['kernels'][0]['phases']; n = len(phs)
    for i, (t, ph) in enumerate(zip(k['phase_terms'], phs)):
        v = max(t['stream_s'], t['latency_s'], t['issue_s']) * t['repetitions'] * 1e6
        ops = ph['issue_warp_instructions']
        kind = 'compute' if any(o.startswith('HMMA') for o in ops) else ('load' if any(o.startswith('LDG') for o in ops) else 'epilogue')
        cats[kind] += v
        if kind == 'load': cats['lat_load'] += t['latency_s'] * t['repetitions'] * 1e6
    cats['barrier'] = k['barrier_s'] * 1e6; cats['launch'] = k['launch_s'] * 1e6
    return cats
print('%-34s %7s %7s | %8s %8s %8s %7s | %8s %8s | %6s %6s' % ('cell', 'meas', 'pred', 'm.load', 'm.comp', 'm.epi', 'm.bar', 'F-NL', 'F-NC', 'ld x', 'cmp x'))
rows = []
for c, x in sorted(A.items()):
    if x['pred'].get('primary_s') is None or x['set'].startswith('validation'): continue
    key = c.split('/')[-3].replace('fresh_g_ml_tensor_matmul', 'x') + '/' + c.split('/')[-2] + '/' + c.split('/')[-1]
    name = ('tc_gemm' if 'tensor' in c else 'attn_fwd') ; cell = c.split('/')[-2] + '/' + c.split('/')[-1]
    a = abl.get(name + '|' + cell) if False else abl.get(cell)
    # ablation cells are keyed 'regime/cand' per kernel: separate by kernel
    pass
abl2 = {}
for l in open(HERE / 'raw/ablation.jsonl'):
    r = json.loads(l); abl2.setdefault((r['kernel'], r['cell']), {})[r['variant']] = r['us_median']
for c, x in sorted(A.items()):
    if x['pred'].get('primary_s') is None or x['set'].startswith('validation'): continue
    kern = 'tc_gemm' if 'tensor' in c else 'attn_fwd'; cell = '/'.join(c.split('/')[-2:]); a = abl2.get((kern, cell))
    if not a: continue
    cs = cat(x); mload, mcomp = a['full'] - a['noload'], a['full'] - a['nocomp']
    pred = x['pred']['primary_s'] * 1e6; meas = x['measured_s'] * 1e6
    print('%-34s %7.1f %7.1f | %8.1f %8.1f %8.1f %7.1f | %8.1f %8.1f | %6.2f %6.2f' % (kern + ' ' + cell, meas, pred, cs['load'], cs['compute'], cs['epilogue'], cs['barrier'], mload, mcomp, mload / cs['load'], mcomp / cs['compute']))
    rows.append(dict(kernel=kern, cell=cell, meas=meas, pred=pred, ablation=a, model=cs))
json.dump(rows, open(HERE / 'ablation_vs_model.json', 'w'), indent=1)
