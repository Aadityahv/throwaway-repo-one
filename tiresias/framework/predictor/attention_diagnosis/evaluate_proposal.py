"""Before/after scores of the proposed rules on all 131 evaluation cells and the 12 validation cells (RETROSPECTIVE: every one of these cells was known when the rules were found).
Variants: current (the registry model, unchanged), stage rule (partial concurrency of the shared-load and MMA pipes with the measured coefficient, and the rest of the phase serialised after the MMA stage),
busiest-SM rule (per-SM pipe work set by ceil(blocks / SMs) blocks), both.   python3 evaluate_proposal.py > proposal_scores.txt   (writes proposal_cells.csv, proposal_scores.json)"""
import csv, json
import numpy as np
from mech_lib import *
from mechanisms import Mech, ape
cells = load_cells()
lab = {r['cell_id']: r for r in csv.DictReader(open(ST / 'cells_scored.csv'))}
VARS = {'current': Mech(), 'stage': Mech(beta_tc=True, stage=True), 'busiest': Mech(tail=True), 'both': Mech(beta_tc=True, stage=True, tail=True)}
rows = []
for c in cells:
    r = dict(cell_id=c['cid'], validation=c['validation'], kernel=c['kernel'], tier=c['tier'], measured_s=c['measured_s'], energy_j=c['energy_j'],
             group=lab[c['cid']]['group'] if c['cid'] in lab else 'validation (prospective for the current model, retrospective for these rules)',
             power=float(lab[c['cid']]['window_power_w']) if c['cid'] in lab else None)
    for v, m in VARS.items():
        rt = compose(c, m) if c['pred_primary_s'] is not None else None
        r['rt_' + v] = rt; r['e_' + v] = energy_static(c, rt) if rt else None
    rows.append(r)
with open(HERE / 'proposal_cells.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def st(sel, key, v):
    x = [ape(r[key + '_' + v], r['measured_s'] if key == 'rt' else r['energy_j']) for r in rows if sel(r)]
    return (float(np.median(x)), float(np.percentile(x, 90)), int(np.isinf(x).sum()), len(x)) if x else None
E_VAL = lambda r: r['validation']
groups = {'ALL 131 evaluation cells': lambda r: not r['validation'], 'predicted before measurement (72)': lambda r: (not r['validation']) and r['group'].startswith('Predicted'),
          'seen during model building (59)': lambda r: (not r['validation']) and r['group'].startswith('Seen'), 'evaluation cells below 570 W': lambda r: (not r['validation']) and r['power'] < 570,
          '12 validation cells': E_VAL}
out = {}
print('Runtime error (median / p90 / refused counted as failures), energy with static runtime (median / p90). RETROSPECTIVE for all cells shown.\n')
print('| Group | n | current runtime | stage rule | busiest-SM rule | both | current energy | stage rule | busiest-SM rule | both |\n|---|---:|---|---|---|---|---|---|---|---|')
for g_, sel in groups.items():
    rt = {v: st(sel, 'rt', v) for v in VARS}; en = {v: st(sel, 'e', v) for v in VARS}
    out[g_] = dict(runtime=rt, energy=en)
    f = lambda s: '%.1f / %.1f%s' % (s[0], s[1], (' (%d refused)' % s[2]) if s[2] else '')
    print('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |' % ((g_, rt['current'][3]) + tuple(f(rt[v]) for v in VARS) + tuple(f(en[v]) for v in VARS)))
kernels = sorted(set(r['kernel'] for r in rows))
print('\nPer-kernel median error, evaluation cells (runtime | energy with static runtime); regression = median rises by more than 2 points (marked R)\n')
print('| Kernel | n | runtime current | stage rule | busiest-SM rule | both | energy current | stage rule | busiest-SM rule | both |\n|---|---:|---|---|---|---|---|---|---|---|')
regress = []
for k in kernels:
    sel = lambda r, k=k: (not r['validation']) and r['kernel'] == k
    rt = {v: st(sel, 'rt', v) for v in VARS}; en = {v: st(sel, 'e', v) for v in VARS}
    if rt['current'] is None: continue
    cell = lambda s, b: '%.1f%s' % (s[0], ' R' if s[0] > b[0] + 2 else '')
    for v in ('stage', 'busiest', 'both'):
        if rt[v][0] > rt['current'][0] + 2: regress.append((k, 'runtime', v, rt['current'][0], rt[v][0]))
        if en[v] and en['current'] and en[v][0] > en['current'][0] + 2: regress.append((k, 'energy_static', v, en['current'][0], en[v][0]))
    print('| %s | %d | %.1f | %s | %s | %s | %.1f | %s | %s | %s |' % (k, rt['current'][3], rt['current'][0], cell(rt['stage'], rt['current']), cell(rt['busiest'], rt['current']), cell(rt['both'], rt['current']), en['current'][0], cell(en['stage'], en['current']), cell(en['busiest'], en['current']), cell(en['both'], en['current'])))
    out.setdefault('kernels', {})[k] = dict(runtime=rt, energy=en)
print('\nKernel-median regressions above 2 points:', regress if regress else 'none')
print('\nValidation cells (runtime error %: current | A | B | A+B ; energy static error %)\n\n| Cell | measured us | current | stage rule | busiest-SM rule | both | energy current | both |\n|---|---:|---|---|---|---|---|---|')
for r in rows:
    if not r['validation']: continue
    print('| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |' % (r['cell_id'][10:], r['measured_s'] * 1e6, *(ape(r['rt_' + v], r['measured_s']) for v in VARS), ape(r['e_current'], r['energy_j']), ape(r['e_both'], r['energy_j'])))
print('\nCells whose runtime error rises by more than 2 points under both rules (evaluation and validation):')
n = 0
for r in rows:
    a, b = ape(r['rt_current'], r['measured_s']), ape(r['rt_both'], r['measured_s'])
    if b > a + 2 and not np.isinf(a): n += 1; print('  %-70s %5.1f -> %5.1f' % (r['cell_id'], a, b))
print('  (%d cells)' % n)
json.dump(out, open(HERE / 'proposal_scores.json', 'w'), indent=1, default=str)
