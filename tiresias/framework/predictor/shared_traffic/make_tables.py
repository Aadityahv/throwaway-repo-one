"""Markdown tables of RESULT.md from scores.json (written by evaluate.py). CPU only.  python3 make_tables.py > tables.md"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'scores.json').read_text())
f = lambda x: '-' if x is None else ('fail' if x != x or x == float('inf') else '%.1f' % x)
ORDER = ['Predicted before measurement (72)', 'Seen during model building (59)', 'ALL 131']
SETN = [k for k in S if k.startswith('set: ')]; KERN = [k for k in S if k.startswith('kernel: ')]


def cell(d, key, stat):
    a, b, c = d[key + '/cur'], d[key + '/new'], d[key + '/L']
    return '%s → %s [%s]' % (f(a[stat]) if a else '-', f(b[stat]) if b else '-', f(c[stat]) if c else '-')


def table(keys, header):
    out = ['| %s | cells | runtime median | runtime p90 | energy, static runtime: median | p90 | below 570 W median | energy, measured runtime: median | p90 | below 570 W median |' % header, '|---|---:|---|---|---|---|---|---|---|---|']
    for k in keys:
        d = S[k]; name = k.replace('set: ', '').replace('kernel: ', '')
        out.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |' % (name, d['cells'], cell(d, 'runtime', 'median'), cell(d, 'runtime', 'p90'), cell(d, 'energy_static', 'median'), cell(d, 'energy_static', 'p90'),
                   cell(d, 'energy_static_below_cap', 'median') if d['energy_static_below_cap/cur'] else '-', cell(d, 'energy_measured', 'median'), cell(d, 'energy_measured', 'p90'),
                   cell(d, 'energy_measured_below_cap', 'median') if d['energy_measured_below_cap/cur'] else '-'))
    return '\n'.join(out)


print('Format: current → candidate with the per-wave L2 rule [candidate with the whole-launch L2 rule]\n')
print('### Groups\n\n' + table(ORDER, 'Group'))
print('\n### Sets\n\n' + table(SETN, 'Set'))
print('\n### Kernels\n\n' + table(KERN, 'Kernel'))
# ablation
print('\n### Ablation (median absolute error, %)\n')
print('| Group | runtime: current | C1 | energy static runtime: current | C1 only | C2 only | C1+C2 | energy measured runtime: current | C2 only (= C1+C2) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for k in ORDER:
    d = S[k]; m = lambda key: f(d[key]['median'])
    print('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (k, m('runtime/cur'), m('runtime/A'), m('energy_static/cur'), m('energy_static/A'), m('energy_static/B'), m('energy_static/new'), m('energy_measured/cur'), m('energy_measured/new')))
print('\n(energy below 570 W, same variants)\n')
print('| Group | cells below cap | static: current | C1 only | C2 only | C1+C2 | measured: current | C1+C2 |\n|---|---:|---:|---:|---:|---:|---:|---:|')
for k in ORDER:
    d = S[k]; m = lambda key: f(d[key]['median']) if d[key] else '-'
    print('| %s | %d | %s | %s | %s | %s | %s | %s |' % (k, d['below_cap_cells'], m('energy_static_below_cap/cur'), m('energy_static_below_cap/A'), m('energy_static_below_cap/B'), m('energy_static_below_cap/new'), m('energy_measured_below_cap/cur'), m('energy_measured_below_cap/new')))
# regressions (>2 points, median per kernel, every metric)
print('\n### Kernels whose median error rises by more than 2 points\n')
print('| Kernel | metric | current | candidate | change |\n|---|---|---:|---:|---:|')
n = 0
for k in KERN:
    d = S[k]
    for m, lab in (('runtime', 'runtime'), ('energy_static', 'energy, static runtime'), ('energy_measured', 'energy, measured runtime'), ('energy_static_below_cap', 'energy, static runtime, below 570 W'), ('energy_measured_below_cap', 'energy, measured runtime, below 570 W')):
        if not d.get(m + '/cur'): continue
        a, b = d[m + '/cur']['median'], d[m + '/new']['median']
        if b - a > 2: print('| %s | %s | %.1f | %.1f | %+.1f |' % (k.replace('kernel: ', ''), lab, a, b, b - a)); n += 1
if not n: print('| (none) | | | | |')
