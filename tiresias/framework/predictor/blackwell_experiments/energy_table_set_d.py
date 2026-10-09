"""Energy of the 27 set-D cells scored against the frozen predictions (predictions_set_d.json, committed before this file was run). CPU only.
Label: accepted application_energy_raw rows, window energy per launch (no idle subtraction); window power = energy / (counted interval / launches). Below cap: window power under 570 W (95% of 600 W).
Rejected or unmeasured cells would count as failures; the one cell whose correctness check failed in timing is listed as not measured."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(HERE))
import runtime_table as RT  # noqa: E402
pred = json.loads((SR.parent / 'evaluation_data/measured/kernel_sets/predictions_set_d.json').read_text()); rows = pred['rows']
E, W = {}, {}
for r in csv.DictReader(open(SR / 'fresh_d/energy_raw_d/application_energy_raw.csv')):
    cid = 'blackwell/%s/%s/%s' % (r['parent_id'], r['regime'], r['candidate_id'])
    if cid in E: raise SystemExit('duplicate ' + cid)
    e = float(r['board_energy_j_per_launch']); t = float(r['counted_launch_interval_s']) / int(r['launch_count']); E[cid] = e; W[cid] = e / t
    if r['correctness_check'] not in ('True', 'true', '1', 'ok', 'CHECK_OK') and not str(r['correctness_check']).lower().startswith(('true', 'check_ok')): print('NOTE correctness_check', cid, r['correctness_check'])
ids = sorted(rows); missing = [c for c in ids if c not in E]
dc = {c['cell_id']: c for c in json.loads((SR / 'fresh_d/fresh_cells_d.json').read_text())['cells']}
grp = {c: RT.group_d(dc[c]) for c in ids}; below = np.array([W.get(c, 0) < 570 for c in ids])
Ev = np.array([E.get(c, np.nan) for c in ids])
M = [('Ours (calibrator only)', 'ours'), ('AccelWattch-style', 'component'), ('FlipFlop', 'flipflop'), ('Alavani', 'alavani')]
out = dict(cells=len(ids), measured=len(ids) - len(missing), missing=missing, not_measured_correctness_failed=pred['correctness_failed_not_predicted'], below_cap=int(below.sum()), tables={})
def ape(key): return np.array([abs(rows[c]['energy_j'][key] / E[c] - 1) * 100 if rows[c]['energy_j'].get(key) and c in E else np.inf for c in ids])
for lab, suffix in (('static runtime', 'predicted'), ('measured runtime', 'measured')):
    t = {}
    for name, m in M:
        a = ape('%s_%s' % (m, suffix)); d = {g: float(np.median(a[[grp[c] == g for c in ids]])) for g in sorted(set(grp.values()))}
        d['ALL %d' % len(ids)] = float(np.median(a)); d['below cap (%d)' % below.sum()] = float(np.median(a[below])); d['p90 all'] = float(np.percentile(a, 90)); d['p90 below cap'] = float(np.percentile(a[below], 90))
        d['signed median all'] = float(np.median([(rows[c]['energy_j']['%s_%s' % (m, suffix)] / E[c] - 1) * 100 for c in ids])); t[name] = d
    out['tables'][lab] = t
# runtime-free published rows (as published)
for name, key in (('FlipFlop as published (own time model)', 'flipflop_static'), ('Delestrac per-level energies', 'delestrac'), ("O'Connor / Keckler constants", 'literature')):
    a = ape(key); out['tables'].setdefault('no runtime input', {})[name] = dict(median_all=float(np.median(a)), median_below_cap=float(np.median(a[below])), p90_all=float(np.percentile(a, 90)))
(HERE / 'energy_table_set_d.json').write_text(json.dumps(out, indent=1))
print('measured %d of %d (missing %s); below cap %d; window power range %.0f-%.0f W' % (out['measured'], len(ids), missing, below.sum(), min(W.values()), max(W.values())))
for lab, t in out['tables'].items():
    print('\n' + lab); keys = list(next(iter(t.values())))
    print('%-46s' % 'kernel' + ''.join('%-26s' % n for n in t))
    for k in keys: print('%-46s' % k + ''.join('%-26s' % ('%6.1f%%' % t[n][k]) for n in t))
