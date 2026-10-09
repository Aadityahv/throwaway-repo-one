"""Pre-registered criterion 2 (DESIGN.md addendum): with the implemented predictor file v3k, no kernel median runtime error of the 131 evaluation cells is up by more than 2 points against the current
model (v3j). Also writes the 131-cell predictions of v3k (runtime, energy with static runtime) to scores_131_v3k.csv for the rescoring. CPU only.   python3 check_criterion2.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import prosp_common as C
from cal import traffic as T
import predict as P
import predict_runtime_v3k as K3
lab = {r['cell_id']: r for r in csv.DictReader(open(C.ST / 'cells_scored.csv'))}
ape = lambda p, m: abs(p / m - 1) * 100 if p else float('inf')
rows = []; docs = {}
for s in C.iter_sets(('eval',)):
    cur = T.predict_candidate(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188); new = K3.predict_portable(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188)
    d = docs.setdefault(s['docp'], P.load_calibration(s['docp'], allow_incomplete=True))
    for r in s['feats']['rows']:
        c = r['cell_id']
        if c not in lab: continue
        L = lab[c]; m = float(L['runtime_measured_s']); out = dict(cell_id=c, kernel=L['kernel'], group=L['group'], tier=L['tier'], rt_meas=m, e_meas=float(L['energy_measured_j']), power=float(L['window_power_w']))
        for tag, p in (('cur', cur), ('v3k', new)):
            rt = p[c].get('primary_s'); tr = p[c].get('traffic')
            es = T.energy_rows_traffic(d, {'rows': [r]}, {c: rt} if rt else {}, {c: tr} if tr else {})[c]; em = T.energy_rows_traffic(d, {'rows': [r]}, {c: m}, {c: tr} if tr else {})[c]
            out.update({'rt_' + tag: rt, 'e_static_' + tag: es.get('energy_j'), 'e_meas_' + tag: em.get('energy_j')})
        rows.append(out)
assert len(rows) == 131, len(rows)
for r in rows:   # consistency with the committed scoring of the current model
    assert abs((r['rt_cur'] or 0) - float(lab[r['cell_id']]['rt_new'] or 0)) <= 1e-12 * max(1, r['rt_cur'] or 1), r['cell_id']
with open(HERE / 'scores_131_v3k.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
med = lambda xs: float(np.median(xs)); res = {}
print('| Kernel | n | current median | v3k median | change |\n|---|---:|---:|---:|---:|'); worst = -1e9
for k in sorted(set(r['kernel'] for r in rows)):
    g = [r for r in rows if r['kernel'] == k]; a = med([ape(r['rt_cur'], r['rt_meas']) for r in g]); b = med([ape(r['rt_v3k'], r['rt_meas']) for r in g]); worst = max(worst, b - a)
    res[k] = (a, b); print('| %s | %d | %.1f | %.1f | %+.1f%s |' % (k, len(g), a, b, b - a, ' **>2**' if b - a > 2 else ''))
A = med([ape(r['rt_cur'], r['rt_meas']) for r in rows]); B = med([ape(r['rt_v3k'], r['rt_meas']) for r in rows])
print('\nAll 131: current %.1f%%, v3k %.1f%% (refused cells count as failures). Largest kernel-median increase: %+.2f points. Criterion 2: %s' % (A, B, worst, 'PASS' if worst <= 2 else 'FAIL'))
json.dump(dict(kernels=res, all131=[A, B], worst_increase=worst, criterion2_pass=worst <= 2), open(HERE / 'criterion2.json', 'w'), indent=1)
