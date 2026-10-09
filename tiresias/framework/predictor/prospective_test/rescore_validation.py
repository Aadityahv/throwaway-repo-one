"""Rescore the 12 validation cells with the adopted model v3k (RETROSPECTIVE for v3k: the cells were frozen and measured for the shared-traffic model v3j; they were known when the rules were found).
Writes scores_validation_v3k.csv. CPU only.   python3 rescore_validation.py"""
import csv, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import prosp_common as C
from cal import traffic as T
import predict as P
import predict_runtime_v3k as K3
lab = {r['cell_id']: r for r in json.loads((C.ST / 'validation/scores_validation_cells.json').read_text())}
fz = json.loads((C.ST / 'validation/PREDICTIONS_FROZEN.json').read_text())['rows']
rows = []; docs = {}
for s in C.iter_sets(('validation',)):
    new = K3.predict_portable(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188); d = docs.setdefault(s['docp'], P.load_calibration(s['docp'], allow_incomplete=True))
    for r in s['feats']['rows']:
        c = r['cell_id']; v = lab[c]; rt = new[c].get('primary_s'); tr = new[c].get('traffic')
        es = T.energy_rows_traffic(d, {'rows': [r]}, {c: rt} if rt else {}, {c: tr} if tr else {})[c]; em = T.energy_rows_traffic(d, {'rows': [r]}, {c: v['rt_meas']}, {c: tr} if tr else {})[c]
        rows.append(dict(cell_id=c, kernel=v['kernel'], rt_meas=v['rt_meas'], rt_v3j=v['rt_new'], rt_v3k=rt, e_meas=v['E'], e_static_v3j=v['e_new_static'], e_static_v3k=es.get('energy_j'), e_meas_v3j=v['e_new_meas'], e_meas_v3k=em.get('energy_j'), power=v['power']))
assert len(rows) == 12
for r in rows: assert abs(r['rt_v3j'] / fz[r['cell_id']]['candidate_runtime_s'] - 1) < 1e-12 and abs(r['e_meas_v3j'] / r['e_meas_v3k'] - 1) < 1e-9, r['cell_id']
with open(HERE / 'scores_validation_v3k.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r['cell_id']))
ape = lambda p, m: abs(p / m - 1) * 100
for tag in ('v3j', 'v3k'):
    print(tag, 'runtime median %.1f p90 %.1f; energy static median %.1f; energy measured runtime median %.1f' % (np.median([ape(r['rt_' + tag], r['rt_meas']) for r in rows]), np.percentile([ape(r['rt_' + tag], r['rt_meas']) for r in rows], 90),
          np.median([ape(r['e_static_' + tag], r['e_meas']) for r in rows]), np.median([ape(r['e_meas_' + tag], r['e_meas']) for r in rows])))
