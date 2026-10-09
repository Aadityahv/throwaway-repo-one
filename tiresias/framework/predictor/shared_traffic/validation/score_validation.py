"""Score the 12 frozen validation cells against the measured runtime and board energy (raw in raw/). CPU only.
Previous model = the frozen `current_*` fields (the model the registry held before M488); new model = the frozen `candidate_*` fields (the current model since M488). Energy with measured runtime is
computed here from the measured runtime with each model's columns (previous: logical-byte columns; new: traffic columns), same calibration documents.  python3 score_validation.py > SCORES.md"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ST = HERE.parent; SR = ST.parent
sys.path.insert(0, str(ST)); sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate'))
import evaluate as EV, predict as P
from cal import traffic as T
fz = json.loads((HERE / 'PREDICTIONS_FROZEN.json').read_text())['rows']
rt = {}; ok = {}
for f in ('f', 'g', 'h', 'e', 'u'):
    for c in json.loads((HERE / 'raw' / ('timing_%s.json' % f)).read_text())['cells']: rt[c['cell_id']] = c['per_launch_runtime_s']; ok[c['cell_id']] = c.get('correct')
E_, W_, status = {}, {}, {}
for d in sorted((HERE / 'raw').glob('energy_*/application_energy_raw.csv')):
    for r in csv.DictReader(open(d)):
        cid = 'blackwell/%s/%s/%s' % (r['parent_id'], r['regime'], r['candidate_id']); e = float(r['board_energy_j_per_launch'])
        E_[cid] = e; W_[cid] = e / (float(r['counted_launch_interval_s']) / int(r['launch_count']))
tabs = {}; docs = {}
for lib in ('f', 'e', 'classic', 'g', 'h'):
    for c, v in json.loads((HERE / 'tables' / ('%s.json' % lib)).read_text()).items(): tabs[c] = (v['features'], EV.TENSOR_DOC if lib in ('g', 'h') else EV.RUN_DOC)
doc = lambda p: docs.setdefault(p, P.load_calibration(p, allow_incomplete=True))
ape = lambda p, m: abs(p / m - 1) * 100 if p else float('inf')
rows = []
for cid, r in sorted(fz.items()):
    if cid not in rt or cid not in E_: rows.append(dict(cell_id=cid, missing=True)); continue
    feat, dp = tabs[cid]; d = doc(dp); m = rt[cid]; E = E_[cid]; tr = r['candidate_traffic_bytes']
    old_meas = P.energy_rows(d, {'rows': [feat]}, {cid: m})[cid]; old_meas = old_meas['energy_j'] if old_meas['status'] == 'ok' else None
    tr_full = dict(l2_read_bytes=tr['l2_served'], l2_write_bytes=0, dram_read_bytes=tr['dram'], dram_write_bytes=0) if tr else None
    new_meas = T.energy_rows_traffic(d, {'rows': [feat]}, {cid: m}, {cid: tr_full} if tr_full else {})[cid]; new_meas = new_meas['energy_j'] if new_meas['status'] == 'ok' else None
    k = cid.split('/')[1].replace('validation_', '')
    rows.append(dict(cell_id=cid, kernel=k, tier=r['tier'], power=W_[cid], rt_meas=m, rt_old=r['current_runtime_s'], rt_new=r['candidate_runtime_s'], E=E, e_old_static=r['current_energy_static_j'], e_new_static=r['candidate_energy_static_j'],
                     e_old_meas=old_meas, e_new_meas=new_meas, correct=ok[cid]))
good = [r for r in rows if not r.get('missing')]
f = lambda x: 'fail' if x == float('inf') else '%.1f' % x
print('# Validation scores (12 frozen cells, Blackwell GPU 1, 3 October 2026)\n\nAbsolute percentage error, previous model | new (current) model. "fail" = refused or unsupported. Power in W from the 15 s window.\n')
print('| Cell | tier | power W | runtime measured us | runtime err | energy static err | energy measured-runtime err |\n|---|---|---:|---:|---|---|---|')
for r in good:
    print('| %s | %s | %.0f | %.1f | %s | %s | %s |' % (r['cell_id'][10:], r['tier'], r['power'], r['rt_meas'] * 1e6, ' / '.join([f(ape(r['rt_old'], r['rt_meas'])), f(ape(r['rt_new'], r['rt_meas']))]), ' / '.join([f(ape(r['e_old_static'], r['E'])), f(ape(r['e_new_static'], r['E']))]), ' / '.join([f(ape(r['e_old_meas'], r['E'])), f(ape(r['e_new_meas'], r['E']))])))
print('\n| Kernel | cells | runtime median (prev / new) | energy static median | energy measured-runtime median | runtime p90 | energy static p90 | energy measured p90 |\n|---|---:|---|---|---|---|---|---|')
groups = {}
for r in good: groups.setdefault(r['kernel'], []).append(r)
groups['ALL'] = good; groups['below 570 W'] = [r for r in good if r['power'] < 570]
for k, g in groups.items():
    if not g: continue
    def st(a, b, m, s): 
        x = [ape(r[a], r[m]) for r in g]; y = [ape(r[b], r[m]) for r in g]; fn = np.median if s == 'med' else (lambda v: np.percentile(v, 90)); return '%s / %s' % (f(fn(x)), f(fn(y)))
    print('| %s | %d | %s | %s | %s | %s | %s | %s |' % (k, len(g), st('rt_old', 'rt_new', 'rt_meas', 'med'), st('e_old_static', 'e_new_static', 'E', 'med'), st('e_old_meas', 'e_new_meas', 'E', 'med'), st('rt_old', 'rt_new', 'rt_meas', 'p90'), st('e_old_static', 'e_new_static', 'E', 'p90'), st('e_old_meas', 'e_new_meas', 'E', 'p90')))
json.dump(rows, open(HERE / 'scores_validation_cells.json', 'w'), indent=1, default=str)
