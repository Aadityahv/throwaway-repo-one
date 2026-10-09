"""Development check (run before the candidate is scored on the sets predicted before measurement): the Blackwell development corpus (132 cells, the cells the runtime model and its constants
were built on), current model versus candidate, runtime and energy. Labels: compile_evidence/development/development_cells.csv (runtime_s, energy_j, measured_power_w), read only.
Cells the current model does not predict (no static table) are listed and excluded from both columns. CPU only.  python3 dev_check.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import evaluate as EV

rows = [r for r in csv.DictReader(open(EV.PW / 'framework/compile_evidence/development/development_cells.csv')) if r['gpu'] == 'blackwell']
feats = {r['cell_id']: r for r in json.loads((EV.SR / 'features_blackwell.json').read_text())['rows']}
EV.DEV_IDS.update(r['cell_id'] for r in rows if r['cell_id'] in feats)
pred = EV.predict_all([EV.DEV])
labels = [dict(cell_id=r['cell_id'], group='Development', set='development corpus', kernel=r['operator_id'], tier=r['tier'], runtime_measured_s=r['runtime_s'], energy_measured_j=r['energy_j'],
               window_power_w=r['measured_power_w']) for r in rows if r['cell_id'] in pred]
scored = EV.score_cells(labels, pred)
ok = [r for r in scored if r['rt_cur']]
print('development cells with a current-model prediction: %d of %d' % (len(ok), len(scored)))
refused = [(r['cell_id'], r['new_reason']) for r in ok if r['rt_new'] is None]
print('newly refused by the candidate:', refused)
agg = EV.aggregate(ok)
out = {k: v for k, v in agg.items() if k == 'ALL 131' or k.startswith('kernel: ')}
out['ALL'] = out.pop('ALL 131')
json.dump(dict(cells_scored=len(ok), cells_total=len(scored), newly_refused=refused, aggregate=out), open(HERE / 'dev_check.json', 'w'), indent=1, sort_keys=True)
changed = [r for r in ok if r['rt_new'] != r['rt_cur'] or r['e_new_static'] != r['e_cur_static']]
print('cells whose prediction changed: runtime', sum(r['rt_new'] != r['rt_cur'] for r in ok), ' energy static', sum(r['e_new_static'] != r['e_cur_static'] for r in ok))
for name, d in out.items():
    print('%-48s n=%3d  runtime %5.1f -> %5.1f | energy(static rt) %5.1f -> %5.1f | energy(measured rt) %5.1f -> %5.1f' % (name[:48], d['cells'], d['runtime/cur']['median'], d['runtime/new']['median'],
          d['energy_static/cur']['median'], d['energy_static/new']['median'], d['energy_measured/cur']['median'], d['energy_measured/new']['median']))
