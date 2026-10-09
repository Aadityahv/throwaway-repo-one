#!/usr/bin/env python3
"""Write data/application_cells_blackwell.csv: the labeled Blackwell application cells (120 exposed development cells and the 48 target cells) in the generic format of
fit_application_model.py. Read-only on the repo's committed measurements; runtime_pred_s is the frozen runtime model's prediction (v3i for the target cells, v3e for development)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent; SR = HERE.parent; U = SR.parent
feats = {}
for p in ('fresh_e/features_fresh_e.json', 'unseen_kernels/frozen/features_unseen.json', 'features_blackwell.json'):
    for r in json.loads((SR / p).read_text())['rows']:
        if r['status'] in ('supported', 'supported_with_assumptions'): feats[r['cell_id']] = r
rt = {}
for p in ('fresh_e/predictions_fresh_e_v3i.json', 'bank/predictions_unseen_v3i.json'): rt.update({k: v['primary_s'] for k, v in json.loads((SR / p).read_text()).items()})
v3e = json.loads((SR / 'predictions_v3e_dev_blackwell.json').read_text())
rows = []
for r in csv.DictReader((U / 'evaluation_data/measured/kernel_sets/per_cell.csv').open()):
    if r['role'] != 'ablation' or r['cell_id'] not in feats: continue
    rows.append([r['cell_id'], r['family'], r['measured_energy_j'], r['measured_short_runtime_s'], rt[r['cell_id']], r['measured_window_power_w'], 'target'])
for r in csv.DictReader((U / 'compile_evidence/development/development_cells.csv').open()):
    if r['gpu'] != 'blackwell' or r['cell_id'] not in feats: continue
    pred = (v3e.get(r['cell_id']) or {}).get('primary_s')
    rows.append([r['cell_id'], r['operator_group'].split('/')[-1], r['energy_j'], r['runtime_s'], pred if pred else '', r['measured_power_w'], 'dev'])
out = HERE / 'data' / 'application_cells_blackwell.csv'
with out.open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['cell_id', 'operator', 'energy_j', 'runtime_s', 'runtime_pred_s', 'window_power_w', 'source']); w.writerows(rows)
print('wrote', out, len(rows), 'cells')
