"""Development check, second part: the 59 evaluation cells seen during model building (classic kernels, CUDA samples at new shapes), scored before the 72 cells predicted before measurement.
python3 seen_check.py > seen_check.txt"""
import csv, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import evaluate as EV
labels = [r for r in csv.DictReader(open(EV.PW / 'evaluation/results/eval_cells.csv')) if r['group'].startswith('Seen')]
pred = EV.predict_all([s for s in EV.SETS if s[0] in ('classic', 'd')])
rows = EV.score_cells(labels, pred); EV.check_reproduction(labels, rows)
agg = EV.aggregate(rows); print('cells', len(rows), 'refused by candidate:', [(r['cell_id'], r['new_reason']) for r in rows if r['rt_new'] is None])
for k in ['Seen during model building (59)'] + [k for k in agg if k.startswith(('set:', 'kernel:'))]:
    d = agg[k]; print('%-52s n=%2d runtime %5.1f -> %5.1f | energy(static) %5.1f -> %5.1f | energy(measured) %5.1f -> %5.1f' % (k[:52], d['cells'], d['runtime/cur']['median'], d['runtime/new']['median'], d['energy_static/cur']['median'], d['energy_static/new']['median'], d['energy_measured/cur']['median'], d['energy_measured/new']['median']))
