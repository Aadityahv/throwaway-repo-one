"""Energy table of the evaluation (CPU only): median absolute percentage error per kernel on the 48 measured target cells (unseen kernels and the scalar-product / Walsh set), calibrator-only profile of the
repeat calibration run (as-designed admission), against the AccelWattch-style, FlipFlop and Alavani rows, with static and with measured runtime, all cells. Set D (27 cells) has no energy labels: its energy was never measured."""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; SR = HERE.parent; U = SR.parent
sys.path.insert(0, str(SR / 'calibrate')); sys.path.insert(0, str(HERE))
import predict as P  # noqa: E402
import cpu_experiments as X  # noqa: E402
doc = P.load_calibration(SR / 'calibrate/runs/energy_repeat_20261002/run_full/calibration_sm_120_0e63baea.json', allow_incomplete=True)
feats, tgt, rts, _, _ = X.load_cells(); pc = pd.DataFrame(list(csv.DictReader((U / 'evaluation_data/measured/kernel_sets/per_cell.csv').open())))
for c in ('measured_energy_j', 'predicted_energy_j', 'measured_window_power_w', 'measured_short_runtime_s'): pc[c] = pc[c].astype(float)
ids = list(tgt.cell_id); E = tgt.measured_energy_j.values; fam = tgt.family.values; below = (tgt.measured_window_power_w < 570).values
tm = dict(zip(ids, tgt.measured_short_runtime_s)); out = {}
def ours(rt):
    r = P.energy_rows(doc, {'rows': [feats[c] for c in ids]}, rt); return np.array([r[c]['energy_j'] for c in ids])
def base(m, key):
    s = pc[pc.method == m % key].set_index('cell_id').predicted_energy_j; return np.array([s[c] for c in ids])
meths = {'Ours (calibrator only)': None, 'AccelWattch-style': 'AccelWattch-style component model with %s runtime', 'FlipFlop': 'FlipFlop power with %s runtime', 'Alavani': 'Alavani static-feature power with %s runtime'}
for lab, rt, key in (('static runtime', rts, 'our predicted'), ('measured runtime', tm, 'measured')):
    out[lab] = {}
    for name, m in meths.items():
        p = ours(rt) if m is None else base(m, key); err = np.abs(p / E - 1) * 100
        d = {f: float(np.median(err[fam == f])) for f in sorted(set(fam))}; d['ALL 48'] = float(np.median(err)); d['below cap (%d)' % below.sum()] = float(np.median(err[below])); out[lab][name] = d
print('median absolute percentage error, all cells of each kernel (8 cells per kernel; 6 for none below cap shown separately)')
for lab in out:
    print('\n' + lab); keys = list(next(iter(out[lab].values())))
    print('%-28s' % 'kernel' + ''.join('%-24s' % n for n in meths))
    for k in keys: print('%-28s' % k + ''.join('%-24s' % ('%5.1f%%' % out[lab][n][k]) for n in meths))
(HERE / 'energy_table.json').write_text(json.dumps(out, indent=1))
