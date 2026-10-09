#!/usr/bin/env python3
"""Utility analysis of the energy prediction on the new cell sets: what decision does it support, and what does it cost? Written before any measured energy of
set E or the unseen cells was read. Estimation accuracy, energy saved by a decision, and measurement cost saved are three different claims and are reported separately.

Decision task (the one the cells support): for every (kernel, size) pair choose the lower-energy of the two candidates (c1, c2: grid/block size or input shape) WITHOUT executing
either. Metrics per set: top-1 accuracy of the choice; regret = E(chosen) / E(best) - 1 (median, mean, worst); energy saved against always taking c1; and, as an ablation of our own model
(not a baseline), the same choice made by predicted RUNTIME alone (fastest is assumed lowest-energy).
Cost task: calibration is a one-time cost per GPU; measuring the energy of one candidate directly costs `per_cell_energy_seconds`. Break-even number of candidate comparisons
= calibration_gpu_seconds / (2 * per_cell_energy_seconds) (two candidates per comparison). Both inputs are passed in and recorded; nothing is assumed.
Usage: score_utility.py --set e|unseen --energy-csv <application_energy_raw.csv> --timing <timing json> --calibration-seconds S --per-cell-energy-seconds S --out <json>
       score_utility.py --selftest"""
import argparse, collections, csv, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent; UK = SR / 'unseen_kernels'
sys.path.insert(0, str(UK))

def decide(E_meas, E_pred, T_pred, pairs):
    rows = []
    for key, (c1, c2) in sorted(pairs.items()):
        if not all(c in E_meas and E_pred.get(c) is not None for c in (c1, c2)): rows.append(dict(pair=key, status='not_scored')); continue
        best = min((c1, c2), key=lambda c: E_meas[c]); by_e = min((c1, c2), key=lambda c: E_pred[c])
        by_t = min((c1, c2), key=lambda c: T_pred[c]) if all(T_pred.get(c) is not None for c in (c1, c2)) else None
        r = lambda c: E_meas[c] / E_meas[best] - 1
        rows.append(dict(pair=key, status='scored', best=best, chosen_by_predicted_energy=by_e, correct=by_e == best, regret_pct=100 * r(by_e),
                         chosen_by_predicted_runtime=by_t, runtime_choice_correct=(by_t == best) if by_t else None, runtime_choice_regret_pct=100 * r(by_t) if by_t else None,
                         saving_vs_c1_pct=100 * (1 - E_meas[by_e] / E_meas[c1]), measured_gap_pct=100 * (max(E_meas[c1], E_meas[c2]) / E_meas[best] - 1)))
    sc = [r for r in rows if r['status'] == 'scored']
    q = lambda xs, f: float(f(xs)) if xs else None
    summ = dict(pairs=len(rows), scored=len(sc), top1_accuracy=q([r['correct'] for r in sc], np.mean), median_regret_pct=q([r['regret_pct'] for r in sc], np.median),
                mean_regret_pct=q([r['regret_pct'] for r in sc], np.mean), worst_regret_pct=q([r['regret_pct'] for r in sc], max),
                mean_saving_vs_c1_pct=q([r['saving_vs_c1_pct'] for r in sc], np.mean), median_measured_gap_pct=q([r['measured_gap_pct'] for r in sc], np.median),
                ablation_runtime_only_top1_accuracy=q([r['runtime_choice_correct'] for r in sc if r['runtime_choice_correct'] is not None], np.mean),
                ablation_runtime_only_median_regret_pct=q([r['runtime_choice_regret_pct'] for r in sc if r['runtime_choice_regret_pct'] is not None], np.median))
    return rows, summ

def pairs_of(cells):
    d = collections.defaultdict(dict)
    for cid, c in cells.items(): d[(c['operator_id'], c['regime'])][c['candidate_id']] = cid
    return {k: (v['c1'], v['c2']) for k, v in d.items() if 'c1' in v and 'c2' in v}

def selftest():
    cells = {f'x/{k}/{r}/{c}': dict(operator_id=k, regime=r, candidate_id=c) for k in ('a', 'b') for r in ('s', 'm') for c in ('c1', 'c2')}
    E = {c: 1.0 for c in cells}; E['x/a/s/c2'] = 0.5; E['x/b/m/c1'] = 2.0; Ep = dict(E); T = {c: E[c] for c in cells}
    rows, s = decide(E, Ep, T, pairs_of(cells)); assert s['top1_accuracy'] == 1.0 and s['median_regret_pct'] == 0.0 and s['scored'] == 4
    Ep['x/a/s/c1'] = 0.1; rows, s = decide(E, Ep, T, pairs_of(cells)); assert s['top1_accuracy'] == 0.75 and s['worst_regret_pct'] == 100.0
    print('selftest OK')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); ap.add_argument('--set', choices=['e', 'unseen']); ap.add_argument('--energy-csv', type=Path)
    ap.add_argument('--run-prefix'); ap.add_argument('--calibration-seconds', type=float); ap.add_argument('--per-cell-energy-seconds', type=float); ap.add_argument('--out', type=Path); a = ap.parse_args()
    if a.selftest: return selftest()
    if not (a.set and a.energy_csv and a.calibration_seconds and a.per_cell_energy_seconds and a.out): ap.error('all of --set --energy-csv --calibration-seconds --per-cell-energy-seconds --out are required')
    if a.out.exists(): raise SystemExit('REFUSED: %s exists; written once' % a.out)
    import score_unseen as SU
    if a.set == 'e':
        cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_e.json').read_text())['cells']}
        en = json.loads((HERE / 'energy_predictions_fresh_e.json').read_text())['cells']; pred = json.loads((HERE / 'predictions_fresh_e_v3i.json').read_text()); prefix = a.run_prefix or 'FRESHE-ENERGY'
    else:
        cells = {c['cell_id']: c for c in json.loads((UK / 'frozen/cells_unseen.json').read_text())['cells']}
        en = json.loads((SR / 'bank/energy_predictions_unseen_v3i.json').read_text())['cells']; pred = json.loads((SR / 'bank/predictions_unseen_v3i.json').read_text()); prefix = a.run_prefix or 'UNSEEN-ENERGY'
    E = {r['cell_id']: float(r['board_energy_j_per_launch']) for r in SU.load_energy_csv(a.energy_csv, prefix)}
    Ep = {c: (en[c].get('with_v3i_runtime') or {}).get('energy_j') for c in cells}; Tp = {c: pred[c].get('primary_s') for c in cells}
    rows, summ = decide(E, Ep, Tp, pairs_of(cells))
    cost = dict(calibration_gpu_seconds=a.calibration_seconds, per_cell_energy_seconds=a.per_cell_energy_seconds, breakeven_candidate_comparisons=a.calibration_seconds / (2 * a.per_cell_energy_seconds),
                note='A comparison of two candidates needs two energy measurements if done directly; the static model needs none after the one-time calibration. Does not include the per-kernel static analysis time (CPU).')
    rep = dict(schema='energy_utility_score/1', set=a.set, decision=summ, per_pair=rows, cost=cost, note='Three separate claims: estimation accuracy (score_energy_*.py), energy saved by the decision (this file), measurement cost saved (cost block).')
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n'); print(json.dumps(summ, indent=1))
if __name__ == '__main__': main()
