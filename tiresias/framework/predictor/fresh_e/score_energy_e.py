"""Score the frozen set-E energy predictions against measured energy. Run once, after the energy run. Uses the helpers of the frozen unseen-kernel scorer.
Criteria (the unseen-kernel test's, applied with the current runtime model; fixed before any set-E energy measurement):
 E1  on below-cap cells the decomposed energy model at PREDICTED runtime has a median error within 5 percentage points of the same model at MEASURED runtime;
 E2  on below-cap cells it has a lower median error than the frozen runtime-plus-traffic formula at predicted runtime.
Reported, not criteria: 90th percentile; per kernel and tier; all measured cells including capped ones; time-the-kernel at typical power and at the kernel's
measured medium/c1 power; signed errors. Unsupported or unmeasured cells count as failures in the coverage line; nothing is dropped."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; UK = HERE.parent / 'unseen_kernels'
sys.path.insert(0, str(UK)); sys.path.insert(0, str(HERE))
import energy_model as EM
import score_unseen as SU
CAP_FRACTION = 0.95
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timing', type=Path, required=True); ap.add_argument('--energy-csv', type=Path, required=True)
    ap.add_argument('--run-prefix', default='FRESHE-ENERGY'); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    if a.out.exists(): raise SystemExit('REFUSED: %s exists; the score is written once' % a.out)
    cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_e.json').read_text())['cells']}
    en = json.loads((HERE / 'energy_predictions_fresh_e.json').read_text()); base = json.loads((HERE / 'baseline_inputs_fresh_e.json').read_text()); model = en['model']
    pred = json.loads((HERE / 'predictions_fresh_e_v3i.json').read_text())
    t_meas = {c['cell_id']: float(c['per_launch_runtime_s']) for c in json.loads(a.timing.read_text())['cells'] if c.get('per_launch_runtime_s') and c.get('correct')}
    rows = SU.load_energy_csv(a.energy_csv, a.run_prefix); E = {r['cell_id']: float(r['board_energy_j_per_launch']) for r in rows}
    tw = {r['cell_id']: float(r['counted_launch_interval_s']) / float(r['launch_count']) for r in rows}
    ids = sorted(cells); fam = lambda c: cells[c]['operator_id'].replace('fresh_e_cuda_samples_', ''); P0, cap = model['P0_w'], model['cap_w']; eps = base['runtime_plus_traffic_eps_pJ_per_byte']
    below = [c for c in ids if c in E and E[c] / tw[c] < CAP_FRACTION * cap]
    anchors = {fam(c): E[c] / tw[c] for c in ids if cells[c]['regime'] == 'medium' and cells[c]['candidate_id'] == 'c1' and c in E}
    e_dec = lambda c, t: EM.predict(en['cells'][c]['terms'], t, model)['energy_j']
    e_rpt = lambda c, t: min(P0 * t + eps[cells[c]['tier']] * 1e-12 * cells[c]['logical_bytes_per_launch'], cap * t)
    rep = dict(schema='fresh_e_energy_score/1', cells_total=len(ids), cells_energy=len(E), cells_timed=len(t_meas), below_cap_cells=len(below), capped_cells=[c for c in ids if c in E and c not in below])
    def block(cset):
        ok = [c for c in cset if en['cells'][c].get('status') == 'ok' and c in t_meas and pred[c].get('primary_s')]
        S = lambda fn, cs=None: SU.stats([SU.pct(fn(c), E[c]) for c in (cs if cs is not None else ok)])
        anch = [c for c in ok if fam(c) in anchors and not (cells[c]['regime'] == 'medium' and cells[c]['candidate_id'] == 'c1')]
        o = dict(cells=len(ok), decomposed_at_predicted_runtime=S(lambda c: en['cells'][c]['with_v3i_runtime']['energy_j']), decomposed_at_measured_runtime=S(lambda c: e_dec(c, t_meas[c])),
                 runtime_plus_traffic_at_predicted_runtime=S(lambda c: e_rpt(c, pred[c]['primary_s'])), runtime_plus_traffic_at_measured_runtime=S(lambda c: e_rpt(c, t_meas[c])),
                 time_the_kernel_x_typical_power=S(lambda c: base['typical_power_w'] * t_meas[c]), time_the_kernel_x_anchor_power=S(lambda c: anchors[fam(c)] * t_meas[c], anch),
                 per_kernel={f: SU.stats([SU.pct(en['cells'][c]['with_v3i_runtime']['energy_j'], E[c]) for c in ok if fam(c) == f]) for f in sorted({fam(c) for c in ok})},
                 per_tier={t: SU.stats([SU.pct(en['cells'][c]['with_v3i_runtime']['energy_j'], E[c]) for c in ok if cells[c]['tier'] == t]) for t in ('L2', 'DRAM')},
                 per_cell={c: dict(measured_j=E[c], measured_power_w=round(E[c] / tw[c], 1), predicted_j=en['cells'][c]['with_v3i_runtime']['energy_j'],
                                   signed_error_pct=round((en['cells'][c]['with_v3i_runtime']['energy_j'] / E[c] - 1) * 100, 1)) for c in ok})
        return o
    rep['below_cap'] = block(below); rep['all_measured_cells'] = block([c for c in ids if c in E]); b = rep['below_cap']
    d, m, r = b['decomposed_at_predicted_runtime'], b['decomposed_at_measured_runtime'], b['runtime_plus_traffic_at_predicted_runtime']
    rep['E1_energy_vs_measured_runtime'] = dict(median_at_predicted=d and d['median_pct'], median_at_measured=m and m['median_pct'], pass_=bool(d and m and d['median_pct'] - m['median_pct'] <= 5))
    rep['E2_beats_runtime_plus_traffic'] = dict(decomposed=d and d['median_pct'], runtime_plus_traffic=r and r['median_pct'], pass_=bool(d and r and d['median_pct'] < r['median_pct']))
    rep['note'] = 'Estimation accuracy only. Energy saved by a decision and measurement cost saved are not established by these criteria.'
    rep['inputs_sha256'] = {'timing': SU.sha(a.timing), 'energy_csv': SU.sha(a.energy_csv), 'freeze': SU.sha(HERE / 'PREDICTION_FREEZE_ENERGY_FRESH_E.json')}
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: rep[k] for k in ('below_cap_cells', 'E1_energy_vs_measured_runtime', 'E2_beats_runtime_plus_traffic')}, indent=1))
if __name__ == '__main__': main()
