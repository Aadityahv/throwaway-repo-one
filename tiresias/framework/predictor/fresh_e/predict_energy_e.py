"""Energy predictions and baseline inputs for fresh set E, frozen before any set-E energy measurement. No energy or power value of any set-E cell is read.

Energy: the frozen decomposed energy model (unseen_kernels/energy_model.py, rates in energy_model_frozen.json, never refitted) evaluated at the runtime
predicted by the current runtime model (predict_runtime_v3i.py, predictions frozen in 855ea27a) and, for reference, at the bank-conflict model's runtime.
Baseline inputs follow unseen_kernels/baselines_unseen.py (memory-plus-compute roofline, typical development power, the runtime-plus-traffic formula)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent; UK = SR / 'unseen_kernels'
sys.path.insert(0, str(UK))
import energy_model as EM
import baselines_unseen as BU
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    feats = json.loads((HERE / 'features_fresh_e.json').read_text()); cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_e.json').read_text())['cells']}
    pred = {m: json.loads((HERE / f'predictions_fresh_e_{m}.json').read_text()) for m in ('v3h', 'v3i')}
    model = json.loads(EM.FROZEN.read_text()); en = {}
    for f in feats['rows']:
        cid = f['cell_id']; c = cells[cid]; t = f.get('per_launch_totals')
        if f['status'] == 'missing_features' or not t: en[cid] = dict(status='unsupported', reason='static features unavailable', energy_j=None); continue
        terms = EM.terms_from_totals(t, c['logical_bytes_per_launch'], c['tier'])
        e = dict(status='ok', terms=terms, counts_exact=bool(t['all_counts_exact']))
        for m in ('v3h', 'v3i'):
            rt = pred[m][cid].get('primary_s')
            if rt: e['with_%s_runtime' % m] = EM.predict(terms, rt, model); e['%s_runtime_s' % m] = rt
        en[cid] = e
    S = json.loads((SR / 'constants/stream_constants.json').read_text())['constants']; sm = feats['hardware_from_ground_truth']['sm_count']
    peak = sm * BU.FP32_LANES_PER_SM * BU.SM_CLOCK_HZ; bw = {'L2': S['L2_read_sector_TBps'] * 1e12, 'DRAM': S['DRAM_read_TBps'] * 1e12}; roof = {}
    for f in feats['rows']:
        cid = f['cell_id']; c = cells[cid]; t = f.get('per_launch_totals')
        if not t: roof[cid] = dict(roofline_s=None); continue
        fp32 = sum(n for op, n in t['opcode_lane_counts'].items() if op.split('.')[0] in ('FFMA', 'FADD', 'FMUL'))
        mem_s = c['logical_bytes_per_launch'] / bw[c['tier']]; comp_s = fp32 / peak
        roof[cid] = dict(roofline_s=S['t0_us'] * 1e-6 * t['kernels_per_launch'] + max(mem_s, comp_s), memory_s=mem_s, compute_s=comp_s, fp32_lane_instructions=fp32)
    tp, n = BU.typical_power()
    base = dict(schema='fresh_e_baseline_inputs/1', frozen_before_energy=True, typical_power_w=tp, typical_power_cells=n,
                runtime_plus_traffic_eps_pJ_per_byte={'L2': 80.642125766766, 'DRAM': 179.59310363418655}, runtime_plus_traffic_P0_w=152.52913482177462, cap_w=600.0, roofline=roof)
    (HERE / 'energy_predictions_fresh_e.json').write_text(json.dumps(dict(model=model, cells=en), indent=1, sort_keys=True) + '\n')
    (HERE / 'baseline_inputs_fresh_e.json').write_text(json.dumps(base, indent=1, sort_keys=True) + '\n')
    freeze = dict(schema='fresh_e_energy_freeze/1', frozen_before_energy_measurement=True, runtime_already_timed=True,
        inputs_sha256={n: sha(HERE / n) for n in ('features_fresh_e.json', 'fresh_cells_e.json', 'PREDICTION_FREEZE_FRESH_E.json', 'predictions_fresh_e_v3i.json', 'predictions_fresh_e_v3h.json')},
        code_sha256={'predict_energy_e.py': sha(__file__), 'energy_model.py': sha(UK / 'energy_model.py'), 'baselines_unseen.py': sha(UK / 'baselines_unseen.py')}, energy_model_sha256=sha(EM.FROZEN),
        outputs_sha256={n: sha(HERE / n) for n in ('energy_predictions_fresh_e.json', 'baseline_inputs_fresh_e.json')}, supported=sum(v.get('status') == 'ok' for v in en.values()),
        note='Written and committed before any set-E energy measurement. The set-E runtime was timed before this file (the runtime freeze 855ea27a precedes the timing); no energy or power value of any set-E cell was read.')
    (HERE / 'PREDICTION_FREEZE_ENERGY_FRESH_E.json').write_text(json.dumps(freeze, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(supported=freeze['supported'], typical_power_w=tp)))
if __name__ == '__main__': main()
