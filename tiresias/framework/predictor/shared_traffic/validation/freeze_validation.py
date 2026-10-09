"""Freeze the candidate's (and, for reference, the current model's) runtime and energy predictions for the 12 prospective validation cells, before any of them is timed or measured.
Reads only the static tables built by build_validation.py and the committed constants and calibration documents; no label of any kind exists for these cells.
Writes PREDICTIONS_FROZEN.json (predictions + sha256 of every input, code file and constant file). Refuses to overwrite an existing freeze.  python3 freeze_validation.py"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ST = HERE.parent; SR = ST.parent
sys.path.insert(0, str(ST)); sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate'))
import evaluate as EV
import predict as P
from cal import portable_predict as PP, traffic as T
import predict_runtime_v3j as J
OUT = HERE / 'PREDICTIONS_FROZEN.json'
if OUT.exists(): raise SystemExit('REFUSED: validation predictions already frozen')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
LIBS = {'f': (SR / 'constants', EV.RUN_DOC), 'e': (SR / 'constants', EV.RUN_DOC), 'classic': (SR / 'constants', EV.RUN_DOC), 'g': (SR / 'fresh_g/constants_tensor', EV.TENSOR_DOC), 'h': (SR / 'fresh_g/constants_tensor', EV.TENSOR_DOC)}
L2 = EV.l2_capacity_bytes(); rows = {}; inputs = {}
for lib, (cd, docp) in LIBS.items():
    tp = HERE / 'tables' / ('%s.json' % lib); inputs['validation/tables/%s.json' % lib] = sha(tp); t = json.loads(tp.read_text())
    ok = {c: v for c, v in t.items() if 'refused' not in v}
    for c, v in t.items():
        if 'refused' in v: rows[c] = dict(status='not_derivable', reason=v['refused'])
    feats = {'hardware_from_ground_truth': json.loads((SR / 'fresh_g/features_fresh_g.json').read_text())['hardware_from_ground_truth'], 'rows': [v['features'] for v in ok.values()]}; ph = {c: v['phases'] for c, v in ok.items()}; un = {c: v['unique'] for c, v in ok.items()}; bk = {c: v['bank'] for c, v in ok.items()}
    fps = {c: v['footprints'] for c, v in ok.items()}; un_fp = J.attach_footprints(un, fps, L2)
    C = PP.load_constants(cd); C['smem'] = None; sm = 188; doc = P.load_calibration(docp, allow_incomplete=True)
    cur = PP.predict(feats, ph, un, bk, C, sm); new = T.predict_candidate(feats, ph, un_fp, bk, C, sm, read_footprint=True)
    for r in feats['rows']:
        cid = r['cell_id']; rt = new[cid].get('primary_s'); tr = new[cid].get('traffic')
        e_new = T.energy_rows_traffic(doc, {'rows': [r]}, {cid: rt} if rt else {}, {cid: tr} if tr else {})[cid]
        rc = cur[cid].get('primary_s'); e_cur = P.energy_rows(doc, {'rows': [r]}, {cid: rc} if rc else {})[cid]
        rows[cid] = dict(status='ok' if rt else 'refused', tier=r['memory']['tier'], kernel_family=ok[cid]['cell']['family'], candidate_runtime_s=rt, candidate_refusal=new[cid].get('unsupported_reason'),
                         candidate_energy_static_j=e_new.get('energy_j'), candidate_traffic_bytes=None if not tr else dict(l2_served=tr['l2_read_bytes'] + tr['l2_write_bytes'], dram=tr['dram_read_bytes'] + tr['dram_write_bytes']),
                         current_runtime_s=rc, current_energy_static_j=e_cur.get('energy_j') if e_cur['status'] == 'ok' else None, energy_document=str(docp.relative_to(SR)))
    inputs['constants:%s' % cd.relative_to(SR)] = {p.name: sha(p) for p in sorted(Path(cd).glob('*.json'))}; inputs['calibration:%s' % docp.relative_to(SR)] = sha(docp)
code = {str(p.relative_to(SR)): sha(p) for p in [SR / f for f in ('predict_runtime_v2.py', 'predict_runtime_v3.py', 'predict_runtime_v3f.py', 'predict_runtime_v3h.py', 'predict_runtime_v3i.py', 'predict_runtime_v3j.py', 'calibrate/cal/traffic.py',
        'calibrate/cal/energy.py', 'calibrate/cal/portable_predict.py', 'calibrate/predict.py', 'shared_traffic/footprint.py', 'shared_traffic/validation/freeze_validation.py')]}
OUT.write_text(json.dumps(dict(schema='validation_prediction_freeze/1', frozen_before_any_measurement=True, model='registry current model (bank-conflict and overlap runtime model, component energy model: current_* fields) and the candidate (v3j, per-wave L2 rule, traffic energy columns: candidate_* fields); the candidate was not adopted, both frozen',
                               energy_with_measured_runtime='computed after timing with the same columns: energy_rows_traffic(doc, features, measured runtime, traffic)', rows=rows, inputs_sha256=inputs, code_sha256=code), indent=1, sort_keys=True) + '\n')
print(json.dumps({c: (r['status'], r.get('candidate_runtime_s'), r.get('candidate_energy_static_j')) for c, r in sorted(rows.items())}, indent=0))
