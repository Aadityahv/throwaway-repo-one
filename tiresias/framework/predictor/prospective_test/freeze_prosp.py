"""Freeze the predictions of BOTH models for the 24 prospective cells, before any of them is timed or measured. Reads only the static tables built by build_prosp.py and the committed constants and
calibration documents; no label of any kind exists for these cells. Writes PREDICTIONS_FROZEN.json (predictions + sha256 of every input and code file); refuses to overwrite an existing freeze.
  current  = the registry model (shared-traffic model v3j with per-wave L2 rule; component energy model with traffic columns)
  proposed = v3k (v3j + pair overlap, stage serialisation, busiest-SM rule; nothing else changes), same energy model fed its runtime and traffic
Fused-kernel cells whose static analysis or a model rule refuses are recorded as refused (they count as failures in every score).   python3 freeze_prosp.py"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import prosp_common as C
from cal import traffic as T
import predict as P, evaluate as EV
import predict_runtime_v3k as K3
SR = C.SR
OUT = HERE / 'PREDICTIONS_FROZEN.json'
if OUT.exists(): raise SystemExit('REFUSED: prospective predictions already frozen')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
s = next(C.iter_sets(('prosp',))); doc = P.load_calibration(s['docp'], allow_incomplete=True)
cur = T.predict_candidate(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188)
new = K3.predict_portable(s['feats'], s['ph'], s['un'], s['bk'], s['C'], 188)
rows = {}
for cid, why in s['refused'].items(): rows[cid] = dict(status='not_derivable', reason=why)
def pack(p, r):
    rt = p[r['cell_id']].get('primary_s'); tr = p[r['cell_id']].get('traffic')
    e = T.energy_rows_traffic(doc, {'rows': [r]}, {r['cell_id']: rt} if rt else {}, {r['cell_id']: tr} if tr else {})[r['cell_id']]
    return dict(runtime_s=rt, refusal=p[r['cell_id']].get('unsupported_reason'), energy_static_j=e.get('energy_j'), energy_status=e.get('status'),
                traffic_bytes=None if not tr else dict(l2_served=tr['l2_read_bytes'] + tr['l2_write_bytes'], dram=tr['dram_read_bytes'] + tr['dram_write_bytes'], l2_read=tr['l2_read_bytes'], l2_write=tr['l2_write_bytes'], dram_read=tr['dram_read_bytes'], dram_write=tr['dram_write_bytes']))
for r in s['feats']['rows']:
    cid = r['cell_id']; c = s['cells'][cid]
    rows[cid] = dict(status='ok', tier=r['memory']['tier'], family=c['family'], kernel_group=c['operator_id'], regime=c['regime'], candidate=c['candidate_id'], current=pack(cur, r), proposed=pack(new, r))
code = {str(p.relative_to(SR)): sha(p) for p in [SR / f for f in ('predict_runtime_v2.py', 'predict_runtime_v3.py', 'predict_runtime_v3f.py', 'predict_runtime_v3h.py', 'predict_runtime_v3i.py', 'predict_runtime_v3j.py', 'predict_runtime_v3k.py',
        'calibrate/cal/traffic.py', 'calibrate/cal/energy.py', 'calibrate/cal/portable_predict.py', 'calibrate/predict.py', 'shared_traffic/footprint.py', 'constants/pair_overlap_constants.json',
        'prospective_test/freeze_prosp.py', 'prospective_test/prosp_common.py', 'prospective_test/cells_prosp.py', 'prospective_test/prosp_lib.py')]}
inputs = {'tables/prosp.json': sha(HERE / 'tables/prosp.json'), 'constants:fresh_g/constants_tensor': {p.name: sha(p) for p in sorted((SR / 'fresh_g/constants_tensor').glob('*.json'))},
          'calibration:' + str(s['docp'].relative_to(SR)): sha(s['docp']), 'compiled/sass/prosp.sass': sha(HERE / 'compiled/sass/prosp.sass'), 'compiled/cubin/prosp.cubin': sha(HERE / 'compiled/cubin/prosp.cubin'),
          'src/prosp_kernels.cuh': sha(HERE / 'src/prosp_kernels.cuh'), 'gpu/driver_prosp.cu': sha(HERE / 'gpu/driver_prosp.cu'), 'cells_prosp.json': sha(HERE / 'cells_prosp.json')}
OUT.write_text(json.dumps(dict(schema='prospective_prediction_freeze/1', frozen_before_any_measurement=True, models=dict(current='registry current model (v3j per-wave rule, component energy model with traffic columns)', proposed='v3k = v3j + pair overlap, stage serialisation and busiest-SM rule (attention_diagnosis/DESIGN.md)'),
                               energy_with_measured_runtime='computed after timing with the same columns: energy_rows_traffic(doc, features, measured runtime, traffic)', rows=rows, inputs_sha256=inputs, code_sha256=code), indent=1, sort_keys=True) + '\n')
print('frozen', OUT.name, sha(OUT))
for cid, r in sorted(rows.items()):
    if r['status'] != 'ok': print(cid, r['status']); continue
    f = lambda x: 'refused' if x['runtime_s'] is None else '%.1f us' % (x['runtime_s'] * 1e6)
    print('%-62s %-4s current %-12s proposed %-12s' % (cid[10:], r['tier'], f(r['current']), f(r['proposed'])))
