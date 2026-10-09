"""Fresh-cell predictions for v2 and the frozen v3d (both use the corrected IMAD.HI.U32 phase table). No runtime or energy label is read."""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR))
import predict_runtime_v2 as V2
import predict_runtime_v3 as V3
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    features = json.loads((HERE / 'features_fresh_b.json').read_text()); phases = json.loads((HERE / 'phases_fresh_b.json').read_text())['rows']
    uniq = json.loads((HERE / 'phases_unique_fresh_b.json').read_text()); uniq = uniq.get('rows', uniq)
    stream = json.loads((SR / 'constants/stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((SR / 'constants/microbench_constants_v2.json').read_text()))
    v3c = json.loads((SR / 'constants/v3c_constants.json').read_text()); v3 = json.loads((SR / 'constants/v3_constants.json').read_text())
    v3e = json.loads((SR / 'constants/v3e_constants.json').read_text())
    p2 = V2.build(features, phases, K); p3 = V3.build(features, phases, uniq, K, v3, v3c, True); p4 = V3.build(features, phases, uniq, K, v3, v3c, True, v3e)
    (HERE / 'predictions_fresh_b_v2.json').write_text(json.dumps(p2, indent=1, sort_keys=True) + '\n')
    (HERE / 'predictions_fresh_b_v3d.json').write_text(json.dumps(p3, indent=1, sort_keys=True) + '\n')
    (HERE / 'predictions_fresh_b_v3e.json').write_text(json.dumps(p4, indent=1, sort_keys=True) + '\n')
    inputs = ['features_fresh_b.json', 'phases_fresh_b.json', 'phases_unique_fresh_b.json', 'fresh_cells_b.json']
    freeze = dict(schema='fresh_prediction_freeze/1', frozen_before_timing=True,
        inputs_sha256={n: sha(HERE / n) for n in inputs},
        code_sha256={'predict_runtime_v2.py': sha(SR / 'predict_runtime_v2.py'), 'predict_runtime_v3.py': sha(SR / 'predict_runtime_v3.py'), 'predict_fresh_b.py': sha(__file__)},
        constants_sha256={n: sha(SR / 'constants' / n) for n in ['stream_constants.json', 'microbench_constants_v2.json', 'v3_constants.json', 'v3c_constants.json', 'v3e_constants.json']},
        outputs_sha256={n: sha(HERE / n) for n in ['predictions_fresh_b_v2.json', 'predictions_fresh_b_v3d.json', 'predictions_fresh_b_v3e.json']},
        supported={'v2': sum(v['primary_s'] is not None for v in p2.values()), 'v3d': sum(v['primary_s'] is not None for v in p3.values()), 'v3e': sum(v['primary_s'] is not None for v in p4.values())},
        note='Written and committed before any fresh cell was timed. No runtime or energy value was read.')
    (HERE / 'PREDICTION_FREEZE_FRESH_B.json').write_text(json.dumps(freeze, indent=1, sort_keys=True) + '\n'); print(json.dumps(freeze['supported']))
if __name__ == '__main__': main()
