"""Round-C predictions for v2, v3e and v3f. No runtime or energy label is read. Cells the static pipeline refuses get a null prediction and a reason."""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR))
import predict_runtime_v2 as V2, predict_runtime_v3 as V3, predict_runtime_v3f as V3F
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    features = json.loads((HERE / 'features_fresh_c.json').read_text()); phases = json.loads((HERE / 'phases_fresh_c.json').read_text())['rows']
    uniq = json.loads((HERE / 'phases_unique_fresh_c.json').read_text()); uniq = uniq.get('rows', uniq)
    C = SR / 'constants'; stream = json.loads((C / 'stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((C / 'microbench_constants_v2.json').read_text()))
    v3 = json.loads((C / 'v3_constants.json').read_text()); v3c = json.loads((C / 'v3c_constants.json').read_text()); v3e = json.loads((C / 'v3e_constants.json').read_text())
    preds = {'v2': V2.build(features, phases, K), 'v3e': V3.build(features, phases, uniq, K, v3, v3c, True, v3e), 'v3f': V3F.build(features, phases, uniq, K, v3, v3c, v3e)}
    for m, p in preds.items(): (HERE / f'predictions_fresh_c_{m}.json').write_text(json.dumps(p, indent=1, sort_keys=True) + '\n')
    inputs = ['features_fresh_c.json', 'phases_fresh_c.json', 'phases_unique_fresh_c.json', 'fresh_cells_c.json']
    freeze = dict(schema='fresh_prediction_freeze/1', frozen_before_timing=True, inputs_sha256={n: sha(HERE / n) for n in inputs},
        code_sha256={n: sha(SR / n) for n in ['predict_runtime_v2.py', 'predict_runtime_v3.py', 'predict_runtime_v3f.py']} | {'predict_fresh_c.py': sha(__file__)},
        constants_sha256={n: sha(C / n) for n in ['stream_constants.json', 'microbench_constants_v2.json', 'v3_constants.json', 'v3c_constants.json', 'v3e_constants.json']},
        outputs_sha256={f'predictions_fresh_c_{m}.json': sha(HERE / f'predictions_fresh_c_{m}.json') for m in preds},
        supported={m: sum(v['primary_s'] is not None for v in p.values()) for m, p in preds.items()}, primary_model='v3f',
        note='Written and committed before any round-C cell was timed. No runtime or energy value was read.')
    (HERE / 'PREDICTION_FREEZE_FRESH_C.json').write_text(json.dumps(freeze, indent=1, sort_keys=True) + '\n'); print(json.dumps(freeze['supported']))
if __name__ == '__main__': main()
