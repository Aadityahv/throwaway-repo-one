"""Set-E predictions for the earlier runtime models and the current model (bank-conflict cost plus partial phase overlap). No runtime or energy label is
read. Cells the static pipeline refuses get a null prediction and a reason."""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR))
import predict_runtime_v2 as V2, predict_runtime_v3 as V3, predict_runtime_v3f as V3F, predict_runtime_v3h as V3H, predict_runtime_v3i as V3I
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    features = json.loads((HERE / 'features_fresh_e.json').read_text()); phases = json.loads((HERE / 'phases_fresh_e.json').read_text())['rows']
    uniq = json.loads((HERE / 'phases_unique_fresh_e.json').read_text()); uniq = uniq.get('rows', uniq)
    bank = json.loads((SR / 'bank' / 'bank_conflicts_fresh_e.json').read_text())['rows']
    C = SR / 'constants'; stream = json.loads((C / 'stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((C / 'microbench_constants_v2.json').read_text()))
    v3 = json.loads((C / 'v3_constants.json').read_text()); v3c = json.loads((C / 'v3c_constants.json').read_text()); v3e = json.loads((C / 'v3e_constants.json').read_text())
    OV = json.loads((C / 'overlap_constants.json').read_text())
    preds = {'v2': V2.build(features, phases, K), 'v3e': V3.build(features, phases, uniq, K, v3, v3c, True, v3e), 'v3f': V3F.build(features, phases, uniq, K, v3, v3c, v3e),
             'v3h': V3H.build(features, phases, uniq, bank, K, v3, v3c, v3e), 'v3i': V3I.build(features, phases, uniq, bank, K, v3, v3c, v3e, OV)}
    for m, p in preds.items(): (HERE / f'predictions_fresh_e_{m}.json').write_text(json.dumps(p, indent=1, sort_keys=True) + '\n')
    inputs = ['features_fresh_e.json', 'phases_fresh_e.json', 'phases_unique_fresh_e.json', 'fresh_cells_e.json', '../bank/bank_conflicts_fresh_e.json']
    freeze = dict(schema='fresh_prediction_freeze/1', frozen_before_timing=True, inputs_sha256={n: sha(HERE / n) for n in inputs},
        code_sha256={n: sha(SR / n) for n in ['predict_runtime_v2.py', 'predict_runtime_v3.py', 'predict_runtime_v3f.py', 'predict_runtime_v3h.py', 'predict_runtime_v3i.py', 'bank/bank_conflicts.py', 'bank/bank_unseen.py']}
            | {'predict_fresh_e.py': sha(__file__), 'fresh_e_lib.py': sha(HERE / 'fresh_e_lib.py'), 'cells_e.py': sha(HERE / 'cells_e.py')},
        constants_sha256={n: sha(C / n) for n in ['stream_constants.json', 'microbench_constants_v2.json', 'v3_constants.json', 'v3c_constants.json', 'v3e_constants.json', 'overlap_constants.json']},
        compiled_sha256={n: sha(HERE / n) for n in ['compiled/sass/scalarProd.sass', 'compiled/sass/fastWalshTransform.sass', 'compiled/cubin/scalarProd.cubin', 'compiled/cubin/fastWalshTransform.cubin']},
        outputs_sha256={f'predictions_fresh_e_{m}.json': sha(HERE / f'predictions_fresh_e_{m}.json') for m in preds},
        supported={m: sum(v['primary_s'] is not None for v in p.values()) for m, p in preds.items()}, primary_model='v3i',
        note='Written and committed before any set-E cell was timed. No runtime or energy value was read.')
    (HERE / 'PREDICTION_FREEZE_FRESH_E.json').write_text(json.dumps(freeze, indent=1, sort_keys=True) + '\n'); print(json.dumps(freeze['supported']))
if __name__ == '__main__': main()
