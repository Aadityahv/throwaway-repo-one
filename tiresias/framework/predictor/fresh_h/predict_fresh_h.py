"""Set-G predictions (fused attention) of the CURRENT runtime model (bank-conflict cost plus partial phase overlap, frozen constants) with the tensor instruction class: HMMA is issued on the tensor
pipe at the cost measured by the tensor stage (`constants_tensor/microbench_constants_v2.json` = the committed Blackwell constants plus `tensor_mma`), through calibrate/cal/portable_predict.py. No runtime
or energy label is read. Cells the static pipeline refuses get a null prediction and a reason."""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate'))
from cal import portable_predict as PP  # noqa: E402
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    features = json.loads((HERE / 'features_fresh_h.json').read_text()); phases = json.loads((HERE / 'phases_fresh_h.json').read_text())['rows']
    uniq = json.loads((HERE / 'phases_unique_fresh_h.json').read_text()); uniq = uniq.get('rows', uniq); bank = json.loads((HERE / 'bank_conflicts_fresh_h.json').read_text())['rows']
    C = PP.load_constants(HERE.parent / 'fresh_g' / 'constants_tensor'); C['smem'] = None      # frozen model: no shared-memory re-cost, as in the earlier set tables
    out = PP.predict(features, phases, uniq, bank, C, features['hardware_from_ground_truth']['sm_count'])
    (HERE / 'predictions_fresh_h_v3i.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    micro = json.loads((HERE / '../fresh_g/constants_tensor/microbench_constants_v2.json').read_text())
    freeze = dict(schema='fresh_prediction_freeze/1', frozen_before_timing=True, primary_model='v3i with the tensor_mma issue class', tensor_mma_issue_cycles=micro['issue_cycles_per_warp_instruction_per_sm'].get('tensor_mma'),
                  inputs_sha256={n: sha(HERE / n) for n in ['features_fresh_h.json', 'phases_fresh_h.json', 'phases_unique_fresh_h.json', 'fresh_cells_h.json', 'bank_conflicts_fresh_h.json', '../fresh_g/constants_tensor/microbench_constants_v2.json']},
                  code_sha256={'predict_fresh_h.py': sha(__file__), 'cells_h.py': sha(HERE / 'cells_h.py'), 'fresh_h_lib.py': sha(HERE / 'fresh_h_lib.py'), 'calibrate/cal/portable_predict.py': sha(SR / 'calibrate/cal/portable_predict.py')},
                  outputs_sha256={'predictions_fresh_h_v3i.json': sha(HERE / 'predictions_fresh_h_v3i.json')}, supported=sum(v['primary_s'] is not None for v in out.values()), cells=len(out),
                  note='Written and committed before any set-H cell was timed. No runtime or energy value was read.')
    (HERE / 'PREDICTION_FREEZE_FRESH_H.json').write_text(json.dumps(freeze, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(supported=freeze['supported'], cells=freeze['cells'])))
if __name__ == '__main__': main()
