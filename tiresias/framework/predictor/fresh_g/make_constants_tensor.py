"""constants_tensor/: the committed Blackwell runtime constants (../constants) with ONE addition, the tensor instruction class measured by the packaged tensor stage (calibrate/runs/tensor_20261003):
microbench_constants_v2.json gets `tensor_mma` in `issue_cycles_per_warp_instruction_per_sm` and `dependent_latency_cycles`. Every other constant is byte-identical to the committed file."""
import json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
out = HERE / 'constants_tensor'; out.mkdir(exist_ok=True)
for p in (SR / 'constants').glob('*.json'): shutil.copy(p, out / p.name)
t = json.loads((HERE / 'calibration_with_tensor.json').read_text())['constants']['tensor']['issue']
m = json.loads((SR / 'constants/microbench_constants_v2.json').read_text())
m['issue_cycles_per_warp_instruction_per_sm']['tensor_mma'] = t['issue_cycles_per_warp_instruction_per_sm']; m['dependent_latency_cycles']['tensor_mma'] = t['dependent_latency_cycles']
(out / 'microbench_constants_v2.json').write_text(json.dumps(m, indent=1, sort_keys=True) + '\n'); print('tensor_mma issue %.3f cycles, latency %.2f cycles' % (t['issue_cycles_per_warp_instruction_per_sm'], t['dependent_latency_cycles']))
