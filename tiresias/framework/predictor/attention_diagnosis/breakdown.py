"""Missing-time breakdown per cell by mechanism steps: current model -> + tensor/shared pair overlap coefficient -> + stage serialisation -> + busiest-SM rule -> residual. python3 breakdown.py > breakdown.txt"""
from mech_lib import *
from mechanisms import Mech
cells = load_cells()
steps = [('current', Mech()), ('+pair overlap', Mech(beta_tc=True)), ('+stage rule', Mech(beta_tc=True, stage=True)), ('+busiest-SM rule', Mech(beta_tc=True, stage=True, tail=True))]
print('%-46s %-5s %8s %8s | %8s %8s %8s %8s | %8s %6s' % ('cell', 'tier', 'meas us', 'pred us', 'd beta', 'd stage', 'd tail', 'final', 'resid us', 'ratio'))
for c in cells:
    if not any(s in c['cid'] for s in ('tensor_matmul', 'fused_attention')) or c['pred_primary_s'] is None: continue
    v = [compose(c, m) * 1e6 for _, m in steps]; m = c['measured_s'] * 1e6
    print('%-46s %-5s %8.1f %8.1f | %+8.1f %+8.1f %+8.1f %8.1f | %+8.1f %6.2f' % (c['cid'].split('/', 1)[1][:46], c['tier'], m, v[0], v[1] - v[0], v[2] - v[1], v[3] - v[2], v[3], m - v[3], v[3] / m))
