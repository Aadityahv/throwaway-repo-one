"""Per-pipe issue cycles per SM, per kernel, with the model's own classes and constants (tensor class and bank-conflict shared cost as in the current model). python3 pipes.py"""
import collections, json
from common import *
import predict_runtime_v2 as V2
import predict_runtime_v3 as V3
K = V2.make_constants(PP.load_constants(CONST)['stream'], PP.load_constants(CONST)['micro'])
A = load_all(); out = {}
for c, x in sorted(A.items()):
    p = x['pred']
    if p.get('primary_s') is None: continue
    ph = x['ph']['kernels'][0]['phases']; bk = x['bank']['kernels'][0]['phases']
    pipes = collections.Counter(); total = 0; ops = collections.Counter()
    for q, b in zip(ph, bk):
        for op, n in q['issue_warp_instructions'].items():
            total += n * q['repetitions']; ops[op] += n * q['repetitions']
            base = op.split('.')[0]
            if base.startswith(('HMMA',)): cls, mult = 'tensor_mma', 1.0
            else: cls, mult = V2.classify(op)
            if cls == 'shared_load': continue
            if cls: pipes[cls] += n * mult * K['issue'][cls] * q['repetitions']
        pipes['shared_load'] += float(b['shared']['shared_cost_cycles'] or 0) * q['repetitions']
    pipes['dispatch'] = V3.DISPATCH_CYCLES * total
    sm = 188 * x['feat']['occupancy']['active_sm_fraction']; clk = K['clock']
    meas = x['measured_s']
    row = dict(cell=c.split('/', 1)[1], total_warp_instr=total, pipes_cycles_per_sm={k: v / sm for k, v in pipes.items()},
               pipe_us={k: v / sm / clk * 1e6 for k, v in pipes.items()}, meas_us=meas * 1e6 if meas else None, pred_us=p['primary_s'] * 1e6,
               top_ops={o: n for o, n in ops.most_common(14)})
    out[c] = row
    pu = row['pipe_us']; top = sorted(pu.items(), key=lambda kv: -kv[1])[:5]
    print(row['cell'], 'meas %.1f pred %.1f' % (row['meas_us'] or 0, row['pred_us']), 'pipes_us', {k: round(v, 1) for k, v in top})
json.dump(out, open(HERE / 'pipes.json', 'w'), indent=1)
