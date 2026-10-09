"""Per-kernel/per-phase term breakdown and missing time for every tensor-core matrix multiply and fused attention cell. python3 dump_terms.py > terms.txt"""
import json
from common import *
A = load_all(); rows = []
for c, x in sorted(A.items()):
    p = x['pred']; occ = x['feat']['occupancy']; m = x['measured_s']
    if p.get('primary_s') is None:
        print(c, 'REFUSED', p['unsupported_reason'][:80]); continue
    k = p['kernels'][0]; pt = k['phase_terms']
    gsum = lambda key: sum(t[key] * t['repetitions'] for t in pt)
    # serial = sum of per-phase max + barrier; full = max of totals + barrier
    body = sum(max(t['stream_s'], t['latency_s'], t['issue_s']) * t['repetitions'] for t in pt)
    # which term dominates each phase, summed by kind
    dom = {'stream': 0.0, 'latency': 0.0, 'issue': 0.0}
    for t in pt:
        key = max(('stream', 'latency', 'issue'), key=lambda n: t[n + '_s']); dom[key] += t[key + '_s'] * t['repetitions']
    r = dict(cell=c.split('/', 1)[1], set=x['set'], tier=x['feat']['memory']['tier'], bps=occ['blocks_per_sm'], warps=occ.get('warps_per_block'), waves=occ['waves'],
             blocks=(x['feat']['geometry'].get('grid_blocks')), n_phases=len(pt), alpha=k['overlap_alpha'], pred_us=p['primary_s'] * 1e6, meas_us=(m * 1e6 if m else None),
             ratio=(p['primary_s'] / m if m else None), serial_us=(k['primary_v3h_s'] * 1e6), full_us=k['max_s'] * 1e6, barrier_us=k['barrier_s'] * 1e6, launch_us=k['launch_s'] * 1e6, dispatch_us=k['dispatch_s'] * 1e6,
             sum_stream_us=gsum('stream_s') * 1e6, sum_latency_us=gsum('latency_s') * 1e6, sum_issue_us=gsum('issue_s') * 1e6, dominated_by_us={a: b * 1e6 for a, b in dom.items()},
             missing_us=((m - p['primary_s']) * 1e6 if m else None), releases=(x['feat'].get('structure') or {}).get('barrier_releases_per_block_estimate'))
    rows.append(r)
    print(json.dumps(r, default=lambda o: round(o, 3)))
json.dump(rows, open(HERE / 'terms.json', 'w'), indent=1)
