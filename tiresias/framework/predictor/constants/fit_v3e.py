"""v3e constants: stream-stage bandwidth by read fraction and per-kernel fixed overhead, from micro_v3/out_stream3.jsonl.

L2 (graph-launched, two sizes per mix): per mix the marginal bandwidth BW = d(bytes)/d(time) and the fixed overhead
a = t(6 MiB) - bytes/BW, from the 6 MiB and 12 MiB per-buffer points (best per_thread).
DRAM: best over per_thread of bytes / time (kernels >= 100 us, launch effects negligible).
Read-only L2/DRAM bandwidth is the mean of the R1W0 and R2W0 mixes. No operator data.
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent / 'micro_v3'
MIXES = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2)]
def main():
    r = [json.loads(l) for l in (M / 'out_stream3.jsonl').read_text().splitlines()]
    l2, over, dram = {}, [], {}
    for mix in MIXES:
        pts = []
        for tier in ('l2a', 'l2b'):
            b = min((x for x in r if x['tier'] == tier and (x['R'], x['W']) == mix), key=lambda x: x['us']); pts.append((b['bytes_read'] + b['bytes_written'], b['us']))
        (b1, t1), (b2, t2) = pts; bw = (b2 - b1) / ((t2 - t1) * 1e-6) / 1e12; over.append(t1 - b1 / (bw * 1e12) * 1e6)
        l2[mix] = bw
        dram[mix] = max((x['bytes_read'] + x['bytes_written']) / (x['us'] * 1e-6) / 1e12 for x in r if x['tier'] == 'dram' and (x['R'], x['W']) == mix)
    def curve(d):
        by = {}
        for (R, W), v in d.items(): by.setdefault(round(R / (R + W), 4), []).append(v)
        xs = sorted(by); return xs, [float(np.mean(by[x])) for x in xs]
    lx, ly = curve(l2); dx, dy = curve(dram)
    out = dict(l2_total_bandwidth_TBps=dict(read_fraction=lx, value=ly), dram_total_bandwidth_TBps=dict(read_fraction=dx, value=dy),
               kernel_fixed_overhead_us=float(np.median(over)), overhead_points_us=[round(o, 3) for o in over],
               source='tiresias/framework/predictor/micro_v3/out_stream3.jsonl')
    (HERE / 'v3e_constants.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(out, indent=1))
if __name__ == '__main__': main()
