"""v3c constants: memory-level-parallelism (Little's law) service latency per tier, from micro_v3/out_mlp.jsonl.

Per configuration the read bytes in flight per SM = achieved blocks/SM x 4 warps x 32 lanes x W x M. In the linear region
(in-flight <= 2 KB per SM) read throughput per SM = in-flight / lambda, so lambda = in-flight / (read bytes per second per SM).
Read bytes per second = bytes_each_way / time (a copy reads and writes the same bytes). lambda is the median over the linear
region points (in-flight <= 2 KB per SM). No operator data.
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; M = HERE.parent / 'micro_v3'
def main():
    rows = [json.loads(l) for l in (M / 'out_mlp.jsonl').read_text().splitlines()]
    out = {'service_latency_us': {}, 'linear_region_points': {}}
    for tier in ('l2', 'dram'):
        lam = []
        for x in rows:
            if x['tier'] != tier: continue
            inflight = x['achieved_blocks_per_sm'] * 4 * 32 * x['W'] * x['M']
            if inflight > 2048: continue
            read_rate_per_sm = x['bytes_each_way'] / (x['us'] * 1e-6) / 188
            lam.append(inflight / read_rate_per_sm * 1e6)
        out['service_latency_us'][tier.upper()] = float(np.median(lam)); out['linear_region_points'][tier.upper()] = [round(v, 3) for v in lam]
    out['rule'] = ('per phase with global loads: read bytes / (active SMs x in-flight bytes per SM / service latency) is a lower bound on the stream '
                   'stage; in-flight per SM = resident warps x (read bytes per load request) x min(loads per warp / dependent load depth, 1 or more)')
    out['source'] = 'tiresias/framework/predictor/micro_v3/out_mlp.jsonl'
    (HERE / 'v3c_constants.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(out['service_latency_us']), {k: len(v) for k, v in out['linear_region_points'].items()})
if __name__ == '__main__': main()
