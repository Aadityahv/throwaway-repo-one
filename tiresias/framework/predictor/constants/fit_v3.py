"""v3 constants from the micro_v3 measurements (Blackwell GPU 1, 2026-10-02). No operator data.

launch  : per extra kernel in a graph, us = a + b*(blocks-1), two-point fit on grid 1 and grid 2048 (grid 188 is a check).
l1      : sectors of a block's re-read pass are L1-served when the per-SM footprint is at most L1_CAPACITY_BYTES.
          Capacity = largest measured per-SM footprint whose extra pass was L1-speed (96 KB); 128 KB per SM behaved as L2.
          L1 bandwidth = median over the 1-block-per-SM points with 48..96 KB per SM (noise-free, cleanest).
mix     : validation of the max-over-pipes composition rule (reported, not fitted).
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; M = HERE.parent / 'micro_v3'

def main():
    chain = [json.loads(l) for l in (M / 'out_chain.jsonl').read_text().splitlines()]
    inc = {}
    for g in (1, 188, 2048):
        t = {x['kernels_per_launch']: x['us_per_launch'] for x in chain if x['grid'] == g}
        inc[g] = float(np.mean([t[k + 1] - t[k] for k in (1, 2, 3)]))
    b = (inc[2048] - inc[1]) / 2047; a = inc[1]
    check188 = a + b * 187
    reuse = [json.loads(l) for l in (M / 'out_reuse.jsonl').read_text().splitlines()]
    pts = [x['footprint_kb_per_block'] * 1024 / (x['extra_pass_us'] * 1e-6) / 1e9 for x in reuse
           if x['blocks_per_sm'] == 1 and 48 <= x['footprint_kb_per_block'] <= 96 and x['extra_pass_us'] > 0.1]
    l1_bw = float(np.median(pts))
    # Smooth L1 re-read curve (v3d): share of re-read sectors that miss L1 and are served by L2, versus per-SM footprint.
    # Series = 4 resident blocks per SM (closest analog to layer norm: many small row-blocks resident). ratio = extra pass / first pass;
    # L1-hit cost ratio r_hit = ratio at 64 KB per SM; p = clip((ratio - r_hit) / (1 - r_hit)); beyond 512 KB per SM p ramps to 1 at 1024 KB.
    s4 = sorted((x['blocks_per_sm'] * x['footprint_kb_per_block'], x['extra_pass_us'] / x['us_1pass']) for x in reuse if x['blocks_per_sm'] == 4 and x['blocks_per_sm'] * x['footprint_kb_per_block'] <= 512)
    r_hit = float(np.median([r for kb, r in s4 if kb == 64]))
    curve = [(kb, float(np.clip((r - r_hit) / (1 - r_hit), 0, 1))) for kb, r in s4 if kb >= 64]
    curve = [(64, 0.0)] + [(kb, v) for kb, v in curve if kb > 64] + [(1024, 1.0)]
    mono = []
    for kb, v in curve: mono.append((kb, max(v, mono[-1][1]) if mono else v))
    pipe = dict(fma=0.251, int=0.312, shfl=2.0, mufu=2.0, lds=4.0)
    ratios = {}
    for x in (json.loads(l) for l in (M / 'out_mix.jsonl').read_text().splitlines()):
        ops = dict(fma=x['NF'], int=2 * x['NI'], shfl=x['NS'], mufu=x['NM'], lds=x['NL']); ops = {k: v * 4 * 32 for k, v in ops.items()}
        tot = sum(ops.values()); mx = max(tot * 0.25, max(ops[k] * pipe[k] for k in ops)); sm = sum(ops[k] * pipe[k] for k in ops)
        meas = x['last_warp_cycles'] / x['loops']
        ratios[x['name']] = dict(measured_over_max_rule=round(meas / mx, 3), measured_over_sum_rule=round(meas / sm, 3))
    out = dict(l1_reread_l2_fraction_curve=dict(per_sm_kb=[k for k, _ in mono], p_l2=[round(v, 4) for _, v in mono], hit_cost_ratio=r_hit),
               launch_us_per_kernel=dict(intercept_at_one_block=a, per_extra_block=b, check_at_188_blocks_pred=check188,
                                         check_at_188_blocks_measured=inc[188]),
               l1=dict(capacity_bytes_per_sm=96 * 1024, bandwidth_bytes_per_s_per_sm=l1_bw * 1e9, points_gbps=[round(p, 1) for p in pts]),
               composition_rule='issue cycles = max(0.25 * total warp instructions, busiest pipe) per phase',
               composition_validation=ratios,
               source='tiresias/framework/predictor/micro_v3/out_{chain,mix,reuse}.jsonl')
    (HERE / 'v3_constants.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(out, indent=1, sort_keys=True))

if __name__ == '__main__':
    main()
