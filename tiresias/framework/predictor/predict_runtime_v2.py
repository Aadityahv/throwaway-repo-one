"""Static runtime predictions, model v2: v1's frozen static inputs, measured hardware constants.

Preregistered in planning/STATIC_PREDICTOR_V2_OVERNIGHT_2026-10-02.md. Reads static features/phases,
the stream constants (fitted on streaming microbenchmarks) and constants/microbench_constants_v2.json
(fitted on the isolated microbenchmark). No operator runtime, energy or profiler metric is opened.

Per kernel: t0 + sum over phases of max(stream, latency, issue) * repetitions + barrier term, where
 latency = waves * (dependent_global_load_depth * L_tier + critical_path_compute_instructions * L_alu)
 issue   = sum over opcodes of warp_instructions * issue_cycles[class] / (active_SMs * clock)
 barrier = waves * barrier_releases_per_block * B(warps_per_block) / clock   (barriers serialise phases)
L_alu is the median measured dependent latency of the integer and fp32 families.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import predict_runtime as V1

HERE = Path(__file__).resolve().parent
LATENCY_NS = {'L1': 16, 'L2': 134, 'DRAM': 311}  # pointer-chase grid, fixed before v2 (plan, amendment 1)


def classify(op):
    base = op.split('.')[0]
    if base in ('BAR', 'WARPSYNC'): return None, 0.0
    if base == 'MUFU' or base.startswith(('F2I', 'I2F', 'I2FP', 'UI2FP')): return 'mufu_ex2', 1.0
    if base == 'SHFL': return 'shuffle', 1.0
    if base in ('LDS', 'STS'):
        width = next((int(x) for x in op.split('.')[1:] if x.isdigit()), 32)
        return 'shared_load', width / 32
    if base in ('FADD', 'FMUL', 'FFMA', 'UFADD', 'UFMUL', 'UFFMA'): return 'fp32_fma', 1.0
    return 'integer', 1.0  # integer, logic, compare, move and address arithmetic


def make_constants(stream, micro):
    issue = micro['issue_cycles_per_warp_instruction_per_sm']
    needed = {'mufu_ex2', 'shuffle', 'shared_load', 'fp32_fma', 'integer'}
    if not needed <= set(issue): raise ValueError('missing measured issue cost: ' + ', '.join(sorted(needed - set(issue))))
    lat = micro['dependent_latency_cycles']
    alu = float(np.median([lat[k] for k in ('integer', 'fp32_add', 'fp32_fma') if k in lat]))
    bar = micro['barrier_latency_cycles_by_warps']
    xs = sorted(int(k) for k in bar)
    if not xs: raise ValueError('missing measured barrier latency')
    clock = micro.get('effective_sm_clock_hz') or 2.617e9
    return dict(stream=stream, issue=issue, alu_latency_cycles=alu, barrier_xs=xs, barrier_ys=[bar[str(x)] for x in xs], clock=clock)


def kernel_terms(phases, occ, tier, K, sm_count, barrier_releases, logical_scale):
    if not occ.get('blocks_per_sm') or not occ.get('waves'): raise ValueError('static occupancy or waves unavailable')
    S = K['stream']; active = sm_count * occ['active_sm_fraction']; scale = S['active_sms_microbench'] / active; terms = []
    for ph in phases:
        for key in ('read_sectors', 'write_sectors', 'lines', 'read_bytes', 'write_bytes', 'dependent_global_load_depth', 'critical_path_compute_instructions'):
            if ph.get(key) is None: raise ValueError('unknown phase transaction or dependency input')
        l2 = (ph['read_sectors'] * 32 / (S['L2_read_sector_TBps'] * 1e12) + ph['write_sectors'] * 32 / (S['L2_write_sector_TBps'] * 1e12)
              + ph['lines'] * S['c_line_ns'] * 1e-9) * scale
        dram = ((ph['read_bytes'] * logical_scale / (S['DRAM_read_TBps'] * 1e12) + ph['write_bytes'] * logical_scale / (S['DRAM_write_TBps'] * 1e12)) * scale) if tier == 'DRAM' else 0.0
        latency = occ['waves'] * (ph['dependent_global_load_depth'] * LATENCY_NS[tier] * 1e-9 + ph['critical_path_compute_instructions'] * K['alu_latency_cycles'] / K['clock'])
        issue = 0.0
        for op, n in ph['issue_warp_instructions'].items():
            cls, mult = classify(op)
            if cls: issue += n * mult * K['issue'][cls]
        issue /= active * K['clock']
        terms.append(dict(stream_s=max(l2, dram), latency_s=latency, issue_s=issue, repetitions=ph['repetitions']))
    w = occ.get('warps_per_block') or 1
    barrier = occ['waves'] * (barrier_releases or 0) * float(np.interp(w, K['barrier_xs'], K['barrier_ys'])) / K['clock']
    totals = {k: sum(p[k] * p['repetitions'] for p in terms) for k in ('stream_s', 'latency_s', 'issue_s')}
    t0 = S['t0_us'] * 1e-6
    return dict(primary_s=t0 + sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms) + barrier,
                max_s=t0 + max(totals.values()) + barrier, sum_s=t0 + sum(totals.values()) + barrier, barrier_s=barrier, phase_terms=terms)


def build(features, phase_rows, K):
    out = {}
    sm_count = features['hardware_from_ground_truth']['sm_count']
    for f in features['rows']:
        cid = f['cell_id']; ph = phase_rows[cid]
        try:
            if ph['status'] == 'unsupported': raise ValueError(ph['reason'])
            tier = f['memory']['tier']; secondary = {k['kernel_id']: k for k in f.get('secondary_kernels', [])}
            executed = sum(p['read_bytes'] + p['write_bytes'] for k in ph['kernels'] for p in k['phases'])
            if not executed: raise ValueError('no static bytes to partition the logical-byte input')
            scale = f['memory']['logical_bytes_per_launch'] / executed; preds = []
            for k in ph['kernels']:
                meta = secondary.get(k.get('kernel_id'), f)
                releases = (meta.get('structure') or {}).get('barrier_releases_per_block_estimate')
                preds.append(kernel_terms(k['phases'], meta['occupancy'], tier, K, sm_count, releases, scale))
            if not preds: raise ValueError('no static kernel phases')
            out[cid] = {key: sum(p[key] for p in preds) for key in ('primary_s', 'max_s', 'sum_s')}
            out[cid].update(unsupported_reason=None, kernels=preds, logical_byte_allocation_scale=scale)
        except (ValueError, KeyError, TypeError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason=str(ex))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--micro', type=Path, default=HERE / 'constants/microbench_constants_v2.json')
    ap.add_argument('--out', type=Path, default=HERE / 'predictions_v2_blackwell.json'); a = ap.parse_args()
    features = json.loads((HERE / 'features_blackwell.json').read_text()); phases = json.loads((HERE / 'phases_blackwell.json').read_text())['rows']
    stream = json.loads((HERE / 'constants/stream_constants.json').read_text())['constants']
    K = make_constants(stream, json.loads(a.micro.read_text()))
    out = build(features, phases, K)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cells=len(out), supported=sum(v['primary_s'] is not None for v in out.values()))))


if __name__ == '__main__':
    main()
