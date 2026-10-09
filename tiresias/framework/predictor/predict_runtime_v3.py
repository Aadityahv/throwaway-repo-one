"""Static runtime predictions, model v3 (preregistered in planning/STATIC_RUNTIME_V3_PREREGISTRATION_2026-10-02.md).

v2 plus: issue = pipe roofline; grid-dependent launch overhead per kernel; L1 reuse via first-touch sectors.
Reads static features/phases/first-touch tables and microbenchmark constants only; no runtime or energy label.
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

import predict_runtime_v2 as V2

HERE = Path(__file__).resolve().parent
DISPATCH_CYCLES = 0.25  # 4 schedulers x 1 warp instruction per cycle (matches the measured FP32 issue of 0.251)


def issue_cycles(ph, K):
    pipes = collections.Counter(); total = 0
    for op, n in ph['issue_warp_instructions'].items():
        total += n
        cls, mult = V2.classify(op)
        if cls: pipes[cls] += n * mult * K['issue'][cls]
    return max(DISPATCH_CYCLES * total, max(pipes.values(), default=0.0))


def mlp_seconds(ph, occ, blocks, tier, V3C, active):
    """v3c: Little's-law bound. Read bytes / (active SMs x in-flight bytes per SM / service latency)."""
    requests = sum(n for op, n in ph['issue_warp_instructions'].items() if op.startswith('LDG'))
    if not requests or not ph['read_sectors']: return 0.0
    warps_block = occ.get('warps_per_block') or 1
    resident_warps = min(occ['blocks_per_sm'], math.ceil(blocks / 188)) * warps_block
    bytes_per_request = ph['read_sectors'] * 32 / requests
    loads_per_warp = requests / max(1, blocks * warps_block)
    mlp = max(1.0, loads_per_warp / max(1, ph['dependent_global_load_depth']))
    inflight = resident_warps * bytes_per_request * mlp
    lam = V3C['service_latency_us'][tier] * 1e-6
    return (ph['read_sectors'] * 32) / (active * inflight / lam)


def l2_share(V3, footprint_bytes, curved):
    """Share of re-read sectors served by L2. v3b/v3c: hard 96 KB gate. v3d: measured smooth curve."""
    if not curved: return 0.0 if footprint_bytes <= V3['l1']['capacity_bytes_per_sm'] else 1.0
    c = V3['l1_reread_l2_fraction_curve']; return float(np.interp(footprint_bytes / 1024, c['per_sm_kb'], c['p_l2']))


def curve_bw(c, rho):
    return float(np.interp(rho, c['read_fraction'], c['value'])) * 1e12


def kernel_terms(phases, uphases, ukernel, occ, geometry, tier, K, V3, sm_count, barrier_releases, logical_scale, V3C=None, curved=False, V3E=None):
    if not occ.get('blocks_per_sm') or not occ.get('waves'): raise ValueError('static occupancy or waves unavailable')
    S = K['stream']; active = sm_count * occ['active_sm_fraction']; scale = S['active_sms_microbench'] / active
    if ukernel is None or ukernel.get('unique_read_bytes_per_block') is None: raise ValueError('first-touch read footprint unavailable')
    blocks = geometry.get('grid_blocks') or geometry.get('grid_blocks_dispatch')
    if not blocks: raise ValueError('grid size unavailable')
    resident = min(occ['blocks_per_sm'], math.ceil(blocks / sm_count))
    footprint = resident * ukernel['unique_read_bytes_per_block']
    share = l2_share(V3, footprint, curved); reuse_gate = share < 1.0
    terms = []
    for ph, up in zip(phases, uphases):
        for key in ('read_sectors', 'write_sectors', 'lines', 'read_bytes', 'write_bytes', 'dependent_global_load_depth', 'critical_path_compute_instructions'):
            if ph.get(key) is None: raise ValueError('unknown phase transaction or dependency input')
        if up.get('read_sectors_first_touch') is None: raise ValueError('first-touch sectors unavailable')
        r_total = ph['read_sectors']; r_ft = min(up['read_sectors_first_touch'], r_total)
        l2_reads = r_ft + share * (r_total - r_ft); l1_sectors = (1 - share) * (r_total - r_ft)
        if V3E:  # v3e: total L2 bytes / measured bandwidth at this read fraction (graph-launched microbenchmark)
            rb, wb = l2_reads * 32, ph['write_sectors'] * 32
            l2 = ((rb + wb) / curve_bw(V3E['l2_total_bandwidth_TBps'], rb / (rb + wb)) if rb + wb else 0.0) * scale + ph['lines'] * S['c_line_ns'] * 1e-9 * scale
        else:
            l2 = (l2_reads * 32 / (S['L2_read_sector_TBps'] * 1e12) + ph['write_sectors'] * 32 / (S['L2_write_sector_TBps'] * 1e12)
                  + ph['lines'] * S['c_line_ns'] * 1e-9) * scale
        l1 = l1_sectors * 32 / (V3['l1']['bandwidth_bytes_per_s_per_sm'] * active)
        if tier != 'DRAM': dram = 0.0
        elif V3E:
            drb, dwb = ph['read_bytes'] * logical_scale, ph['write_bytes'] * logical_scale
            dram = ((drb + dwb) / curve_bw(V3E['dram_total_bandwidth_TBps'], drb / (drb + dwb)) if drb + dwb else 0.0) * scale
        else:
            dram = (ph['read_bytes'] * logical_scale / (S['DRAM_read_TBps'] * 1e12) + ph['write_bytes'] * logical_scale / (S['DRAM_write_TBps'] * 1e12)) * scale
        latency = occ['waves'] * (ph['dependent_global_load_depth'] * V2.LATENCY_NS[tier] * 1e-9 + ph['critical_path_compute_instructions'] * K['alu_latency_cycles'] / K['clock'])
        issue = issue_cycles(ph, K) / (active * K['clock'])
        mlp = mlp_seconds(ph, occ, blocks, tier, V3C, active) if V3C else 0.0
        terms.append(dict(stream_s=max(l2, l1, dram, mlp), mlp_s=mlp, l2_s=l2, l1_s=l1, dram_s=dram, latency_s=latency, issue_s=issue, repetitions=ph['repetitions']))
    w = occ.get('warps_per_block') or 1
    barrier = occ['waves'] * (barrier_releases or 0) * float(np.interp(w, K['barrier_xs'], K['barrier_ys'])) / K['clock']
    totals = {k: sum(p[k] * p['repetitions'] for p in terms) for k in ('stream_s', 'latency_s', 'issue_s')}
    # v3b: only the fixed launch cost is additive; block dispatch is a rate bound (blocks x per-block cost) inside the max,
    # because dispatch of later blocks overlaps execution of earlier ones (v3a added it and over-predicted large grids).
    L = V3['launch_us_per_kernel']; t0 = (V3E['kernel_fixed_overhead_us'] if V3E else L['intercept_at_one_block']) * 1e-6; dispatch = L['per_extra_block'] * (blocks - 1) * 1e-6
    body = sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms) + barrier
    return dict(primary_s=t0 + max(body, dispatch),
                max_s=t0 + max(max(totals.values()) + barrier, dispatch), sum_s=t0 + max(sum(totals.values()) + barrier, dispatch), barrier_s=barrier, launch_s=t0, dispatch_s=dispatch,
                reuse_gate=reuse_gate, per_sm_unique_read_bytes=footprint, phase_terms=terms)


def build(features, phase_rows, unique_rows, K, V3, V3C=None, curved=False, V3E=None):
    out = {}; sm_count = features['hardware_from_ground_truth']['sm_count']
    for f in features['rows']:
        cid = f['cell_id']; ph = phase_rows[cid]
        try:
            if ph['status'] == 'unsupported': raise ValueError(ph['reason'])
            u = unique_rows.get(cid)
            if not u or u.get('status') not in ('ok', 'derived', 'conditional_static_phases'): raise ValueError('first-touch analysis unavailable: ' + str((u or {}).get('reason')))
            tier = f['memory']['tier']; secondary = {k['kernel_id']: k for k in f.get('secondary_kernels', [])}
            executed = sum(p['read_bytes'] + p['write_bytes'] for k in ph['kernels'] for p in k['phases'])
            if not executed: raise ValueError('no static bytes to partition the logical-byte input')
            scale = f['memory']['logical_bytes_per_launch'] / executed; preds = []
            for kk, uk in zip(ph['kernels'], u['kernels']):
                meta = secondary.get(kk.get('kernel_id'), f)
                releases = (meta.get('structure') or {}).get('barrier_releases_per_block_estimate')
                preds.append(kernel_terms(kk['phases'], uk['phases'], uk, meta['occupancy'], meta['geometry'], tier, K, V3, sm_count, releases, scale, V3C, curved, V3E))
            if not preds: raise ValueError('no static kernel phases')
            out[cid] = {key: sum(p[key] for p in preds) for key in ('primary_s', 'max_s', 'sum_s')}
            out[cid].update(unsupported_reason=None, kernels=preds, logical_byte_allocation_scale=scale)
        except (ValueError, KeyError, TypeError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason=str(ex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, default=HERE / 'features_blackwell.json'); ap.add_argument('--phases', type=Path, default=HERE / 'phases_blackwell.json')
    ap.add_argument('--unique', type=Path, default=HERE / 'reuse/phases_unique_blackwell.json')
    ap.add_argument('--micro', type=Path, default=HERE / 'constants/microbench_constants_v2.json'); ap.add_argument('--v3', type=Path, default=HERE / 'constants/v3_constants.json')
    ap.add_argument('--variant', choices=['v3b', 'v3c', 'v3d', 'v3e'], default='v3b'); ap.add_argument('--v3c', type=Path, default=HERE / 'constants/v3c_constants.json')
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    features = json.loads(a.features.read_text()); phases = json.loads(a.phases.read_text())['rows']
    uniq = json.loads(a.unique.read_text()); uniq = uniq.get('rows', uniq)
    stream = json.loads((HERE / 'constants/stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads(a.micro.read_text())); V3 = json.loads(a.v3.read_text())
    V3C = json.loads(a.v3c.read_text()) if a.variant in ('v3c', 'v3d', 'v3e') else None
    V3E = json.loads((HERE / 'constants/v3e_constants.json').read_text()) if a.variant == 'v3e' else None
    out = build(features, phases, uniq, K, V3, V3C, a.variant in ('v3d', 'v3e'), V3E)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cells=len(out), supported=sum(v['primary_s'] is not None for v in out.values()))))


if __name__ == '__main__':
    main()
