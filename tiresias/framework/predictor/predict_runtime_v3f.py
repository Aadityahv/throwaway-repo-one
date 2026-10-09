"""Static runtime predictions, model v3f (preregistered in planning/STATIC_RUNTIME_V3F_PREREGISTRATION_2026-10-02.md).

v3e plus three corrections found by round B and its term-by-term diagnosis:
 1. The read fraction used to pick the measured L2/DRAM bandwidth is taken per KERNEL (reads and writes overlap across blocks),
    and the kernel's stream time is allocated to its barrier phases in proportion to their bytes.
 2. DRAM traffic per kernel = grid-wide unique read sectors (static, from the first-touch table) + written sectors, replacing the
    logical-byte partition (which counted layer norm's X three times and left the padded copy kernel's own traffic out).
 3. The memory-level-parallelism bound uses only the bytes served by the tier (first-touch / DRAM-unique reads), not L1-served re-reads.
Reads static features/phases/first-touch tables and microbenchmark constants only; no runtime or energy label.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import predict_runtime_v2 as V2
import predict_runtime_v3 as V3

HERE = Path(__file__).resolve().parent
KERNEL_LEVEL_RHO = True  # False: apply the measured bandwidth curve per barrier phase (v3e behaviour) while keeping the DRAM and MLP corrections


def grid_unique_reads(ukernel, ph_ft_total):
    """Grid-wide unique read sectors: blocks x (first-touch per block - sectors shared with other blocks) + shared once."""
    blocks = ukernel['blocks']; per_block = ukernel['first_touch_read_sectors_per_block']
    note = ukernel.get('read_sectors_shared_across_blocks_note') or {}
    shared = note.get('shared_with_at_least_one_other_sampled_block') or 0
    return min(ph_ft_total, blocks * (per_block - shared) + shared) if shared else ph_ft_total


def mlp_seconds(served_bytes, ph, occ, blocks, tier, V3C, active):
    requests = sum(n for op, n in ph['issue_warp_instructions'].items() if op.startswith('LDG'))
    if not requests or not ph['read_sectors'] or not served_bytes: return 0.0
    warps_block = occ.get('warps_per_block') or 1
    resident_warps = min(occ['blocks_per_sm'], math.ceil(blocks / 188)) * warps_block
    bytes_per_request = ph['read_sectors'] * 32 / requests
    mlp = max(1.0, requests / max(1, blocks * warps_block) / max(1, ph['dependent_global_load_depth']))
    return served_bytes / (active * resident_warps * bytes_per_request * mlp / (V3C['service_latency_us'][tier] * 1e-6))


def kernel_terms(phases, uphases, ukernel, occ, geometry, tier, K, V3c, sm_count, barrier_releases, V3C, V3E):
    if not occ.get('blocks_per_sm') or not occ.get('waves'): raise ValueError('static occupancy or waves unavailable')
    S = K['stream']; active = sm_count * occ['active_sm_fraction']; scale = S['active_sms_microbench'] / active
    if ukernel is None or ukernel.get('unique_read_bytes_per_block') is None: raise ValueError('first-touch read footprint unavailable')
    blocks = geometry.get('grid_blocks') or geometry.get('grid_blocks_dispatch')
    if not blocks: raise ValueError('grid size unavailable')
    share = V3.l2_share(V3c, min(occ['blocks_per_sm'], math.ceil(blocks / sm_count)) * ukernel['unique_read_bytes_per_block'], True)
    rows = []
    for ph, up in zip(phases, uphases):
        for key in ('read_sectors', 'write_sectors', 'lines', 'dependent_global_load_depth', 'critical_path_compute_instructions'):
            if ph.get(key) is None: raise ValueError('unknown phase transaction or dependency input')
        if up.get('read_sectors_first_touch') is None: raise ValueError('first-touch sectors unavailable')
        r_total = ph['read_sectors']; r_ft = min(up['read_sectors_first_touch'], r_total)
        rows.append(dict(ph=ph, r_total=r_total, r_ft=r_ft, l2_reads=r_ft + share * (r_total - r_ft), l1_sectors=(1 - share) * (r_total - r_ft), writes=ph['write_sectors']))
    # 1) kernel-level L2 stage
    l2_r = sum(x['l2_reads'] for x in rows) * 32; l2_w = sum(x['writes'] for x in rows) * 32; l2_tot = l2_r + l2_w
    l2_kernel = (l2_tot / V3.curve_bw(V3E['l2_total_bandwidth_TBps'], l2_r / l2_tot) if l2_tot else 0.0) * scale
    # 2) kernel-level DRAM stage from grid-unique sectors
    ft_total = sum(x['r_ft'] for x in rows); uniq = grid_unique_reads(ukernel, ft_total)
    for x in rows: x['dram_r'] = uniq * 32 * (x['r_ft'] / ft_total) if (tier == 'DRAM' and ft_total) else 0.0
    d_r = sum(x['dram_r'] for x in rows); d_w = sum(x['writes'] for x in rows) * 32 if tier == 'DRAM' else 0.0
    dram_kernel = ((d_r + d_w) / V3.curve_bw(V3E['dram_total_bandwidth_TBps'], d_r / (d_r + d_w)) if d_r + d_w else 0.0) * scale
    terms = []
    for x in rows:
        ph = x['ph']; b = x['l2_reads'] * 32 + x['writes'] * 32
        db = x['dram_r'] + (x['writes'] * 32 if tier == 'DRAM' else 0.0)
        if KERNEL_LEVEL_RHO:
            l2 = l2_kernel * (b / l2_tot) if l2_tot else 0.0
            dram = dram_kernel * (db / (d_r + d_w)) if d_r + d_w else 0.0
        else:
            rb = x['l2_reads'] * 32; wb = x['writes'] * 32
            l2 = ((rb + wb) / V3.curve_bw(V3E['l2_total_bandwidth_TBps'], rb / (rb + wb)) if rb + wb else 0.0) * scale
            dram = (db / V3.curve_bw(V3E['dram_total_bandwidth_TBps'], x['dram_r'] / db) if db else 0.0) * scale
        l1 = x['l1_sectors'] * 32 / (V3c['l1']['bandwidth_bytes_per_s_per_sm'] * active) + ph['lines'] * S['c_line_ns'] * 1e-9 * scale
        served = x['dram_r'] if tier == 'DRAM' else x['l2_reads'] * 32
        mlp = mlp_seconds(served, ph, occ, blocks, tier, V3C, active)
        latency = occ['waves'] * (ph['dependent_global_load_depth'] * V2.LATENCY_NS[tier] * 1e-9 + ph['critical_path_compute_instructions'] * K['alu_latency_cycles'] / K['clock'])
        issue = V3.issue_cycles(ph, K) / (active * K['clock'])
        terms.append(dict(stream_s=max(l2, l1, dram, mlp), l2_s=l2, l1_s=l1, dram_s=dram, mlp_s=mlp, latency_s=latency, issue_s=issue, repetitions=ph['repetitions']))
    w = occ.get('warps_per_block') or 1
    barrier = occ['waves'] * (barrier_releases or 0) * float(np.interp(w, K['barrier_xs'], K['barrier_ys'])) / K['clock']
    totals = {k: sum(p[k] * p['repetitions'] for p in terms) for k in ('stream_s', 'latency_s', 'issue_s')}
    L = V3c['launch_us_per_kernel']; t0 = V3E['kernel_fixed_overhead_us'] * 1e-6; dispatch = L['per_extra_block'] * (blocks - 1) * 1e-6
    body = sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms) + barrier
    return dict(primary_s=t0 + max(body, dispatch), max_s=t0 + max(max(totals.values()) + barrier, dispatch), sum_s=t0 + max(sum(totals.values()) + barrier, dispatch),
                barrier_s=barrier, launch_s=t0, dispatch_s=dispatch, dram_unique_read_sectors=uniq, phase_terms=terms)


def build(features, phase_rows, unique_rows, K, V3c, V3C, V3E):
    out = {}; sm_count = features['hardware_from_ground_truth']['sm_count']
    for f in features['rows']:
        cid = f['cell_id']; ph = phase_rows[cid]
        try:
            if ph['status'] == 'unsupported': raise ValueError(ph['reason'])
            u = unique_rows.get(cid)
            if not u or u.get('status') not in ('ok', 'derived', 'conditional_static_phases'): raise ValueError('first-touch analysis unavailable: ' + str((u or {}).get('reason')))
            tier = f['memory']['tier']; secondary = {k['kernel_id']: k for k in f.get('secondary_kernels', [])}; preds = []
            for kk, uk in zip(ph['kernels'], u['kernels']):
                meta = secondary.get(kk.get('kernel_id'), f)
                preds.append(kernel_terms(kk['phases'], uk['phases'], uk, meta['occupancy'], meta['geometry'], tier, K, V3c, sm_count,
                                          (meta.get('structure') or {}).get('barrier_releases_per_block_estimate'), V3C, V3E))
            if not preds: raise ValueError('no static kernel phases')
            out[cid] = {key: sum(p[key] for p in preds) for key in ('primary_s', 'max_s', 'sum_s')}
            out[cid].update(unsupported_reason=None, kernels=preds)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason=str(ex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, default=HERE / 'features_blackwell.json'); ap.add_argument('--phases', type=Path, default=HERE / 'phases_blackwell.json')
    ap.add_argument('--unique', type=Path, default=HERE / 'reuse/phases_unique_blackwell.json'); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    features = json.loads(a.features.read_text()); phases = json.loads(a.phases.read_text())['rows']; uniq = json.loads(a.unique.read_text()); uniq = uniq.get('rows', uniq)
    C = HERE / 'constants'; stream = json.loads((C / 'stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((C / 'microbench_constants_v2.json').read_text()))
    out = build(features, phases, uniq, K, json.loads((C / 'v3_constants.json').read_text()), json.loads((C / 'v3c_constants.json').read_text()), json.loads((C / 'v3e_constants.json').read_text()))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(cells=len(out), supported=sum(v['primary_s'] is not None for v in out.values()))))


if __name__ == '__main__':
    main()
