"""Static runtime predictions, candidate model v3j: the current model (v3i: bank-conflict shared-memory cost plus partial phase overlap) with ONE traffic correction and
traffic accounting for the energy model. Design and mechanism: shared_traffic/DESIGN.md.

Change 1 (read footprint). v3f/v3g/v3h/v3i charge DRAM read traffic as `blocks x (first-touch sectors per block - sectors shared with another sampled block)`. In a tiled kernel
every block loads its own tile, so that estimate charges every tile load to DRAM although the grid as a whole touches each input sector from DRAM once and the L2 serves the other
blocks' loads. v3j replaces it with the grid-wide unique footprint derived from the interpreted address traces, per pointer argument
(shared_traffic/footprint.py): the DRAM read sectors of a kernel are the sectors the whole grid touches, bounded above by the per-block first-touch total. A kernel whose
inter-block reuse working set exceeds the L2 capacity is refused (the premise that L2 captures inter-block reuse fails), and a DRAM-tier kernel without a footprint is refused.
Everything else (L2 stage, memory-level parallelism bound, latency and issue terms, overlap) is unchanged. No constant is added or fitted.

Traffic accounting. Each kernel prediction also reports the bytes the L2 serves (first touches, L2-served re-reads, writes) and the DRAM bytes, from the same terms as the
runtime stage, so the energy model can charge exactly the traffic the runtime model times.

With `read_footprint=False` the model is byte-for-byte the arithmetic of v3f/v3i (verified by shared_traffic/test_shared_traffic.py on all evaluation cells).
The v3f..v3i files are imported unchanged; their frozen hashes stay valid.
"""
import contextlib
import math

import numpy as np

import predict_runtime_v2 as V2
import predict_runtime_v3 as V3
import predict_runtime_v3f as V3F
import predict_runtime_v3i as V3I

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'shared_traffic'))
from footprint import reuse_working_set_bytes, wave_working_set_bytes  # noqa: E402

OPTIONS = dict(read_footprint=True, write_footprint=False, l2_rule='wave')   # l2_rule: 'wave' (Amendment 2) or 'launch' (whole-launch working set, first version)


def dram_read_sectors(ukernel, ft_total, tier, wave_blocks=None):
    """DRAM read sectors of a kernel launch (see module docstring)."""
    legacy = V3F.grid_unique_reads(ukernel, ft_total)
    if not OPTIONS['read_footprint'] or tier != 'DRAM': return legacy
    fp = ukernel.get('grid_footprint')
    if not fp: raise ValueError('DRAM-tier kernel without a derived grid footprint (refused, no fallback to logical bytes)')
    if OPTIONS['l2_rule'] == 'wave':
        if wave_blocks is None: raise ValueError('wave size unavailable for the per-wave L2 rule (refused)')
        ws, what = wave_working_set_bytes(fp, wave_blocks) if 'pointers' in fp else fp['wave_working_set_bytes'], 'per-wave'
    else: ws, what = fp['reuse_working_set_bytes'], 'whole-launch'
    if ws > fp['l2_capacity_bytes']:
        raise ValueError('inter-block reuse %s working set %.1f MiB exceeds the L2 capacity %.1f MiB; the L2-captures-reuse assumption fails (refused)' % (what, ws / 2**20, fp['l2_capacity_bytes'] / 2**20))
    return min(ft_total, fp['read_footprint_sectors'])


def kernel_terms(phases, uphases, ukernel, occ, geometry, tier, K, V3c, sm_count, barrier_releases, V3C, V3E):
    """V3F.kernel_terms with the DRAM read estimate above, and the L2/DRAM byte totals added to the result (`traffic`)."""
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
    l2_r = sum(x['l2_reads'] for x in rows) * 32; l2_w = sum(x['writes'] for x in rows) * 32; l2_tot = l2_r + l2_w
    l2_kernel = (l2_tot / V3.curve_bw(V3E['l2_total_bandwidth_TBps'], l2_r / l2_tot) if l2_tot else 0.0) * scale
    ft_total = sum(x['r_ft'] for x in rows); wave_blocks = min(blocks, occ['blocks_per_sm'] * active)
    uniq = dram_read_sectors(ukernel, ft_total, tier, wave_blocks)
    for x in rows: x['dram_r'] = uniq * 32 * (x['r_ft'] / ft_total) if (tier == 'DRAM' and ft_total) else 0.0
    d_r = sum(x['dram_r'] for x in rows); d_w = sum(x['writes'] for x in rows) * 32 if tier == 'DRAM' else 0.0
    wscale = 1.0
    if OPTIONS['write_footprint'] and tier == 'DRAM' and d_w:   # post-hoc option C3: DRAM write bytes = the grid-wide unique written footprint (the L2 merges partial-sector writes before write-back)
        fp = ukernel.get('grid_footprint')
        if not fp: raise ValueError('DRAM-tier kernel without a derived grid footprint (refused, no fallback)')
        wscale = min(1.0, fp['write_footprint_sectors'] * 32 / d_w); d_w *= wscale
    dram_kernel = ((d_r + d_w) / V3.curve_bw(V3E['dram_total_bandwidth_TBps'], d_r / (d_r + d_w)) if d_r + d_w else 0.0) * scale
    terms = []
    for x in rows:
        ph = x['ph']; b = x['l2_reads'] * 32 + x['writes'] * 32
        db = x['dram_r'] + (x['writes'] * 32 * wscale if tier == 'DRAM' else 0.0)
        l2 = l2_kernel * (b / l2_tot) if l2_tot else 0.0
        dram = dram_kernel * (db / (d_r + d_w)) if d_r + d_w else 0.0
        l1 = x['l1_sectors'] * 32 / (V3c['l1']['bandwidth_bytes_per_s_per_sm'] * active) + ph['lines'] * S['c_line_ns'] * 1e-9 * scale
        served = x['dram_r'] if tier == 'DRAM' else x['l2_reads'] * 32
        mlp = V3F.mlp_seconds(served, ph, occ, blocks, tier, V3C, active)
        latency = occ['waves'] * (ph['dependent_global_load_depth'] * V2.LATENCY_NS[tier] * 1e-9 + ph['critical_path_compute_instructions'] * K['alu_latency_cycles'] / K['clock'])
        issue = V3.issue_cycles(ph, K) / (active * K['clock'])
        terms.append(dict(stream_s=max(l2, l1, dram, mlp), l2_s=l2, l1_s=l1, dram_s=dram, mlp_s=mlp, latency_s=latency, issue_s=issue, repetitions=ph['repetitions']))
    w = occ.get('warps_per_block') or 1
    barrier = occ['waves'] * (barrier_releases or 0) * float(np.interp(w, K['barrier_xs'], K['barrier_ys'])) / K['clock']
    totals = {k: sum(p[k] * p['repetitions'] for p in terms) for k in ('stream_s', 'latency_s', 'issue_s')}
    L = V3c['launch_us_per_kernel']; t0 = V3E['kernel_fixed_overhead_us'] * 1e-6; dispatch = L['per_extra_block'] * (blocks - 1) * 1e-6
    body = sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms) + barrier
    traffic = dict(l2_read_bytes=l2_r, l2_write_bytes=l2_w, dram_read_bytes=d_r, dram_write_bytes=d_w, first_touch_read_bytes=ft_total * 32.0, tier=tier)
    return dict(primary_s=t0 + max(body, dispatch), max_s=t0 + max(max(totals.values()) + barrier, dispatch), sum_s=t0 + max(sum(totals.values()) + barrier, dispatch),
                barrier_s=barrier, launch_s=t0, dispatch_s=dispatch, dram_unique_read_sectors=uniq, phase_terms=terms, traffic=traffic)


@contextlib.contextmanager
def _installed(**options):
    old_opt = dict(OPTIONS); old_kt = V3F.kernel_terms
    OPTIONS.update(options); V3F.kernel_terms = kernel_terms
    try: yield
    finally:
        OPTIONS.clear(); OPTIONS.update(old_opt); V3F.kernel_terms = old_kt


def attach_footprints(unique_rows, footprints, l2_capacity_bytes):
    """Copy of the first-touch tables with each kernel's derived footprint (footprints: {cell_id: {status, kernels: [...]}} from run_footprints.py) attached.
    A cell or kernel without a footprint gets none; the model refuses it only if it needs one (DRAM tier)."""
    out = {}
    for cid, row in unique_rows.items():
        row = dict(row); fp = footprints.get(cid)
        if fp and fp.get('status') == 'ok' and row.get('kernels') and len(fp['kernels']) == len(row['kernels']):
            row['kernels'] = [dict(k, grid_footprint=dict(f, l2_capacity_bytes=l2_capacity_bytes, reuse_working_set_bytes=reuse_working_set_bytes(f))) for k, f in zip(row['kernels'], fp['kernels'])]
        out[cid] = row
    return out


def build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E, OV, read_footprint=True, write_footprint=False, l2_rule='wave'):
    """Same signature as predict_runtime_v3i.build plus the option. Cell results gain `traffic` (summed over the cell's kernels)."""
    with _installed(read_footprint=read_footprint, write_footprint=write_footprint, l2_rule=l2_rule):
        out = V3I.build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E, OV)
    for cid, o in out.items():
        if o.get('primary_s') is None: continue
        ks = o.get('kernels') or []
        if not ks or any('traffic' not in k for k in ks): continue
        o['traffic'] = {key: sum(k['traffic'][key] for k in ks) for key in ('l2_read_bytes', 'l2_write_bytes', 'dram_read_bytes', 'dram_write_bytes', 'first_touch_read_bytes')}
    return out
