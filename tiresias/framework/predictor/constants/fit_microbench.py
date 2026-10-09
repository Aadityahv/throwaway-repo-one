"""Fit hardware constants from the isolated microbenchmark grid (static runtime v2).

Preregistered in planning/STATIC_PREDICTOR_V2_OVERNIGHT_2026-10-02.md. Reads only the retained grid
output of one campaign (cells/<slot>/{sampleN.bin.gz,result.json}) and the packet rows; no operator data.

Per compute family f, stream count S, thread count, blocks, the device-clock cycles of the timed loop obey
    cycles = c0 + k*c_loop + k*u*slope        (k = loops, u = unroll; both vary in the packet)
fitted by least squares. Then, with w = warps per SM:
    dependent latency   L_f  = slope / (S*ops) at S=1, one block, one warp (threads=32)
    issue cost          I_f  = slope / (w*S*ops) at S=4, one block per SM, 1024 threads (cycles per warp-instruction per SM)
    barrier latency     B(w) = slope at family 0, S=1, one block, w = threads/32
ops = integer instructions per step (2 for the integer family, else 1).
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY_NAMES = ['barrier', 'integer', 'fp32_add', 'fp32_fma', 'mufu_ex2', 'shuffle', 'shared_load', 'shared_store_load']
OPS_PER_STEP = {1: 2}  # x = x + (x ^ c): two integer ops per step; every other family issues one


def load_cell(cell_dir, row):
    """Device-cycle span of the timed loop per SM, and the median event time (ms).

    The span of a block is the span of its LAST-finishing warp (maximum over its warps), averaged over
    blocks; the median over the three samples is returned. A median-warp span understates SM time when
    warps run unevenly (fp32 cells: first warp 68k cycles, last 277k, event time 298k), see plan
    amendment 4. For one-warp groups the two definitions coincide.
    """
    spans = []
    for s in range(3):
        raw = gzip.open(Path(cell_dir) / f'sample{s}.bin.gz', 'rb').read()
        lanes = row['blocks'] * row['threads']; warps = lanes // 32
        out_bytes = lanes * row['streams'] * 4
        cycles = np.frombuffer(raw[out_bytes:out_bytes + warps * 8], dtype='<u8').astype(float)
        spans.append(float(cycles.reshape(row['blocks'], row['threads'] // 32).max(axis=1).mean()))
    result = json.loads((Path(cell_dir) / 'result.json').read_text())
    return float(np.median(spans)), float(np.median(result['event_ms']))


def fit_group(points):
    """points: list of (k, u, cycles). Returns (c0, c_loop, slope, max_rel_residual)."""
    A = np.array([[1.0, k, k * u] for k, u, _ in points]); y = np.array([c for _, _, c in points])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ coef - y) / np.maximum(np.abs(y), 1.0)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(resid.max())


def fit(table, rows_by_slot):
    """table: {slot: (cycles, event_ms)} for compute cells. Returns the constants document."""
    groups = {}
    for slot, (cycles, _) in table.items():
        r = rows_by_slot[slot]
        if r['kind'] != 'compute': continue
        groups.setdefault((r['family'], r['streams'], r['threads'], r['blocks']), []).append((r['loops'], r['unroll'], cycles))
    fits = {key: fit_group(pts) for key, pts in groups.items() if len({(k, u) for k, u, _ in pts}) >= 4}
    out = {'dependent_latency_cycles': {}, 'issue_cycles_per_warp_instruction_per_sm': {}, 'barrier_latency_cycles_by_warps': {},
           'loop_overhead_cycles_per_iteration': {}, 'fit_quality_max_relative_residual': {}}
    # Effective SM clock during the grid: device cycles over event time, on the longest compute cells.
    longest = sorted((c for c, _ in table.values()), reverse=True)[:max(1, len(table) // 20)]
    clocks = [c / (ms * 1e-3) for c, ms in table.values() if c in longest and ms > 0]
    out['effective_sm_clock_hz'] = float(np.median(clocks)) if clocks else None
    for (f, S, t, b), (c0, c_loop, slope, res) in fits.items():
        name = FAMILY_NAMES[f]; ops = OPS_PER_STEP.get(f, 1)
        if f == 0 and S == 1 and b == 1:
            out['barrier_latency_cycles_by_warps'][str(t // 32)] = slope
        if f != 0 and S == 1 and b == 1 and t == 32:
            out['dependent_latency_cycles'][name] = slope / (S * ops)
            out['loop_overhead_cycles_per_iteration'][name] = c_loop
        if f != 0 and S == 4 and b == 188 and t == 1024:
            out['issue_cycles_per_warp_instruction_per_sm'][name] = slope / ((t // 32) * S * ops)
        out['fit_quality_max_relative_residual'][f'{name}/S{S}/t{t}/b{b}'] = res
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--campaign-dir', type=Path, required=True)
    ap.add_argument('--packet', type=Path, required=True); ap.add_argument('--out', type=Path, default=HERE / 'microbench_constants_v2.json')
    a = ap.parse_args()
    rows = {r['slot']: r for r in json.loads(a.packet.read_text())['rows']}
    table = {slot: load_cell(a.campaign_dir / 'cells' / slot, rows[slot]) for slot in rows if rows[slot]['kind'] == 'compute'}
    doc = fit(table, rows); doc['source'] = {'campaign_dir': str(a.campaign_dir), 'packet': str(a.packet), 'cells_used': len(table)}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n'); print(json.dumps(doc, indent=1, sort_keys=True)[:2500])


if __name__ == '__main__':
    main()
