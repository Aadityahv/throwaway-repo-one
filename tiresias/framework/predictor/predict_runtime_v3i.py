"""Static runtime predictions, model v3i (preregistered in planning/STATIC_RUNTIME_V3I_PREREGISTRATION_2026-10-02.md).

The bank-conflict model (v3h) with one change in how barrier phases combine. v3h sums a per-phase maximum (phases strictly serial);
the plain maximum over whole-kernel resource totals assumes phases of co-resident blocks overlap fully. The overlap microbenchmark
(micro_v3/out_overlap.jsonl) measured the actual overlap fraction alpha as a function of resident blocks per SM (0.12, 0.48, 0.50, 0.59 at
1, 2, 3, 6). Per kernel:  T = (1 - alpha) * T_serial + alpha * T_full,  with T_serial = v3h's per-phase-maximum sum and T_full = the plain
maximum, both already including the fixed overhead and dispatch bound. alpha = interp(resident blocks per SM) with
resident = min(blocks per SM from occupancy, ceil(grid blocks / SM count)), clamped to the measured range. Nothing is fitted on any operator cell.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import predict_runtime_v2 as V2
import predict_runtime_v3h as V3H

HERE = Path(__file__).resolve().parent


def build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E, OV):
    out = V3H.build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E)
    sm = features['hardware_from_ground_truth']['sm_count']
    for f in features['rows']:
        cid = f['cell_id']; o = out[cid]
        if o.get('primary_s') is None: continue
        secondary = {k['kernel_id']: k for k in f.get('secondary_kernels', [])}; ph = phase_rows[cid]
        try:
            total = 0.0; total_max = 0.0; alphas = []
            for kk, pred in zip(ph['kernels'], o['kernels']):
                meta = secondary.get(kk.get('kernel_id'), f)
                geo = meta['geometry']; blocks = geo.get('grid_blocks') or geo.get('grid_blocks_dispatch')
                resident = min(meta['occupancy']['blocks_per_sm'], math.ceil(blocks / sm))
                a = float(np.interp(resident, OV['resident_blocks_per_sm'], OV['alpha'])); alphas.append(a)
                pred['overlap_alpha'] = a; pred['primary_v3h_s'] = pred['primary_s']
                pred['primary_s'] = (1 - a) * pred['primary_s'] + a * pred['max_s']
                total += pred['primary_s']
            o['primary_v3h_s'] = o['primary_s']; o['primary_s'] = total; o['overlap_alphas'] = alphas
        except (KeyError, TypeError, ValueError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason='overlap input unavailable: ' + str(ex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True); ap.add_argument('--phases', type=Path, required=True)
    ap.add_argument('--unique', type=Path, required=True); ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    features = json.loads(a.features.read_text()); phases = json.loads(a.phases.read_text()); phases = phases.get('rows', phases)
    uniq = json.loads(a.unique.read_text()); uniq = uniq.get('rows', uniq); bank = json.loads(a.bank.read_text())['rows']
    C = HERE / 'constants'; stream = json.loads((C / 'stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((C / 'microbench_constants_v2.json').read_text()))
    out = build(features, phases, uniq, bank, K, json.loads((C / 'v3_constants.json').read_text()), json.loads((C / 'v3c_constants.json').read_text()),
                json.loads((C / 'v3e_constants.json').read_text()), json.loads((C / 'overlap_constants.json').read_text()))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(cells=len(out), supported=sum(v['primary_s'] is not None for v in out.values()))))


if __name__ == '__main__':
    main()
