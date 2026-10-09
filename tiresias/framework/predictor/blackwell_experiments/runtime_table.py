"""Runtime table of the evaluation (CPU only): median absolute percentage error of the static runtime model per kernel, on the unseen kernels (32 cells), the CUDA-samples kernels at new shapes
(set D, 27 scored cells) and the scalar-product / Walsh-transform set (set E, 16 cells). Same layout as the table the user specified. Two columns of predictions:
  frozen model: the runtime model as frozen before timing (committed constants, no shared-memory re-cost)  -- the number the earlier tables report
  calibrated model: the packaged calibration tool's constants from the final validation run of the current code (smem re-cost from the calibrated floor and slope), through cal/portable_predict.py
Measured runtime is the committed timing of each set (never re-timed here). Cells that failed their correctness check are excluded and listed; unsupported cells would count as failures."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR / 'calibrate'))
from cal import portable_predict as PP  # noqa: E402
CAL_CONST = SR / 'calibrate/runs/validation_20261002/attempt8/legacy_constants'


def load(d, f, p, u, b):
    g = lambda *x: json.loads((SR.joinpath(*x)).read_text())
    ph = g(d, p); ph = ph.get('rows', ph); uq = g(d, u); uq = uq.get('rows', uq)
    return g(d, f), ph, uq, g('bank', b)['rows']


def timing(path, key='cells'):
    out, bad = {}, []
    for r in json.loads(path.read_text())[key]:
        if r.get('correct', r.get('check_ok', True)) is False: bad.append(r['cell_id'])
        else: out[r['cell_id']] = r['per_launch_runtime_s']
    return out, bad


def group_d(c):
    op = c['operator_id']
    return {'fresh_d_cuda_samples_copy': 'Copy', 'fresh_d_cuda_samples_transposefine': 'Fine- and coarse-grained transpose', 'fresh_d_cuda_samples_transpose': 'Transposes (naive, coalesced, padded, diagonal)',
            'fresh_d_cuda_samples_vector_add': 'Vector add', 'fresh_d_cuda_samples_reduction': 'Reductions (reduce0/1/6)', 'fresh_d_cuda_samples_reduce2': 'Reduce2'}[op]


UNSEEN = {'unseen_cuda_samples_black_scholes': 'Black-Scholes', 'unseen_cuda_samples_separable_convolution': 'Separable convolution', 'unseen_cuda_samples_scan': 'Prefix-sum scan', 'unseen_cuda_samples_matrix_multiply': 'Tiled matrix multiply'}
EOPS = {'fresh_e_cuda_samples_scalar_product': 'Scalar product', 'fresh_e_cuda_samples_fast_walsh_transform': 'Fast Walsh transform'}
SM = 188


def run():
    sets = []
    CC = PP.load_constants(CAL_CONST); CF = PP.load_constants(SR / 'constants'); CF['smem'] = None
    # unseen
    feats, ph, uq, bank = load('unseen_kernels/frozen', 'features_unseen.json', 'phases_unseen.json', 'phases_unique_unseen.json', 'bank_conflicts_unseen_wavefront.json')
    cells = {c['cell_id']: c for c in json.loads((SR / 'unseen_kernels/frozen/cells_unseen.json').read_text())['cells']} if 'cells' in json.loads((SR / 'unseen_kernels/frozen/cells_unseen.json').read_text()) else None
    T, bad = timing(SR / 'unseen_kernels/measured/timing_unseen_result.json')
    sets.append(('Unseen kernels', feats, ph, uq, bank, T, bad, lambda cid: UNSEEN[cid.split('/')[1]]))
    feats, ph, uq, bank = load('fresh_d', 'features_fresh_d.json', 'phases_fresh_d.json', 'phases_unique_fresh_d.json', 'bank_conflicts_fresh_d.json')
    dc = {c['cell_id']: c for c in json.loads((SR / 'fresh_d/fresh_cells_d.json').read_text())['cells']}
    T, bad = timing(SR / 'fresh_d/timing_fresh_d_result.json')
    sets.append(('CUDA-samples kernels at new shapes', feats, ph, uq, bank, T, bad, lambda cid: group_d(dc[cid])))
    feats, ph, uq, bank = load('fresh_e', 'features_fresh_e.json', 'phases_fresh_e.json', 'phases_unique_fresh_e.json', 'bank_conflicts_fresh_e_wavefront.json')
    T, bad = timing(SR / 'fresh_e/timing_fresh_e_result.json')
    sets.append(('Scalar product and Walsh transform (prospective)', feats, ph, uq, bank, T, bad, lambda cid: EOPS[cid.split('/')[1]]))
    rows, notes = [], []
    for name, feats, ph, uq, bank, T, bad, grp in sets:
        pred = {}
        for lab, C in (('frozen', CF), ('calibrated', CC)):
            try: pred[lab] = PP.predict(feats, ph, uq, bank, C, SM)
            except Exception as e: notes.append('%s %s: %s' % (name, lab, repr(e)[:200])); pred[lab] = {}
        ids = [k for k in T if k in {r['cell_id'] for r in feats['rows']}]
        by = {}
        for cid in ids:
            e = {}
            for lab in pred:
                v = pred[lab].get(cid); v = v.get('primary_s') if isinstance(v, dict) else v
                e[lab] = abs(v / T[cid] - 1) * 100 if v else None
            by.setdefault(grp(cid), []).append(e)
        for g, es in by.items():
            rows.append((name, g, len(es), {lab: (float(np.median([x[lab] for x in es if x[lab] is not None])) if any(x[lab] is not None for x in es) else None) for lab in pred}, sum(1 for x in es if x['frozen'] is None)))
        allv = {lab: [x[lab] for es in by.values() for x in es if x[lab] is not None] for lab in pred}
        rows.append((name, 'ALL', sum(len(es) for es in by.values()), {lab: float(np.median(v)) if v else None for lab, v in allv.items()}, 0))
        notes.append('%s: correctness-failed cells excluded: %s' % (name, bad))
    return rows, notes


if __name__ == '__main__':
    rows, notes = run()
    f = lambda x: '   n/a' if x is None else '%5.1f%%' % x
    print('%-48s %-48s %5s %10s %12s' % ('set', 'kernel / operator', 'cells', 'frozen', 'calibrated'))
    for s, g, n, m, u in rows: print('%-48s %-48s %5d %10s %12s' % (s, g, n, f(m.get('frozen')), f(m.get('calibrated'))))
    print('\n'.join(notes))
    (HERE / 'runtime_table.json').write_text(json.dumps([dict(set=s, kernel=g, cells=n, median_ape_pct=m, unsupported=u) for s, g, n, m, u in rows], indent=1))
