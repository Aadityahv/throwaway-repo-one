"""Score the frozen fresh-cell predictions (v2, v3d) against timed fresh runtimes. Run once, after timing_fresh.py."""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P0, EPS_L2, CAP = 152.52913482177462, 80.642125766766, 600.0  # frozen energy model, Blackwell L2-tier constants from the development table
def energy(t, nbytes): return min(P0 * t + EPS_L2 * 1e-12 * nbytes, CAP * t)
def stats(v):
    v = np.asarray(v, float)
    return None if not v.size else dict(n=int(v.size), median_pct=round(float(np.median(v)), 2), p90_pct=round(float(np.percentile(v, 90)), 2), mean_pct=round(float(v.mean()), 2))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timing', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args(); a.timing = a.timing.resolve()
    cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells.json').read_text())['cells']}
    T = {r['cell_id']: r['per_launch_runtime_s'] for r in json.loads(a.timing.read_text())['cells']}
    if set(T) != set(cells): raise SystemExit('timing must cover exactly the 24 fresh cells')
    P = {m: json.loads((HERE / f'predictions_fresh_{m}.json').read_text()) for m in ('v2', 'v3d')}
    rep = dict(inputs={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.timing, HERE / 'predictions_fresh_v2.json', HERE / 'predictions_fresh_v3d.json']}, models={})
    for m, pred in P.items():
        sup = [k for k in cells if pred[k].get('primary_s')]
        err = {k: abs(pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}; sgn = {k: (pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}
        eng = [abs(energy(pred[k]['primary_s'], cells[k]['logical_read_bytes'] + cells[k]['logical_write_bytes']) / energy(T[k], cells[k]['logical_read_bytes'] + cells[k]['logical_write_bytes']) - 1) * 100 for k in sup]
        grp = lambda f: stats([err[k] for k in sup if f(cells[k])])
        rep['models'][m] = dict(supported=len(sup), unsupported=len(cells) - len(sup), runtime_error=stats(list(err.values())),
            runtime_error_unsupported_as_failure=stats(list(err.values()) + [float('inf')] * (len(cells) - len(sup))),
            signed_median_pct=round(float(np.median(list(sgn.values()))), 2) if sgn else None,
            implied_energy_error_from_runtime_only=stats(eng), per_operator={op: grp(lambda c, op=op: c['operator_id'] == op) for op in sorted({c['operator_id'] for c in cells.values()})},
            per_regime={r: grp(lambda c, r=r: c['regime'] == r) for r in ('small', 'medium', 'large')},
            per_kernel_count={'one_kernel_c1': grp(lambda c: c['candidate_id'] == 'c1'), 'two_kernel_padded_c2_c4': grp(lambda c: c['candidate_id'] != 'c1')},
            per_cell={k: dict(measured_us=round(T[k] * 1e6, 3), predicted_us=round(pred[k]['primary_s'] * 1e6, 3) if pred[k].get('primary_s') else None,
                              signed_error_pct=round(sgn[k], 1) if k in sgn else None, unsupported_reason=pred[k].get('unsupported_reason')) for k in sorted(cells)})
    both = [k for k in cells if P['v2'][k].get('primary_s') and P['v3d'][k].get('primary_s')]
    e2 = [abs(P['v2'][k]['primary_s'] / T[k] - 1) * 100 for k in both]; e3 = [abs(P['v3d'][k]['primary_s'] / T[k] - 1) * 100 for k in both]
    rep['same_cells_v2_vs_v3d'] = dict(n=len(both), v2_median_pct=round(float(np.median(e2)), 2), v3d_median_pct=round(float(np.median(e3)), 2), cells_v3d_better=int(sum(b < c for b, c in zip(e3, e2))))
    m3 = rep['models']['v3d']; rep['verdict'] = dict(
        runtime_median_le_15_supported=bool(m3['runtime_error'] and m3['runtime_error']['median_pct'] <= 15),
        runtime_median_le_15_unsupported_as_failure=bool(m3['runtime_error_unsupported_as_failure'] and m3['runtime_error_unsupported_as_failure']['median_pct'] is not None and m3['runtime_error_unsupported_as_failure']['median_pct'] <= 15),
        v3d_beats_v2_on_same_cells=bool(rep['same_cells_v2_vs_v3d']['v3d_median_pct'] < rep['same_cells_v2_vs_v3d']['v2_median_pct']),
        note='Fresh set: new shapes of the two PyTorch kernels, all L2-tier. Not new source lineages; energy not measured (implied energy only).')
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n'); print(json.dumps({'v2': rep['models']['v2']['runtime_error'], 'v3d': rep['models']['v3d']['runtime_error'], 'same_cells': rep['same_cells_v2_vs_v3d'], 'verdict': rep['verdict']}, indent=1))
if __name__ == '__main__': main()
