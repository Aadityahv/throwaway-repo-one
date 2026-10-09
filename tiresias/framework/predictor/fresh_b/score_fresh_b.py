"""Score the frozen round-B predictions (v2, v3d, v3e) against timed fresh-B runtimes. Run once, after timing_fresh_b.py."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P0, CAP = 152.52913482177462, 600.0  # frozen energy model (Blackwell); eps per tier from the development table
EPS = {'L2': 80.642125766766, 'DRAM': 179.59310363418655}
def energy(t, nbytes, tier): return min(P0 * t + EPS[tier] * 1e-12 * nbytes, CAP * t)
def stats(v):
    v = np.asarray(v, float)
    return None if not v.size else dict(n=int(v.size), median_pct=round(float(np.median(v)), 2), p90_pct=round(float(np.percentile(v, 90)), 2), mean_pct=round(float(v.mean()), 2))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timing', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args(); a.timing = a.timing.resolve()
    cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_b.json').read_text())['cells']}
    T = {r['cell_id']: r['per_launch_runtime_s'] for r in json.loads(a.timing.read_text())['cells']}
    if set(T) != set(cells): raise SystemExit('timing must cover exactly the 32 set-B cells')
    models = ('v2', 'v3d', 'v3e'); P = {m: json.loads((HERE / f'predictions_fresh_b_{m}.json').read_text()) for m in models}
    rep = dict(inputs={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.timing] + [HERE / f'predictions_fresh_b_{m}.json' for m in models]}, models={})
    for m, pred in P.items():
        sup = [k for k in cells if pred[k].get('primary_s')]
        err = {k: abs(pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}; sgn = {k: (pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}
        nb = lambda k: cells[k]['logical_read_bytes'] + cells[k]['logical_write_bytes']
        eng = [abs(energy(pred[k]['primary_s'], nb(k), cells[k]['tier']) / energy(T[k], nb(k), cells[k]['tier']) - 1) * 100 for k in sup]
        grp = lambda f: stats([err[k] for k in sup if f(cells[k])])
        rep['models'][m] = dict(supported=len(sup), runtime_error=stats(list(err.values())), runtime_error_unsupported_as_failure=stats(list(err.values()) + [float('inf')] * (len(cells) - len(sup))),
            signed_median_pct=round(float(np.median(list(sgn.values()))), 2) if sgn else None, implied_energy_error_from_runtime_only=stats(eng),
            per_operator={op: grp(lambda c, op=op: c['operator_id'] == op) for op in sorted({c['operator_id'] for c in cells.values()})},
            per_regime={r: grp(lambda c, r=r: c['regime'] == r) for r in ('small', 'medium', 'large', 'xlarge')},
            per_tier={t: grp(lambda c, t=t: c['tier'] == t) for t in ('L2', 'DRAM')},
            per_kernel_count={'one_kernel_c1': grp(lambda c: c['candidate_id'] == 'c1'), 'two_kernel_padded_c2_c4': grp(lambda c: c['candidate_id'] != 'c1')},
            per_cell={k: dict(measured_us=round(T[k] * 1e6, 3), predicted_us=round(pred[k]['primary_s'] * 1e6, 3) if pred[k].get('primary_s') else None, signed_error_pct=round(sgn[k], 1) if k in sgn else None) for k in sorted(cells)})
    e = {m: [abs(P[m][k]['primary_s'] / T[k] - 1) * 100 for k in cells] for m in models}
    rep['same_cells'] = {m: round(float(np.median(e[m])), 2) for m in models}; rep['cells_v3e_closer_than_v2'] = int(sum(b < c for b, c in zip(e['v3e'], e['v2']))); rep['cells_v3e_closer_than_v3d'] = int(sum(b < c for b, c in zip(e['v3e'], e['v3d'])))
    m = rep['models']['v3e']; rep['verdict'] = dict(v3e_runtime_median_le_15=bool(m['runtime_error']['median_pct'] <= 15), v3e_beats_v2=bool(rep['same_cells']['v3e'] < rep['same_cells']['v2']),
        v3e_beats_v3d=bool(rep['same_cells']['v3e'] < rep['same_cells']['v3d']), note='Round B: new shapes of the two PyTorch kernels incl. DRAM-tier cells; not new source lineages; energy not measured (implied only).')
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n'); print(json.dumps({'medians': rep['same_cells'], 'v3e': rep['models']['v3e']['runtime_error'], 'verdict': rep['verdict']}, indent=1))
if __name__ == '__main__': main()
