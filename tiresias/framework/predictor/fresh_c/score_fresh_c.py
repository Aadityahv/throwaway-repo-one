"""Score the frozen round-C predictions (v2, v3e, v3f) against timed runtimes. Run once, after timing_fresh_c.py. Runtime error only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MODELS = ('v2', 'v3e', 'v3f')
def stats(v):
    v = np.asarray(v, float)
    return None if not v.size else dict(n=int(v.size), median_pct=round(float(np.median(v)), 2), p90_pct=round(float(np.percentile(v, 90)), 2), mean_pct=round(float(v.mean()), 2))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timing', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args(); a.timing = a.timing.resolve()
    cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_c.json').read_text())['cells']}
    T = {r['cell_id']: r['per_launch_runtime_s'] for r in json.loads(a.timing.read_text())['cells']}
    if set(T) != set(cells): raise SystemExit('timing must cover exactly the 40 set-C cells')
    P = {m: json.loads((HERE / f'predictions_fresh_c_{m}.json').read_text()) for m in MODELS}
    rep = dict(inputs={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.timing] + [HERE / f'predictions_fresh_c_{m}.json' for m in MODELS]}, models={})
    for m, pred in P.items():
        sup = [k for k in cells if pred[k].get('primary_s')]; err = {k: abs(pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}; sgn = {k: (pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}
        grp = lambda f: stats([err[k] for k in sup if f(cells[k])])
        rep['models'][m] = dict(supported=len(sup), unsupported=len(cells) - len(sup), runtime_error=stats(list(err.values())),
            runtime_error_unsupported_as_failure=stats(list(err.values()) + [float('inf')] * (len(cells) - len(sup))), signed_median_pct=round(float(np.median(list(sgn.values()))), 2),
            per_operator={op: grp(lambda c, op=op: c['operator_id'] == op) for op in sorted({c['operator_id'] for c in cells.values()})},
            per_regime={r: grp(lambda c, r=r: c['regime'] == r) for r in ('small', 'medium', 'large', 'dram1', 'dram2')}, per_tier={t: grp(lambda c, t=t: c['tier'] == t) for t in ('L2', 'DRAM')},
            per_kernel_count={'one_kernel_c1': grp(lambda c: c['candidate_id'] == 'c1'), 'two_kernel_padded_c2_c4': grp(lambda c: c['candidate_id'] != 'c1')},
            per_cell={k: dict(measured_us=round(T[k] * 1e6, 3), predicted_us=round(pred[k]['primary_s'] * 1e6, 3) if pred[k].get('primary_s') else None, signed_error_pct=round(sgn[k], 1) if k in sgn else None, unsupported_reason=pred[k].get('unsupported_reason')) for k in sorted(cells)})
    common = [k for k in cells if all(P[m][k].get('primary_s') for m in MODELS)]
    e = {m: [abs(P[m][k]['primary_s'] / T[k] - 1) * 100 for k in common] for m in MODELS}
    rep['same_supported_cells'] = dict(n=len(common), **{m: round(float(np.median(e[m])), 2) for m in MODELS}, v3f_closer_than_v2=int(sum(b < c for b, c in zip(e['v3f'], e['v2']))), v3f_closer_than_v3e=int(sum(b < c for b, c in zip(e['v3f'], e['v3e']))))
    f = rep['models']['v3f']; s = rep['same_supported_cells']
    rep['verdict'] = dict(v3f_median_le_15_supported=bool(f['runtime_error']['median_pct'] <= 15), v3f_median_le_15_unsupported_as_failure=bool(f['runtime_error_unsupported_as_failure']['median_pct'] is not None and f['runtime_error_unsupported_as_failure']['median_pct'] <= 15),
        v3f_beats_v2=bool(s['v3f'] < s['v2']), v3f_beats_v3e=bool(s['v3f'] < s['v3e']), note='Round C: new shapes of two PyTorch kernels incl. two DRAM sizes; 4 cells unsupported (refused by the static pipeline); runtime only.')
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n'); print(json.dumps({'common_medians': s, 'v3f': f['runtime_error'], 'v3f_unsupported_as_failure': f['runtime_error_unsupported_as_failure'], 'verdict': rep['verdict']}, indent=1))
if __name__ == '__main__': main()
