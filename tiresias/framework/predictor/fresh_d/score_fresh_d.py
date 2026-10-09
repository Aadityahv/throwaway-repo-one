"""Score the frozen set-D predictions (four runtime models) against timed runtimes. Run once. Runtime error only.
Cells whose correctness check failed are excluded from the criteria (and listed). Roofline baseline: t0 + logical bytes / measured peak tier bandwidth."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MODELS = ('v2', 'v3e', 'v3f', 'v3h')
def stats(v):
    v = np.asarray(v, float)
    return None if not v.size else dict(n=int(v.size), median_pct=round(float(np.median(v)), 2), p90_pct=round(float(np.percentile(v, 90)), 2), mean_pct=round(float(v.mean()), 2))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timing', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args(); a.timing = a.timing.resolve()
    cells = {c['cell_id']: c for c in json.loads((HERE / 'fresh_cells_d.json').read_text())['cells']}
    R = {r['cell_id']: r for r in json.loads(a.timing.read_text())['cells']}
    if set(R) != set(cells): raise SystemExit('timing must cover exactly the 28 set-D cells')
    T = {k: r['per_launch_runtime_s'] for k, r in R.items()}; bad = sorted(k for k, r in R.items() if not r['correct']); ok = [k for k in cells if k not in bad]
    S = json.loads((HERE.parent / 'constants/stream_constants.json').read_text())['constants']
    roof = {k: S['t0_us'] * 1e-6 + cells[k]['logical_bytes_per_launch'] / ((S['L2_read_sector_TBps'] if cells[k]['tier'] == 'L2' else S['DRAM_read_TBps']) * 1e12) for k in cells}
    P = {m: json.loads((HERE / f'predictions_fresh_d_{m}.json').read_text()) for m in MODELS}; P['roofline'] = {k: dict(primary_s=roof[k]) for k in cells}
    rep = dict(correctness_failed_excluded=bad, inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.timing] + [HERE / f'predictions_fresh_d_{m}.json' for m in MODELS]}, models={})
    for m, pred in P.items():
        sup = [k for k in ok if pred[k].get('primary_s')]; err = {k: abs(pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}; sgn = {k: (pred[k]['primary_s'] / T[k] - 1) * 100 for k in sup}
        grp = lambda f: stats([err[k] for k in sup if f(cells[k])])
        rep['models'][m] = dict(supported=len(sup), runtime_error=stats(list(err.values())), signed_median_pct=round(float(np.median(list(sgn.values()))), 2),
            per_operator={op: grp(lambda c, op=op: c['operator_id'] == op) for op in sorted({c['operator_id'] for c in cells.values()})}, per_tier={t: grp(lambda c, t=t: c['tier'] == t) for t in ('L2', 'DRAM')},
            per_cell={k: dict(measured_us=round(T[k] * 1e6, 3), predicted_us=round(pred[k]['primary_s'] * 1e6, 3), signed_error_pct=round(sgn[k], 1)) for k in sup})
    h = rep['models']['v3h']; r = rep['models']['roofline']; f = rep['models']['v3f']
    rep['verdict'] = dict(bank_model_median_le_15=bool(h['runtime_error']['median_pct'] <= 15), beats_roofline=bool(h['runtime_error']['median_pct'] < r['runtime_error']['median_pct']),
        beats_previous_model=bool(h['runtime_error']['median_pct'] < f['runtime_error']['median_pct']))
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
    for m in P: print(m, rep['models'][m]['runtime_error'], 'signed', rep['models'][m]['signed_median_pct'])
    print(rep['verdict'], 'excluded', bad)
if __name__ == '__main__': main()
