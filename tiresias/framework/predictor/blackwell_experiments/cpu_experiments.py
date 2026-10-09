"""CPU-only Blackwell experiments from archived data (A3, A4, E5). No GPU, no remote machine.

    python3 cpu_experiments.py [--out results.json]

Reads: the 27 archived energy windows of the packaged-calibrator run (`calibrate/runs/energy_20261002/run_full/stages/energy/windows.json`,
as-designed admission), the 48 target cells (with published-method rows) and the 120 exposed development cells, exactly as
`energy_stage_assistant_a_20261002/evaluate_calibration.py` loads them. Never changes a pre-registered bar.

A4 and E5 refit the energy profile with the packaged fitter (`cal.energy.fit_rates`) after removing one term or a subset of windows,
and score on the cells that were never used in a fit. The terms are a fitted attribution, not a physical split of power.
A3 is descriptive: how much of the variation in measured energy is explained by runtime alone, traffic alone, or both.
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; SR = HERE.parent; U = SR.parent
sys.path.insert(0, str(SR / 'calibrate'))
import predict as P  # noqa: E402
from cal import energy as En  # noqa: E402

RUN = SR / 'calibrate/runs/energy_20261002/run_full'
BELOW_CAP_W = 570  # same below-cap rule as evaluate_calibration.py


def load_cells():
    feats = {}
    for p in (SR / 'fresh_e/features_fresh_e.json', SR / 'unseen_kernels/frozen/features_unseen.json', SR / 'features_blackwell.json'):
        for r in json.loads(p.read_text())['rows']: feats[r['cell_id']] = r
    pc = pd.DataFrame(list(csv.DictReader((U / 'evaluation_data/measured/kernel_sets/per_cell.csv').open())))
    for c in ('measured_energy_j', 'predicted_energy_j', 'measured_window_power_w', 'measured_short_runtime_s'): pc[c] = pc[c].astype(float)
    tgt = pc[pc.role == 'ablation'].reset_index(drop=True)
    rt_static = {}
    for p in (SR / 'fresh_e/predictions_fresh_e_v3i.json', SR / 'bank/predictions_unseen_v3i.json'): rt_static.update({k: v['primary_s'] for k, v in json.loads(p.read_text()).items()})
    d = pd.read_csv(U / 'compile_evidence/development/development_cells.csv'); d = d[(d.gpu == 'blackwell') & d.cell_id.isin(feats)].reset_index(drop=True)
    dv = json.loads((SR / 'predictions_v3e_dev_blackwell.json').read_text())
    return feats, tgt, rt_static, d, {c: (dv.get(c) or {}).get('primary_s') for c in d.cell_id}


def score(doc, feats, ids, E, below, rt):
    rows = P.energy_rows(doc, {'rows': [feats[c] for c in ids]}, {c: t for c, t in rt.items() if t})
    ok = np.array([rows[c]['status'] == 'ok' for c in ids])
    pred = np.array([rows[c]['energy_j'] if rows[c]['status'] == 'ok' else 1e-12 for c in ids])  # unsupported counted as failures
    e = np.abs(pred / E - 1) * 100
    m = below
    return dict(median=float(np.median(e[m])), p90=float(np.percentile(e[m], 90)), median_all=float(np.median(e)), n_below=int(m.sum()), n_unsupported=int((~ok).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=HERE / 'cpu_experiments_results.json'); ap.add_argument('--draws', type=int, default=300); a = ap.parse_args()
    windows = json.loads((RUN / 'stages/energy/windows.json').read_text())
    doc0 = P.load_calibration(next(RUN.glob('calibration_sm_*[0-9a-f].json')), allow_incomplete=True); cap = doc0['constants']['energy']['cap_w']
    feats, tgt, rts_t, dev, rts_d = load_cells()
    ids_t = list(tgt.cell_id); Et = tgt.measured_energy_j.values; bt = (tgt.measured_window_power_w < BELOW_CAP_W).values
    rtm_t = dict(zip(tgt.cell_id, tgt.measured_short_runtime_s))
    ids_d = list(dev.cell_id); Ed = dev.energy_j.values; bd = (dev.measured_power_w < BELOW_CAP_W).values; rtm_d = dict(zip(dev.cell_id, dev.runtime_s))
    fitw = [w for w in windows if w['role'] == 'fit' and w['admitted']]; held = [w for w in windows if w['role'] == 'heldout' and w['admitted']]

    def evaluate(profile):
        doc = copy.deepcopy(doc0); doc['constants']['energy']['profile'] = profile
        h = En.heldout_errors(profile, held, cap)
        return dict(heldout_abs_max=max(abs(v) for v in h.values()), heldout=h,
                    target_static=score(doc, feats, ids_t, Et, bt, rts_t), target_measured=score(doc, feats, ids_t, Et, bt, rtm_t),
                    dev_static=score(doc, feats, ids_d, Ed, bd, rts_d), dev_measured=score(doc, feats, ids_d, Ed, bd, rtm_d))

    res = dict(note='as-designed admission (21 of 27 windows); below cap = measured window power < %d W; unsupported cells counted as failures' % BELOW_CAP_W,
               n_fit_windows=len(fitw), n_heldout_windows=len(held))
    full = En.fit_rates(fitw); res['full'] = dict(profile={k: full[k] for k in ('base_power_w', 'rates_pJ', 'status')}, **evaluate(full))

    # A4 / E5a: leave-one-term-out. Zeroing a column removes its support, so the fitter drops it and the runtime term absorbs it.
    res['leave_one_term_out'] = {}
    for c in En.COLUMNS:
        ws = copy.deepcopy(fitw)
        for w in ws: w['columns'][c] = 0.0
        try: p = En.fit_rates(ws)
        except ValueError as e: res['leave_one_term_out'][c] = dict(error=str(e)); continue
        res['leave_one_term_out'][c] = dict(base_power_w=p['base_power_w'], **evaluate(p))
    # base-power term removed: runtime carries no free constant (base fixed to zero)
    p0 = En.fit_rates(fitw, base_w=0.0); res['no_runtime_term'] = dict(base_power_w=0.0, **evaluate(p0))

    # A4: attributed share of predicted energy per term on the target cells (measured runtime, below cap): fitted attribution only
    doc = copy.deepcopy(doc0); rows = P.energy_rows(doc, {'rows': [feats[c] for c in ids_t]}, rtm_t)
    share = {c: [] for c in ['runtime'] + En.COLUMNS}
    for c, m in zip(ids_t, bt):
        r = rows[c]
        if not m or r['status'] != 'ok': continue
        tot = r['base_term_j'] + sum(r['term_j'].values()); share['runtime'].append(r['base_term_j'] / tot)
        for k in En.COLUMNS: share[k].append(r['term_j'][k] / tot)
    res['attributed_share_median_target_below_cap'] = {k: float(np.median(v)) for k, v in share.items()}

    # E5b: calibration-window count sensitivity (random subsets of admitted fit windows that keep every column's support >= 2)
    rng = np.random.default_rng(20261002); n = len(fitw); res['window_count'] = {}
    for k in (8, 10, 12, 14, 16, 18, n):
        meds, p90s, fails = [], [], 0
        for _ in range(a.draws if k < n else 1):
            idx = rng.choice(n, k, replace=False) if k < n else np.arange(n)
            sub = [fitw[i] for i in idx]
            try: p = En.fit_rates(sub)
            except ValueError: fails += 1; continue
            if any(v.startswith('UNIDENTIFIED') for v in p['status'].values()): fails += 1; continue  # a term without support is not a usable calibration
            s = evaluate(p)['target_measured']; meds.append(s['median']); p90s.append(s['p90'])
        res['window_count'][str(k)] = dict(draws_ok=len(meds), draws_rejected=fails, median_of_median=float(np.median(meds)) if meds else None,
                                           p10=float(np.percentile(meds, 10)) if meds else None, p90=float(np.percentile(meds, 90)) if meds else None, p90_cell_error_median=float(np.median(p90s)) if p90s else None)

    # E5c: leave-one-window-out influence (one admitted fit window removed at a time)
    res['leave_one_window_out'] = {}
    for i, w in enumerate(fitw):
        sub = fitw[:i] + fitw[i + 1:]
        try: p = En.fit_rates(sub)
        except ValueError as e: res['leave_one_window_out'][w['design']] = dict(error=str(e)); continue
        unid = [c for c, v in p['status'].items() if v.startswith('UNIDENTIFIED')]
        res['leave_one_window_out'][w['design']] = dict(unidentified=unid, target_measured_median=evaluate(p)['target_measured']['median'] if not unid else None)

    # A3: energy vs runtime vs traffic on the measured cells (target + development), log-log R^2, all and below cap
    def r2(y, X):
        X = np.column_stack([np.ones(len(y))] + X); b, *_ = np.linalg.lstsq(X, y, rcond=None); return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    cells = []
    for c, E, t, pw in list(zip(ids_t, Et, tgt.measured_short_runtime_s, tgt.measured_window_power_w)) + list(zip(ids_d, Ed, dev.runtime_s, dev.measured_power_w)):
        f = feats[c]
        if not P.supported(f): continue
        tot = f.get('per_launch_totals') or f['work']; mem = f['memory']
        cells.append(dict(E=E, t=t, P=pw, B=mem['logical_bytes_per_launch'], N=tot['total_lane_instructions']))
    C = pd.DataFrame(cells); C = C[(C.E > 0) & (C.t > 0) & (C.B > 0) & (C.N > 0)]
    res['A3'] = {}
    for lab, m in (('all', np.ones(len(C), bool)), ('below_cap', (C.P < BELOW_CAP_W).values)):
        S = C[m]; y = np.log(S.E.values); lt, lb, ln = np.log(S.t.values), np.log(S.B.values), np.log(S.N.values)
        res['A3'][lab] = dict(n=int(len(S)), r2_runtime_only=r2(y, [lt]), r2_traffic_only=r2(y, [lb]), r2_instructions_only=r2(y, [ln]), r2_traffic_and_instructions=r2(y, [lb, ln]),
                              r2_runtime_and_traffic=r2(y, [lt, lb]), r2_runtime_traffic_instructions=r2(y, [lt, lb, ln]),
                              power_w_median=float(S.P.median()), share_within_5pct_of_cap=float((S.P >= 0.95 * cap).mean()))
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
