#!/usr/bin/env python3
"""Fit the energy rates from calibration windows PLUS labeled application kernels measured on the same GPU, and score the fit leave-one-operator-out. CPU only.

This is the fit that produced the Blackwell results of 2 Oct 2026 (static runtime 9.9% median below cap, measured runtime 9.7%). It is NOT calibrator-only: the application kernels
(their static features and their measured energy and runtime) are training data, so a new GPU needs those kernels measured too. Calibration windows alone are the other route
(`run_calibration.py` energy stage; `calibrate/predict.py` uses whichever profile the document carries).

    python3 calibrate/tools/fit_application_model.py --labels cells.csv --features f1.json [f2.json ...] \\
        (--calibration calibration_<arch>_<uuid8>.json | --legacy-windows compiled_design.json result.json) \\
        [--measured-base-w 152.529] [--write-profile out.json]

labels CSV columns: cell_id, operator, energy_j, runtime_s, runtime_pred_s (blank when no static runtime prediction exists), window_power_w, source.
Model (unchanged from the Blackwell fit): E = min(cap * t, base * t + e_L2*B_L2 + e_DRAM*B_DRAM + e_fma*N_fma/mul + e_shared*N_shared + e_shb*N_shuffle/barrier + e_sfu*N_sfu),
non-negative least squares on relative error, training on below-cap cells (window power under 570 W). Static-runtime fit: free base power, trained and scored on the runtime
model's predictions. Measured-runtime fit: base power fixed to --measured-base-w when given (the Blackwell development constant), else free."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import energy as En  # noqa: E402

FIT_COLUMNS = ['B_l2', 'B_dr', 'ffma', 'shm', 'shb', 'sfu']
BELOW_CAP_W = 570.0


def cell_columns(row):
    tot = row.get('per_launch_totals') or row['work']; mem = row['memory']
    c = En.columns_from_feature_row(tot, mem['logical_bytes_per_launch'], mem['tier'], tot.get('executed_global_store_bytes_lane_level', mem.get('executed_global_store_bytes_lane_level', 0)))
    c['shb'] = c['shfl'] + c['bar']
    return c


def load_cells(labels_csv, feature_files):
    feats = {}
    for p in feature_files:
        for r in json.loads(Path(p).read_text())['rows']:
            if r.get('status') in ('supported', 'supported_with_assumptions'): feats[r['cell_id']] = r
    rows = []
    for r in csv.DictReader(open(labels_csv)):
        if r['cell_id'] not in feats: continue
        c = cell_columns(feats[r['cell_id']]); pred = float(r['runtime_pred_s']) if r['runtime_pred_s'] else np.nan
        rows.append(dict(id=r['cell_id'], op=r['operator'], src=r['source'], E=float(r['energy_j']), t=float(r['runtime_s']), tp=pred, P=float(r['window_power_w']), **c))
    return rows


def windows_from_document(path):
    doc = json.loads(Path(path).read_text()); out = []
    for w in doc['constants']['energy']['windows']:
        if w['role'] != 'fit' or not w['admitted']: continue
        c = dict(w['columns']); c['shb'] = c.get('shfl', 0) + c.get('bar', 0)
        out.append(dict(id='CAL:' + w['id'], op='CAL:' + w['id'], src='cal', E=w['energy_j_per_launch'], t=w['runtime_s_per_launch'], tp=np.nan, P=w['power_w'], **c))
    return out


def windows_from_legacy(design_json, result_json):
    des = {r['design_id']: r for r in json.loads(Path(design_json).read_text())['rows']}; out = []
    for r in json.loads(Path(result_json).read_text())['rows']:
        if r['role'] != 'fit': continue
        d = des[r['design_id']]; fam = d['work']['families']; g = lambda *fs: float(sum(fam.get(f, {}).get('lane_instructions', 0) for f in fs)); dram = d['tier'] == 'DRAM'; B = float(d['logical_bytes'])
        t, E = r['counted_runtime_s_per_launch'], r['energy_j_per_launch']
        out.append(dict(id='CAL:' + r['design_id'], op='CAL:' + d['mix'], src='cal', E=E, t=t, tp=np.nan, P=E / t, B_l2=0.0 if dram else B, B_dr=B if dram else 0.0, ffma=g('fp32_fma', 'fp32_mul'),
                        shm=g('shared_load', 'shared_store', 'shared_matrix_load'), shb=g('shuffle', 'barrier'), sfu=g('special_function')))
    return out


def design(rows, train_t):
    X = np.array([[r[c] * 1e-12 for c in FIT_COLUMNS] for r in rows]); return X, train_t


def fit(rows, train_t, fixed_base_w=None):
    X = np.array([[r[c] * 1e-12 for c in FIT_COLUMNS] for r in rows]); E = np.array([r['E'] for r in rows]); w = 1 / E
    if fixed_base_w is None:
        sol, _ = nnls(np.hstack([X, train_t[:, None]]) * w[:, None], E * w); return sol[:-1], float(sol[-1])
    sol, _ = nnls(X * w[:, None], (E - fixed_base_w * train_t) * w); return sol, float(fixed_base_w)


def predict_rows(rates, base, rows, test_t, cap_w):
    X = np.array([[r[c] * 1e-12 for c in FIT_COLUMNS] for r in rows]); return np.minimum(X @ rates + base * test_t, cap_w * test_t)


def stats(pred, E, mask):
    e = np.abs(pred / E - 1)[mask] * 100
    return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), signed=float(np.median((pred / E - 1)[mask] * 100)), n=int(mask.sum()))


def loo_operator(app, cal, mode, measured_base_w, cap_w):
    """Leave one application operator out; calibration windows always train. Returns predictions for application cells (NaN where a cell has no static prediction in static mode)."""
    pred = np.full(len(app), np.nan); ops = np.array([r['op'] for r in app]); below = np.array([r['P'] < BELOW_CAP_W for r in app])
    tp = np.array([r['tp'] for r in app]); t = np.array([r['t'] for r in app]); has = ~np.isnan(tp)
    for o in np.unique(ops):
        tr_idx = [i for i in range(len(app)) if ops[i] != o and below[i] and (has[i] if mode == 'static' else True)]
        cal_ok = [c for c in cal if c['P'] < BELOW_CAP_W]
        rows = [app[i] for i in tr_idx] + cal_ok
        if mode == 'static': train_t = np.array([tp[i] for i in tr_idx] + [c['t'] for c in cal_ok]); fixed = None
        else: train_t = np.array([t[i] for i in tr_idx] + [c['t'] for c in cal_ok]); fixed = measured_base_w
        rates, base = fit(rows, train_t, fixed)
        te = [i for i in range(len(app)) if ops[i] == o and (has[i] if mode == 'static' else True)]
        if te: pred[te] = predict_rows(rates, base, [app[i] for i in te], np.array([tp[i] if mode == 'static' else t[i] for i in te]), cap_w)
    return pred


def evaluate(app, cal, measured_base_w=None, cap_w=600.0):
    E = np.array([r['E'] for r in app]); below = np.array([r['P'] < BELOW_CAP_W for r in app]); src = np.array([r['src'] for r in app]); out = {}
    for mode in ('static', 'measured'):
        p = loo_operator(app, cal, mode, measured_base_w, cap_w); ok = ~np.isnan(p)
        out[mode] = {s: dict(below=stats(np.where(ok, p, 1.0), E, below & (src == s) & ok), all=stats(np.where(ok, p, 1.0), E, (src == s) & ok)) for s in ('target', 'dev')}
    return out


def final_profile(app, cal, cap_w=600.0):
    """Static-runtime profile fitted on every below-cap application cell with a runtime prediction plus the admitted windows (the profile `predict.py` uses with the runtime model)."""
    use = [r for r in app if r['P'] < BELOW_CAP_W and not np.isnan(r['tp'])] + [c for c in cal if c['P'] < BELOW_CAP_W]
    train_t = np.array([r['tp'] if r['src'] != 'cal' else r['t'] for r in use]); rates, base = fit(use, train_t)
    r = dict(zip(FIT_COLUMNS, [float(x) for x in rates])); full = {c: 0.0 for c in En.COLUMNS}
    for k in ('B_l2', 'B_dr', 'ffma', 'shm', 'sfu'): full[k] = r[k]
    full['shfl'] = r['shb']; full['bar'] = r['shb']
    return dict(base_power_w=base, rates_pJ=full, status={c: 'ok' for c in En.COLUMNS}, fit_kind='application_fit (calibration windows + labeled application kernels)', n_training_rows=len(use), cap_w=cap_w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', type=Path, required=True); ap.add_argument('--features', type=Path, nargs='+', required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--calibration', type=Path); g.add_argument('--legacy-windows', type=Path, nargs=2, metavar=('DESIGN_JSON', 'RESULT_JSON'))
    ap.add_argument('--measured-base-w', type=float); ap.add_argument('--cap-w', type=float, default=600.0); ap.add_argument('--write-profile', type=Path); ap.add_argument('--write-document', type=Path, help='copy of --calibration whose energy profile is the application fit (the calibrator-only profile is kept under profile_calibrator_only); usable with predict.py')
    a = ap.parse_args(argv)
    app = load_cells(a.labels, a.features); cal = windows_from_document(a.calibration) if a.calibration else windows_from_legacy(*a.legacy_windows)
    ev = evaluate(app, cal, a.measured_base_w, a.cap_w)
    print('application cells: %d (%d below cap); calibration windows: %d' % (len(app), sum(r['P'] < BELOW_CAP_W for r in app), len(cal)))
    for mode in ('static', 'measured'):
        for s in ('target', 'dev'):
            v = ev[mode][s]; print('%-9s runtime, %-6s cells: below cap %5.1f / %5.1f / %+5.1f (n=%d) | all %5.1f / %5.1f / %+5.1f (n=%d)   [median / 90th pct / signed, leave-one-operator-out]' % (
                mode, s, v['below']['median'], v['below']['p90'], v['below']['signed'], v['below']['n'], v['all']['median'], v['all']['p90'], v['all']['signed'], v['all']['n']))
    if a.write_profile:
        prof = final_profile(app, cal, a.cap_w); a.write_profile.write_text(json.dumps(dict(profile=prof, loo_evaluation=ev), indent=1) + '\n'); print('wrote', a.write_profile)
    if a.write_document:
        if not a.calibration: print('--write-document needs --calibration', file=sys.stderr); return 2
        doc = json.loads(a.calibration.read_text()); e = doc['constants']['energy']; prof = final_profile(app, cal, a.cap_w)
        e['profile_calibrator_only'] = e['profile']; e['profile'] = prof; e['fit_kind'] = prof['fit_kind']; e['loo_evaluation'] = ev
        doc['warnings'] = list(doc.get('warnings', [])) + ['energy profile fitted on calibration windows PLUS labeled application kernels (tools/fit_application_model.py); the calibrator-only profile is kept as profile_calibrator_only']
        a.write_document.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n'); print('wrote', a.write_document)
    return 0


if __name__ == '__main__':
    sys.exit(main())
