"""E3: error attribution at identical predictions (CPU only). Calibrator-only profile of the archived first run, as-designed admission.

For each target cell (48, with measured energy): energy error with the frozen runtime model's runtime (static) and with the measured runtime, the runtime error itself,
and whether the 600 W cap clipped the prediction. The static-runtime log error splits exactly as
  log(Epred_static/E) = log(Epred_meas/E) [energy-model residual] + log(Epred_static/Epred_meas) [carried through from runtime error].
"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpu_experiments as X  # noqa: E402
P, np_ = X.P, np


def main():
    windows = json.loads((X.RUN / 'stages/energy/windows.json').read_text())
    doc = P.load_calibration(next(X.RUN.glob('calibration_sm_*[0-9a-f].json')), allow_incomplete=True)
    feats, tgt, rts, dev, rtd = X.load_cells(); ids = list(tgt.cell_id); E = tgt.measured_energy_j.values; tm = tgt.measured_short_runtime_s.values
    below = (tgt.measured_window_power_w < X.BELOW_CAP_W).values
    rows_s = P.energy_rows(doc, {'rows': [feats[c] for c in ids]}, rts); rows_m = P.energy_rows(doc, {'rows': [feats[c] for c in ids]}, dict(zip(ids, tm)))
    ps = np.array([rows_s[c]['energy_j'] for c in ids]); pm = np.array([rows_m[c]['energy_j'] for c in ids])
    capped_s = np.array([rows_s[c]['capped'] for c in ids]); capped_m = np.array([rows_m[c]['capped'] for c in ids])
    tstat = np.array([rts[c] for c in ids]); rt_err = tstat / tm - 1
    out = dict(n=len(ids), n_below=int(below.sum()))
    for lab, m in (('below_cap', below), ('all', np.ones(len(ids), bool))):
        d = dict(runtime_abs_err_median=float(np.median(np.abs(rt_err[m])) * 100), runtime_signed_median=float(np.median(rt_err[m]) * 100),
                 energy_err_static_median=float(np.median(np.abs(ps / E - 1)[m]) * 100), energy_err_measured_runtime_median=float(np.median(np.abs(pm / E - 1)[m]) * 100),
                 carried_from_runtime_median_abs=float(np.median(np.abs(ps / pm - 1)[m]) * 100),
                 capped_static=int(capped_s[m].sum()), capped_measured=int(capped_m[m].sum()))
        # cells where static is worse than measured by > 5 points: what drives it
        w = (np.abs(ps / E - 1) - np.abs(pm / E - 1))[m] * 100; d['static_worse_than_measured_by_5pts'] = int((w > 5).sum()); d['static_better_than_measured_by_5pts'] = int((w < -5).sum())
        out[lab] = d
    # per family
    fam = {}
    for f in sorted(set(tgt.family)):
        m = (tgt.family == f).values & below
        if m.sum() == 0: continue
        fam[f] = dict(n=int(m.sum()), runtime_abs_err=float(np.median(np.abs(rt_err[m])) * 100), energy_static=float(np.median(np.abs(ps / E - 1)[m]) * 100), energy_measured=float(np.median(np.abs(pm / E - 1)[m]) * 100))
    out['per_family_below_cap'] = fam
    # cap clipping: cells at or above 570 W measured
    ab = ~below; out['above_cap_cells'] = dict(n=int(ab.sum()), static_median_abs=float(np.median(np.abs(ps / E - 1)[ab]) * 100), measured_median_abs=float(np.median(np.abs(pm / E - 1)[ab]) * 100),
                                               static_signed=float(np.median((ps / E - 1)[ab]) * 100), predicted_capped=int(capped_s[ab].sum()))
    (HERE / 'e3_error_attribution.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
