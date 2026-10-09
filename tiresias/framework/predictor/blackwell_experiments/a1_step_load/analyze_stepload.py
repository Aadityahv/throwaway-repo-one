"""A1 analysis (CPU only). Input: raw/samples.csv.gz and raw/edges.json from micro_stepload with COUNTER_POLL_MS=1000 (power polled at about 3 kHz; the cumulative energy counter read at the load edges and at 1 Hz).
raw_counter_every_sample/ holds the first run, in which the counter was read in every sampler iteration; that counter is corrupted (see RESULT) and is used only for its update period.
Measured: power update period, response lag to a step (rise and fall), the error of a window that starts and stops at the load edges, with and without padding, against a reference energy, and the
shortest window inside a steady load whose mean power agrees with the long-window mean. Reference net energy of a step = power-sample integral over [t_on - 2 s, t_off + 5 s] minus idle power times that span."""
import gzip, json
from pathlib import Path
import numpy as np
_trap = getattr(np, 'trapezoid', None) or np.trapz
H = Path(__file__).resolve().parent
S = np.loadtxt(gzip.open(H / 'raw/samples.csv.gz'), delimiter=',', skiprows=1); E = json.load(open(H / 'raw/edges.json')); t0 = S[0, 0]
t = (S[:, 0] - t0) * 1e-9; p = S[:, 2] * 1e-3
edges = [{k: (v if k == 'on_s' else v) for k, v in x.items()} for x in E['edges']]
res = dict(device=E['device'], power_limit_w=E['power_limit_mw'] / 1000, samples=len(t), power_poll_rate_hz=len(t) / t[-1])

def integ(a, b):
    m = (t >= a) & (t <= b); return float(_trap(p[m], t[m]))
ch = t[np.flatnonzero(np.diff(p) != 0) + 1]; dc = np.diff(ch)
res['power_value_updates'] = dict(n=int(len(ch)), median_interval_ms=float(np.median(dc) * 1e3), p10_ms=float(np.percentile(dc, 10) * 1e3), p90_ms=float(np.percentile(dc, 90) * 1e3),
                                  share_of_intervals_within_5ms_of_500ms=float(np.mean(np.abs(dc - 0.5) < 0.005)))
steps = []
for g in edges:
    on, off, d = (g['t_on_ns'] - t0) * 1e-9, (g['t_off_ns'] - t0) * 1e-9, g['on_s']
    idle = float(np.median(p[(t > on - 2.5) & (t < on - 0.5)])); r = dict(on_s=d, idle_w=idle)
    ref = integ(on - 2, off + 5) - idle * (off + 5 - (on - 2)); r['reference_net_energy_j'] = ref
    r['naive_power_integral_vs_reference_pct'] = (integ(on, off) - idle * (off - on)) / ref * 100 - 100
    r['padded_1s_vs_reference_pct'] = (integ(on - 1, off + 1) - idle * (d + 2)) / ref * 100 - 100
    r['padded_2s_vs_reference_pct'] = (integ(on - 2, off + 2) - idle * (d + 4)) / ref * 100 - 100
    # counter at the load edges against the power-sample integral over the same interval (validates the counter)
    r['counter_on_to_off_j'] = (g['e_off_mj'] - g['e_on_mj']) * 1e-3; r['power_integral_on_to_off_j'] = integ(on, off)
    r['counter_over_power_integral_on_to_off'] = r['counter_on_to_off_j'] / r['power_integral_on_to_off_j'] if r['power_integral_on_to_off_j'] else None
    r['counter_before_to_after_over_power_integral'] = (g['e_after_mj'] - g['e_before_mj']) * 1e-3 / integ(t[0], t[-1]) if False else None
    if d >= 5:
        plateau = float(np.median(p[(t > on + d * 0.5) & (t < off)])); r['plateau_w'] = plateau
        lo, hi = idle + 0.5 * (plateau - idle), idle + 0.9 * (plateau - idle)
        w = (t > on - 0.2) & (t < on + 6); tt, pp = t[w], p[w]; r['rise_to_50pct_s'] = float(tt[np.argmax(pp >= lo)] - on); r['rise_to_90pct_s'] = float(tt[np.argmax(pp >= hi)] - on)
        wd = (t > off - 0.05) & (t < off + 8); td, pd = t[wd], p[wd]; r['fall_to_50pct_s'] = float(td[np.argmax(pd <= lo)] - off); r['fall_to_10pct_of_step_s'] = float(td[np.argmax(pd <= idle + 0.1 * (plateau - idle))] - off)
    steps.append(r)
res['steps'] = steps
# 1 Hz counter reads: counter slope on each long plateau against the power mean
polled = np.flatnonzero(S[:, 3] > 0); ce = (S[polled, 0] - t0) * 1e-9; ev = S[polled, 3] * 1e-3
res['counter_polled_samples'] = int(len(polled)); plat = []
for g in edges:
    if g['on_s'] >= 40:
        a, b = (g['t_on_ns'] - t0) * 1e-9 + 10, (g['t_off_ns'] - t0) * 1e-9 - 2; m = (ce >= a) & (ce <= b)
        if m.sum() > 5: plat.append(dict(counter_slope_w=float((ev[m][-1] - ev[m][0]) / (ce[m][-1] - ce[m][0])), power_mean_w=float(p[(t >= ce[m][0]) & (t <= ce[m][-1])].mean())))
res['counter_vs_power_on_40s_plateaus'] = plat
# shortest window inside a steady 40 s load that agrees with the long-window mean power (power samples)
long = [g for g in edges if g['on_s'] >= 40]; agree = {}
for w_s in (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20):
    errs = []
    for g in long:
        on, off = (g['t_on_ns'] - t0) * 1e-9, (g['t_off_ns'] - t0) * 1e-9; ref = integ(on + 10, off - 5) / (off - 5 - on - 10)
        for a in np.arange(on + 10, off - 5 - w_s, 0.5): errs.append(abs(integ(a, a + w_s) / w_s / ref - 1) * 100)
    agree[str(w_s)] = dict(p50=float(np.median(errs)), p90=float(np.percentile(errs, 90)), max=float(np.max(errs)))
res['window_inside_steady_load_error_vs_long_mean_power_pct'] = agree
json.dump(res, open(H / 'a1_results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'steps'}, indent=1))
for r in steps: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if v is not None})
