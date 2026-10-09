#!/usr/bin/env python3
"""Power-sensor update test: analysis (CPU only). Reproduces the method of the Blackwell step-load analysis
(tiresias/framework/predictor/blackwell_experiments/a1_step_load/analyze_stepload.py, not edited) as an importable, path-parameterised function with
robust handling of a level that is never reached, several runs, and an explicit minimum-window decision.

Input per run directory: samples.csv.gz (or samples.csv) and edges.json written by micro_stepload, with the board power polled at about 3 kHz and the cumulative energy counter read at the
load edges and at 1 Hz only (COUNTER_POLL_MS=1000). Optional: a counter-update-period diagnostic directory (the same program with the counter read in every sampler iteration; its counter VALUES
are not usable, only the interval between value changes).

Measured: how often the board-power reading changes value; the delay of that reading after a load step (rise to 50% and 90% of the step, fall to 50% and to within 10%); for a window that starts
and stops at the load edges, with 0, 1 and 2 s of padding on each side, the error of the integrated sampled power against a reference net energy (the sampled-power integral over
[t_on - 2 s, t_off + 5 s] minus idle power times that span); the same window taken from the cumulative energy counter; the error of a short window inside a steady load against the long mean;
and the counter slope against the sampled mean on long loads. The decision block gives, per padding, the shortest load length from which every window of that length or longer is within the
tolerance (default 3%), over all runs (the largest length wins).

    python analyze_power_sensor.py --raw <run dir> [<run dir> ...] [--diag <diag dir>] --out power_sensor_results.json [--tolerance-pct 3]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_trap = getattr(np, 'trapezoid', None) or np.trapz
PADDINGS_S = (0, 1, 2)
STEADY_WINDOWS_S = (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20)


def load_samples(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        header = f.readline().strip().split(',')
        if header[:4] != ['monotonic_ns', 'power_call_end_ns', 'board_power_mw', 'energy_counter_mj']:
            raise ValueError('%s: unexpected header %r' % (path, header))
        # integer columns are kept as float64: nanosecond values are below 2**53 for any uptime of interest, and differences are taken after subtracting the first sample
        return np.loadtxt(f, delimiter=',', dtype=np.float64, ndmin=2)


def _find_samples(d):
    d = Path(d)
    for name in ('samples.csv.gz', 'samples.csv'):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError('no samples.csv.gz or samples.csv in %s' % d)


def _stats_ms(dc):
    return dict(n=int(len(dc)), median_interval_ms=float(np.median(dc) * 1e3), p10_ms=float(np.percentile(dc, 10) * 1e3), p90_ms=float(np.percentile(dc, 90) * 1e3))


def _crossing(tt, pp, level, rising):
    idx = np.flatnonzero(pp >= level) if rising else np.flatnonzero(pp <= level)
    return float(tt[idx[0]]) if idx.size else None


def _delta(a, b):
    return None if (a is None or b is None) else float(a - b)


def analyze_run(S, E, tolerance_pct=3.0):
    """One run: samples array (columns: monotonic_ns, power_call_end_ns, power_mW, counter_mJ or 0) and the parsed edges.json."""
    t0 = S[0, 0]
    t = (S[:, 0] - t0) * 1e-9
    p = S[:, 2] * 1e-3
    edges = E['edges']
    res = dict(device=E.get('device'), power_limit_w=E['power_limit_mw'] / 1000 if E.get('power_limit_mw') else None, samples=int(len(t)),
               power_poll_rate_hz=float(len(t) / t[-1]) if t[-1] > 0 else None, nvml_ok=E.get('nvml_ok'), kernel_trips=E.get('kernel_trips'), blocks=E.get('blocks'))

    def integ(a, b):
        m = (t >= a) & (t <= b)
        return float(_trap(p[m], t[m])) if m.sum() > 1 else 0.0

    changes = t[np.flatnonzero(np.diff(p) != 0) + 1]
    if len(changes) >= 3:
        dc = np.diff(changes)
        u = _stats_ms(dc)
        u['changes'] = int(len(changes))
        u['share_of_intervals_within_5ms_of_median'] = float(np.mean(np.abs(dc - np.median(dc)) < 0.005))
        res['power_value_updates'] = u
    else:
        res['power_value_updates'] = dict(n=int(len(changes)), note='fewer than 3 value changes: update period not measurable')

    steps = []
    for g in edges:
        on, off, d = (g['t_on_ns'] - t0) * 1e-9, (g['t_off_ns'] - t0) * 1e-9, float(g['on_s'])
        pre = p[(t > on - 2.5) & (t < on - 0.5)]
        if pre.size == 0:
            pre = p[t < on]
        idle = float(np.median(pre))
        r = dict(on_s=d, idle_w=idle, actual_load_s=float(off - on))
        span = (off + 5) - (on - 2)
        ref = integ(on - 2, off + 5) - idle * span
        r['reference_net_energy_j'] = ref
        for pad in PADDINGS_S:
            r['padded_%ds_vs_reference_pct' % pad] = ((integ(on - pad, off + pad) - idle * (off - on + 2 * pad)) / ref * 100 - 100) if ref > 0 else None
        # counter at the load edges (energy of the load interval, net of idle) against the same reference
        c_on_off = (g['e_off_mj'] - g['e_on_mj']) * 1e-3
        r['counter_on_to_off_j'] = c_on_off
        r['power_integral_on_to_off_j'] = integ(on, off)
        r['counter_over_power_integral_on_to_off'] = c_on_off / r['power_integral_on_to_off_j'] if r['power_integral_on_to_off_j'] else None
        r['counter_edge_window_vs_reference_pct'] = ((c_on_off - idle * (off - on)) / ref * 100 - 100) if ref > 0 and g['e_off_mj'] and g['e_on_mj'] else None
        r['counter_edge_mean_power_w'] = c_on_off / (off - on) if off > on and g['e_off_mj'] else None
        if d >= 5:
            plateau = float(np.median(p[(t > on + d * 0.5) & (t < off)]))
            r['plateau_w'] = plateau
            lo, hi, lo10 = idle + 0.5 * (plateau - idle), idle + 0.9 * (plateau - idle), idle + 0.1 * (plateau - idle)
            w = (t > on - 0.2) & (t < off); tt, pp = t[w], p[w]
            r['rise_to_50pct_s'] = _delta(_crossing(tt, pp, lo, True), on); r['rise_to_90pct_s'] = _delta(_crossing(tt, pp, hi, True), on)
            wd = (t > off - 0.05) & (t < off + 5.5); td, pd = t[wd], p[wd]
            r['fall_to_50pct_s'] = _delta(_crossing(td, pd, lo, False), off); r['fall_to_10pct_of_step_s'] = _delta(_crossing(td, pd, lo10, False), off)
        steps.append(r)
    res['steps'] = steps
    fr = [r['plateau_w'] / res['power_limit_w'] for r in steps if 'plateau_w' in r and res['power_limit_w']]
    res['max_plateau_fraction_of_power_limit'] = float(max(fr)) if fr else None
    res['plateau_near_power_limit_warning'] = bool(fr and max(fr) > 0.9)   # a load at the cap would let power-limit throttling confound the step response

    lag = {}
    for k in ('rise_to_50pct_s', 'rise_to_90pct_s', 'fall_to_50pct_s', 'fall_to_10pct_of_step_s'):
        v = [r[k] for r in steps if r.get(k) is not None]
        n_missing = sum(1 for r in steps if 'plateau_w' in r and r.get(k) is None)
        lag[k] = dict(n=len(v), n_level_not_reached=n_missing, min=float(min(v)) if v else None, median=float(np.median(v)) if v else None, max=float(max(v)) if v else None)
    res['step_lag_s'] = lag

    # counter slope on the long plateaus, from the 1 Hz reads, against the sampled mean (the counter is never read faster; see the README)
    polled = np.flatnonzero(S[:, 3] > 0)
    ce, ev = (S[polled, 0] - t0) * 1e-9, S[polled, 3] * 1e-3
    res['counter_polled_samples'] = int(len(polled))
    plat = []
    for g in edges:
        if g['on_s'] >= 20:
            a, b = (g['t_on_ns'] - t0) * 1e-9 + 10, (g['t_off_ns'] - t0) * 1e-9 - 2
            m = (ce >= a) & (ce <= b)
            if m.sum() > 5 and ce[m][-1] > ce[m][0]:
                pm = p[(t >= ce[m][0]) & (t <= ce[m][-1])].mean()
                sl = float((ev[m][-1] - ev[m][0]) / (ce[m][-1] - ce[m][0]))
                plat.append(dict(on_s=g['on_s'], counter_slope_w=sl, power_mean_w=float(pm), counter_over_power_mean_pct=float(sl / pm * 100 - 100)))
    res['counter_vs_power_on_long_plateaus'] = plat

    # window inside a steady load against the long-window mean of sampled power
    long_steps = [g for g in edges if g['on_s'] >= 40] or [g for g in edges if g['on_s'] >= 20]
    agree = {}
    for w_s in STEADY_WINDOWS_S:
        errs = []
        for g in long_steps:
            on, off = (g['t_on_ns'] - t0) * 1e-9, (g['t_off_ns'] - t0) * 1e-9
            lo, hi = on + 10, off - 5
            if hi - lo <= w_s:
                continue
            ref = integ(lo, hi) / (hi - lo)
            for a in np.arange(lo, hi - w_s, 0.5):
                errs.append(abs(integ(a, a + w_s) / w_s / ref - 1) * 100)
        if errs:
            agree[str(w_s)] = dict(p50=float(np.median(errs)), p90=float(np.percentile(errs, 90)), max=float(np.max(errs)), n=len(errs))
    res['window_inside_steady_load_error_vs_long_mean_power_pct'] = agree
    res['steady_load_lengths_used_s'] = [g['on_s'] for g in long_steps]
    return res


def min_load_within(items, tol):
    """items: (load_s, error_pct or None). Shortest load length L such that every item with load >= L has |error| <= tol; None if even the longest load fails or no data."""
    items = [(l, e) for l, e in items if e is not None]
    if not items:
        return None
    for L in sorted({l for l, _ in items}):
        if all(abs(e) <= tol for l, e in items if l >= L):
            return L
    return None


def decision(runs, tol):
    """Minimum load length, and the total measured window it implies, per padding, over every run (largest wins; None when any run never gets there)."""
    out = {}
    keys = [('padding_%ds' % pad, 'padded_%ds_vs_reference_pct' % pad, pad) for pad in PADDINGS_S] + [('energy_counter_edge_window', 'counter_edge_window_vs_reference_pct', 0)]
    for name, key, pad in keys:
        per_run = [min_load_within([(r['on_s'], r.get(key)) for r in run['steps']], tol) for run in runs]
        known = [x for x in per_run if x is not None]
        L = None if (not per_run or len(known) != len(per_run)) else max(known)
        out[name] = dict(min_load_s_per_run=per_run, min_load_s=L, total_window_s=None if L is None else float(L + 2 * pad),
                         meaning='shortest load length from which every window of that length or longer is within %.1f%% of the reference net energy, in every run' % tol)
    return out


def counter_update_period(diag_dir):
    S = load_samples(_find_samples(diag_dir)); t = (S[:, 0] - S[0, 0]) * 1e-9
    m = S[:, 3] > 0
    tc, ec = t[m], S[m, 3]
    idx = np.flatnonzero(np.diff(ec) != 0) + 1
    if len(idx) < 5:
        return dict(measured=False, note='fewer than 5 counter value changes in the diagnostic run')
    d = np.diff(tc[idx])
    out = _stats_ms(d); out['measured'] = True; out['counter_reads'] = int(m.sum())
    out['note'] = 'interval between counter value changes seen by a counter read in every sampler iteration; the counter values of that run are NOT used (high-rate reads corrupt the counter, see the README)'
    return out


def analyze(raw_dirs, diag_dir=None, tolerance_pct=3.0):
    runs, names = [], []
    for d in raw_dirs:
        d = Path(d)
        E = json.load(open(d / 'edges.json'))
        runs.append(analyze_run(load_samples(_find_samples(d)), E, tolerance_pct)); names.append(str(d))
    res = dict(schema='power_sensor_results_v1', tolerance_pct=tolerance_pct, run_dirs=names, runs=runs, decision=decision(runs, tolerance_pct))
    res['power_reading_update_period_ms'] = [r['power_value_updates'].get('median_interval_ms') for r in runs]
    res['energy_counter_update_period'] = counter_update_period(diag_dir) if diag_dir and Path(diag_dir).exists() else dict(
        measured=False, note='no counter-update-period diagnostic was run (it reads the counter in every sampler iteration and is opt-in)')
    return res


def _print_summary(res):
    for name, run in zip(res['run_dirs'], res['runs']):
        print('run', name, 'device', run['device'], 'poll %.0f Hz' % (run['power_poll_rate_hz'] or 0), 'power updates:', run['power_value_updates'])
        print('  lag:', json.dumps(run['step_lag_s']))
        for r in run['steps']:
            print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if v is not None and k in (
                'on_s', 'padded_0s_vs_reference_pct', 'padded_1s_vs_reference_pct', 'padded_2s_vs_reference_pct', 'counter_edge_window_vs_reference_pct', 'plateau_w')})
    print('decision:', json.dumps(res['decision'], indent=1))
    print('energy counter update period:', res['energy_counter_update_period'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', nargs='+', required=True, help='run directories holding samples.csv.gz and edges.json')
    ap.add_argument('--diag', help='optional counter-update-period diagnostic directory'); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--tolerance-pct', type=float, default=3.0)
    a = ap.parse_args(argv)
    if a.out.exists():
        print('refusing to overwrite %s' % a.out, file=sys.stderr); return 2
    res = analyze(a.raw, a.diag, a.tolerance_pct)
    a.out.write_text(json.dumps(res, indent=1) + '\n'); _print_summary(res)
    return 0


if __name__ == '__main__':
    sys.exit(main())
