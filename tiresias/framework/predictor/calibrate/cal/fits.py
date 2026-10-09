"""Fits of the calibration constants from raw microbenchmark rows. Pure functions (rows in, constants out); ports of constants/fit_*.py with every Blackwell literal (188 SMs, 96 KB L1 capacity) replaced by device facts or a rule derived from the data. No operator data is read."""
import collections
import gzip
import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------- launch gap and L1 re-read (micro_launch)

def fit_launch_reuse(chain, reuse, sm):
    inc = {}
    for g in (1, sm, 2048):
        t = {x['kernels_per_launch']: x['us_per_launch'] for x in chain if x['grid'] == g}
        inc[g] = float(np.mean([t[k + 1] - t[k] for k in (1, 2, 3)]))
    b = (inc[2048] - inc[1]) / 2047; a = inc[1]
    pts = [x['footprint_kb_per_block'] * 1024 / (x['extra_pass_us'] * 1e-6) / 1e9 for x in reuse
           if x['blocks_per_sm'] == 1 and 48 <= x['footprint_kb_per_block'] <= 96 and x['extra_pass_us'] > 0.1]
    l1_bw = float(np.median(pts))
    s4 = sorted((x['blocks_per_sm'] * x['footprint_kb_per_block'], x['extra_pass_us'] / x['us_1pass']) for x in reuse if x['blocks_per_sm'] == 4 and x['blocks_per_sm'] * x['footprint_kb_per_block'] <= 512)
    r_hit = float(np.median([r for kb, r in s4 if kb == 64]))
    curve = [(kb, float(np.clip((r - r_hit) / (1 - r_hit), 0, 1))) for kb, r in s4 if kb >= 64]
    curve = [(64, 0.0)] + [(kb, v) for kb, v in curve if kb > 64] + [(1024, 1.0)]
    mono = []
    for kb, v in curve: mono.append((kb, max(v, mono[-1][1]) if mono else v))
    # L1 capacity: largest per-SM footprint (one block per SM) whose re-read pass ran at L1 speed: ratio at most a quarter of the way from the L1-hit ratio to an all-L2 pass (1.0).
    # This formalises the earlier judgement (96 KB on Blackwell: 128 KB per SM already behaved as L2); the rule, not the number, is what the tool carries to other GPUs.
    s1 = sorted((x['footprint_kb_per_block'], x['extra_pass_us'] / x['us_1pass']) for x in reuse if x['blocks_per_sm'] == 1)
    thr = r_hit + 0.25 * (1.0 - r_hit)
    cap_kb = max([kb for kb, r in s1 if r <= thr] or [0])
    return dict(l1_reread_l2_fraction_curve=dict(per_sm_kb=[k for k, _ in mono], p_l2=[round(v, 4) for _, v in mono], hit_cost_ratio=r_hit),
                launch_us_per_kernel=dict(intercept_at_one_block=a, per_extra_block=b, check_at_sm_blocks_pred=a + b * (sm - 1), check_at_sm_blocks_measured=inc[sm]),
                l1=dict(capacity_bytes_per_sm=cap_kb * 1024, capacity_rule='largest one-block-per-SM footprint whose re-read pass ratio is at most hit_ratio + 0.25 * (1 - hit_ratio)', bandwidth_bytes_per_s_per_sm=l1_bw * 1e9, points_gbps=[round(p, 1) for p in pts]),
                composition_rule='issue cycles = max(0.25 * total warp instructions, busiest pipe) per phase')

# ----------------------------------------------------------------------------------------------- memory-level parallelism (micro_mlp)

def fit_mlp(rows, sm):
    out = {'service_latency_us': {}, 'linear_region_points': {}}
    for tier in ('l2', 'dram'):
        lam = []
        for x in rows:
            if x['tier'] != tier: continue
            inflight = x['achieved_blocks_per_sm'] * 4 * 32 * x['W'] * x['M']
            if inflight > 2048: continue
            read_rate_per_sm = x['bytes_each_way'] / (x['us'] * 1e-6) / sm
            lam.append(inflight / read_rate_per_sm * 1e6)
        if not lam: raise ValueError('no linear-region points for tier %s' % tier)
        out['service_latency_us'][tier.upper()] = float(np.median(lam)); out['linear_region_points'][tier.upper()] = [round(v, 3) for v in lam]
    out['rule'] = ('per phase with global loads: read bytes / (active SMs x in-flight bytes per SM / service latency) is a lower bound on the stream stage; in-flight per SM = resident warps x (read bytes per load request) x min(loads per warp / dependent load depth, 1 or more)')
    return out

# ----------------------------------------------------------------------------------------------- stream bandwidth by read fraction (micro_stream)
MIXES = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2)]

def fit_stream(r):
    l2, over, dram = {}, [], {}
    for mix in MIXES:
        pts = []
        for tier in ('l2a', 'l2b'):
            b = min((x for x in r if x['tier'] == tier and (x['R'], x['W']) == mix), key=lambda x: x['us']); pts.append((b['bytes_read'] + b['bytes_written'], b['us']))
        (b1, t1), (b2, t2) = pts; bw = (b2 - b1) / ((t2 - t1) * 1e-6) / 1e12; over.append(t1 - b1 / (bw * 1e12) * 1e6)
        l2[mix] = bw
        dram[mix] = max((x['bytes_read'] + x['bytes_written']) / (x['us'] * 1e-6) / 1e12 for x in r if x['tier'] == 'dram' and (x['R'], x['W']) == mix)
    def curve(d):
        by = {}
        for (R, W), v in d.items(): by.setdefault(round(R / (R + W), 4), []).append(v)
        xs = sorted(by); return xs, [float(np.mean(by[x])) for x in xs]
    lx, ly = curve(l2); dx, dy = curve(dram)
    return dict(l2_total_bandwidth_TBps=dict(read_fraction=lx, value=ly), dram_total_bandwidth_TBps=dict(read_fraction=dx, value=dy),
                kernel_fixed_overhead_us=float(np.median(over)), overhead_points_us=[round(o, 3) for o in over])

# ----------------------------------------------------------------------------------------------- store/triad fixtures: legacy stream constants (micro_store)

def store_pattern(stride, threads=256):
    """Mean sectors and 128-B lines per 32-lane warp request for index=(tid*stride)&mask, 4-byte stores."""
    s = l = 0; warps = threads // 32
    for w in range(warps):
        addrs = [((32 * w + i) * stride) * 4 for i in range(32)]
        s += len({a // 32 for a in addrs}); l += len({a // 128 for a in addrs})
    return s / warps, l / warps

def nnls3(A, y):
    """Nonnegative least squares for at most a handful of unknowns by enumerating the active sets (exact; no scipy). Returns (coefficients, condition number of A)."""
    import itertools
    n = A.shape[1]; best = None
    for r in range(1, n + 1):
        for cols in itertools.combinations(range(n), r):
            sub = A[:, cols]; c, *_ = np.linalg.lstsq(sub, y, rcond=None)
            if (c < 0).any(): continue
            full = np.zeros(n); full[list(cols)] = c; res = float(np.sum((A @ full - y) ** 2))
            if best is None or res < best[0]: best = (res, full)
    return best[1], float(np.linalg.cond(A))


STORE_METHODS = ('legacy', 'stream_rates', 'guarded')
STORE_DERIVATION_VERSION = 'store_derivation/1'
MIN_READ_INCREMENT_FRACTION = 0.05  # identification rule, the booking log 2026-10-03 12:00 (fixed before any comparison)


def _store_rows(rows):
    pick = lambda **kw: next(x for x in rows if all(x.get(k) == v for k, v in kw.items()))
    return dict(tiny=pick(fixture='tiny'), tri=pick(fixture='triad', tier='below'), s1=pick(fixture='store', tier='below', stride=1), s7=pick(fixture='store', tier='below', stride=7),
                s33=pick(fixture='store', tier='below', stride=33), sa=pick(fixture='store', tier='above', stride=1), ta=pick(fixture='triad', tier='above'))


def _store_design(w):
    """Design rows of the four below-L2 windows: (read bytes, write-sector bytes, lines, measured us); the triad first, then strides 1, 7, 33."""
    tri = w['tri']; rows = [(tri['bytes_read'], tri['bytes_written'], (tri['bytes_read'] + tri['bytes_written']) / 128, tri['us'])]
    for st, key in ((1, 's1'), (7, 's7'), (33, 's33')):
        sec, lin = store_pattern(st); req = w[key]['bytes_written'] / 128
        rows.append((0.0, req * sec * 32, req * lin, w[key]['us']))
    return rows


def store_identification(rows, coef=None):
    """Identification check of the read rate in the existing (legacy) store fit: the triad's time over the stride-1 store (its extra reads) must be positive and at least
    MIN_READ_INCREMENT_FRACTION of the triad's time above the launch floor, and no fitted coefficient may sit at the nonnegativity bound. `coef` = the legacy NNLS
    coefficients (read-sector, write-sector, line); None skips the bound test."""
    w = _store_rows(rows); inc = w['tri']['us'] - w['s1']['us']; above = w['tri']['us'] - w['tiny']['us']
    frac = inc / above if above > 0 else float('nan')
    at_bound = None if coef is None else [bool(c <= 0) for c in coef]
    reasons = []
    if not inc > 0: reasons.append('the triad reads add no time over the store (increment %.4f us)' % inc)
    elif not frac >= MIN_READ_INCREMENT_FRACTION: reasons.append('the triad reads add only %.1f%% of its time above the launch floor (< %.0f%%)' % (100 * frac, 100 * MIN_READ_INCREMENT_FRACTION))
    if at_bound is not None and any(at_bound): reasons.append('a coefficient is at the nonnegativity bound (read-sector, write-sector, line): %s' % at_bound)
    return dict(read_increment_us=float(inc), triad_above_floor_us=float(above), read_increment_fraction=float(frac), coefficient_at_bound=at_bound, passes=not reasons, reasons=reasons)


def _curve_endpoint(curve, fraction):
    xs = list(curve['read_fraction'])
    if not xs or abs(xs[0] - 0.0) > 1e-9 or abs(xs[-1] - 1.0) > 1e-9: raise ValueError('stream curve lacks the read-fraction 0 and 1 endpoints: %s' % xs)
    return float(curve['value'][0 if fraction == 0 else -1])


def stream_endpoint_rates(stream_curves):
    """Rates in TB/s (bytes of one kind per second / 1e12, the unit make_constants consumers divide 32-byte-sector bytes by) taken from the stream stage: the write-only
    (read fraction 0) and read-only (fraction 1) endpoints of the L2 and DRAM total-bandwidth curves. At a pure-read (pure-write) mix total bandwidth is read (write) bandwidth."""
    if not stream_curves: raise ValueError("method needs the stream stage's curves (stream_curves is None)")
    l2, dr = stream_curves['l2_total_bandwidth_TBps'], stream_curves['dram_total_bandwidth_TBps']
    return dict(L2_read_sector_TBps=_curve_endpoint(l2, 1), L2_write_sector_TBps=_curve_endpoint(l2, 0), DRAM_read_TBps=_curve_endpoint(dr, 1), DRAM_write_TBps=_curve_endpoint(dr, 0))


def _fit_store_legacy(rows, sm, latency_ns):
    w = _store_rows(rows); t0 = float(w['tiny']['us'])
    A, y = [], []
    for rb, wb, lines, us in _store_design(w):
        A.append([rb, wb, lines]); y.append(us - t0)
    coef, cond = nnls3(np.array(A), np.array(y))
    inv_r, inv_w, c_line = coef
    warnings = []
    if cond > 1e8: warnings.append('store fit is ill-conditioned (condition number %.3g): coefficients are a nonnegative best fit, not independently identified' % cond)
    if min(coef) <= 0: warnings.append('a stream coefficient was driven to zero by the nonnegativity constraint: %s' % [float(c) for c in coef])
    sa, ta = w['sa'], w['ta']
    inv_dw = (sa['us'] - t0) / sa['bytes_written']
    inv_dr = (ta['us'] - t0 - ta['bytes_written'] * inv_dw) / ta['bytes_read']
    c_line = float(c_line)   # inv_r, inv_w stay numpy scalars: a coefficient at the bound gives an infinite rate exactly as in the archived documents (1/0 must not raise)
    out = dict(t0_us=t0, L2_read_sector_TBps=1 / inv_r / 1e6, L2_write_sector_TBps=1 / inv_w / 1e6, c_line_ns=c_line * 1e3,
               DRAM_read_TBps=1 / inv_dr / 1e6, DRAM_write_TBps=1 / inv_dw / 1e6, latency_ns=latency_ns, active_sms_microbench=sm)
    return out, warnings, coef


def _fit_store_stream_rates(rows, sm, latency_ns, stream_curves):
    """Candidate A: the four traffic rates are the stream stage's measured endpoints; the launch floor is the directly measured empty-launch time (the tiny fixture, exactly as
    the legacy derivation takes it; recorded decision 2026-10-04, not refitted); only the per-line cost is fitted (nonnegative least squares, one unknown) on the four below-L2
    windows with the rates and the launch floor fixed: us = t0 + read bytes / R_read + write-sector bytes / R_write + lines * c_line."""
    w = _store_rows(rows); rate = stream_endpoint_rates(stream_curves); t0 = float(w['tiny']['us'])
    lines, y, base_us = [], [], []
    design = _store_design(w)
    for rb, wb, ln, us in design:
        base = rb / (rate['L2_read_sector_TBps'] * 1e6) + wb / (rate['L2_write_sector_TBps'] * 1e6)
        lines.append(ln); y.append(us - t0 - base); base_us.append(base)
    A = np.array(lines).reshape(-1, 1)
    coef, cond = nnls3(A, np.array(y)); c_line = float(coef[0])
    pred = [t0 + b + c_line * a for a, b in zip(lines, base_us)]
    warnings = []
    if c_line <= 0: warnings.append('stream_rates refit: the line cost was driven to zero by the nonnegativity constraint (launch floor and rates fixed): %s' % [float(c_line)])
    out = dict(t0_us=t0, L2_read_sector_TBps=rate['L2_read_sector_TBps'], L2_write_sector_TBps=rate['L2_write_sector_TBps'], c_line_ns=c_line * 1e3,
               DRAM_read_TBps=rate['DRAM_read_TBps'], DRAM_write_TBps=rate['DRAM_write_TBps'], latency_ns=latency_ns, active_sms_microbench=sm)
    fitinfo = dict(window_measured_us=[float(r[3]) for r in design], window_predicted_us=[float(p) for p in pred], coefficients_at_bound=[bool(c_line <= 0)],
                   tiny_launch_floor_us=t0, launch_floor_source='measured empty-launch time (tiny fixture), not refitted')
    return out, warnings, fitinfo


DRAM_AGREEMENT_TOL = 0.10        # above-L2 algebra rate must agree with the stream stage's direct measurement of the same kind within this fraction
MIN_FOOTPRINT_OVER_L2 = 4.0      # above-L2 windows must move at least this multiple of the board's L2 size


def dram_plausibility(rows, legacy, stream_curves=None, l2_bytes=None, dram_peak_TBps=None):
    """General plausibility check of the DRAM rates that the above-L2 store/triad algebra produced (`legacy` = its constants). Checks: (1) both above-L2 windows move at least
    MIN_FOOTPRINT_OVER_L2 x the board's L2 size (else L2 hits contaminate them); (2) each DRAM rate is within DRAM_AGREEMENT_TOL of the stream stage's direct DRAM measurement of the
    same kind (read: read-only endpoint, write: write-only endpoint); (3) neither exceeds the board's verified DRAM peak, when one is supplied (from HARDWARE_GROUND_TRUTH.md).
    A check whose input is missing is recorded as not evaluated, never as passed; `passes` is False only when an evaluated check fails."""
    w = _store_rows(rows); checks = []
    footprint = min(w['sa']['bytes_written'], w['ta']['bytes_read'] + w['ta']['bytes_written'])
    if l2_bytes: checks.append(dict(check='footprint_over_L2', value=footprint / l2_bytes, limit=MIN_FOOTPRINT_OVER_L2, ok=bool(footprint / l2_bytes >= MIN_FOOTPRINT_OVER_L2)))
    else: checks.append(dict(check='footprint_over_L2', not_evaluated='no L2 size supplied'))
    if stream_curves:
        sr = stream_endpoint_rates(stream_curves)
        for kind in ('read', 'write'):
            k = 'DRAM_%s_TBps' % kind; rel = legacy[k] / sr[k] - 1
            checks.append(dict(check='agrees_with_stream_%s_endpoint' % kind, algebra_TBps=float(legacy[k]), stream_TBps=sr[k], relative_difference=float(rel), limit=DRAM_AGREEMENT_TOL, ok=bool(abs(rel) <= DRAM_AGREEMENT_TOL)))
    else: checks.append(dict(check='agrees_with_stream_endpoints', not_evaluated='no stream curves supplied'))
    if dram_peak_TBps:
        for kind in ('read', 'write'):
            k = 'DRAM_%s_TBps' % kind; checks.append(dict(check='below_verified_peak_%s' % kind, rate_TBps=float(legacy[k]), peak_TBps=float(dram_peak_TBps), ok=bool(legacy[k] <= dram_peak_TBps)))
    else: checks.append(dict(check='below_verified_peak', not_evaluated='no verified DRAM peak supplied'))
    bad = [c for c in checks if c.get('ok') is False]
    return dict(passes=not bad, checks=checks, reasons=['%s failed: %s' % (c['check'], {k: v for k, v in c.items() if k != 'check'}) for c in bad])


def fit_store_detailed(rows, sm, latency_ns, stream_curves=None, method='legacy', l2_bytes=None, dram_peak_TBps=None):
    """Traffic-rate constants of the runtime model (`store_legacy`) by one of three derivations; returns (constants, warnings, derivation).
    legacy: additive nonnegative least squares of the four below-L2 windows (read-sector, write-sector, line coefficients), t0 = tiny-kernel time, DRAM rates by above-L2
    algebra; byte-identical to every archived document. A legacy fit that fails the L2 identification check or the DRAM plausibility check carries a loud warning.
    stream_rates (candidate A): L2/DRAM read and write rates = stream endpoints, launch floor = measured empty-launch time, only the line cost refitted with the rates and floor fixed.
    guarded (candidate B), decided per block by general rules: L2 block (L2 rates, line cost, launch floor) = legacy if the identification check passes, else the stream_rates
    refit; DRAM block = the above-L2 algebra if the plausibility check passes, else the stream endpoints. The derivation dict records the method requested and used and all diagnostics."""
    if method not in STORE_METHODS: raise ValueError('unknown store derivation method %r (choose from %s)' % (method, STORE_METHODS))
    leg, wleg, coef = _fit_store_legacy(rows, sm, latency_ns)
    ident = store_identification(rows, coef)
    dplaus = dram_plausibility(rows, leg, stream_curves, l2_bytes, dram_peak_TBps)
    der = dict(version=STORE_DERIVATION_VERSION, method_requested=method, identification=ident, dram_plausibility=dplaus, legacy_coefficients_us_per_byte_or_line=[float(c) for c in coef])
    if method == 'legacy':
        warnings = list(wleg)
        if not ident['passes']:
            warnings.append('UNIDENTIFIED TRAFFIC RATES (legacy store fit): ' + '; '.join(ident['reasons']) + '. The fitted L2 read, L2 write and line coefficients are not independent measurements; prefer method stream_rates or guarded')
        if not dplaus['passes']:
            warnings.append('IMPLAUSIBLE DRAM RATES (legacy above-L2 algebra): ' + '; '.join(dplaus['reasons']) + '. Prefer method stream_rates or guarded')
        der['method_used'] = 'legacy'; return leg, warnings, der
    stream_dram = {k: stream_endpoint_rates(stream_curves)[k] for k in ('DRAM_read_TBps', 'DRAM_write_TBps')}
    use_legacy_l2 = method == 'guarded' and ident['passes']; use_legacy_dram = method == 'guarded' and dplaus['passes']
    if use_legacy_l2: out, warnings = dict(leg), list(wleg)
    else: out, warnings, fitinfo = _fit_store_stream_rates(rows, sm, latency_ns, stream_curves); der['stream_rates_fit'] = fitinfo
    if use_legacy_dram: out.update({k: leg[k] for k in ('DRAM_read_TBps', 'DRAM_write_TBps')})
    else: out.update(stream_dram)
    der.update(stream_endpoint_rates=stream_endpoint_rates(stream_curves), legacy_constants_for_comparison={k: leg[k] for k in leg if k != 'latency_ns'},
               l2_block='legacy' if use_legacy_l2 else 'stream_rates', dram_block='legacy' if use_legacy_dram else 'stream_rates')
    der['method_used'] = 'legacy' if (use_legacy_l2 and use_legacy_dram) else 'stream_rates' if not (use_legacy_l2 or use_legacy_dram) else 'mixed(l2=%s,dram=%s)' % (der['l2_block'], der['dram_block'])
    if method == 'guarded': der['guard_reason'] = ident['reasons'] + dplaus['reasons']
    return out, warnings, der


def fit_store(rows, sm, latency_ns, stream_curves, method='legacy', l2_bytes=None, dram_peak_TBps=None):
    """Constants and warnings of fit_store_detailed (default `legacy`, identical to the original derivation)."""
    out, warnings, _ = fit_store_detailed(rows, sm, latency_ns, stream_curves, method, l2_bytes, dram_peak_TBps)
    return out, warnings

# ----------------------------------------------------------------------------------------------- shared-memory cost (micro_smem)

def fit_smem(vol, plain):
    """Cost of a warp-level shared request in cycles per SM versus lane word-stride. Rule: max(floor, degree) with degree = min(32, stride) for power-of-two strides >= 4."""
    v = {(x['stride'], x['store']): x['cycles_per_warp_instruction_per_sm'] for x in vol}
    free = [c for (s, st), c in v.items() if s in (0, 1, 3, 5, 33)]
    floor = float(np.median(free))
    conflicted = [(s, c) for (s, st), c in v.items() if s in (4, 8, 16, 32)]
    slope = float(np.median([c / s for s, c in conflicted]))
    p = {(x['width'], x['stride_words'], x['ffma_per_load']): x['cycles_per_warp_instruction_per_sm'] for x in plain}
    return dict(floor_cycles=floor, degree_slope_cycles=slope, conflict_degree_rule='cost = max(floor, slope * degree), degree = largest number of distinct 32-bit words mapping to one of 32 banks per request group',
                volatile_by_stride={str(s): round(float(np.median([c for (ss, st), c in v.items() if ss == s])), 4) for s in sorted({s for s, _ in v})},
                plain_32bit_conflict_free=float(np.median([c for (w, s, f), c in p.items() if w == 32 and s in (0, 1, 2, 33) and f == 0])),
                plain_64bit_conflict_free=float(np.median([c for (w, s, f), c in p.items() if w == 64 and s == 2 and f == 0])),
                plain_128bit_conflict_free=float(np.median([c for (w, s, f), c in p.items() if w == 128 and s == 4 and f == 0])),
                ffma_interleave_changes_cost=bool(abs(np.median([c for (w, s, f), c in p.items() if w == 32 and s == 1 and f == 1]) - np.median([c for (w, s, f), c in p.items() if w == 32 and s == 1 and f == 0])) > 0.2))

# ----------------------------------------------------------------------------------------------- phase overlap (micro_overlap)

def fit_overlap(rows):
    g = collections.defaultdict(list)
    for x in rows:
        if x.get('mode') == 'overlap' and x['pairs'] >= 8 and x['T_mem_us'] >= 20 and x['T_cmp_us'] >= 20: g[x['resident_blocks_per_sm']].append(x['alpha'])
    if not g: raise ValueError('no overlap rows with at least 8 phase pairs and phases of at least 20 us')
    bs = sorted(g)
    return dict(resident_blocks_per_sm=bs, alpha=[round(float(np.median(g[b])), 3) for b in bs], configs_per_point=[len(g[b]) for b in bs],
                overlap_definition='alpha = (Tm + Tc - T) / (Tm + Tc - max(Tm, Tc)); 0 = different blocks phases fully serialised, 1 = fully overlapped',
                extrapolation='clamped to the measured range')

# ----------------------------------------------------------------------------------------------- pipes: latency, issue, barrier, clock, pointer chase (micro_pipes)
FAMILY_NAMES = ['barrier', 'integer', 'fp32_add', 'fp32_fma', 'mufu_ex2', 'shuffle', 'shared_load', 'shared_store_load']
OPS_PER_STEP = {1: 2}

def load_cell(cell_dir, row):
    """Device-cycle span of the timed loop per SM (last-finishing warp per block, averaged over blocks, median over three samples) and the median event time (ms)."""
    spans = []
    for s in range(3):
        p = Path(cell_dir) / ('sample%d.bin' % s)
        raw = p.read_bytes() if p.exists() else gzip.open(str(p) + '.gz', 'rb').read()
        lanes = row['blocks'] * row['threads']; warps = lanes // 32
        out_bytes = lanes * row['streams'] * 4
        cycles = np.frombuffer(raw[out_bytes:out_bytes + warps * 8], dtype='<u8').astype(float)
        spans.append(float(cycles.reshape(row['blocks'], row['threads'] // 32).max(axis=1).mean()))
    result = json.loads((Path(cell_dir) / 'result.json').read_text())
    return float(np.median(spans)), float(np.median(result['event_ms']))

def chase_cycles_per_step(cell_dir, row):
    """Median over samples and warps of device cycles per dependent load of a pointer-chase cell."""
    steps = row['table_bytes'] // 32 * 2; vals = []
    for s in range(3):
        raw = (Path(cell_dir) / ('sample%d.bin' % s)).read_bytes()
        lanes = row['blocks'] * row['threads']; warps = lanes // 32; out_bytes = lanes * row['streams'] * 4
        cycles = np.frombuffer(raw[out_bytes:out_bytes + warps * 8], dtype='<u8').astype(float)
        vals.append(float(np.median(cycles)) / steps)
    return float(np.median(vals))

def fit_group(points):
    A = np.array([[1.0, k, k * u] for k, u, _ in points]); y = np.array([c for _, _, c in points])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ coef - y) / np.maximum(np.abs(y), 1.0)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(resid.max())

def fit_pipes(table, rows_by_slot, sm):
    """table: {slot: (cycles, event_ms)} for compute cells. Identical definitions to constants/fit_microbench.py with the SM count of the device in place of 188."""
    groups = {}
    for slot, (cycles, _) in table.items():
        r = rows_by_slot[slot]
        if r['kind'] != 'compute': continue
        groups.setdefault((r['family'], r['streams'], r['threads'], r['blocks']), []).append((r['loops'], r['unroll'], cycles))
    fits = {key: fit_group(pts) for key, pts in groups.items() if len({(k, u) for k, u, _ in pts}) >= 4}
    out = {'dependent_latency_cycles': {}, 'issue_cycles_per_warp_instruction_per_sm': {}, 'barrier_latency_cycles_by_warps': {}, 'loop_overhead_cycles_per_iteration': {}, 'fit_quality_max_relative_residual': {}}
    longest = sorted((c for c, _ in table.values()), reverse=True)[:max(1, len(table) // 20)]
    clocks = [c / (ms * 1e-3) for c, ms in table.values() if c in longest and ms > 0]
    out['effective_sm_clock_hz'] = float(np.median(clocks)) if clocks else None
    for (f, S, t, b), (c0, c_loop, slope, res) in fits.items():
        name = FAMILY_NAMES[f]; ops = OPS_PER_STEP.get(f, 1)
        if f == 0 and S == 1 and b == 1: out['barrier_latency_cycles_by_warps'][str(t // 32)] = slope
        if f != 0 and S == 1 and b == 1 and t == 32:
            out['dependent_latency_cycles'][name] = slope / (S * ops); out['loop_overhead_cycles_per_iteration'][name] = c_loop
        if f != 0 and S == 4 and b == sm and t == 1024: out['issue_cycles_per_warp_instruction_per_sm'][name] = slope / ((t // 32) * S * ops)
        out['fit_quality_max_relative_residual']['%s/S%d/t%d/b%d' % (name, S, t, b)] = res
    return out

def fit_chase(cycles_by_tier, clock_hz):
    """cycles_by_tier: {'L1': cycles per dependent load, ...}; returns latency in ns at the effective SM clock."""
    return {k: round(v / clock_hz * 1e9, 2) for k, v in cycles_by_tier.items()}
