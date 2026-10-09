"""Tensor-core stage of the packaged calibrator (optional, additive). CPU-only analysis; the GPU part is csrc/micro_tensor.cu.

What it adds to a calibration document (`constants.tensor`), without touching the validated runtime and energy stages:
 * issue cost and dependent latency of the bf16 m16n8k16 MMA (`issue_cycles_per_warp_instruction_per_sm`, `dependent_latency_cycles`), exported to the runtime model as the instruction class `tensor_mma`;
 * a tensor energy rate (pJ per lane-instruction, one warp-level MMA = 32 lane-instructions) DERIVED from three energy windows at 1, 4 and 16 warps per SM, using the base power and the rates the energy stage
   already fitted: residual = window energy - base power * runtime - sum(other fitted rates * the window's other counts); rate = relative-error-weighted least squares of the residual on the MMA count.
 A rate whose windows cannot be admitted (power rule, SASS check) or whose windows disagree with the fitted rate by more than 10% is NOT used: the status is UNIDENTIFIED or UNSTABLE and a kernel with tensor
 instructions is then reported unsupported by predict.py. Nothing is defaulted and no other board's number is substituted.
"""
import numpy as np

from . import energy as En

WINDOWS = ['tc_w1', 'tc_w4', 'tc_w16']        # warps per SM: 1, 4, 16
M_PER_TRIP = 32                               # MMAs per loop iteration in csrc/micro_tensor.cu (mma_energy<32>)
MAX_WINDOW_ERROR = 0.10
MAX_ISSUE_FIT_RESIDUAL = 0.10   # only the issue-cost fit feeds the runtime model; the pipe stage records the residual of every fit and rejects none (its one-warp latency fits reach 0.21), so the latency fit is recorded the same way


def fit_issue(doc, sm_count):
    """The pipe stage's own fit (`fits.fit_group`): cycles = c0 + loop_overhead * loops + step_cycles * loops * unroll over the six (loops, unroll) points of each cell group; the dependent latency is the
    step cost of the one-warp, one-chain cell, the issue cost the step cost of the 32-warp, four-chain cell divided by the warp instructions it issues per step."""
    from .fits import fit_group
    groups = {}
    for r in doc['cells']: groups.setdefault((r['streams'], r['threads'], r['blocks']), []).append((r['loops'], r['unroll'], r['cycles_median']))
    need = {(1, 32, 1), (4, 1024, sm_count)}
    if not need <= set(groups): raise ValueError('tensor issue cells missing: %s' % sorted(need - set(groups)))
    out = dict(dependent_latency_cycles=None, issue_cycles_per_warp_instruction_per_sm=None, loop_overhead_cycles_per_iteration=None, fit_quality_max_relative_residual={})
    for (S, t, b), pts in groups.items():
        if (S, t, b) not in need: continue
        if len({(k, u) for k, u, _ in pts}) < 4: raise ValueError('tensor %s: fewer than four distinct (loops, unroll) points' % ((S, t, b),))
        c0, c_loop, slope, resid = fit_group(pts)
        out['fit_quality_max_relative_residual']['S%d_t%d_b%d' % (S, t, b)] = resid
        if S > 1 and resid > MAX_ISSUE_FIT_RESIDUAL: raise ValueError('tensor %s cells do not fit the loop model (max relative residual %.3f)' % ((S, t, b), resid))
        if S == 1: out['dependent_latency_cycles'] = float(slope); out['loop_overhead_cycles_per_iteration'] = float(c_loop)
        else: out['issue_cycles_per_warp_instruction_per_sm'] = float(slope / ((t // 32) * S))
    return out


def verify_design(in_loop, m=M_PER_TRIP):
    bad = []
    if in_loop.get('tensor_core', 0) != m: bad.append('tensor_core: SASS %d per iteration, design %d' % (in_loop.get('tensor_core', 0), m))
    if in_loop.get('global_load', 0) != 1: bad.append('global_load: SASS %d per iteration, design 1' % in_loop.get('global_load', 0))
    for fam in ('fp32_fma', 'fp32_add', 'special_function', 'shared_store', 'shared_load', 'shuffle', 'barrier', 'global_store'):
        if in_loop.get(fam, 0): bad.append('%s: SASS %d per iteration but the design has none' % (fam, in_loop[fam]))
    return bad


def derive_rate(windows, energy):
    """windows: analysed tensor windows (En.analyse_window records with `columns` incl. `tc`). energy: constants.energy of the same document (profile with base power and rates)."""
    out = dict(status='UNIDENTIFIED', rate_pJ_per_lane_instruction=None, windows=[], reason='')
    if not energy or 'profile' not in energy: out['reason'] = 'no energy profile in the document'; return out
    prof = energy['profile']; base = prof['base_power_w']; rates = prof['rates_pJ']
    adm = [w for w in windows if w['admitted']]
    out['admitted'] = [w['id'] for w in adm]; out['not_admitted'] = [w['id'] for w in windows if not w['admitted']]
    if len(adm) < 2: out['reason'] = 'fewer than two admitted tensor windows (%d)' % len(adm); return out
    used = sorted({c for w in adm for c in En.COLUMNS if w['columns'].get(c, 0) > 0})
    missing = [c for c in used if c not in rates]
    if missing: out['reason'] = 'the energy profile has no rate for %s, which the tensor windows use' % ', '.join(missing); return out
    rows = []
    for w in adm:
        t, E = w['runtime_s_per_launch'], w['energy_j_per_launch']; tc = float(w['columns']['tc'])
        if not (t > 0 and E > 0 and tc > 0): out['reason'] = 'non-positive energy, runtime or MMA count in %s' % w['id']; return out
        other = sum(rates[c] * w['columns'].get(c, 0.0) * 1e-12 for c in En.COLUMNS if c in rates)
        rows.append(dict(id=w['id'], E=E, t=t, tc=tc, resid=E - base * t - other))
    wts = np.array([1.0 / r['E'] for r in rows]); resid = np.array([r['resid'] for r in rows]); tc = np.array([r['tc'] for r in rows])
    rate = float(np.sum(wts ** 2 * resid * tc) / np.sum(wts ** 2 * tc ** 2))           # joules per lane-instruction
    if not rate > 0: out['reason'] = 'the fitted tensor rate is not positive'; return out
    worst = 0.0
    for r in rows:
        pred_resid = rate * r['tc']; err = (base * r['t'] + (r['E'] - base * r['t'] - r['resid']) + pred_resid) / r['E'] - 1
        out['windows'].append(dict(id=r['id'], residual_energy_j=r['resid'], tensor_lane_instructions=r['tc'], implied_pJ_per_lane_instruction=r['resid'] / r['tc'] * 1e12, window_error_pct=round(100 * err, 2)))
        worst = max(worst, abs(err))
    out.update(rate_pJ_per_lane_instruction=rate * 1e12, max_window_error_pct=round(100 * worst, 2), status='ok' if worst <= MAX_WINDOW_ERROR else 'UNSTABLE',
               reason='' if worst <= MAX_WINDOW_ERROR else 'a window differs from the fitted rate by more than %d%%' % int(100 * MAX_WINDOW_ERROR))
    return out


def finalize(raw, energy, sm_count):
    """raw: results of the tensor stage (issue document and analysed windows). Returns constants.tensor."""
    c = dict(status='ok', issue=None, energy=None, windows=[{k: w[k] for k in ('id', 'design', 'energy_j_per_launch', 'runtime_s_per_launch', 'power_w', 'temp_start_c', 'temp_end_c', 'clock_median_mhz', 'admitted', 'columns')} for w in raw.get('windows', [])],
             sass_verified=raw.get('sass_verified'))
    try: c['issue'] = fit_issue(raw['issue'], sm_count)
    except (KeyError, ValueError) as ex: c['issue'] = dict(status='failed', reason=str(ex)); c['status'] = 'incomplete'
    c['energy'] = derive_rate(raw.get('windows', []), energy)
    if c['energy']['status'] != 'ok': c['status'] = 'incomplete'
    return c
