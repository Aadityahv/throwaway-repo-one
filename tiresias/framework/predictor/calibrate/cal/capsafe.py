"""Cap-safe energy windows: the pure part of the rule written in calibrate/CAP_SAFE_WINDOWS.md.

The probe itself runs in-process in csrc/cap_safe.h (NVML power while the window's own graph replays). This module holds the same choice formula in Python, so that
 * the scaling choice is unit-testable with synthetic probe powers, and
 * every window's recorded probe steps are replayed after the run (`verify_record`): a recorded block count that the rule would not have chosen refuses the window.
`next_blocks` must stay identical to `cap_next_blocks` in cap_safe.h (inputs are quantised to 0.001 mW on both sides so the recorded values reproduce the choice exactly).
"""
import math

CAP_SAFE_FRACTION = 0.90    # window power target as a fraction of the enforced limit; the admission rule (98.5% of the limit, cal/energy.py) is unchanged and separate
MAX_PROBES = 8          # rule version 2 (CAP_SAFE_WINDOWS.md Amendment 1)
MAX_PROBES_V1 = 4       # rule version 1, kept only to replay archived documents
RULE_VERSION = 2


def q(x):
    return math.floor(x * 1000.0 + 0.5) / 1000.0


def threshold_mw(fraction, limit_mw):
    return q(fraction * float(limit_mw))


def settle_measure_s(warmup_s):
    return min(6.0, max(1.0, 0.1 * warmup_s))


def next_blocks(blocks, power_mw, idle_mw, thr_mw):
    """Block count after a probe at `blocks` that read `power_mw`: unchanged when at or below the threshold, otherwise the linear proposal, always a strict decrease, at least one block."""
    if power_mw <= thr_mw: return blocks
    dyn = power_mw - idle_mw
    k = math.floor(blocks * (thr_mw - idle_mw) / dyn) if dyn > 0 else float(blocks - 1)
    return int(max(1.0, min(float(blocks - 1), float(k))))


def next_blocks_v2(steps, sm, idle_mw, thr_mw):
    """Rule version 2: block count after the last of `steps` = [(blocks, power_mw), ...]. Mirrors cap_next_blocks in cap_safe.h.
    At or below the threshold, or at one block: unchanged. More blocks than SMs: one block per SM. Fewer blocks than SMs (whole SMs idle) and still above: halve.
    At exactly one block per SM: linear proposal in dynamic power over active SMs. Always a strict decrease, at least one block."""
    b, p = steps[-1]
    if p <= thr_mw or b <= 1: return b
    if b > sm: return sm
    if b < sm: return max(1, b // 2)
    dyn = p - idle_mw
    k = math.floor(sm * (thr_mw - idle_mw) / dyn) if dyn > 0 else float(b - 1)
    return int(max(1.0, min(float(b - 1), float(k))))


def _replay_v2(cs, steps, bad):
    sm = cs.get('sm_count'); idle = cs['idle_mw']; thr = cs['threshold_mw']; expect_final = None; converged = True
    if not isinstance(sm, int) or sm < 1: bad.append('rule version 2 needs a positive sm_count'); return
    hist = [(s['blocks'], s['mean_mw']) for s in steps]
    for i, s in enumerate(steps):
        nb = next_blocks_v2(hist[:i + 1], sm, idle, thr)
        if nb == s['blocks']:
            if i != len(steps) - 1: bad.append('probe continued after the rule had settled at %d blocks' % nb)
            expect_final = nb; converged = s['mean_mw'] <= thr; break
        if i + 1 < len(steps):
            if steps[i + 1]['blocks'] != nb: bad.append('probe %d ran at %d blocks, the rule gives %d' % (i + 2, steps[i + 1]['blocks'], nb))
        else:
            if len(steps) != MAX_PROBES: bad.append('probing stopped after %d steps without settling' % len(steps))
            expect_final = s['blocks']; converged = False      # the probe budget is spent: the last measured grid is used
    if expect_final is not None and expect_final != cs['blocks_chosen']: bad.append('chosen %d blocks, the rule gives %d' % (cs['blocks_chosen'], expect_final))
    if bool(cs.get('converged')) != converged: bad.append('converged flag disagrees with the probe steps')


def verify_record(cs, cap_w=None):
    """Problems (empty list = the recorded choice is exactly what the rule gives) in the `cap_safe` object of a window row."""
    if not cs or not cs.get('enabled'): return []
    bad = []; full = cs['blocks_full']; chosen = cs['blocks_chosen']
    if cap_w is not None and abs(cs['limit_mw'] - cap_w * 1000.0) > 1: bad.append('probe power limit %s mW differs from the stage cap %s W' % (cs['limit_mw'], cap_w))
    if abs(cs['fraction'] - round(cs['fraction'], 4)) > 1e-12: bad.append('fraction %r has more than 4 decimals' % cs['fraction'])
    thr = threshold_mw(round(cs['fraction'], 4), cs['limit_mw'])
    if abs(thr - cs['threshold_mw']) > 5e-4: bad.append('threshold %.3f mW is not fraction x limit (%.3f)' % (cs['threshold_mw'], thr))
    if not 1 <= chosen <= full: bad.append('chosen blocks %d outside 1..%d' % (chosen, full))
    if cs.get('forced'):
        if cs.get('probe_steps'): bad.append('a forced window must not carry probe steps')
        if cs.get('scaled') != (chosen != full): bad.append('scaled flag disagrees with the block counts')
        return bad
    steps = cs.get('probe_steps') or []
    if not steps: return bad + ['no probe steps recorded']
    version = cs.get('rule_version', 1)      # absent = the original rule (archived documents)
    if version not in (1, 2): return bad + ['unknown cap-safe rule version %r' % (version,)]
    limit = MAX_PROBES if version == 2 else MAX_PROBES_V1
    if len(steps) > limit: bad.append('more than %d probe steps' % limit)
    if steps[0]['blocks'] != full: bad.append('first probe at %d blocks, not the full grid %d' % (steps[0]['blocks'], full))
    if version == 2:
        _replay_v2(cs, steps, bad)
        if cs.get('scaled') != (chosen != full): bad.append('scaled flag disagrees with the block counts')
        return bad
    idle = cs['idle_mw']; thr = cs['threshold_mw']; expect_final = None; converged = True
    for i, s in enumerate(steps):
        nb = next_blocks(s['blocks'], s['mean_mw'], idle, thr)
        if nb == s['blocks']:
            if i != len(steps) - 1: bad.append('probe continued after the rule had settled at %d blocks' % nb)
            expect_final = nb; converged = s['mean_mw'] <= thr; break
        if i + 1 < len(steps):
            if steps[i + 1]['blocks'] != nb: bad.append('probe %d ran at %d blocks, the rule gives %d' % (i + 2, steps[i + 1]['blocks'], nb))
        else:
            if len(steps) != MAX_PROBES_V1: bad.append('probing stopped after %d steps without settling' % len(steps))
            expect_final = nb; converged = False
    if expect_final is not None and expect_final != chosen: bad.append('chosen %d blocks, the rule gives %d' % (chosen, expect_final))
    if bool(cs.get('converged')) != converged: bad.append('converged flag disagrees with the probe steps')
    if cs.get('scaled') != (chosen != full): bad.append('scaled flag disagrees with the block counts')
    return bad


def window_args(cap_fraction, forced_blocks=None):
    """Extra command-line arguments of the window programs (empty = the previous behaviour, full grid, no probe)."""
    if not cap_fraction: return []
    return [str(cap_fraction)] + ([str(int(forced_blocks))] if forced_blocks else [])
