"""Static runtime predictions, model v3k: the shared-traffic model v3j with three composition rules for tensor-core kernels and grids above one block per SM (proposal and diagnosis:
attention_diagnosis/DESIGN.md, RESULT.md; prospective test: prospective_test/DESIGN addendum). v3j and everything below it are imported unchanged.

Rule 1, pair overlap. In a barrier phase containing both shared loads and tensor-core MMA, the two pipes do not overlap fully: phase time of the pair = max(s, m) + beta * min(s, m), beta from the
dependent fragment-load microbenchmark (constants/pair_overlap_constants.json), linearly interpolated in resident warps per SM (resident blocks x warps per block).
Rule 2, stage serialisation. In a phase containing MMA, the remaining pipes (exp, shuffles, FP32, integer) belong to other stages of the dependent chain: issue time = pair + max(remaining pipes),
bounded below by the dispatch limit. Phases without MMA are untouched by rules 1 and 2.
Rule 3, busiest SM. For grids above one block per SM, per-SM issue work is set by the busiest SM: issue terms are multiplied by ceil(blocks / SMs) / (blocks / SMs). Applies to every kernel.
No constant is fitted on any operator cell. With all three rules off (OPTIONS) the model is v3j exactly; a phase or kernel that no rule touches keeps v3j's numbers bit for bit.
Implementation: v3j's kernel_terms is wrapped (its result recomputed for the affected phases); the overlap of phases across resident blocks (v3i) and the traffic accounting are v3j's.
"""
import collections
import contextlib
import json
import math
from pathlib import Path

import numpy as np

import predict_runtime_v2 as V2
import predict_runtime_v3 as V3
import predict_runtime_v3h as V3H
import predict_runtime_v3j as J

HERE = Path(__file__).resolve().parent
OPTIONS = dict(pair=True, stage=True, busiest=True)
TENSOR_PREFIXES = ('HMMA', 'IMMA', 'QMMA', 'DMMA')
_PAIR = json.loads((HERE / 'constants/pair_overlap_constants.json').read_text())


def beta(resident_warps):
    return float(np.interp(resident_warps, _PAIR['resident_warps_per_sm'], _PAIR['beta_hmma']))


def phase_pipes(ph, K):
    """(pipe seconds-numerators in cycles, dispatch cycles) of a phase: the classes of V3H.issue_cycles_v3h (tensor class through the patched V2.classify), shared cost from the bank table."""
    shared = V3H._SHARED_BY_PHASE.get(id(ph))
    if shared is None: raise ValueError('bank-conflict shared cost unavailable for a phase')
    pipes = collections.Counter(); total = 0
    for op, n in ph['issue_warp_instructions'].items():
        total += n
        cls, mult = V2.classify(op)
        if cls == 'shared_load': continue
        if cls: pipes[cls] += n * mult * K['issue'][cls]
    pipes['shared_load'] = shared
    return pipes, V3.DISPATCH_CYCLES * total


_kernel_terms_v3j = J.kernel_terms


def kernel_terms_v3k(phases, uphases, ukernel, occ, geometry, tier, K, V3c, sm_count, barrier_releases, V3C, V3E):
    r = _kernel_terms_v3j(phases, uphases, ukernel, occ, geometry, tier, K, V3c, sm_count, barrier_releases, V3C, V3E)
    if not (OPTIONS['pair'] or OPTIONS['stage'] or OPTIONS['busiest']): return r
    blocks = geometry.get('grid_blocks') or geometry.get('grid_blocks_dispatch')
    active = sm_count * occ['active_sm_fraction']; clock = K['clock']
    resident = min(occ['blocks_per_sm'], math.ceil(blocks / sm_count)); warps = resident * (occ.get('warps_per_block') or 1)
    tail = (math.ceil(blocks / sm_count) / (blocks / sm_count)) if (OPTIONS['busiest'] and blocks > sm_count) else 1.0
    terms = [dict(t) for t in r['phase_terms']]; changed = False
    for ph, t in zip(phases, terms):
        has_mma = any(op.startswith(TENSOR_PREFIXES) for op in ph['issue_warp_instructions'])
        if not (has_mma and (OPTIONS['pair'] or OPTIONS['stage'])) and tail == 1.0: continue
        pipes, disp = phase_pipes(ph, K); issue_cycles = None
        if has_mma and (OPTIONS['pair'] or OPTIONS['stage']) and pipes.get('tensor_mma', 0) > 0 and pipes.get('shared_load', 0) > 0:
            sh, tc = pipes.pop('shared_load'), pipes.pop('tensor_mma'); pair = max(sh, tc) + beta(warps) * min(sh, tc)
            if OPTIONS['stage']: issue_cycles = max(disp, pair + max(pipes.values(), default=0.0))
            else: issue_cycles = max(disp, pair, max(pipes.values(), default=0.0))
        else:
            issue_cycles = max(disp, max(pipes.values(), default=0.0))
        issue = issue_cycles / (active * clock) * tail
        if issue != t['issue_s']: t['issue_s'] = issue; changed = True
    if not changed: return r
    rep = lambda k: sum(p[k] * p['repetitions'] for p in terms)
    totals = {k: rep(k) for k in ('stream_s', 'latency_s', 'issue_s')}
    body = sum(max(p['stream_s'], p['latency_s'], p['issue_s']) * p['repetitions'] for p in terms) + r['barrier_s']
    t0, dispatch, bar = r['launch_s'], r['dispatch_s'], r['barrier_s']
    return dict(r, phase_terms=terms, primary_s=t0 + max(body, dispatch), max_s=t0 + max(max(totals.values()) + bar, dispatch), sum_s=t0 + max(sum(totals.values()) + bar, dispatch))


@contextlib.contextmanager
def _installed(**options):
    old = dict(OPTIONS); OPTIONS.update(options); J.kernel_terms = kernel_terms_v3k
    try: yield
    finally: OPTIONS.clear(); OPTIONS.update(old); J.kernel_terms = _kernel_terms_v3j


def build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E, OV, read_footprint=True, write_footprint=False, l2_rule='wave', pair=True, stage=True, busiest=True):
    """Same signature as predict_runtime_v3j.build plus the three rule switches (all False reproduces v3j)."""
    with _installed(pair=pair, stage=stage, busiest=busiest):
        return J.build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E, OV, read_footprint=read_footprint, write_footprint=write_footprint, l2_rule=l2_rule)


class _Shim:
    def __init__(self, **rules): self.rules = rules
    def build(self, features, phase_rows, unique_rows, bank_rows, K, v3c, V3C, V3E, OV):
        return build(features, phase_rows, unique_rows, bank_rows, K, v3c, V3C, V3E, OV, **self.rules)


def predict_portable(features, phase_rows, unique_rows, bank_rows, constants, sm_count, pair=True, stage=True, busiest=True):
    """calibrate/cal/portable_predict.predict with this model behind it (the SM count, tensor class and calibrated shared-memory cost patches apply unchanged); energy: cal.traffic.energy_rows_traffic."""
    import sys
    sys.path.insert(0, str(HERE / 'calibrate'))
    from cal import portable_predict as PP
    old = PP.V3I; PP.V3I = _Shim(pair=pair, stage=stage, busiest=busiest)
    try: return PP.predict(features, phase_rows, unique_rows, bank_rows, constants, sm_count)
    finally: PP.V3I = old
