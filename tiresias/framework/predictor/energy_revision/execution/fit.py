"""Predeclared nonnegative component fitting and admission; calibration labels only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import component_candidate as C

MIXES = ('memory', 'arithmetic', 'special_function', 'other_shared')
FOOTPRINTS = ('small_candidate', 'l2_candidate', 'dram_candidate')
FIT_IDS = tuple(f'{mix}/{dose}/{foot}' for mix in MIXES for dose in ('low', 'high') for foot in FOOTPRINTS)
HOLD_IDS = ('mixed/small_candidate', 'mixed/l2_candidate', 'mixed/dram_candidate',
            'mixed_altered/dram_candidate', 'anchor_memory/l2_candidate', 'anchor_arithmetic/small_candidate')
POOLED = ('base_time', 'pooled_activity', 'lookup_bytes', 'dram_bytes')


def fingerprint(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def audit_matrix(x, names, max_condition):
    scale = np.linalg.norm(x, axis=0)
    if np.any(scale <= 0):
        raise ValueError('REFUSED: unexcited component column')
    normalized = x / scale
    rank = int(np.linalg.matrix_rank(normalized))
    condition = float(np.linalg.cond(normalized))
    if rank != len(names) or condition > max_condition:
        raise ValueError(f'REFUSED: unidentifiable calibration: rank={rank}, normalized condition={condition:.6g}')
    return dict(rank=rank, normalized_condition=condition, column_norms=scale.tolist())


def solve(x, y, names, limit):
    audit = audit_matrix(x, names, limit)
    # Fixed relative-joule squared loss; normalize the columns, not the physical units.
    scale = np.asarray(audit['column_norms'])
    design = (x / scale) / y[:, None]
    weights, residual = nnls(design, np.ones(len(y)), maxiter=100 * len(names))
    coef = weights / scale
    return coef, dict(audit, relative_squared_residual=float(residual ** 2))


def metrics(pred, actual):
    error = np.abs(pred / actual - 1) * 100
    return dict(median_ape_pct=float(np.median(error)), p90_ape_pct=float(np.percentile(error, 90)), max_ape_pct=float(np.max(error)))


def fit_profiles(design, acquisition, policy):
    if design['schema'] != 'energy_component_calibration_design/1' or acquisition['schema'] != 'energy_component_calibration_measurements/1':
        raise ValueError('unsupported calibration schema')
    if design['status'] != 'compiled_counts_and_correctness_frozen':
        raise ValueError('REFUSED: source, binary, ABI, exact counts and correctness gates are not frozen')
    if acquisition['design_sha256'] != fingerprint(design):
        raise ValueError('REFUSED: measured/static design hash mismatch')
    if acquisition.get('synthetic', False):
        raise ValueError('REFUSED: synthetic fixtures are not calibration evidence')
    if acquisition['target_role'] != 'calibration_only':
        raise ValueError('REFUSED: target labels cannot fit this head')
    slots = {r['design_id']: r for r in design['rows']}
    records = {r['design_id']: r for r in acquisition['rows']}
    required = set(FIT_IDS + HOLD_IDS)
    if len(slots) != len(design['rows']) or len(records) != len(acquisition['rows']):
        raise ValueError('duplicate calibration slot')
    if set(slots) != required or set(records) != required:
        raise ValueError('REFUSED: complete 24-fit/6-heldout grid required; no silent exclusions')
    cap = design['cap_w']
    if not math.isfinite(cap) or cap <= 0 or cap != policy['cap_w']:
        raise ValueError('unverified or mismatched cap')
    x, y = [], []
    for did in FIT_IDS + HOLD_IDS:
        s, r = slots[did], records[did]
        expected_role = 'fit' if did in FIT_IDS else 'heldout'
        if s['role'] != expected_role or r['role'] != expected_role:
            raise ValueError('REFUSED: fit/heldout assignment changed')
        if r['status'] != 'accepted' or r['attempts'] != 1 or not r['correctness_pass']:
            raise ValueError(f'REFUSED: failed/retried slot {did}; preserve it, do not shrink grid')
        if s['count_status'] != 'exact' or s['abi_status'] != 'verified':
            raise ValueError('REFUSED: calibration count/ABI proof missing')
        for name in ('source_sha256', 'binary_sha256', 'count_evidence_sha256'):
            if not isinstance(s.get(name), str) or len(s[name]) != 64 or any(c not in '0123456789abcdef' for c in s[name]):
                raise ValueError('REFUSED: incomplete source/binary/count provenance')
            if r.get(name) != s[name]:
                raise ValueError('REFUSED: acquisition binary/count provenance mismatch')
        if not isinstance(r.get('trace_sha256'), str) or len(r['trace_sha256']) != 64:
            raise ValueError('REFUSED: raw trace hash missing')
        t, e, count = r['counted_runtime_s_per_launch'], r['energy_j_per_launch'], r['counted_launches']
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (t, e, count)):
            raise ValueError('invalid calibration measurement')
        if count != int(count):
            raise ValueError('fractional counted launch count')
        # Both intervals are REAL recorded durations, not requested window lengths.
        for key in ('counted_interval_s', 'precondition_cuda_s'):
            if not math.isfinite(r[key]) or r[key] <= 0:
                raise ValueError('missing actual acquisition interval')
        if not math.isclose(t * count, r['counted_interval_s'], rel_tol=1e-6):
            raise ValueError('REFUSED: launch normalization/time mismatch')
        if e / t >= policy['below_cap_fraction'] * cap:
            raise ValueError('REFUSED: cap-limited calibration slot; no silent row removal')
        a = C.activity(s['work'], s['logical_bytes'], s['tier']); a['base_time'] = t
        x.append([a[k] for k in C.FEATURES]); y.append(e)
    x, y = np.asarray(x), np.asarray(y)
    train = slice(0, 24); hold = slice(24, 30)
    six, six_audit = solve(x[train], y[train], C.FEATURES, policy['max_normalized_condition'])
    z = np.column_stack((x[:, 0], x[:, 1:4].sum(axis=1), x[:, 4], x[:, 5]))
    four, four_audit = solve(z[train], y[train], POOLED, policy['max_normalized_condition'])
    constant = float(np.mean(y[train] / x[train, 0]))
    held = dict(class_aware=metrics(x[hold] @ six, y[hold]), pooled=metrics(z[hold] @ four, y[hold]), constant=metrics(x[hold, 0] * constant, y[hold]))
    dose_checks = {}
    for dose in ('low', 'high'):
        ix = [i for i, did in enumerate(FIT_IDS) if did.split('/')[1] == dose]
        beta, audit = solve(x[ix], y[ix], C.FEATURES, policy['max_normalized_condition'])
        # Stability of effective energy predictions; coefficients are NOT physical prices.
        gap = float(np.max(np.abs((x @ beta) / (x @ six) - 1)) * 100)
        dose_checks[dose] = dict(audit, max_prediction_shift_pct=gap)
    reasons = []
    if held['class_aware']['median_ape_pct'] > policy['heldout_max_median_ape_pct']:
        reasons.append('heldout median exceeds fixed limit')
    if held['class_aware']['max_ape_pct'] > policy['heldout_max_cell_ape_pct']:
        reasons.append('heldout cell error exceeds fixed limit')
    if held['class_aware']['median_ape_pct'] > held['pooled']['median_ape_pct'] + policy['heldout_max_median_degradation_vs_pooled_pct']:
        reasons.append('heldout class separation harms the matched pooled head')
    if any(a['max_prediction_shift_pct'] > policy['max_dose_prediction_shift_pct'] for a in dose_checks.values()):
        reasons.append('effective charges unstable across dose fits')
    costs = dict(acquired_windows=30,fit_windows=24,heldout_windows=6,
                 counted_cuda_s=sum(r['counted_interval_s'] for r in acquisition['rows']),
                 precondition_cuda_s=sum(r['precondition_cuda_s'] for r in acquisition['rows']),
                 total_wall_s=acquisition.get('total_wall_s'),
                 note='Measured CUDA intervals only unless total_wall_s is supplied. Compilation, gates and probes must also be logged; never treated as free.')
    result = dict(schema='admitted_class_aware_component/1', status='rejected' if reasons else 'calibrated_and_frozen',
                  rejection_reasons=reasons, cap_w=cap, coefficient_unit_system='W_and_J_per_activity',
                  coefficients=dict(zip(C.FEATURES, map(float, six))), matched_pooled_coefficients=dict(zip(POOLED, map(float, four))),
                  matched_constant_power_w=constant, admission=dict(class_aware_matrix=six_audit,pooled_matrix=four_audit,heldout=held,dose_checks=dose_checks),
                  calibration_budget=costs, inputs_content_sha256=dict(design=fingerprint(design), acquisition=fingerprint(acquisition), policy=fingerprint(policy)),
                  library_versions=dict(numpy=np.__version__, scipy=scipy.__version__),
                  qualification='Effective empirical charges, not physical component prices. Training uses measured calibration runtime; target inference uses the common static runtime vector.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', required=True, type=Path)
    p.add_argument('--measurements', required=True, type=Path)
    p.add_argument('--policy', default=HERE / 'fit_policy.json', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    docs = [json.loads(f.read_text()) for f in (a.design, a.measurements, a.policy)]
    result = fit_profiles(*docs)
    result['code_sha256'] = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), HERE.parent / 'component_candidate.py')}
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(result['status'])
    if result['status'] != 'calibrated_and_frozen':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
