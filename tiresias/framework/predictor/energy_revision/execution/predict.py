"""Static energy heads on ONE external runtime vector; never imports CUDA or labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import component_candidate as candidate


def positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"invalid {name}")
    return value


def component(row, profile):
    """Published-structure adaptation with its ORIGINAL source-operation units."""
    if profile['activity_convention'] != 'source_operations_plus_logical_bytes_div_128':
        raise ValueError('REFUSED: legacy coefficients require the original source-operation proxy')
    t = positive(row['predicted_runtime_s'], 'runtime')
    B = positive(row['logical_bytes'], 'logical bytes', zero=True)
    operations = positive(row['source_operations'], 'source operations', zero=True)
    if row['tier'] not in ('L2', 'DRAM'):
        raise ValueError('REFUSED: original component calibration supports L2/DRAM only')
    c = profile['coefficients']
    parts = dict(base_time=c['P_const'] * t,
                 source_operation_proxy=c['e_inst'] * (operations + B / 128),
                 lookup_bytes=c['e_B'] * B,
                 dram_bytes=c['e_DRAM'] * (B if row['tier'] == 'DRAM' else 0))
    total = sum(parts.values())
    return dict(energy_j=min(total, profile['cap_w'] * t), uncapped_j=total,
                capped=total > profile['cap_w'] * t, components_j=parts)


def predict_rows(request, reference, calibrated=None):
    if request['schema'] != 'static_energy_request/1':
        raise ValueError('unsupported request schema')
    if reference['schema'] != 'frozen_component_reference/1':
        raise ValueError('unsupported reference schema')
    if reference['status'] != 'frozen':
        raise ValueError('unfrozen reference')
    if reference.get('coefficient_unit_system') != 'W_and_J_per_activity':
        raise ValueError('REFUSED: reference units must be explicit')
    positive(reference['cap_w'], 'cap')
    if set(reference['coefficients']) != {'P_const', 'e_inst', 'e_B', 'e_DRAM'}:
        raise ValueError('invalid source-proxy coefficient basis')
    for v in reference['coefficients'].values():
        positive(v, 'reference coefficient', zero=True)
    positive(reference['constant_power_w'], 'constant power')
    ids = [r['cell_id'] for r in request['rows']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate target grid')
    rows = []
    for r in request['rows']:
        t = positive(r['predicted_runtime_s'], 'runtime')
        methods = {}
        # A refused head remains a row in the declared denominator, never a silent fallback.
        try:
            methods['source_proxy_component'] = dict(status='ok', **component(r, reference))
        except (ValueError, KeyError) as exc:
            methods['source_proxy_component'] = dict(status='unsupported', reason=str(exc))
        methods['constant_power'] = dict(status='ok', energy_j=min(reference['constant_power_w'], reference['cap_w']) * t)
        if calibrated is not None:
            try:
                if calibrated['status'] != 'calibrated_and_frozen':
                    raise ValueError('REFUSED: calibration has not passed admission')
                if calibrated['cap_w'] != reference['cap_w']:
                    raise ValueError('REFUSED: comparison cap mismatch')
                positive(calibrated['matched_constant_power_w'], 'matched constant power')
                methods['budget_matched_constant_power'] = dict(status='ok', energy_j=min(calibrated['matched_constant_power_w'], reference['cap_w']) * t)
            except (ValueError, KeyError) as exc:
                for name in ('class_aware_component', 'budget_matched_pooled_component', 'budget_matched_constant_power'):
                    methods[name] = dict(status='unsupported', reason=str(exc))
                rows.append(dict(cell_id=r['cell_id'], predicted_runtime_s=t, methods=methods))
                continue
            try:
                work = r['work']; B = r['logical_bytes']; tier = r['tier']
                methods['class_aware_component'] = dict(status='ok', **candidate.predict(work, B, tier, t, calibrated))
                x = candidate.activity(work, B, tier)
                p = calibrated['matched_pooled_coefficients']
                if set(p) != {'base_time', 'pooled_activity', 'lookup_bytes', 'dram_bytes'}:
                    raise ValueError('REFUSED: incomplete pooled coefficient basis')
                for v in p.values(): positive(v, 'pooled coefficient', zero=True)
                energy = (p['base_time'] * t + p['pooled_activity'] * sum(x[k] for k in ('fp32_work', 'other_non_global', 'special_function'))
                          + p['lookup_bytes'] * B + p['dram_bytes'] * x['dram_bytes'])
                methods['budget_matched_pooled_component'] = dict(status='ok', energy_j=min(energy, reference['cap_w'] * t), uncapped_j=energy)
            except (ValueError, KeyError) as exc:
                for name in ('class_aware_component', 'budget_matched_pooled_component'):
                    methods[name] = dict(status='unsupported', reason=str(exc))
        rows.append(dict(cell_id=r['cell_id'], predicted_runtime_s=t, methods=methods))
    return dict(schema='runtime_matched_static_energy/1', requested_cells=len(rows), rows=rows,
                runtime_vector_sha256=hashlib.sha256(json.dumps([[r['cell_id'], r['predicted_runtime_s']] for r in request['rows']], separators=(',', ':')).encode()).hexdigest(),
                information_class='S', note='All heads share one supplied static runtime. Unsupported rows retained; no target execution or energy labels.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True, type=Path)
    p.add_argument('--reference', type=Path, default=HERE / 'reference_profile.json')
    p.add_argument('--calibrated', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    inputs = [a.request, a.reference] + ([a.calibrated] if a.calibrated else [])
    docs = [json.loads(f.read_text()) for f in inputs]
    out = predict_rows(docs[0], docs[1], docs[2] if len(docs) > 2 else None)
    out['inputs_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    out['code_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), HERE.parent / 'component_candidate.py')}
    # Exclusive creation prevents replacement of a prediction freeze.
    with a.out.open('x') as f:
        json.dump(out, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(f"{len(out['rows'])} targets retained; common runtime vector hashed")


if __name__ == '__main__':
    main()
