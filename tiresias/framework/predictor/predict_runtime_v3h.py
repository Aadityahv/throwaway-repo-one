"""Static runtime predictions, model v3h (preregistered in planning/STATIC_RUNTIME_V3H_PREREGISTRATION_2026-10-02.md).

The component model fitted on the stream grid (v3f) with ONE change: the shared-memory pipe cost of a barrier phase is no longer
(shared warp instructions x flat 4.0 cycles) but the statically derived bank-conflict cost from bank/bank_conflicts_*.json:
cycles per warp request = max(2.0, conflict degree), the rule measured by micro_v3/out_smem.jsonl. Every other term, constant and
rule is v3f's. A phase with unknown shared addresses, or whose bank table is missing or refused, is refused (never a flat fallback).
Reads static tables and microbenchmark constants only; no runtime or energy label.
"""
import argparse
import collections
import json
from pathlib import Path

import predict_runtime_v2 as V2
import predict_runtime_v3 as V3
import predict_runtime_v3f as V3F

HERE = Path(__file__).resolve().parent
_SHARED_BY_PHASE = {}  # id(phase dict) -> shared cost cycles for that phase, filled per cell by build()


def issue_cycles_v3h(ph, K):
    shared = _SHARED_BY_PHASE.get(id(ph))
    if shared is None: raise ValueError('bank-conflict shared cost unavailable for a phase')
    pipes = collections.Counter(); total = 0
    for op, n in ph['issue_warp_instructions'].items():
        total += n
        cls, mult = V2.classify(op)
        if cls == 'shared_load': continue  # replaced by the bank-conflict cost below
        if cls: pipes[cls] += n * mult * K['issue'][cls]
    pipes['shared_load'] = shared
    return max(V3.DISPATCH_CYCLES * total, max(pipes.values(), default=0.0))


def build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E):
    original = V3.issue_cycles; V3.issue_cycles = issue_cycles_v3h
    try:
        return _build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E)
    finally:
        V3.issue_cycles = original


def _build(features, phase_rows, unique_rows, bank_rows, K, V3c, V3C, V3E):
    out = {}
    for f in features['rows']:
        cid = f['cell_id']; ph = phase_rows[cid]; b = bank_rows.get(cid)
        try:
            if ph['status'] == 'unsupported': raise ValueError(ph['reason'])
            if not b or 'kernels' not in b: raise ValueError('bank-conflict table unavailable: ' + str((b or {}).get('reason') or (b or {}).get('status')))
            for kk, bk in zip(ph['kernels'], b['kernels']):
                if len(kk['phases']) != len(bk['phases']): raise ValueError('bank table phase count differs from the phase table')
                for p, bp in zip(kk['phases'], bk['phases']):
                    sh = bp['shared']
                    if sh.get('shared_requests_unknown'): raise ValueError('shared address unknown in a phase (' + str(sh.get('shared_unknown_reasons')) + ')')
                    _SHARED_BY_PHASE[id(p)] = float(sh['shared_cost_cycles'] or 0.0)
            out.update(V3F.build({**features, 'rows': [f]}, {cid: ph}, unique_rows, K, V3c, V3C, V3E))
        except (ValueError, KeyError, TypeError) as ex:
            out[cid] = dict(primary_s=None, max_s=None, sum_s=None, unsupported_reason=str(ex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True); ap.add_argument('--phases', type=Path, required=True)
    ap.add_argument('--unique', type=Path, required=True); ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    features = json.loads(a.features.read_text()); phases = json.loads(a.phases.read_text()); phases = phases.get('rows', phases)
    uniq = json.loads(a.unique.read_text()); uniq = uniq.get('rows', uniq); bank = json.loads(a.bank.read_text())['rows']
    C = HERE / 'constants'; stream = json.loads((C / 'stream_constants.json').read_text())['constants']
    K = V2.make_constants(stream, json.loads((C / 'microbench_constants_v2.json').read_text()))
    out = build(features, phases, uniq, bank, K, json.loads((C / 'v3_constants.json').read_text()), json.loads((C / 'v3c_constants.json').read_text()), json.loads((C / 'v3e_constants.json').read_text()))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n'); print(json.dumps(dict(cells=len(out), supported=sum(v['primary_s'] is not None for v in out.values()))))


if __name__ == '__main__':
    main()
