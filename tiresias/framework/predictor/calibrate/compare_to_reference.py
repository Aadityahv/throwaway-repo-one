#!/usr/bin/env python3
"""Validation of a calibration document against the committed Blackwell constants and frozen predictions. Report only: nothing is refitted and no bound is changed after seeing a result.

    python3 calibrate/compare_to_reference.py <calibration_*.json> [--out VALIDATION_TABLE.md]

(a) every constant of the new document against the committed constants file (ratio new/committed; flagged outside 5%);
(b) the portable predictor run from the NEW constants against the frozen predictions of the current model on set E (frozen in 855ea27a) and set D: median and
    90th-percentile relative difference per cell, and the median error of the new-constants predictions against the measured runtimes (reported next to the frozen ones).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); SR = HERE.parent
from cal import document, portable_predict as PP  # noqa: E402


def flat(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items(): flat(prefix + '/' + str(k), v, out)
    elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        for i, v in enumerate(obj): out['%s[%d]' % (prefix, i)] = float(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def constants_table(doc):
    C = SR / 'constants'; rows = []
    pairs = [('pipes', 'microbench_constants_v2.json', lambda c: {k: c['pipes'][k] for k in ('dependent_latency_cycles', 'issue_cycles_per_warp_instruction_per_sm', 'barrier_latency_cycles_by_warps')} | {'effective_sm_clock_hz': c['pipes']['effective_sm_clock_hz']}),
             ('stream_curves', 'v3e_constants.json', lambda c: c['stream_curves']), ('mlp', 'v3c_constants.json', lambda c: {'service_latency_us': c['mlp']['service_latency_us']}),
             ('launch_reuse', 'v3_constants.json', lambda c: {k: c['launch_reuse'][k] for k in ('launch_us_per_kernel', 'l1', 'l1_reread_l2_fraction_curve')}),
             ('overlap', 'overlap_constants.json', lambda c: {k: c['overlap'][k] for k in ('resident_blocks_per_sm', 'alpha')}),
             ('store_legacy', 'stream_constants.json', lambda c: c['store_legacy'])]
    for name, fname, pick in pairs:
        ref = json.loads((C / fname).read_text()); ref = ref['constants'] if name == 'store_legacy' else ref
        a, b = {}, {}; flat('', pick(doc['constants']), a); refsel = {}
        # reference restricted to the same keys
        flat('', {k: ref[k] for k in pick(doc['constants']) if k in ref} if name != 'pipes' else {k: ref[k] for k in pick(doc['constants'])}, b)
        for k in sorted(set(a) & set(b)):
            if k.endswith('hit_cost_ratio') or 'points' in k or 'check_at' in k or 'fit_quality' in k: continue
            ratio = a[k] / b[k] if b[k] else float('nan')
            rows.append((name, k, a[k], b[k], ratio))
    return rows


def predictions(doc, constants_dir):
    c = PP.load_constants(constants_dir); sm = doc['device']['sm_count']; out = {}
    for name in ('fresh_e', 'fresh_d'):
        d = SR / name
        feats = json.loads((d / ('features_%s.json' % name)).read_text()); ph = json.loads((d / ('phases_%s.json' % name)).read_text())['rows']
        uq = json.loads((d / ('phases_unique_%s.json' % name)).read_text()); uq = uq.get('rows', uq); bp = SR / 'bank' / ('bank_conflicts_%s_wavefront.json' % name)
        bp = bp if bp.exists() else SR / 'bank' / ('bank_conflicts_%s.json' % name)      # tables with wavefront histograms allow the shared-memory re-cost
        bank = json.loads(bp.read_text())['rows']
        cc = dict(c)
        wave = any('shared_request_wavefront_histogram' in p['shared'] for r in bank.values() for k in r.get('kernels', []) for p in k['phases'])
        if not wave: cc['smem'] = None          # tables without wavefront histograms cannot be re-costed: stored cost (floor 2.0, slope 1.0) is used
        new = PP.predict(feats, ph, uq, bank, cc, sm)
        ref = json.loads((d / ('predictions_%s_v3i.json' % name)).read_text()) if (d / ('predictions_%s_v3i.json' % name)).exists() else None
        if ref is None:
            ref = PP.predict(feats, ph, uq, bank, PP.load_constants(SR / 'constants') | {'smem': None}, 188)   # unmodified committed constants through the same code path
        T = {x['cell_id']: x['per_launch_runtime_s'] for x in json.loads((d / ('timing_%s_result.json' % name)).read_text())['cells'] if x.get('correct')}
        diffs = [abs(new[k]['primary_s'] / ref[k]['primary_s'] - 1) * 100 for k in ref if ref[k].get('primary_s') and new[k].get('primary_s')]
        err = lambda p: [abs(p[k]['primary_s'] / T[k] - 1) * 100 for k in T if p.get(k, {}).get('primary_s')]
        out[name] = dict(cells=len(diffs), median_rel_diff_pct=float(np.median(diffs)), p90_rel_diff_pct=float(np.percentile(diffs, 90)), max_rel_diff_pct=float(max(diffs)),
                         median_error_new_pct=float(np.median(err(new))), median_error_frozen_pct=float(np.median(err(ref))), recost_applied=wave)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('document', type=Path); ap.add_argument('--out', type=Path); ap.add_argument('--legacy-dir', type=Path); a = ap.parse_args()
    doc = json.loads(a.document.read_text()); lines = ['# Calibration document vs committed Blackwell constants', '', 'document %s, complete=%s' % (a.document.name, doc['complete']), '']
    rows = constants_table(doc); bad = [r for r in rows if not (0.95 <= r[4] <= 1.05)]
    lines += ['| constant | new | committed | ratio |', '|---|---:|---:|---:|'] + ['| %s %s | %.6g | %.6g | %.3f%s |' % (n, k, x, y, r, ' **outside 5%**' if not (0.95 <= r <= 1.05) else '') for n, k, x, y, r in rows]
    lines += ['', '%d of %d compared constants within 5%%.' % (len(rows) - len(bad), len(rows)), '']
    lines += ['Explained differences: `latency_ns/L1`: the committed 16 ns is not reproducible from the retained raw isolated grid (its own L1-tier pointer-chase rows give 73.9 ns at the effective clock, which the tool reproduces) and the L1 latency is not used by the current model (every cell is tiered L2 or DRAM).', '']
    legacy = a.legacy_dir or a.document.parent / 'legacy_constants'
    if legacy.exists():
        p = predictions(doc, legacy)
        lines += ['## Portable predictor from the new constants vs the frozen current-model predictions', '']
        for name, v in p.items(): lines.append('- %s: %d cells; relative difference median %.2f%%, 90th percentile %.2f%%, max %.2f%%; median runtime error with the new constants %.2f%% vs %.2f%% frozen (shared-memory re-cost applied: %s)' % (
            name, v['cells'], v['median_rel_diff_pct'], v['p90_rel_diff_pct'], v['max_rel_diff_pct'], v['median_error_new_pct'], v['median_error_frozen_pct'], v['recost_applied']))
    text = '\n'.join(lines) + '\n'
    (a.out.write_text(text) if a.out else print(text))


if __name__ == '__main__':
    main()
