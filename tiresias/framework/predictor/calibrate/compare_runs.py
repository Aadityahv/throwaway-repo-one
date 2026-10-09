#!/usr/bin/env python3
"""Run-to-run repeatability of two calibration documents of the same GPU (the acceptance test of the tool): every numeric constant of the two documents is compared and the
share within a bound (default 5%) is reported with the list of those outside.
    python3 calibrate/compare_runs.py <doc_a.json> <doc_b.json> [--bound 0.05]"""
import argparse
import json
import sys
from pathlib import Path

SKIP = ('points', 'check_at', 'fit_quality', 'overhead_points', 'composition', 'linear_region', 'configs_per_point', 'read_fraction', 'per_sm_kb', 'resident_blocks', 'volatile_by_stride', 'hit_cost_ratio')


def flat(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items(): flat(prefix + '/' + str(k), v, out)
    elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        for i, v in enumerate(obj): out['%s[%d]' % (prefix, i)] = float(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def compare(a, b, bound=0.05):
    fa, fb = {}, {}; flat('', a['constants'], fa); flat('', b['constants'], fb); rows = []
    for k in sorted(set(fa) & set(fb)):
        if any(s in k for s in SKIP): continue
        if fa[k] == 0 and fb[k] == 0: continue
        rows.append((k, fa[k], fb[k], fb[k] / fa[k] if fa[k] else float('nan')))
    bad = [r for r in rows if not (1 - bound <= r[3] <= 1 + bound)]
    return rows, bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('a', type=Path); ap.add_argument('b', type=Path); ap.add_argument('--bound', type=float, default=0.05); x = ap.parse_args()
    A, B = json.loads(x.a.read_text()), json.loads(x.b.read_text())
    if A['device']['uuid'] != B['device']['uuid']: sys.exit('REFUSED: the two documents are from different devices')
    rows, bad = compare(A, B, x.bound)
    print('%d constants compared; %d within %.0f%%' % (len(rows), len(rows) - len(bad), 100 * x.bound))
    for k, p, q, r in bad: print('  OUTSIDE %-76s %.5g %.5g ratio %.3f' % (k[-76:], p, q, r))
    return 0 if not bad else 3


if __name__ == '__main__':
    sys.exit(main())
