#!/usr/bin/env python3
"""Retrospective Blackwell scoring of the CURRENT runtime model (and the calibrator-only energy model fed with static runtime) under different traffic-rate (`store_legacy`) derivations.

CPU only; Blackwell data only. Uses exactly the code path and inputs of blackwell_experiments/runtime_table.py and energy_table.py (cal/portable_predict.py with the committed
constants, no shared-memory re-cost, 188 SMs; the committed timings; the energy profile of the repeat calibration run), changing ONLY the six traffic-rate keys of the store constants.
Variants: `committed` = the committed constants/stream_constants.json (reproduces the headline numbers); `legacy` = this tool's legacy derivation on the Blackwell calibration document;
`A` = stream_rates; `B` = guarded. Every set with committed runtime timing and committed frozen predictions is scored (F, G, H are the machine-learning kernel sets).

    python3 calibrate/tools/score_rate_variants.py [--source-run energy_repeat_20261002/run_full] [--out result.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SR = HERE.parent; U = SR.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'tools')); sys.path.insert(0, str(SR / 'blackwell_experiments'))
from cal import fits, portable_predict as PP  # noqa: E402
import predict as P  # noqa: E402
import rederive_store as R  # noqa: E402

RATE_KEYS = ('t0_us', 'L2_read_sector_TBps', 'L2_write_sector_TBps', 'c_line_ns', 'DRAM_read_TBps', 'DRAM_write_TBps')
SM = 188
g = lambda *x: json.loads(SR.joinpath(*x).read_text())
rows_of = lambda d: d.get('rows', d)


def load_set(d, f, p, u, b):
    return g(d, f), rows_of(g(d, p)), rows_of(g(d, u)), g(*b)['rows']


SETS = [  # name, loader args, timing path, base constants dir
    ('Unseen kernels (4 kernels)', ('unseen_kernels/frozen', 'features_unseen.json', 'phases_unseen.json', 'phases_unique_unseen.json', ('bank', 'bank_conflicts_unseen_wavefront.json')), 'unseen_kernels/measured/timing_unseen_result.json', 'constants'),
    ('CUDA-samples kernels at new shapes (set D)', ('fresh_d', 'features_fresh_d.json', 'phases_fresh_d.json', 'phases_unique_fresh_d.json', ('bank', 'bank_conflicts_fresh_d.json')), 'fresh_d/timing_fresh_d_result.json', 'constants'),
    ('Scalar product and Walsh transform (set E, prospective)', ('fresh_e', 'features_fresh_e.json', 'phases_fresh_e.json', 'phases_unique_fresh_e.json', ('bank', 'bank_conflicts_fresh_e_wavefront.json')), 'fresh_e/timing_fresh_e_result.json', 'constants'),
    ('Machine-learning kernels (set F)', ('fresh_f', 'features_fresh_f.json', 'phases_fresh_f.json', 'phases_unique_fresh_f.json', ('fresh_f', 'bank_conflicts_fresh_f.json')), 'fresh_f/timing_fresh_f_result.json', 'constants'),
    ('Tensor-core matrix multiply (set G)', ('fresh_g', 'features_fresh_g.json', 'phases_fresh_g.json', 'phases_unique_fresh_g.json', ('fresh_g', 'bank_conflicts_fresh_g.json')), 'fresh_g/timing_fresh_g_result.json', 'fresh_g/constants_tensor'),
    ('Fused attention (set H)', ('fresh_h', 'features_fresh_h.json', 'phases_fresh_h.json', 'phases_unique_fresh_h.json', ('fresh_h', 'bank_conflicts_fresh_h.json')), 'fresh_h/timing_fresh_h_result.json', 'fresh_g/constants_tensor'),
]


def variants_from(run_dir, committed):
    """Store dicts per variant: committed constants, and the three derivations applied to the archived Blackwell raw store/stream stage outputs of `run_dir`."""
    doc_path = [x for x in sorted(Path(run_dir).glob('calibration_sm_*.json')) if 'application' not in x.name][0]
    out = {'committed': committed}; info = {}
    import tempfile
    for key, m in (('legacy', 'legacy'), ('A', 'stream_rates'), ('B', 'guarded')):
        p, new = R.rederive(doc_path, m, Path(tempfile.mkdtemp()) / 'x.json'); out[key] = {k: float(new['constants']['store_legacy'][k]) for k in RATE_KEYS}; info[key] = new['store_derivation']['method_used']
    return out, info, doc_path


def with_store(C, store):
    C = dict(C); s = dict(C['stream']); s.update({k: store[k] for k in RATE_KEYS}); C['stream'] = s; return C


def timing(path):
    out, bad = {}, []
    for r in g(path)['cells']:
        if r.get('correct', r.get('check_ok', True)) is False: bad.append(r['cell_id'])
        else: out[r['cell_id']] = r['per_launch_runtime_s']
    return out, bad


def stats(pred, T, ids):
    ape = np.array([abs(pred[c] / T[c] - 1) * 100 if pred.get(c) else np.inf for c in ids]); sg = np.array([(pred[c] / T[c] - 1) * 100 if pred.get(c) else np.nan for c in ids])
    return dict(cells=len(ids), no_prediction=int(np.isinf(ape).sum()), median_ape_pct=float(np.median(ape)), p90_ape_pct=float(np.percentile(ape, 90)), signed_median_pct=float(np.nanmedian(sg)))


def predict_runtime(feats, ph, uq, bank, C):
    out = PP.predict(feats, ph, uq, bank, C, SM)
    return {k: (v.get('primary_s') if isinstance(v, dict) else v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-run', default='energy_repeat_20261002/run_full', help='Blackwell calibration run (under calibrate/runs) whose raw store/stream outputs the variants are derived from')
    ap.add_argument('--out', type=Path); a = ap.parse_args(argv)
    committed_store = json.loads((SR / 'constants/stream_constants.json').read_text())['constants']
    variants, info, doc_path = variants_from(HERE / 'runs' / a.source_run, {k: float(committed_store[k]) for k in RATE_KEYS})
    res = dict(source_run=a.source_run, variants=variants, methods_used=info, runtime={}, energy={})
    pred_cache = {}
    for name, args, tpath, cdir in SETS:
        feats, ph, uq, bank = load_set(*args); T, bad = timing(tpath)
        base = PP.load_constants(SR / cdir); base['smem'] = None
        ids = [k for k in T if k in {r['cell_id'] for r in feats['rows']}]
        res['runtime'][name] = dict(excluded_failed_correctness=bad)
        for v, store in variants.items():
            pr = predict_runtime(feats, ph, uq, bank, with_store(base, store)); pred_cache[(name, v)] = pr
            res['runtime'][name][v] = stats(pr, T, ids)
    # energy: 48 evaluation cells (unseen kernels + set E), calibrator-only profile of the repeat run, static runtime from the variant
    import cpu_experiments as X  # noqa
    import pandas as pd
    edoc = P.load_calibration(HERE / 'runs/energy_repeat_20261002/run_full/calibration_sm_120_0e63baea.json', allow_incomplete=True)
    feats_all, tgt, rts_committed, _, _ = X.load_cells(); ids48 = list(tgt.cell_id); E = tgt.measured_energy_j.values; below = (tgt.measured_window_power_w < 570).values
    for v in variants:
        rt = {}
        for name in (SETS[0][0], SETS[2][0]): rt.update({k: x for k, x in pred_cache[(name, v)].items()})
        r = P.energy_rows(edoc, {'rows': [feats_all[c] for c in ids48]}, {c: t for c, t in rt.items() if t})
        pe = np.array([r[c]['energy_j'] if r[c]['status'] == 'ok' else np.inf for c in ids48]); err = np.abs(pe / E - 1) * 100
        res['energy'][v] = dict(cells=len(ids48), below_cap_cells=int(below.sum()), median_all_pct=float(np.median(err)), median_below_cap_pct=float(np.median(err[below])), p90_all_pct=float(np.percentile(err, 90)), unsupported=int(np.isinf(err).sum()))
    if a.out: a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    for name, d in res['runtime'].items():
        print(name, '(excluded: %s)' % d['excluded_failed_correctness'])
        for v in variants: s = d[v]; print('   %-9s cells %3d  nopred %d  median %6.2f%%  p90 %6.2f%%  signed %+6.2f%%' % (v, s['cells'], s['no_prediction'], s['median_ape_pct'], s['p90_ape_pct'], s['signed_median_pct']))
    print('energy (static runtime, calibrator-only profile, 48 cells)')
    for v, s in res['energy'].items(): print('   %-9s all %6.2f%%  below cap(%d) %6.2f%%  p90 %6.2f%%  unsupported %d' % (v, s['median_all_pct'], s['below_cap_cells'], s['median_below_cap_pct'], s['p90_all_pct'], s['unsupported']))
    print('variant store constants:', json.dumps(variants, indent=1)); print('methods used:', info)
    return 0


if __name__ == '__main__':
    sys.exit(main())
