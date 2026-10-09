"""Score the current model and the candidate (v3j runtime model + traffic energy columns) on all 131 evaluation cells, CPU only.

    python3 evaluate.py            # writes cells_scored.csv, scores.json
Labels (measured runtime, energy, window power, set, kernel) come from tiresias/evaluation/results/eval_cells.csv, which is read only. Predictions are recomputed here from the stored
static tables, each set loaded exactly as its committed prediction script loads it; the current-model values are checked against eval_cells.csv before anything else is reported
(the run refuses to continue if they do not reproduce). Error definition as evaluation/results/build_eval_cells.py: |predicted / measured - 1| * 100; median over cells; 90th percentile with
numpy's default linear interpolation; "below cap" = window power under 570 W. A cell the model refuses counts as a failure (infinite error) and is listed.
Variants (ablation): current; A = candidate runtime only (energy columns unchanged); B = traffic energy columns only (current runtime, DRAM bytes as before = logical bytes);
A+B = the candidate."""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; SR = HERE.parent; PW = SR.parents[1]
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate')); sys.path.insert(0, str(HERE))
import predict as P  # noqa: E402
from cal import portable_predict as PP  # noqa: E402
from cal import traffic as T  # noqa: E402
import predict_runtime_v3j as J  # noqa: E402

BELOW = 570.0
RUN_DOC = SR / 'calibrate/runs/energy_repeat_20261002/run_full/calibration_sm_120_0e63baea.json'
TENSOR_DOC = SR / 'fresh_g/calibration_with_tensor.json'
SETS = [  # name, directory, features, phases, unique, bank, constants dir, energy document
    ('classic', SR / 'unseen_kernels/frozen', 'features_unseen.json', 'phases_unseen.json', 'phases_unique_unseen.json', SR / 'bank/bank_conflicts_unseen_wavefront.json', SR / 'constants', RUN_DOC),
    ('e', SR / 'fresh_e', 'features_fresh_e.json', 'phases_fresh_e.json', 'phases_unique_fresh_e.json', SR / 'bank/bank_conflicts_fresh_e_wavefront.json', SR / 'constants', RUN_DOC),
    ('d', SR / 'fresh_d', 'features_fresh_d.json', 'phases_fresh_d.json', 'phases_unique_fresh_d.json', SR / 'bank/bank_conflicts_fresh_d.json', SR / 'constants', RUN_DOC),
    ('f', SR / 'fresh_f', 'features_fresh_f.json', 'phases_fresh_f.json', 'phases_unique_fresh_f.json', SR / 'fresh_f/bank_conflicts_fresh_f.json', SR / 'constants', RUN_DOC),
    ('g', SR / 'fresh_g', 'features_fresh_g.json', 'phases_fresh_g.json', 'phases_unique_fresh_g.json', SR / 'fresh_g/bank_conflicts_fresh_g.json', SR / 'fresh_g/constants_tensor', TENSOR_DOC),
    ('h', SR / 'fresh_h', 'features_fresh_h.json', 'phases_fresh_h.json', 'phases_unique_fresh_h.json', SR / 'fresh_h/bank_conflicts_fresh_h.json', SR / 'fresh_g/constants_tensor', TENSOR_DOC),
]
g = lambda p: json.loads(Path(p).read_text())
DEV_IDS = set()   # filled by dev_check.py


def load_footprints(name):
    out = {}
    for p in sorted((HERE / 'footprints' / name).glob('*.json')):
        d = json.loads(p.read_text()); out[d['cell_id']] = d
    return out


def l2_capacity_bytes():
    feats = g(SR / 'fresh_g/features_fresh_g.json'); return int(feats['hardware_from_ground_truth']['l2_bytes'])   # parsed from HARDWARE_GROUND_TRUTH.md by the feature builders


DEV = ('dev', SR, 'features_blackwell.json', 'phases_blackwell.json', 'reuse/phases_unique_blackwell.json', SR / 'bank/bank_conflicts_dev.json', SR / 'constants', RUN_DOC)


def predict_all(sets=None):
    """-> {cell_id: dict(set, row, doc, cur, oj, new)}: cur = current model, oj = candidate with the runtime correction off, new = candidate."""
    L2 = l2_capacity_bytes(); res = {}
    for name, d, ff, pf, uf, bf, cd, docp in (sets or SETS):
        feats = g(d / ff); ph = g(d / pf); ph = ph.get('rows', ph); un = g(d / uf); un = un.get('rows', un); bk = g(bf)['rows']
        if name == 'dev': feats = dict(feats, rows=[r for r in feats['rows'] if r['cell_id'] in DEV_IDS])
        C = PP.load_constants(cd); C['smem'] = None; sm = feats['hardware_from_ground_truth']['sm_count']
        un_fp = J.attach_footprints(un, load_footprints(name), L2)
        cur = PP.predict(feats, ph, un, bk, C, sm)
        old_through_j = T.predict_candidate(feats, ph, un_fp, bk, C, sm, read_footprint=False)
        new = T.predict_candidate(feats, ph, un_fp, bk, C, sm, read_footprint=True)
        lw = T.predict_candidate(feats, ph, un_fp, bk, C, sm, read_footprint=True, l2_rule='launch')
        for r in feats['rows']:
            cid = r['cell_id']
            res[cid] = dict(set=name, row=r, doc=docp, cur=cur[cid], oj=old_through_j[cid], new=new[cid], lw=lw[cid])
    return res


def ape(p, m): return abs(p / m - 1) * 100 if p and p > 0 else float('inf')


def score_cells(labels, pred):
    docs = {}
    def doc(p):
        if p not in docs: docs[p] = P.load_calibration(p, allow_incomplete=True)
        return docs[p]
    rows = []
    for lab in labels:
        cid = lab['cell_id']; p = pred[cid]; rm = float(lab['runtime_measured_s']); em = float(lab['energy_measured_j']); d = doc(p['doc']); r = {'rows': [p['row']]}
        rt_cur, rt_new = p['cur'].get('primary_s'), p['new'].get('primary_s')
        out = dict(cell_id=cid, group=lab['group'], set=lab['set'], kernel=lab['kernel'], tier=lab['tier'], runtime_measured_s=rm, energy_measured_j=em, window_power_w=float(lab['window_power_w']),
                   rt_cur=rt_cur, rt_new=rt_new, new_reason=p['new'].get('unsupported_reason'), cur_reason=p['cur'].get('unsupported_reason'))
        # current model energy: current columns
        def en_cur(rt):
            x = P.energy_rows(d, r, {cid: rt} if rt else {})[cid]; return x['energy_j'] if x['status'] == 'ok' else None
        def en_traffic(rt, tr, dram_override=None):
            x = T.energy_rows_traffic(d, r, {cid: rt} if rt else {}, {cid: tr} if tr else {}, dram_bytes_override=dram_override)[cid]; return x['energy_j'] if x['status'] == 'ok' else None
        mem = p['row']['memory']; logical_dram = float(mem['logical_bytes_per_launch']) if mem['tier'] == 'DRAM' else 0.0
        tr_oj, tr_new = p['oj'].get('traffic'), p['new'].get('traffic')
        out.update(
            e_cur_static=en_cur(rt_cur), e_cur_measured=en_cur(rm),
            e_A_static=en_cur(rt_new), e_A_measured=en_cur(rm),
            e_B_static=en_traffic(rt_cur, tr_oj, {cid: logical_dram}) if tr_oj else None, e_B_measured=en_traffic(rm, tr_oj, {cid: logical_dram}) if tr_oj else None,
            e_new_static=en_traffic(rt_new, tr_new), e_new_measured=en_traffic(rm, tr_new),
            rt_L=p['lw'].get('primary_s'), e_L_static=en_traffic(p['lw'].get('primary_s'), p['lw'].get('traffic')) if p['lw'].get('traffic') else None, e_L_measured=en_traffic(rm, p['lw'].get('traffic')) if p['lw'].get('traffic') else None, L_reason=p['lw'].get('unsupported_reason'),
            l2_bytes_new=(tr_new['l2_read_bytes'] + tr_new['l2_write_bytes']) if tr_new else None, dram_bytes_new=(tr_new['dram_read_bytes'] + tr_new['dram_write_bytes']) if tr_new else None,
            dram_bytes_cur=(tr_oj['dram_read_bytes'] + tr_oj['dram_write_bytes']) if tr_oj else None, logical_bytes=float(mem['logical_bytes_per_launch']))
        rows.append(out)
    return rows


def check_reproduction(labels, rows):
    bad = 0
    for lab, r in zip(labels, rows):
        for key, col, mc in (('rt_cur', 'prev_runtime_s', None), ('e_cur_static', 'prev_e_static', None), ('e_cur_measured', 'prev_e_measured', None)):
            ref = float(lab[col]); v = r[key]
            if v is None or abs(v / ref - 1) > 1e-9: bad += 1; print('MISMATCH', r['cell_id'], key, v, ref)
    if bad: raise SystemExit('refusing: the shared-traffic model does not reproduce the prev_* columns of eval_cells.csv (%d mismatches)' % bad)
    print('the shared-traffic model reproduces the prev_* columns of eval_cells.csv exactly on all %d cells (runtime, energy static, energy measured)' % len(rows))


def stats(vals):
    a = np.array(vals, float); return dict(n=len(a), median=float(np.median(a)), p90=float(np.percentile(a, 90)), failures=int(np.isinf(a).sum()))


METRICS = [('runtime', lambda r, v: ape(r['rt_' + v], r['runtime_measured_s'])),
           ('energy_static', lambda r, v: ape(r['e_%s_static' % v], r['energy_measured_j'])),
           ('energy_measured', lambda r, v: ape(r['e_%s_measured' % v], r['energy_measured_j']))]
VARIANTS = {'cur': 'current', 'A': 'A', 'B': 'B', 'new': 'candidate'}


def metric_value(rows_r, metric, variant):
    if metric == 'runtime': return ape(rows_r['rt_' + {'new': 'new', 'A': 'new', 'L': 'L'}.get(variant, 'cur')], rows_r['runtime_measured_s'])
    suffix = 'static' if metric == 'energy_static' else 'measured'
    return ape(rows_r['e_%s_%s' % (variant, suffix)], rows_r['energy_measured_j'])


def aggregate(rows):
    sel = {'ALL 131': lambda r: True, 'Predicted before measurement (72)': lambda r: r['group'].startswith('Predicted'), 'Seen during model building (59)': lambda r: r['group'].startswith('Seen')}
    for s in dict.fromkeys(r['set'] for r in rows): sel['set: ' + s] = (lambda s: lambda r: r['set'] == s)(s)
    for k in dict.fromkeys(r['kernel'] for r in rows): sel['kernel: ' + k] = (lambda k: lambda r: r['kernel'] == k)(k)
    out = {}
    for name, f in sel.items():
        sub = [r for r in rows if f(r)]
        if not sub: continue
        below = [r for r in sub if r['window_power_w'] < BELOW]; d = {'cells': len(sub), 'below_cap_cells': len(below)}
        for m, _ in METRICS:
            for v in ('cur', 'A', 'B', 'new', 'L'):
                d['%s/%s' % (m, v)] = stats([metric_value(r, m, v) for r in sub])
                if m != 'runtime': d['%s_below_cap/%s' % (m, v)] = stats([metric_value(r, m, v) for r in below]) if below else None
        out[name] = d
    return out


def main():
    labels = list(csv.DictReader(open(PW / 'evaluation/results/eval_cells.csv')))
    pred = predict_all()
    rows = score_cells(labels, pred); check_reproduction(labels, rows)
    # the candidate with its correction switched off reproduces the current model exactly (test of the new code path)
    bad = [r['cell_id'] for r in rows if (pred[r['cell_id']]['oj'].get('primary_s') or 0) != (pred[r['cell_id']]['cur'].get('primary_s') or 0)]
    if bad: raise SystemExit('candidate with read_footprint=False differs from the current model: %s' % bad[:5])
    print('candidate with the correction disabled reproduces the current runtime exactly on all cells')
    with open(HERE / 'cells_scored.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = aggregate(rows); (HERE / 'scores.json').write_text(json.dumps(agg, indent=1, sort_keys=True) + '\n')
    refused = [(r['cell_id'], r['new_reason']) for r in rows if r['rt_new'] is None]
    print('cells refused by the candidate:', len(refused), refused[:5])
    for name in ('Predicted before measurement (72)', 'Seen during model building (59)', 'ALL 131'):
        d = agg[name]; print(name, {m: (round(d[m + '/cur']['median'], 1), round(d[m + '/new']['median'], 1)) for m, _ in METRICS})


if __name__ == '__main__':
    main()
