"""Score frozen energy predictions; coverage failures never become full-grid wins."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

NAMES = {
    'source_proxy_component': 'Original source-operation component adaptation',
    'constant_power': 'Original calibration-mean constant power',
    'class_aware_component': 'Class-aware component revision',
    'budget_matched_pooled_component': 'Component adaptation with pooled activity and matched calibration',
    'budget_matched_constant_power': 'Constant power with matched calibration',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return dict(n=len(values), median_ape_pct=statistics.median(values) if values else None,
                mean_ape_pct=statistics.mean(values) if values else None)


def score(predictions, labels, cap_w):
    if predictions['schema'] != 'runtime_matched_static_energy/1' or labels['schema'] != 'energy_evaluation_labels/1':
        raise ValueError('REFUSED: unsupported prediction/label schema')
    if labels['exposure'] not in ('already_exposed_diagnostic', 'fresh_once'):
        raise ValueError('REFUSED: declare exposure explicitly')
    if not math.isfinite(cap_w) or cap_w <= 0:
        raise ValueError('REFUSED: invalid cap')
    pr = predictions['rows']; lr = labels['rows']
    pids = [r['cell_id'] for r in pr]; lids = [r['cell_id'] for r in lr]
    if not pids or len(set(pids)) != len(pids) or len(set(lids)) != len(lids) or set(pids) != set(lids):
        raise ValueError('REFUSED: complete identical target coverage required')
    label_by_id = {r['cell_id']: r for r in lr}
    methods = sorted(set().union(*(r['methods'] for r in pr)))
    if any(set(r['methods']) != set(methods) for r in pr):
        raise ValueError('REFUSED: missing method rows must be explicit unsupported entries')
    rows = []
    for p in pr:
        l = label_by_id[p['cell_id']]
        e, t = l['energy_j'], l['measured_runtime_s']
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in (e, t)):
            raise ValueError('REFUSED: invalid measured energy/runtime')
        row = dict(cell_id=p['cell_id'], family=l['family'], below_cap=e/t < .95*cap_w,
                   runtime_ape_pct=100*abs(p['predicted_runtime_s']/t-1), methods={})
        for m in methods:
            v = p['methods'][m]
            if v['status'] == 'unsupported':
                row['methods'][m] = dict(status='unsupported', reason=v['reason'])
            elif v['status'] == 'ok':
                ep = v['energy_j']
                if not math.isfinite(ep) or ep <= 0:
                    raise ValueError('REFUSED: invalid frozen energy prediction')
                row['methods'][m] = dict(status='ok', ape_pct=100*abs(ep/e-1), signed_error_pct=100*(ep/e-1))
            else:
                raise ValueError('REFUSED: unknown prediction status')
        rows.append(row)
    summaries = {}
    for m in methods:
        ok = [r for r in rows if r['methods'][m]['status'] == 'ok']
        below = [r for r in ok if r['below_cap']]
        summaries[m] = dict(name=NAMES.get(m,m), supported=len(ok), requested=len(rows),
                            full_grid_win_eligible=len(ok)==len(rows),
                            all_supported=summarize([r['methods'][m]['ape_pct'] for r in ok]),
                            below_cap_supported=summarize([r['methods'][m]['ape_pct'] for r in below]),
                            families={f:summarize([r['methods'][m]['ape_pct'] for r in below if r['family']==f])
                                      for f in sorted({r['family'] for r in rows})})
    return dict(schema='runtime_matched_energy_score/1', exposure=labels['exposure'], requested=len(rows),
                below_cap_requested=sum(r['below_cap'] for r in rows), cap_w=cap_w,
                runtime_vector_sha256=predictions['runtime_vector_sha256'], summaries=summaries, rows=rows,
                qualification='Supported-only accuracy cannot establish a full-grid win. No decision-energy savings or measurement-cost savings established by this score.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze=json.loads(a.freeze.read_text())
    # Validate the pushed numerical/scorer freeze BEFORE opening labels.
    if freeze.get('predictions_sha256')!=sha(a.predictions) or freeze.get('scorer_sha256')!=sha(Path(__file__)) or not freeze.get('pushed_commit'):
        raise ValueError('REFUSED: pushed numerical/scorer freeze absent or changed')
    predictions=json.loads(a.predictions.read_text());labels=json.loads(a.labels.read_text())
    result=score(predictions,labels,freeze['cap_w'])
    result['inputs_sha256']={str(f):sha(f) for f in (a.predictions,a.freeze,a.labels,Path(__file__))}
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('Scored full declared grid once; exposure and unsupported coverage retained')


if __name__=='__main__':main()
