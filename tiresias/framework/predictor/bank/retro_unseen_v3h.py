"""RETROSPECTIVE (not prospective): the frozen bank-conflict model and the earlier models on the 32 exposed unseen-kernel cells.
The measured values were seen before this model existed; nothing here is tuned on them. Runtime error only."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent; UK = SR / 'unseen_kernels'; sys.path.insert(0, str(SR))
import predict_runtime_v2 as V2, predict_runtime_v3 as V3, predict_runtime_v3f as V3F, predict_runtime_v3h as V3H
def stats(v): v = np.asarray(v, float); return dict(n=int(v.size), median_pct=round(float(np.median(v)), 2), p90_pct=round(float(np.percentile(v, 90)), 2)) if v.size else None
F = UK / 'frozen'
features = json.loads((F / 'features_unseen.json').read_text()); phases = json.loads((F / 'phases_unseen.json').read_text())['rows']
uniq = json.loads((F / 'phases_unique_unseen.json').read_text()); uniq = uniq.get('rows', uniq); bank = json.loads((HERE / 'bank_conflicts_unseen.json').read_text())['rows']
C = SR / 'constants'; K = V2.make_constants(json.loads((C / 'stream_constants.json').read_text())['constants'], json.loads((C / 'microbench_constants_v2.json').read_text()))
v3, v3c, v3e = (json.loads((C / n).read_text()) for n in ('v3_constants.json', 'v3c_constants.json', 'v3e_constants.json'))
P = {'v3e': V3.build(features, phases, uniq, K, v3, v3c, True, v3e), 'v3f': V3F.build(features, phases, uniq, K, v3, v3c, v3e), 'v3h': V3H.build(features, phases, uniq, bank, K, v3, v3c, v3e)}
T = {c['cell_id']: c for c in json.loads((UK / 'measured/timing_unseen_result.json').read_text())['cells']}
rep = dict(label='RETROSPECTIVE: exposed cells, model frozen without tuning on them', models={})
for m, p in P.items():
    sup = [k for k in T if p[k].get('primary_s')]; err = {k: abs(p[k]['primary_s'] / T[k]['per_launch_runtime_s'] - 1) * 100 for k in sup}; sg = {k: (p[k]['primary_s'] / T[k]['per_launch_runtime_s'] - 1) * 100 for k in sup}
    fam = lambda f: stats([err[k] for k in sup if T[k]['family'] == f]); tier = lambda t: stats([err[k] for k in sup if T[k]['tier'] == t])
    rep['models'][m] = dict(supported=len(sup), error=stats(list(err.values())), signed_median_pct=round(float(np.median(list(sg.values()))), 2) if sg else None,
        per_family={f: fam(f) for f in ('matmul', 'bs', 'scan', 'conv')}, per_tier={t: tier(t) for t in ('L2', 'DRAM')},
        per_cell={k: dict(measured_us=round(T[k]['per_launch_runtime_s'] * 1e6, 3), predicted_us=round(p[k]['primary_s'] * 1e6, 3), signed_pct=round(sg[k], 1)) for k in sup},
        unsupported={k: p[k]['unsupported_reason'] for k in T if not p[k].get('primary_s')})
(HERE / 'RETRO_unseen_v3h.json').write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
for m, r in rep['models'].items(): print(m, r['supported'], r['error'], r['signed_median_pct'], r['per_family'], r['per_tier'])
