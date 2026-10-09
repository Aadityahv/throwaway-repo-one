"""Score an unseen machine-learning kernel set (FRESH_SET=f: GELU, SwiGLU, RMSNorm, rotary embedding, FP32 matrix multiply; g: tensor-core matrix multiply; h: fused attention) against the frozen predictions. CPU only.
Inputs: predictions_set_<set>.json (frozen static-runtime evaluations), predictions_set_<set>_measured.json (frozen measured-runtime evaluations), the committed timing and the raw energy output.
Runtime: median absolute percentage error of the static runtime model per kernel (a cell without a prediction or without a correct timing counts as a failure). Energy: per kernel, ours (calibrator-only,
with the tensor term for the tensor set) against the AccelWattch-style refit, FlipFlop and Alavani, with static and with measured runtime, below cap (window power < 570 W) and all cells, plus the rows that
take no runtime input. Label: accepted application_energy_raw rows, energy per launch, no idle subtraction."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SR = HERE.parent; U = SR.parent
SET = os.environ.get('FRESH_SET', 'f'); F = SR / ('fresh_' + SET); PUB = U / 'evaluation_data/measured/kernel_sets'
NAME = {'gelu': 'GELU (tanh form)', 'swiglu': 'SwiGLU gate', 'rmsnorm': 'RMSNorm', 'rope': 'Rotary position embedding', 'sgemm': 'FP32 matrix multiply (register-tiled)', 'tcgemm': 'Tensor-core matrix multiply (bf16, fp32 accumulate)', 'attn': 'Fused attention (bf16 tensor cores, online softmax)'}
static = json.loads((PUB / ('predictions_set_%s.json' % SET)).read_text())['rows']; meas = json.loads((PUB / ('predictions_set_%s_measured.json' % SET)).read_text())['rows']
timing = {r['cell_id']: r for r in json.loads((F / ('timing_fresh_%s_result.json' % SET)).read_text())['cells']}
E, W = {}, {}
for r in csv.DictReader(open(F / ('energy_raw_%s/application_energy_raw.csv' % SET))):
    cid = 'blackwell/%s/%s/%s' % (r['parent_id'], r['regime'], r['candidate_id'])
    if cid in E: raise SystemExit('duplicate ' + cid)
    e = float(r['board_energy_j_per_launch']); t = float(r['counted_launch_interval_s']) / int(r['launch_count']); E[cid] = e; W[cid] = e / t
ids = sorted(static); fam = np.array([static[c]['family'] for c in ids]); below = np.array([W.get(c, 0) < 570 for c in ids]); pinned = np.array([W.get(c, 0) >= 599 for c in ids])
Ev = np.array([E.get(c, np.nan) for c in ids]); out = dict(set=SET, cells=len(ids), energy_measured=sum(c in E for c in ids), below_cap=int(below.sum()), cap_pinned=int(pinned.sum()), missing_energy=[c for c in ids if c not in E])
tm = {c: (timing[c]['per_launch_runtime_s'] if c in timing and timing[c].get('correct') else None) for c in ids}
rt = np.array([abs(static[c]['static_runtime_s'] / tm[c] - 1) * 100 if static[c]['static_runtime_s'] and tm[c] else np.inf for c in ids]); rts = np.array([(static[c]['static_runtime_s'] / tm[c] - 1) * 100 if static[c]['static_runtime_s'] and tm[c] else np.nan for c in ids])
out['runtime'] = {NAME[f]: dict(cells=int((fam == f).sum()), no_prediction=int(np.isinf(rt[fam == f]).sum()), median_ape_pct=float(np.median(rt[fam == f])), signed_median_pct=float(np.nanmedian(rts[fam == f]))) for f in dict.fromkeys(fam)}
out['runtime']['ALL'] = dict(cells=len(ids), no_prediction=int(np.isinf(rt).sum()), median_ape_pct=float(np.median(rt)), p90_ape_pct=float(np.percentile(rt, 90)), signed_median_pct=float(np.nanmedian(rts)))
M = [('Ours (calibrator only)', 'ours'), ('AccelWattch-style (refit)', 'component'), ('FlipFlop', 'flipflop'), ('Alavani', 'alavani')]
def ape(src, key): return np.array([abs(src[c]['energy_j'][key] / E[c] - 1) * 100 if src[c]['energy_j'].get(key) and c in E else np.inf for c in ids])
out['energy'] = {}
for lab, src, suf in (('static runtime', static, 'predicted'), ('measured runtime', meas, 'measured')):
    t = {}
    for name, m in M:
        a = ape(src, '%s_%s' % (m, suf)); d = {NAME[f]: float(np.median(a[fam == f])) for f in dict.fromkeys(fam)}
        d['ALL %d' % len(ids)] = float(np.median(a)); d['below cap (%d)' % below.sum()] = float(np.median(a[below])) if below.any() else None; d['p90 all'] = float(np.percentile(a, 90)); d['no prediction'] = int(np.isinf(a).sum()); t[name] = d
    out['energy'][lab] = t
nr = {}
for name, key in (('FlipFlop as published (own time model)', 'flipflop_static'), ('Delestrac per-level energies', 'delestrac'), ("O'Connor / Keckler constants", 'literature')):
    a = ape(static, key); nr[name] = dict(median_all=float(np.median(a)), median_below_cap=float(np.median(a[below])) if below.any() else None)
out['energy']['no runtime input'] = nr
# decision utility (a separate claim): choose the lower-energy of the two candidates of each (kernel, size) without executing either; compared with choosing the faster by the runtime model and with always taking c1
dec = {}
pairs = {}
for c in ids: pairs.setdefault((static[c]['family'], static[c]['regime']), {})[static[c]['candidate']] = c
def choose(score):
    res = []
    for (f, rg), d in sorted(pairs.items()):
        if set(d) != {'c1', 'c2'} or any(d[k] not in E for k in d): continue
        vals = {k: score(d[k]) for k in d}
        if any(v is None for v in vals.values()): res.append(dict(family=f, regime=rg, status='failure')); continue
        pick = min(vals, key=lambda k: (vals[k], k)); best = min(E[d['c1']], E[d['c2']]); res.append(dict(family=f, regime=rg, status='ok', picked=pick, regret_pct=(E[d[pick]] / best - 1) * 100, optimal=E[d[pick]] == best))
    ok = [r for r in res if r['status'] == 'ok']
    return dict(pairs=len(res), failures=len(res) - len(ok), top1_optimal=sum(r['optimal'] for r in ok), median_regret_pct=float(np.median([r['regret_pct'] for r in ok])) if ok else None, mean_regret_pct=float(np.mean([r['regret_pct'] for r in ok])) if ok else None, max_regret_pct=max([r['regret_pct'] for r in ok], default=None))
dec['ours (static runtime, calibrator only)'] = choose(lambda c: static[c]['energy_j'].get('ours_predicted'))
dec['faster by the static runtime model'] = choose(lambda c: static[c]['static_runtime_s'])
dec['always the first candidate'] = choose(lambda c: 0 if static[c]['candidate'] == 'c1' else 1)
dec['AccelWattch-style refit (static runtime)'] = choose(lambda c: static[c]['energy_j'].get('component_predicted'))
out['decision'] = dec
(HERE / ('score_set_%s.json' % SET)).write_text(json.dumps(out, indent=1))
print('set %s: %d cells, energy measured %d, below cap %d, pinned at cap %d' % (SET, len(ids), out['energy_measured'], out['below_cap'], out['cap_pinned']))
print('\ndecision utility (lower-energy candidate per kernel and size):'); [print('  %-44s pairs %d failures %d top-1 optimal %d median regret %s max %s' % (k, v['pairs'], v['failures'], v['top1_optimal'], 'n/a' if v['median_regret_pct'] is None else '%.2f%%' % v['median_regret_pct'], 'n/a' if v['max_regret_pct'] is None else '%.2f%%' % v['max_regret_pct'])) for k, v in dec.items()]
print('\nruntime (static runtime model), median absolute percentage error'); [print('  %-52s cells %2d  no prediction %2d  %6.1f%%  signed %+.1f%%' % (k, v['cells'], v['no_prediction'], v['median_ape_pct'], v['signed_median_pct'])) for k, v in out['runtime'].items()]
for lab, t in out['energy'].items():
    if lab == 'no runtime input': print('\nno runtime input:', json.dumps(t)); continue
    print('\nenergy, ' + lab); keys = list(next(iter(t.values())))
    print('  %-52s' % 'kernel' + ''.join('%-27s' % n for n in t))
    for k in keys: print('  %-52s' % k + ''.join('%-27s' % ('-' if t[n][k] is None else ('%.1f%%' % t[n][k] if k != 'no prediction' else '%d' % t[n][k])) for n in t))
