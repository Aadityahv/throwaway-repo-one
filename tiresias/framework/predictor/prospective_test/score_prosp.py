"""Score the 24 frozen prospective cells against the measured runtime and board energy (raw in raw/) and apply the pre-registered criteria (DESIGN.md addendum). CPU only.
current = frozen `current` fields, proposed = frozen `proposed` fields (v3k). Energy with measured runtime is computed here from the measured runtime with the frozen traffic of each model.
python3 score_prosp.py > SCORES.md   (writes scores_prosp_cells.json)"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import prosp_common as C
from cal import traffic as T
import predict as P
fz = json.loads((HERE / 'PREDICTIONS_FROZEN.json').read_text())['rows']
tj = json.loads((HERE / 'raw/timing_prosp.json').read_text()); rt = {c['cell_id']: c['per_launch_runtime_s'] for c in tj['cells']}; ok = {c['cell_id']: c.get('correct') for c in tj['cells']}
E_, W_ = {}, {}
for d in sorted((HERE / 'raw').glob('energy*/application_energy_raw.csv')):
    for r in csv.DictReader(open(d)):
        cid = 'blackwell/%s/%s/%s' % (r['parent_id'], r['regime'], r['candidate_id']); e = float(r['board_energy_j_per_launch']); E_[cid] = e; W_[cid] = e / (float(r['counted_launch_interval_s']) / int(r['launch_count']))
s = next(C.iter_sets(('prosp',))); feats = {r['cell_id']: r for r in s['feats']['rows']}; doc = P.load_calibration(s['docp'], allow_incomplete=True)
ape = lambda p, m: abs(p / m - 1) * 100 if p else float('inf')
def meas_energy(cid, m, tr):
    if not tr: return None
    x = T.energy_rows_traffic(doc, {'rows': [feats[cid]]}, {cid: m}, {cid: dict(l2_read_bytes=tr['l2_read'], l2_write_bytes=tr['l2_write'], dram_read_bytes=tr['dram_read'], dram_write_bytes=tr['dram_write'])})[cid]
    return x['energy_j'] if x['status'] == 'ok' else None
rows = []
for cid, r in sorted(fz.items()):
    if r['status'] != 'ok':   # not derivable by the static pipeline: failure for both models
        rows.append(dict(cell_id=cid, kernel=cid.split('/')[1], family='?', tier='?', refused_static=r['reason'], rt_meas=rt.get(cid), E=E_.get(cid), power=W_.get(cid), rt_cur=None, rt_pro=None, e_static_cur=None, e_static_pro=None, e_meas_cur=None, e_meas_pro=None, correct=ok.get(cid))); continue
    if cid not in rt: rows.append(dict(cell_id=cid, missing=True)); continue
    m = rt[cid]; E = E_.get(cid)
    rows.append(dict(cell_id=cid, kernel=r['kernel_group'], family=r['family'], tier=r['tier'], regime=r['regime'], cand=r['candidate'], power=W_.get(cid), rt_meas=m, E=E, correct=ok[cid],
                     rt_cur=r['current']['runtime_s'], rt_pro=r['proposed']['runtime_s'], e_static_cur=r['current']['energy_static_j'], e_static_pro=r['proposed']['energy_static_j'],
                     e_meas_cur=meas_energy(cid, m, r['current']['traffic_bytes']), e_meas_pro=meas_energy(cid, m, r['proposed']['traffic_bytes']), refusal_cur=r['current']['refusal'], refusal_pro=r['proposed']['refusal']))
good = [r for r in rows if not r.get('missing')]
f = lambda x: 'fail' if x == float('inf') else '%.1f' % x
print('# Prospective test scores (24 frozen cells, Blackwell GPU 1)\n\nAbsolute percentage error, current model | proposed model. "fail" = refused or unsupported (counts as a failure in every median). Power from the 15 s window.\n')
print('| Cell | tier | power W | measured us | runtime err | energy static err | energy measured-runtime err |\n|---|---|---:|---:|---|---|---|')
for r in good:
    E = r['E']
    en = lambda a, b, m: ' / '.join([f(ape(a, E)) if E else 'n/a', f(ape(b, E)) if E else 'n/a'])
    print('| %s | %s | %s | %.1f | %s / %s | %s | %s |' % (r['cell_id'][10:], r['tier'], '%.0f' % r['power'] if r['power'] else 'n/a', (r['rt_meas'] or 0) * 1e6, f(ape(r['rt_cur'], r['rt_meas'])), f(ape(r['rt_pro'], r['rt_meas'])), en(r['e_static_cur'], r['e_static_pro'], 0), en(r['e_meas_cur'], r['e_meas_pro'], 0)))
FAM = {'tcgemm2': 'tensor-core matrix multiply', 'tcrelu': 'tensor-core matrix multiply', 'attn2': 'attention', 'attnnm': 'attention'}
groups = {}
for r in good:
    groups.setdefault(r['kernel'], []).append(r); groups.setdefault('FAMILY: ' + FAM.get(r['family'], '?'), []).append(r); groups.setdefault('ALL 24', []).append(r); groups.setdefault('TIER: ' + r['tier'], []).append(r)
print('\n| Group | cells | runtime median cur / pro | runtime p90 cur / pro | energy static median cur / pro | energy measured-runtime median cur / pro |\n|---|---:|---|---|---|---|')
summ = {}
for k, g in sorted(groups.items()):
    def st(a, b, m, fn):
        x = [ape(r[a], r[m]) for r in g if r.get(m)]; y = [ape(r[b], r[m]) for r in g if r.get(m)]
        if not x: return 'n/a'
        return '%s / %s' % (f(fn(x)), f(fn(y)))
    med, p90 = np.median, lambda v: np.percentile(v, 90)
    summ[k] = dict(n=len(g), rt_med=[float(np.median([ape(r['rt_cur'], r['rt_meas']) for r in g])), float(np.median([ape(r['rt_pro'], r['rt_meas']) for r in g]))])
    print('| %s | %d | %s | %s | %s | %s |' % (k, len(g), st('rt_cur', 'rt_pro', 'rt_meas', med), st('rt_cur', 'rt_pro', 'rt_meas', p90), st('e_static_cur', 'e_static_pro', 'E', med), st('e_meas_cur', 'e_meas_pro', 'E', med)))
print('\n## Criteria (fixed in DESIGN.md before any cell was built)')
c1 = {k: summ[k]['rt_med'][1] for k in ('ALL 24', 'FAMILY: tensor-core matrix multiply', 'FAMILY: attention')}
for k, v in c1.items(): print('- Criterion 1, %s: proposed median runtime error %.1f%% (limit 15%%): %s' % (k, v, 'PASS' if v <= 15 else 'FAIL'))
crit2 = json.loads((HERE / 'criterion2.json').read_text())
print('- Criterion 2, no kernel median of the 131 up by more than 2 points: largest increase %+.2f points: %s' % (crit2['worst_increase'], 'PASS' if crit2['criterion2_pass'] else 'FAIL'))
refused = [r['cell_id'][10:] for r in good if r['rt_pro'] is None]; refused_c = [r['cell_id'][10:] for r in good if r['rt_cur'] is None]
print('- Criterion 3, refusals reported: proposed model refused %d cell(s) %s; current model refused %d cell(s) %s' % (len(refused), refused, len(refused_c), refused_c))
wrong = [r['cell_id'] for r in good if not r.get('correct')]
print('- Output-correctness failures of the timing harness: %s' % (wrong or 'none'))
adopt = all(v <= 15 for v in c1.values()) and crit2['criterion2_pass']
print('\n**DECISION: %s**' % ('ADOPT the proposed model (all pre-registered criteria hold)' if adopt else 'DO NOT ADOPT (a pre-registered criterion fails)'))
print('\nCells whose runtime error is worse under the proposed model by more than 2 points:')
for r in good:
    a, b = ape(r['rt_cur'], r['rt_meas']), ape(r['rt_pro'], r['rt_meas'])
    if b > a + 2 and a != float('inf'): print('- %s: %s -> %s' % (r['cell_id'][10:], f(a), f(b)))
json.dump(dict(rows=rows, summary=summ, criterion1=c1, criterion2=crit2, adopt=adopt), open(HERE / 'scores_prosp_cells.json', 'w'), indent=1, default=str)
