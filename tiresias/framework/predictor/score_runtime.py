"""Score static runtime predictions once against the frozen kill rule.

Model v1 has no parameter fitted on operator cells (all constants come from microbenchmarks), so every
operator cell is held out by construction; results are still broken down per source lineage and
operator as the plan requires. Energy uses the frozen runtime-plus-traffic model
E = min(P0*t + eps*bytes, cap*t), reproduced exactly from the development table.

Input: a predictions JSON {cell_id: {"primary_s": float|None, "max_s": float|None, "sum_s": float|None,
"unsupported_reason": str|None}} covering every Blackwell development cell.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
CELLS=REPO/'tiresias/framework/compile_evidence/development/development_cells.csv'
ROOFLINE=REPO/'tiresias/app_runners/iteration4_fit_score_blackwell.json'
KILL={'runtime_median_max_pct':15.0,'energy_margin_pp':5.0,'energy_p90_margin_pp':10.0}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cells():
    out={}
    for x in csv.DictReader(CELLS.open(newline='')):
        if x['gpu']!='blackwell':continue
        out[x['cell_id']]=dict(operator=x['operator_id'],lineage=x['source_file_group'],onchip=x['onchip_heavy']=='True',
            t=float(x['runtime_s']),B=float(x['actual_logical_bytes_per_launch']),p0=float(x['physics_p0_w']),
            eps=float(x['calibration_epsilon_pj_per_byte'])*1e-12,cap=float(x['cap_w']),E=float(x['energy_j']),frozen=float(x['physics_pred_energy_j']))
    for c in out.values():
        if abs(min(c['p0']*c['t']+c['eps']*c['B'],c['cap']*c['t'])/c['frozen']-1)>1e-12:raise SystemExit('frozen energy model not reproduced')
        c['below']=c['E']/c['t']<.95*c['cap']
    if len(out)!=132:raise SystemExit('Blackwell denominator must be 132')
    return out

def energy(c,t):return min(c['p0']*t+c['eps']*c['B'],c['cap']*t)

def stats(values):
    v=np.asarray(values,dtype=float)
    if v.size == 0: return None
    def percentile(q):
        # Infinite errors are explicit unsupported failures. Avoid inf-inf interpolation.
        ordered = sorted(v)
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        if not math.isfinite(ordered[hi]): return None
        return round(float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)), 2)
    return {'n':int(v.size),'median_pct':percentile(.5),'p90_pct':percentile(.9),
            'unsupported_failures':int(np.count_nonzero(~np.isfinite(v)))}

def score(pred_path):
    C=cells();P=json.loads(Path(pred_path).read_text())
    if set(P) - set(C): raise SystemExit('unexpected prediction cells')
    for row in P.values():
        for key in ['primary_s','max_s','sum_s']:
            val = row.get(key)
            if val is not None and (not math.isfinite(val) or val <= 0):
                raise SystemExit('predictions must be finite and positive or explicitly null')
    missing=set(C)-set(P)
    if missing:raise SystemExit('predictions must list every cell (use null + reason): %d missing'%len(missing))
    report={'inputs':{str(Path(p).resolve().relative_to(REPO)):sha(p) for p in [pred_path,CELLS,ROOFLINE,__file__]},'kill_rule':KILL,'variants':{}}
    for variant in ['primary_s','max_s','sum_s']:
        sup=[k for k in C if P[k].get(variant)]
        rt=[abs(P[k][variant]/C[k]['t']-1)*100 for k in sup]
        e_pred=[abs(energy(C[k],P[k][variant])/C[k]['E']-1)*100 for k in sup]
        e_meas=[abs(C[k]['frozen']/C[k]['E']-1)*100 for k in sup]
        def sel(f,vals):return [v for k,v in zip(sup,vals) if f(C[k])]
        below=lambda c:c['below'];onchip=lambda c:c['below'] and c['onchip']
        res={'supported_cells':len(sup),'unsupported_cells':len(C)-len(sup),
            'runtime_error':stats(rt),
            'energy_error_predicted_runtime':{'all':stats(e_pred),'below_cap':stats(sel(below,e_pred)),'below_cap_onchip':stats(sel(onchip,e_pred))},
            'energy_error_measured_runtime_same_cells':{'all':stats(e_meas),'below_cap':stats(sel(below,e_meas)),'below_cap_onchip':stats(sel(onchip,e_meas))},
            'signed_runtime_error_median_pct':round(float(np.median([(P[k][variant]/C[k]['t']-1)*100 for k in sup])),2) if sup else None,
            'per_operator':{},'per_lineage':{}}
        for key,field in [('per_operator','operator'),('per_lineage','lineage')]:
            for g in sorted({C[k][field] for k in C}):
                ks=[k for k in C if C[k][field]==g];s=[k for k in ks if P[k].get(variant)]
                res[key][g]={'cells':len(ks),'supported':len(s),
                    'runtime_median_pct':round(float(np.median([abs(P[k][variant]/C[k]['t']-1)*100 for k in s])),2) if s else None,
                    'signed_runtime_median_pct':round(float(np.median([(P[k][variant]/C[k]['t']-1)*100 for k in s])),2) if s else None,
                    'unsupported_reasons':sorted({str(P[k].get('unsupported_reason')) for k in ks if not P[k].get(variant)})}
        # Unsupported counted as failures: runtime error treated as infinite for the median.
        rt_fail=rt+[float('inf')]*(len(C)-len(sup))
        res['runtime_median_unsupported_as_failure_pct']=stats(rt_fail)['median_pct']
        res['runtime_unsupported_as_failure']=stats(rt_fail)
        below_missing=sum(C[k]['below'] for k in C if k not in sup)
        res['energy_below_cap_unsupported_as_failure']=stats(sel(below,e_pred)+[float('inf')]*below_missing)
        res['energy_measured_runtime_full_below_cap']=stats([abs(C[k]['frozen']/C[k]['E']-1)*100 for k in C if C[k]['below']])
        res['per_cell']={k:dict(runtime_error_pct=abs(P[k][variant]/C[k]['t']-1)*100,
            energy_error_pct=abs(energy(C[k],P[k][variant])/C[k]['E']-1)*100) if k in sup else
            dict(runtime_error_pct=None,energy_error_pct=None,unsupported_reason=P[k].get('unsupported_reason')) for k in C}
        report['variants'][variant]=res
    # Roofline comparison on shared cells, both against the development-table runtime.
    r=json.loads(ROOFLINE.read_text())['exploratory']['cells'];roof={}
    for x in r:
        k=f"blackwell/{x['parent_id']}/{x['regime']}/{x['candidate_id']}"
        if k in C:roof[k]=x['runtime_pred_us']*1e-6
    shared=[k for k in roof if P[k].get('primary_s')]
    report['roofline_comparison']={'shared_cells':len(shared),
        'roofline_runtime':stats([abs(roof[k]/C[k]['t']-1)*100 for k in shared]),
        'primary_runtime':stats([abs(P[k]['primary_s']/C[k]['t']-1)*100 for k in shared]),
        'roofline_energy_below_cap':stats([abs(energy(C[k],roof[k])/C[k]['E']-1)*100 for k in shared if C[k]['below']]),
        'primary_energy_below_cap':stats([abs(energy(C[k],P[k]['primary_s'])/C[k]['E']-1)*100 for k in shared if C[k]['below']])}
    v=report['variants']['primary_s'];rc=report['roofline_comparison']
    eb=v['energy_error_predicted_runtime']['below_cap'];mb=v['energy_error_measured_runtime_same_cells']['below_cap']
    checks={'runtime_median_le_15':bool(v['runtime_error'] and v['runtime_error']['median_pct']<=KILL['runtime_median_max_pct']),
        'runtime_beats_roofline_on_shared':bool(rc['primary_runtime'] and rc['roofline_runtime'] and rc['primary_runtime']['median_pct']<rc['roofline_runtime']['median_pct']),
        'energy_below_cap_within_5pp':bool(eb and mb and eb['median_pct']<=mb['median_pct']+KILL['energy_margin_pp']),
        'energy_p90_within_10pp':bool(eb and mb and eb['p90_pct']<=mb['p90_pct']+KILL['energy_p90_margin_pp'])}
    ef=v['energy_below_cap_unsupported_as_failure'];mf=v['energy_measured_runtime_full_below_cap']
    full_checks=dict(checks)
    full_checks.update(runtime_median_le_15=bool(v['runtime_unsupported_as_failure']['median_pct'] is not None and v['runtime_unsupported_as_failure']['median_pct']<=15),
        energy_below_cap_within_5pp=bool(ef and ef['median_pct'] is not None and ef['median_pct']<=mf['median_pct']+5),
        energy_p90_within_10pp=bool(ef and ef['p90_pct'] is not None and ef['p90_pct']<=mf['p90_pct']+10))
    report['verdict']={'checks_unsupported_as_failure':full_checks,'passes_kill_rule_unsupported_as_failure':all(full_checks.values()),'checks':checks,'passes_kill_rule_on_supported_cells':all(checks.values()),
        'note':'Unsupported cells are reported separately and also scored as failures (runtime_median_unsupported_as_failure_pct).'}
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rep=score(a.predictions);Path(a.out).write_text(json.dumps(rep,indent=1)+'\n')
    print(json.dumps({k:rep[k] for k in ['verdict','roofline_comparison']},indent=1))
