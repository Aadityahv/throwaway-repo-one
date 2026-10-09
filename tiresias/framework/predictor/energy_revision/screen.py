"""Fixed CPU-only energy development screen; never reads application test labels."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
U = HERE.parents[1]
DATA = U / 'compile_evidence/development/development_cells.csv'
FEATURES = HERE.parent / 'features_blackwell.json'
ALPHAS = (0.0, 0.001, 0.01, 0.1)
FORMS = {
    'legacy': ('Original energy recipe refitted within folds', ('l2', 'dram', 'nonmem', 'sfu'), True),
    'traffic': ('Joint base power and tier traffic', ('time', 'l2', 'dram'), False),
    'arithmetic': ('Joint base power, traffic and arithmetic', ('time', 'l2', 'dram', 'arithmetic', 'sfu'), False),
    'useful': ('Joint base power, traffic, arithmetic and shared work', ('time', 'l2', 'dram', 'arithmetic', 'sfu', 'shared'), False),
    'pooled': ('Joint base power and pooled instruction costs', ('time', 'l2', 'dram', 'nonmem', 'sfu'), False),
    'lookup': ('Joint base power with lookup-plus-DRAM accounting', ('time', 'lookup', 'dram', 'nonmem', 'sfu'), False),
    'grouped': ('Joint base power and disjoint instruction groups', ('time', 'l2', 'dram', 'integer', 'floating', 'sfu', 'shared', 'overhead'), False),
}
CONFIGS = [(form, alpha) for form in FORMS for alpha in ALPHAS]
FLOATING = ('fp32_add', 'fp32_mul', 'fp32_fma', 'fp_other')
SHARED = ('shared_load', 'shared_store', 'shared_matrix_load', 'shuffle')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(work):
    f = work['families']
    n = {k: float(v['lane_instructions']) for k, v in f.items()}
    if any(not np.isfinite(v) or v < 0 for v in n.values()):
        raise ValueError('negative/nonfinite instruction count')
    total = float(work['total_lane_instructions'])
    if abs(sum(n.values()) - total) > max(1e-6, total * 1e-12):
        raise ValueError('family census is not complete')
    nonmem = total - n.get('global_load', 0) - n.get('global_store', 0)
    integer = n.get('integer_alu', 0)
    floating = sum(n.get(k, 0) for k in FLOATING)
    sfu = n.get('special_function', 0)
    shared = sum(n.get(k, 0) for k in SHARED)
    overhead = nonmem - integer - floating - sfu - shared
    if overhead < -1e-6:
        raise ValueError('disjoint group sum exceeds non-global count')
    return dict(nonmem=nonmem, integer=integer, floating=floating,
                arithmetic=integer+floating, sfu=sfu, shared=shared,
                overhead=max(0, overhead))


def feature_values(work, logical_bytes, tier, runtime_s):
    if tier not in ('L2', 'DRAM'):
        raise ValueError('this calibration has no L1 energy rate')
    if not np.isfinite(runtime_s) or runtime_s <= 0 or logical_bytes < 0:
        raise ValueError('invalid workload/runtime')
    return dict(time=runtime_s, l2=logical_bytes if tier == 'L2' else 0,
                dram=logical_bytes if tier == 'DRAM' else 0,
                lookup=logical_bytes, **counts(work))


def load():
    feats = {r['cell_id']: r for r in json.loads(FEATURES.read_text())['rows']}
    out = []
    with DATA.open(newline='') as handle:
        input_rows=list(csv.DictReader(handle))
    for r in input_rows:
        if r['gpu'] != 'blackwell':
            continue
        f = feats.get(r['cell_id'])
        supported = f is not None and f['status'] in ('supported', 'supported_with_assumptions')
        t, energy, cap = map(float, (r['runtime_s'], r['energy_j'], r['cap_w']))
        if not all(np.isfinite(v) and v > 0 for v in (t, energy, cap)):
            raise ValueError('invalid measured development input')
        out.append(dict(cell_id=r['cell_id'], group=r['operator_group'],
                        description=r['operator_description'], family=r['family'],
                        tier=r['tier'], regime=r['regime'], candidate=r['candidate_id'],
                        t=t, energy=energy, cap=cap, p0=round(float(r['physics_p0_w']),9),
                        below=r['below_cap'] == 'True', supported=supported,
                        exact=f['work']['all_counts_exact'] if supported else False,
                        values=feature_values(f['work'], float(r['actual_logical_bytes_per_launch']), r['tier'], t) if supported else None))
    if len(out) != 132 or sum(r['supported'] for r in out) != 120:
        raise ValueError('declared full grid changed')
    if len({r['group'] for r in out if r['supported']}) != 9:
        raise ValueError('supported source-lineage grid changed')
    if len({r['p0'] for r in out}) != 1 or len({r['cap'] for r in out}) != 1:
        raise ValueError('calibration base power/cap changed within board')
    return sorted(out, key=lambda r:r['cell_id'])


def matrix(rows, form):
    features = FORMS[form][1]
    return np.array([[r['values'][k] * (1 if k == 'time' else 1e-12)
                      for k in features] for r in rows])


def fit(rows, config):
    form, alpha = config
    if not rows or not all(r['supported'] and r['below'] for r in rows):
        raise ValueError('fit expects supported below-cap training rows only')
    X = matrix(rows, form)
    e = np.array([r['energy'] for r in rows])
    fixed = FORMS[form][2]
    target = (e-np.array([r['p0']*r['t'] for r in rows]) if fixed else e)/e
    weighted = X/e[:, None]
    norm = np.linalg.norm(weighted, axis=0)
    zero = norm == 0
    norm[zero] = 1
    Z = weighted/norm
    penalty = np.array([0 if k == 'time' else 1 for k in FORMS[form][1]])
    augmented = np.vstack((Z, np.sqrt(alpha)*np.diag(penalty)))
    coeff, _ = nnls(augmented, np.r_[target, np.zeros(len(norm))])
    coeff = coeff/norm
    coeff[zero] = 0
    rank = int(np.linalg.matrix_rank(Z))
    condition = float(np.linalg.cond(Z))
    return dict(form=form, name=FORMS[form][0], alpha=alpha,
                features=list(FORMS[form][1]), coefficients=coeff.tolist(),
                fixed_base_power_w=rows[0]['p0'] if fixed else 0,
                n_train=len(rows), rank=rank, columns=len(norm),
                condition_number=condition if np.isfinite(condition) else None,
                zero_training_features=[k for k,z in zip(FORMS[form][1],zero) if z])


def predict(rows, model):
    if not all(r['supported'] for r in rows):
        raise ValueError('unsupported counts must refuse inference')
    return np.minimum(matrix(rows, model['form'])@np.array(model['coefficients']) +
                      model['fixed_base_power_w']*np.array([r['t'] for r in rows]),
                      np.array([r['cap']*r['t'] for r in rows]))


def choose(rows, group_key='group'):
    groups = sorted({r[group_key] for r in rows})
    losses = []
    for config in CONFIGS:
        group_losses = []
        for held in groups:
            train = [r for r in rows if r[group_key] != held and r['below']]
            test = [r for r in rows if r[group_key] == held and r['below']]
            if not test:
                continue
            p = predict(test, fit(train, config))
            group_losses.append(float(np.mean(np.abs(p/np.array([r['energy'] for r in test])-1)))*100)
        if not group_losses:
            raise ValueError('no eligible inner validation lineage')
        losses.append(dict(form=config[0], alpha=config[1], mean_lineage_ape_pct=float(np.mean(group_losses))))
    best = min(x['mean_lineage_ape_pct'] for x in losses)
    tied = [x for x in losses if x['mean_lineage_ape_pct'] <= best+1e-9]
    selected = min(tied, key=lambda x:(len(FORMS[x['form']][1]), -x['alpha'], list(FORMS).index(x['form'])))
    return (selected['form'], selected['alpha']), losses


def summarize(rows, preds):
    valid = [r for r in rows if r['cell_id'] in preds]
    errors = np.array([abs(preds[r['cell_id']]/r['energy']-1)*100 for r in valid])
    signed = np.array([(preds[r['cell_id']]/r['energy']-1)*100 for r in valid])
    failures = len(rows)-len(valid)
    def quant(a, q):
        # Nearest-rank percentile makes any failure at that quantile explicitly infinite.
        if not len(a): return None
        v=sorted(a)[max(0, int(np.ceil(q*len(a)))-1)]
        return float(v) if np.isfinite(v) else 'failure'
    return dict(declared=len(rows), supported=len(valid), failures=failures,
                median_ape_pct=float(np.median(errors)) if len(errors) else None,
                p90_ape_pct=float(np.percentile(errors,90)) if len(errors) else None,
                signed_median_pct=float(np.median(signed)) if len(signed) else None,
                failure_aware_median_pct=quant(list(errors)+[float('inf')]*failures,.5),
                failure_aware_p90_pct=quant(list(errors)+[float('inf')]*failures,.9))


def stratified(rows, preds):
    strata = {'all':rows, 'below_cap':[r for r in rows if r['below']],
              'exact_counts':[r for r in rows if r['supported'] and r['exact']],
              'upper_bound_counts':[r for r in rows if r['supported'] and not r['exact']]}
    for key in ('family','tier','group'):
        for value in sorted({r[key] for r in rows}):
            strata[key+':'+value] = [r for r in rows if r[key] == value]
            strata['below_cap_'+key+':'+value] = [r for r in rows if r[key] == value and r['below']]
    return {k:summarize(v,preds) for k,v in strata.items()}


def bootstrap(rows, reference, selected):
    rows = [r for r in rows if r['below'] and r['supported']]
    groups = sorted({r['group'] for r in rows})
    err = {g:([abs(reference[r['cell_id']]/r['energy']-1)*100 for r in rows if r['group']==g],
              [abs(selected[r['cell_id']]/r['energy']-1)*100 for r in rows if r['group']==g]) for g in groups}
    rng = np.random.default_rng(20261002)
    deltas=[]
    for _ in range(2000):
        drawn=rng.choice(groups,len(groups),replace=True)
        deltas.append(float(np.median([v for g in drawn for v in err[g][0]])-
                            np.median([v for g in drawn for v in err[g][1]])))
    return dict(independent_source_lineages=len(groups), draws=2000,
                scope='supported below-cap development cells; exploratory',
                reference_minus_selected_median_points=float(np.median([v for g in groups for v in err[g][0]])-
                                                             np.median([v for g in groups for v in err[g][1]])),
                interval_95_points=np.percentile(deltas,[2.5,97.5]).tolist())


def verify_freeze():
    doc=json.loads((HERE/'SCREEN_FREEZE.json').read_text())
    for name,digest in doc['files_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('local screening freeze changed: '+name)
        remote=subprocess.check_output(['git','show','origin/cross-gpu-prep:'+name],cwd=ROOT)
        if hashlib.sha256(remote).hexdigest() != digest:
            raise ValueError('screening freeze must be pushed first: '+name)
    return doc


def main():
    if (HERE/'SCREEN_SUPERSEDED.json').exists():
        raise SystemExit('SUPERSEDED: do not execute the earlier broad screen. See DIAGNOSTIC_PROTOCOL.md and RECOMMENDATION.md; the user requires mechanism-led, runtime-matched development.')
    output=HERE/'development_result.json'
    if output.exists():
        raise SystemExit('REFUSED: fixed development screen already completed')
    freeze=verify_freeze()
    rows=load();supported=[r for r in rows if r['supported']]
    outer=[];grid={f'{f}/{a}':{} for f,a in CONFIGS};selected={};reference={}
    for held in sorted({r['group'] for r in supported}):
        train=[r for r in supported if r['group']!=held]
        test=[r for r in supported if r['group']==held]
        config,losses=choose(train)
        model=fit([r for r in train if r['below']],config)
        for r,p in zip(test,predict(test,model)):selected[r['cell_id']]=float(p)
        for f,a in CONFIGS:
            m=fit([r for r in train if r['below']],(f,a))
            for r,p in zip(test,predict(test,m)):grid[f'{f}/{a}'][r['cell_id']]=float(p)
        reference.update({r['cell_id']:grid['legacy/0.0'][r['cell_id']] for r in test})
        extrap={r['cell_id']:[k for k in model['zero_training_features'] if r['values'][k]>0] for r in test}
        outer.append(dict(held_out_source_lineage=held, test_cells=[r['cell_id'] for r in test],
                          training_cells=[r['cell_id'] for r in train if r['below']], selected=model,
                          all_inner_losses=losses, unseen_activity={k:v for k,v in extrap.items() if v}))
    semantic={};semantic_folds=[]
    for held in sorted({r['family'] for r in supported}):
        train=[r for r in supported if r['family']!=held]
        test=[r for r in supported if r['family']==held]
        config,losses=choose(train,'family')
        m=fit([r for r in train if r['below']],config)
        for r,p in zip(test,predict(test,m)):semantic[r['cell_id']]=float(p)
        semantic_folds.append(dict(held_out_semantic_family=held,selected=m,all_inner_losses=losses))
    stats=stratified(rows,selected);refstats=stratified(rows,reference)
    final_config,final_losses=choose(supported)
    final_model=fit([r for r in supported if r['below']],final_config)
    median=stats['below_cap']['median_ape_pct'];old=refstats['below_cap']['median_ape_pct']
    family_deltas={f:stats['below_cap_family:'+f]['median_ape_pct']-refstats['below_cap_family:'+f]['median_ape_pct']
                   for f in sorted({r['family'] for r in supported})}
    admission=dict(median_relative_improvement_at_least_10pct=median<=.9*old,
                   p90_degradation_at_most_2points=stats['below_cap']['p90_ape_pct']<=refstats['below_cap']['p90_ape_pct']+2,
                   no_family_median_degradation_above_5points=all(v<=5 for v in family_deltas.values()))
    result=dict(schema='energy_revision_development/1',note='Development only; energy-window runtime, not prospective or static-runtime evidence',
                screening_freeze=freeze,declared_cells=len(rows),supported_cells=len(supported),
                exact_count_cells=sum(r['exact'] for r in supported),source_lineages=9,semantic_families=6,
                nested_selected_summary=stats,reference_refitted_summary=refstats,
                all_config_outer_summaries={k:stratified(rows,v) for k,v in grid.items()},
                nested_predictions=selected,reference_predictions=reference,all_config_outer_predictions=grid,
                outer_folds=outer,semantic_holdout_summary=stratified(rows,semantic),semantic_folds=semantic_folds,
                paired_development_bootstrap=bootstrap(rows,reference,selected),
                family_median_delta_points=family_deltas,admission_checks=admission,
                admitted_for_new_validation=all(admission.values()),final_selected_model=final_model,
                final_all_inner_losses=final_losses,
                selected_configuration_frequency=dict(Counter(FORMS[x['selected']['form']][0]+'; penalty '+str(x['selected']['alpha']) for x in outer)))
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    proposed=dict(schema='energy_revision_proposed_model/1',status='candidate for fresh validation' if all(admission.values()) else 'not admitted; development gate failed',
                  model=final_model,cap_w=rows[0]['cap'],input_contract='supported static family census + logical bytes + declared L2/DRAM tier + external runtime; no target execution required by energy formula',
                  training_input_hashes=freeze['files_sha256'],units='time coefficient W; byte/instruction coefficients pJ/unit',
                  limitations='120/132 development support; inexact counts disclosed; coefficients trained using measured energy-window runtime; no fresh validation or static-runtime claim')
    (HERE/'proposed_model.json').write_text(json.dumps(proposed,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(reference_below_cap=refstats['below_cap'],selected_below_cap=stats['below_cap'],
                          admission=admission,final_form=final_model['name'],final_penalty=final_model['alpha'],
                          paired_development_bootstrap=result['paired_development_bootstrap']),indent=2))


if __name__=='__main__':main()
