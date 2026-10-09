"""All exposed cells, exact runtime-matched diagnostic. No fitting on target labels."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
U=HERE.parents[1]
BASE=U/'evaluation_data/measured/kernel_sets'
RESET=ROOT/'tiresias/app_runners'
sys.path.insert(0,str(RESET))
import operator_work_time_diagnostic as CAL


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(rows,key):
    e=np.array([r[key]/r['actual_j']-1 for r in rows])*100
    return dict(n=len(rows),median_ape_pct=float(np.median(abs(e))),
                p90_ape_pct=float(np.percentile(abs(e),90)),signed_median_pct=float(np.median(e)))


def pieces(terms,t,model):
    rate=model['rates_pJ_per_unit']
    return dict(base=model['P0_w']*t,
                memory=(rate['bytes_L2']*terms['bytes_L2']+rate['bytes_DRAM']*terms['bytes_DRAM'])*1e-12,
                compute=(rate['nonmem_lane_instructions']*terms['nonmem_lane_instructions']+
                         rate['sfu_lane_instructions']*terms['sfu_lane_instructions'])*1e-12)


def clipped(part,t,cap):
    total=sum(part.values());pred=min(total,cap*t)
    return pred,total-pred


def main():
    out=HERE/'matched_diagnostic.json'
    if out.exists():raise SystemExit('REFUSED: diagnostic already retained')
    model=read(HERE.parent/'unseen_kernels/energy_model_frozen.json')
    recipe=read(BASE/'recipe.json');co=recipe['component_fit']['params']
    static=read(BASE/'predictions_static.json')['rows']
    runtime=read(BASE/'predictions_runtime.json')['rows']
    own={**read(HERE.parent/'fresh_e/energy_predictions_fresh_e.json')['cells'],
         **read(HERE.parent/'bank/energy_predictions_unseen_v3i.json')['cells']}
    cal=CAL.load_calibration();fit=[r for r in cal if r['excluded'] is None]
    if len(cal)!=32 or len(fit)!=22:raise ValueError('frozen calibration population changed')
    const=float(np.mean([r['power_w'] for r in fit]))
    raw=list(csv.DictReader((CAL.CAL_DIR/'raw_records.csv').open(newline='')))
    budget=dict(acquired_windows=32,used_windows=22,constant_power_w=const,
                sum_counted_launch_interval_s=sum(float(r['counted_launch_interval_s']) for r in raw),
                sum_precondition_cuda_s=sum(float(r['precondition_cuda_seconds']) for r in raw),
                note='Actual summed device intervals; acquisition wall time and compile/idle/transfer overhead not reconstructed. Existing data reused, zero new calibration windows.',
                exclusion_counts={reason:sum(r['excluded']==reason for r in cal) for reason in sorted({r['excluded'] for r in cal if r['excluded']})})
    rows=[];errors=[]
    for scope in ('prospective','retrospective'):
        score=read(BASE/f'score_{scope}.json')
        for cid,label in score['per_cell'].items():
            c=static[cid]['cell'];p=runtime[cid];term=own[cid]['terms']
            t=own[cid]['v3i_runtime_s'];tm=label['measured_short_runtime_s'];actual=label['measured_energy_j']
            if actual<=0 or t<=0 or tm<=0:raise ValueError('missing target value')
            B=c['logical_bytes_per_launch'];N=static[cid]['component']['instruction_proxy']
            a=pieces(term,t,model);b=dict(base=co['P_const']*t,
                    memory=co['e_B']*B+co['e_DRAM']*(B if c['tier']=='DRAM' else 0),compute=co['e_inst']*N)
            ea,ca=clipped(a,t,model['cap_w']);eb,cb=clipped(b,t,model['cap_w'])
            am=pieces(term,tm,model);bm=dict(b,base=co['P_const']*tm)
            eam,cam=clipped(am,tm,model['cap_w']);ebm,cbm=clipped(bm,tm,model['cap_w'])
            for k,v in [('ours_static',ea),('component_predicted',eb),('ours_measured',eam),('component_measured',ebm)]:
                errors.append(abs(v-p['energy_j'][k])/max(abs(v),1e-30))
            delta={k:a[k]-b[k] for k in a};delta['cap_relief_difference']=cb-ca
            if not np.isclose(sum(delta.values()),ea-eb,rtol=1e-10,atol=1e-12):raise ValueError('nonadditive model difference')
            rows.append(dict(cell_id=cid,scope=scope,family=label['family'],tier=label['tier'],regime=label['regime'],
                             candidate=label['candidate'],below_cap=label['measured_window_power_w']<.95*model['cap_w'],
                             actual_j=actual,predicted_runtime_s=t,measured_short_runtime_s=tm,
                             own_predicted_j=ea,component_predicted_j=eb,own_measured_j=eam,component_measured_j=ebm,
                             constant_predicted_j=min(const,model['cap_w'])*t,constant_measured_j=min(const,model['cap_w'])*tm,
                             own_parts=a,component_parts=b,own_cap_relief_j=ca,component_cap_relief_j=cb,
                             own_runtime_error_j=ea-eam,component_runtime_error_j=eb-ebm,
                             own_energy_side_residual_j=eam-actual,component_energy_side_residual_j=ebm-actual,
                             difference_j=ea-eb,difference_parts_j=delta,
                             lane_nonmem_count=term['nonmem_lane_instructions'],source_operation_proxy=N,
                             own_compute_to_component_compute=a['compute']/b['compute'] if b['compute'] else None))
    if len(rows)!=48 or max(errors)>1e-9:raise ValueError('frozen prediction reproduction failed')
    summaries={}
    for scope in ('prospective','retrospective'):
        for stratum in ('all','below_cap'):
            q=[r for r in rows if r['scope']==scope and (stratum=='all' or r['below_cap'])]
            summaries[scope+'/'+stratum]={k:stats(q,k) for k in ('own_predicted_j','component_predicted_j','constant_predicted_j',
                                                                 'own_measured_j','component_measured_j','constant_measured_j')}
    family={}
    for name in sorted({r['family'] for r in rows}):
        q=[r for r in rows if r['family']==name and r['below_cap']]
        d={k:stats(q,k) for k in ('own_predicted_j','component_predicted_j','constant_predicted_j','own_measured_j','component_measured_j')}
        d['difference_parts_median_pct_of_actual']={k:float(np.median([r['difference_parts_j'][k]/r['actual_j']*100 for r in q])) for k in rows[0]['difference_parts_j']}
        d['runtime_error_signed_median_pct']={k:float(np.median([r[k]/r['actual_j']*100 for r in q])) for k in ('own_runtime_error_j','component_runtime_error_j')}
        d['energy_side_signed_median_pct']={k:float(np.median([r[k]/r['actual_j']*100 for r in q])) for k in ('own_energy_side_residual_j','component_energy_side_residual_j')}
        d['own_compute_to_component_compute_median']=float(np.median([r['own_compute_to_component_compute'] for r in q]))
        d['cap_activated_cells']={k:sum(r[k]>0 for r in q) for k in ('own_cap_relief_j','component_cap_relief_j')}
        family[name]=d
    inputs=[Path(__file__),HERE/'DIAGNOSTIC_PROTOCOL.md',BASE/'recipe.json',BASE/'predictions_static.json',BASE/'predictions_runtime.json',
            BASE/'score_prospective.json',BASE/'score_retrospective.json',HERE.parent/'unseen_kernels/energy_model_frozen.json',
            HERE.parent/'fresh_e/energy_predictions_fresh_e.json',HERE.parent/'bank/energy_predictions_unseen_v3i.json',
            CAL.CAL_DIR/'raw_records.csv',CAL.CAL_DIR/'manifest.jsonl',RESET/'operator_work_time_diagnostic.py']
    result=dict(schema='runtime_matched_exposed_energy_diagnosis/1',all_targets_exposed=True,
                note='Exact prediction reproduction and additive per-cell error diagnosis, not a new validation. Same runtime/cap/coverage; different calibration and instruction conventions remain confounded.',
                max_relative_prediction_reproduction_error=max(errors),calibration_budget=budget,summaries=summaries,
                per_family_below_cap=family,per_cell=rows,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(summaries=summaries,calibration_budget=budget,per_family_below_cap=family),indent=2))


if __name__=='__main__':main()
