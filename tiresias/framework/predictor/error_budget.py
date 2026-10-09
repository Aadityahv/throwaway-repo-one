"""How much runtime error the frozen runtime-plus-traffic energy model tolerates.

CPU only, existing exposed development cells (Blackwell and Ada primary; A100
diagnostic). The frozen model is E = min(P0*t + eps*bytes, cap*t), reproduced
exactly from the development table. Runtime t is replaced by t*(1+d) for a
fixed signed error d, and by t*exp(N(0,s)) for random log-normal errors whose
median absolute runtime error is reported. Measured runtime stays the reference.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
CELLS=HERE.parent/'compile_evidence/development/development_cells.csv'

def load():
    rows=list(csv.DictReader(CELLS.open(newline='')))
    f=lambda k,s=1.0:np.array([float(x[k])*s for x in rows])
    d=dict(gpu=np.array([x['gpu'] for x in rows]),onchip=np.array([x['onchip_heavy']=='True' for x in rows]),
        t=f('runtime_s'),B=f('actual_logical_bytes_per_launch'),p0=f('physics_p0_w'),eps=f('calibration_epsilon_pj_per_byte',1e-12),
        cap=f('cap_w'),E=f('energy_j'),frozen=f('physics_pred_energy_j'))
    model=np.minimum(d['p0']*d['t']+d['eps']*d['B'],d['cap']*d['t'])
    if np.max(np.abs(model/d['frozen']-1))>1e-12:raise SystemExit('frozen model not reproduced')
    d['below']=d['E']/d['t']<.95*d['cap'];return d

def err(d,t):return np.abs(np.minimum(d['p0']*t+d['eps']*d['B'],d['cap']*t)/d['E']-1)*100

def summary(d,mask,t):
    e=err(d,t)[mask];return {'cells':int(mask.sum()),'median_pct':round(float(np.median(e)),2),'p90_pct':round(float(np.percentile(e,90)),2)}

def main():
    d=load();rng=np.random.default_rng(20261001);out={'model':'min(P0*t + eps*bytes, cap*t), frozen','groups':{}}
    for gpu in ['blackwell','ada','a100']:
        g=d['gpu']==gpu
        groups={'below_cap':g&d['below'],'below_cap_onchip':g&d['below']&d['onchip'],'all':g}
        res={'measured_runtime':{k:summary(d,m,d['t']) for k,m in groups.items()},'fixed_signed':{},'random_lognormal':{}}
        # Share of energy that scales with runtime, per below-cap cell.
        share=(d['p0']*d['t']/(d['p0']*d['t']+d['eps']*d['B']))[groups['below_cap']]
        res['runtime_term_share_of_predicted_energy_below_cap']={'median':round(float(np.median(share)),3),'p10':round(float(np.percentile(share,10)),3),'p90':round(float(np.percentile(share,90)),3)}
        for delta in [-.3,-.2,-.1,.1,.2,.3]:
            res['fixed_signed'][f'{delta:+.0%}']={k:summary(d,m,d['t']*(1+delta)) for k,m in groups.items() if k!='all'}
        for target in [.05,.10,.15,.20,.30]:
            s=target/0.6745  # median |N(0,s)| = 0.6745 s; log errors ~ relative errors at these sizes
            meds={k:[] for k in groups if k!='all'};p90={k:[] for k in meds}
            for _ in range(500):
                t=d['t']*np.exp(rng.normal(0,s,len(d['t'])))
                for k in meds:
                    e=err(d,t)[groups[k]];meds[k].append(np.median(e));p90[k].append(np.percentile(e,90))
            res['random_lognormal'][f'median_runtime_error_{target:.0%}']={k:{'median_pct_mean':round(float(np.mean(meds[k])),2),
                'median_pct_95th':round(float(np.percentile(meds[k],95)),2),'p90_pct_mean':round(float(np.mean(p90[k])),2)} for k in meds}
        out['groups'][gpu]=res
    (HERE/'error_budget.json').write_text(json.dumps(out,indent=1)+'\n');return out

if __name__=='__main__':
    o=main()
    for gpu,res in o['groups'].items():
        print(gpu,'measured',res['measured_runtime']['below_cap'],'onchip',res['measured_runtime']['below_cap_onchip'],'share',res['runtime_term_share_of_predicted_energy_below_cap'])
        for k,v in res['random_lognormal'].items():print('  ',k,'below_cap',v['below_cap'],'onchip',v['below_cap_onchip'])
