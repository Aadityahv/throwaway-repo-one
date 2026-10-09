"""Explicit two-control replacement; charge all 32 windows, never hide retired data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPLACED={'memory/low/dram_candidate','memory/high/dram_candidate'}
base_path=HERE.parent/'execution/fit.py'
spec=importlib.util.spec_from_file_location('frozen_initial_energy_fit',base_path)
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)


def assemble(original_design,original_acquisition,new_design,new_acquisition,policy):
    for design,acq,n in [(original_design,original_acquisition,30),(new_design,new_acquisition,2)]:
        if design['schema']!='energy_component_calibration_design/1' or acq['schema']!='energy_component_calibration_measurements/1' or design['cap_w']!=policy['cap_w']:
            raise ValueError('REFUSED: calibration schema/cap mismatch')
        if not acq.get('complete') or len(acq['rows'])!=n or len(design['rows'])!=n:
            raise ValueError('REFUSED: both complete original and replacement grids required')
        if acq['design_sha256']!=BASE.fingerprint(design):raise ValueError('REFUSED: acquired design drift')
        if design['status']!='compiled_counts_and_correctness_frozen' or acq['target_role']!='calibration_only' or acq.get('synthetic'):
            raise ValueError('REFUSED: native calibration proof missing')
        if any(r['status']!='accepted' or r['attempts']!=1 or not r['correctness_pass'] for r in acq['rows']):
            raise ValueError('REFUSED: rejected/retried measurement')
    old={r['design_id']:r for r in original_acquisition['rows']};new={r['design_id']:r for r in new_acquisition['rows']}
    if set(old)!=set(BASE.FIT_IDS+BASE.HOLD_IDS) or set(new)!=REPLACED:
        raise ValueError('REFUSED: undeclared retirement/replacement')
    cutoff=policy['cap_w']*policy['below_cap_fraction']
    cap_ids={k for k,r in old.items() if r['energy_j_per_launch']/r['counted_runtime_s_per_launch']>=cutoff}
    if cap_ids!=REPLACED:raise ValueError('REFUSED: original failure differs from the two-control proposal')
    if any(r['energy_j_per_launch']/r['counted_runtime_s_per_launch']>=cutoff for r in new.values()):
        raise ValueError('REFUSED: replacement control still near cap')
    nd={r['design_id']:r for r in new_design['rows']}
    if set(nd)!=REPLACED:raise ValueError('REFUSED: replacement static grid mismatch')
    od={r['design_id']:r for r in original_design['rows']}
    for k,s in nd.items():
        for field in ('role','dose','n','grid','block','logical_bytes','tier'):
            if s[field]!=od[k][field]:raise ValueError('REFUSED: undeclared geometry/activity-dose change')
        if s['source_sha256']==od[k]['source_sha256']:
            raise ValueError('REFUSED: same source is a forbidden retry, not a new control')
        if s['binary_sha256']==od[k]['binary_sha256'] or s['count_evidence_sha256']==od[k]['count_evidence_sha256']:
            raise ValueError('REFUSED: new native binary/count proof required')
        if s.get('dependent_checksum_steps_per_memory_load')!=9:
            raise ValueError('REFUSED: unreviewed memory dependence')
    ids=BASE.FIT_IDS+BASE.HOLD_IDS
    design=dict(original_design,rows=[nd.get(k,od[k]) for k in ids],
                certificates={**original_design['certificates'],**new_design['certificates']},
                qualification='Explicitly revised calibration: 28 original controls and two new memory-dependence controls. No target labels.')
    acquisition=dict(original_acquisition,rows=[new.get(k,old[k]) for k in ids],design_sha256=BASE.fingerprint(design),
                     total_wall_s=original_acquisition['total_wall_s']+new_acquisition['total_wall_s'])
    return design,acquisition,[old[k] for k in sorted(REPLACED)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original-design','original-measurements','replacement-design','replacement-measurements'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--policy',type=Path,default=HERE/'fit_policy.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();files=[a.original_design,a.original_measurements,a.replacement_design,a.replacement_measurements,a.policy]
    docs=[json.loads(f.read_text()) for f in files]
    design,acq,retired=assemble(*docs);result=BASE.fit_profiles(design,acq,docs[-1])
    cost=result['calibration_budget'];cost.update(acquired_windows=32,retired_near_cap_windows=2,fit_windows=24,heldout_windows=6)
    for key,field in [('counted_cuda_s','counted_interval_s'),('precondition_cuda_s','precondition_cuda_s')]:cost[key]+=sum(r[field] for r in retired)
    cost['note']='All original 30 and new two windows charged. Explicit retirement is calibration-design development, never a hidden exclusion. Gates/build/count costs separately reported.'
    result['calibration_revision']=dict(retired_controls=sorted(REPLACED),reason='Two original controls exceed the unchanged 95%-cap gate',original_transfer_exposure='Acquired in original calibration; never fitted',target_exposure='No target labels used')
    result['inputs_byte_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    result['code_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),base_path,HERE.parent/'component_candidate.py')}
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(result['status'])
    if result['status']!='calibrated_and_frozen':raise SystemExit(2)


if __name__=='__main__':main()
