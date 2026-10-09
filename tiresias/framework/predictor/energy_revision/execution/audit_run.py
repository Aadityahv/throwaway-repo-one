"""Audit retained calibration traces and costs without fitting or target labels."""
import argparse
import hashlib
import json
from pathlib import Path

MIX_NAMES={'memory':'Memory lookup','arithmetic':'FP32 arithmetic','special_function':'Special function',
           'other_shared':'Other and shared work','mixed':'Mixed work','mixed_altered':'Altered mixed work'}
FOOT_NAMES={'small_candidate':'Small bypass-L1 control','l2_candidate':'L2-sized control',
            'dram_candidate':'Larger-than-L2 control'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(run_dir, policy):
    design=json.loads((run_dir/'correctness/result.json').read_text())
    acquisition=json.loads((run_dir/'energy/result.json').read_text())
    freeze=json.loads((run_dir/'ENERGY_ACQUISITION_FREEZE.json').read_text())
    if sha(run_dir/'correctness/result.json')!=freeze['design_sha256']:
        raise ValueError('REFUSED: correctness design changed')
    if sha(run_dir/'native/calibration')!=freeze['binary_sha256']:
        raise ValueError('REFUSED: native binary changed')
    fp=hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if acquisition['design_sha256']!=fp:raise ValueError('REFUSED: acquisition/design mismatch')
    records={r['design_id']:r for r in acquisition['rows']}
    if len(records)!=len(acquisition['rows']):raise ValueError('REFUSED: duplicate acquired slot')
    rows=[];near_cap=[];counted=0.;pre=0.
    for s in design['rows']:
        r=records.get(s['design_id']);status=r['status'] if r else 'not_acquired'
        row=dict(design_id=s['design_id'],activity=MIX_NAMES[s['mix']],dose=s['dose'],
                 footprint=FOOT_NAMES[s['footprint_candidate']],role=s['role'],status=status)
        if status=='accepted':
            for key in ('source_sha256','binary_sha256','count_evidence_sha256'):
                if r[key]!=s[key]:raise ValueError('REFUSED: acquired provenance mismatch')
            remote=Path(r['raw_record']['trace_dir']);parts=remote.parts
            i=len(parts)-1-list(reversed(parts)).index('energy')
            trace=run_dir/'energy'/Path(*parts[i+1:])
            hashes={str(f.relative_to(trace)):sha(f) for f in sorted(trace.rglob('*')) if f.is_file()}
            if hashes!=r['trace_files_sha256']:raise ValueError('REFUSED: missing/changed raw trace')
            digest=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if digest!=r['trace_sha256']:raise ValueError('REFUSED: raw trace registry mismatch')
            power=r['energy_j_per_launch']/r['counted_runtime_s_per_launch']
            row.update(power_w=power,energy_j_per_launch=r['energy_j_per_launch'],runtime_s_per_launch=r['counted_runtime_s_per_launch'],
                       counted_cuda_s=r['counted_interval_s'],precondition_cuda_s=r['precondition_cuda_s'],
                       below_calibration_cutoff=power<policy['below_cap_fraction']*policy['cap_w'])
            counted+=row['counted_cuda_s'];pre+=row['precondition_cuda_s']
            if not row['below_calibration_cutoff']:near_cap.append(s['design_id'])
        elif r:row['reason']=r.get('rejection_reason','measurement rejected')
        rows.append(row)
    complete=bool(acquisition['complete']) and len(records)==30 and all(r['status']=='accepted' for r in rows)
    result=dict(schema='component_calibration_audit/1',complete_measurement_grid=complete,requested=30,
                accepted=sum(r['status']=='accepted' for r in rows),near_cap_slots=near_cap,
                calibration_power_cutoff_w=policy['below_cap_fraction']*policy['cap_w'],
                fit_preconditions_pass=complete and not near_cap,rows=rows,
                costs=dict(energy_acquisition_wall_s=acquisition['total_wall_s'],counted_cuda_s=counted,precondition_cuda_s=pre,
                           correctness_wall_s=design['correctness_elapsed_wall_s']),
                qualification='Measurement acceptance is distinct from model admission. Proximity to the cap is not proof of throttling. No coefficients or target errors computed.')
    result['inputs_sha256']={str(p.relative_to(run_dir)):sha(p) for p in (run_dir/'correctness/result.json',run_dir/'energy/result.json',run_dir/'ENERGY_ACQUISITION_FREEZE.json')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=Path(__file__).parent/'fit_policy.json');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=audit(a.run_dir,json.loads(a.policy.read_text()));r['policy_sha256']=sha(a.policy);r['code_sha256']=sha(Path(__file__))
    with a.out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('Verified calibration traces; fit preconditions:',r['fit_preconditions_pass'])


if __name__=='__main__':main()
