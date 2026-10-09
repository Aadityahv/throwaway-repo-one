"""Full-grid correctness or single-attempt energy acquisition on approved GPU 1.

No implicit booking, no build, no retries, no model fitting or fresh target access.
Correctness and energy are separate invocations so gates can be committed first.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from prepare import FIT_IDS, HOLD_IDS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
UUID='GPU-0e63baea-0bb1-e90c-f19d-8aa5f2995894'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fingerprint(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(p.read_text())


def validate_design(design,binary,mode):
    required='compiled_counts_frozen_correctness_pending' if mode=='correctness' else 'compiled_counts_and_correctness_frozen'
    if design['schema']!='energy_component_calibration_design/1' or design['status']!=required:
        raise ValueError('REFUSED: compiled count/correctness admission missing')
    if len(design['rows'])!=30 or len({r['design_id'] for r in design['rows']})!=30:
        raise ValueError('REFUSED: full 30-slot grid required')
    if [r['design_id'] for r in design['rows']]!=list(FIT_IDS+HOLD_IDS):
        raise ValueError('REFUSED: frozen slot ordering/grid changed')
    for r in design['rows']:
        if r['count_status']!='exact' or r['abi_status']!='verified' or r['binary_sha256']!=sha(binary):
            raise ValueError('REFUSED: binary/count/ABI drift')
        if r['source_sha256']!=sha(HERE/'calibration.cu'):
            raise ValueError('REFUSED: source drift')
        proof=design['certificates'].get(r['count_evidence_sha256'])
        if proof is None or fingerprint(proof)!=r['count_evidence_sha256'] or r['work'].get('all_counts_exact') is not True:
            raise ValueError('REFUSED: count certificate absent or modified')
    if mode=='energy' and any(not r.get('correctness_pass') for r in design['rows']):
        raise ValueError('REFUSED: entire correctness grid must pass before energy')


def append(path,doc):
    with path.open('a') as f:f.write(json.dumps(doc,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['correctness','energy']);p.add_argument('--design',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--cells-dir',type=Path,required=True,help='same persistent private cell directory across correctness/energy; binds CPU oracle hashes')
    p.add_argument('--booking-ref',required=True);p.add_argument('--i-have-a-booking',action='store_true')
    p.add_argument('--child-timeout-s',type=float,default=900);p.add_argument('--total-timeout-s',type=float,default=14400)
    p.add_argument('--freeze-manifest',type=Path,help='required for energy; committed/pushed proof hashes prepared after correctness')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    design=read(a.design);validate_design(design,a.binary,a.mode)
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode,slots=30,energy_attempts_per_slot=1,window_target_s=15,precondition_target_s=90,
            gpu_index=1,uuid=UUID,booking_ref=a.booking_ref,rows=[dict(design_id=r['design_id'],argv=[str(a.binary),*r['argv_prefix'][:-1],'CELL_OUTPUT.bin']) for r in design['rows']]),indent=2));return
    if not a.i_have_a_booking or not a.booking_ref.strip():raise ValueError('REFUSED: booking log booking required; check that every other GPU 1 booking is closed')
    if a.mode=='energy':
        if a.freeze_manifest is None:raise ValueError('REFUSED: pushed correctness/count freeze manifest required before energy')
        freeze=read(a.freeze_manifest)
        for name,path in [('design_sha256',a.design),('binary_sha256',a.binary),('source_sha256',HERE/'calibration.cu'),('oracle_code_sha256',HERE/'oracle.hpp'),('acquirer_sha256',Path(__file__))]:
            if freeze.get(name)!=sha(path):raise ValueError('REFUSED: energy freeze input drift')
        if not freeze.get('pushed_commit'):raise ValueError('REFUSED: pushed proof commit not recorded')
    if not 0<a.child_timeout_s<=900 or not 0<a.total_timeout_s<=14400:raise ValueError('REFUSED: invalid bounded session timeouts')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or os.environ.get('UNSEEN_EXPECT_UUID')!=UUID:
        raise ValueError('REFUSED: exact physical GPU 1 visibility/UUID required')
    os.environ['ENERGY_BOOKING_REF']=a.booking_ref;os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    sys.path.insert(0,str(ROOT/'energy_harness'))
    import application_energy_harness as H
    # Local imported instance only: no repository mutation, retargeting or retry.
    H.MAX_DURATION_ATTEMPTS=1
    start=time.monotonic();deadline=start+a.total_timeout_s
    a.out_dir.mkdir(parents=True,exist_ok=False);a.cells_dir.mkdir(parents=True,exist_ok=True)
    ledger=a.out_dir/'ledger.jsonl';result=[]

    def run(argv):
        timeout=min(a.child_timeout_s,deadline-time.monotonic())
        if timeout<=0:raise TimeoutError('session deadline exhausted')
        return subprocess.run(argv,capture_output=True,text=True,timeout=timeout)

    def harness_child(binary,argv_prefix,repeat,batch,trace_dir):
        argv=[str(binary),*map(str,argv_prefix),'--repeat',str(repeat),'--graph-batch',str(batch),'--trace-dir',str(trace_dir)]
        begin=time.monotonic()
        proc=run(argv)
        # Duration probes share a directory in the common harness; preserve every
        # observed window before the next invocation could overwrite windows.csv.
        child_log=trace_dir/'children.jsonl'
        ordinal=len(child_log.read_text().splitlines()) if child_log.exists() else 0
        prefix=f'child-{ordinal:02d}'
        (trace_dir/(prefix+'.stdout')).write_text(proc.stdout);(trace_dir/(prefix+'.stderr')).write_text(proc.stderr)
        win=trace_dir/'windows.csv'
        if win.exists():(trace_dir/(prefix+'.windows.csv')).write_bytes(win.read_bytes())
        append(child_log,dict(argv=argv,returncode=proc.returncode,wall_s=time.monotonic()-begin))
        return proc
    H._run_binary=harness_child

    def settle_idle():
        # The driver's process exits before utilisation reporting necessarily settles.
        # Wait only for this reporting lag; a new foreign process causes refusal.
        for _ in range(90):
            if H._compute_process_pids(1):raise ValueError('REFUSED: GPU 1 acquired by another process')
            q=run(['nvidia-smi','-i','1','--query-gpu=uuid,power.limit,utilization.gpu','--format=csv,noheader,nounits'])
            if q.returncode:raise ValueError('REFUSED: idle query failed')
            fields=[s.strip() for s in q.stdout.strip().split(',')]
            if len(fields)!=3 or fields[0]!=UUID or float(fields[1])!=design['cap_w']:
                raise ValueError('REFUSED: live GPU identity/power limit differs from frozen design')
            if int(fields[2])==0:return
            if time.monotonic()>=deadline:raise TimeoutError('session deadline during idle settle')
            time.sleep(1)
        raise ValueError('REFUSED: GPU 1 did not settle idle within bounded interval')
    for index,row in enumerate(design['rows']):
        append(ledger,dict(event='start',design_id=row['design_id'],mode=a.mode,monotonic_s=time.monotonic(),booking_ref=a.booking_ref))
        try:
            settle_idle()
            uuid=H.preflight_before_context(1,'blackwell')
            if uuid!=UUID:raise ValueError('REFUSED: live physical UUID changed')
            d=a.cells_dir/f'{index:02d}';d.mkdir(exist_ok=True)
            output=d/'out.bin';argv=row['argv_prefix'][:-1]+[str(output)]
            if a.mode=='correctness':
                child=run([str(a.binary),*argv])
                (d/'correctness.stdout').write_text(child.stdout);(d/'correctness.stderr').write_text(child.stderr)
                check=output.with_suffix(output.suffix+'.check')
                passed=child.returncode==0 and check.is_file() and check.read_text().startswith('CHECK_OK ')
                rec=dict(row,correctness_pass=passed,correctness_returncode=child.returncode,
                    correctness_stdout_sha256=sha(d/'correctness.stdout'),correctness_stderr_sha256=sha(d/'correctness.stderr'),
                    actual_output_sha256=sha(output) if passed else None,
                    check_sha256=sha(check) if check.is_file() else None,
                    oracle_sha256=sha(Path(str(output)+'.oracle')) if passed else None)
            else:
                oracle=Path(str(output)+'.oracle')
                if not oracle.is_file() or sha(oracle)!=row['oracle_sha256']:raise ValueError('REFUSED: CPU oracle cache changed')
                # Two timing probes control graph batch; these are calibration costs,
                # not target timing and not additional counted energy windows.
                pd=d/'graph_probe';pd.mkdir(exist_ok=False)
                probe=harness_child(a.binary,argv,1,1,pd)
                (pd/'stdout.txt').write_text(probe.stdout);(pd/'stderr.txt').write_text(probe.stderr)
                if probe.returncode:raise ValueError('graph-batch probe failed')
                timing=next(csv.DictReader((pd/'windows.csv').open()))
                t=float(timing['cuda_seconds'])/int(timing['launches'])
                if not math.isfinite(t) or t<=0:raise ValueError('nonpositive graph probe time')
                batch=min(2048,max(1,math.ceil(.005/t)))
                ctx=H.BinaryEnergyContext(parent_id='synthetic_component_calibration',regime=row['footprint_candidate'],
                    candidate_id=row['design_id'],source_revision='new_component_calibration',source_sha256=row['source_sha256'],
                    source_path=str(HERE/'calibration.cu'),controls=dict(design_id=row['design_id'],tier=row['tier'],n=row['n'],dose=row['dose'],mix=row['mix']),
                    runtime_info=dict(binary_sha256=row['binary_sha256'],count_evidence_sha256=row['count_evidence_sha256'],oracle_sha256=row['oracle_sha256']),
                    binary=a.binary,argv_prefix=argv,check=lambda o=output:Path(str(o)+'.check').read_text().startswith('CHECK_OK '))
                raw=H.run_binary_energy(ctx,window_target_seconds=15,session=1,run_id=f'energy-component-{index:02d}',out_dir=a.out_dir,
                    gpu_index=1,platform='blackwell',runner_name='component_calibration',gpu_uuid=uuid,graph_batch=batch)
                H.check_gpu_unshared_excluding_self(1)
                H.append_raw_or_rejected(a.out_dir,raw);r=raw['row']
                rec=dict(design_id=row['design_id'],role=row['role'],status='accepted' if raw['status']=='raw' else 'rejected',attempts=1,
                    correctness_pass=r.get('correctness_check',False),source_sha256=row['source_sha256'],binary_sha256=row['binary_sha256'],
                    count_evidence_sha256=row['count_evidence_sha256'],raw_record=r)
                if raw['status']=='raw':
                    td=Path(r['trace_dir']);hashes={str(f.relative_to(td)):sha(f) for f in sorted(td.rglob('*')) if f.is_file()}
                    rec.update(energy_j_per_launch=r['board_energy_j_per_launch'],counted_launches=r['launch_count'],
                        counted_runtime_s_per_launch=r['counted_launch_interval_s']/r['launch_count'],counted_interval_s=r['counted_launch_interval_s'],
                        precondition_cuda_s=r['precondition_cuda_seconds'],trace_sha256=fingerprint(hashes),trace_files_sha256=hashes)
                else:rec['rejection_reason']=r.get('rejection_reason','unknown')
            result.append(rec);append(ledger,dict(event='finish',design_id=row['design_id'],mode=a.mode,record=rec,monotonic_s=time.monotonic()))
            print(index+1,'/30',row['design_id'],rec.get('status',rec.get('correctness_pass')),flush=True)
            if a.mode=='energy' and rec['status']!='accepted':break
            if a.mode=='correctness' and not rec['correctness_pass']:break
        except Exception as exc:
            append(ledger,dict(event='failure',design_id=row['design_id'],mode=a.mode,reason=str(exc),monotonic_s=time.monotonic()))
            break
    complete=len(result)==30 and all(r.get('status')=='accepted' if a.mode=='energy' else r['correctness_pass'] for r in result)
    if a.mode=='correctness':
        out=dict(design,status='compiled_counts_and_correctness_frozen' if complete else 'correctness_rejected',rows=result,
                 full_correctness_grid_pass=complete,correctness_elapsed_wall_s=time.monotonic()-start)
    else:
        out=dict(schema='energy_component_calibration_measurements/1',target_role='calibration_only',synthetic=False,
                 design_sha256=fingerprint(design),rows=result,complete=complete,total_wall_s=time.monotonic()-start,
                 note='Actual total for this acquisition session, including idle/probes/setup/cache reads. Earlier build/count/full oracle correctness costs are separate.')
    with (a.out_dir/'result.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    if not complete:raise SystemExit('REFUSED: partial/failed full grid; raw ledger retained, no profile admitted')


if __name__=='__main__':main()
