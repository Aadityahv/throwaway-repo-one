"""Use disjoint physical cores for actual 1/4/8-worker suite comparisons."""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
BASE=Path('/home/user/tiresias_review_cpu_20261005');ROOT=BASE/'source';OUT=BASE/'results'
BENCH=ROOT/'tiresias/framework/cpu_prediction_benchmark/cpu_campaign/benchmark_cpu.py'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cpus=[];seen=set()
for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
    if line.startswith('#'):continue
    cpu,core,socket=map(int,line.split(','))
    if (socket,core) not in seen and cpu in os.sched_getaffinity(0):cpus.append(cpu);seen.add((socket,core))
assert len(cpus)>=64
mapping={1:[cpus[55]],4:cpus[:4],8:cpus[4:12]}
state={'status':'running','start_epoch':time.time(),'source_unchanged':None,'runs':[],
       'methodology':'Actual full-inventory wall times on disjoint physical CPU cores. One-worker run overlaps the tail of the 55-worker run; four- and eight-worker runs overlap each other and the one-worker run. Host load/memory samples are retained. These are observed shared-host throughputs, not isolated scaling guarantees.',
       'selected_logical_cpus':mapping,'physical_cores_reserved':8,
       'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def save():
    tmp=OUT/'parallel_campaign_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'parallel_campaign_state.json')
def run(jobs):
    command=['taskset','-c',','.join(map(str,mapping[jobs])),sys.executable,str(BENCH),'--out',str(OUT),'--mode','cold','--jobs',str(jobs)]
    with (BASE/f'cold_{jobs}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=log,check=True)
    report=json.loads((OUT/f'cold_workers_{jobs}/report.json').read_text());assert report['configurations']==167 and report['predictions_match_frozen']
    return {'workers':jobs,'wall_s':report['wall_s'],'finished_epoch':time.time()}
save()
try:
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,1)]
        while not (OUT/'cold_workers_55/report.json').exists():
            if futures[0].done() and futures[0].exception():raise futures[0].exception()
            for p in (OUT/'cold_workers_55/logs').glob('*.txt'):
                if 'Traceback' in p.read_text():raise RuntimeError('Full-inventory gate failure: '+str(p))
            time.sleep(5)
        state['runs'].append({'workers':55,'wall_s':json.loads((OUT/'cold_workers_55/report.json').read_text())['wall_s']});save()
        futures += [pool.submit(run,4),pool.submit(run,8)]
        for future in cf.as_completed(futures):state['runs'].append(future.result());save()
    manifest=json.loads((OUT/'executed_source_sha256.json').read_text())
    actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert manifest==actual,'Analysis source snapshot changed'
    state.update(status='complete',finished_epoch=time.time(),source_unchanged=True);save()
except BaseException as ex:state.update(status='failed',error=str(ex),finished_epoch=time.time());save();raise
