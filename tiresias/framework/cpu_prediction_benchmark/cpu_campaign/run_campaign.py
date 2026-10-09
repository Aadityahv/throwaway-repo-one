"""Sequential CPU benchmarks, private source/output, no GPU execution."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

BASE=Path('/home/user/tiresias_review_cpu_20261005')
SOURCE=BASE/'source'; HERE=SOURCE/'tiresias/framework/cpu_prediction_benchmark/cpu_campaign'
OUT=BASE/'results'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (OUT/'campaign_state.json').exists(),'An existing campaign must not be duplicated'

def save(data):
    tmp=OUT/'campaign_state.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(OUT/'campaign_state.json')
state={'status':'waiting_for_gates','start_epoch':time.time(),'runs':[]};save(state)
# The long FP32 interpretation gate runs independently; do not time it as a suite run.
while True:
    log=(BASE/'gates.log').read_text()
    completed={line.split()[0] for line in log.splitlines() if line.endswith(' 0')}
    if 'ALL GATES PASS' in log or {'classic','e','g','h','prosp','validation_classic'} <= completed:break
    if 'Traceback' in log:raise RuntimeError('Workload gate failed: '+str(BASE/'gates.log'))
    time.sleep(5)

available=os.sched_getaffinity(0); cpus=[];seen=set()
topology=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
for line in topology.splitlines():
    if line.startswith('#'):continue
    cpu,core,socket=map(int,line.split(','));key=(socket,core)
    if cpu in available and key not in seen:cpus.append(cpu);seen.add(key)
assert len(cpus)>=64,'Expected host topology changed; review before running'
selected=cpus[:55];os.sched_setaffinity(0,set(selected))
# One extra physical core is reserved for the already-running FP32 gate.
# This is our own private process; no other user's affinity changes.
try:os.sched_setaffinity(1009541,{cpus[55]})
except ProcessLookupError:pass
host={'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,'selected_logical_cpus':selected,
      'physical_cores_available':len(cpus),'physical_cores_reserved':len(cpus)-len(selected),'cpu_topology':topology,
      'lscpu':subprocess.check_output(['lscpu'],text=True),'memory':subprocess.check_output(['free','-m'],text=True),
      'gpu_preflight':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
      'nvcc':subprocess.check_output(['/usr/local/cuda-13.2/bin/nvcc','--version'],text=True),
      'initial_gate_log':log,'scope':'Cold means fresh derived statistics and a fresh interpreter for every cell; operating-system file caches are not flushed. Existing compiled SASS/cubins are inputs. Compilation/disassembly is a separate benchmark. Cached runs recompute every prediction. No target kernel executes.'}
(OUT/'host.json').write_text(json.dumps(host,indent=2)+'\n')
manifest={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(OUT/'executed_source_sha256.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
try:
    for jobs in [55,8,4,1]:
        state.update(status='running',mode='cold',workers=jobs,run_start_epoch=time.time());save(state)
        with (BASE/f'cold_{jobs}.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'benchmark_cpu.py'),'--out',str(OUT),'--mode','cold','--jobs',str(jobs)],stdout=log,stderr=log,check=True)
        report=json.loads((OUT/f'cold_workers_{jobs}/report.json').read_text())
        assert report['configurations']==167 and report['predictions_match_frozen']
        state['runs'].append({'workers':jobs,'wall_s':report['wall_s'],'completed_epoch':time.time()});save(state)
    after={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert manifest==after,'Frozen source snapshot changed during campaign'
    state.update(status='complete',finished_epoch=time.time(),source_unchanged=True);save(state)
except BaseException as ex:
    state.update(status='failed',error=str(ex),finished_epoch=time.time());save(state);raise
