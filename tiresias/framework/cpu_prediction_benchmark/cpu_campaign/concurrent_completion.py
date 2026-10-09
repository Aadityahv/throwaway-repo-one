"""Adopt existing suites and start remaining comparisons without duplicate work."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
BASE = Path('/home/user/tiresias_review_cpu_20261005')
ROOT = BASE / 'source'; OUT = BASE / 'results'
BENCH = ROOT / 'tiresias/framework/cpu_prediction_benchmark/cpu_campaign/benchmark_cpu.py'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
cpus = []; seen = set()
for line in subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True).splitlines():
    if line.startswith('#'): continue
    cpu, core, socket = map(int, line.split(','))
    if (socket, core) not in seen and cpu in os.sched_getaffinity(0):
        cpus.append(cpu); seen.add((socket, core))
assert len(cpus) >= 64
mapping = {4: cpus[:4], 8: cpus[4:12]}
state = {'status': 'running', 'start_epoch': time.time(), 'runs': [], 'source_unchanged': None,
         'adopted_existing_suites': [1, 55], 'physical_cores_reserved': 8,
         'new_suite_logical_cpus': mapping,
         'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'methodology': 'Actual shared-host full-inventory wall times. The one-, four- and eight-worker suites overlap; the four/eight suites also overlap the final matrix-multiply case of the initial 55-worker suite. Four/eight/one use separate CPU masks; the initial suite mask includes the four/eight CPUs. Load and memory samples are retained. These are not isolated scaling guarantees. No active analysis process is stopped or duplicated.'}
def save():
    path = OUT / 'concurrent_campaign_state.tmp'
    path.write_text(json.dumps(state, indent=2) + '\n')
    path.replace(OUT / 'concurrent_campaign_state.json')
processes = {}; logs = {}
save()
try:
    for workers in [4, 8]:
        assert not (OUT / f'cold_workers_{workers}').exists(), 'Refuse duplicate suite'
        logs[workers] = (BASE / f'cold_{workers}.log').open('x')
        processes[workers] = subprocess.Popen(['taskset', '-c', ','.join(map(str, mapping[workers])),
            sys.executable, str(BENCH), '--out', str(OUT), '--mode', 'cold', '--jobs', str(workers)],
            stdout=logs[workers], stderr=logs[workers])
        state.setdefault('launched_pids', {})[workers] = processes[workers].pid; save()
    pending = {1, 4, 8, 55}
    while pending:
        for workers in sorted(pending):
            path = OUT / f'cold_workers_{workers}/report.json'
            if path.exists():
                report = json.loads(path.read_text())
                assert report['configurations'] == 167 and report['predictions_match_frozen']
                state['runs'].append({'workers': workers, 'wall_s': report['wall_s'], 'observed_finished_epoch': time.time()})
                pending.remove(workers); save()
            elif workers in processes and processes[workers].poll() is not None:
                raise RuntimeError(f'Suite {workers} ended without verified report; inspect cold_{workers}.log')
            elif 'Traceback' in (BASE / f'cold_{workers}.log').read_text():
                raise RuntimeError(f'Suite {workers} failed; inspect cold_{workers}.log')
        if pending: time.sleep(5)
    manifest = json.loads((OUT / 'executed_source_sha256.json').read_text())
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual == manifest, 'Analysis snapshot changed'
    state.update(status='complete', finished_epoch=time.time(), source_unchanged=True); save()
except BaseException as ex:
    state.update(status='failed', finished_epoch=time.time(), error=str(ex)); save(); raise
finally:
    for log in logs.values(): log.close()
