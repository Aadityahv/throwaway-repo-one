"""One-shot collector for this already running CPU job; never starts experiments.

Run --once to collect only completed suites, or --wait to await all four suites.
--publish permits one scoped completion commit on the configured project branch.
No manuscript files, model code, shared checkout or GPU settings are written.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/user/tiresias')
REL = Path('tiresias/framework/cpu_prediction_benchmark/cpu_completion')
REMOTE = '/home/user/tiresias_review_cpu_20261005'
EXPECTED_CONTROLLER = '4d26650e46027462f58966379c813610eb8095178caa4356b7117d57a02d12b8'
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--once', action='store_true')
group.add_argument('--wait', action='store_true')
parser.add_argument('--publish', action='store_true')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert not args.publish or args.wait, 'Never publish a partial completion'
assert args.out.resolve().is_relative_to(REPO), 'Only this authorized workspace may receive outputs'
args.out.mkdir(parents=True, exist_ok=True)
status_path = args.out / 'collection_status.json'
if status_path.exists():
    old = json.loads(status_path.read_text())
    assert old['status'] not in ['waiting', 'publishing', 'complete'], 'Refuse duplicate collector'
state = {'status': 'waiting' if args.wait else 'collecting', 'pid': os.getpid(),
         'started_epoch': time.time(), 'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'scope': 'One-shot collection of this existing CPU campaign; no scheduling, new experiments, GPU calls or paper changes.'}
def save():
    p = status_path.with_suffix('.tmp')
    p.write_text(json.dumps(state, indent=2) + '\n'); p.replace(status_path)
def run(argv, **kw):
    return subprocess.run(argv, check=True, timeout=kw.pop('timeout', 180), **kw)
def ssh(command, **kw):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'blackwell', command], **kw)
def publish():
    assert args.out.resolve() == (REPO / REL).resolve()
    name = run(['git', '-C', str(REPO), 'config', 'user.name'], capture_output=True, text=True).stdout.strip()
    email = run(['git', '-C', str(REPO), 'config', 'user.email'], capture_output=True, text=True).stdout.strip()
    assert (name, email) == ('the authors', 'author@example.org'), 'Configured identity changed; refuse'
    for attempt in range(3):
        run(['git', '-C', str(REPO), 'fetch', 'origin', 'cross-gpu-prep'], capture_output=True)
        root = Path(tempfile.mkdtemp(prefix='tiresias-cpu-completion-publish-'))
        tree = root / 'checkout'
        run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(tree), 'origin/cross-gpu-prep'], capture_output=True)
        try:
            target = tree / REL
            assert not target.exists(), 'Completion path already exists upstream; refuse to overwrite'
            shutil.copytree(args.out, target)
            note = ('\n\n## CPU review campaign completed — 5 October 2026 launch\n\n'
                    'All 167 valid Blackwell configurations reproduce frozen runtime/energy predictions and refusals '
                    'at one, four, eight and 55 workers. Executed source checksums remain unchanged. '
                    'Compilation, fresh static analysis and cached prediction costs are reported separately '
                    'in tiresias/framework/cpu_prediction_benchmark/cpu_completion/CPU_COST_RESULTS.md. '
                    'These are observed shared-host throughputs with recorded overlapping load, not isolated scaling. '
                    'The private CPU jobs completed; no GPU execution, shared settings, paper edits or rebuilds occurred. '
                    'Every manuscript/table/figure proposal still requires the author\'s exact-edit approval.\n')
            with (tree / 'the booking log').open('a') as f: f.write(note)
            run(['git', '-C', str(tree), 'add', str(REL), 'the booking log'], capture_output=True)
            run(['git', '-C', str(tree), 'commit', '-m', 'Record completed CPU analysis throughput and costs'], capture_output=True)
            commit = run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
            pushed = subprocess.run(['git', '-C', str(tree), 'push', 'origin', 'HEAD:cross-gpu-prep'], capture_output=True, text=True, timeout=180)
            if pushed.returncode == 0:
                return commit
            if 'non-fast-forward' not in pushed.stderr and 'fetch first' not in pushed.stderr:
                raise RuntimeError('Completion push failed; inspect private collector log')
        finally:
            run(['git', '-C', str(REPO), 'worktree', 'remove', str(tree)], capture_output=True)
            root.rmdir()
    raise RuntimeError('Branch changed during three completion publication attempts; outputs remain local')

save()
try:
    deadline = time.time() + 48 * 3600
    while True:
        try:
            remote = json.loads(ssh(f'cat {REMOTE}/results/concurrent_campaign_state.json', capture_output=True, text=True).stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as ex:
            state['last_connection_error'] = type(ex).__name__; save()
            if not args.wait or time.time() >= deadline: raise
            time.sleep(60); continue
        assert remote['controller_sha256'] == EXPECTED_CONTROLLER
        assert remote['status'] != 'failed', remote.get('error')
        completed = sorted(r['workers'] for r in remote['runs'])
        state.update(completed_workers=completed, remote_status=remote['status'], checked_epoch=time.time()); save()
        if not args.wait or remote['status'] == 'complete': break
        assert time.time() < deadline, '48-hour collection deadline; remote evidence remains available'
        time.sleep(60)
    if args.wait:
        assert completed == [1, 4, 8, 55] and remote['source_unchanged'] is True
    assert completed, 'No complete suites to collect'
    members = ['results/host.json', 'results/inventory.json', 'results/executed_source_sha256.json',
               'results/concurrent_campaign_state.json', 'concurrent_completion.py', 'concurrent_campaign.log']
    for w in completed: members += [f'results/cold_workers_{w}', f'cold_{w}.log']
    archive = args.out / 'completed_results.tar.gz'
    temp_archive = archive.with_suffix('.tmp')
    with temp_archive.open('wb') as f:
        ssh(f'tar -czf - -C {REMOTE} ' + ' '.join(members), stdout=f, timeout=600)
    assert 0 < temp_archive.stat().st_size < 100 * 2**20, 'Unexpected evidence archive size'
    temp_archive.replace(archive)
    with tempfile.TemporaryDirectory(prefix='tiresias-verified-cpu-results-') as directory:
        unpack = Path(directory)
        with tarfile.open(archive) as t:
            assert sum(m.size for m in t.getmembers()) < 1024 * 2**20
            for member in t.getmembers():
                p = Path(member.name)
                assert not p.is_absolute() and '..' not in p.parts and not member.issym() and not member.islnk()
                assert member.isfile() or member.isdir()
            t.extractall(unpack, filter='data')
        original = json.loads((HERE / 'evidence/executed_source_sha256.json').read_text())
        assert json.loads((unpack / 'results/executed_source_sha256.json').read_text()) == original
        run([sys.executable, str(HERE / 'summarize_cpu_cost.py'), '--results', str(unpack / 'results'), '--out', str(args.out)], capture_output=True, text=True)
    report = json.loads((args.out / 'cpu_cost_results.json').read_text())
    assert sorted(map(int, report['cold'])) == completed
    if args.wait: assert report['complete_required_worker_comparisons'] and not report['pending_cold_workers']
    state.update(status='publishing' if args.publish else 'collected',
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), completed_epoch=time.time())
    save()
    if args.publish:
        # Put final completion status in the committed package, then record its commit locally.
        state['status'] = 'complete'; save()
        state['published_commit'] = publish(); save()
    print(json.dumps(state, indent=2), flush=True)
except BaseException as ex:
    state.update(status='failed', error=str(ex), failed_epoch=time.time()); save(); raise
