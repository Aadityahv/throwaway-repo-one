"""Compile and run the calibration programs with full provenance. `run` is injectable so the orchestration can be tested on a CPU with a stub."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CSRC = HERE / 'csrc'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arch_flag(facts):
    major, minor = facts['compute_capability'].split('.')
    return 'sm_%s%s' % (major, minor)


class Toolchain:
    def __init__(self, nvcc, run=subprocess.run):
        self.nvcc, self.run = nvcc, run
        r = run([nvcc, '--version'], capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError('nvcc not runnable: %s' % nvcc)
        self.version = [l for l in r.stdout.splitlines() if 'release' in l][-1].strip()

    def compile(self, program, facts, build_dir):
        src = CSRC / (program + '.cu'); out = Path(build_dir) / program; Path(build_dir).mkdir(parents=True, exist_ok=True)
        cmd = [self.nvcc, '-O3', '-std=c++17', '-arch=%s' % arch_flag(facts), '-I', str(CSRC), '-o', str(out), str(src)] + (['-ldl'] if program in ('micro_energy', 'micro_tensor') else [])
        t = time.time(); r = self.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError('compile of %s failed for %s (CUDA toolkit too old for this architecture?):\n%s' % (program, arch_flag(facts), (r.stderr or '')[-800:]))
        return dict(program=program, source=str(src.relative_to(HERE.parent)), source_sha256=sha256(src), binary_sha256=sha256(out), command=cmd, compile_seconds=round(time.time() - t, 1), binary=str(out))


def program_env(uuid, booking):
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = uuid; env['CAL_EXPECT_UUID'] = uuid; env['CAL_BOOKING_REF'] = booking
    return env


def run_program(binary, args, uuid, booking, timeout, run=subprocess.run):
    """Run one program; returns dict(returncode, stdout, stderr, seconds). Never retries."""
    t = time.time()
    try:
        r = run([str(binary)] + [str(a) for a in args], capture_output=True, text=True, env=program_env(uuid, booking), timeout=timeout)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr, seconds=round(time.time() - t, 2))
    except subprocess.TimeoutExpired as ex:
        return dict(returncode=-9, stdout=(ex.stdout or b'').decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ''), stderr='TIMEOUT after %ss' % timeout, seconds=round(time.time() - t, 2))


def parse_jsonl(text):
    rows = []
    for l in text.splitlines():
        l = l.strip()
        if l.startswith('{'):
            rows.append(json.loads(l))
    return rows
