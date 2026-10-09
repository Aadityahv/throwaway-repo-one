#!/usr/bin/env python3
"""CPU-only entry point for Tiresias.

  python3 reproduce.py list                show the available commands
  python3 reproduce.py predict [--jobs N]  re-predict all 167 evaluated configurations from the retained compiled kernels
                                           and compare each prediction (or refusal) with the frozen record
  python3 reproduce.py test                run the unit tests of the framework and the calibrator

No GPU is used.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PRED = ROOT / 'tiresias/framework/predictor'
BENCH = ROOT / 'tiresias/framework/cpu_prediction_benchmark/cpu_campaign/benchmark_cpu.py'
TEST_TARGETS = [PRED / 'calibrate/tests', PRED / 'test_extract_features.py', PRED / 'test_runtime_completion.py']
SCRIPT_TESTS = [PRED / 'test_runtime_v3k.py', PRED / 'port_common/test_port_ext.py', PRED / 'port_common/test_set_d_port.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    p = sub.add_parser('predict'); p.add_argument('--jobs', type=int, default=4); p.add_argument('--out', default=str(ROOT / 'predict_out'))
    sub.add_parser('test')
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if a.cmd == 'list':
        print('predict   re-predict all 167 configurations and compare with the frozen record')
        print('test      run the unit tests (pytest) and the script-style tests')
        return 0
    if a.cmd == 'predict':
        for extra in (['--prepare'], ['--mode', 'cached', '--jobs', str(a.jobs)]):
            rc = subprocess.run([PY, str(BENCH), '--out', a.out] + extra, env=env).returncode
            if rc:
                return rc
        return 0
    rc = 0
    for t in TEST_TARGETS:
        rc |= subprocess.run([PY, '-m', 'pytest', '-q', str(t)], env=env).returncode
    for t in SCRIPT_TESTS:
        rc |= subprocess.run([PY, t.name], cwd=t.parent, env=env).returncode
    return rc


if __name__ == '__main__':
    sys.exit(main())
