#!/usr/bin/env python3
"""Compile-only check of the tensor stage's energy kernel against the real SASS (no GPU context; run before booking, like verify_energy_sass.py).

    python3 calibrate/tools/verify_tensor_sass.py --nvcc /usr/local/cuda-13.2/bin/nvcc --arch sm_120

Compiles csrc/micro_tensor.cu, dumps the SASS, and requires: the mma_energy<32> kernel has exactly 32 tensor-core instructions and one global load per loop iteration, one backward loop, no spill, and none of
the priced classes of the other energy designs in the loop. Exit 0 on success, 1 on any mismatch."""
import argparse, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cal import energy as En, tensor as Tn  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nvcc', required=True); ap.add_argument('--arch', required=True); a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / 'micro_tensor'
        r = subprocess.run([a.nvcc, '-O3', '-std=c++17', '-arch=' + a.arch, '-I', str(HERE / 'csrc'), '-o', str(exe), str(HERE / 'csrc/micro_tensor.cu'), '-ldl'], capture_output=True, text=True)
        if r.returncode: print(r.stderr[-2000:]); return 1
        cuobjdump = str(Path(a.nvcc).with_name('cuobjdump'))
        s = subprocess.run([cuobjdump, '-sass', str(exe)], capture_output=True, text=True)
        if s.returncode: print(s.stderr[-500:]); return 1
    funcs = En.split_functions(s.stdout); sym = next((f for f in funcs if f.startswith('_Z10mma_energyILi%dE' % Tn.M_PER_TRIP)), None)
    if sym is None: print('mma_energy kernel not found'); return 1
    try: c = En.count_kernel(funcs[sym], 2048)
    except ValueError as ex: print('refused:', ex); return 1
    bad = Tn.verify_design(c['in_loop']); print(sym, 'in loop:', c['in_loop'])
    if bad: print('MISMATCH:', '; '.join(bad)); return 1
    print('tensor energy kernel verified against the real SASS'); return 0


if __name__ == '__main__':
    sys.exit(main())
