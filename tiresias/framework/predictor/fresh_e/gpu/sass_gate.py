#!/usr/bin/env python3
"""SASS identity gate: the kernels linked into the freshly built timing binaries must have exactly the instruction sequence that was analysed
(fresh_e/isolated/<kernel>.isolated.sass). Usage: sass_gate.py <workdir> [cuobjdump]. Exit 1 on any mismatch. CPU only."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; STATIC = HERE.parents[0].parent  # predictor/
sys.path.insert(0, str(STATIC / 'fresh_d'))
from timing_fresh_d import sass_identical  # noqa: E402  (imported, never edited)
CHECKS = [('sp', 'fresh_e_sp', '_Z13scalarProdGPUPfS_S_ii'), ('fwt1', 'fresh_e_fwt', '_Z15fwtBatch1KernelPfS_i'), ('fwt2', 'fresh_e_fwt', '_Z15fwtBatch2KernelPfS_i')]
def main():
    work = Path(sys.argv[1]).expanduser(); cuo = sys.argv[2] if len(sys.argv) > 2 else '/usr/local/cuda-13.2/bin/cuobjdump'; ok = True; res = []
    for kid, binary, sym in CHECKS:
        dump = subprocess.run([cuo, '-sass', '-fun', sym, '-arch', 'sm_120', str(work / 'bin' / binary)], capture_output=True, text=True, check=True).stdout
        same, detail = sass_identical((HERE.parent / 'isolated' / (kid + '.isolated.sass')).read_text(), dump)
        res.append(dict(kernel=kid, symbol=sym, binary=binary, sass_equal=same, detail=detail)); ok &= same
    print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
if __name__ == '__main__': main()
