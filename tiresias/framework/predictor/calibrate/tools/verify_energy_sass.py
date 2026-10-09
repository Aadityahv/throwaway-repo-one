#!/usr/bin/env python3
"""Compile-only check of the energy stage for a GPU architecture. NO CUDA context, NO GPU is used.

    python3 calibrate/tools/verify_energy_sass.py --nvcc /usr/local/cuda/bin/nvcc --arch sm_90 [--out-dir build_check]
    python3 calibrate/tools/verify_energy_sass.py --sass-file saved_cuobjdump_output.txt      # offline: verify an existing dump

Builds csrc/micro_energy.cu for the architecture, dumps the real SASS with cuobjdump, and checks every design of the energy stage against it: each priced instruction class appears exactly
as many times per loop iteration as the design promises, there is exactly one backward (trip) branch, no register spill and no predicate-guarded instruction inside the loop. Run this on
every new architecture BEFORE booking the GPU: a mismatch means the compiler changed a priced instruction (the energy windows would then not measure what they claim) and the opcode
families or the kernel need attention. Exit code 0 = all designs verified, 1 = a problem (listed), 2 = tool error."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import energy as En  # noqa: E402


def verify(sass_text):
    table = {}
    for sym, lines in En.split_functions(sass_text).items():
        p = En.design_of_symbol(sym)
        if p: table[p] = lines
    problems, report = [], {}
    for name, d in En.DESIGNS.items():
        p = d[:9]
        if p not in table: problems.append('%s: no kernel instantiation in the SASS' % name); continue
        try: c = En.count_kernel(table[p], trips=100)
        except ValueError as ex: problems.append('%s: %s' % (name, ex)); continue
        bad = En.verify_design(name, c['in_loop'])
        report[name] = c['in_loop']
        problems += ['%s: %s' % (name, b) for b in bad]
    return problems, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nvcc'); ap.add_argument('--arch', help='e.g. sm_80, sm_89, sm_90, sm_120'); ap.add_argument('--out-dir', type=Path, default=Path('energy_sass_check')); ap.add_argument('--sass-file', type=Path)
    a = ap.parse_args(argv)
    try:
        if a.sass_file: text = a.sass_file.read_text()
        else:
            if not (a.nvcc and a.arch): print('give --nvcc and --arch, or --sass-file', file=sys.stderr); return 2
            a.out_dir.mkdir(parents=True, exist_ok=True); exe = a.out_dir / 'micro_energy'
            r = subprocess.run([a.nvcc, '-O3', '-std=c++17', '-arch=%s' % a.arch, '-I', str(HERE / 'csrc'), '-o', str(exe), str(HERE / 'csrc' / 'micro_energy.cu'), '-ldl'], capture_output=True, text=True)
            if r.returncode: print('compile failed for %s:\n%s' % (a.arch, r.stderr[-1500:]), file=sys.stderr); return 2
            cud = Path(a.nvcc).parent / 'cuobjdump'; r = subprocess.run([str(cud if cud.is_file() else 'cuobjdump'), '-sass', str(exe)], capture_output=True, text=True)
            if r.returncode: print('cuobjdump failed:\n%s' % r.stderr[-800:], file=sys.stderr); return 2
            text = r.stdout; (a.out_dir / 'sass.txt').write_text(text)
    except OSError as ex:
        print('tool error: %s' % ex, file=sys.stderr); return 2
    problems, report = verify(text)
    for n, il in report.items(): print('%-12s %s' % (n, dict(sorted(il.items()))))
    if problems:
        print('\nPROBLEMS (%d):' % len(problems)); [print('  ' + x) for x in problems]; return 1
    print('\nall %d designs verified against the real SASS' % len(En.DESIGNS)); return 0


if __name__ == '__main__':
    sys.exit(main())
