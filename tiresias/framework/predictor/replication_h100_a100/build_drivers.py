"""Compile the unchanged complete-suite drivers; no GPU context or execution.

Commands come from the Blackwell harness build functions through the existing
recording adapter. Only target architecture and nvcc change. Compiler failures
are recorded and remain failures; every driver is attempted to expose coverage.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import board as B

NVCC = '/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p/bin/nvcc'
PINNED = '5443602d89ed99aede2e4b7bf329daddeadb320e'
VECTOR_HOST_HEADER = '''// Host-only compatibility for the pinned sample's launch-grid helper.
// vecAdd itself uses no symbols from this header; its source is untouched.
#pragma once
namespace cuda {
constexpr int ceil_div(int n, int d) { return n / d + (n % d != 0); }
}
'''


def manifest(label):
    b = B.binding(label)
    b.CUDA_NVCC = NVCC
    m = B.load_module('_cluster_build_manifest', B.ADA / 'make_build_manifest.py')
    m.B, m.NVCC = b, NVCC
    entries = m.entries()
    for e in entries:
        # Keep mechanical commands and kernel lists; no Ada-specific prose.
        e.pop('covers', None)
        e.pop('ada_sass_exists', None)
        e.pop('writes_driver_source', None)
        e['source_of_command'] = e['source_of_command'].replace('sm_89', b.ARCH)
        for k in ('dump_sass', 'dump_res', 'deposit'):
            e['outputs'].pop(k, None)
    return dict(schema='cluster_complete_suite_build/1', board=label, arch=b.ARCH,
                nvcc=NVCC, samples_revision=PINNED,
                source_sha256=m.source_hashes(), builds=entries)


def verify_samples(root):
    if (root / '.source_rev').is_file() and (root / 'MANIFEST.sha256').is_file():
        if (root / '.source_rev').read_text().strip() != PINNED:
            raise SystemExit('REFUSED: archived samples revision mismatch')
        subprocess.run(['sha256sum', '--quiet', '-c', 'MANIFEST.sha256'], cwd=root, check=True)
        return
    p = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if p.returncode or p.stdout.strip() != PINNED:
        raise SystemExit('REFUSED: samples must be the pinned Git checkout')
    p = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=True)
    if p.stdout.strip():
        raise SystemExit('REFUSED: samples checkout has changes')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board', required=True, choices=B.CONFIG)
    ap.add_argument('--samples', type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--only', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--vector-host-compat', action='store_true', help='provide only the host ceil_div helper missing from CUDA 12.1; pinned source unchanged')
    a = ap.parse_args()
    man = manifest(a.board)
    builds = [e for e in man['builds'] if not a.only or e['id'] == a.only]
    if not builds:
        raise SystemExit('REFUSED: unknown build id')
    if a.dry_run:
        print(json.dumps(dict(man, builds=builds), indent=1, sort_keys=True))
        return
    if not a.samples:
        raise SystemExit('REFUSED: --samples required for compilation')
    verify_samples(a.samples)
    if a.work.exists():
        raise SystemExit('REFUSED: work directory already exists; never overwrite')
    a.work.mkdir(parents=True)
    os.nice(19)
    ver = subprocess.run([NVCC, '--version'], capture_output=True, text=True, check=True).stdout
    if 'release 12.1' not in ver:
        raise SystemExit('REFUSED: expected CUDA 12.1')
    os.environ['HARNESS_CUDA_ARCH'], os.environ['HARNESS_NVCC'] = man['arch'], NVCC
    for rel in ['tiresias/app_runners', 'tiresias/framework/unseen_operators/runners', 'energy_harness']:
        sys.path.insert(0, str(B.REPO / rel))
    def expand(s):
        return str(s).replace('<REPO>', str(B.REPO)).replace('<SAMPLES>', str(a.samples)).replace('<WORK>', str(a.work))
    results = []
    for e in builds:
        wd = a.work / e['id']
        wd.mkdir()
        result = dict(id=e['id'], status='failed')
        old_prepend = os.environ.get('NVCC_PREPEND_FLAGS')
        try:
            if e['id'] == 'copy_runner' and a.vector_host_compat:
                header = wd / 'host_compat' / 'cuda' / 'cmath'
                header.parent.mkdir(parents=True)
                header.write_text(VECTOR_HOST_HEADER)
                include = str(header.parent.parent)
                if any(c.isspace() for c in include):
                    raise RuntimeError('compatibility include path contains whitespace')
                os.environ['NVCC_PREPEND_FLAGS'] = (old_prepend or '') + ' -I' + include
                result['host_compat'] = dict(header_sha256=B.sha(header), NVCC_PREPEND_FLAGS=os.environ['NVCC_PREPEND_FLAGS'], scope='pinned sample host ceil_div; device source byte unchanged')
            if e.get('python_entry'):
                # build_binary consumes explicit nvcc; it does not run the live
                # architecture guard in find_nvcc or execute its resulting binary.
                mod = __import__(e['id'])
                mod.build_binary(a.samples, wd, NVCC)
            else:
                for step in e['steps']:
                    cmd = [expand(s) for s in step['cmd']]
                    r = subprocess.run(cmd, capture_output=True, text=True, cwd=wd)
                    (wd / 'build.log').write_text(' '.join(cmd) + '\n' + r.stdout + r.stderr)
                    r.check_returncode()
            binary = Path(expand(e['binary']))
            result['binary_sha256'] = B.sha(binary)
            for kind, flag in [('sass', '-sass'), ('res', '-res-usage')]:
                dst = a.work / 'compiled' / e['outputs'][kind]
                dst.parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([str(Path(NVCC).with_name('cuobjdump')), flag, '-arch', man['arch'], str(binary)], capture_output=True, text=True, check=True)
                if not r.stdout.strip():
                    raise RuntimeError('empty compiled dump')
                dst.write_text(r.stdout)
                result[kind + '_sha256'] = B.sha(dst)
            result['status'] = 'built'
        except Exception as ex:
            result['reason'] = str(ex)
        finally:
            if old_prepend is None:
                os.environ.pop('NVCC_PREPEND_FLAGS', None)
            else:
                os.environ['NVCC_PREPEND_FLAGS'] = old_prepend
        results.append(result)
        (a.work / 'build_results.json').write_text(json.dumps(dict(man, nvcc_version=ver, results=results), indent=1, sort_keys=True) + '\n')
        print(e['id'], result['status'], result.get('reason', ''), flush=True)
    if any(r['status'] != 'built' for r in results):
        raise SystemExit('INCOMPLETE: driver failures retained in build_results.json')


if __name__ == '__main__':
    main()
