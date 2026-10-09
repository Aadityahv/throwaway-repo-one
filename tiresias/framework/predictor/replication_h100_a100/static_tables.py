"""Target-board static tables through the existing complete-suite analyser.

CPU only; no measured labels. Runs serially so each isolated target keeps its
board binding. Missing vector-add SASS delays those four cells only. Every cell
remains in the merged inventory, including compiler and interpreter refusals.
"""
import argparse
import json
from pathlib import Path

import board as B


def setup(label):
    cells = B.cells_module(label)
    mod = B.load_module('_cluster_static_' + label, B.ADA / 'static_ada.py')
    mod.B, mod.MA, mod.HERE = cells.B, cells, cells.HERE
    mod.OUTDIR, mod.BUILD = cells.HERE / 'static', cells.HERE / 'build'
    return mod


def cell_sass_ready(mod, cell):
    origin = mod.origin_of(cell)
    if origin != 'base':
        return mod.sass_state()[origin][0]
    # The previous all-base gate also blocked unrelated classic kernels when
    # vectorAdd alone was missing. Keep that candidate pending while others run.
    stems = ['vectorAdd'] if any(k['kid'] == 'd_vecAdd' for k in cell['kernels']) else [s for s in mod.BASE_STEMS if s != 'vectorAdd']
    return all((mod.B.SASS_BASE / folder / (s + ext)).is_file()
               for s in stems for folder,ext in [('sass','.sass'),('log','.res')])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board', required=True, choices=B.CONFIG)
    ap.add_argument('--only', default='')
    ap.add_argument('--merge-only', action='store_true')
    a = ap.parse_args()
    mod = setup(a.board)
    if not a.merge_only:
        for group in mod.MA.GROUPS:
            for cell in mod.MA.load_group(group)['cells']:
                if a.only and a.only not in cell['cell_id']:
                    continue
                if not cell_sass_ready(mod, cell) or mod.cache_path(group, cell['cell_id']).exists():
                    continue
                print('%s %s' % mod.build_one((group, cell['cell_id'])), flush=True)
        if a.only:
            return
    hw,_ = mod.B.read_ada_hardware()
    rows = [mod.merge(g, hw) for g in mod.MA.GROUPS]
    # Reuse the analyser outputs, with a descriptive target-board status report.
    (mod.HERE / 'static_status.json').write_text(json.dumps(dict(board=a.board,groups=rows),indent=1,sort_keys=True)+'\n')
    for row in rows:
        print(a.board, row['group'], row['cells'], 'cells;', row['supported'], 'supported;', row['pending'], 'pending;', len(row['unsupported']), 'refused')


if __name__ == '__main__':
    main()
