"""Point the existing GPU harnesses at the 12 validation cells inside a STAGING copy of the repository (never the committed tree). The harnesses (fresh_f/run_f.py with FRESH_SET=f|g|h,
fresh_e/gpu/run_e.py, unseen_kernels/gpu/run_unseen.py) take their cell lists from fresh_cells_<set>.json or from `cells_e.py` / `cells.py: define_cells`; this script writes the validation
cells there (the same cell dictionaries the static tables were built from). Nothing else in the harnesses changes.
    python3 stage_cells.py --tree ~/stage/tiresias"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--tree', type=Path, required=True); a = ap.parse_args()
SR = a.tree / 'tiresias/framework/predictor'
tab = lambda lib: [v['cell'] for _, v in sorted(json.loads((HERE / 'tables' / ('%s.json' % lib)).read_text()).items()) if 'cell' in v]
for lib, d in (('f', 'fresh_f'), ('g', 'fresh_g'), ('h', 'fresh_h')):
    p = SR / d / ('fresh_cells_%s.json' % lib); old = json.loads(p.read_text()); old['cells'] = tab(lib); old['staged_validation_cells'] = True
    p.write_text(json.dumps(old, indent=1, sort_keys=True) + '\n'); print('wrote', p, len(old['cells']))
for lib, p in (('e', SR / 'fresh_e/cells_e.py'), ('classic', SR / 'unseen_kernels/cells.py')):
    cells = tab(lib)
    p.write_text('"""STAGED validation cells (written by shared_traffic/validation/stage_cells.py); replaces the evaluation cell definitions in the staging copy only."""\nimport json\nCELLS = json.loads(%r)\n\ndef define_cells(hw):\n    return [dict(c, _dev=dict(c.get("_dev") or {})) for c in CELLS]\n' % json.dumps(cells, sort_keys=True))
    print('wrote', p, len(cells))
