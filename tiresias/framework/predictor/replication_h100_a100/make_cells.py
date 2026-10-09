"""Generate the complete A100/H100 intended inventory; CPU only, no labels."""
import argparse
import collections
import json
from pathlib import Path

import board as B

EXPECTED = {'samples': 76, 'ml': 40, 'tensor': 16, 'validation': 12, 'prospective': 24}
SOURCES = {
    'samples': ['unseen_kernels/frozen/cells_unseen.json', 'fresh_e/fresh_cells_e.json', 'fresh_d/fresh_cells_d.json'],
    'ml': ['fresh_f/fresh_cells_f.json'], 'tensor': ['fresh_g/fresh_cells_g.json', 'fresh_h/fresh_cells_h.json'],
    'prospective': ['prospective_test/cells_prosp.json']}


def key(c):
    # Kernel classes distinguish variants. Transform stage repetition changes
    # with resized dimensions, while the original launch-building rule stays.
    kernels = c.get('kernels', c.get('kernel_launches'))
    kids = ['d_' + c['kernel']] if kernels is None else [k['kid'] for k in kernels]
    return c['operator_id'], c['regime'], c['candidate_id'], tuple(dict.fromkeys(kids))


def blackwell_sources(mod):
    result = {}
    for group, paths in SOURCES.items():
        rows = []
        for rel in paths:
            rows.extend(json.loads((B.SR / rel).read_text())['cells'])
        result[group] = rows
    result['validation'], _ = mod._blackwell_cells()
    return result


def documents(label, out=None):
    mod = B.cells_module(label, out)
    docs = mod.build()
    source = blackwell_sources(mod)
    refs = {}
    for group, rows in source.items():
        refs[group] = {key(c): c for c in rows}
        if len(refs[group]) != EXPECTED[group]:
            raise ValueError('incomplete/duplicate Blackwell source inventory for ' + group)
    all_ids = []
    for group, doc in docs.items():
        if doc['unplaceable'] or len(doc['cells']) != EXPECTED[group]:
            raise ValueError('target inventory incomplete: %s/%s: %s' % (label, group, doc['unplaceable']))
        doc['derivation']['rule'] = ('Unchanged complete-suite geometry/shape functions from the Ada replication and H100 generators, explicitly bound to ' + label + "'s verified hardware; source-cell crosswalk below. No labels read.")
        doc['derivation']['l1_tier'] = 'none: no verified target L1 capacity/residency gate'
        for c in doc['cells']:
            original = refs[group].get(key(c))
            if original is None:
                raise ValueError('no exact Blackwell candidate/stage mapping: ' + c['cell_id'])
            c['blackwell_source_cell_id'] = original['cell_id']
            if '_dev' in c:
                c['_dev']['geometry_source'] = 'complete-suite generators bound to ' + label + ' hardware'
            all_ids.append(c['cell_id'])
    if len(all_ids) != 168 or len(set(all_ids)) != 168:
        raise ValueError('target inventory is not exactly 168 unique cells')
    return mod, docs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board', required=True, choices=B.CONFIG)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    mod, docs = documents(a.board, a.out)
    out = mod.HERE
    if not a.check:
        out.mkdir(parents=True, exist_ok=True)
    for g, d in docs.items():
        path = out / mod.FILES[g]
        contents = mod.dump(d).encode()
        if a.check:
            if not path.is_file() or path.read_bytes() != contents:
                raise SystemExit('REFUSED: manifest not reproduced: ' + str(path))
        else:
            path.write_bytes(contents)
    sources = {p: B.sha(B.SR / p) for paths in SOURCES.values() for p in paths}
    sources.update({str(p.relative_to(B.SR)): B.sha(p) for p in [B.ADA / 'make_cells_ada.py', B.ADA / 'board_ada.py', B.SR / 'h100_eval_freeze/make_cells_h100.py', B.SR / 'h100_eval_freeze/ml_sets/make_cells_ml.py', B.SR / 'shared_traffic/validation/cells_validation.py']})
    inventory = dict(schema='cluster_replication_inventory/1', board=a.board, arch=mod.B.ARCH,
                     intended_cells=168, group_counts=EXPECTED, source_sha256=sources,
                     cell_files_sha256={mod.FILES[g]: B.sha(out / mod.FILES[g]) for g in docs},
                     tier_counts=dict(collections.Counter(c['tier'] for d in docs.values() for c in d['cells'])))
    contents = mod.dump(inventory).encode()
    path = out / 'inventory.json'
    if a.check:
        if path.read_bytes() != contents:
            raise SystemExit('REFUSED: inventory hashes not reproduced')
    else:
        path.write_bytes(contents)
    print(a.board, '168 intended cells:', inventory['tier_counts'], 'byte-reproduced' if a.check else 'written')


if __name__ == '__main__':
    main()
