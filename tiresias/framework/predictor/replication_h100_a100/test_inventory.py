"""CPU checks for full source coverage and target hardware binding."""
import collections
import json
import pytest

import board as B
import make_cells as M
import build_drivers as D
import static_tables as S


@pytest.mark.parametrize('label,sm,l2,arch', [('a100',108,40*2**20,'sm_80'), ('h100',132,50*2**20,'sm_90')])
def test_full_crosswalk_and_target(label,sm,l2,arch):
    mod, docs = M.documents(label)
    hw,_ = mod.B.read_ada_hardware()
    assert (hw['sm_count'], hw['l2_bytes'],mod.B.ARCH) == (sm,l2,arch)
    original = M.blackwell_sources(mod)
    tiers = collections.Counter()
    for group,doc in docs.items():
        assert len(doc['cells']) == M.EXPECTED[group]
        assert not doc['unplaceable']
        refs = {c['cell_id']: c for c in original[group]}
        assert {c['blackwell_source_cell_id'] for c in doc['cells']} == set(refs)
        for c in doc['cells']:
            assert M.key(c) == M.key(refs[c['blackwell_source_cell_id']])
            tiers[c['tier']] += 1
        assert (mod.HERE / mod.FILES[group]).read_bytes() == mod.dump(doc).encode()
    assert tiers == {'L2':96,'DRAM':72}


@pytest.mark.parametrize('label,arch', [('a100','sm_80'),('h100','sm_90')])
def test_build_commands_use_target_and_unchanged_sources(label,arch):
    man = D.manifest(label)
    assert len(man['builds']) == 12
    assert {e['id'] for e in man['builds']} >= {'driver_ml','driver_prosp','copy_runner','tile_family'}
    for e in man['builds']:
        for s in e['steps']:
            assert s['cmd'][0] == D.NVCC
            assert '-arch='+arch in s['cmd']
            assert not any('sm_89' in v or 'sm_120' in v for v in s['cmd'])
    for rel,digest in man['source_sha256'].items():
        assert B.sha(B.SR / rel) == digest


@pytest.mark.parametrize('label', ['a100','h100'])
def test_missing_vector_dump_only_blocks_vector_cells(label, tmp_path):
    mod = S.setup(label)
    # Exercise the incomplete original base independently of the repaired
    # private composite folder, if one is present.
    mod.B.SASS_BASE = B.SR / ('port_' + label) / 'compiled_cluster_cuda12.1'
    cells = [c for c in mod.MA.load_cells() if mod.origin_of(c) == 'base']
    vector = [c for c in cells if any(k['kid']=='d_vecAdd' for k in c['kernels'])]
    assert len(vector) == 4
    assert all(not S.cell_sass_ready(mod,c) for c in vector)
    assert sum(S.cell_sass_ready(mod,c) for c in cells) == len(cells)-4
