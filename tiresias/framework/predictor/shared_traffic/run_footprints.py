"""Derive per-kernel grid-wide memory footprints (footprint.py) for every cell of an evaluation set by re-interpreting the sampled blocks with the existing static pipeline.
CPU only. One JSON per cell in footprints/<set>/ (resumable). Sets: classic (unseen kernels), e (scalar product, Walsh), f, g, h, d.

    python3 run_footprints.py --set f --jobs 6 [--only substring]
The pipelines are imported and called unchanged (nothing under predictor/ is edited): UP.binding, UP.trace_kernel for the UP-based sets; fresh_d_lib.analyse with a hook
on phases_unique.kernel_entry for set D. Sampled blocks: the pipeline's own choice (first and last block) for exactly-uniform kernels; the grid corners for kernels that use block classes."""
import argparse, csv, itertools, json, math, multiprocessing as mp, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(HERE))
import footprint as FP

EVAL = {r['cell_id'] for r in csv.DictReader(open(SR.parent.parent / 'evaluation/results/eval_cells.csv'))}
SETS = {'classic': ('unseen_kernels', None, None), 'e': ('fresh_e', 'fresh_e_lib', 'cells_e'), 'f': ('fresh_f', 'fresh_f_lib', 'cells_f'),
        'g': ('fresh_g', 'fresh_g_lib', 'cells_g'), 'h': ('fresh_h', 'fresh_h_lib', 'cells_h'), 'd': ('fresh_d', 'fresh_d_lib', None)}


def corners(grid):
    ids = set()
    for xyz in itertools.product(*[sorted({0, g - 1}) for g in grid]):
        ids.add(xyz[0] + xyz[1] * grid[0] + xyz[2] * grid[0] * grid[1])
    return sorted(ids)


def load(setname):
    d, lib, cells = SETS[setname]
    sys.path.insert(0, str(SR / d))
    if setname == 'classic':
        sys.path.insert(0, str(SR / 'unseen_kernels')); import unseen_pipeline as UP, cells as CE
        return UP, CE.define_cells
    if setname == 'd':
        import fresh_d_lib as L, build_fresh_d as BD
        return (L, BD), None
    L = __import__(lib); CE = __import__(cells)
    return L.UP, CE.define_cells


def cell_up(UP, cell, hw):
    out = []
    for kern in cell['kernels']:
        kid = kern['kid']; consts, grid, block, prov = UP.binding(kid, kern)
        nblocks = math.prod(grid)
        sub = dict(kern, sample_blocks=corners(grid)) if kern.get('sampling') == 'classes' else kern
        sigs, obs = UP.trace_kernel(kid, sub, consts, grid, block)
        fp = FP.kernel_footprint(UP.U, obs, nblocks); fp['kernel_id'] = kid; fp['sampled_block_ids'] = UP.sample_blocks(grid, sub)
        out.append(fp)
    return out


G = {}


def one(cell):
    s, outdir, env, hw = G['s'], G['outdir'], G['env'], G['hw']
    path = outdir / (cell['cell_id'].replace('/', '__') + '.json')
    if path.exists(): return cell['cell_id'], 'skip'
    t = time.time()
    try:
        if s == 'd': kernels = cell_d(env, cell, hw)
        else: kernels = cell_up(env, cell, hw)
        doc = dict(cell_id=cell['cell_id'], status='ok', kernels=kernels)
    except (FP.FootprintRefusal, Exception) as ex:   # a refusal is recorded with its reason, never replaced by a default
        doc = dict(cell_id=cell['cell_id'], status='refused', reason='%s: %s' % (type(ex).__name__, ex), kernels=[])
    doc['seconds'] = round(time.time() - t, 1)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
    return cell['cell_id'], '%s %.0fs' % (doc['status'], doc['seconds'])



def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', required=True); ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--only', default='')
    a = ap.parse_args(); s = a.set; outdir = HERE / 'footprints' / s; outdir.mkdir(parents=True, exist_ok=True)
    env, define = load(s)
    if s == 'd':
        L, BD = env; hw, l2 = L.l2_bytes_from_ground_truth(); cells = L.define_cells(l2)
    else:
        UP = env; hw = UP.X.load_hardware(UP.X.read_text(UP.X.GROUND_TRUTH)); cells = define(hw)
    cells = [c for c in cells if c['cell_id'] in EVAL and a.only in c['cell_id']]
    print(s, len(cells), 'cells', flush=True)

    G.update(s=s, outdir=outdir, env=env, hw=hw)
    cells.sort(key=lambda c: -math.prod(c['kernels'][-1]['grid']) if 'kernels' in c else 0)
    with mp.get_context('fork').Pool(a.jobs) as pool:
        for cid, msg in pool.imap_unordered(one, cells, chunksize=1): print('%-72s %s' % (cid, msg), flush=True)


def cell_d(env, cell, hw):
    L, BD = env; PU = L._pu(); orig = PU.kernel_entry; got = []
    def hook(observers, frozen_kernel, bps):
        got.append((list(observers), observers[0].blocks)); return orig(observers, frozen_kernel, bps)
    PU.kernel_entry = hook
    try:
        out = L.analyse(BD.cell_spec(cell, hw['sm_count']))
    finally:
        PU.kernel_entry = orig
    if not got: raise FP.FootprintRefusal('no first-touch analysis for this cell: ' + str(out['unique'].get('reason')))
    res = []
    for obs, nb in got:
        fp = FP.kernel_footprint(PU, obs, nb); fp['kernel_id'] = 'retained'; res.append(fp)
    return res


if __name__ == '__main__':
    main()
