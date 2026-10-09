"""Set B copy of ../fresh/phases_unique_fresh.py (that file is not edited). Differences: set B file names, and the copy-kernel
cols % 32 == 0 requirement is dropped (small regime cols 112); periodicity is still checked.
First-touch unique-sector analysis for the 32 fresh set B cells (CPU only; no measured runtime or energy read).

Same method and output schema as ../reuse/phases_unique.py (imported, not edited). The interpretation is the
SAME corrected one as phases_fresh_b.json: IMAD.HI.U32 with the 64-bit addend pair
(divider_fix.corrected_interp_class), and the padded copy kernel (kernel k1) analysed by block class
(copy_kernel_phases_by_block_class logic of build_fresh.py), each class weighted by its grid population.
Every recomputed per-phase read sectors, write sectors and lines must equal the corrected table exactly,
otherwise the cell is refused.

Copy kernel: per-block first-touch counts differ between block classes (row alignment), so the per-block fields of
that kernel are grid-population-weighted means (may be fractional) with min/max reported; grid totals are exact sums.
Run: python3 phases_unique_fresh.py   (several minutes)
"""
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / 'fresh'))
sys.path.insert(0, str(SR / 'reuse'))
import phases_unique as U  # noqa: E402  (imported, never edited)
import divider_fix  # noqa: E402

PH, C, A = U.PH, U.C, U.PH.A
assert PH.P is A.P, 'phases module must share the interpreter module with the adapter'
SECTOR = U.SECTOR


def copy_kernel_entry(kern, cell, frozen_kernel, blocks_per_sm):
    """Block-class analysis of the padded copy kernel, one phase, grid-weighted; refuses on any mismatch."""
    constants, grid, block = PH.bind_pytorch('k1', kern, cell)
    rows, cols = cell['input']['shape']
    nblocks = math.prod(grid)
    C.require((rows * cols) % 256 == 0, 'copy kernel tail block')  # set B: cols % 32 == 0 not required (cols 112); periodicity is checked below
    period = (cols * 32) // math.gcd(cols * 32, 256)
    threads = math.prod(block)
    sites = A.P.parse((SR / 'libtorch_sm120' / 'k1.isolated.sass').read_text())

    def one(b):
        obs = U.CapturingObserver(threads)
        obs.block_x = block[0]
        interp = A.P.Interp(C.D, ext=True, fchk_fast_path=True, trace=obs.pytorch)

        def coords(l):
            return {k: A.P.V.exact(v) for k, v in {'SR_CTAID.X': b % grid[0], 'SR_CTAID.Y': b // grid[0], 'SR_CTAID.Z': 0,
                                                   'SR_CgaCtaId': 0, 'SR_TID.X': l % block[0], 'SR_TID.Y': l // block[0],
                                                   'SR_TID.Z': 0, 'SR_LANEID': l % 32}.items()}
        interp.run_block(sites, {k: A.P.V.exact(v) for k, v in constants.items()}, coords, threads)
        obs.finish(1)
        return obs

    reps = [one(b) for b in range(min(period, nblocks))]
    an = [U.analyse_block(o) for o in reps]
    if any(a['unknown'] for a in an):
        return dict(status='unsupported', reason='data-dependent global address in copy kernel')
    # periodicity check on the first-touch quantities (the table builder checks the full signature)
    for b in sorted({period, period + 1, nblocks - 1}):
        if b < nblocks:
            ab = U.analyse_block(one(b))
            if [(r['req_read'], r['req_write'], r['lines'], r['unique_in_phase'], r['first_touch']) for r in ab['rows']] != \
               [(r['req_read'], r['req_write'], r['lines'], r['unique_in_phase'], r['first_touch']) for r in an[b % period]['rows']]:
                return dict(status='refused', reason='block %d does not reproduce class %d: periodicity not proved' % (b, b % period))
    counts = [nblocks // period + (1 if c < nblocks % period else 0) for c in range(len(reps))]
    fph = frozen_kernel['phases']
    if any(len(a['rows']) != len(fph) for a in an):
        return dict(status='refused', reason='recomputed phase count differs from corrected table')
    phases = []
    for i, f in enumerate(fph):
        rr = [a['rows'][i] for a in an]
        tot = {k: sum(r[k] * n for r, n in zip(rr, counts)) for k in ('req_read', 'req_write', 'lines', 'first_touch', 'unique_in_phase',
                                                                       'readback_of_prior_phase_writes', 'read_also_written_same_phase')}
        if tot['req_read'] != f['read_sectors'] or tot['req_write'] != f['write_sectors'] or tot['lines'] != f['lines']:
            return dict(status='refused', reason='recomputed read/write sectors or lines differ from phases_fresh_b.json')
        phases.append(dict(
            index=f['index'], repetitions=f['repetitions'], read_sectors=f['read_sectors'], write_sectors=f['write_sectors'], lines=f['lines'],
            read_sectors_total_requested=f['read_sectors'],
            read_sectors_unique_in_phase_per_block=tot['unique_in_phase'] / nblocks,
            read_sectors_first_touch_per_block=tot['first_touch'] / nblocks,
            read_sectors_first_touch=tot['first_touch'],
            read_sectors_first_touch_per_block_min=min(r['first_touch'] for r in rr),
            read_sectors_first_touch_per_block_max=max(r['first_touch'] for r in rr),
            read_sectors_readback_of_prior_phase_writes_per_block=tot['readback_of_prior_phase_writes'] / nblocks,
            read_sectors_also_written_same_phase_per_block=tot['read_also_written_same_phase'] / nblocks))
    w = lambda f: sum(f(a) * n for a, n in zip(an, counts)) / nblocks
    return dict(
        status='ok', blocks=nblocks, blocks_per_sm=blocks_per_sm, phases=phases,
        per_block_values_are_grid_weighted_means_over_block_classes=dict(period_blocks=period, classes=len(reps)),
        total_requested_read_sectors_per_block=w(lambda a: sum(r['req_read'] for r in a['rows'])),
        unique_read_sectors_per_block=w(lambda a: len(a['cum_read'])),
        unique_read_bytes_per_block=w(lambda a: len(a['cum_read'])) * SECTOR,
        unique_read_or_written_bytes_per_block=w(lambda a: len(a['cum_touched'])) * SECTOR,
        first_touch_read_sectors_per_block=w(lambda a: sum(r['first_touch'] for r in a['rows'])),
        read_sectors_shared_across_blocks_note=dict(note='not computed for the block-class copy kernel (classes are not neighbouring blocks)'))


def main():
    corrected = json.loads((HERE / 'phases_fresh_b.json').read_text())['rows']
    feats = {r['cell_id']: r for r in json.loads((HERE / 'features_fresh_b.json').read_text())['rows']}
    dispatch = json.loads((HERE / 'dispatch_trace_fresh_b.json').read_text())
    A.P.Interp = divider_fix.corrected_interp_class(A.P)   # same interpretation as the corrected table
    out, cache = {}, {}
    for cell in dispatch['cells']:
        cid = cell['cell_id']
        f = corrected[cid]
        bps = feats[cid]['occupancy'].get('blocks_per_sm')
        if f['status'] != 'conditional_static_phases':
            out[cid] = dict(status='unsupported', reason='corrected phase table unsupported: ' + str(f.get('reason')), kernels=[])
            print(cid, out[cid]['reason'], flush=True); continue
        entries = []
        try:
            for kern, fk in zip(cell['kernels'], f['kernels']):
                kid = A.kernel_id(kern)
                assert fk['kernel_id'] == kid
                key = json.dumps([kid, kern, cell['input'] if kid == 'k1' else None, bps, [p['read_sectors'] for p in fk['phases']]], sort_keys=True)
                if key not in cache:
                    if kid == 'k1':
                        cache[key] = copy_kernel_entry(kern, cell, fk, bps)
                    else:
                        del U.CAPTURED[:]
                        PH.pytorch(kid, kern, cell)
                        cache[key] = U.kernel_entry(list(U.CAPTURED), fk, bps)
                e = dict(cache[key]); e['kernel_id'] = kid
                entries.append(e)
        except (C.Refusal, A.P.Refusal) as ex:
            out[cid] = dict(status='unsupported', reason=str(ex), kernels=[]); print(cid, out[cid]['reason'], flush=True); continue
        ok = all(e['status'] == 'ok' for e in entries)
        bad = next((e for e in entries if e['status'] != 'ok'), None)
        out[cid] = dict(status='ok' if ok else bad['status'], reason=None if ok else bad['reason'],
                        kernels_per_launch=feats[cid].get('kernels_per_launch'),
                        blocks_per_sm_note='features occupancy, cell level (main kernel)' if len(entries) > 1 else 'features occupancy',
                        kernels=entries if ok else [])
        print(cid, out[cid]['status'], out[cid]['reason'] or '', flush=True)
    result = dict(schema='first_touch_unique_sectors/1', sector_bytes=SECTOR,
                  derived_from='tiresias/framework/predictor/fresh_b/phases_fresh_b.json (unchanged)',
                  interpretation='corrected IMAD.HI.U32 (divider_fix.py); copy kernel by block class, grid-weighted',
                  uses_measured_runtime_or_energy=False, rows=out)
    (HERE / 'phases_unique_fresh_b.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(collections.Counter(r['status'] for r in out.values()))
    # every recomputed total equals the phase table (also enforced per cell above: a mismatch refuses the cell)
    assert all(r['status'] == 'ok' for r in out.values()), 'some cells not analysed'
    for cid, r in out.items():
        for ko, kc in zip(r['kernels'], corrected[cid]['kernels']):
            for po, pc in zip(ko['phases'], kc['phases']):
                for f in ('read_sectors', 'write_sectors', 'lines'):
                    assert po[f] == pc[f], (cid, f)


if __name__ == '__main__':
    main()
