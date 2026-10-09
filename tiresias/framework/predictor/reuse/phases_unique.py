"""First-touch unique-sector analysis on top of the frozen barrier-phase tables.

Reuses phases.py unchanged: its retained() and pytorch() are called as-is and an
Observer subclass (substituted into the phases module namespace) captures the exact
per-lane global addresses the frozen analysis already interpreted. No measured runtime
or energy is read. Frozen read/write sectors and lines are recomputed from the captured
addresses and must equal phases_blackwell.json exactly, otherwise the cell is refused.

Definitions (per block, sector = 32 B granule, buffers are distinct synthetic bases so
addresses never alias across pointer parameters):
  total_requested  sum over warp requests of distinct sectors in the request (frozen).
  unique_in_phase  distinct read sectors over all requests of the phase.
  first_touch      unique_in_phase minus sectors read OR written by the same block in
                   any earlier barrier phase of the same kernel.
Within one phase nothing is subtracted (instruction order across lanes is not recorded),
so a same-phase write-then-read is counted as first-touch (conservative for L1 credit).
Repetition counts are 1 in the frozen table: loop trips are already unrolled in the trace
with their real per-trip addresses.
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / 'coalescing'))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # before exec (dataclass caveat)
    spec.loader.exec_module(mod)
    return mod


PH = _load('phases_frozen_for_reuse', SR / 'phases.py')
C = PH.C
A = PH.A
P = PH.P
SECTOR = 32

CAPTURED = []


class CapturingObserver(PH.Observer):
    def finish(self, blocks):
        self.blocks = blocks
        CAPTURED.append(self)
        return super().finish(blocks)


PH.Observer = CapturingObserver


def buffer_of(addr):
    return (addr - C.PTR_BASE0) // C.PTR_STRIDE


def per_phase_sets(obs):
    """-> list over phases of dict(read=set, write=set, req_read, req_write, lines, unknown)."""
    nph = max(obs.phase) + 1
    out = [dict(read=set(), write=set(), req_read=0, req_write=0, lines=0, unknown=False,
                reads_per_buffer=collections.Counter()) for _ in range(nph)]
    for (warp, ph, pc, k, direction, width), addresses in obs.memory.items():
        o = out[ph]
        if any(a is None for a in addresses):
            o['unknown'] = True
            continue
        gs = {g for a in addresses for g in C.granules(a, width, SECTOR)}
        o['req_' + direction] += len(gs)
        o['lines'] += len({g for a in addresses for g in C.granules(a, width, 128)})
        o[direction] |= gs
    return out


def analyse_block(obs):
    phases = per_phase_sets(obs)
    cum = set()
    rows = []
    cum_read = set()
    for i, o in enumerate(phases):
        if o['unknown']:
            rows.append(None)
            continue
        prior = cum
        ft = o['read'] - prior
        readback = o['read'] & prior_written(phases, i)
        rows.append(dict(
            unique_in_phase=len(o['read']), first_touch=len(ft),
            readback_of_prior_phase_writes=len(readback),
            read_also_written_same_phase=len(o['read'] & o['write']),
            req_read=o['req_read'], req_write=o['req_write'], lines=o['lines'],
            write_unique_in_phase=len(o['write'])))
        cum = cum | o['read'] | o['write']
        cum_read |= o['read']
    return dict(rows=rows, cum_read=cum_read, cum_touched=cum,
                unknown=any(o['unknown'] for o in phases))


def prior_written(phases, i):
    w = set()
    for o in phases[:i]:
        w |= o['write']
    return w


def kernel_entry(observers, frozen_kernel, blocks_per_sm):
    """observers: one per sampled block, in the order phases.py created them."""
    blocks = observers[0].blocks
    an = [analyse_block(o) for o in observers]
    if any(a['unknown'] for a in an):
        return dict(status='unsupported', reason='data-dependent global address in a sampled block; first-touch unknown')
    fph = frozen_kernel['phases']
    for a in an:
        if len(a['rows']) != len(fph):
            return dict(status='refused', reason='recomputed phase count differs from frozen')
        for r, f in zip(a['rows'], fph):
            if (r['req_read'] * blocks != f['read_sectors'] or r['req_write'] * blocks != f['write_sectors']
                    or r['lines'] * blocks != f['lines']):
                return dict(status='refused', reason='recomputed read/write sectors or lines differ from frozen phases_blackwell.json')
    base = an[0]
    for a in an[1:]:
        if [r['first_touch'] for r in a['rows']] != [r['first_touch'] for r in base['rows']] or \
           [r['unique_in_phase'] for r in a['rows']] != [r['unique_in_phase'] for r in base['rows']]:
            return dict(status='refused', reason='first-touch counts differ across sampled blocks; per-block class unproved',
                        per_block_first_touch=[[r['first_touch'] for r in x['rows']] for x in an])
    phases = []
    for i, (r, f) in enumerate(zip(base['rows'], fph)):
        phases.append(dict(
            index=f['index'], repetitions=f['repetitions'],
            read_sectors=f['read_sectors'], write_sectors=f['write_sectors'], lines=f['lines'],
            read_sectors_total_requested=f['read_sectors'],
            read_sectors_unique_in_phase_per_block=r['unique_in_phase'],
            read_sectors_first_touch_per_block=r['first_touch'],
            read_sectors_first_touch=r['first_touch'] * blocks,
            read_sectors_readback_of_prior_phase_writes_per_block=r['readback_of_prior_phase_writes'],
            read_sectors_also_written_same_phase_per_block=r['read_also_written_same_phase']))
    # cross-block sharing: block-0 read sectors also read by at least one other sampled block
    b0 = observers and base['cum_read']
    others = set()
    for a in an[1:]:
        others |= a['cum_read']
    shared = len(b0 & others)
    return dict(
        status='ok', blocks=blocks, blocks_per_sm=blocks_per_sm, phases=phases,
        total_requested_read_sectors_per_block=sum(r['req_read'] for r in base['rows']),
        unique_read_sectors_per_block=len(base['cum_read']),
        unique_read_bytes_per_block=len(base['cum_read']) * SECTOR,
        unique_read_or_written_bytes_per_block=len(base['cum_touched']) * SECTOR,
        first_touch_read_sectors_per_block=sum(r['first_touch'] for r in base['rows']),
        read_sectors_shared_across_blocks_note=dict(
            block0_distinct_read_sectors=len(b0), shared_with_at_least_one_other_sampled_block=shared,
            sampled_blocks_compared=len(an) - 1,
            note='lower bound from 1-2 other sampled blocks; NOT used in first_touch'))


def main():
    frozen = json.loads((SR / 'phases_blackwell.json').read_text())['rows']
    feats = {r['cell_id']: r for r in json.loads((SR / 'features_blackwell.json').read_text())['rows']}
    out = {}

    def bps(cid):
        return feats[cid]['occupancy'].get('blocks_per_sm')

    def finish(cid, entries, note=None):
        ok = all(e['status'] == 'ok' for e in entries)
        bad = next((e for e in entries if e['status'] != 'ok'), None)
        out[cid] = dict(status='ok' if ok else bad['status'], reason=None if ok else bad['reason'],
                        kernels_per_launch=feats[cid].get('kernels_per_launch'),
                        blocks_per_sm_note='features occupancy, cell level (main kernel)' if len(entries) > 1 else 'features occupancy',
                        kernels=entries if ok else [])
        print(cid, out[cid]['status'], out[cid]['reason'] or '', flush=True)

    # retained-binary cells: call frozen retained() unchanged
    manifest_frozen = json.loads((SR / 'coalescing/static_sectors_frozen.json').read_text())
    fz = {r['cell_id']: r for r in manifest_frozen['rows']}
    for corpus, root in C.D.CORPORA.items():
        for row in json.loads((root / 'retention_manifest.json').read_text())['rows']:
            cid = 'blackwell/' + row['operator_id'] + '/' + row['cell']
            f = frozen[cid]
            if f['status'] != 'conditional_static_phases':
                out[cid] = dict(status='unsupported', reason='frozen phase table unsupported: ' + str(f.get('reason')), kernels=[])
                print(cid, 'skip', out[cid]['reason'], flush=True); continue
            del CAPTURED[:]
            try:
                PH.retained(corpus, row, root, fz[cid])
                finish(cid, [kernel_entry(list(CAPTURED), f['kernels'][0], bps(cid))])
            except (C.Refusal, P.Refusal) as ex:
                out[cid] = dict(status='unsupported', reason=str(ex), kernels=[]); print(cid, out[cid]['reason'], flush=True)

    dispatch = json.loads((SR / 'pytorch_dispatch/dispatch_trace.json').read_text())
    cache = {}
    for cell in dispatch['cells']:
        cid = cell['cell_id']; f = frozen[cid]
        if f['status'] != 'conditional_static_phases':
            out[cid] = dict(status='unsupported', reason='frozen phase table unsupported: ' + str(f.get('reason')), kernels=[])
            print(cid, 'skip', out[cid]['reason'], flush=True); continue
        entries = []
        try:
            for kern, fk in zip(cell['kernels'], f['kernels']):
                kid = A.kernel_id(kern)
                assert fk['kernel_id'] == kid
                key = json.dumps([kid, kern, cell['input'] if kid == 'k1' else None], sort_keys=True)
                if key not in cache:
                    del CAPTURED[:]
                    PH.pytorch(kid, kern, cell)
                    cache[key] = list(CAPTURED)
                e = kernel_entry(cache[key], fk, bps(cid)); e['kernel_id'] = kid
                entries.append(e)
            finish(cid, entries)
        except (C.Refusal, P.Refusal) as ex:
            out[cid] = dict(status='unsupported', reason=str(ex), kernels=[]); print(cid, out[cid]['reason'], flush=True)
    for cid in frozen:
        out.setdefault(cid, dict(status='unsupported', reason='frozen phase table unsupported: ' + str(frozen[cid].get('reason')), kernels=[]))
    result = dict(schema='first_touch_unique_sectors/1', sector_bytes=SECTOR,
                  derived_from='tiresias/framework/predictor/phases_blackwell.json (frozen, unchanged)',
                  uses_measured_runtime_or_energy=False, rows=out)
    (HERE / 'phases_unique_blackwell.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(collections.Counter(r['status'] for r in out.values()))


if __name__ == '__main__':
    main()
