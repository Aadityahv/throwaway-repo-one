"""Grid-wide unique memory footprint per pointer argument, derived from the same interpreted address traces the existing static analysis uses. CPU only; reads no measured value.

For one kernel launch, the static analysis interprets a few sampled blocks (first and last block, or the grid corners for kernels with border classes) and records
every global address each lane touches. From those traces this module builds, per pointer argument and per direction (reads, writes):

  per block   the set of 32-byte sectors the block touches for the first time (reads: not read or written by the same block in an earlier barrier phase;
              writes: distinct sectors written). This is the definition of `first_touch` in reuse/phases_unique.py.
  naive       grid blocks x mean over the sampled blocks of that count (every block touches its own sectors; no sharing between blocks).
  span        (highest - lowest sector) + 1 over the union of the sampled blocks' sectors: the touched address range between the extreme blocks.
  union       distinct sectors in the union of the sampled blocks (a lower bound on the footprint).
  footprint   max(union, min(naive, span)).

The footprint is the number of distinct sectors the whole grid touches in that buffer, assuming (a) block-to-address maps are monotone enough that the first and last
sampled blocks bound the touched range (true for every tiled, streaming, row-wise and reduction access pattern of the evaluation; stated in DESIGN.md), and (b) when the
blocks share sectors the touched range is dense (no gaps inside the span). A pointer whose naive count exceeds its footprint is `reused`: several blocks read the same
sectors, which the L2 serves after the first fetch.

`reuse_working_set_bytes` = the summed footprint of the reused read pointers. The runtime model refuses a kernel whose reused working set exceeds the L2 capacity
(HARDWARE_GROUND_TRUTH.md), because the assumption that L2 captures inter-block reuse then no longer holds.

Nothing here is tuned; a kernel with an unknown (data-dependent) address is refused.
"""
import collections
import math


class FootprintRefusal(RuntimeError):
    pass


def block_sets(U, obs):
    """-> (reads, writes): dicts pointer index -> set of first-touch read sectors / distinct write sectors for one interpreted block."""
    C = U.C
    phases = U.per_phase_sets(obs)
    if any(p['unknown'] for p in phases):
        raise FootprintRefusal('data-dependent global address in a sampled block; footprint unknown')
    cum = set(); ft = set(); wr = set()
    for p in phases:
        ft |= p['read'] - cum
        wr |= p['write']
        cum = cum | p['read'] | p['write']
    def bucket(sectors):
        out = collections.defaultdict(set)
        for g in sectors:
            out[(g * 32 - C.PTR_BASE0) // C.PTR_STRIDE].add(g)
        return out
    return bucket(ft), bucket(wr)


def _estimate(per_block_sets, nblocks):
    """per_block_sets: list (one per sampled block) of sector sets for one pointer and direction."""
    n = [len(s) for s in per_block_sets]
    union = set().union(*per_block_sets) if per_block_sets else set()
    if not union:
        return dict(sampled_blocks=len(per_block_sets), mean_per_block=0.0, union=0, span=0, naive=0.0, footprint=0, reused=False)
    mean = sum(n) / len(n); span = max(union) - min(union) + 1; naive = nblocks * mean
    footprint = max(len(union), min(naive, span))
    return dict(sampled_blocks=len(per_block_sets), mean_per_block=mean, union=len(union), span=span, naive=naive, footprint=footprint,
                reused=is_reused(naive, footprint, nblocks))


def is_reused(naive, footprint, nblocks):
    """Blocks share sectors when the no-sharing count exceeds the footprint by at least one sector per block on average. (Border blocks of a kernel with no sharing make the naive count and
    the span differ by a few sectors in total; that is not reuse.)"""
    return bool(naive - footprint >= nblocks)


def reuse_working_set_bytes(kfp):
    """Summed read footprint of the pointers whose sectors several blocks share (recomputed from the stored per-pointer numbers)."""
    return 32 * sum(v['read']['footprint'] for v in kfp['pointers'].values() if is_reused(v['read']['naive'], v['read']['footprint'], kfp['blocks']))


def kernel_footprint(U, observers, nblocks):
    """Footprint of one kernel launch from the observers of its sampled blocks."""
    if not observers: raise FootprintRefusal('no sampled block')
    per = [block_sets(U, o) for o in observers]
    ptrs = sorted({p for r, w in per for p in list(r) + list(w)})
    out = dict(blocks=nblocks, sampled_blocks=len(observers), pointers={})
    for p in ptrs:
        out['pointers'][str(p)] = dict(read=_estimate([r.get(p, set()) for r, _ in per], nblocks), write=_estimate([w.get(p, set()) for _, w in per], nblocks))
    rd = sum(v['read']['footprint'] for v in out['pointers'].values()); wr = sum(v['write']['footprint'] for v in out['pointers'].values())
    out['read_footprint_sectors'] = rd; out['write_footprint_sectors'] = wr
    out['read_naive_sectors'] = sum(v['read']['naive'] for v in out['pointers'].values())
    out['write_naive_sectors'] = sum(v['write']['naive'] for v in out['pointers'].values())
    out['reuse_working_set_bytes'] = reuse_working_set_bytes(out)
    out['first_touch_sectors_per_sampled_block'] = [sum(len(s) for s in r.values()) for r, _ in per]
    return out


def wave_working_set_bytes(kfp, wave_blocks):
    """Per-wave shared-read working set (DESIGN.md Amendment 2): for each shared read pointer, min(footprint, wave blocks x mean first-touch sectors per block), summed, in bytes."""
    tot = 0.0
    for v in kfp['pointers'].values():
        r = v['read']
        if is_reused(r['naive'], r['footprint'], kfp['blocks']): tot += min(r['footprint'], wave_blocks * r['mean_per_block'])
    return 32 * tot
