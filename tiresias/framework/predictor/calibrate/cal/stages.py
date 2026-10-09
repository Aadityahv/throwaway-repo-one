"""Stage table and the lean pipe/latency/pointer-chase grid. Every size is derived from device facts (SM count, L2 size), never fixed to one GPU."""
import hashlib
import itertools
import json
import math

THREADS = (32, 64, 128, 256, 384, 512, 768, 1024)
LOOP_POINTS = ((127, 1), (127, 4), (127, 16), (509, 1), (509, 4), (509, 16))      # (loops, unroll): six points per fit group, as in the full isolated grid

# name, program, arguments, expected minutes (measured on Blackwell: the overlap stage takes about ten minutes, everything else under a minute each), what it feeds
STAGES = [
    ('launch_chain', 'micro_launch', ['chain'], 0.1, 'kernel launch gap by grid size'),
    ('launch_reuse', 'micro_launch', ['reuse'], 0.2, 'L1 capacity, L1 bandwidth, re-read L2-share curve'),
    ('stream', 'micro_stream', [], 0.1, 'L2 and DRAM bandwidth by read fraction, fixed kernel overhead'),
    ('mlp', 'micro_mlp', [], 0.1, 'memory-level-parallelism service latency per tier'),
    ('smem_volatile', 'micro_smem', ['volatile'], 0.1, 'shared-memory request cost versus bank-conflict degree (volatile)'),
    ('smem_plain', 'micro_smem', ['plain'], 0.1, 'shared-memory request cost, plain loads of 32, 64 and 128 bits'),
    ('overlap', 'micro_overlap', [], 10, 'phase-overlap fraction by resident blocks per SM'),
    ('store', 'micro_store', [], 0.1, 'store and triad fixtures: launch floor, per-sector and per-line costs'),
    ('pipes', 'micro_pipes', None, 1, 'pipe issue costs, dependent latencies, barrier cost, effective clock, pointer-chase latency'),
    ('tensor', 'micro_tensor', None, 6, 'tensor-core MMA issue cost and dependent latency, and three tensor energy windows (1, 4, 16 warps per SM; 60 s warm-up + 20 s measurement); the tensor energy rate is derived from them and the energy stage fit'),
    ('energy', 'micro_energy', None, 43, 'energy per byte by tier, per instruction class and base power (27 windows of 60 s warm-up + 20 s measurement)'),
]


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':')).encode()


def tier_tables(facts):
    """Pointer-chase table bytes per block for the three tiers: L1 64 KiB; L2: 30% of L2 across all SMs; DRAM: at least 3x L2 across all SMs (and at least 2 MiB per block)."""
    sm, l2 = facts['sm_count'], facts['l2_bytes']
    l2_table = max(65536 + 4096, int(0.3 * l2 / sm) // 4096 * 4096)
    dram_table = max(2 * 1024 * 1024, math.ceil(3 * l2 / sm / 65536) * 65536)
    if dram_table > 8 * 1024 * 1024: raise ValueError('DRAM pointer-chase table %d bytes per block exceeds the 8 MiB limit' % dram_table)
    return dict(L1=65536, L2=l2_table, DRAM=dram_table)


def pipes_grid(facts):
    sm = facts['sm_count']; rows = []
    for family in range(1, 8):
        for (S, t, b) in ((1, 32, 1), (4, 1024, sm)):
            for loops, unroll in LOOP_POINTS:
                rows.append(dict(kind='compute', family=family, streams=S, threads=t, blocks=b, unroll=unroll, loops=loops, table_bytes=0, width_bytes=4, cache_policy='ca', stride=1))
    for t in THREADS:
        for loops, unroll in LOOP_POINTS:
            rows.append(dict(kind='compute', family=0, streams=1, threads=t, blocks=1, unroll=unroll, loops=loops, table_bytes=0, width_bytes=4, cache_policy='ca', stride=1))
    tables = tier_tables(facts)
    for tier, policy in (('L1', 'ca'), ('L2', 'cg'), ('DRAM', 'cg')):
        rows.append(dict(kind='memory', family=8, streams=2, threads=32, blocks=sm, unroll=1, loops=1, table_bytes=tables[tier], width_bytes=4, cache_policy=policy, stride=1, tier=tier))
    for r in rows:
        r['sms'] = sm; r['slot'] = hashlib.sha256(canonical(dict(r, sms=sm))).hexdigest()[:20]
    if len({r['slot'] for r in rows}) != len(rows): raise ValueError('duplicate pipe cell')
    return rows


def pipes_args(out_dir, r):
    """Argument vector of micro_pipes: <out> kind family streams unroll loops threads blocks table width policy stride sms."""
    return [out_dir, r['kind'], r['family'], r['streams'], r['unroll'], r['loops'], r['threads'], r['blocks'], r['table_bytes'], r['width_bytes'], r['cache_policy'], r['stride'], r['sms']]
