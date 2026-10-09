"""Synthetic-data test: the regression recovers known constants. Synthetic only, never evidence."""
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_microbench as F


def make(latency, issue, bar_per_warp, c_loop=3.0, c0=40.0):
    rows, table = {}, {}
    for f, S, t, b, u, k in itertools.product(range(8), (1, 4), (32, 1024), (1, 188), (1, 4, 16), (127, 509)):
        ops = F.OPS_PER_STEP.get(f, 1); w = t // 32
        if f == 0: slope = bar_per_warp * w
        elif S == 1: slope = latency[f] * ops
        else: slope = issue[f] * w * S * ops
        slot = f'{f}/{S}/{t}/{b}/{u}/{k}'
        rows[slot] = dict(kind='compute', family=f, streams=S, threads=t, blocks=b, unroll=u, loops=k)
        table[slot] = (c0 + k * c_loop + k * u * slope, 0.0)
    return rows, table


class FitTests(unittest.TestCase):
    def test_recovers_constants(self):
        latency = {f: 4.0 + f for f in range(1, 8)}; issue = {f: 0.25 * (1 + f % 3) for f in range(1, 8)}
        rows, table = make(latency, issue, bar_per_warp=7.0)
        out = F.fit(table, rows)
        for f in range(1, 8):
            name = F.FAMILY_NAMES[f]
            self.assertAlmostEqual(out['dependent_latency_cycles'][name], latency[f], places=6)
            self.assertAlmostEqual(out['issue_cycles_per_warp_instruction_per_sm'][name], issue[f], places=6)
            self.assertAlmostEqual(out['loop_overhead_cycles_per_iteration'][name], 3.0, places=6)
        self.assertAlmostEqual(out['barrier_latency_cycles_by_warps']['1'], 7.0, places=6)
        self.assertAlmostEqual(out['barrier_latency_cycles_by_warps']['32'], 7.0 * 32, places=4)
        self.assertLess(max(out['fit_quality_max_relative_residual'].values()), 1e-9)

    def test_underdetermined_group_is_skipped(self):
        rows, table = make({f: 4.0 for f in range(1, 8)}, {f: 0.5 for f in range(1, 8)}, 5.0)
        for slot in [s for s in rows if rows[s]['family'] == 2 and rows[s]['threads'] == 32 and rows[s]['streams'] == 1 and rows[s]['blocks'] == 1 and rows[s]['loops'] == 509]:
            del table[slot]
        out = F.fit(table, rows)
        # Three of six (loops, unroll) points remain: under-determined, so no constant is reported (never guessed).
        self.assertNotIn('fp32_add', out['dependent_latency_cycles'])
        self.assertIn('fp32_fma', out['dependent_latency_cycles'])


if __name__ == '__main__':
    unittest.main()
