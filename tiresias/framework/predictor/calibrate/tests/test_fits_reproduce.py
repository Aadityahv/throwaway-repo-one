"""The packaged fits applied to the committed Blackwell raw microbenchmark outputs must reproduce the committed constants files (same definitions, device facts in place of 188)."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SR = HERE.parent
sys.path.insert(0, str(HERE))
from cal import fits, stages  # noqa: E402

M = SR / 'micro_v3'; C = SR / 'constants'
rows = lambda name: [json.loads(l) for l in (M / name).read_text().splitlines() if l.strip().startswith('{')]
load = lambda name: json.loads((C / name).read_text())
close = lambda a, b, tol=1e-9: abs(a - b) <= tol * max(1.0, abs(b))


class Reproduce(unittest.TestCase):
    def test_stream_curves_reproduce_v3e(self):
        got = fits.fit_stream(rows('out_stream3.jsonl')); ref = load('v3e_constants.json')
        self.assertTrue(close(got['kernel_fixed_overhead_us'], ref['kernel_fixed_overhead_us']))
        for k in ('l2_total_bandwidth_TBps', 'dram_total_bandwidth_TBps'):
            self.assertEqual(got[k]['read_fraction'], ref[k]['read_fraction'])
            for x, y in zip(got[k]['value'], ref[k]['value']): self.assertTrue(close(x, y))

    def test_mlp_reproduces_v3c(self):
        got = fits.fit_mlp(rows('out_mlp.jsonl'), 188); ref = load('v3c_constants.json')
        for t in ('L2', 'DRAM'): self.assertTrue(close(got['service_latency_us'][t], ref['service_latency_us'][t]))

    def test_launch_and_reuse_reproduce_v3(self):
        got = fits.fit_launch_reuse(rows('out_chain.jsonl'), rows('out_reuse.jsonl'), 188); ref = load('v3_constants.json')
        for k in ('intercept_at_one_block', 'per_extra_block'): self.assertTrue(close(got['launch_us_per_kernel'][k], ref['launch_us_per_kernel'][k]))
        self.assertTrue(close(got['l1']['bandwidth_bytes_per_s_per_sm'], ref['l1']['bandwidth_bytes_per_s_per_sm']))
        self.assertEqual(got['l1_reread_l2_fraction_curve']['per_sm_kb'], ref['l1_reread_l2_fraction_curve']['per_sm_kb'])
        # the capacity RULE (derived, not the old literal) must land on the committed 96 KB
        self.assertEqual(got['l1']['capacity_bytes_per_sm'], ref['l1']['capacity_bytes_per_sm'])

    def test_overlap_reproduces_committed_table(self):
        got = fits.fit_overlap(rows('out_overlap.jsonl')); ref = json.loads((C / 'overlap_constants.json').read_text())
        self.assertEqual(got['resident_blocks_per_sm'], ref['resident_blocks_per_sm'])
        for x, y in zip(got['alpha'], ref['alpha']): self.assertAlmostEqual(x, y, places=3)

    def test_smem_rule_matches_measurement(self):
        got = fits.fit_smem(rows('out_smem.jsonl'), rows('out_smem_plain.jsonl'))
        self.assertAlmostEqual(got['floor_cycles'], 2.0, delta=0.05)
        self.assertAlmostEqual(got['degree_slope_cycles'], 1.0, delta=0.05)
        self.assertAlmostEqual(got['plain_64bit_conflict_free'], 2.0, delta=0.1)
        self.assertAlmostEqual(got['plain_128bit_conflict_free'], 4.0, delta=0.1)
        self.assertFalse(got['ffma_interleave_changes_cost'])

    def test_nnls3_recovers_nonnegative_and_clips_negative(self):
        import numpy as np
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        c, _ = fits.nnls3(A, np.array([2.0, 3.0, 5.0, 10.0])); self.assertTrue(np.allclose(c, [2, 3, 5]))
        c, _ = fits.nnls3(A, np.array([2.0, 3.0, -5.0, 0.0])); self.assertTrue((c >= 0).all()); self.assertEqual(c[2], 0.0)

    def test_store_pattern_counts(self):
        self.assertEqual(fits.store_pattern(1), (4.0, 1.0))
        s7, l7 = fits.store_pattern(7); s33, l33 = fits.store_pattern(33)
        self.assertGreater(s7, 4.0); self.assertGreater(s33, s7 - 1e-9); self.assertEqual(l33, 32.0)


class Pipes(unittest.TestCase):
    ARCHIVES = sorted((SR.parent / 'compile_evidence' / 'acquisition_runs').glob('isolated_runtime_blackwell_2026100*'))

    @unittest.skipUnless(ARCHIVES, 'isolated campaign archive not present')
    def test_pipes_reproduce_committed_microbench_constants(self):
        ref = load('microbench_constants_v2.json')
        cells = None
        for arc in self.ARCHIVES:
            if (arc / 'cells').exists() and ref['source']['campaign_dir'].endswith(arc.name): cells = arc / 'cells'; break
        if cells is None: self.skipTest('the campaign directory of the committed constants is not retained locally')
        packet = json.loads((SR / 'repair/next_iteration/prepared_packet_v3.json').read_text()); by = {r['slot']: r for r in packet['rows']}
        table = {s: fits.load_cell(cells / s, by[s]) for s in by if by[s]['kind'] == 'compute'}
        got = fits.fit_pipes(table, by, 188)
        for k in ('dependent_latency_cycles', 'issue_cycles_per_warp_instruction_per_sm', 'barrier_latency_cycles_by_warps'):
            for name, v in ref[k].items(): self.assertTrue(close(got[k][name], v), (k, name, got[k][name], v))
        self.assertTrue(close(got['effective_sm_clock_hz'], ref['effective_sm_clock_hz']))
        # lean grid (the cells the packaged tool runs): same constants, clock within 1%
        lean = {(r['family'], r['streams'], r['threads'], r['blocks'], r['loops'], r['unroll']) for r in stages.pipes_grid(dict(sm_count=188, l2_bytes=134217728)) if r['kind'] == 'compute'}
        sub = {s: v for s, v in table.items() if (by[s]['family'], by[s]['streams'], by[s]['threads'], by[s]['blocks'], by[s]['loops'], by[s]['unroll']) in lean}
        self.assertGreater(len(sub), 100)
        got2 = fits.fit_pipes(sub, by, 188)
        for k in ('dependent_latency_cycles', 'issue_cycles_per_warp_instruction_per_sm', 'barrier_latency_cycles_by_warps'):
            for name, v in ref[k].items(): self.assertTrue(close(got2[k][name], v, 1e-6), (k, name, got2[k][name], v))
        self.assertLess(abs(got2['effective_sm_clock_hz'] / ref['effective_sm_clock_hz'] - 1), 0.01)


if __name__ == '__main__':
    unittest.main()
