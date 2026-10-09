"""Tests of the shared memory-traffic candidate (v3j runtime model, footprint estimator, traffic energy columns). CPU only.

    python3 -m unittest shared_traffic/test_shared_traffic.py   (from predictor/)
The reproduction test reads eval_cells.csv and the stored tables; it needs the footprints/ directories produced by run_footprints.py."""
import csv, json, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR)); sys.path.insert(0, str(SR / 'calibrate')); sys.path.insert(0, str(HERE))
import footprint as FP  # noqa: E402
import predict_runtime_v3j as J  # noqa: E402


class FakeC:
    PTR_BASE0 = 0; PTR_STRIDE = 1 << 40


class FakeU:
    C = FakeC
    @staticmethod
    def per_phase_sets(obs): return obs.phases


class Obs:
    def __init__(self, phases): self.phases = phases


def phase(read=(), write=(), unknown=False): return dict(read=set(read), write=set(write), unknown=unknown)


def ptr_sector(ptr, k): return (ptr * (1 << 40)) // 32 + k      # sector index inside pointer `ptr`'s buffer


class EstimatorTests(unittest.TestCase):
    def test_streaming_blocks_own_disjoint_chunks(self):
        # 100 blocks, each reads its own 10 sectors of pointer 0; blocks 0 and 99 sampled
        obs = [Obs([phase(read=[ptr_sector(0, b * 10 + i) for i in range(10)])]) for b in (0, 99)]
        f = FP.kernel_footprint(FakeU, obs, 100)
        self.assertEqual(f['read_footprint_sectors'], 1000); self.assertEqual(f['reuse_working_set_bytes'], 0)

    def test_tiled_blocks_share_inputs(self):
        # 4x4 grid: block (bx,by) reads A row-tile by (8 sectors) and B column-tile bx (8 sectors, strided by 4); corners (0,0) and (3,3) sampled
        A = lambda by: [ptr_sector(0, by * 8 + i) for i in range(8)]
        B = lambda bx: [ptr_sector(1, bx * 8 + i) for i in range(8)]
        obs = [Obs([phase(read=A(by) + B(bx))]) for bx, by in ((0, 0), (3, 3))]
        f = FP.kernel_footprint(FakeU, obs, 16)
        self.assertEqual(f['read_footprint_sectors'], 64)          # A 32 + B 32; the naive count is 16 blocks x 16 = 256
        self.assertEqual(f['read_naive_sectors'], 256); self.assertEqual(f['reuse_working_set_bytes'], 64 * 32)

    def test_shared_vector_read_by_every_block(self):
        obs = [Obs([phase(read=[ptr_sector(0, i) for i in range(16)])]) for _ in range(2)]
        f = FP.kernel_footprint(FakeU, obs, 50)
        self.assertEqual(f['read_footprint_sectors'], 16)

    def test_gapped_no_sharing_is_not_collapsed_to_the_span(self):
        # each block touches 4 sectors spread over a 1000-sector range, blocks disjoint: naive 4 x blocks is the footprint, the span would overstate
        obs = [Obs([phase(read=[ptr_sector(0, b * 4 + 1000 * i) for i in range(4)])]) for b in (0, 1)]
        f = FP.kernel_footprint(FakeU, obs, 2)
        self.assertEqual(f['read_footprint_sectors'], 8)

    def test_prior_phase_writes_are_not_first_touch(self):
        obs = [Obs([phase(write=[ptr_sector(0, i) for i in range(8)]), phase(read=[ptr_sector(0, i) for i in range(8)])])]
        f = FP.kernel_footprint(FakeU, obs * 2, 4)
        self.assertEqual(f['read_footprint_sectors'], 0)

    def test_unknown_address_is_refused(self):
        with self.assertRaises(FP.FootprintRefusal): FP.kernel_footprint(FakeU, [Obs([phase(unknown=True)])], 1)


class ModelGuardTests(unittest.TestCase):
    UK = dict(first_touch_read_sectors_per_block=10, blocks=4)

    def test_dram_tier_without_footprint_is_refused(self):
        with self.assertRaises(ValueError): J.dram_read_sectors(dict(self.UK), 40, 'DRAM')

    def test_l2_tier_needs_no_footprint(self):
        self.assertEqual(J.dram_read_sectors(dict(self.UK), 40, 'L2'), 40)

    def test_reuse_beyond_l2_is_refused_whole_launch(self):
        uk = dict(self.UK, grid_footprint=dict(read_footprint_sectors=10, reuse_working_set_bytes=2 << 20, l2_capacity_bytes=1 << 20))
        J.OPTIONS['l2_rule'] = 'launch'
        try:
            with self.assertRaises(ValueError): J.dram_read_sectors(uk, 40, 'DRAM')
        finally: J.OPTIONS['l2_rule'] = 'wave'

    def test_per_wave_rule_refuses_only_when_one_wave_exceeds_l2(self):
        # one shared pointer: footprint 1e6 sectors (32 MB), 1000 blocks x 2000 sectors each; L2 = 16 MB
        ptr = dict(read=dict(footprint=1_000_000, naive=2_000_000, mean_per_block=2000.0), write=dict(footprint=0, naive=0, mean_per_block=0))
        fp = dict(blocks=1000, pointers={'0': ptr}, read_footprint_sectors=1_000_000, reuse_working_set_bytes=32_000_000, l2_capacity_bytes=16_000_000)
        uk = dict(self.UK, grid_footprint=fp)
        self.assertEqual(J.dram_read_sectors(uk, 2_000_000, 'DRAM', wave_blocks=100), 1_000_000)      # one wave touches 100 x 2000 sectors = 6.4 MB < L2
        with self.assertRaises(ValueError): J.dram_read_sectors(uk, 2_000_000, 'DRAM', wave_blocks=400)   # 25.6 MB > L2
        with self.assertRaises(ValueError): J.dram_read_sectors(uk, 2_000_000, 'DRAM')                # wave size unknown: refused

    def test_footprint_bounded_by_first_touch_and_switch_off_gives_legacy(self):
        uk = dict(self.UK, grid_footprint=dict(read_footprint_sectors=10_000, reuse_working_set_bytes=0, l2_capacity_bytes=1 << 20, pointers={}, blocks=4))
        self.assertEqual(J.dram_read_sectors(uk, 40, 'DRAM', wave_blocks=4), 40)
        J.OPTIONS['read_footprint'] = False
        try: self.assertEqual(J.dram_read_sectors(dict(self.UK), 40, 'DRAM'), 40)
        finally: J.OPTIONS['read_footprint'] = True


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import evaluate as EV
        cls.EV = EV; cls.labels = list(csv.DictReader(open(EV.PW / 'evaluation/results/eval_cells.csv'))); cls.pred = EV.predict_all(); cls.rows = EV.score_cells(cls.labels, cls.pred)

    def test_candidate_with_correction_off_is_the_current_runtime(self):
        for r in self.rows:
            self.assertEqual(self.pred[r['cell_id']]['oj'].get('primary_s'), self.pred[r['cell_id']]['cur'].get('primary_s'), r['cell_id'])

    def test_streaming_cells_keep_their_energy_columns(self):
        from cal import traffic as T
        import predict as P
        from cal import energy as En
        n = 0
        for cid, p in self.pred.items():
            row = p['row']; tr = p['new'].get('traffic')
            if 'ml_gelu' not in cid or not tr or p['row']['memory']['tier'] != 'DRAM': continue
            tot = row.get('per_launch_totals') or row['work']; mem = row['memory']; store = tot.get('executed_global_store_bytes_lane_level', 0)
            old = En.columns_from_feature_row(tot, mem['logical_bytes_per_launch'], 'DRAM', store)
            new = T.columns_from_traffic(tot, tr['l2_read_bytes'] + tr['l2_write_bytes'], tr['dram_read_bytes'] + tr['dram_write_bytes'], store)
            for c in ('B_l2', 'B_dr', 'B_wr'): self.assertAlmostEqual(new[c], old[c], delta=1e-6 * max(1, old[c]), msg=cid + c)
            n += 1
        self.assertGreater(n, 0)


if __name__ == '__main__':
    unittest.main()
