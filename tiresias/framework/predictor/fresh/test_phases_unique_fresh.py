"""CPU tests: python3 test_phases_unique_fresh.py (needs phases_unique_fresh.json)."""
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Fresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = json.loads((HERE / 'phases_unique_fresh.json').read_text())['rows']
        cls.corr = json.loads((HERE / 'phases_fresh_divider_corrected.json').read_text())['rows']

    def test_all_24_cells_ok_and_same_kernel_order(self):
        self.assertEqual(len(self.out), 24)
        for cid, o in self.out.items():
            self.assertEqual(o['status'], 'ok', (cid, o['reason']))
            self.assertEqual([k['kernel_id'] for k in o['kernels']], [k['kernel_id'] for k in self.corr[cid]['kernels']])

    def test_totals_equal_corrected_table(self):
        for cid, o in self.out.items():
            for ko, kc in zip(o['kernels'], self.corr[cid]['kernels']):
                self.assertEqual(len(ko['phases']), len(kc['phases']))
                for po, pc in zip(ko['phases'], kc['phases']):
                    for f in ('read_sectors', 'write_sectors', 'lines'):
                        self.assertEqual(po[f], pc[f], (cid, f))
                    self.assertEqual(po['read_sectors_total_requested'], pc['read_sectors'])

    def test_unique_le_total(self):
        for cid, o in self.out.items():
            for k in o['kernels']:
                for p in k['phases']:
                    self.assertLessEqual(p['read_sectors_first_touch'], p['read_sectors'], cid)
                self.assertLessEqual(k['unique_read_sectors_per_block'], k['total_requested_read_sectors_per_block'] + 1e-9, cid)
                self.assertLessEqual(k['first_touch_read_sectors_per_block'], k['unique_read_sectors_per_block'] + 1e-9, cid)

    def test_copy_kernel_first_touch_vs_total(self):
        # Unaligned warp requests of the padded copy kernel share a boundary sector with the next request of the
        # same row, so first touch may be below total; it must never exceed it, and for the largest row alignment
        # it is within the total. Recorded ratio is reported in REUSE_FRESH.md.
        n = 0
        for cid, o in self.out.items():
            for k in o['kernels']:
                if k['kernel_id'] != 'k1':
                    continue
                n += 1
                for p in k['phases']:
                    self.assertLessEqual(p['read_sectors_first_touch'], p['read_sectors'])
                    self.assertGreater(p['read_sectors_first_touch'], 0)
                    # measured by this analysis, not assumed: boundary-sector sharing gives 1.11-1.19
                    self.assertTrue(1.05 < p['read_sectors'] / p['read_sectors_first_touch'] < 1.25, cid)
        self.assertEqual(n, 18)

    def test_blocks_per_sm_from_features(self):
        feats = {r['cell_id']: r for r in json.loads((HERE / 'features_fresh.json').read_text())['rows']}
        for cid, o in self.out.items():
            for k in o['kernels']:
                self.assertEqual(k['blocks_per_sm'], feats[cid]['occupancy']['blocks_per_sm'])

    def test_layer_norm_main_kernel_shows_reuse(self):
        cid = 'blackwell/fresh_pytorch_layer_norm/medium/c1'
        k = self.out[cid]['kernels'][0]
        self.assertGreater(sum(p['read_sectors'] for p in k['phases']), sum(p['read_sectors_first_touch'] for p in k['phases']))

    def test_softmax_no_reuse(self):
        for cid, o in self.out.items():
            if 'softmax' in cid:
                for k in o['kernels']:
                    if k['kernel_id'] != 'k1':
                        for p in k['phases']:
                            self.assertEqual(p['read_sectors_first_touch'], p['read_sectors'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
