"""CPU tests: python3 reuse/test_phases_unique.py  (needs phases_unique_blackwell.json for the real-cell tests)."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases_unique as U

B0 = U.C.PTR_BASE0


def fake_obs(blocks, reqs, nphases):
    """reqs: list of (phase, direction, [lane addresses], width)."""
    o = U.CapturingObserver(32)
    o.phase = [nphases - 1] * 32
    for i, (ph, d, addrs, w) in enumerate(reqs):
        o.memory[(0, ph, i, 0, d, w)] = list(addrs)
    o.blocks = blocks
    return o


def coalesced(base, n=32, w=4):
    return [base + 4 * i for i in range(n)]


class Synthetic(unittest.TestCase):
    def kernel(self, reqs, nph, blocks=4):
        obs = [fake_obs(blocks, reqs, nph) for _ in range(2)]
        an = U.analyse_block(obs[0])
        return an

    def test_two_pass_first_touch_equals_unique(self):
        # pass 1 (phase 0) and pass 2 (phase 1) read the same 128 B row
        reqs = [(0, 'read', coalesced(B0), 4), (1, 'read', coalesced(B0), 4)]
        an = self.kernel(reqs, 2)
        self.assertEqual([r['req_read'] for r in an['rows']], [4, 4])      # frozen-style total 8
        self.assertEqual([r['first_touch'] for r in an['rows']], [4, 0])   # second pass is L1
        self.assertEqual(len(an['cum_read']), 4)
        self.assertEqual(sum(r['first_touch'] for r in an['rows']), len(an['cum_read']))

    def test_streaming_first_touch_equals_total(self):
        reqs = [(0, 'read', coalesced(B0 + 128 * i), 4) for i in range(5)]
        an = self.kernel(reqs, 1)
        r = an['rows'][0]
        self.assertEqual(r['first_touch'], r['req_read'])

    def test_distinct_buffers_do_not_alias(self):
        reqs = [(0, 'read', coalesced(B0), 4), (1, 'read', coalesced(B0 + U.C.PTR_STRIDE), 4)]
        an = self.kernel(reqs, 2)
        self.assertEqual([r['first_touch'] for r in an['rows']], [4, 4])

    def test_written_then_read_counts_touched_and_reported(self):
        reqs = [(0, 'write', coalesced(B0), 4), (1, 'read', coalesced(B0), 4)]
        an = self.kernel(reqs, 2)
        self.assertEqual(an['rows'][1]['first_touch'], 0)
        self.assertEqual(an['rows'][1]['readback_of_prior_phase_writes'], 4)

    def test_same_phase_repeat_counts_unique_once(self):
        reqs = [(0, 'read', coalesced(B0), 4), (0, 'read', coalesced(B0), 4)]
        an = self.kernel(reqs, 1)
        r = an['rows'][0]
        self.assertEqual((r['req_read'], r['unique_in_phase'], r['first_touch']), (8, 4, 4))

    def test_kernel_entry_refuses_mismatch(self):
        reqs = [(0, 'read', coalesced(B0), 4)]
        obs = [fake_obs(4, reqs, 1) for _ in range(2)]
        bad = dict(phases=[dict(index=0, repetitions=1, read_sectors=999, write_sectors=0, lines=0)])
        self.assertEqual(U.kernel_entry(obs, bad, 3)['status'], 'refused')
        good = dict(phases=[dict(index=0, repetitions=1, read_sectors=16, write_sectors=0, lines=4)])
        e = U.kernel_entry(obs, good, 3)
        self.assertEqual(e['status'], 'ok')
        self.assertEqual(e['phases'][0]['read_sectors_first_touch'], 16)


class RealCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = json.loads((HERE / 'phases_unique_blackwell.json').read_text())['rows']
        cls.frozen = json.loads((HERE.parent / 'phases_blackwell.json').read_text())['rows']

    def check_equal(self, cid):
        o, f = self.out[cid], self.frozen[cid]
        self.assertEqual(o['status'], 'ok', o)
        for ko, kf in zip(o['kernels'], f['kernels']):
            for po, pf in zip(ko['phases'], kf['phases']):
                for field in ('read_sectors', 'write_sectors', 'lines'):
                    self.assertEqual(po[field], pf[field])
                self.assertLessEqual(po['read_sectors_first_touch'], po['read_sectors'])
        return o

    def first(self, op):
        return next(c for c in sorted(self.out) if '/' + op + '/' in c and self.out[c]['status'] == 'ok')

    def test_streaming_copy_first_touch_equals_total(self):
        o = self.check_equal(self.first('final_cuda_samples_copy'))
        for k in o['kernels']:
            for p in k['phases']:
                self.assertEqual(p['read_sectors_first_touch'], p['read_sectors'])

    def test_triton_layer_norm(self):
        o = self.check_equal(self.first('final_triton_layer_norm'))

    def test_pytorch_layer_norm(self):
        self.check_equal(self.first('final_pytorch_layer_norm'))

    def test_unique_le_total_everywhere(self):
        n = 0
        for cid, o in self.out.items():
            if o['status'] != 'ok':
                continue
            for k in o['kernels']:
                self.assertLessEqual(k['unique_read_sectors_per_block'], k['total_requested_read_sectors_per_block'], cid)
                self.assertLessEqual(k['first_touch_read_sectors_per_block'], k['unique_read_sectors_per_block'], cid)
                n += 1
        self.assertGreater(n, 50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
