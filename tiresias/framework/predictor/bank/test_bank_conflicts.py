"""CPU unit tests for the static bank-conflict analysis (no GPU, no measured value).

Synthetic tests exercise the arithmetic; the cell tests read the committed outputs bank_conflicts_dev.json (and run one small
retained cell live for the end-to-end check against the frozen phase table).
Run: python3 -m pytest -q bank/test_bank_conflicts.py   (or python3 bank/test_bank_conflicts.py)
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bank_conflicts as B  # noqa: E402


def lanes(f, n=32):
    return {i: f(i) for i in range(n)}


class Synthetic(unittest.TestCase):
    def deg(self, f, width=4, n=32, atomic=False):
        return B.request_degrees(lanes(f, n), width, atomic)

    def test_stride_1_conflict_free(self):
        self.assertEqual(self.deg(lambda i: 4 * i), (1,))

    def test_odd_strides_conflict_free(self):
        for s in (3, 5, 33):
            self.assertEqual(self.deg(lambda i, s=s: 4 * s * i), (1,), s)

    def test_microbenchmark_strides(self):
        # measured cycles per request: 2.0 for strides 0,1,2,3,5,33 and degree for 4,8,16,32
        want = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 4.0, 5: 2.0, 8: 8.0, 16: 16.0, 32: 32.0, 33: 2.0}
        for s, cyc in want.items():
            d = self.deg(lambda i, s=s: 4 * s * i)
            self.assertEqual(B.request_costs(d, 32)[0], cyc, s)

    def test_degrees_by_stride(self):
        for s, d in {2: 2, 4: 4, 8: 8, 16: 16, 32: 32}.items():
            self.assertEqual(self.deg(lambda i, s=s: 4 * s * i), (d,))

    def test_broadcast_counts_once(self):
        self.assertEqual(self.deg(lambda i: 128), (1,))
        self.assertEqual(self.deg(lambda i: 0), (1,))

    def test_same_word_different_lanes_with_other_conflict(self):
        # lanes 0..15 read word 0, lanes 16..31 read word 32 (same bank, different word): degree 2
        self.assertEqual(self.deg(lambda i: 0 if i < 16 else 128), (2,))

    def test_atomics_serialise_same_word(self):
        self.assertEqual(self.deg(lambda i: 0, atomic=True), (32,))
        self.assertEqual(self.deg(lambda i: 4 * i, atomic=True), (1,))

    def test_subword_maps_to_word(self):
        # 8-bit accesses of consecutive lanes: 4 lanes share each word -> broadcast within a word, 8 banks used
        self.assertEqual(B.request_degrees(lanes(lambda i: i), 1), (1,))
        # 16-bit, stride 2 bytes
        self.assertEqual(B.request_degrees(lanes(lambda i: 2 * i), 2), (1,))
        # 8-bit with a 128-byte stride: every lane hits bank 0, different words
        self.assertEqual(B.request_degrees(lanes(lambda i: 128 * i), 1), (32,))

    def test_64bit_halfwarp_groups(self):
        # consecutive 64-bit elements: each half-warp covers 32 words once
        self.assertEqual(self.deg(lambda i: 8 * i, 8), (1, 1))
        # 64-bit with a 256-byte stride: all 16 lanes of a half-warp hit banks {0,1}: degree 16 per group
        self.assertEqual(self.deg(lambda i: 256 * i, 8), (16, 16))
        # 64-bit broadcast: every lane reads the same 8 bytes
        self.assertEqual(self.deg(lambda i: 64, 8), (1, 1))

    def test_128bit_quarterwarp_groups(self):
        self.assertEqual(self.deg(lambda i: 16 * i, 16), (1, 1, 1, 1))
        self.assertEqual(self.deg(lambda i: 512 * i, 16), (8, 8, 8, 8))

    def test_partial_warp_and_inactive_lanes(self):
        # only lanes 0..7 active, stride 4 words: 8 lanes in banks 0,4,...,28 -> degree 1
        self.assertEqual(B.request_degrees({i: 16 * i for i in range(8)}, 4), (1,))
        # only lane 31 active
        self.assertEqual(B.request_degrees({31: 4}, 4), (1,))
        # empty request has no groups
        self.assertEqual(B.request_degrees({}, 4), ())

    def test_costs(self):
        self.assertEqual(B.request_costs((1,), 32), (2.0, 2.0, 2.0))
        self.assertEqual(B.request_costs((8,), 32), (8.0, 8.0, 8.0))
        # 64-bit conflict-free: primary max(2, 1+1)=2; per-group floor 2+2=4; lane-scaled (2+2)*16/32 = 2
        self.assertEqual(B.request_costs((1, 1), 16), (2.0, 4.0, 2.0))
        # 128-bit conflict-free: primary 4; per-group floor 8; lane-scaled 2
        self.assertEqual(B.request_costs((1, 1, 1, 1), 8), (4.0, 8.0, 2.0))

    def test_operand_evaluation(self):
        regs = {'R5': 64, 'UR4': 0x400}
        g = regs.get
        self.assertEqual(B.eval_terms('R5+UR4+0x20', g), 64 + 0x400 + 0x20)
        self.assertEqual(B.eval_terms('UR4+-0x40', g), 0x400 - 0x40)
        self.assertEqual(B.eval_terms('RZ', g), 0)
        self.assertIsNone(B.eval_terms('R9', g))     # unknown register stays unknown
        with self.assertRaises(B.C.Refusal):
            B.eval_terms('R5.X4+UR4', g)

    def test_opcode_widths(self):
        self.assertEqual(B.shared_info('LDS')['width'], 4)
        self.assertEqual(B.shared_info('LDS.U8')['width'], 1)
        self.assertEqual(B.shared_info('STS.64')['group_lanes'], 16)
        self.assertEqual(B.shared_info('LDS.128')['group_lanes'], 8)
        self.assertEqual(B.shared_info('LDSM.16.M88')['lanes_used'], 8)
        self.assertEqual(B.shared_info('LDSM.16.M88.4')['lanes_used'], 32)
        self.assertTrue(B.shared_info('ATOMS.ADD')['atomic'])
        self.assertIsNone(B.shared_info('LDG.E'))
        self.assertIsNone(B.shared_info('IMAD'))
        with self.assertRaises(B.C.Refusal):
            B.shared_info('LDS.BOGUS')


def _load_dev():
    p = HERE / 'bank_conflicts_dev.json'
    if not p.exists():
        raise unittest.SkipTest('bank_conflicts_dev.json not generated yet')
    return json.loads(p.read_text())


class RealCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = _load_dev()['rows']

    def cell(self, operator, regime, cand):
        return self.dev['blackwell/%s/%s/%s' % (operator, regime, cand)]

    def max_degree(self, row):
        degs = [int(k) for k in row['kernels'][0]['phases'][0]['shared']['shared_conflict_degree_histogram']]
        for ph in row['kernels'][0]['phases']:
            degs += [int(k) for k in ph['shared']['shared_conflict_degree_histogram']]
        return max(degs, default=0)

    def test_reduction_conflicts_follow_the_addressing_scheme(self):
        # reduce0 (alt c1), reduce6 (alt c3) and the sequential-addressing reduce2 (train c1): every request has degree <= 2.
        # reduce1 (alt c2) uses the strided index 2*s*tid, a textbook bank-conflict pattern: the degree is min(2s, active
        # lanes' bank sharing) = 2, 4, 8, 8, 8, 4, 2, 1 over its eight reduction phases for the 256-thread block.
        def degs(cid):
            out = []
            for ph in self.dev[cid]['kernels'][0]['phases']:
                out += [int(k) for k in ph['shared']['shared_conflict_degree_histogram']]
            return out
        for cid in ('alt_cuda_samples_reduction/large/c1', 'alt_cuda_samples_reduction/large/c3', 'train_cuda_samples_reduction/large/c1'):
            self.assertLessEqual(max(degs('blackwell/' + cid)), 2, cid)
        phases = self.dev['blackwell/alt_cuda_samples_reduction/large/c2']['kernels'][0]['phases']
        seq = [max(int(k) for k in p['shared']['shared_conflict_degree_histogram']) for p in phases[1:9]]
        self.assertEqual(seq, [2, 4, 8, 8, 8, 4, 2, 1])

    def test_naive_transpose_tile_conflicts_padded_does_not(self):
        naive = self.cell('train_cuda_samples_transpose', 'small', 'c2')     # transposeCoalesced: 32x32 tile, column read
        padded = self.cell('train_cuda_samples_transpose', 'small', 'c3')    # transposeNoBankConflicts: 32x33 tile
        self.assertEqual(self.max_degree(naive), 32)
        self.assertLessEqual(self.max_degree(padded), 2)
        # same request counts; the column-read phase costs 32 vs 2 cycles per request, and the tile write is 2 in both:
        # (2 + 32) / (2 + 2) = 8.5 of total shared cycles
        self.assertEqual(naive['shared_summary']['shared_requests'], padded['shared_summary']['shared_requests'])
        ratio = naive['shared_summary']['shared_cost_cycles'] / padded['shared_summary']['shared_cost_cycles']
        self.assertAlmostEqual(ratio, 8.5)

    def test_gate_recorded_for_every_analysed_cell(self):
        for cid, row in self.dev.items():
            for k in row['kernels']:
                self.assertTrue(k['gate'].startswith('passed') or k['gate'].startswith('no shared'), cid)
        self.assertEqual(_load_dev()['gate_summary'].get('GATE_FAILED', 0), 0)

    def test_request_counts_equal_frozen_warp_counts(self):
        frozen = json.loads((B.STATIC / 'phases_blackwell.json').read_text())['rows']
        for cid, row in self.dev.items():
            for k, fk in zip(row['kernels'], frozen[cid]['kernels']):
                for ph, fp in zip(k['phases'], fk['phases']):
                    want = {op: n for op, n in fp['issue_warp_instructions'].items() if B._is_shared_opcode(op) and not op.startswith('LDGSTS')}
                    self.assertEqual(ph['shared']['shared_requests_by_opcode'], want, cid)


class LiveCell(unittest.TestCase):
    def test_small_transpose_cells_live(self):
        lk = B._dev_lookup()
        frozen = json.loads((B.STATIC / 'phases_blackwell.json').read_text())['rows']
        out = {}
        for cand in ('c2', 'c3'):
            cid = 'blackwell/train_cuda_samples_transpose/small/' + cand
            corpus, row, root = lk[cid]
            base, shared, _ = B.run_retained(corpus, row, root)
            B.gate_kernel(base, frozen[cid]['kernels'][0], shared)      # raises if the regression gate fails
            out[cand] = shared
        self.assertEqual(out['c2'][1]['shared_conflict_degree_histogram'], {'32': 32768})
        self.assertEqual(out['c3'][1]['shared_conflict_degree_histogram'], {'1': 32768})


if __name__ == '__main__':
    unittest.main()
