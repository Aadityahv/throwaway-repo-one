"""Synthetic-SASS checks of the static sector counter (CPU only, no corpus access)."""
import unittest
import coalescing as C

BASE = C.PTR_BASE0            # 256-byte aligned (in fact 4 GiB aligned)


def snippet(lines):
    return C.parse('\n'.join('/*%04x*/ %s;' % (i * 16, l) for i, l in enumerate(lines)))


def consts(base=BASE, extra=None):
    d = {0x380: base & C.M32, 0x384: base >> 32}
    d.update(extra or {})
    return d


def run(lines, threads=32, base=BASE, bdim=(32, 1), ctaid=(0, 0), extra=None):
    sites = snippet(lines)
    reqs, evt, info = C.block_requests(sites, consts(base, extra), ctaid, bdim, threads)
    return reqs


PRE = ['S2R R0, SR_TID.X', 'LDC.64 R4, c[0x0][0x380]']


class Basic(unittest.TestCase):
    def one(self, reqs):
        self.assertEqual(len(reqs), 1)
        (lst,) = reqs.values()
        self.assertEqual(len(lst), 1)
        return lst[0]

    def test_unit_stride_4B_load_is_4_sectors_1_line(self):
        r = self.one(run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'EXIT']))
        self.assertEqual((r['sectors'], r['lines'], r['active'], r['bytes'], r['status']), (4, 1, 32, 128, 'known'))

    def test_stride_33_floats_is_32_sectors(self):
        r = self.one(run(PRE + ['IMAD R1, R0, 0x21, RZ', 'IMAD.WIDE.U32 R2, R1, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'EXIT']))
        self.assertEqual(r['sectors'], 32)
        self.assertEqual(r['lines'], len({(132 * l) // 128 for l in range(32)}))

    def test_stride_7_floats_differs_from_stride_33(self):
        r = self.one(run(PRE + ['IMAD R1, R0, 0x7, RZ', 'IMAD.WIDE.U32 R2, R1, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'EXIT']))
        self.assertEqual(r['sectors'], len({(28 * l) // 32 for l in range(32)}))
        self.assertEqual(r['sectors'], 28)          # 28-byte pitch: lanes 7,14,21,28 share a sector with a neighbour
        self.assertNotEqual(r['sectors'], 32)

    def test_vector_16B_unit_stride(self):
        r = self.one(run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x10, R4', 'LDG.E.128 R8, desc[UR4][R2.64]', 'EXIT']))
        self.assertEqual((r['sectors'], r['lines'], r['bytes']), (16, 4, 512))

    def test_64bit_unit_stride(self):
        r = self.one(run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x8, R4', 'LDG.E.64 R8, desc[UR4][R2.64]', 'EXIT']))
        self.assertEqual((r['sectors'], r['lines'], r['bytes']), (8, 2, 256))

    def test_broadcast_is_one_sector(self):
        r = self.one(run(PRE + ['LDG.E R6, desc[UR4][R4.64]', 'EXIT']))
        self.assertEqual((r['sectors'], r['lines'], r['active']), (1, 1, 32))

    def test_misaligned_base_adds_a_sector(self):
        r = self.one(run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'EXIT'], base=BASE + 16))
        self.assertEqual((r['sectors'], r['lines']), (5, 2))

    def test_lane_subset_and_store(self):
        reqs = run(PRE + ['ISETP.LT.U32.AND P0, PT, R0, 5, PT', 'IMAD.WIDE.U32 R2, R0, 0x4, R4', '@P0 STG.E desc[UR4][R2.64], R6', 'EXIT'])
        (lst,) = reqs.values()
        self.assertEqual((lst[0]['active'], lst[0]['bytes'], lst[0]['sectors']), (5, 20, 1))

    def test_fully_predicated_off_warp_issues_no_request(self):
        reqs = run(PRE + ['ISETP.LT.U32.AND P0, PT, R0, 0, PT', 'IMAD.WIDE.U32 R2, R0, 0x4, R4', '@P0 LDG.E R6, desc[UR4][R2.64]', 'EXIT'])
        self.assertEqual(reqs, {})

    def test_data_dependent_address_is_null(self):
        reqs = run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'IMAD.WIDE.U32 R8, R6, 0x4, R4', 'LDG.E R7, desc[UR4][R8.64]', 'EXIT'])
        first, second = reqs[0x30][0], reqs[0x50][0]
        self.assertEqual(first['sectors'], 4)
        self.assertIsNone(second['sectors'])
        self.assertEqual(second['status'], 'data_dependent_address')
        self.assertEqual(second['bytes'], 128)       # active lanes are still known

    def test_data_dependent_predicate_is_flagged(self):
        reqs = run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'FSETP.GT.AND P0, PT, R6, RZ, PT', '@P0 LDG.E R7, desc[UR4][R2.64]', 'EXIT'])
        # unknown predicate on a priced memory instruction: site is reported, not guessed
        r = reqs[0x50][0]
        self.assertEqual((r['status'], r['sectors'], r['active']), ('data_dependent_predicate', None, None))

    def test_lea_carry_across_32bit_boundary(self):
        base = 0x1_ffff_ff80      # low word wraps inside the warp's 128 bytes
        r = self.one(run(['S2R R0, SR_TID.X', 'LDC.64 R4, c[0x0][0x380]', 'LEA R6, P0, R0, R4, 0x2', 'LEA.HI.X R7, R0, R5, RZ, 0x2, P0',
                          'LDG.E R8, desc[UR4][R6.64]', 'EXIT'], base=base))
        self.assertEqual((r['sectors'], r['lines']), (4, 1))
        # direct check of the 64-bit address produced
        sites = snippet(['S2R R0, SR_TID.X', 'LDC.64 R4, c[0x0][0x380]', 'LEA R6, P0, R0, R4, 0x2', 'LEA.HI.X R7, R0, R5, RZ, 0x2, P0', 'LDG.E R8, desc[UR4][R6.64]', 'EXIT'])
        _, log, _, _ = C.run_thread(sites, consts(base), {'SR_TID.X': 31})
        self.assertEqual(log[0][4], base + 31 * 4)

    def test_ldgsts_trailing_predicate_gates_global_read(self):
        on = run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x10, R4', 'ISETP.GE.U32.AND P1, PT, R0, RZ, PT', 'LDGSTS.E.BYPASS.128 [R20], desc[UR14][R2.64], P1', 'EXIT'])
        (lst,) = on.values()
        self.assertEqual([(r['sectors'], r['ignore_src_pred']) for r in lst if not r['ignore_src_pred']], [(16, False)])
        off = run(PRE + ['IMAD.WIDE.U32 R2, R0, 0x10, R4', 'ISETP.LT.U32.AND P1, PT, R0, RZ, PT', 'LDGSTS.E.BYPASS.128 [R20], desc[UR14][R2.64], P1', 'EXIT'])
        (lst,) = off.values()
        self.assertEqual([r for r in lst if not r['ignore_src_pred']], [])
        self.assertEqual([r['sectors'] for r in lst if r['ignore_src_pred']], [16])

    def test_second_warp_and_loop_arrivals_are_separate_requests(self):
        reqs = run(PRE + ['MOV R9, 0', 'IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'IADD3 R9, PT, PT, R9, 0x1, RZ',
                          'IADD3 R2, PT, PT, R2, 0x400, RZ', 'ISETP.LT.U32.AND P0, PT, R9, 0x3, PT', '@P0 BRA 0x30', 'EXIT'], threads=64)
        (lst,) = reqs.values()
        self.assertEqual(len(lst), 2 * 3)    # 2 warps x 3 trips

    def test_unknown_control_refuses(self):
        sites = snippet(PRE + ['IMAD.WIDE.U32 R2, R0, 0x4, R4', 'LDG.E R6, desc[UR4][R2.64]', 'ISETP.NE.U32.AND P0, PT, R6, RZ, PT', '@P0 BRA 0x80', 'EXIT', 'EXIT'])
        with self.assertRaises(C.Refusal):
            C.block_requests(sites, consts(), (0, 0), (32, 1), 32)

    def test_unsupported_width_refuses(self):
        with self.assertRaises(C.Refusal):
            C.mem_kind('LDG.E.U8')


class Blocks(unittest.TestCase):
    LINES = ['S2R R0, SR_TID.X', 'S2R R1, SR_CTAID.X', 'LDC.64 R4, c[0x0][0x380]', 'LDC R7, c[0x0][0x390]',
             'IMAD R3, R1, 0x20, R0', 'ISETP.GE.AND P0, PT, R3, R7, PT', 'IMAD.WIDE.U32 R2, R3, 0x4, R4', '@!P0 LDG.E R6, desc[UR4][R2.64]', 'EXIT']

    def cell(self, nblocks, n):
        sites = snippet(self.LINES)
        cb = consts(extra={0x390: n})
        blocks, exh, total, sigs, evts, info = C.analyse_blocks(sites, cb, (nblocks, 1), (32, 1), 32)
        (agg, _), cov = C.combine(blocks, exh, total, sigs, evts)
        return agg, cov

    def test_sampled_grid_matches_closed_form_with_tail(self):
        nb = 400; n = nb * 32 - 5
        agg, cov = self.cell(nb, n)
        (v,) = [x for k, x in agg.items()]
        self.assertFalse(cov['exhaustive'])
        self.assertEqual(v['bytes'], n * 4)               # 5 inactive lanes in the final warp
        self.assertEqual(v['requests'], nb)
        self.assertEqual(v['sectors'], (nb - 1) * 4 + 4)  # tail warp: 27 lanes = 108 B = 4 sectors

    def test_small_grid_is_exhaustive(self):
        agg, cov = self.cell(10, 320)
        self.assertTrue(cov['exhaustive'])
        (v,) = agg.values()
        self.assertEqual((v['requests'], v['sectors']), (10, 40))

    def test_change_point_is_located_by_bisection(self):
        nb = 400; n = 100 * 32            # blocks >= 100 are fully predicated off (like a reduction reading 2 elements per thread)
        agg, cov = self.cell(nb, n)
        (v,) = agg.values()
        self.assertEqual((v['requests'], v['bytes'], v['sectors']), (100, 100 * 128, 400))
        self.assertGreaterEqual(cov['change_points_located_by_bisection'], 1)
        self.assertEqual(cov['blocks_in_unresolved_gaps'], 0)


class FloatReciprocalDivision(unittest.TestCase):
    def test_hfma2_constant_register(self):
        sites = snippet(['HFMA2 R5, -RZ, RZ, 0, 2.384185791015625e-07', 'HFMA2 R6, -RZ, RZ, 1.5, 0', 'EXIT'])
        m_events, log, arr, reach = C.run_thread(sites, {}, {})  # no memory sites: only checks it runs
        self.assertEqual(C.half_bits(2.384185791015625e-07), 4)
        self.assertEqual(C.half_bits(1.5), 0x3e00)

    def test_compiler_unsigned_division_sequence_is_exact(self):
        # the I2F.RP / MUFU.RCP / F2I sequence the compiler emits for 32-bit x / y and x % y (copied from the retained transpose SASS)
        lines = ['I2F.U32.RP R0, R7', 'MUFU.RCP R0, R0', 'IADD R4, R0, 0xffffffe', 'F2I.FTZ.U32.TRUNC.NTZ R5, R4', 'HFMA2 R4, -RZ, RZ, 0, 0',
                 'IADD R6, RZ, -R5', 'IMAD R9, R6, R7, RZ', 'IMAD.HI.U32 R5, R5, R9, R4', 'IMAD.HI.U32 R5, R5, R10, RZ', 'IADD R0, -R5, RZ',
                 'IMAD R4, R7, R0, R10', 'ISETP.GE.U32.AND P0, PT, R4, R7, PT', '@P0 IADD R4, R4, -R7', '@P0 IADD R5, R5, 0x1',
                 'ISETP.GE.U32.AND P0, PT, R4, R7, PT', '@P0 IADD R4, R4, -R7', '@P0 IADD R5, R5, 0x1', 'EXIT']
        sites = C.parse('\n'.join('/*%04x*/ %s;' % (i * 16, l) for i, l in enumerate(lines)))
        import random
        rnd = random.Random(1)
        cases = [(x, y) for y in (1, 2, 3, 7, 255, 256, 1000, 8192, 65535, 65536) for x in (0, 1, y - 1, y, y + 1, 12345, 65535, 2 ** 20 + 3)]
        cases += [(rnd.randrange(2 ** 24), rnd.randrange(1, 2 ** 13)) for _ in range(2000)]
        for x, y in cases:
            m = C.Mach({}, {}); m.r['R7'] = y; m.r['R10'] = x
            # run via run_thread by seeding registers through immediates
            pre = ['MOV R7, %d' % y, 'MOV R10, %d' % x]
            ss = C.parse('\n'.join('/*%04x*/ %s;' % (i * 16, l) for i, l in enumerate(pre + lines)))
            # expose final registers with an LDG whose address register is the quotient: use probe
            probe_sites = C.parse('\n'.join('/*%04x*/ %s;' % (i * 16, l) for i, l in enumerate(pre + lines[:-1] + ['LDG.E R1, desc[UR4][R4.64]', 'EXIT'])))
            pr = {}
            C.run_thread(probe_sites, {}, {}, probe=pr)
            regs = pr[(len(pre) + len(lines) - 1) * 16][0][2]
            self.assertEqual((regs['R5'], regs['R4']), (x // y, x % y), (x, y))


class CorpusChecks(unittest.TestCase):
    def test_diagonal_transpose_block_remap_is_a_bijection(self):
        import json
        corpus, root = list(C.D.CORPORA.items())[1]
        rows = [r for r in json.loads((root / 'retention_manifest.json').read_text())['rows'] if r['operator_id'] == 'train_cuda_samples_transpose' and r['cell'] == 'small/c4']
        if not rows: self.skipTest('corpus absent')
        row = rows[0]
        consts, coords, threads, blocks, launch = C.D.binding(corpus, row, root)
        meta = C.D.parameter_layout(root, row)
        cb = {k: v.exact_value for k, v in consts.items()}; pc, _ = C.pointer_consts(meta); cb.update(pc)
        g, bd = launch['grid'], launch['block']
        for off, v in zip((0x360, 0x364, 0x368), (bd[0], bd[1], 1)): cb.setdefault(off, v)
        for off, v in zip((0x370, 0x374, 0x378), (g[0], g[1], 1)): cb.setdefault(off, v)
        sites = C.parse((root / row['disassembly_path']).read_text())
        tiles = set()
        for b in range(g[0] * g[1]):
            c = {'SR_CTAID.X': b % g[0], 'SR_CTAID.Y': b // g[0], 'SR_CTAID.Z': 0, 'SR_CgaCtaId': 0, 'SR_TID.X': 0, 'SR_TID.Y': 0, 'SR_TID.Z': 0, 'SR_LANEID': 0}
            _, log, _, _ = C.run_thread(sites, cb, c)
            self.assertIsNotNone(log[0][4])
            tiles.add(log[0][4])
        self.assertEqual(len(tiles), g[0] * g[1])          # every block reads a distinct input tile


if __name__ == '__main__':
    unittest.main()
