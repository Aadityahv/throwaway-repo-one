import math
import unittest
from types import SimpleNamespace

from dependencies import decode, Graph, Refusal
from reconstruct_calibration import fit, response, sector_count


def site(op, *args, pc=0, pred=None):
    return SimpleNamespace(op=op,a=args,pc=pc,pred=pred)


class Dependencies(unittest.TestCase):
    def test_address_carry_is_produced_not_consumed(self):
        r=decode(site('LEA','R4','P0','R2','UR16','0x2'))
        self.assertEqual(r.definitions,{'R4','P0'})
        self.assertEqual(r.uses,{'R2','UR16'})
        self.assertIn('P0',decode(site('LEA.HI.X','R5','R2','UR17','R3','0x2','P0')).uses)

    def test_register_pairs_and_scalar_multiplicands(self):
        r=decode(site('IMAD.WIDE','R4','R2','R3','R6'))
        self.assertEqual(r.definitions,{'R4','R5'})
        self.assertEqual(r.uses,{'R2','R3','R6','R7'})
        r=decode(site('IADD.64','R4','R6','UR8'))
        self.assertEqual(r.uses,{'R6','R7','UR8','UR9'})
        r=decode(site('ISETP.EQ.S64.OR','P0','PT','R18','RZ','P1'))
        self.assertEqual(r.uses,{'R18','R19','P1'})

    def test_vector_store_and_add_carry_outputs(self):
        r=decode(site('STG.E.128','desc[UR4][R2.64]','R8'))
        self.assertEqual(r.uses,{'UR4','R2','R3','R8','R9','R10','R11'})
        r=decode(site('IADD3','R2','P0','P1','R4','R5','R6'))
        self.assertEqual(r.definitions,{'R2','P0','P1'})
        self.assertEqual(r.uses,{'R4','R5','R6'})

    def test_predicate_register_mask(self):
        self.assertEqual(decode(site('P2R','R2','PR','RZ','0x42')).uses,{'P1','P6'})

    def test_unknown_format_refuses(self):
        with self.assertRaises(Refusal):decode(site('LDSM.16.M88','R11','[R6+UR6]'))
        with self.assertRaises(Refusal):decode(site('MADEUP','R0','R1'))
        with self.assertRaises(Refusal):decode(site('FADD.UNKNOWN','R0','R1','R2'))
        with self.assertRaises(Refusal):decode(site('IADD3','R0','R1','R2','R3'))

    def test_independent_loads_do_not_become_a_pointer_chase_at_backedge(self):
        g=Graph()
        for pc in [16,32,16]:g.add(site('LDG.E','R2','desc[UR4][R8.64]',pc=pc))
        self.assertEqual(g.summary()['register_path_global_loads'],1)
        chase=Graph()
        for pc in [16,32,16]:chase.add(site('LDG.E.64','R2','desc[UR4][R2.64]',pc=pc))
        self.assertEqual(chase.summary()['register_path_global_loads'],3)

    def test_loop_accumulator_is_preserved_across_backward_pc(self):
        g=Graph()
        for pc in [16,32,16]:g.add(site('FADD','R2','R2','R4',pc=pc))
        self.assertEqual(g.longest({'FADD':4}),12)

    def test_global_load_high_word_reaches_next_address(self):
        g=Graph();g.add(site('LDG.E.64','R4','desc[UR8][R10.64]'))
        g.add(site('IMAD','R12','R5','4','RZ'))
        g.add(site('LDG.E','R6','desc[UR8][R12.64]'))
        self.assertEqual(g.summary()['register_path_global_loads'],2)

    def test_weighted_max_uses_one_real_path(self):
        g=Graph();g.add(site('LDG.E','R2','desc[UR4][R8.64]'))
        for _ in range(5):g.add(site('FADD','R4','R4','R6'))
        self.assertEqual(g.critical_time({'LDG.E':100,'FADD':2}),100)
        # Independent component maxima would incorrectly give 100+10.

    def test_shared_memory_links_to_other_lane_store(self):
        g=Graph();g.add(site('LDG.E','R2','desc[UR4][R8.64]'),lane=0)
        g.add(site('STS','[R8]','R2'),lane=0,shared_address=0)
        g.add(site('BAR.SYNC','0'),participants=[0,1])
        g.add(site('LDS','R4','[R8]'),lane=1,shared_address=0)
        self.assertEqual(g.critical_time({'LDG.E':100,'STS':1,'BAR.SYNC':0,'LDS':4}),105)

    def test_shuffle_uses_peer_snapshot(self):
        g=Graph();g.add(site('LDG.E','R2','desc[UR4][R8.64]'),lane=0)
        g.add(site('MOV','R2','7'),lane=1)
        sh=site('SHFL.BFLY','PT','R2','R2','1','0x1f')
        g.add_batch([dict(site=sh,lane=0,peer_lane=1),dict(site=sh,lane=1,peer_lane=0)])
        weights={'LDG.E':100,'MOV':1,'SHFL.BFLY':2}
        self.assertEqual(g.critical_time(weights),102)
        node=g.nodes[-1]
        self.assertIn(0,node['dependencies'])
        self.assertNotIn(2,node['dependencies'])

    def test_missing_shared_peer_or_weights_refuses_timing(self):
        g=Graph();g.add(site('LDS','R4','[R8]'))
        with self.assertRaises(Refusal):g.critical_time({'LDS':4})
        h=Graph();h.add(site('FADD','R2','R2','R4'))
        with self.assertRaises(Refusal):h.critical_time({})
        with self.assertRaises(Refusal):h.critical_time({'FADD':math.nan})
        with self.assertRaises(Refusal):h.add(site('MOV','R2','3'),guard=None)

    def test_shared_race_is_not_silently_resolved(self):
        g=Graph()
        with self.assertRaises(Refusal):g.add_batch([
            dict(site=site('STS','[R8]','R2'),lane=i,shared_address=0) for i in [0,1]])


class Calibration(unittest.TestCase):
    def test_pointer_width_and_wrap_sector_formula(self):
        self.assertEqual(sector_count(1,32,65536),8)
        self.assertEqual(sector_count(2,32,65536),16)
        self.assertEqual(sector_count(4,32,65536),32)
        self.assertEqual(sector_count(64,1024,65536),32)

    def test_recovers_known_latency_and_pressure_response(self):
        rows=[dict(sectors_per_warp=s,launched_warps_per_sm=w) for s in [8,16,32] for w in [1,2,4,8,16,32,48]]
        truth=[30,2,.6]
        for r,p in zip(rows,response(truth,rows)):r['effective_hop_ns']=float(p)
        fitted=fit(rows)
        for a,b in zip(fitted,truth):self.assertAlmostEqual(a,b,places=3)
        for p,r in zip(response(fitted,rows),rows):self.assertAlmostEqual(p,r['effective_hop_ns'],places=3)


if __name__=='__main__':unittest.main()
