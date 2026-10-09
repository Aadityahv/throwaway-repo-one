import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases as F
import predict_runtime as R
import score_runtime as S
sys.path.insert(0, str(HERE / 'constants'))
from fit_dependency import nonnegative_cost


class RuntimeCompletion(unittest.TestCase):
    def site(self, pc, op, args=()):
        return SimpleNamespace(pc=pc, op=op, a=args, pred=None)

    def test_barrier_serializes_phases_and_resets_chain(self):
        obs = F.Observer(1)
        obs.event(0, self.site(0, 'LDG.E', ('R2', 'desc[UR4][R4.64]')), True, 0)
        obs.event(0, self.site(16, 'BAR.SYNC', ('0',)), True)
        obs.event(0, self.site(32, 'LDG.E', ('R2', 'desc[UR4][R4.64]')), True, 32)
        ps = obs.finish(1)['phases']
        self.assertEqual([p['read_sectors'] for p in ps], [1, 1])
        self.assertEqual([p['dependent_global_load_depth'] for p in ps], [1, 1])

    def test_executed_loop_trips_charge_latency(self):
        obs = F.Observer(1)
        for _ in range(3):
            obs.event(0, self.site(0, 'LDG.E', ('R2', 'desc[UR4][R4.64]')), True, 0)
            obs.event(0, self.site(16, 'BRA', ('0x0',)), True)
        self.assertEqual(obs.finish(1)['phases'][0]['dependent_global_load_depth'], 3)

    def test_unknown_address_stays_null(self):
        obs = F.Observer(1)
        obs.event(0, self.site(0, 'LDG.E', ('R2', 'desc[UR4][R4.64]')), True)
        self.assertIsNone(obs.finish(1)['phases'][0]['read_sectors'])

    def test_nonnegative_dependency_fit(self):
        self.assertEqual(nonnegative_cost([dict(compute_depth=2, measured_s=5, fixed_s=1)]), 2)
        self.assertEqual(nonnegative_cost([dict(compute_depth=2, measured_s=0, fixed_s=1)]), 0)

    def test_phase_max_is_between_diagnostics(self):
        c = json.loads((HERE/'constants/stream_constants.json').read_text())['constants']
        p = dict(read_sectors=10000, write_sectors=0, lines=2500, read_bytes=320000, write_bytes=0,
                 dependent_global_load_depth=1, critical_path_compute_instructions=1,
                 issue_warp_instructions={'MUFU.EX2':100000}, repetitions=1)
        q = dict(p, read_sectors=0, lines=0, read_bytes=0, dependent_global_load_depth=100)
        v = R.kernel_terms([p,q], dict(blocks_per_sm=1, waves=1, active_sm_fraction=1), 'L2', c, 1e-9, 188, 2.617e9)
        self.assertLessEqual(v['max_s'],v['primary_s'])
        self.assertLessEqual(v['primary_s'],v['sum_s'])
        self.assertGreater(v['primary_s'],v['max_s'])

    def test_unsupported_failure_percentiles_remain_explicit(self):
        values=[1.0]*80+[float('inf')]*20
        v=S.stats(values)
        self.assertEqual(v['median_pct'],1.0)
        self.assertIsNone(v['p90_pct'])
        self.assertEqual(v['unsupported_failures'],20)

    def test_pytorch_wide_shift_preserves_address_high_word(self):
        sites=F.C.parse('/*0000*/ MOV R2, 0xffffffff;\n/*0010*/ MOV R3, 1;\n/*0020*/ SHF.L.U64.HI R5, R2, 0x4, R3;\n/*0030*/ EXIT;')
        seen=[]
        def trace(coords,s,guard,r,p):
            if s.op=='EXIT': seen.append(r['R5'].lo)
        i=F.P.Interp(F.C.D,ext=True,trace=trace)
        i.run_block(sites,{},lambda l:{},1)
        self.assertEqual(seen,[31])

    def test_pytorch_carry_crosses_low_word_boundary(self):
        sites=F.C.parse('\n'.join('/*%04x*/ %s;'%(i*16,op) for i,op in enumerate([
            'MOV R0, 31', 'MOV R4, 0xffffff80', 'MOV R5, 1',
            'LEA R6, P0, R0, R4, 0x2', 'LEA.HI.X R7, R0, R5, RZ, 0x2, P0',
            'LDG.E R8, desc[UR4][R6.64]', 'EXIT'])))
        seen=[]
        def trace(coords,s,guard,r,p):
            if s.op=='LDG.E': seen.append((r['R7'].lo<<32)|r['R6'].lo)
        i=F.P.Interp(F.C.D,ext=True,trace=trace)
        i.run_block(sites,{},lambda l:{},1)
        self.assertEqual(seen,[0x1fffffffc])
        # Actual overflow case.
        sites=F.C.parse('\n'.join('/*%04x*/ %s;'%(i*16,op) for i,op in enumerate([
            'MOV R0, 32', 'MOV R4, 0xffffff80', 'MOV R5, 1',
            'LEA R6, P0, R0, R4, 0x2', 'LEA.HI.X R7, R0, R5, RZ, 0x2, P0',
            'LDG.E R8, desc[UR4][R6.64]', 'EXIT'])))
        i.run_block(sites,{},lambda l:{},1)
        self.assertEqual(seen[-1],0x200000000)


if __name__=='__main__': unittest.main()
