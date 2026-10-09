"""CPU independent analytical invariants and fail-closed interpreter cases."""
import copy,json,math,unittest
from pathlib import Path
from derive import *

def synthetic(lines):
    return parse('\n'.join('/*%04x*/ %s;'%(i*16,line) for i,line in enumerate(lines)))
class Paths(unittest.TestCase):
    def test_lane_subset_and_width(self):
        sites=synthetic(['S2R R0, SR_TID.X','ISETP.LT.U32.AND P0, PT, R0, 5, PT','@P0 LDS.128 R2, [RZ]','EXIT'])
        n=0
        for tid in range(32):
            e,_,_=run_lane(sites,{}, {'SR_TID.X':V.exact(tid)});n+=e[(32,'shared_load',128)]
        self.assertEqual(n,5)
    def test_unknown_unpriced_does_not_break_target(self):
        e,_,u=run_lane(synthetic(['LDG.E R0, [RZ]','FSETP.GT.AND P0, PT, R0, RZ, PT','@P0 FMUL R1, R0, R0','SHFL.BFLY PT, R2, R0, 1, 0x1f','EXIT']),{}, {})
        self.assertEqual(sum(e.values()),1);self.assertEqual(u['FMUL'],1)
    def test_data_control_refuses(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):run_lane(synthetic(['LDG.E R0, [RZ]','ISETP.NE.U32.AND P0, PT, R0, RZ, PT','@P0 BRA 0x40','EXIT','EXIT']),{}, {})
    def test_unknown_priced_predicate_refuses(self):
        with self.assertRaisesRegex(Refusal,'unknown target'):run_lane(synthetic(['FSETP.GT.AND P0, PT, R0, RZ, PT','@P0 MUFU.EX2 R1, R0','EXIT']),{}, {})
    def test_poison_output_not_stale(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):run_lane(synthetic(['MOV R0, 0','@P0 MOV R0, 1','ISETP.NE.U32.AND P1, PT, R0, RZ, PT','@P1 BRA 0x50','EXIT','EXIT']),{}, {})
    def test_unknown_lop3_predicate_poisons_register_output(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):
            run_lane(synthetic(['MOV R0, 0','@P0 LOP3.LUT P1, R0, R2, R3, RZ, 0xc0, !PT','ISETP.NE.U32.AND P2, PT, R0, RZ, PT','@P2 BRA 0x50','EXIT','EXIT']),{}, {})
    def test_unknown_lop3_poisons_predicate_output(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):
            run_lane(synthetic(['ISETP.GE.U32.AND P1, PT, RZ, RZ, PT','@P0 LOP3.LUT P1, R0, R2, R3, RZ, 0xc0, !PT','@P1 BRA 0x40','EXIT','EXIT']),{}, {})
    def test_address_carry_predicate_cannot_stay_stale(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):
            run_lane(synthetic(['ISETP.GE.U32.AND P1, PT, RZ, RZ, PT','LEA R0, P1, RZ, RZ, 2','@P1 BRA 0x40','EXIT','EXIT']),{}, {})
    def test_unknown_wide_constant_load_poisons_high_word(self):
        with self.assertRaisesRegex(Refusal,'unknown control'):
            run_lane(synthetic(['MOV R1, 0','@P0 LDC.64 R0, c[0x0][0x380]','ISETP.NE.U32.AND P2, PT, R1, RZ, PT','@P2 BRA 0x50','EXIT','EXIT']),{}, {})
    def test_generalized_carry_combiner_and_funnel_refuse(self):
        for op in ['IADD3 R0, P0, PT, RZ, RZ, RZ','LOP3.LUT R0, RZ, RZ, RZ, 0xc0, PT','SHF.R.U32.HI R0, 1, 2, 3','ISETP.GE.U32.AND P0, P1, RZ, RZ, PT']:
            with self.assertRaises(Refusal):run_lane(synthetic([op,'EXIT']),{}, {})
    def test_unknown_opcode_and_collective_refuse(self):
        for op in ['SURPRISE R0, R1','BRA.DIV UR4, 0x10']:
            with self.assertRaises(Refusal):run_lane(synthetic([op,'EXIT']),{}, {})
    def test_nonterminating_refuses(self):
        with self.assertRaisesRegex(Refusal,'step limit'):run_lane(synthetic(['BRA 0x0']),{}, {},max_steps=10)
    def test_signed_boundary_and_wrap(self):
        self.assertTrue(cmp(V.exact(-1),V.exact(0),'LT',False));self.assertFalse(cmp(V.exact(-1),V.exact(0),'LT',True))
        self.assertIsNone(calc([V(0xfffffffe,0xffffffff),V.exact(1)],lambda a,b:a+b))
        self.assertEqual(calc([V.exact(-1),V.exact(1)],lambda a,b:a+b),V.exact(0))
    def test_bitmask_and_funnel_loop_count(self):
        s=synthetic(['MOV R0, 256','MOV R1, 0','SHF.R.U32.HI R0, RZ, 1, R0','SHFL.BFLY PT, R2, R1, 1, 0x1f','ISETP.GT.U32.AND P0, PT, R0, 1, PT','@P0 BRA 0x20','EXIT']);e,_,_=run_lane(s,{},{});self.assertEqual(sum(e.values()),8)

class Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]
        for corpus,root in CORPORA.items():
            for r in json.loads((root/'retention_manifest.json').read_text())['rows']:
                try:q=derive(corpus,r,root)
                except Refusal as e:q={'status':'refused','reason':str(e)}
                cls.rows.append({**q,'oid':r['operator_id'],'cell':r['cell']})
    def test_grid_and_all_refusals(self):
        self.assertEqual(len(self.rows),93);self.assertEqual(sum(r['status']=='candidate_class_counts' for r in self.rows),84)
        self.assertEqual({r['reason'] for r in self.rows if r['status']=='refused'},{'unsupported control BRA.DIV'})
    def test_reduction_tree_analytical_sums(self):
        for r in self.rows:
            if r['oid']!='train_cuda_samples_reduction' or r['cell'].split('/')[1] not in ('c1','c2'):continue
            t=r['launch']['block'][0];b=r['launch']['grid'][0];c=r['classes']
            self.assertEqual(c['shared_store']['predicate_true_thread_instruction'],(2*t-1)*b)
            loads=2*(t-1)+1 if r['cell'].endswith('c1') else t-1
            self.assertEqual(c['shared_load']['predicate_true_thread_instruction'],loads*b)
            self.assertEqual(c['barrier']['predicate_true_thread_instruction'],t*(int(math.log2(t))+1)*b)
    def test_two_reductions_shuffle_schedule(self):
        for r in self.rows:
            if r['oid'] not in ('dev_triton_softmax','final_triton_layer_norm'):continue
            t=r['launch']['block'][0];cols=r['launch']['recorded_compile_arguments'][-2] if r['oid']=='final_triton_layer_norm' else r['launch']['recorded_compile_arguments'][-1];w=min(t//32,cols//32);b=r['launch']['grid'][0]
            self.assertEqual(r['classes']['shuffle']['predicate_true_thread_instruction'],(10+2*int(math.log2(w)))*t*b)
    def test_hash_tamper_refuses(self):
        root=CORPORA['cuda'];row=json.loads((root/'retention_manifest.json').read_text())['rows'][0];row=copy.deepcopy(row);row['cubin_sha256']='0'*64
        with self.assertRaisesRegex(Refusal,'hash mismatch'):verify(root,row)
    def test_unit_and_admission_distinction(self):
        for r in self.rows:
            if r['status']=='refused':continue
            self.assertFalse(r['scientifically_admitted']);self.assertFalse(r['all_opcode_counts_complete']);self.assertFalse(r['profiler_validated'])
            for c in r['classes'].values():self.assertIsNone(c['warp_issued_instruction'])
if __name__=='__main__':unittest.main()
