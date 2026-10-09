"""Tests for the BRA.DIV collective extension (collective.py)."""
import collections,json,unittest
from pathlib import Path
import derive as D
from derive import Refusal,V,parse,CORPORA,sha
from collective import run_lane_div,check_warp_convergence,derive_collective

def synthetic(lines):
    return parse('\n'.join('/*%04x*/ %s;'%(i*16,line) for i,line in enumerate(lines)))
def lane(sites,tid,consts=None):
    arr=collections.Counter();e,v,u=run_lane_div(sites,consts or {},{'SR_TID.X':V.exact(tid)},arrivals=arr);return e,arr
# Mimics the CUDA warp reduction: lanes >=20 skip, others reach BRA.DIV then 2 shuffles; lane<5 additionally do an LDS.
PATH=['S2R R0, SR_TID.X','ISETP.GT.U32.AND P0, PT, R0, 19, PT','@P0 BRA 0x80','UMOV UR4, 0xffffffff','BRA.DIV UR4, 0x90',
      'SHFL.DOWN PT, R5, R4, 0x1, 0x1f','SHFL.DOWN PT, R6, R5, 0x2, 0x1f','EXIT','EXIT','WARPSYNC.COLLECTIVE R10, 0xa0']
class Divergence(unittest.TestCase):
    def test_per_lane_enumeration_counts_each_lane_path(self):
        # Branch predicate depends only on tid: 20 lanes run both shuffles, 12 skip; no BRA.DIV involvement here.
        sites=synthetic(['S2R R0, SR_TID.X','ISETP.GT.U32.AND P0, PT, R0, 19, PT','@P0 BRA 0x50','SHFL.DOWN PT, R5, R4, 0x1, 0x1f','SHFL.DOWN PT, R6, R5, 0x2, 0x1f','EXIT'])
        n=sum(sum(lane(sites,t)[0].values()) for t in range(32));self.assertEqual(n,40)
        sites=synthetic(['S2R R0, SR_TID.X','ISETP.LT.U32.AND P0, PT, R0, 5, PT','@P0 LDS.128 R2, [RZ]','@!P0 LDS R3, [RZ]','EXIT'])
        c=collections.Counter()
        for t in range(32):c.update(lane(sites,t)[0])
        self.assertEqual({k[2]:v for k,v in c.items()},{128:5,32:27})
    def test_bra_div_converged_fast_path_not_fallback(self):
        # Fully converged warp: BRA.DIV falls through, fallback (WARPSYNC.COLLECTIVE) never entered.
        sites=synthetic(['UMOV UR4, 0xffffffff','BRA.DIV UR4, 0x40','SHFL.DOWN PT, R5, R4, 0x1, 0x1f','EXIT','WARPSYNC.COLLECTIVE R10, 0x50'])
        arrs=[];n=0
        for t in range(32):
            e,a=lane(sites,t);n+=sum(e.values());arrs.append(a)
        self.assertEqual(n,32);check_warp_convergence(arrs,32)
        self.assertEqual(arrs[0],collections.Counter({(0x10,0xffffffff):1}))
    def test_divergent_arrival_refuses(self):
        # Only 20 of 32 lanes reach BRA.DIV: warp is divergent at the collective, so the cell must refuse.
        sites=synthetic(PATH);arrs=[lane(sites,t)[1] for t in range(32)]
        with self.assertRaisesRegex(Refusal,'diverges at BRA.DIV'):check_warp_convergence(arrs,32)
    def test_partial_warp_and_partial_mask_refuse(self):
        sites=synthetic(['UMOV UR4, 0xffffffff','BRA.DIV UR4, 0x40','EXIT','EXIT','EXIT']);arrs=[lane(sites,t)[1] for t in range(16)]
        with self.assertRaisesRegex(Refusal,'partial warp'):check_warp_convergence(arrs,16)
        sites=synthetic(['UMOV UR4, 0xffff','BRA.DIV UR4, 0x40','EXIT','EXIT','EXIT']);arrs=[lane(sites,t)[1] for t in range(32)]
        with self.assertRaisesRegex(Refusal,'full-warp mask'):check_warp_convergence(arrs,32)
    def test_fallback_block_still_refuses_if_reached(self):
        with self.assertRaisesRegex(Refusal,'unsupported control'):lane(synthetic(['WARPSYNC.COLLECTIVE R10, 0x20','EXIT']),0)

class DataDependent(unittest.TestCase):
    def test_data_dependent_branch_refuses(self):
        sites=synthetic(['LDG.E R0, [RZ]','ISETP.NE.U32.AND P0, PT, R0, RZ, PT','@P0 BRA 0x40','EXIT','EXIT'])
        with self.assertRaisesRegex(Refusal,'unknown control'):lane(sites,0)
    def test_data_dependent_bra_div_predicate_and_mask_refuse(self):
        sites=synthetic(['LDG.E R0, [RZ]','ISETP.NE.U32.AND P0, PT, R0, RZ, PT','UMOV UR4, 0xffffffff','@P0 BRA.DIV UR4, 0x40','EXIT'])
        with self.assertRaisesRegex(Refusal,'unknown control'):lane(sites,0)
        sites=synthetic(['LDG.E R0, [RZ]','MOV UR4, R0','BRA.DIV UR4, 0x40','EXIT'])
        with self.assertRaisesRegex(Refusal,'unknown BRA.DIV mask'):lane(sites,0)
    def test_data_dependent_branch_after_bra_div_still_refuses(self):
        sites=synthetic(['UMOV UR4, 0xffffffff','BRA.DIV UR4, 0x60','LDG.E R0, [RZ]','ISETP.NE.U32.AND P0, PT, R0, RZ, PT','@P0 BRA 0x60','EXIT','EXIT'])
        with self.assertRaisesRegex(Refusal,'unknown control'):lane(sites,0)
    def test_unknown_priced_predicate_still_refuses(self):
        with self.assertRaisesRegex(Refusal,'unknown target'):lane(synthetic(['FSETP.GT.AND P0, PT, R0, RZ, PT','@P0 MUFU.EX2 R1, R0','EXIT']),0)

class Corpus(unittest.TestCase):
    base=json.loads((Path(D.__file__).parent/'candidate_counts.json').read_text())
    def test_existing_derived_cells_unchanged(self):
        old={(r['operator_id'],r['cell']):r for r in self.base['rows']};n=0
        for corpus,root in CORPORA.items():
            for row in json.loads((root/'retention_manifest.json').read_text())['rows']:
                b=old[(row['operator_id'],row['cell'])]
                if b['status']!='candidate_class_counts':continue
                c=derive_collective(corpus,row,root);n+=1
                self.assertEqual(json.dumps(c['classes'],sort_keys=True),json.dumps(b['classes'],sort_keys=True))
                self.assertEqual([(s['pc'],s['class'],s['width_bits'],s['predicate_true_thread_instruction']) for s in c['sites']],[(s['pc'],s['class'],s['width_bits'],s['predicate_true_thread_instruction']) for s in b['sites']])
                self.assertEqual(c['divergence_model']['bra_div_sites'],[])
        self.assertEqual(n,84)
    def test_derive_py_untouched_and_frozen_hashes(self):
        self.assertEqual(sha(D.__file__),self.base['implementation_sha256'])
        fz=json.loads((Path(D.__file__).parent/'collective_counts.json').read_text())
        self.assertEqual(fz['derive_py_sha256'],self.base['implementation_sha256'])
        self.assertEqual(fz['total_cells'],9);self.assertTrue(all(r['status']=='candidate_class_counts' for r in fz['rows']))
        for r in fz['rows']:
            self.assertEqual(list(r['shuffle_modes_lane_invocations']),['SHFL.DOWN']);self.assertFalse(r['divergence_model']['fallback_block_entered'])
    def test_nine_collective_cells_derive_with_analytic_shuffle_count(self):
        fz=json.loads((Path(D.__file__).parent/'collective_counts.json').read_text())
        for r in fz['rows']:  # one warp per block runs the 5-step fast-path shuffle tree
            self.assertEqual(r['classes']['shuffle']['predicate_true_thread_instruction'],5*32*r['launch']['grid'][0])
if __name__=='__main__':unittest.main()
