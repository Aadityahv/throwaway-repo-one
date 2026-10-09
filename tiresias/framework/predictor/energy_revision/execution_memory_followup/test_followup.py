"""Synthetic in-memory provenance checks; never emits measurement evidence."""
import copy
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'execution'))
import test_execution as T
import fit_followup as F


class ReplacementChecks(unittest.TestCase):
    def fixture(self):
        d,a,p,_=T.CalibrationChecks().fixtures();d['certificates']={};a.update(complete=True,total_wall_s=4000)
        for i,s in enumerate(d['rows']):
            s.update(dose=16,n=1024,grid=[188,1,1],block=[256,1,1])
            if s['design_id'] in F.REPLACED:a['rows'][i]['energy_j_per_launch']=583*a['rows'][i]['counted_runtime_s_per_launch']
        a['design_sha256']=F.BASE.fingerprint(d)
        nd=copy.deepcopy(d);nd['rows']=[s for s in nd['rows'] if s['design_id'] in F.REPLACED]
        na=copy.deepcopy(a);na['rows']=[r for r in na['rows'] if r['design_id'] in F.REPLACED]
        for s,r in zip(nd['rows'],na['rows']):
            s.update(source_sha256='e'*64,binary_sha256='f'*64,count_evidence_sha256='1'*64,dependent_checksum_steps_per_memory_load=9)
            r.update(source_sha256='e'*64,binary_sha256='f'*64,count_evidence_sha256='1'*64,energy_j_per_launch=400*r['counted_runtime_s_per_launch'])
        na['design_sha256']=F.BASE.fingerprint(nd)
        return d,a,nd,na,p

    def test_retired_data_retained_and_full_grid_assembled(self):
        d,a,nd,na,p=self.fixture();merged,acq,retired=F.assemble(d,a,nd,na,p)
        self.assertEqual(len(merged['rows']),30);self.assertEqual(len(acq['rows']),30);self.assertEqual(len(retired),2)
        self.assertEqual(sum(s['source_sha256']=='e'*64 for s in merged['rows']),2)
        self.assertEqual(acq['total_wall_s'],8000)

    def test_other_cap_failure_refuses_proposal(self):
        d,a,nd,na,p=self.fixture();a['rows'][0]['energy_j_per_launch']=583*a['rows'][0]['counted_runtime_s_per_launch']
        with self.assertRaises(ValueError):F.assemble(d,a,nd,na,p)

    def test_same_source_retry_refuses(self):
        d,a,nd,na,p=self.fixture()
        for s in nd['rows']:s['source_sha256']='a'*64
        na['design_sha256']=F.BASE.fingerprint(nd)
        with self.assertRaises(ValueError):F.assemble(d,a,nd,na,p)

    def test_geometry_change_refuses(self):
        d,a,nd,na,p=self.fixture();nd['rows'][0]['block']=[128,1,1];na['design_sha256']=F.BASE.fingerprint(nd)
        with self.assertRaises(ValueError):F.assemble(d,a,nd,na,p)


if __name__=='__main__':unittest.main()
