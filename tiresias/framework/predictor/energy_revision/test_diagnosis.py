import json
import unittest
from pathlib import Path
import component_candidate as C


class DiagnosticChecks(unittest.TestCase):
    def test_full_exposed_grid_and_exact_error_accounting(self):
        d=json.loads((Path(__file__).parent/'matched_diagnostic.json').read_text())
        self.assertEqual(len(d['per_cell']),48)
        self.assertLess(d['max_relative_prediction_reproduction_error'],1e-12)
        for r in d['per_cell']:
            self.assertAlmostEqual(r['own_predicted_j']-r['actual_j'],r['own_runtime_error_j']+r['own_energy_side_residual_j'])
            self.assertAlmostEqual(r['component_predicted_j']-r['actual_j'],r['component_runtime_error_j']+r['component_energy_side_residual_j'])
            self.assertAlmostEqual(r['difference_j'],sum(r['difference_parts_j'].values()))

    def test_constant_uses_same_runtime_and_cap(self):
        d=json.loads((Path(__file__).parent/'matched_diagnostic.json').read_text())
        p=d['calibration_budget']['constant_power_w']
        for r in d['per_cell']:
            self.assertAlmostEqual(r['constant_predicted_j'],p*r['predicted_runtime_s'])
            self.assertAlmostEqual(r['constant_measured_j'],p*r['measured_short_runtime_s'])

    def work(self):
        f=dict(fp32_fma=10,fp32_add=4,fp32_mul=2,integer_alu=7,control=3,special_function=2,global_load=1,global_store=1)
        return dict(all_counts_exact=True,total_lane_instructions=sum(f.values()),families={k:dict(lane_instructions=v) for k,v in f.items()})

    def test_fma_weighting_and_disjoint_accounting(self):
        x=C.activity(self.work(),128,'DRAM')
        self.assertEqual(x['fp32_work'],26)
        self.assertEqual(x['other_non_global'],10)
        self.assertEqual(x['special_function'],2)
        self.assertEqual(x['lookup_bytes'],128)
        self.assertEqual(x['dram_bytes'],128)

    def test_uncalibrated_or_inexact_inference_refuses(self):
        with self.assertRaises(ValueError):C.predict(self.work(),128,'L2',.01,None)
        w=self.work();w['all_counts_exact']=False
        with self.assertRaises(ValueError):C.activity(w,128,'L2')
        with self.assertRaises(ValueError):C.activity(self.work(),128,'L1')

    def test_class_coefficients_cannot_reuse_legacy_proxy(self):
        p=dict(status='calibrated_and_frozen',coefficient_unit_system='W_and_J_per_activity',cap_w=600,coefficients={'e_inst':1})
        with self.assertRaises(ValueError):C.predict(self.work(),128,'L2',.01,p)

    def test_explicit_units_cap_and_sfu_single_charge(self):
        p=dict(status='calibrated_and_frozen',cap_w=100,coefficient_unit_system='W_and_J_per_activity',
               coefficients=dict(base_time=10,fp32_work=.001,other_non_global=.002,
                                 special_function=.003,lookup_bytes=.004,dram_bytes=.005))
        d=C.predict(self.work(),128,'DRAM',.01,p)
        self.assertAlmostEqual(d['uncapped_j'],1.304)
        self.assertEqual(d['energy_j'],1)
        self.assertEqual(d['components_j']['special_function'],.006)
        del p['coefficient_unit_system']
        with self.assertRaises(ValueError):C.predict(self.work(),128,'L2',.01,p)


if __name__=='__main__':unittest.main()
