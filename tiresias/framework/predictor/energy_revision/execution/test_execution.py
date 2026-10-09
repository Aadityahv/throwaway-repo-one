import copy
import json
import unittest
from pathlib import Path

from predict import predict_rows
import fit as F
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((HERE / 'exposed_parity_request.json').read_text())
        self.profile = json.loads((HERE / 'reference_profile.json').read_text())

    def test_original_full_grid_reproduction(self):
        frozen = json.loads((ROOT / 'tiresias/framework/evaluation_data/measured/kernel_sets/predictions_runtime.json').read_text())['rows']
        out = predict_rows(self.request, self.profile)
        self.assertEqual(len(out['rows']), 48)
        for r in out['rows']:
            expected = frozen[r['cell_id']]['energy_j']['component_predicted']
            self.assertLess(abs(r['methods']['source_proxy_component']['energy_j'] / expected - 1), 1e-12)
            self.assertEqual(r['predicted_runtime_s'], frozen[r['cell_id']]['predicted_runtime_s'])

    def test_missing_source_convention_is_failure_not_fallback(self):
        request = copy.deepcopy(self.request)
        del request['rows'][0]['source_operations']
        request['rows'][0]['work'] = {'total_lane_instructions': 1000}
        out = predict_rows(request, self.profile)
        self.assertEqual(len(out['rows']), 48)
        self.assertEqual(out['rows'][0]['methods']['source_proxy_component']['status'], 'unsupported')
        self.assertEqual(out['rows'][0]['methods']['constant_power']['status'], 'ok')

    def test_invalid_runtime_or_duplicate_grid_refuses(self):
        for value in (0, -1, float('nan'), True):
            request = copy.deepcopy(self.request); request['rows'][0]['predicted_runtime_s'] = value
            with self.assertRaises(ValueError): predict_rows(request, self.profile)
        self.request['rows'].append(self.request['rows'][0])
        with self.assertRaises(ValueError): predict_rows(self.request, self.profile)

    def test_no_unadmitted_class_candidate(self):
        out = predict_rows(self.request, self.profile, {'status': 'uncalibrated'})
        for row in out['rows']:
            self.assertEqual(row['methods']['class_aware_component']['status'], 'unsupported')

    def test_static_count_refusals_preserve_reference_and_constant_coverage(self):
        request=json.loads((HERE/'exposed_class_activity_request.json').read_text())
        profile=dict(status='calibrated_and_frozen',cap_w=600,coefficient_unit_system='W_and_J_per_activity',
            coefficients=dict(base_time=100,fp32_work=1e-11,other_non_global=1e-11,special_function=1e-11,lookup_bytes=1e-11,dram_bytes=1e-11),
            matched_pooled_coefficients=dict(base_time=100,pooled_activity=1e-11,lookup_bytes=1e-11,dram_bytes=1e-11),matched_constant_power_w=200)
        out=predict_rows(request,self.profile,profile)
        count=lambda name:sum(r['methods'][name]['status']=='ok' for r in out['rows'])
        self.assertEqual(len(out['rows']),48)
        self.assertEqual(count('class_aware_component'),40)
        self.assertEqual(count('budget_matched_pooled_component'),40)
        for name in ('source_proxy_component','constant_power','budget_matched_constant_power'):
            self.assertEqual(count(name),48)


class CalibrationChecks(unittest.TestCase):
    def fixtures(self):
        """Synthetic test data ONLY; no file/profile is emitted as measured evidence."""
        beta = np.array([100, 2e-11, 1e-11, 5e-11, 5e-11, 7e-11])
        design = dict(schema='energy_component_calibration_design/1', status='compiled_counts_and_correctness_frozen', cap_w=600, rows=[])
        acquired = dict(schema='energy_component_calibration_measurements/1', target_role='calibration_only', rows=[])
        for i, did in enumerate(F.FIT_IDS + F.HOLD_IDS):
            fields = did.split('/'); foot = fields[-1]; mixed = did in F.HOLD_IDS
            dose = 1 if len(fields) < 3 or fields[1] == 'low' else 4
            B = {'small_candidate': 1e6, 'l2_candidate': 16e6, 'dram_candidate': 256e6}[foot]
            n = B / 4; kind = F.MIXES.index(fields[0]) if not mixed else (i - 24) % 4
            A = n * 64 * dose if kind == 1 else n * 8 * dose if kind == 2 else 0
            S = n * 8 * dose if kind == 2 else 0
            H = n * 64 * dose if kind == 3 else n * (4 + dose) + 1e7
            if mixed: A += n * 7; S += n * 3; H += n * 2
            tier = 'DRAM' if foot == 'dram_candidate' else 'L2'
            families = dict(fp32_add=A, special_function=S, integer_alu=H, global_load=n, global_store=1)
            work = dict(all_counts_exact=True, total_lane_instructions=sum(families.values()), families={k:dict(lane_instructions=v) for k,v in families.items()})
            t = .0001 * (1 + kind ** 2 + dose ** 2) + B / 1e12 + (A + S + H) / 1e13
            energy = np.dot([t, A, H, S, B, B if tier == 'DRAM' else 0], beta)
            s = dict(design_id=did, role='heldout' if mixed else 'fit', tier=tier, work=work, logical_bytes=B,
                     count_status='exact', abi_status='verified', source_sha256='a'*64, binary_sha256='b'*64, count_evidence_sha256='c'*64)
            design['rows'].append(s)
            acquired['rows'].append(dict(design_id=did,role=s['role'],status='accepted',attempts=1,correctness_pass=True,
                source_sha256=s['source_sha256'],binary_sha256=s['binary_sha256'],count_evidence_sha256=s['count_evidence_sha256'],trace_sha256='d'*64,
                counted_runtime_s_per_launch=t,energy_j_per_launch=float(energy),counted_launches=1000,counted_interval_s=t*1000,precondition_cuda_s=90))
        acquired['design_sha256'] = F.fingerprint(design)
        policy = json.loads((HERE / 'fit_policy.json').read_text())
        return design, acquired, policy, beta

    def test_known_coefficients_units_and_common_budget(self):
        d, a, p, beta = self.fixtures()
        r = F.fit_profiles(d, a, p)
        self.assertEqual(r['status'], 'calibrated_and_frozen')
        np.testing.assert_allclose(list(r['coefficients'].values()), beta, rtol=1e-9, atol=1e-20)
        self.assertLess(r['admission']['heldout']['class_aware']['max_ape_pct'], 1e-8)
        self.assertEqual(r['calibration_budget']['acquired_windows'], 30)
        self.assertEqual(r['calibration_budget']['precondition_cuda_s'], 2700)
        expected = np.mean([r['energy_j_per_launch'] / r['counted_runtime_s_per_launch'] for r in a['rows'][:24]])
        self.assertAlmostEqual(r['matched_constant_power_w'], expected)

    def test_heldout_labels_do_not_change_coefficients(self):
        d, a, p, _ = self.fixtures(); original = F.fit_profiles(d, a, p)
        for r in a['rows'][24:]: r['energy_j_per_launch'] *= 1.1
        changed = F.fit_profiles(d, a, p)
        self.assertEqual(original['coefficients'], changed['coefficients'])
        self.assertEqual(original['matched_pooled_coefficients'], changed['matched_pooled_coefficients'])
        self.assertEqual(original['matched_constant_power_w'], changed['matched_constant_power_w'])

    def test_synthetic_or_target_labels_refuse(self):
        d, a, p, _ = self.fixtures(); a['synthetic'] = True
        with self.assertRaises(ValueError): F.fit_profiles(d, a, p)
        del a['synthetic']; a['target_role'] = 'fresh_target'
        with self.assertRaises(ValueError): F.fit_profiles(d, a, p)

    def test_missing_failed_retried_or_capped_slot_refuses(self):
        for change in ('missing', 'rejected', 'retry', 'capped', 'provenance', 'normalization'):
            d, a, p, _ = self.fixtures(); r = a['rows'][0]
            if change == 'missing': a['rows'].pop()
            if change == 'rejected': r['status'] = 'rejected'
            if change == 'retry': r['attempts'] = 2
            if change == 'capped': r['energy_j_per_launch'] = 600 * r['counted_runtime_s_per_launch']
            if change == 'provenance': r['binary_sha256'] = 'e'*64
            if change == 'normalization': r['counted_launches'] = 100
            with self.subTest(change=change), self.assertRaises(ValueError): F.fit_profiles(d, a, p)

    def test_unexcited_and_aliased_designs_refuse(self):
        x = np.ones((24, 6))
        with self.assertRaises(ValueError): F.audit_matrix(x, F.C.FEATURES, 10000)
        x[:, 3] = 0
        with self.assertRaises(ValueError): F.audit_matrix(x, F.C.FEATURES, 10000)

    def test_bad_transfer_rejects_profile_without_posthoc_refit(self):
        d, a, p, _ = self.fixtures(); original = F.fit_profiles(d, a, p)
        for r in a['rows'][24:]: r['energy_j_per_launch'] *= .5
        changed = F.fit_profiles(d, a, p)
        self.assertEqual(changed['status'], 'rejected')
        self.assertEqual(original['coefficients'], changed['coefficients'])


if __name__ == '__main__':
    unittest.main()
