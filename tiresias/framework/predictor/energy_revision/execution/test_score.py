import unittest
import score


class CoverageChecks(unittest.TestCase):
    def fixture(self):
        p=dict(schema='runtime_matched_static_energy/1',runtime_vector_sha256='fixture',rows=[
            dict(cell_id='a',predicted_runtime_s=1,methods=dict(reference=dict(status='ok',energy_j=100),candidate=dict(status='ok',energy_j=90))),
            dict(cell_id='b',predicted_runtime_s=1,methods=dict(reference=dict(status='ok',energy_j=590),candidate=dict(status='unsupported',reason='inexact')))])
        l=dict(schema='energy_evaluation_labels/1',exposure='already_exposed_diagnostic',rows=[
            dict(cell_id='b',family='second',energy_j=590,measured_runtime_s=1),dict(cell_id='a',family='first',energy_j=90,measured_runtime_s=1)])
        return p,l

    def test_refusal_remains_full_grid_failure(self):
        p,l=self.fixture();r=score.score(p,l,600)
        self.assertEqual(r['requested'],2);self.assertEqual(r['below_cap_requested'],1)
        self.assertEqual(r['summaries']['candidate']['supported'],1)
        self.assertFalse(r['summaries']['candidate']['full_grid_win_eligible'])
        self.assertEqual(r['summaries']['candidate']['all_supported']['median_ape_pct'],0)

    def test_no_missing_label_exclusion(self):
        p,l=self.fixture();l['rows'].pop()
        with self.assertRaises(ValueError):score.score(p,l,600)

    def test_no_silent_missing_method(self):
        p,l=self.fixture();del p['rows'][1]['methods']['candidate']
        with self.assertRaises(ValueError):score.score(p,l,600)


if __name__=='__main__':unittest.main()
