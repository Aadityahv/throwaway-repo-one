"""Gate of the regenerated bank tables: for every cell that exists in both the frozen table and its wavefront-histogram regeneration, the cost recomputed from the histogram with floor 2.0 and slope 1.0
must equal the cost stored in the frozen table, and the portable predictor with the committed constants must reproduce the frozen predictions when run on the regenerated table."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; SR = HERE.parent
sys.path.insert(0, str(HERE))
from cal import portable_predict as PP  # noqa: E402

PAIRS = (('bank_conflicts_fresh_e.json', 'bank_conflicts_fresh_e_wavefront.json'), ('bank_conflicts_unseen.json', 'bank_conflicts_unseen_wavefront.json'))


class Regen(unittest.TestCase):
    def test_histogram_cost_equals_frozen_cost(self):
        checked = 0
        for frozen_name, new_name in PAIRS:
            f, n = SR / 'bank' / frozen_name, SR / 'bank' / new_name
            if not n.exists(): continue
            frozen = json.loads(f.read_text())['rows']; new = json.loads(n.read_text())['rows']
            self.assertEqual(set(frozen), set(new), new_name)
            re = PP.recost_bank(new, 2.0, 1.0)
            for cid, row in frozen.items():
                if 'kernels' not in row: continue
                for k, k2 in zip(row['kernels'], re[cid]['kernels']):
                    for p, p2 in zip(k['phases'], k2['phases']):
                        a, b = p['shared']['shared_cost_cycles'], p2['shared']['shared_cost_cycles']
                        if a is None: self.assertIsNone(b)
                        else: self.assertAlmostEqual(a, b, places=6, msg=cid); checked += 1
        if not checked: self.skipTest('wavefront tables not generated yet (python3 calibrate/tools/regen_bank_wavefront.py e|unseen)')

    def test_set_e_portable_prediction_on_regenerated_table_reproduces_frozen(self):
        n = SR / 'bank' / 'bank_conflicts_fresh_e_wavefront.json'
        if not n.exists(): self.skipTest('set-E wavefront table not generated yet')
        C = SR / 'constants'
        c = dict(stream=json.loads((C / 'stream_constants.json').read_text())['constants'], micro=json.loads((C / 'microbench_constants_v2.json').read_text()), v3=json.loads((C / 'v3_constants.json').read_text()),
                 v3c=json.loads((C / 'v3c_constants.json').read_text()), v3e=json.loads((C / 'v3e_constants.json').read_text()), overlap=json.loads((C / 'overlap_constants.json').read_text()),
                 smem=dict(floor_cycles=2.0, degree_slope_cycles=1.0))          # re-cost active, with the committed rule
        d = SR / 'fresh_e'
        got = PP.predict(json.loads((d / 'features_fresh_e.json').read_text()), json.loads((d / 'phases_fresh_e.json').read_text())['rows'],
                         (lambda u: u.get('rows', u))(json.loads((d / 'phases_unique_fresh_e.json').read_text())), json.loads(n.read_text())['rows'], c, 188)
        ref = json.loads((d / 'predictions_fresh_e_v3i.json').read_text())
        for cid in ref:
            a, b = got[cid].get('primary_s'), ref[cid].get('primary_s')
            self.assertEqual(a is None, b is None)
            if a is not None: self.assertAlmostEqual(a / b, 1.0, places=9, msg=cid)


if __name__ == '__main__':
    unittest.main()
