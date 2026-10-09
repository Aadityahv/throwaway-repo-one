"""The portable wrapper, fed the committed Blackwell constants and the committed bank tables, must reproduce the frozen predictions of the current model (set E and set D)."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; SR = HERE.parent
sys.path.insert(0, str(HERE))
from cal import portable_predict as PP  # noqa: E402


def sets():
    for d, f, p, u, b in (('fresh_e', 'features_fresh_e.json', 'phases_fresh_e.json', 'phases_unique_fresh_e.json', 'bank_conflicts_fresh_e.json'),
                          ('fresh_d', 'features_fresh_d.json', 'phases_fresh_d.json', 'phases_unique_fresh_d.json', 'bank_conflicts_fresh_d.json')):
        yield d, json.loads((SR / d / f).read_text()), json.loads((SR / d / p).read_text())['rows'], json.loads((SR / d / u).read_text()), json.loads((SR / 'bank' / b).read_text())['rows']


class Portable(unittest.TestCase):
    def setUp(self):
        C = SR / 'constants'
        self.c = dict(stream=json.loads((C / 'stream_constants.json').read_text())['constants'], micro=json.loads((C / 'microbench_constants_v2.json').read_text()),
                      v3=json.loads((C / 'v3_constants.json').read_text()), v3c=json.loads((C / 'v3c_constants.json').read_text()), v3e=json.loads((C / 'v3e_constants.json').read_text()),
                      overlap=json.loads((C / 'overlap_constants.json').read_text()), smem=None)

    def test_reproduces_frozen_predictions_without_smem_recost(self):
        for name, feats, phases, uniq, bank in sets():
            uniq = uniq.get('rows', uniq)
            got = PP.predict(feats, phases, uniq, bank, self.c, 188)
            if name == 'fresh_e': ref = json.loads((SR / name / 'predictions_fresh_e_v3i.json').read_text())     # frozen before timing
            else:       # set D froze four earlier models only: compare with the unmodified predictor modules called directly
                import predict_runtime_v2 as V2, predict_runtime_v3i as V3I
                K = V2.make_constants(self.c['stream'], self.c['micro']); ref = V3I.build(feats, phases, uniq, bank, K, self.c['v3'], self.c['v3c'], self.c['v3e'], self.c['overlap'])
            self.assertEqual(set(got), set(ref))
            for cid in ref:
                a, b = got[cid].get('primary_s'), ref[cid].get('primary_s')
                self.assertEqual(a is None, b is None, cid)
                if a is not None: self.assertAlmostEqual(a / b, 1.0, places=12, msg=cid)

    def test_recost_with_committed_rule_equals_stored_cost(self):
        name, feats, phases, uniq, bank = [s for s in sets() if s[0] == 'fresh_d'][0]      # set D tables carry the wavefront histograms
        re = PP.recost_bank(bank, 2.0, 1.0)
        n = 0
        for cid, row in bank.items():
            for k, k2 in zip(row['kernels'], re[cid]['kernels']):
                for p, p2 in zip(k['phases'], k2['phases']):
                    a, b = p['shared']['shared_cost_cycles'], p2['shared']['shared_cost_cycles']
                    if a is None: self.assertIsNone(b)
                    else: self.assertAlmostEqual(a, b, places=6); n += 1
        self.assertGreater(n, 50)

    def test_smem_constants_change_the_prediction(self):
        name, feats, phases, uniq, bank = [s for s in sets() if s[0] == 'fresh_d'][0]; uniq = uniq.get('rows', uniq)
        base = PP.predict(feats, phases, uniq, bank, self.c, 188)
        c2 = dict(self.c, smem=dict(floor_cycles=1.0, degree_slope_cycles=1.0))
        alt = PP.predict(feats, phases, uniq, bank, c2, 188)
        diff = [cid for cid in base if base[cid].get('primary_s') and abs(alt[cid]['primary_s'] / base[cid]['primary_s'] - 1) > 1e-6]
        self.assertTrue(diff)

    def test_originals_are_restored_after_a_call(self):
        import predict_runtime_v3 as V3, predict_runtime_v3f as V3F
        a, b = V3.mlp_seconds, V3F.mlp_seconds
        name, feats, phases, uniq, bank = next(sets()); PP.predict(feats, phases, uniq.get('rows', uniq), bank, self.c, 188)
        self.assertIs(V3.mlp_seconds, a); self.assertIs(V3F.mlp_seconds, b)

    def test_other_sm_count_changes_memory_bound_phase_prediction(self):
        name, feats, phases, uniq, bank = next(sets()); uniq = uniq.get('rows', uniq)
        a = PP.predict(feats, phases, uniq, bank, self.c, 188); b = PP.predict(feats, phases, uniq, bank, self.c, 108)
        self.assertTrue(any(a[c].get('primary_s') and abs(a[c]['primary_s'] / b[c]['primary_s'] - 1) > 1e-9 for c in a if b[c].get('primary_s')))


if __name__ == '__main__':
    unittest.main()
