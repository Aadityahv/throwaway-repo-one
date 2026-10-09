"""CPU checks for census, folds, units, missing coverage and original-fit parity."""
import unittest
from unittest.mock import patch
import json
import numpy as np
import screen as S


class ScreenChecks(unittest.TestCase):
    def test_disjoint_instruction_census(self):
        w=dict(total_lane_instructions=30,families={k:dict(lane_instructions=v) for k,v in
               dict(global_load=4,global_store=2,integer_alu=5,fp32_fma=3,special_function=2,
                    shared_load=4,shuffle=1,control=6,move_const_special=3).items()})
        c=S.counts(w)
        self.assertEqual(c['nonmem'],24)
        self.assertEqual(sum(c[k] for k in ('integer','floating','sfu','shared','overhead')),24)
        self.assertEqual(c['arithmetic'],8)

    def test_incomplete_or_negative_census_refuses(self):
        for w in [dict(total_lane_instructions=3,families={'control':dict(lane_instructions=2)}),
                  dict(total_lane_instructions=-1,families={'control':dict(lane_instructions=-1)})]:
            with self.assertRaises(ValueError):S.counts(w)

    def test_unsupported_tier_refuses(self):
        with self.assertRaises(ValueError):S.feature_values({},10,'L1',.1)

    def synthetic_rows(self):
        out=[]
        for i in range(15):
            t=(i%3+1)*.001; l2=(i//3+1)*1e7; dram=1e7 if i%2 else 0
            out.append(dict(cell_id=str(i),group=str(i%3),supported=True,below=True,t=t,
                            p0=10,cap=600,energy=80*t+5e-10*l2+8e-10*dram,
                            values=dict(time=t,l2=l2,dram=dram)))
        return out

    def test_time_and_pj_units_recover_known_energy(self):
        rows=self.synthetic_rows();m=S.fit(rows,('traffic',0))
        np.testing.assert_allclose(m['coefficients'],[80,500,800],rtol=1e-10)
        np.testing.assert_allclose(S.predict(rows,m),[r['energy'] for r in rows],rtol=1e-10)

    def test_zero_training_activity_is_explicit(self):
        rows=self.synthetic_rows()
        for r in rows:r['values']['dram']=0
        m=S.fit(rows,('traffic',.01))
        self.assertIn('dram',m['zero_training_features'])
        self.assertEqual(m['coefficients'][2],0)

    def test_training_guard_and_unsupported_inference(self):
        rows=self.synthetic_rows();m=S.fit(rows,('traffic',0))
        bad=dict(rows[0],below=False)
        with self.assertRaises(ValueError):S.fit([bad],('traffic',0))
        with self.assertRaises(ValueError):S.predict([dict(rows[0],supported=False)],m)

    def test_full_development_grid_and_source_groups(self):
        rows=S.load()
        self.assertEqual(len(rows),132)
        self.assertEqual(sum(r['supported'] for r in rows),120)
        self.assertEqual(sum(r['supported'] and r['exact'] for r in rows),72)
        self.assertEqual(len({r['group'] for r in rows if r['supported']}),9)
        reductions=[r for r in rows if r['family']=='reduction']
        self.assertEqual(len({r['group'] for r in reductions}),1)

    def test_original_recipe_reproduction(self):
        rows=[r for r in S.load() if r['supported'] and r['below']]
        self.assertEqual(len(rows),73)
        m=S.fit(rows,('legacy',0))
        old=json.loads((S.HERE.parent/'unseen_kernels/energy_model_frozen.json').read_text())
        expected=[old['rates_pJ_per_unit'][k] for k in ('bytes_L2','bytes_DRAM','nonmem_lane_instructions','sfu_lane_instructions')]
        np.testing.assert_allclose(m['coefficients'],expected,rtol=1e-9,atol=1e-9)

    def test_inner_fit_excludes_validation_source(self):
        rows=self.synthetic_rows();origfit=S.fit;origpred=S.predict
        def fit(train,config):
            model=origfit(train,config);model['training_ids']=[r['cell_id'] for r in train];return model
        def predict(test,model):
            self.assertFalse(set(model['training_ids'])&{r['cell_id'] for r in test})
            self.assertEqual(len({r['group'] for r in test}),1)
            test_group=test[0]['group']
            self.assertFalse(any(r['group']==test_group for r in rows if r['cell_id'] in model['training_ids']))
            return origpred(test,model)
        with patch.object(S,'CONFIGS',[('traffic',0),('traffic',.01)]),patch.object(S,'fit',fit),patch.object(S,'predict',predict):
            selected,losses=S.choose(rows)
        self.assertEqual(selected,('traffic',0))
        self.assertEqual(len(losses),2)

    def test_missing_cells_count_as_failures(self):
        rows=self.synthetic_rows();p={r['cell_id']:r['energy'] for r in rows[:10]}
        d=S.summarize(rows,p)
        self.assertEqual(d['failures'],5)
        self.assertEqual(d['median_ape_pct'],0)
        self.assertEqual(d['failure_aware_p90_pct'],'failure')


if __name__=='__main__':unittest.main()
