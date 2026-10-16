import copy
import math
import unittest

import strata_workloads as w
import strata_experiment as ex
from score_api import EngineProtocolError


WORK = [
    {'id':'W1','kind':'score','shape':'S','weight':.4,'candidates':[15,16,17,18],'quality':'structure/1'},
    {'id':'W2','kind':'conditional','shape':'L','weight':.4,'candidates':[15,16,17,18],
     'max_new':128,'explain_if':{'mode':'always'},'quality':'structure/1'},
    {'id':'W3','kind':'chat','shape':'S','weight':.2,'max_new':32,'quality':'structure/1'}]


def arm(ms=1000, branch='explain', n=128):
    return {'metrics':{'P1':{'prefill_ms':800},
        **{k+'1':{'workflow_observed_ms':ms,'branch':branch,'n':n,'finish':'length'} for k in ['W1','W2','W3']}}}


class WorkloadTests(unittest.TestCase):
    def spec(self):
        return {'workloads':copy.deepcopy(WORK),'baseline':{'config':{'ctx':4096}},
                'shapes':[{'id':'S','prompt_tokens':3896},{'id':'L','prompt_tokens':3817}]}

    def test_manifest_explicit_weights_shapes_and_quality(self):
        s=self.spec();w.validate(s)
        for key,value in [('weight',.8),('kind','unknown'),('quality','business-approved')]:
            bad=copy.deepcopy(s);bad['workloads'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):w.validate(bad)
        self.assertEqual(ex.benchmark(s)['version'],'business-workflows/1')

    def test_explanation_policy_is_explicit_and_not_low_mass(self):
        q=copy.deepcopy(WORK[1]);self.assertTrue(w.should_explain(q,15))
        q['explain_if']={'mode':'never'};self.assertFalse(w.should_explain(q,15))
        q['explain_if']={'mode':'choice-in','ids':[17]}
        self.assertTrue(w.should_explain(q,17));self.assertFalse(w.should_explain(q,15))

    def test_probabilities_zero_generation_and_full_vocab_mass(self):
        a={'tokens':[],'done':{'finish':'score'},'lines':['CANDLOGPROBS 3896 2 15:-2.0 16:-3.0']}
        p=w.probabilities(a,[15,16]);self.assertAlmostEqual(p['label_mass'],math.exp(-2)+math.exp(-3))
        self.assertFalse(p['calibrated']);self.assertEqual(p['choice'],15)
        for bad in [{'tokens':[1]},{'lines':['CANDLOGPROBS 3896 2 15:0.0 16:0.0']}]:
            with self.assertRaises((ValueError,EngineProtocolError)):w.probabilities({**a,**bad},[15,16])

    def test_weighted_latency_not_mixed_raw_tps(self):
        policy={'min_gain':.05}
        good=w.compare([arm(),arm(900),arm(900),arm()],policy,WORK)
        self.assertEqual(good['verdict'],'PASS');self.assertAlmostEqual(good['objective_gain_fraction'],.1)
        self.assertEqual(w.compare([arm(),arm(995),arm(995),arm()],policy,WORK)['verdict'],'NO-GO')

    def test_branch_or_actual_length_changes_cannot_fake_speedup(self):
        for b in [arm(500,'score-only',0),arm(500,n=64)]:
            self.assertEqual(w.compare([arm(),b,b,arm()],{'min_gain':.05},WORK)['verdict'],'NO-GO')


if __name__=='__main__':unittest.main()
