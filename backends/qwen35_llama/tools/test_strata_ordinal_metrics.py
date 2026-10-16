import unittest
from strata_ordinal_metrics import evaluate, paired


def row(i, g, p, **kw):
    return dict(sample_id=str(i), conversation_id=str(i // 2), gold_risk=g, parsed_risk=p, **kw)


class OrdinalTests(unittest.TestCase):
    def test_direction_and_missing(self):
        m = evaluate([row(0, 3, 2), row(1, 3, 0), row(2, 3, None), row(3, 0, 0)])
        self.assertEqual(m['risk3_to2'], 1)
        self.assertEqual(m['risk3_recall'], 0)
        self.assertEqual(m['risk3_severe_underestimate_rate'], 1/3)
        self.assertEqual(m['risk3_invalid'], 1)
        self.assertEqual(m['exact_accuracy'], .25)
        self.assertEqual(m['within_one_accuracy'], .5)
        self.assertEqual(m['mae_failure_penalty_3'], 7/4)
        self.assertEqual(m['severe_or_invalid_rate'], .5)

    def test_all_invalid(self):
        m = evaluate([row(0, 3, None, invalid_reason='timeout')])
        self.assertEqual(m['exact_accuracy'], 0)
        self.assertIsNone(m['mae_valid_only'])
        self.assertEqual(m['mae_failure_penalty_3'], 3)
        self.assertEqual(m['flow_failure_rate'], 1)

    def test_bool_not_risk(self):
        self.assertEqual(evaluate([row(0, 0, True)])['invalid'], 1)
        with self.assertRaises(ValueError): evaluate([row(0, True, 0)])

    def test_empty(self):
        self.assertIsNone(evaluate([])['exact_accuracy'])

    def test_pair_clusters(self):
        a = [row(i, 3, 2) for i in range(4)]
        b = [row(i, 3, 3) for i in range(4)]
        m = paired(a, list(reversed(b)), draws=100)
        self.assertEqual(m['clusters'], 2)
        self.assertEqual(m['corrected_by_b'], 4)
        self.assertEqual(m['deltas']['exact_accuracy']['ci95'], [1, 1])
        self.assertEqual(m['status'], 'INCONCLUSIVE')

    def test_pair_refusals(self):
        with self.assertRaises(ValueError): paired([row(0, 1, 1)], [])
        with self.assertRaises(ValueError): paired([row(0, 1, 1)], [row(0, 2, 1)])
        with self.assertRaises(ValueError): paired([], [], draws=1)


if __name__ == '__main__': unittest.main()
