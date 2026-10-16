import threading
import time
import unittest
from strata_mixed_load import bounded,summarize


class MixedTests(unittest.TestCase):
    def test_serial_and_backpressure(self):
        state={'active':0,'peak':0}
        def work(r):
            state['active']+=1;state['peak']=max(state['peak'],state['active'])
            time.sleep(.02);state['active']-=1
            return dict(r,generated_tokens=1,invalid_reason=None)
        rows,wall=bounded([{'sample_id':str(i)} for i in range(12)],work,clients=8,capacity=1)
        self.assertEqual(state['peak'],1)
        self.assertEqual(len(rows),12)
        self.assertTrue(any(r['http_status']==503 for r in rows))
        m=summarize(rows,wall)
        self.assertEqual(m['submitted'],12)
        self.assertEqual(m['completed']+m['failed'],12)
        self.assertIsNone(m['quality_qualified_per_s'])

    def test_one_client_no_loss(self):
        rows,_=bounded([{'sample_id':str(i)} for i in range(4)],lambda r:dict(r,invalid_reason=None),clients=1,capacity=1)
        self.assertEqual([r['submission_index'] for r in rows],list(range(4)))
        self.assertTrue(all(r['http_status']==200 for r in rows))

    def test_failure_stops(self):
        def fail(r):raise ValueError('unsafe')
        with self.assertRaises(RuntimeError):bounded([{'sample_id':'x'}],fail,clients=1,capacity=1)

    def test_bounds(self):
        for c,q in [(0,1),(9,1),(1,0),(True,1)]:
            with self.assertRaises(ValueError):bounded([{}],lambda r:r,clients=c,capacity=q)


if __name__=='__main__':unittest.main()
