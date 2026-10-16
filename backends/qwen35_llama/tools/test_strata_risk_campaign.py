import io
import json
from pathlib import Path
import tempfile
import unittest
from strata_risk_campaign import select,templates,instrument,completed_rows
from strata_risk_eval import write


class CampaignTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=str(i),gold_risk=i//4,scenario='a' if i%2 else 'b',
                messages=[{'role':'user','content':'unique '+chr(65+i)*30}]) for i in range(16)]

    def test_frozen_balanced_selection(self):
        rows=self.rows();lengths=[{'eligible':True,'A0_tokens':500+i*100} for i in range(16)]
        a,_,audit=select(rows,lengths,per_risk=3,seed=1)
        b,_,_=select(rows,lengths,per_risk=3,seed=1)
        self.assertEqual(a,b);self.assertEqual(len(set(a)),12)
        self.assertEqual([sum(rows[i]['gold_risk']==g for i in a) for g in range(4)],[3]*4)
        self.assertEqual(audit['duplicate_content'],0)

    def test_duplicate_and_scarcity_refusal(self):
        rows=self.rows();lengths=[{'eligible':True,'A0_tokens':500} for _ in rows]
        rows[1]['messages']=rows[0]['messages']
        with self.assertRaises(ValueError):select(rows,lengths,per_risk=3)
        with self.assertRaises(ValueError):select(self.rows(),lengths,per_risk=5)

    def test_template_digits_normalize(self):
        rows=[{'messages':[{'content':'shared template number '+str(i)+' and a sufficiently long normal paragraph'}]} for i in range(3)]
        groups,_,audit=templates(rows)
        self.assertEqual(len(set(groups)),1)
        self.assertGreater(audit['confirmed_edges'],0)

    def test_corrupted_completed_checkpoint_refused(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'private-results.json',[{'x':1}])
            sha=hashlib.sha256((p/'private-results.json').read_bytes()).hexdigest()
            write(p/'completed.json',{'files':{'private-results.json':sha}})
            self.assertEqual(completed_rows(p),[{'x':1}])
            write(p/'private-results.json',[])
            with self.assertRaises(ValueError):completed_rows(p)

    def test_observation_tap_keeps_strict_parser(self):
        import score_api
        from unittest.mock import patch
        # Parser tap coverage; the real existing parser has separate wire tests.
        class Link:
            def __init__(self):self.lines=[]
            def send(self,line):self.lines=["T 18","DONE 1 2 3.0 4.0 stop 0 0 2 0 0 0 0 0 0 0"]
            def _take_until(self,*args):return self.lines.pop(0)
        link=Link();instrument(link);link.send('GEN 1 cont=1')
        self.assertEqual(link._take_until(1,1),'T 18')
        with patch.object(score_api,'_parse_done',return_value={'produced':1,'decode_ms':4.0}):
            self.assertTrue(link._take_until(1,1).startswith('DONE'))
        self.assertEqual(len(link.measurements),1)
        self.assertEqual(link.measurements[0]['done']['produced'],1)


if __name__=='__main__':unittest.main()
