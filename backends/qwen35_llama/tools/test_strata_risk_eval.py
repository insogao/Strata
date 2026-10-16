import json
from pathlib import Path
import tempfile
import unittest
from strata_risk_eval import Client,audit,dataset,parse_generated,report,split,tokenizer_audit,visible_generation


def sample(i=0):
    return dict(sample_id=str(i),conversation_id=str(i),messages=[{'role':'user','content':'synthetic '+str(i)}],
        gold_risk=3,gold_status='provisional',scenario='other',source_id='unit-test',
        label_policy_version='test/1',reference_kind='synthetic_design_label')


class RiskTests(unittest.TestCase):
    def test_strict_parse(self):
        self.assertEqual(parse_generated(' 3\n','A1'),3)
        for s in ['3 because','{"risk":3}','03','-1','3.0']:self.assertIsNone(parse_generated(s,'A1'))
        self.assertEqual(parse_generated('{"risk":3,"violation":true}','A0'),3)
        for s in ['{"risk":3,"violation":false}','{"risk":true,"violation":false}','[]']:
            self.assertIsNone(parse_generated(s,'A0'))

    def test_no_gold(self):
        self.assertEqual(audit([sample()])['accuracy_status'],'BLOCKED_NO_GOLD_DATA')

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl';s=sample()
            p.write_text(json.dumps(s));self.assertEqual(len(dataset(p)),1)
            s['reference_kind']='human_adjudicated';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):dataset(p)
            s=sample();s['gold_status']='disputed';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):dataset(p)

    def test_split_groups_and_duplicates(self):
        rows=[sample(i) for i in range(10)]
        rows[0]['messages']=rows[1]['messages']
        rows[2]['leakage_group']=rows[3]['leakage_group']='same-family'
        s=split(rows,'fixed')['assignments']
        self.assertEqual(s['0'],s['1']);self.assertEqual(s['2'],s['3'])
        self.assertEqual(s,split(rows,'fixed')['assignments'])

    def test_incomplete_refusal_and_token_match(self):
        rows=[sample()]
        with self.assertRaises(ValueError):report(rows,[])
        predictions=[dict(sample_id='0',method=m,prompt_sha256='same',e2e_ms=1,parsed_risk=2,invalid_reason=None) for m in ('A0','A1','B1')]
        r=report(rows,predictions)
        self.assertEqual(r['human_gold']['B1']['n'],0)
        self.assertEqual(r['synthetic_reference_diagnostics']['B1']['risk3_to2'],1)
        predictions[-1]['prompt_sha256']='different'
        with self.assertRaises(ValueError):report(rows,predictions)

    def test_single_token_guard(self):
        class Bad:
            special_tokens={}
            def encode(self,*a,**kw):return [1,2]
        with self.assertRaises(ValueError):tokenizer_audit(Bad())

    def test_eos_is_not_explanation(self):
        class Tokenizer:
            special_tokens={'<eos>':99};control_tokens=['<eos>']
            def decode(self,ids):return ''.join('text' if i==1 else '<eos>' for i in ids)
        self.assertEqual(visible_generation(Tokenizer(),[99]),'')
        self.assertEqual(visible_generation(Tokenizer(),[1,99]),'text')

    def test_empty_explanation_fails_whole_flow_and_releases(self):
        class Tokenizer:
            special_tokens={'<eos>':99};control_tokens=['<eos>']
            def encode(self,s,**kw):return [int(s.strip())+15] if s.strip() in '0123' else [2]
            def decode(self,ids):return str(ids[0]-15) if ids and ids[0] in range(15,19) else ''
        class Service:
            def encode_messages(self,ms,t):return [100,101]
        client=Client('',Service(),Tokenizer(),{'label_system':'label','label_max_new':16,'explanation_suffix':'explain'},4096)
        calls=[]
        def post(path,body):
            calls.append((path,body))
            if path=='/v1/decisions':
                return {'candidates':[{'token_ids':[i],'logprob':-2.0} for i in range(15,19)],'label_mass':.4,'calibrated':False,'usage':{'prompt_ms':1,'prompt_read':2},'session':{'id':'test','revision':1}}
            if path=='/v1/continue':
                return {'session':{'id':'test','revision':2},'generated_tokens':1,'tokens':[99],'finish':'stop','prompt_read':1,'reused':2}
            return {'session':None}
        client.post=post
        r=client.one(sample(),'B1',conditional=True,route=lambda _:True)
        self.assertEqual(r['invalid_reason'],'explanation_empty')
        self.assertEqual(calls[-1][0],'/v1/score')
        self.assertEqual(calls[-1][1]['session']['revision'],2)

    def test_real_api_candidate_schema(self):
        import score_api
        class Tokens:
            special_tokens={}
            def encode(self,s,**kw):return [int(s.strip())+15]
            def decode(self,ids):return str(ids[0]-15)
        class Service:
            def encode_messages(self,ms,t):return [99,100]
        client=Client('',Service(),Tokens(),{'label_system':'label','label_max_new':16},4096)
        def post(path,body):
            self.assertEqual(score_api._parse_candidates(body['candidates']),[15,16,17,18])
            return {'candidates':[{'token_ids':[i],'logprob':-2.0-i/100} for i in range(15,19)],
                    'label_mass':.4,'calibrated':False,'usage':{'prompt_ms':1,'prompt_read':2}}
        client.post=post
        self.assertEqual(client.one(sample(),'B1')['parsed_risk'],0)

    def test_new_turn_uses_real_label_and_current_revision(self):
        class Tokens:
            special_tokens={'<eos>':99};control_tokens=['<eos>']
            def encode(self,s,**kw):return [int(s.strip())+15] if s.strip() in {'0','1','2','3'} else [2]
            def decode(self,ids):
                if ids==[100,101]:return '<|im_start|>user\nexplain'
                return ''.join(str(i-15) if i in range(15,19) else 'reason' if i==42 else '' for i in ids)
        class Service:
            def encode_messages(self,*a):return [100,101]
        client=Client('',Service(),Tokens(),{'label_system':'label','label_max_new':16,'explanation_suffix':'explain','explanation_mode':'new_turn'},4096)
        calls=[]
        def post(path,body):
            calls.append((path,body))
            if path=='/v1/decisions':return {'candidates':[{'token_ids':[i],'logprob':-2.0} for i in range(15,19)],'label_mass':.4,'calibrated':False,'usage':{'prompt_ms':1,'prompt_read':2},'session':{'id':'x','revision':1}}
            if path=='/v1/continue' and 'ids' not in body:return {'session':{'id':'x','revision':2},'generated_tokens':2,'tokens':[15,99],'finish':'stop','prompt_read':0,'reused':2}
            if path=='/v1/continue':return {'session':{'id':'x','revision':3},'generated_tokens':2,'tokens':[42,99],'finish':'stop','prompt_read':3,'reused':4}
            return {'session':None}
        client.post=post;client.get=lambda _: {'armed':True,'engine_state':{'pos':9}}
        r=client.one(sample(),'B1',conditional=True,route=lambda _:True)
        self.assertIsNone(r['invalid_reason']);self.assertEqual(r['generated_tokens'],4)
        self.assertEqual(r['text'],'reason');self.assertEqual(r['state_cursor_verified'],9)
        self.assertEqual(calls[-1][1]['session']['revision'],3)


if __name__=='__main__':unittest.main()
