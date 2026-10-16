import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from strata_risk_data import import_jev


@unittest.skipUnless(importlib.util.find_spec('yaml'),'optional PyYAML importer dependency')
class DataTests(unittest.TestCase):
    def make(self,root):
        import yaml
        base=root/'crime_chat_dataset';(base/'registry').mkdir(parents=True);(base/'generated/accepted').mkdir(parents=True)
        c=sqlite3.connect(base/'registry/dataset_registry.sqlite')
        c.execute('CREATE TABLE cases(case_id TEXT, family_id TEXT, lineage_id TEXT, risk INTEGER, category TEXT, status TEXT, yaml_path TEXT, has_transfer INTEGER)')
        c.execute('CREATE TABLE pairs(pair_id TEXT,case_a TEXT,case_b TEXT)')
        for i,status in enumerate(['accepted','review','accepted','accepted']):
            cid=str(i);p=base/'generated/accepted'/('case'+cid+'.yaml')
            d={'identity':{'case_id':cid},'pipeline_status':'accepted','labels':{'risk':i},
                'source':{'source_type':'synthetic_subagent'},'conversation':'public unit text '+cid,
                'hidden_case':{'latent_intent':'MUST NOT ENTER PROMPT'},'execution':{'workflow_version':'unit/1'}}
            p.write_text(yaml.safe_dump(d),encoding='utf-8')
            if i==3:p.write_text('broken: [',encoding='utf-8')
            c.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?)',(cid,'family','lineage',i,'other',status,str(p.relative_to(root)),0))
        c.commit();c.close()
        return base/'registry/dataset_registry.sqlite'

    def test_readonly_authority_and_private_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);db=self.make(root);before=db.read_bytes()
            rows,summary,details=import_jev(root)
            self.assertEqual(len(rows),2)
            self.assertEqual(summary['excluded']['registry_review'],1)
            self.assertEqual(summary['excluded']['ParserError'],1)
            self.assertEqual(summary['largest_cluster'],2)
            self.assertEqual(summary['gold_accuracy_status'],'BLOCKED_NO_GOLD_DATA')
            self.assertTrue(all(r['gold_status']=='provisional' for r in rows))
            self.assertNotIn('MUST NOT ENTER PROMPT',json.dumps(rows))
            self.assertEqual(before,db.read_bytes())


if __name__=='__main__':unittest.main()
