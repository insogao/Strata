import sys,time,json,pathlib,datetime,hashlib
sys.path.insert(0,'D:/work/StrataUnified/strata/backends/qwen35_llama/tools')
import strata_experiment as ex
from leased_native import LeasedNative
from strata_diagnostics import Sampler
r=pathlib.Path(__file__).parent;s=ex.st.read_json(r/'spec.json')
m=ex.st.read_json(r/'p3-11-manifest.json');started=datetime.datetime.fromisoformat(m['started_at']).timestamp()
ex.validate(s);fp=ex.fingerprint(s);assert ex.admission(s,fp,s['baseline']['config'])['admitted']
base=pathlib.Path('D:/work/StrataUnified/experiments/next-20261016');base.mkdir(exist_ok=True)
record={'identity':ex.identity(s,fp),'manifest':m,'arms':[]}
for i in range(2):
    out=base/('diagnostic-'+str(i));n=LeasedNative(s,s['baseline']['config'],out,started);sampler=Sampler(n);sampler.start()
    rows={}
    try:
        for shape in s['shapes']:
            ids=','.join(map(str,ex.token_ids(shape['tokens_file'])))
            for j in range(2):
                a=n.ask(shape['id']+str(j),'GEN %d %s'%(shape['max_new'],ids));row=ex.metrics(a)
                assert row['n']==shape['max_new'] and row['finish']=='length' and row['prompt_read']==shape['prompt_tokens']
                rows[shape['id']+str(j)]=row
        n.ask('reset','SCORE 0 topk=1')
        L=ex.token_ids(s['shapes'][2]['tokens_file'])
        score=n.ask('score','SCORE '+','.join(map(str,L))+' topk=1 keep=1')
        assert score['done']['prompt_read']==3817
        a=n.ask('held','GEN 128 cont=1');rows['held']=ex.metrics(a)
        state=json.loads(n.ask('state','STATE')['lines'][-1][6:]);assert state['pos']==3817+len(a['tokens'])
        assert a['done']['prompt_read']==0 and a['done']['reused']==3817 and len(a['tokens'])==128
        n.ask('reset_eos','SCORE 0 topk=1')
        P=ex.token_ids(s['shapes'][0]['tokens_file']);n.ask('score_eos','SCORE '+','.join(map(str,P))+' topk=1 keep=1')
        a=n.ask('eos','GEN 32 cont=1');assert a['done']['finish']=='stop' and a['tokens'][-1]==248046
        rows['eos']=ex.metrics(a)
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':repr(exc),'retry':False}));raise
    finally:
        sampler.finish(out/'telemetry.json');n.close(abort=True)
    receipt=n.receipt();record['arms'].append({'directory':str(out),'metrics':rows,'resources':json.loads((out/'resources.json').read_text()),'native_exit':receipt['commands'][0]['exit_code']})
    (r/'diagnostic-result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'arm':i,'metrics':{k:{x:round(v[x],2) if isinstance(v[x],float) else v[x] for x in ['n','finish','prefill_ms','decode_ms','decode_observed_tps']} for k,v in rows.items()}}),flush=True)
