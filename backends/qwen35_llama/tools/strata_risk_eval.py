"""Opt-in local risk evaluation. Human gold and synthetic references stay separate.

run starts one finite leased engine and loopback HTTP service; no production
server/default/profile is modified. Import/audit/report do not use a GPU.
All per-case output is PRIVATE; publish only aggregate statistics and hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

from strata_risk_data import digest
from strata_ordinal_metrics import evaluate, paired, risk


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def dataset(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    seen = set()
    for r in rows:
        for field in ('sample_id', 'conversation_id', 'scenario', 'source_id', 'label_policy_version'):
            if not isinstance(r.get(field), str) or not r[field]: raise ValueError('missing '+field)
        if r['sample_id'] in seen: raise ValueError('duplicate sample id')
        seen.add(r['sample_id'])
        if not risk(r.get('gold_risk')): raise ValueError('invalid reference risk')
        if r.get('gold_status') not in {'adjudicated', 'disputed', 'provisional'}: raise ValueError('invalid reference status')
        if r.get('reference_kind') not in {'human_adjudicated', 'synthetic_design_label', 'unverified_reference'}: raise ValueError('explicit reference_kind required')
        if r['reference_kind'] == 'human_adjudicated' and not re.fullmatch('[a-f0-9]{64}',str(r.get('annotation_evidence_sha256',''))):
            raise ValueError('human reference requires annotation evidence binding')
        if r['gold_status'] == 'disputed':
            allowed = r.get('allowed_risks')
            if not isinstance(allowed, list) or not allowed or any(not risk(x) for x in allowed):
                raise ValueError('disputed reference requires allowed_risks')
        ms = r.get('messages')
        if not isinstance(ms, list) or not ms or any(not isinstance(m, dict) or set(m) != {'role','content'} or m['role'] not in {'user','assistant'} or not isinstance(m['content'],str) or not m['content'].strip() for m in ms):
            raise ValueError('only nonempty conversation text, never metadata, in messages')
    if not rows: raise ValueError('empty dataset')
    return rows


def audit(rows):
    gold = [r for r in rows if r['reference_kind']=='human_adjudicated' and r['gold_status']=='adjudicated']
    return {'n':len(rows), 'dataset_sha256':digest(rows),
        'risk_counts':dict(Counter(r['gold_risk'] for r in rows)),
        'scenario_counts':dict(Counter(r['scenario'] for r in rows)),
        'reference_counts':dict(Counter(r['reference_kind'] for r in rows)),
        'status_counts':dict(Counter(r['gold_status'] for r in rows)),
        'human_adjudicated_n':len(gold),
        'accuracy_status':'READY_FOR_REGISTERED_EVALUATION' if gold else 'BLOCKED_NO_GOLD_DATA'}


def split(rows, seed):
    """Stable, conservative group assignment, not a promise of balanced holdout."""
    groups = {}
    # Repeated conversations are joined even when caller supplied wrong group IDs.
    parents = {r['sample_id']:r['sample_id'] for r in rows}
    def find(k):
        while parents[k]!=k:
            parents[k]=parents[parents[k]];k=parents[k]
        return k
    def join(a,b): parents[find(b)]=find(a)
    for r in rows:
        for key in (('conversation',r['conversation_id']),('content',digest(r['messages'])),
                    ('group',r.get('leakage_group',r['conversation_id']))):
            join(groups.setdefault(key,r['sample_id']),r['sample_id'])
    assignments = {r['sample_id']: ('final' if int(digest([seed,find(r['sample_id'])])[:8],16)%5==0 else 'development') for r in rows}
    return {'schema':'strata-risk-split/1', 'dataset_sha256':digest(rows), 'seed':seed,
            'assignments':assignments, 'counts':dict(Counter(assignments.values())),
            'groups':len({find(r['sample_id']) for r in rows}),
            'note':'No stratification/independence claim; inspect giant source families before final use.'}


def tokenizer_audit(tokenizer):
    result = []
    for digit in '0123':
        ids = tokenizer.encode(digit, parse_special=False)
        if len(ids)!=1 or tokenizer.decode(ids)!=digit or ids[0] in tokenizer.special_tokens.values():
            raise ValueError('digit is not an ordinary single-token candidate: '+digit)
        result.append({'risk':int(digit),'ids':ids,'decoded':tokenizer.decode(ids),
                       'space_prefix_ids':tokenizer.encode(' '+digit,parse_special=False)})
    if len({r['ids'][0] for r in result})!=4: raise ValueError('duplicate candidate ids')
    return {'labels':result,'candidate_ids':[r['ids'][0] for r in result],
            'special_token_map_sha256':digest(tokenizer.special_tokens), 'status':'PASS'}


def parse_generated(text, method):
    text = text.strip()
    if method=='A1': return int(text) if text in {'0','1','2','3'} else None
    try: d=json.loads(text)
    except (ValueError, TypeError): return None
    if not isinstance(d,dict) or not risk(d.get('risk')) or type(d.get('violation')) is not bool:
        return None
    return d['risk'] if d['violation']==(d['risk']>=2) else None


def visible_generation(tokenizer, tokens):
    """EOS/control-only continuation is not a successful explanation."""
    controls={tokenizer.special_tokens[s] for s in tokenizer.control_tokens if s in tokenizer.special_tokens}
    return tokenizer.decode([t for t in tokens if t not in controls])


def speed(rows):
    rows=list(rows);times=[r['e2e_ms'] for r in rows]
    valid=sum(r.get('parsed_risk') is not None and not r.get('invalid_reason') for r in rows)
    total=sum(times)
    return {'n':len(rows),'valid_completed_reviews':valid,
            'mean_e2e_ms':statistics.mean(times) if times else None,
            'median_e2e_ms':statistics.median(times) if times else None,
            'p90_e2e_ms':sorted(times)[math.ceil(.9*len(times))-1] if len(times)>=100 else None,
            'p95_e2e_ms':sorted(times)[math.ceil(.95*len(times))-1] if len(times)>=100 else None,
            'tail_note':'tails withheld below100; warm requests are not proof of steady state',
            'valid_reviews_per_s_service_sum':valid*1000/total if total else None,
            'mean_actual_output_tokens':statistics.mean(r.get('generated_tokens',0) for r in rows) if rows else None,
            'errors':dict(Counter(r['invalid_reason'] for r in rows if r.get('invalid_reason')))}


def report(rows, results):
    lookup={r['sample_id']:r for r in rows}
    expected={(r['sample_id'],m) for r in rows for m in ('A0','A1','B1')}
    if len(results)!=len(expected) or {(r['sample_id'],r['method']) for r in results}!=expected:
        raise ValueError('incomplete/duplicate paired result set; cannot drop failed samples')
    methods={m:[] for m in ('A0','A1','B1')}
    for r in results:
        d=lookup[r['sample_id']]
        methods[r['method']].append({**r,'gold_risk':d['gold_risk'], 'conversation_id':d['conversation_id'],
                                    'reference_kind':d['reference_kind'],'gold_status':d['gold_status']})
    for d in rows:
        same=[r['prompt_sha256'] for r in results if r['sample_id']==d['sample_id'] and r['method'] in ('A1','B1')]
        if len(set(same))!=1: raise ValueError('A1/B1 prompt token mismatch')
    human=lambda rs:[r for r in rs if r['reference_kind']=='human_adjudicated' and r['gold_status']=='adjudicated']
    out={'audit':audit(rows),'speed':{m:speed(v) for m,v in methods.items()},
         'human_gold':{m:evaluate(human(v)) for m,v in methods.items()},
         'quality_qualified_reviews_per_s':None, 'business_release':'NOT_APPROVED',
         'disputed_n':sum(r['gold_status']=='disputed' for r in rows),
         'synthetic_reference_diagnostics':{m:evaluate(r for r in v if r['reference_kind']=='synthetic_design_label' and r['gold_status']!='disputed') for m,v in methods.items()},
         'paired_human_gold':{}}
    out['by_scenario_human_gold']={m:{s:evaluate(r for r in human(v) if lookup[r['sample_id']]['scenario']==s)
                                    for s in sorted({r['scenario'] for r in rows})} for m,v in methods.items()}
    out['disputed']={m:{'n':sum(lookup[r['sample_id']]['gold_status']=='disputed' for r in v),
        'within_allowed_set':sum(lookup[r['sample_id']]['gold_status']=='disputed' and r.get('parsed_risk') in lookup[r['sample_id']].get('allowed_risks',[]) for r in v)} for m,v in methods.items()}
    out['prediction_agreement_all']={}
    for m in ('A0','A1'):
        out['paired_human_gold'][m+'-B1']=paired(human(methods[m]),human(methods['B1']))
        a={r['sample_id']:r for r in methods[m]};b={r['sample_id']:r for r in methods['B1']}
        out['prediction_agreement_all'][m+'-B1']={'n':len(a), 'agree':sum(risk(a[k].get('parsed_risk')) and not a[k].get('invalid_reason') and not b[k].get('invalid_reason') and a[k]['parsed_risk']==b[k].get('parsed_risk') for k in a), 'note':'agreement is not human accuracy'}
    out['comparison_scope']={'A0-B1':'business prompt and output format both change',
                             'A1-B1':'same prompt token IDs; complete ordinary label generation vs score, no forced label normalization'}
    return out


class Client:
    def __init__(self,url,service,tokenizer,config,ctx):
        self.url,self.service,self.tokenizer,self.config,self.ctx=url,service,tokenizer,config,ctx
        self.ids=tokenizer_audit(tokenizer)['candidate_ids']
        self.counter=0

    def post(self,path,body):
        request=urllib.request.Request(self.url+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
        # Do not send private content through configured proxies.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=65) as response:
            return json.load(response)

    def prompt(self,sample,method):
        system=self.config['a0_system'] if method=='A0' else self.config['label_system']
        return self.service.encode_messages([{'role':'system','content':system}]+sample['messages'],self.tokenizer)

    def one(self,sample,method,*,conditional=False,route=None,max_new=None):
        begin=time.perf_counter(); self.counter+=1
        ids=self.prompt(sample,method)
        maximum=max_new or self.config['a0_max_new' if method=='A0' else 'label_max_new']
        if len(ids)+maximum>self.ctx: raise ValueError('actual prompt plus output exceeds context; no truncation')
        row={'sample_id':sample['sample_id'],'method':method,'prompt_sha256':digest(ids),
             'prompt_tokens':len(ids),'requested_max_new':maximum,'generated_tokens':0,
             'invalid_reason':None,'parsed_risk':None,'finish':None,'eos':False,
             'decode_ms':None,'score_ms':None,'cold_warm':'first_request' if self.counter==1 else 'subsequent_request',
             'timing_scope':'encoding, loopback HTTP, native work, parsing and explicit ownership cleanup; observer wall',
             'session':None,'repeated_prefill':False}
        lease=None
        try:
            if method in ('A0','A1'):
                response=self.post('/v1/chat/completions',{'prompt_ids':ids,'max_tokens':maximum,'temperature':0})
                text=response['choices'][0]['message']['content']
                row.update(parsed_risk=parse_generated(text,method),text=text,
                    generated_tokens=response['usage']['completion_tokens'],finish=response['choices'][0]['finish_reason'],
                    **response['engine_timing'])
                row['session']=response['session']
            else:
                scored_at=time.perf_counter()
                response=self.post('/v1/decisions',{'kind':'choice','prompt_ids':ids,
                    'candidates':[[i] for i in self.ids], 'keep':conditional})
                row['score_ms']=(time.perf_counter()-scored_at)*1000
                pairs=response['candidates'];values=[p['logprob'] for p in pairs]
                if [p['token_ids'] for p in pairs]!=[[i] for i in self.ids] or any(not math.isfinite(x) for x in values):
                    raise ValueError('invalid candidate response')
                row.update(logits=pairs,label_mass=response['label_mass'],calibrated=response['calibrated'],
                    top2_logprob_margin=sorted(values,reverse=True)[0]-sorted(values,reverse=True)[1],
                    parsed_risk=max(range(4),key=lambda i:values[i]),finish='score',prefill_ms=response['usage']['prompt_ms'])
                if response['usage']['prompt_read']!=len(ids): raise ValueError('SCORE unexpectedly reused/omitted prompt')
                lease=response.get('session');row['session']=lease
                if conditional and route(row['parsed_risk']):
                    suffix=self.config.get('explanation_suffix')
                    if not isinstance(suffix,str) or not suffix.strip():raise ValueError('explicit explanation_suffix required')
                    extra=self.tokenizer.encode(suffix,parse_special=False)
                    if len(ids)+len(extra)+maximum>self.ctx:raise ValueError('explanation exceeds context')
                    gen=self.post('/v1/continue',{'session':lease,'max_new':maximum,'ids':extra})
                    lease=gen['session'];row['session']=lease
                    visible=visible_generation(self.tokenizer,gen['tokens'])
                    row.update(generated_tokens=gen['generated_tokens'],finish=gen['finish'],text=visible)
                    if not visible.strip(): row['invalid_reason']='explanation_empty'
                    row['appended_prompt_tokens']=len(extra)
                    row['continuation_prompt_read']=gen['prompt_read'];row['continuation_reused']=gen['reused']
                    if gen['prompt_read']!=len(extra) or gen['reused']!=len(ids): raise ValueError('continuation repeated prefill/state cursor mismatch')
                    row['branch']='explain'
                else: row['branch']='score-only'
                if conditional:
                    current=self.post('/v1/score',{'prompt_ids':[0],'topk':1,'session':lease})
                    row['release_session']=current.get('session');lease=None
            row['eos']=row['finish']=='stop'
            if row['parsed_risk'] is None: row['invalid_reason']='invalid_output'
        except urllib.error.HTTPError as exc:
            row['invalid_reason']='http_'+str(exc.code)
            # Cannot safely resume the same stream after timeout/protocol failure.
            if exc.code>=500: raise RuntimeError('unsafe engine HTTP error '+str(exc.code)) from exc
        finally:
            if lease is not None:
                self.post('/v1/score',{'prompt_ids':[0],'topk':1,'session':lease})
            row['e2e_ms']=(time.perf_counter()-begin)*1000
        return row


def run(args):
    import strata_experiment as ex
    from leased_native import LeasedNative
    from strata_tokenizer import Tokenizer
    from strata_unified import UnifiedService,leased_link,start_http
    rows=dataset(args.data);config=read(args.config);spec=read(args.spec)
    if args.split!='development': raise ValueError('final run requires registered holdout adapter; development only in this release')
    if len(rows)>64: raise ValueError('bounded pilot limit64; register a separate formal campaign')
    if not args.enable_gpu: raise ValueError('explicit --enable-gpu required')
    if not 60<=args.lifetime<=300: raise ValueError('finite lifetime60..300 required')
    if args.mixed and (not 1<=args.clients<=8 or not 1<=args.capacity<=8 or args.explain_max not in (16,64,128)):
        raise ValueError('invalid mixed clients/capacity/output limit; no GPU started')
    if args.mixed and (not isinstance(config.get('explanation_suffix'),str) or not config['explanation_suffix'].strip()):
        raise ValueError('mixed run requires an explicit explanation suffix')
    if not all(isinstance(config.get(k),str) and config[k].strip() for k in ('a0_system','label_system')):
        raise ValueError('explicit versioned A0 and digit-only system templates required')
    for k in ('a0_max_new','label_max_new'):
        if type(config.get(k)) is not int or not 1<=config[k]<=128: raise ValueError('invalid max output')
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    ex.validate(spec);fp=ex.fingerprint(spec)
    if not ex.admission(spec,fp,spec['baseline']['config'])['admitted']:raise ValueError('resource admission refused')
    spec['budget']['native_timeout_s']=args.lifetime
    # Retain the original cumulative20 minute window; never reset between steps.
    started=time.time()
    write(out/'preregistration.json',{'data':audit(rows),'config':config,'config_sha256':digest(config),
        'spec':spec,'identity':ex.identity(spec,fp),'started_at_epoch':started,
        'authorized_steps':['tokenizer_audit','paired_development' if not args.mixed else 'mixed_development'],
        'max_native':1,'max_internal_requests':len(rows)*12+8,
        'accuracy_status':audit(rows)['accuracy_status'],'tools':{p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ('strata_risk_eval.py','strata_risk_data.py','strata_ordinal_metrics.py','strata_mixed_load.py')}})
    tokenizer=Tokenizer.from_gguf(Path(spec['cwd'])/spec['model']['path'])
    write(out/'tokenizer-audit.json',tokenizer_audit(tokenizer))
    native=None;httpd=None;link=None;results=[]
    try:
        native=LeasedNative(spec,spec['baseline']['config'],out/'native',started)
        link=leased_link(native)
        service=UnifiedService(link,engine_path=spec['engine']['path'],timeout=60,
            tokenizer_model=str(Path(spec['cwd'])/spec['model']['path']),backend='qwen35_llama',backend_source='P4-evaluation')
        service._tokenizer=tokenizer;service._tokenizer_loaded=True
        service.service_deadline=native.started+args.lifetime
        httpd=start_http(service,port=0)
        thread=threading.Thread(target=httpd.serve_forever,kwargs={'poll_interval':.05});thread.start()
        client=Client('http://127.0.0.1:%d'%httpd.server_address[1],service,tokenizer,config,spec['baseline']['config']['ctx'])
        def guard():
            if time.time()-started>min(1200,args.lifetime-30) or native.monitor.violation:
                raise RuntimeError('time/resource guard; checkpoint, no next operation')
        regression_ids=ex.token_ids(next(s['tokens_file'] for s in spec['shapes'] if s['id']=='P'))
        guard()
        regression=[client.post('/v1/chat/completions',{'prompt_ids':regression_ids,'max_tokens':1,'temperature':0})]
        # All business requests follow the explicit fixed-P warm-up; never call
        # these cold processes/steady-state estimates.
        client.counter=1
        with (out/'private-results.jsonl').open('x',encoding='utf-8') as stream:
            def record(row):
                results.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if args.mixed:
                from strata_mixed_load import experiment
                summary=experiment(rows,client,guard,record,args.clients,args.capacity,args.explain_max)
            else:
                for i,sample in enumerate(rows):
                    for method in (('A0','A1','B1') if i%2==0 else ('B1','A1','A0')):
                        guard();record(client.one(sample,method))
                summary=report(rows,results)
        write(out/'summary.json',summary)
        guard()
        regression.append(client.post('/v1/chat/completions',{'prompt_ids':regression_ids,'max_tokens':1,'temperature':0}))
        write(out/'prefill-regression.json',regression)
        write(out/'load.json',{'load_ready_s':native.load_ready_s,'scope':'cold native startup; following requests are within this process'})
    except BaseException as exc:
        write(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(results),'no_retry':True})
        raise
    finally:
        if httpd is not None:httpd.shutdown();httpd.server_close();thread.join(5)
        if native is not None:native.close(abort=True)
        if link is not None:link.close()
    write(out/'receipt.json',native.receipt())
    write(out/'evidence-index.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    imp=sub.add_parser('import-jev');imp.add_argument('--root',required=True);imp.add_argument('--out',required=True)
    aud=sub.add_parser('audit');aud.add_argument('--data',required=True)
    spl=sub.add_parser('split');spl.add_argument('--data',required=True);spl.add_argument('--out',required=True);spl.add_argument('--seed',default='P4-v1')
    r=sub.add_parser('run')
    for k in ('data','config','spec','out'):r.add_argument('--'+k,required=True)
    r.add_argument('--split',choices=('development','final'),default='development');r.add_argument('--enable-gpu',action='store_true')
    r.add_argument('--lifetime',type=int,default=300);r.add_argument('--mixed',action='store_true')
    r.add_argument('--clients',type=int,default=2);r.add_argument('--capacity',type=int,default=2);r.add_argument('--explain-max',type=int,default=16)
    a=ap.parse_args(argv)
    if a.action=='import-jev':
        from strata_risk_data import import_jev
        rows,summary,excluded=import_jev(a.root)
        out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
        (out/'private-dataset.jsonl').write_bytes((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)).encode())
        write(out/'summary.json',summary);write(out/'private-exclusions.json',excluded)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    elif a.action=='audit':print(json.dumps(audit(dataset(a.data)),ensure_ascii=False,indent=2))
    elif a.action=='split':
        out=Path(a.out)
        with out.open('x',encoding='utf-8') as f:json.dump(split(dataset(a.data),a.seed),f,ensure_ascii=False,indent=2)
    else:run(a)
    return 0


if __name__=='__main__':raise SystemExit(main())
