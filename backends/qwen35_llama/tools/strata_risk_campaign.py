"""Finite, resumable synthetic-reference campaign; no implicit human-gold gate.

Preparation never uses predictions. Completed atomic triples are immutable.
Each GPU window has a fixed1200s anchor, each native lease<=300s. A failed
window never retries itself. Existing result hashes must verify before skipping.
All source messages, IDs and per-case records remain in a local private folder.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import threading
import time
import zlib

from strata_risk_data import digest
from strata_risk_eval import Client, dataset, read, write, tokenizer_audit
from strata_ordinal_metrics import evaluate, paired

SEED=2026101609
METHODS=('A0','A1','B1')


def normalized(text):
    text=re.sub(r'\d+', '0', text)
    return re.sub(r'[\W_]+','',text).lower()


def templates(rows):
    """Bounded bottom-shingle LSH audit, followed by exact normalized Jaccard.

    This is an approximate template detector, not proof of all near-duplicates.
    Digits/punctuation normalize; actual conversation exact hashes stay separate.
    """
    buckets=defaultdict(list);grams=[];parents=list(range(len(rows)))
    comparisons=edges=0;hot=0;repetition=[]
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,r in enumerate(rows):
        s=normalized(r['messages'][0]['content'])
        gs={s[k:k+8] for k in range(0,max(1,len(s)-7),4)}
        grams.append(gs)
        signature=sorted(zlib.crc32(g.encode()) for g in gs)[:32]
        candidates=set()
        for b in range(8):
            key=(b,tuple(signature[b*4:b*4+4]))
            previous=buckets[key]
            if len(previous)>256:hot+=1
            candidates.update(previous[:256])
            previous.append(i)
        for j in sorted(candidates):
            comparisons+=1
            other=grams[j]
            if min(len(gs),len(other)) < .8*max(len(gs),len(other)):continue
            score=len(gs & other)/len(gs | other) if gs|other else 1
            if score>=.8:
                parents[find(i)]=find(j);edges+=1
        chunks=[s[k:k+40] for k in range(0,len(s)-39,40)]
        repetition.append(1-len(set(chunks))/len(chunks) if chunks else 0)
    groups=[find(i) for i in range(len(rows))]
    return groups,repetition,{'algorithm':'digits/punctuation normalized8-character shingles stride4; bottom32 LSH8x4, exact Jaccard>=.8 confirmation; per-bucket cap256',
        'comparisons':comparisons,'confirmed_edges':edges,'clusters':len(set(groups)),
        'largest_cluster':max(Counter(groups).values(),default=0),'hot_bucket_truncations':hot,
        'scope':'approximate significant-template audit, not exhaustive semantic duplicate detection'}


def select(rows, lengths, *, per_risk=150, seed=SEED):
    """Round-robin scenario/length bins, rare template groups first per class."""
    if type(per_risk) is not int or per_risk<1:raise ValueError('positive per-risk count required')
    groups,repeated,near=templates(rows)
    rng=random.Random(seed);selected=[];fallbacks={};coverage={}
    exact=Counter(digest(re.sub(r'\s+','',r['messages'][0]['content'])) for r in rows)
    if max(exact.values())>1:raise ValueError('duplicate content detected before sampling')
    for g in range(4):
        candidates=[i for i,r in enumerate(rows) if r['gold_risk']==g and lengths[i]['eligible']]
        if len(candidates)<per_risk:raise ValueError('not enough context-legal candidates at risk'+str(g))
        counts=Counter(groups[i] for i in candidates)
        cells=defaultdict(list)
        for i in candidates:
            bin_id=min(3,lengths[i]['A0_tokens']//512)
            cells[(rows[i]['scenario'],bin_id)].append(i)
        for values in cells.values():
            rng.shuffle(values)
            values.sort(key=lambda i:(repeated[i]>=.25,counts[groups[i]]))
        keys=sorted(cells);rng.shuffle(keys)
        chosen=[];used_groups=set()
        # Prefer distinct strong-template clusters; risk0 may have little slack.
        for allow_repeat in (False,True):
            while len(chosen)<per_risk:
                progress=False
                for key in keys:
                    options=cells[key]
                    pos=next((p for p,i in enumerate(options) if allow_repeat or groups[i] not in used_groups),None)
                    if pos is None:continue
                    i=options.pop(pos);chosen.append(i);used_groups.add(groups[i]);progress=True
                    if len(chosen)==per_risk:break
                if not progress:break
        selected+=chosen;fallbacks[str(g)]=per_risk-len(used_groups)
        coverage[str(g)]={'eligible':len(candidates),'selected':len(chosen),'unique_template_clusters':len(used_groups),
            'scenario':dict(Counter(rows[i]['scenario'] for i in chosen)),
            'A0_lengths':{'min':min(lengths[i]['A0_tokens'] for i in chosen),'median':statistics.median(lengths[i]['A0_tokens'] for i in chosen),'max':max(lengths[i]['A0_tokens'] for i in chosen)},
            'repetition_flagged':sum(repeated[i]>=.25 for i in chosen)}
    rng.shuffle(selected)
    return selected,groups,{'seed':seed,'per_risk':per_risk,'near_templates':near,'class_coverage':coverage,
        'template_reuse_needed':fallbacks,'selection':'scenario/length round-robin, seeded order; prefer distinct significant-template clusters; no model predictions used',
        'duplicate_content':0,'excluded_context':sum(not l['eligible'] for l in lengths)}


def prepare(args):
    from strata_tokenizer import Tokenizer
    from strata_unified import UnifiedService
    rows=dataset(args.data);config=read(args.config);spec=read(args.spec)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    tokenizer=Tokenizer.from_gguf(Path(spec['cwd'])/spec['model']['path'])
    service=object.__new__(UnifiedService);service.tokenizer_model=str(Path(spec['cwd'])/spec['model']['path'])
    client=Client('',service,tokenizer,config,spec['baseline']['config']['ctx'])
    lengths=[]
    for r in rows:
        a=client.prompt(r,'A0');b=client.prompt(r,'A1')
        lengths.append({'A0_tokens':len(a),'label_tokens':len(b),'A0_hash':digest(a),'label_hash':digest(b),
            'eligible':len(a)+config['a0_max_new']<=client.ctx and len(b)+config['label_max_new']<=client.ctx})
    indices,groups,audit=select(rows,lengths)
    chosen=[dict(rows[i],template_cluster=str(groups[i])) for i in indices]
    (out/'private-balanced600.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen).encode())
    remaining=[rows[i] for i in range(len(rows)) if i not in set(indices)]
    (out/'private-remainder.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in remaining).encode())
    write(out/'private-lengths.json',lengths)
    write(out/'sampling-audit.json',audit)
    # One fixed interleaving schedule, independent of reference class and prediction.
    orders=[list(METHODS[i%3:]+METHODS[:i%3]) if i%2==0 else list(reversed(METHODS[i%3:]+METHODS[:i%3])) for i in range(len(chosen))]
    batches=[list(range(i,min(i+30,len(chosen)))) for i in range(0,len(chosen),30)]
    manifest={'schema':'strata-synthetic-reference-campaign/1','reference_kind':'synthetic_design_label',
        'evaluation_status':'AUTHORIZED_SYNTHETIC_REFERENCE','human_gold_status':'BLOCKED_NO_GOLD_DATA',
        'source_dataset_sha256':digest(rows),'selected_dataset_sha256':digest(chosen),
        'config':config,'config_sha256':digest(config),'spec':spec,'orders':orders,'batches':batches,
        'max_native':len(batches),'native_limit_s':300,'window_s':1200,'max_batches_per_window':6,
        'budget_exit':'No retry on safety/native/protocol failure; completed triple hashes are immutable; next window only at explicit supervisor boundary.',
        'max_internal_per_batch':30*12+8,'n':len(chosen),'seed':SEED,
        'sampling_sha256':digest(audit),'tokenizer':tokenizer_audit(tokenizer)}
    write(out/'manifest.json',manifest)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


def completed_rows(directory):
    directory=Path(directory);path=directory/'completed.json'
    if not path.exists():return None
    receipt=read(path)
    for name,sha in receipt['files'].items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('completed checkpoint hash mismatch: '+str(p))
    return read(directory/'private-results.json')


class InstrumentedClient(Client):
    """Observer-only strict-parser tap. No native work or service semantics changed."""
    def one(self,*args,**kwargs):
        self.service.link.measurements=[]
        result=super().one(*args,**kwargs)
        if result['method'] in ('A0','A1') and result['finish']!='stop' and not result.get('invalid_reason'):
            result['invalid_reason']='generation_not_eos'
        operations=list(self.service.link.measurements)
        generated=[o for o in operations if o['done']['produced']]
        result['native_operations']=operations
        result['decode_ms']=sum(o['done']['decode_ms'] for o in generated) if generated else 0.0
        if generated:
            # This is the GEN command's first-token observer clock, not full
            # HTTP TTFT; prefix SCORE is reported separately, never hidden.
            result['gen_first_token_observed_ms']=generated[0].get('first_token_ms')
        return result


def instrument(link):
    # Tap transport observation only, including cooperative methods that call
    # super() directly. Existing EngineLink still validates each full response.
    import score_api as api
    original=link._take_until;send=link.send;active={}
    def sending(line):
        if line.startswith(('SCORE ','GEN ')):
            active.clear();active.update(operation=line.split()[0],begin=time.perf_counter(),stamps=[])
        return send(line)
    def take(deadline,timeout):
        line=original(deadline,timeout)
        if line.startswith('T ') and active:active['stamps'].append(time.perf_counter())
        if line.startswith('DONE ') and active:
            stamps=active['stamps'];begin=active['begin']
            link.measurements.append({'operation':active['operation'],'done':api._parse_done(line),
                'observed_ms':1000*(time.perf_counter()-begin),
                'first_token_ms':1000*(stamps[0]-begin) if stamps else None,
                'token_observation_ms':[1000*(s-begin) for s in stamps]})
            active.clear()
        return line
    link.send=sending;link._take_until=take;link.measurements=[]


def run_window(args):
    import strata_experiment as ex
    from leased_native import LeasedNative
    from strata_tokenizer import Tokenizer
    from strata_unified import UnifiedService,leased_link,start_http
    root=Path(args.campaign);manifest=read(root/'manifest.json')
    rows=dataset(root/'private-balanced600.jsonl')
    if digest(rows)!=manifest['selected_dataset_sha256']:raise ValueError('frozen selection changed')
    config=manifest['config'];spec=manifest['spec']
    if digest(config)!=manifest['config_sha256']:raise ValueError('frozen template changed')
    if not args.enable_gpu:raise ValueError('explicit --enable-gpu required')
    if not 1<=args.max_batches<=manifest['max_batches_per_window']:raise ValueError('window batch budget exceeded')
    window=time.time();windowdir=root/('window-'+str(args.window_id));windowdir.mkdir(exist_ok=False)
    ex.validate(spec);fp=ex.fingerprint(spec)
    if not ex.admission(spec,fp,spec['baseline']['config'])['admitted']:raise ValueError('resource admission refused')
    write(windowdir/'preregistration.json',{'started_at_epoch':window,'authorized_steps':['up to'+str(args.max_batches)+' next untouched batches'],'campaign_sha256':digest(manifest),
        'identity':ex.identity(spec,fp),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('strata_risk*.py')},'native_bound_s':300,'max_internal_per_batch':manifest['max_internal_per_batch']})
    tokenizer=Tokenizer.from_gguf(Path(spec['cwd'])/spec['model']['path']);launched=0
    for batch_id,indices in enumerate(manifest['batches']):
        out=root/('batch-%02d'%batch_id)
        done=completed_rows(out)
        if done is not None:continue
        if out.exists():raise ValueError('incomplete batch needs explicit supervisor decision, no automatic retry: '+str(out))
        if launched>=args.max_batches or time.time()-window>manifest['window_s']-360:break
        launched+=1;spec['budget']['native_timeout_s']=300
        native=None;link=None;httpd=None;results=[];started=time.perf_counter()
        try:
            native=LeasedNative(spec,spec['baseline']['config'],out,window)
            link=leased_link(native);instrument(link)
            service=UnifiedService(link,engine_path=spec['engine']['path'],timeout=60,
                tokenizer_model=str(Path(spec['cwd'])/spec['model']['path']),backend='qwen35_llama',backend_source='P4-balanced600')
            service._tokenizer=tokenizer;service._tokenizer_loaded=True
            service.service_deadline=native.started+300
            httpd=start_http(service,port=0);thread=threading.Thread(target=httpd.serve_forever,kwargs={'poll_interval':.05});thread.start()
            client=InstrumentedClient('http://127.0.0.1:%d'%httpd.server_address[1],service,tokenizer,config,spec['baseline']['config']['ctx'])
            client.source_shapes=[]
            with (out/'private-results.jsonl').open('x',encoding='utf-8') as stream:
                for index in indices:
                    for method in manifest['orders'][index]:
                        if time.perf_counter()-started>225 or time.time()-window>=manifest['window_s'] or native.monitor.violation:
                            raise RuntimeError('batch time/resource guard; retained checkpoint, no next request')
                        r=client.one(rows[index],method)
                        r.update(case_index=index,batch=batch_id,reference_risk=rows[index]['gold_risk'],reference_kind='synthetic_design_label')
                        results.append(r);stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    same=[r['prompt_sha256'] for r in results[-3:] if r['method'] in ('A1','B1')]
                    if len(same)!=2 or same[0]!=same[1]:raise ValueError('paired token input mismatch')
            write(out/'private-results.json',results)
        except BaseException as exc:
            if out.exists():write(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(results),'no_retry':True})
            raise
        finally:
            if httpd is not None:httpd.shutdown();httpd.server_close();thread.join(5)
            if native is not None:native.close(abort=True)
            if link is not None:link.close()
        receipt=native.receipt()
        write(out/'completed.json',{'case_count':len(indices),'records':len(results),'native_receipt':receipt,
            'window_id':args.window_id,'wall_ms':1000*(time.perf_counter()-started),
            'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('private-results.json','private-results.jsonl','raw/child_00.log','result.json','resources.json','job.json') }})
        print(json.dumps({'batch':batch_id,'completed_cases':sum(len(b) for j,b in enumerate(manifest['batches']) if (root/('batch-%02d'%j)/'completed.json').exists()),'batch_wall_s':time.perf_counter()-started,'remaining_window_s':manifest['window_s']-(time.time()-window)},allow_nan=False),flush=True)
    write(windowdir/'checkpoint.json',{'launched':launched,'elapsed_s':time.time()-window,'completed_batches':[i for i in range(len(manifest['batches'])) if (root/('batch-%02d'%i)/'completed.json').exists()]})


def summarize(args):
    root=Path(args.campaign);manifest=read(root/'manifest.json');samples=dataset(root/'private-balanced600.jsonl')
    records=[];walls=[]
    for i in range(len(manifest['batches'])):
        rows=completed_rows(root/('batch-%02d'%i))
        if rows is None:raise ValueError('incomplete campaign; cannot report600')
        records+=rows;walls.append(read(root/('batch-%02d'%i)/'completed.json')['wall_ms'])
    if len(records)!=1800 or len({(r['sample_id'],r['method']) for r in records})!=1800:raise ValueError('paired completeness mismatch')
    byid={r['sample_id']:r for r in samples};methods={m:[] for m in METHODS}
    for r in records:
        d=byid[r['sample_id']]
        methods[r['method']].append(dict(r,gold_risk=d['gold_risk'],conversation_id=d['conversation_id']))
    from strata_risk_eval import speed
    result={'reference_kind':'synthetic_design_label','evaluation_status':'COMPLETED_SYNTHETIC_REFERENCE','human_gold_status':'NOT_CLAIMED',
        'n':600,'sampling':read(root/'sampling-audit.json'),'accuracy':{m:evaluate(rs) for m,rs in methods.items()},
        'performance':{m:speed(rs) for m,rs in methods.items()},'prediction_counts':{m:dict(Counter(r.get('parsed_risk') for r in rs)) for m,rs in methods.items()},
        'paired':{m+'-B1':paired(methods[m],methods['B1'],draws=1000,seed=SEED) for m in ('A0','A1')},
        'campaign_wall_including_load_s':sum(walls)/1000,'mixed_method_completed_per_s':sum(not r.get('invalid_reason') for r in records)/(sum(walls)/1000),
        'scope':'balanced synthetic-reference comparison; no human quality claim, no tuning on predictions, no original-distribution extrapolation'}
    for m,rs in methods.items():
        result['performance'][m]['mean_prefill_ms']=statistics.mean(r['prefill_ms'] for r in rs)
        result['performance'][m]['mean_decode_ms']=statistics.mean(r['decode_ms'] for r in rs)
        result['performance'][m]['cold_first']=speed(r for r in rs if r['cold_warm']=='first_request')
        result['performance'][m]['subsequent']=speed(r for r in rs if r['cold_warm']!='first_request')
    write(root/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('n','accuracy','performance','prediction_counts')},ensure_ascii=False,indent=2))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for key in ('data','config','spec','out'):p.add_argument('--'+key,required=True)
    r=sub.add_parser('run-window');r.add_argument('--campaign',required=True);r.add_argument('--window-id',type=int,required=True);r.add_argument('--max-batches',type=int,default=6);r.add_argument('--enable-gpu',action='store_true')
    s=sub.add_parser('summarize');s.add_argument('--campaign',required=True)
    a=ap.parse_args(argv)
    {'prepare':prepare,'run-window':run_window,'summarize':summarize}[a.action](a);return 0


if __name__=='__main__':raise SystemExit(main())
