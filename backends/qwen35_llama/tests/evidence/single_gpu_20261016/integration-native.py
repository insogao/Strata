import datetime, hashlib, json, pathlib, re, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, 'D:/work/StrataUnified/strata/backends/qwen35_llama/tools')
import strata_experiment as ex
import strata_unified as un
from leased_native import LeasedNative
root = pathlib.Path(__file__).parent
spec = ex.st.read_json(root / 'smoke-spec.json')
out = pathlib.Path('D:/work/StrataUnified/experiments/single-gpu-20261015/integration-api-repair1')
started = datetime.datetime.fromisoformat(ex.st.read_json(root / 'p3-8-9-resume-20261016.json')['started_at']).timestamp()
ex.validate(spec)
assert ex.admission(spec, ex.fingerprint(spec), spec['baseline']['config'])['admitted']
P = [int(x) for x in re.split(r'[,\s]+', pathlib.Path(spec['shapes'][0]['tokens_file']).read_text(encoding='utf-8-sig').strip())]
L = [int(x) for x in re.split(r'[,\s]+', pathlib.Path(spec['shapes'][2]['tokens_file']).read_text(encoding='utf-8-sig').strip())]
gates, http_rows = {}, []
n = LeasedNative(spec, spec['baseline']['config'], out, started)
httpd = link = None
def gate(k, v):
    gates[k] = bool(v)
    (out/'gates.json').write_text(json.dumps(gates, indent=2), encoding='utf-8')
    assert v, k
def ask(k,c,**kw): return n.ask(k,c,**kw)
def ids(xs): return ','.join(map(str,xs))
def state(k): return json.loads(ask(k,'STATE')['lines'][-1][6:])
def score(k,xs):
    reset=ask(k+'_reset','SCORE 0 topk=1')
    gate(k+'_reset_ok',reset.get('done',{}).get('finish')=='score')
    answer=ask(k,'SCORE '+ids(xs)+' topk=1 keep=1')
    gate(k+'_score_ok',answer.get('done',{}).get('finish')=='score')
    return answer
def request(method,path,body=None):
    req=urllib.request.Request(url+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method=method)
    try:
        with urllib.request.urlopen(req,timeout=60) as r: code,data=r.status,r.read()
    except urllib.error.HTTPError as e: code,data=e.code,e.read()
    value=json.loads(data); http_rows.append({'method':method,'path':path,'status':code,'response':value})
    (out/'http.json').write_text(json.dumps(http_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return code,value
try:
    score('short',P)
    a=ask('gen16','GEN 16 cont=1')
    gate('short_cursor',state('state16')['pos']==len(P)+len(a['tokens']))
    f1=out/'state1.bin'; f2=out/'state2.bin'
    ask('save1','SAVE '+str(f1)); ask('load1','LOAD '+str(f1)); ask('save2','SAVE '+str(f2))
    gate('save_load_save_sha',hashlib.sha256(f1.read_bytes()).digest()==hashlib.sha256(f2.read_bytes()).digest())
    before=state('load_before'); a=ask('load_pure','GEN 1 cont=1')
    gate('load_no_logits_refused',any('no logits' in s for s in a['lines']))
    gate('load_refusal_state_unchanged',state('load_after')==before)
    a=ask('append','GEN 4 cont=1 '+ids(P[-4:]))
    gate('load_append_cursor',state('append_state')['pos']==before['pos']+4+len(a['tokens']))
    score('long',L)
    a=ask('cancel','GEN 128 cont=1',stop_after=4)
    gate('cancel_actual',a['done']['finish']=='cancel' and 4<=len(a['tokens'])<128)
    pos=len(L)+len(a['tokens']); gate('cancel_cursor',state('cancel_state')['pos']==pos)
    a=ask('recover','GEN 8 cont=1'); pos+=len(a['tokens'])
    gate('cancel_recovery_cursor',state('recover_state')['pos']==pos)
    before=state('overflow_before'); a=ask('overflow','GEN 9223372036854775807 cont=1')
    gate('overflow_refused',a['lines'][-1].startswith('ERR '))
    gate('overflow_state_unchanged',state('overflow_after')==before)
    score('eos_prompt',P); a=ask('eos','GEN 32 cont=1')
    gate('real_eos',a['done']['finish']=='stop' and a['tokens'][-1]==248046)
    gate('eos_cursor',state('eos_state')['pos']==len(P)+len(a['tokens']))
    # Only the HTTP EngineLink reader owns stdout from this boundary onward.
    sent=[]; original=n.send
    def send(c): sent.append(c); original(c)
    n.send=send
    link=un.leased_link(n)
    service=un.UnifiedService(link,engine_path=spec['engine']['path'],timeout=60,tokenizer_model=str(pathlib.Path(spec['cwd'])/spec['model']['path']),backend='qwen35_llama',backend_source='integration-test')
    httpd=un.start_http(service,port=0); url='http://127.0.0.1:'+str(httpd.server_address[1])
    threading.Thread(target=httpd.serve_forever,daemon=True).start()
    gate('models',request('GET','/v1/models')[0]==200)
    c,b=request('GET','/v1/capabilities'); gate('shared_chat',c==200 and b['shared_executor'] and b['chat_completions'])
    c,b=request('POST','/v1/chat/completions',{'prompt_ids':P,'max_tokens':32})
    gate('chat_real_eos_json',c==200 and b['choices'][0]['finish_reason']=='stop' and isinstance(json.loads(b['choices'][0]['message']['content'])['risk'],int))
    c,b=request('POST','/v1/score',{'prompt_ids':P,'keep':True}); gate('score_keep',c==200); owner=b['session']
    count=len(sent); c,b=request('POST','/v1/chat/completions',{'prompt_ids':[1],'max_tokens':1})
    gate('owner_blocks_chat_no_write',c==409 and len(sent)==count)
    c,b=request('POST','/v1/continue',{'max_new':16,'session':owner}); gate('continue',c==200); newer=b['session']
    count=len(sent); c,b=request('POST','/v1/continue',{'max_new':1,'session':owner})
    gate('stale_revision_no_write',c==409 and len(sent)==count)
    c,b=request('POST','/v1/continue',{'max_new':32,'session':newer}); gate('continue_eos',c==200 and b['finish']=='stop'); newest=b['session']
    c,b=request('POST','/v1/decisions',{'kind':'choice','prompt_ids':[1],'candidates':[[15],[16],[17],[18]],'session':newest})
    gate('decisions_uncalibrated',c==200 and b['calibrated'] is False)
    c,b=request('GET','/v1/session'); gate('session_released',c==200 and not b['armed'])
    c,b=request('POST','/v1/chat/completions',{'messages':[{'role':'user','content':'只回答一个字：好。'}],'max_tokens':32})
    gate('chat_real_template',c==200 and len(b['choices'][0]['message']['content'])>0)
    count=len(sent); c,b=request('POST','/v1/chat/completions',{'prompt_ids':[1],'stream':True})
    gate('stream_unsupported_no_write',c==501 and len(sent)==count)
    c,b=request('GET','/v1/session'); gate('chat_private_state_reset',c==200 and not b['armed'])
    (out/'http-commands.json').write_text(json.dumps(sent,indent=2),encoding='utf-8')
except BaseException as e:
    (out/'failure.json').write_text(json.dumps({'error':repr(e),'gates':gates,'retry':False},indent=2),encoding='utf-8')
    raise
finally:
    if httpd is not None: httpd.shutdown(); httpd.server_close()
    n.close(abort=True)
    if link is not None: link.close()
    receipt=n.receipt()
    (out/'cleanup.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({'gates':gates,'native_launches':1,'result':'PASS'}))
