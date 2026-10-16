import pathlib,subprocess,sys,time,json,urllib.request,urllib.error,os,signal,socket,http.client
r=pathlib.Path(__file__).parent;out=pathlib.Path('D:/work/StrataUnified/experiments/next-20261016/stream-service-repair1')
sys.path.insert(0,'D:/work/StrataUnified/strata/backends/qwen35_llama/tools')
import strata_experiment as ex
spec=json.loads((r/'spec.json').read_text());P=ex.token_ids(spec['shapes'][0]['tokens_file']);L=ex.token_ids(spec['shapes'][2]['tokens_file'])
log=(r/'stream-service-repair1.log').open('wb')
p=subprocess.Popen([sys.executable,'-X','utf8','-B','tools/strata_server.py','--spec',str(r/'spec.json'),'--out',str(out),'--port','0','--lifetime','90'],cwd='D:/work/StrataUnified/strata',stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
gates={};responses=[];started=time.monotonic();url=None
def save():
    (r/'stream-result-repair1.json').write_text(json.dumps({'gates':gates,'responses':responses,'server_pid':p.pid,'server_exit':p.poll()},ensure_ascii=False,indent=2),encoding='utf-8')
def gate(k,v):gates[k]=bool(v);save();assert v,k
def req(path,body=None):
    begin=time.monotonic();q=urllib.request.Request(url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(q,timeout=60) as v:code,data=v.status,v.read()
    except urllib.error.HTTPError as e:code,data=e.code,e.read()
    result=json.loads(data);responses.append({'path':path,'status':code,'response':result,'http_e2e_ms':1000*(time.monotonic()-begin)});save();return code,result
def idle(label):
    for i in range(25):
        code,body=req('/v1/session')
        if code==200:
            gate(label,not body['armed'] and body['engine_state']['pos']==0);return
        assert code==503,(code,body)
        time.sleep(.2)
    raise AssertionError('cancel did not drain promptly')
try:
    until=time.monotonic()+60
    while time.monotonic()<until:
        assert p.poll() is None,(r/'stream-service-repair1.log').read_text(errors='replace')[-1500:]
        for line in (r/'stream-service-repair1.log').read_text(errors='replace').splitlines():
            if line.startswith('{'):
                value=json.loads(line)
                if 'url' in value:url=value['url'];break
        if url:break
        time.sleep(.1)
    assert url,'no URL within startup deadline'
    c,b=req('/v1/capabilities');gate('caps',c==200 and b['chat_streaming'] and b['chat_disconnect_cancel'])
    def stream(prompt,max_new,disconnect=False):
        q=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps({'prompt_ids':prompt,'max_tokens':max_new,'stream':True}).encode(),headers={'Content-Type':'application/json'})
        chunks=[]
        with urllib.request.urlopen(q,timeout=60) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data: '):continue
                value=line[6:]
                if value=='[DONE]':return chunks,True
                chunks.append(json.loads(value))
                if disconnect and len(chunks)>=5:return chunks,False
        return chunks,False
    chunks,done=stream(P,32)
    text=''.join(x['choices'][0]['delta'].get('content','') for x in chunks)
    gate('sse_eos_complete_json',done and chunks[-1]['choices'][0]['finish_reason']=='stop' and isinstance(json.loads(text)['risk'],int))
    chunks,done=stream(L,128,True);gate('disconnect_early',not done and len(chunks)==5)
    idle('generation_disconnect_reset')
    c,b=req('/v1/score',{'prompt_ids':P,'keep':True});gate('score_keep',c==200);owner=b['session']
    c,b=req('/v1/chat/completions',{'prompt_ids':P,'max_tokens':32,'stream':True});gate('stream_owner_409',c==409)
    c,b=req('/v1/continue',{'session':owner,'max_new':16});gate('continue16',c==200 and b['generated_tokens']==16);newer=b['session']
    c,b=req('/v1/continue',{'session':owner,'max_new':1});gate('stale_revision',c==409)
    c,b=req('/v1/continue',{'session':newer,'max_new':32});gate('continue_eos',c==200 and b['finish']=='stop');newer=b['session']
    c,b=req('/v1/decisions',{'session':newer,'prompt_ids':P,'candidates':[[15],[16],[17],[18]],'kind':'choice'})
    gate('decisions',c==200 and not b['calibrated']);idle('decision_release')
    port=int(url.rsplit(':',1)[1]);conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    conn.request('POST','/v1/chat/completions',json.dumps({'prompt_ids':L,'max_tokens':128,'stream':True}),{'Content-Type':'application/json'})
    time.sleep(.1);conn.close();time.sleep(.2);idle('prefill_disconnect_reset')
    c,b=req('/v1/chat/completions',{'messages':[{'role':'user','content':'只回答一个字：好。'}],'max_tokens':32})
    gate('next_message_chat',c==200 and bool(b['choices'][0]['message']['content']));idle('final_reset')
finally:
    if p.poll() is None:
        os.kill(p.pid,signal.CTRL_BREAK_EVENT)
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait(timeout=10)
    log.close();save()
receipt=json.loads((out/'result.json').read_text())
gate('graceful_cli_exit',p.returncode==0)
gate('lease_cleanup',receipt['commands'][0]['exit_code']==0 and receipt['child_cleanup']['active_processes_zero'] and not receipt['marker']['retained'])
raw=(out/'raw/child_00.log').read_text(errors='replace')
cancel=[line.split() for line in raw.splitlines() if line.startswith('DONE ') and line.split()[5]=='cancel']
gate('real_gen_cancel',any(0<int(x[1])<128 for x in cancel))
responses.append({'native_prefill_cancel_reason_seen':any(int(x[1])==0 for x in cancel),'semantics':'in-flight Prefill block may finish before cancellation; generation is skipped and private state reset'})
sock=socket.socket();sock.settimeout(1);gate('port_released',sock.connect_ex(('127.0.0.1',int(url.rsplit(':',1)[1])))!=0);sock.close()
save();print(json.dumps({'gates':gates,'elapsed_s':time.monotonic()-started}))
