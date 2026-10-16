"""Recompute public workflow/diagnostic evidence with only the Python stdlib."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(directory):
    root=Path(directory);s=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    checks={}
    for item in s['artifacts']:
        checks['sha:'+item['path']]=hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
    for label,run in s['runs'].items():
        raw=(root/run['raw']).read_text(encoding='utf-8',errors='replace')
        pending=[];done=0
        for row in raw.splitlines():
            if row.startswith('T '):pending.append(int(row.split()[1]))
            if row.startswith('DONE '):
                f=row.split();checks[label+':done:'+str(done)]=int(f[1])==len(pending)
                pending=[];done+=1
        checks[label+':no-trailing-token']=not pending
        receipt=run['receipt']
        checks[label+':cleanup']=(receipt['binary_sha256_actual']==s['engine_sha256']
            and receipt['child_cleanup']['active_processes_zero'] and not receipt['marker']['retained']
            and len(receipt['commands'])==1 and receipt['commands'][0]['exit_code']==0)
        if run.get('responses'):
            rr=json.loads((root/run['responses']).read_text(encoding='utf-8'))
            for name,a in rr.items():
                if 'done' not in a:continue
                d=a['done'];checks[label+':'+name+':count']=d['produced']==len(a['tokens'])
                for row in a['lines']:
                    if row.startswith('CANDLOGPROBS '):
                        f=row.split();pairs=[x.split(':') for x in f[3:]];ps=[float(lp) for _,lp in pairs]
                        checks[label+':'+name+':prob']=(len(pairs)==int(f[2]) and not a['tokens']
                            and d['finish']=='score' and all(math.isfinite(x) and x<=0 for x in ps)
                            and math.fsum(math.exp(x) for x in ps)<=1.0001)
            if run.get('measurements'):
                for w in s['workloads']:
                    for i in range(2):
                        name=w['id']+str(i);m=run['measurements']['metrics'][name]
                        if w['kind']=='conditional':
                            state=json.loads(rr[name+'-state']['lines'][-1][6:]);score=rr[name+'-score']['done']
                            expected=score['prompt_tokens']+m['n']
                            checks[label+':'+name+':cursor']=state['session'] and state['pos']==expected
                            if m['branch']=='explain':
                                gen=rr[name+'-explain']['done']
                                checks[label+':'+name+':no-duplicate-prefill']=gen['prompt_read']==0 and gen['reused']==score['prompt_tokens']
                        if w['kind']=='score':checks[label+':'+name+':score-only']=m['n']==0 and m['finish']=='score'
                        checks[label+':'+name+':rate']=abs(m['requests_per_s']-1000/m['workflow_observed_ms'])<1e-8
    candidate=s['raw_candidate_result'][0]
    ratios={}
    for w in s['workloads']:
        vals=[s['runs']['compact-search/compact-on-'+str(i)]['measurements']['metrics'][w['id']+'1']['workflow_observed_ms'] for i in range(4)]
        ratios[w['id']]=statistics.median(vals[1:3])/statistics.median([vals[0],vals[3]])
        checks[w['id']+':ratio']=abs(ratios[w['id']]-candidate['shapes'][w['id']]['ratio'])<1e-10
    gain=1-sum(w['weight']*ratios[w['id']] for w in s['workloads'])
    checks['raw-math-preserved']=abs(gain-candidate['objective_gain_fraction'])<1e-10
    for i in (1,2):
        raw=(root/s['runs']['compact-search/compact-on-'+str(i)]['raw']).read_text()
        checks['candidate'+str(i)+':not-applied']='staying uniform' in raw and 'native expert tier (compact)' not in raw
    checks['no-compact-speedup-claim']=s['configuration_verdict'].startswith('INVALID_CONFIGURATION') and s['selected_configuration'] is None
    for label,windows in s['diagnostic_windows'].items():
        r=s['runs'][label];responses=json.loads((root/r['responses']).read_text())
        telemetry=json.loads((root/r['telemetry']).read_text())['samples']
        for name,w in windows.items():
            t=responses[name]['token_observation_s'];rate=(len(t)-1)/(t[-1]-t[0])
            checks[label+':'+name+':tps']=abs(rate-w['decode_observed_tps'])<1e-8
            rows=[a for a in telemetry if t[0]<=a['at']<=t[-1]]
            checks[label+':'+name+':power-event']=any(x['gpu'].get('throttle_reasons',0)&4 for x in rows)==w['power_cap_seen']
    checks['stream-gates']=all(s['stream_final']['gates'].values())
    checks['preserve-first-stream-failure']=len(s['stream_first']['gates'])<len(s['stream_final']['gates'])
    return {'pass':all(checks.values()),'checks':len(checks),'failures':[k for k,v in checks.items() if not v],
        'scope':'raw counts/probabilities/state/no-reprefill/timing arithmetic/config fallback; not a causal proof or business quality'}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory')
    result=verify(ap.parse_args().directory);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['pass'] else 1)
