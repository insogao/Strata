"""Business-shaped measurements on the existing SCORE/GEN protocol, no new backend."""
import json
import math
from pathlib import Path
import statistics
import time

KINDS = {'score', 'conditional', 'chat'}


def validate(spec):
    ws = spec['workloads']
    if not isinstance(ws, list) or not 1 <= len(ws) <= 6:
        raise ValueError('workloads must contain1..6 entries')
    if len({w['id'] for w in ws}) != len(ws):
        raise ValueError('duplicate workload IDs')
    import re
    for w in ws:
        if not re.fullmatch('[A-Za-z][A-Za-z0-9-]{0,30}', w['id']) or w['kind'] not in KINDS:
            raise ValueError('invalid workload id/kind')
        if w['shape'] not in {'S', 'L'} or not isinstance(w['weight'], (int, float)) or not math.isfinite(w['weight']) or w['weight'] <= 0:
            raise ValueError('invalid workload shape/weight')
        if w.get('quality') != 'structure/1':
            raise ValueError('explicit structural quality scope required; not business approval')
        if w['kind'] != 'chat':
            cs = w.get('candidates')
            if not isinstance(cs, list) or len(cs) < 2 or len(cs) > 32 or len(set(cs)) != len(cs) or any(type(c) is not int or c < 0 for c in cs):
                raise ValueError('explicit unique single-token score candidates required')
        if w['kind'] != 'score' and (type(w['max_new']) is not int or not 1 <= w['max_new'] <= 128):
            raise ValueError('explanation/chat max_new must be1..128')
        if w['kind'] == 'conditional':
            policy = w['explain_if']
            if policy.get('mode') not in {'always', 'never', 'choice-in'}:
                raise ValueError('explicit explanation policy required')
            if policy['mode'] == 'choice-in' and (not policy.get('ids') or not set(policy['ids']) <= set(w['candidates'])):
                raise ValueError('explanation choices must be registered candidates')
        shape = next(s for s in spec['shapes'] if s['id'] == w['shape'])
        if shape['prompt_tokens'] + w.get('max_new', 0) > spec['baseline']['config']['ctx']:
            raise ValueError('workload exceeds actual context')
    if abs(sum(w['weight'] for w in ws) - 1) > 1e-8:
        raise ValueError('workload weights must sum to1')


def probabilities(answer, candidates):
    import score_api as api
    row = next((r for r in answer['lines'] if r.startswith('CANDLOGPROBS ')), None)
    if row is None or answer['tokens'] or answer.get('done', {}).get('finish') != 'score':
        raise ValueError('SCORE must return candidates and zero generated tokens')
    pos, pairs = api._parse_candlogprobs(row)
    if [p['id'] for p in pairs] != candidates:
        raise ValueError('candidate order/count mismatch')
    ps = [math.exp(p['logprob']) for p in pairs]
    mass = sum(ps)
    if not math.isfinite(mass) or not 0 <= mass <= 1.0001:
        raise ValueError('invalid candidate probability mass')
    return {'pos': pos, 'candidate_logprobs': pairs, 'label_mass': mass,
            'choice': pairs[max(range(len(pairs)), key=lambda i: pairs[i]['logprob'])]['id'],
            'calibrated': False}


def should_explain(w, choice):
    rule = w['explain_if']
    return rule['mode'] == 'always' or rule['mode'] == 'choice-in' and choice in rule['ids']


def run(spec, config, directory, started):
    import strata_experiment as ex
    from leased_native import LeasedNative, save
    native = LeasedNative(spec, config, directory, started)
    sampler = None
    if spec.get('telemetry'):
        from strata_diagnostics import Sampler
        sampler = Sampler(native); sampler.start()
    rows = {}
    def reset(name):
        a = native.ask(name, 'SCORE 0 topk=1')
        if a.get('done', {}).get('finish') != 'score': raise ValueError('reset refused')
    try:
        p = spec['shapes'][0]
        if p['id'] != 'P': raise ValueError('P must be the first regression shape')
        for i in range(2):
            a = native.ask('P'+str(i), 'GEN 1 '+','.join(map(str,ex.token_ids(p['tokens_file']))))
            rows['P'+str(i)] = ex.metrics(a)
            if rows['P'+str(i)]['n'] != 1 or rows['P'+str(i)]['prompt_read'] != p['prompt_tokens']:
                raise ValueError('Prefill regression shape failed')
        for w in spec['workloads']:
            shape = next(s for s in spec['shapes'] if s['id'] == w['shape'])
            ids = ex.token_ids(shape['tokens_file']); prompt = ','.join(map(str, ids))
            for i in range(2):
                name = w['id']+str(i); reset(name+'-reset')
                if w['kind'] == 'chat':
                    a = native.ask(name, 'GEN %d %s' % (w['max_new'], prompt))
                    record = ex.metrics(a); record.update(branch='chat', workflow_observed_ms=record['e2e_observed_ms'])
                else:
                    begin = time.monotonic()
                    scored = native.ask(name+'-score', 'SCORE '+prompt+' topk=0 cand='+','.join(map(str,w['candidates']))+(' keep=1' if w['kind']=='conditional' else ''))
                    prob = probabilities(scored,w['candidates'])
                    if scored['done']['prompt_read'] != len(ids) or scored['done']['reused'] != 0:
                        raise ValueError('score did not consume isolated actual prompt')
                    record = {'n':0, 'finish':'score', 'prefill_ms':scored['done']['prompt_ms'],
                              'score_response_ms':1000*(time.monotonic()-begin), **prob}
                    explain = w['kind']=='conditional' and should_explain(w,prob['choice'])
                    if explain:
                        a = native.ask(name+'-explain','GEN %d cont=1'%w['max_new'])
                        record.update(ex.metrics(a))
                        record['prefill_ms'] = scored['done']['prompt_ms']
                        if a['done']['prompt_read'] != 0 or a['done']['reused'] != len(ids):
                            raise ValueError('conditional explanation repeated prefill')
                    record.update(branch='explain' if explain else 'score-only',
                                  workflow_observed_ms=1000*(time.monotonic()-begin))
                    state = json.loads(native.ask(name+'-state','STATE')['lines'][-1][6:])
                    expected = len(ids)+record['n'] if w['kind']=='conditional' else 0
                    if state['pos'] != expected or state['session'] != (w['kind']=='conditional'):
                        raise ValueError('workflow state cursor/isolation failed')
                record['requests_per_s'] = 1000/record['workflow_observed_ms']
                record['quality'] = 'structure/1; business accuracy unverified'
                rows[name] = record
                reset(name+'-cleanup')
    except BaseException as exc:
        save(Path(directory)/'failure.json', {'error':repr(exc),'retry':False}); raise
    finally:
        if sampler is not None:
            sampler.finish(Path(directory)/'telemetry.json')
        native.close(abort=True)
    native.receipt()
    report = {'metrics':rows,'load_and_ready_s':native.load_ready_s,
              'resources':json.loads((Path(directory)/'resources.json').read_text())}
    save(Path(directory)/'measurements.json', report)
    return report


def compare(arms, policy, workloads):
    if len(arms) != 4: return {'verdict':'NO-GO','reason':'incomplete ABBA'}
    ratios, bands, details = [], [], {}
    for w in workloads:
        rows = [a['metrics'][w['id']+'1'] for a in arms]
        if len({(r['branch'],r['n'],r['finish']) for r in rows}) != 1:
            return {'verdict':'NO-GO','reason':'branch/output count/finish mismatch; not comparable'}
        times = [r['workflow_observed_ms'] for r in rows]
        a,b = [times[0],times[3]],times[1:3];ma,mb=statistics.median(a),statistics.median(b)
        if min(times)<=0 or not all(math.isfinite(t) for t in times): raise ValueError('invalid workflow timing')
        band=max((max(a)-min(a))/ma,(max(b)-min(b))/mb)
        ratios.append(w['weight']*mb/ma);bands.append(band)
        details[w['id']]={'baseline_ms':a,'candidate_ms':b,'ratio':mb/ma,'drift_fraction':band}
    gain=1-sum(ratios)
    paired=all(sum(w['weight']*arms[j]['metrics'][w['id']+'1']['workflow_observed_ms']/arms[k]['metrics'][w['id']+'1']['workflow_observed_ms'] for w in workloads)<1 for j,k in [(1,0),(2,3)])
    p=[a['metrics']['P1']['prefill_ms'] for a in arms]
    pratio=statistics.median(p[1:3])/statistics.median([p[0],p[3]])
    pband=max(abs(p[0]-p[3])/statistics.median([p[0],p[3]]),abs(p[1]-p[2])/statistics.median(p[1:3]))
    ok=gain>=policy['min_gain'] and gain>max(bands+[pband]) and paired and pratio<=1+max(.03,pband) and all(x['ratio']<=1+max(.03,x['drift_fraction']) for x in details.values())
    return {'verdict':'PASS' if ok else 'NO-GO','objective_gain_fraction':gain,'drift_fraction':max(bands+[pband]),'paired_directions_positive':paired,'shapes':details,'prefill_ratio':pratio,'quality':'structure only; not business accuracy','production_approved':False}
