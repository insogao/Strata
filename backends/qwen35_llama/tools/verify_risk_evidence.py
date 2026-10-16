"""Independent CPU arithmetic for anonymized P4 observations, no source text."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(directory):
    p=Path(directory)
    load=lambda name:json.loads((p/name).read_text(encoding='utf-8'))
    manifest=load('sha256.json');checks=0
    for name,sha in manifest.items():
        path=(p/name).resolve()
        assert path.is_relative_to(p.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==sha
        checks+=1
    data=load('observations.json');result=load('results.json')
    paired=data['paired'];mixed=data['mixed']
    for m in ('A0','A1','B1'):
        rows=[r for r in paired if r['method']==m]
        stats=result['paired']['speed'][m]
        assert len(rows)==stats['n']==4;checks+=1
        assert math.isclose(statistics.mean(r['e2e_ms'] for r in rows),stats['mean_e2e_ms']);checks+=1
        assert math.isclose(statistics.mean(r['generated_tokens'] for r in rows),stats['mean_actual_output_tokens']);checks+=1
        diag=result['paired']['synthetic_reference_diagnostics'][m]
        matrix=[[0]*4 for _ in range(4)]
        for r in rows:
            if not r.get('invalid_reason'):matrix[r['reference_risk']][r['parsed_risk']]+=1
        assert matrix==diag['confusion_gold_rows'];checks+=1
        assert sum(matrix[g][g] for g in range(4))/len(rows)==diag['exact_accuracy'];checks+=1
        assert sum(matrix[g][v] for g in range(4) for v in range(4) if abs(g-v)<=1)/len(rows)==diag['within_one_accuracy'];checks+=1
    # Directly sum exp(full-vocabulary logprob), not a softmax of the four scores.
    for r in paired:
        if r['method']=='B1':
            ps=[math.exp(v) for v in r['candidate_logprobs']]
            assert math.isclose(sum(ps),r['label_mass'],rel_tol=1e-10);checks+=1
            assert max(range(4),key=lambda i:r['candidate_logprobs'][i])==r['parsed_risk'];checks+=1
    for arm in ('M0','M1','M2'):
        rows=[r for r in mixed if r['arm']==arm]
        stats=result['mixed_corrected']['arms'][arm]
        assert len(rows)==stats['submitted'];checks+=1
        assert sum(not r.get('invalid_reason') for r in rows)==stats['completed'];checks+=1
        assert math.isclose(stats['valid_completed_per_s'],stats['completed']*1000/stats['wall_ms']);checks+=1
        if arm!='M0':
            assert all(r['finish']=='stop' and r['generated_tokens']==1 and r['invalid_reason']=='explanation_empty' for r in rows);checks+=1
            assert all(r['continuation_prompt_read']==12 and r['continuation_reused']==r['prompt_tokens'] for r in rows);checks+=1
    assert result['paired']['audit']['human_adjudicated_n']==0;checks+=1
    assert result['paired']['audit']['accuracy_status']=='BLOCKED_NO_GOLD_DATA';checks+=1
    assert result['mixed_corrected']['status']=='NO-GO';checks+=1
    return {'checks':checks,'status':'PASS','scope':'CPU arithmetic and SHA binding; private native original hashes are an index, not public independent raw replay'}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');a=ap.parse_args(argv)
    print(json.dumps(verify(a.directory),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
