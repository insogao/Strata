"""Independent arithmetic over anonymized balanced600 CSV; no private data/GPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(directory):
    p=Path(directory);load=lambda f:json.loads((p/f).read_text(encoding='utf-8'));checks=0
    for f,sha in load('sha256.json').items():
        path=(p/f).resolve()
        assert path.is_relative_to(p.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==sha
        checks+=1
    with (p/'observations.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    results=load('results.json')
    assert len(rows)==1800 and len({(r['case_index'],r['method']) for r in rows})==1800;checks+=1
    assert results['reference_kind']=='synthetic_design_label' and results['n']==600;checks+=1
    cold={r['case_index'] for r in rows if r['cold_warm']=='first_request'}
    assert len(cold)==20;checks+=1
    for method in ('A0','A1','B1'):
        rs=[r for r in rows if r['method']==method]
        matrix=[[0]*4 for _ in range(4)]
        for r in rs:
            assert not r['invalid_reason'] and r['finish']==('score' if method=='B1' else 'stop')
            matrix[int(r['reference_risk'])][int(r['parsed_risk'])]+=1
        assert all(sum(row)==150 for row in matrix);checks+=1
        a=results['accuracy'][method];perf=results['performance'][method]
        assert matrix==a['confusion_gold_rows'];checks+=1
        measurements={
            'exact_accuracy':sum(matrix[g][g] for g in range(4))/600,
            'within_one_accuracy':sum(matrix[g][v] for g in range(4) for v in range(4) if abs(g-v)<=1)/600,
            'mae_valid_only':sum(matrix[g][v]*abs(g-v) for g in range(4) for v in range(4))/600,
            'mse_valid_only':sum(matrix[g][v]*(g-v)**2 for g in range(4) for v in range(4))/600,
            'severe_error_rate_all':sum(matrix[g][v] for g in range(4) for v in range(4) if abs(g-v)>=2)/600,
            'risk3_recall':matrix[3][3]/150,
            'risk3_severe_underestimate_rate':sum(matrix[3][:2])/150}
        for key,value in measurements.items():assert math.isclose(a[key],value);checks+=1
        for col,key in [('e2e_ms','mean_e2e_ms'),('prefill_ms','mean_prefill_ms'),('decode_ms','mean_decode_ms'),('generated_tokens','mean_actual_output_tokens')]:
            assert math.isclose(statistics.mean(float(r[col]) for r in rs),perf[key]);checks+=1
        warm=[r for r in rs if r['case_index'] not in cold]
        assert len(warm)==580 and math.isclose(statistics.mean(float(r['e2e_ms']) for r in warm),results['matched_warm'][method]['mean_e2e_ms']);checks+=1
    a={r['case_index']:r for r in rows if r['method']=='A1'};b={r['case_index']:r for r in rows if r['method']=='B1'}
    assert set(a)==set(b) and all(a[k]['prompt_sha256']==b[k]['prompt_sha256'] for k in a);checks+=1
    assert all(a[k]['parsed_risk']==b[k]['parsed_risk'] for k in a);checks+=1
    for n in results['native']:
        assert n['records']==90 and n['case_count']==30 and n['job_subtree_zero'] and not n['fence_retained'] and n['violation'] is None;checks+=1
    repair=load('explanation-repair.json')
    assert len(repair['rows'])==24;checks+=1
    for arm in ('M0','M1','M2'):
        rs=[r for r in repair['rows'] if r['arm']==arm]
        assert len(rs)==8 and all(not r['invalid_reason'] for r in rs);checks+=1
        stats=repair['summary']['arms'][arm]
        assert stats['completed']==8 and math.isclose(stats['valid_completed_per_s'],8000/stats['wall_ms']);checks+=1
    explained=[r for r in repair['rows'] if r.get('branch')=='explain']
    assert len(explained)==12;checks+=1
    for r in explained:
        assert r['finish']=='stop' and r['visible_text_length']>0 and r['explanation_generated_tokens']>1;checks+=1
        assert r['generated_tokens']==r['label_generated_tokens']+r['explanation_generated_tokens'];checks+=1
        assert r['continuation_reused']==r['prompt_tokens']+r['label_generated_tokens'] and r['continuation_prompt_read']==r['appended_prompt_tokens'];checks+=1
        assert r['state_cursor_verified']==r['continuation_reused']+r['appended_prompt_tokens']+r['explanation_generated_tokens'];checks+=1
    assert repair['native_receipt']['child_cleanup']['active_processes_zero'] and not repair['native_receipt']['marker']['retained'] and repair['resources']['violation'] is None;checks+=1
    return {'status':'PASS','checks':checks,'samples':600,'method_records':1800,
            'scope':'public anonymized arithmetic/SHA, not a human quality claim or private native replay'}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');args=ap.parse_args(argv)
    print(json.dumps(verify(args.directory),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
