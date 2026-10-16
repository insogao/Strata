"""CPU-only independent recalculation of the public single-GPU review bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def verify(directory):
    root = Path(directory)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    checks = {}
    for item in summary['raw_artifacts'] + summary['observation_artifacts']:
        p = root / item['path']
        checks['sha:' + p.name] = hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
    for item in summary['observation_artifacts']:
        rows = json.loads((root / item['path']).read_text(encoding='utf-8'))
        metrics = summary['measurements'][item['label']]['metrics']
        raw_item = next(x for x in summary['raw_artifacts'] if x['label'] == item['label'])
        raw = (root / raw_item['path']).read_text(encoding='utf-8', errors='replace')
        done_rows = [s.split() for s in raw.splitlines() if s.startswith('DONE ')]
        token_rows = [int(s.split()[1]) for s in raw.splitlines() if s.startswith('T ')]
        checks['raw-token-order:' + item['label']] = token_rows == sum((v['tokens'] for v in rows.values()), [])
        checks['raw-request-count:' + item['label']] = len(done_rows) == len(rows)
        for (name, a), done in zip(rows.items(), done_rows):
            m = metrics[name]
            stamps = a['token_observation_s']; n = len(a['tokens'])
            checks[item['label'] + ':' + name + ':native'] = (
                n == int(done[1]) == m['n'] and done[5] == m['finish']
                and float(done[3]) == m['prefill_ms'] and int(done[2]) == m['prompt_tokens'])
            checks[item['label'] + ':' + name + ':e2e'] = abs(
                1000 * (a['finished_s'] - a['started_s']) - m['e2e_observed_ms']) < 1e-6
            tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 and stamps[-1] > stamps[0] else None
            checks[item['label'] + ':' + name + ':tps'] = (
                tps is None and m['decode_observed_tps'] is None or
                tps is not None and abs(tps - m['decode_observed_tps']) < 1e-6)
    for c in summary['paired_candidates']:
        ratios = {}
        for shape in ('P', 'S', 'L'):
            key = 'prefill_ms' if shape == 'P' else 'e2e_observed_ms'
            times = [summary['measurements']['search/' + c['id'] + '-' + str(i)]['metrics'][shape + '1'][key]
                     for i in range(4)]
            a, b = [times[0], times[3]], times[1:3]
            ratio = statistics.median(b) / statistics.median(a)
            ratios[shape] = ratio
            checks[c['id'] + ':' + shape + ':ratio'] = abs(ratio - c['shapes'][shape]['ratio']) < 1e-10
        gain = 1 - (.9 * ratios['S'] + .1 * ratios['L'])
        checks[c['id'] + ':gain'] = abs(gain - c['objective_gain_fraction']) < 1e-10
        checks[c['id'] + ':no-go'] = c['verdict'] == 'NO-GO' and gain < .05
    checks['no-install/no-winner'] = (summary['selected_configuration'] is None
                                    and not summary['auto_apply'] and not summary['production_approved'])
    checks['native-cleanup'] = all(r['child_cleanup']['active_processes_zero']
        and not r['marker']['retained'] and all(c['exit_code'] == 0 for c in r['commands'])
        for r in summary['receipts'].values())
    checks['native-sha-bound'] = all(r['binary_sha256_actual'] == summary['engine_sha256']
                                   for r in summary['receipts'].values())
    return {'pass': all(checks.values()), 'checks': len(checks),
            'failures': [k for k, v in checks.items() if not v],
            'scope': 'raw counts/prefill, file-observation timing and paired arithmetic; not business quality'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory')
    result = verify(ap.parse_args().directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
