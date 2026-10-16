"""Ordinal risk diagnostics. Missing predictions stay in accuracy denominators.

MAE/MSE on failures have no defined numerical distance: report both the
valid-only descriptive value and a conservative failure=maximum-distance value.
Neither value turns synthetic references into human gold.
"""
from collections import Counter, defaultdict
import math
import random


def risk(value):
    return type(value) is int and 0 <= value <= 3


def evaluate(rows):
    rows = list(rows)
    if any(not risk(r['gold_risk']) for r in rows):
        raise ValueError('ordinal reference must be an integer 0..3')
    n = len(rows)
    matrix = [[0] * 4 for _ in range(4)]
    missing = [0] * 4
    errors, high = [], Counter()
    failures = Counter()
    for row in rows:
        g, p = row['gold_risk'], row.get('parsed_risk')
        if not risk(p) or row.get('invalid_reason'):
            missing[g] += 1
            failures[row.get('invalid_reason') or 'invalid_output'] += 1
            continue
        matrix[g][p] += 1
        errors.append(abs(p - g))
        if g >= 2 and p < g:
            high[f'{g}->{p}'] += 1
    div = lambda a, b: a / b if b else None
    support = [sum(matrix[g]) + missing[g] for g in range(4)]
    recalls = [div(matrix[g][g], support[g]) for g in range(4)]
    f1 = []
    for g in range(4):
        tp = matrix[g][g]
        fn = support[g] - tp
        fp = sum(matrix[i][g] for i in range(4) if i != g)
        f1.append(div(2 * tp, 2 * tp + fp + fn))
    invalid = sum(missing)
    return {'n': n, 'valid': len(errors), 'invalid': invalid, 'failures': dict(failures),
            'confusion_gold_rows': matrix, 'invalid_by_gold': missing, 'support': support,
            'exact_accuracy': div(sum(e == 0 for e in errors), n),
            'within_one_accuracy': div(sum(e <= 1 for e in errors), n),
            'mae_valid_only': div(sum(errors), len(errors)),
            'mse_valid_only': div(sum(e * e for e in errors), len(errors)),
            'mae_failure_penalty_3': div(sum(errors) + 3 * invalid, n),
            'mse_failure_penalty_9': div(sum(e * e for e in errors) + 9 * invalid, n),
            'severe_error_rate_all': div(sum(e >= 2 for e in errors), n),
            'severe_or_invalid_rate': div(sum(e >= 2 for e in errors) + invalid, n),
            'per_class_recall': recalls,
            'macro_f1_supported': sum(v for v in f1 if v is not None) / sum(v is not None for v in f1) if any(v is not None for v in f1) else None,
            'absent_classes': [g for g in range(4) if support[g] == 0],
            'risk3_recall': recalls[3],
            'risk3_to2': matrix[3][2],
            'risk3_severe_underestimate_rate': div(sum(matrix[3][:2]), support[3]),
            'risk3_invalid': missing[3], 'high_risk_underestimation': dict(high),
            'flow_failure_rate': div(invalid, n)}


def paired(rows_a, rows_b, *, draws=1000, seed=0):
    """Resample entire conversation clusters, never individual correlated rows."""
    if type(draws) is not int or not 100 <= draws <= 10000:
        raise ValueError('bootstrap draws must be 100..10000')
    a, b = list(rows_a), list(rows_b)
    ka = [r['sample_id'] for r in a]
    kb = [r['sample_id'] for r in b]
    if len(set(ka)) != len(ka) or set(ka) != set(kb) or len(set(kb)) != len(kb):
        raise ValueError('paired methods must contain exactly the same unique samples')
    lookup = {r['sample_id']: r for r in b}
    b = [lookup[k] for k in ka]
    groups = defaultdict(list)
    corrected = introduced = agreed = both_valid = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x['gold_risk'] != y['gold_risk'] or x['conversation_id'] != y['conversation_id']:
            raise ValueError('paired reference/cluster mismatch')
        groups[x['conversation_id']].append(i)
        vx = risk(x.get('parsed_risk')) and not x.get('invalid_reason')
        vy = risk(y.get('parsed_risk')) and not y.get('invalid_reason')
        cx, cy = vx and x['parsed_risk'] == x['gold_risk'], vy and y['parsed_risk'] == y['gold_risk']
        corrected += not cx and cy
        introduced += cx and not cy
        both_valid += vx and vy
        agreed += vx and vy and x['parsed_risk'] == y['parsed_risk']
    metrics = ('exact_accuracy', 'within_one_accuracy', 'mae_failure_penalty_3',
               'severe_or_invalid_rate', 'risk3_recall', 'risk3_severe_underestimate_rate')
    aa, bb = evaluate(a), evaluate(b)
    result = {'clusters': len(groups), 'n': len(a), 'corrected_by_b': corrected,
              'introduced_by_b': introduced, 'both_valid': both_valid,
              'agreement_all': agreed / len(a) if a else None,
              'direction': 'B minus A', 'bootstrap_seed': seed, 'bootstrap_draws': draws,
              'status': 'INCONCLUSIVE' if len(groups) < 30 else 'DESCRIPTIVE_CI_NOT_RELEASE_APPROVAL', 'deltas': {}}
    rng = random.Random(seed)
    keys = list(groups)
    samples = {m: [] for m in metrics}
    if keys:
        for _ in range(draws):
            indices = [i for _ in keys for i in groups[rng.choice(keys)]]
            x, y = evaluate(a[i] for i in indices), evaluate(b[i] for i in indices)
            for m in metrics:
                if x[m] is not None and y[m] is not None:
                    samples[m].append(y[m] - x[m])
    for m in metrics:
        values = sorted(samples[m])
        result['deltas'][m] = {'estimate': bb[m] - aa[m] if aa[m] is not None and bb[m] is not None else None,
            'ci95': [values[math.floor(.025 * (len(values)-1))], values[math.ceil(.975 * (len(values)-1))]] if values else None,
            'defined_draws': len(values)}
    return result
