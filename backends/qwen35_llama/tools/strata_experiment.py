"""Explicit, bounded, single-NVIDIA-GPU measurement. No default/profile install.

Offline inspect/screen/plan never launch. `experiment --execute` is the only
launch command, with exact manifest identities and the canonical GPU Job lock.
Current online transport is Windows; other platforms fail before launching.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics as stats
import subprocess
import sys
import time

import strata_tune as st
import strata_fingerprint as sf

TOOLS = Path(__file__).resolve().parent
LOCK_SHA = '337426d63d5e36f199b8bca3e46bea041987a5d68272f426a5976f50da2a7111'
CONFIG_KEYS = {'slots', 'threads', 'ubatch', 'ctx', 'compact'}
ENV = {'STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE', 'STRATA_QWEN35_PACKED_IDS', 'STRATA_QWEN35_EXPERT_ORDER'}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def token_ids(path):
    text = Path(path).read_text(encoding='utf-8-sig').strip()
    ids = [int(x) for x in re.split(r'[\s,]+', text)]
    if not ids or any(x < 0 or x > 2147483647 for x in ids):
        raise ValueError('invalid fixture IDs')
    return ids


def validate(spec):
    if spec.get('schema') != 'strata-single-gpu-experiment/1':
        raise ValueError('unsupported experiment schema')
    for obj, field in ((spec['model'], 'sha256'), (spec['engine'], 'sha256')):
        if not re.fullmatch('[a-f0-9]{64}', str(obj.get(field, ''))):
            raise ValueError('full registered SHA256 required')
    if not Path(spec['cwd']).is_absolute() or not Path(spec['lease']['lock']).is_absolute():
        raise ValueError('cwd and shared lock must be absolute')
    if st.file_sha(spec['engine']['path']) != spec['engine']['sha256']:
        raise ValueError('engine SHA mismatch')
    if st.file_sha(spec['lease']['tool']) != LOCK_SHA:
        raise ValueError('canonical lease SHA mismatch')
    budget = spec['budget']
    for k, maximum in [('max_native', 16), ('native_timeout_s', 300), ('window_s', 1200), ('wait_s', 60)]:
        if type(budget[k]) is not int or not 0 < budget[k] <= maximum:
            raise ValueError('invalid bounded budget: ' + k)
    resources = spec['resources']
    for k in ('max_gpu_used_mib', 'max_host_rss_mib', 'min_available_ram_mib',
              'baseline_gpu_estimate_mib', 'estimated_host_mib'):
        if not isinstance(resources[k], (int, float)) or not math.isfinite(resources[k]) or resources[k] <= 0:
            raise ValueError('invalid resource bound ' + k)
    if set(spec['runtime']['env']) != ENV:
        raise ValueError('runtime env must contain exactly the registered3 production flags')
    if spec['runtime']['env'] != {'STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE': '1',
            'STRATA_QWEN35_PACKED_IDS': '1', 'STRATA_QWEN35_EXPERT_ORDER': 'count_desc'}:
        raise ValueError('unsupported runtime flags')
    candidates = spec['candidates']
    if len(candidates) > 3 or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('at most3 distinct candidate IDs')
    for c in [spec['baseline']] + candidates:
        if not re.fullmatch('[a-zA-Z0-9-]{1,40}', c['id']):
            raise ValueError('unsafe configuration ID')
        cfg = c['config']
        if set(cfg) != CONFIG_KEYS or type(cfg['compact']) is not bool:
            raise ValueError('configuration schema mismatch')
        for k, maximum in [('slots', 10240), ('threads', os.cpu_count() or 1), ('ubatch', 4096), ('ctx', 4608)]:
            if type(cfg[k]) is not int or not 0 < cfg[k] <= maximum:
                raise ValueError('invalid configuration ' + k)
        if cfg['ubatch'] > cfg['ctx']:
            raise ValueError('ubatch exceeds context')
        if c in candidates:
            changed = [k for k in CONFIG_KEYS if cfg[k] != spec['baseline']['config'][k]]
            if len(changed) != 1 or changed == ['ctx']:
                raise ValueError('each candidate changes one meaningful variable; context fixed')
            if cfg['slots'] == 2200:
                raise ValueError('2200-slot route archived; requires a new separately reviewed hypothesis')
    if {s['id'] for s in spec['shapes']} != {'P', 'S', 'L'} or len(spec['shapes']) != 3:
        raise ValueError('exactly P/S/L shapes required')
    for shape in spec['shapes']:
        if st.file_sha(shape['tokens_file']) != shape['sha256']:
            raise ValueError('fixture SHA mismatch')
        ids = token_ids(shape['tokens_file'])
        if len(ids) != shape['prompt_tokens']:
            raise ValueError('actual prompt count mismatch')
        expected = {'P': (3896, 1), 'S': (3896, 16), 'L': (3817, 128)}[shape['id']]
        if (len(ids), shape['max_new']) != expected:
            raise ValueError('this experiment version requires fixed P/S/L dimensions')
        if len(ids) + shape['max_new'] > spec['baseline']['config']['ctx']:
            raise ValueError('illegal context budget')
    if not 0 < spec['policy']['min_gain'] <= .5 or not 0 < spec['policy']['early_regression'] <= .5:
        raise ValueError('invalid decision policy')


def fingerprint(spec):
    fp = sf.build_fingerprint(str(Path(spec['cwd']) / spec['model']['path']),
            {'context': spec['baseline']['config']['ctx'], 'max_new': 128, 'batch': 1}, spec['model']['sha256'])
    rc, output = sf._run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'])
    uuids = output.strip().splitlines()
    if rc != 0 or len(uuids) != 1 or not uuids[0].startswith('GPU-'):
        raise ValueError('exactly one identifiable NVIDIA GPU required')
    fp['gpu']['uuid'] = sf.field(uuids[0], 'nvidia-smi')
    if 'error' in st.values(fp['model']['gguf']):
        raise ValueError('GGUF header unavailable')
    return fp


def identity(spec, fp):
    work = {'context': spec['baseline']['config']['ctx'], 'max_new': 128,
            'batch': 1, 'manifest_sha256': st.digest(spec)}
    binding = st.binding(fp, spec['engine']['path'], work, 'safe', 4, spec['runtime']['id'])
    binding['experiment_tools'] = {p: st.file_sha(TOOLS / p) for p in
                                  ('strata_experiment.py', 'leased_native.py', 'gpu_lock.py',
                                   'strata_tokenizer.py', 'score_api.py')}
    binding['telemetry_source_sha256'] = st.file_sha(TOOLS.parents[2] / 'serve/telemetry.py')
    binding['quant'] = spec['model']['quant']
    binding['python'] = sys.version
    binding['packages'] = {}
    for package in ('psutil', 'pywin32', 'regex', 'jinja2'):
        try:
            binding['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            binding['packages'][package] = None
    if spec.get('baseline_profile'):
        binding['baseline_profile_sha256'] = st.file_sha(spec['baseline_profile'])
    pack = Path(spec['cwd']) / spec['pack']
    binding['pack_index_sha256'] = st.file_sha(pack / 'native_experts.txt')
    binding['pack_files'] = {str(p.relative_to(pack)): [p.stat().st_size, p.stat().st_mtime_ns]
                             for p in sorted(pack.rglob('*')) if p.is_file()}
    return binding


def admission(spec, fp, config, baseline_peak=None):
    """Explicit conservative estimate; live monitoring is still mandatory."""
    model = st.values(fp['model']['gguf'])
    layer_bytes = model['routed_expert_layer_bytes']
    maximum_slot_mib = max(layer_bytes) / model['expert_count'] / 2**20
    baseline = spec['baseline']['config']
    estimate = baseline_peak if baseline_peak is not None else spec['resources']['baseline_gpu_estimate_mib']
    estimate += max(0, config['slots'] - baseline['slots']) * maximum_slot_mib
    # Do not infer extra capacity from a smaller cache or COMPACT until measured.
    reasons = []
    if estimate > spec['resources']['max_gpu_used_mib']:
        reasons.append('predicted GPU memory exceeds registered budget')
    ram = st.values(fp['ram'])
    if ram.get('commit_avail_mib', 0) < spec['resources']['estimated_host_mib'] + 1024:
        reasons.append('host commit budget insufficient')
    if config['ubatch'] > baseline['ubatch']:
        reasons.append('workspace growth unmeasured')
    return {'admitted': not reasons, 'reasons': reasons, 'estimated_gpu_mib': estimate,
            'estimate_scope': 'baseline high-water plus upper slot geometry; no capacity claim for reductions'}


def metrics(answer):
    d = answer['done']; t = answer['token_observation_s']; n = len(answer['tokens'])
    span = t[-1] - t[0] if n > 1 else None
    return {'n': n, 'prompt_tokens': d['prompt_tokens'], 'prompt_read': d['prompt_read'],
            'finish': d['finish'], 'eos': d['finish'] == 'stop', 'prefill_ms': d['prompt_ms'],
            'prefill_tps': d['prompt_read'] * 1000 / d['prompt_ms'] if d['prompt_ms'] > 0 else None,
            'decode_ms': d['decode_ms'], 'ttft_observed_ms': (t[0] - answer['started_s']) * 1000 if n else None,
            'e2e_observed_ms': (answer['finished_s'] - answer['started_s']) * 1000,
            'last_visible_observed_ms': (t[-1] - answer['started_s']) * 1000 if n else None,
            'decode_observed_tps': (n - 1) / span if span and span > 0 else None,
            'tpot_observed_ms': span * 1000 / (n - 1) if span and span > 0 else None,
            'timing_scope': 'file observations after engine fflush,5ms polling; not exact native flush timestamps',
            'token_ids_sha256': st.digest(answer['tokens'])}


def run_arm(spec, config, directory, started):
    from leased_native import LeasedNative, save as overwrite
    native = LeasedNative(spec, config, directory, started)
    rows = {}
    try:
        for shape in spec['shapes']:
            prompt = ','.join(map(str, token_ids(shape['tokens_file'])))
            for request in range(2):
                answer = native.ask(shape['id'] + str(request), 'GEN %d %s' % (shape['max_new'], prompt))
                row = metrics(answer)
                if (row['n'] != shape['max_new'] or row['finish'] != 'length'
                        or row['prompt_read'] != shape['prompt_tokens'] or answer['done']['reused'] != 0):
                    raise ValueError('shape failed actual length/finish/isolation gate')
                rows[shape['id'] + str(request)] = row
    except BaseException as exc:
        native.close(abort=True)
        overwrite(Path(directory) / 'failure.json', {'error': str(exc), 'retry': False})
        raise
    native.close()
    native.receipt()
    # Text inspection is after native timing and before the next launch, never
    # CPU tokenizer work concurrent with a measured decode.
    import strata_tokenizer
    tokenizer = strata_tokenizer.Tokenizer.from_gguf(str(Path(spec['cwd']) / spec['model']['path']))
    for request in range(2):
        text = tokenizer.decode(native.responses['S' + str(request)]['tokens'])
        payload = json.loads(text)
        if (not isinstance(payload, dict) or type(payload.get('risk')) is not int
                or payload['risk'] not in range(4) or type(payload.get('violation')) is not bool):
            raise ValueError('short JSON format/fields failed (not a classification accuracy test)')
        rows['S' + str(request)]['json_complete'] = True
    report = {'metrics': rows, 'load_and_ready_s': native.load_ready_s,
              'resources': json.loads((Path(directory) / 'resources.json').read_text())}
    overwrite(Path(directory) / 'measurements.json', report)
    return report


def compare(arms, policy):
    """A B B A; warm rq2 only. Exact independent text is not a decision gate."""
    if len(arms) != 4:
        return {'verdict': 'NO-GO', 'reason': 'incomplete ABBA evidence'}
    a = [arms[0], arms[3]]; b = [arms[1], arms[2]]
    ratios, shape_stats = {}, {}
    bands = []
    for shape in ('P', 'S', 'L'):
        key = 'prefill_ms' if shape == 'P' else 'e2e_observed_ms'
        av = [r['metrics'][shape + '1'][key] for r in a]
        bv = [r['metrics'][shape + '1'][key] for r in b]
        ma, mb = stats.median(av), stats.median(bv)
        if ma <= 0 or mb <= 0:
            return {'verdict': 'NO-GO', 'reason': 'nonpositive timing'}
        band = max((max(av) - min(av)) / ma, (max(bv) - min(bv)) / mb)
        bands.append(band); ratios[shape] = mb / ma
        shape_stats[shape] = {'baseline_ms': av, 'candidate_ms': bv, 'ratio': mb / ma, 'drift_fraction': band}
    gain = 1 - (.9 * ratios['S'] + .1 * ratios['L'])
    paired = all(.9 * b[i]['metrics']['S1']['e2e_observed_ms'] / a[i]['metrics']['S1']['e2e_observed_ms'] +
                 .1 * b[i]['metrics']['L1']['e2e_observed_ms'] / a[i]['metrics']['L1']['e2e_observed_ms'] < 1
                 for i in range(2))
    passed = (gain >= policy['min_gain'] and gain > max(bands) and paired
              and ratios['P'] <= 1 + max(.03, shape_stats['P']['drift_fraction'])
              and ratios['L'] <= 1 + max(.03, shape_stats['L']['drift_fraction']))
    return {'verdict': 'PASS' if passed else 'NO-GO', 'objective_gain_fraction': gain,
            'drift_fraction': max(bands), 'paired_directions_positive': paired, 'shapes': shape_stats,
            'reason': 'measured workload improvement' if passed else 'no reliable all-gates improvement',
            'production_approved': False, 'quality': 'business classification not evaluated'}


def generate_candidates(spec, measured):
    """Small interpretable space, conditioned on an actual baseline Profile."""
    baseline = spec['baseline']['config']
    rows = measured['candidates'][0]['measurement']['metrics']
    prefill_share = rows['S1']['prefill_ms'] / rows['S1']['e2e_observed_ms']
    changes = [('ubatch', max(512, baseline['ubatch'] // 2)),
               ('threads', max(1, baseline['threads'] - 2)), ('slots', 2700)]
    return [{'id': k + str(v), 'config': {**baseline, k: v},
             'hypothesis': ('S prefill dominates; reduce workspace/CPU contention' if prefill_share >= .5 else
                            'decode contributes materially; test cache/CPU resource balance'),
             'baseline_short_prefill_fraction': prefill_share}
            for k, v in changes if baseline[k] != v]


def execute(spec, out, smoke=False, runner=run_arm, fp=None):
    validate(spec)
    fp = fingerprint(spec) if fp is None else fp
    bound = identity(spec, fp)
    if st.missing_identity(bound):
        raise ValueError('incomplete fingerprint identity')
    out = Path(out); out.mkdir(exist_ok=False)
    started = time.time()
    save(out / 'manifest.json', {'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
         'spec': spec, 'identity': bound, 'profile_smoke': smoke, 'no_retry': True})
    arms_used, all_candidates, artifacts = 0, [], []
    candidates = spec['candidates']
    if not smoke:
        measured = st.read_json(spec['baseline_profile'])
        if not check(measured, measured['spec'], fp)['valid'] or measured['fallback'] != spec['baseline']['config']:
            raise ValueError('baseline Profile invalid or configuration mismatch; profile before searching')
        if not candidates:
            candidates = generate_candidates(spec, measured)
        save(out / 'candidate-plan.json', candidates)
    def arm(label, config):
        nonlocal arms_used
        if arms_used >= spec['budget']['max_native']:
            raise ValueError('native budget exhausted')
        left = spec['budget']['window_s'] - (time.time() - started)
        if left < spec['budget']['native_timeout_s'] + spec['budget']['wait_s']:
            raise ValueError('window exhausted; no native started')
        guard = admission(spec, fp, config)
        if not guard['admitted']:
            raise ValueError('capacity refused: ' + str(guard['reasons']))
        arms_used += 1
        result = runner(spec, config, out / label, started)
        artifacts.append(label)
        return result
    failure = None
    try:
        base = spec['baseline']['config']
        if smoke:
            all_candidates.append({'id': 'baseline-profile', 'measurement': arm('baseline-profile', base)})
        else:
            for c in candidates:
                guard = admission(spec, fp, c['config'])
                if not guard['admitted']:
                    all_candidates.append({'id': c['id'], 'verdict': 'NO-GO', 'admission': guard}); continue
                trials = []
                for i, config in enumerate((base, c['config'], c['config'], base)):
                    trials.append(arm(c['id'] + '-' + str(i), config))
                    if i == 1:
                        reg = max(trials[1]['metrics'][s + '1']['e2e_observed_ms'] /
                                  trials[0]['metrics'][s + '1']['e2e_observed_ms'] - 1 for s in ('S', 'L'))
                        if reg > spec['policy']['early_regression']:
                            break
                all_candidates.append({'id': c['id'], 'config': c['config'], **compare(trials, spec['policy'])})
    except BaseException as exc:
        save(out / 'failure.json', {'error': str(exc), 'native_budget_consumed': arms_used,
                                   'unsafe_feature_stopped': True, 'retry': False})
        failure = str(exc)
        all_candidates.append({'id': 'interrupted-experiment', 'verdict': 'NO-GO', 'reason': failure})
    selected = [c for c in all_candidates if c.get('verdict') == 'PASS']
    winner = max(selected, key=lambda c: c['objective_gain_fraction']) if selected else None
    evidence = []
    for p in sorted(out.rglob('*')):
        if p.is_file():
            evidence.append({'path': str(p.resolve()), 'sha256': st.file_sha(p), 'size_bytes': p.stat().st_size})
    profile = {'schema': 'strata-measured-profile/1', 'status': 'validated' if winner else 'measured',
        'validation_scope': 'paired performance and structural workload gates only; not business quality',
        'selected_configuration': winner['config'] if winner else None,
        'fallback': spec['baseline']['config'], 'auto_apply': False, 'production_approved': False,
        'identity': bound, 'identity_sha256': st.digest(bound), 'spec': spec, 'candidates': all_candidates,
        'native_attempts': arms_used,
        'native_launches': sum(bool(c.get('pid')) for p in out.rglob('result.json')
                               for c in st.read_json(p).get('commands', [])),
        'candidate_plan': candidates, 'evidence': evidence,
        'invalidation': 'any engine/model/quant/device/runtime/input/tool/evidence change'}
    profile['failure'] = failure
    if failure:
        profile['status'] = 'measured'
        profile['selected_configuration'] = None
    profile['content_sha256'] = st.digest(profile)
    save(out / 'profile.json', profile)
    return profile


def check(profile, spec, fp):
    reasons = []
    if profile.get('schema') != 'strata-measured-profile/1':
        reasons.append('schema mismatch')
    if profile.get('content_sha256') != st.digest({k: v for k, v in profile.items() if k != 'content_sha256'}):
        reasons.append('profile content changed')
    if profile.get('auto_apply') is not False or profile.get('production_approved') is not False:
        reasons.append('unsafe approval/install claim')
    winners = [c for c in profile.get('candidates', []) if c.get('verdict') == 'PASS']
    selected = profile.get('selected_configuration')
    if profile.get('status') not in {'measured', 'validated'}:
        reasons.append('invalid measurement maturity')
    if ((selected is not None and (profile.get('status') != 'validated' or
            not any(c.get('config') == selected for c in winners))) or
            (selected is None and profile.get('status') == 'validated')):
        reasons.append('selected configuration lacks measured PASS')
    if profile.get('identity') != identity(spec, fp) or profile.get('spec') != spec:
        reasons.append('current identity/spec changed')
    for e in profile.get('evidence', []):
        if not Path(e['path']).is_file() or st.file_sha(e['path']) != e['sha256']:
            reasons.append('evidence missing/changed: ' + e['path'])
    return {'valid': not reasons, 'reasons': reasons, 'production_approved': False, 'auto_apply': False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['inspect', 'screen', 'plan', 'experiment', 'check-experiment'])
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out'); ap.add_argument('--profile')
    ap.add_argument('--execute', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(argv)
    try:
        spec = st.read_json(a.spec); validate(spec)
        fp = fingerprint(spec)
        if a.command in {'inspect', 'screen', 'plan'}:
            value = {'schema': 'strata-experiment-plan/1', 'executed': False,
                     'fingerprint': fp if a.command == 'inspect' else None,
                     'baseline': admission(spec, fp, spec['baseline']['config']),
                     'candidates': [{'id': c['id'], 'config': c['config'], 'admission': admission(spec, fp, c['config'])}
                                    for c in spec['candidates']],
                     'interleave': ['A1', 'B1', 'B2', 'A2'], 'selected_configuration': None,
                     'notice': 'capacity only; no predicted speed recommendation'}
            if a.out: save(a.out, value)
        elif a.command == 'check-experiment':
            if not a.profile: raise ValueError('--profile required')
            value = check(st.read_json(a.profile), spec, fp)
        else:
            if not a.execute or not a.out:
                raise ValueError('real GPU runs require explicit --execute and a new --out directory')
            if os.name != 'nt': raise ValueError('current leased online transport is Windows only')
            # Fail before taking a lease if an optional online dependency is absent.
            import importlib
            for module in ('psutil', 'win32pipe', 'win32file', 'regex', 'jinja2'):
                importlib.import_module(module)
            value = execute(spec, a.out, a.smoke, fp=fp)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 2 if value.get('valid') is False or value.get('failure') else 0
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ImportError) as exc:
        print(json.dumps({'error': str(exc), 'auto_retry': False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
