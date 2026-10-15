#!/usr/bin/env python3
"""Recompute the bounded long-session regression from original engine rows.

No model, GPU or engine is started. Input is an ordered command plan and raw
gpu-lock child output. Interactive STOP has no response of its own. A requested
maximum is never evidence of actual output length. File-reader timestamps are
not used as token emission times. Container checks require local payload files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from verify_session_native import parse


def done(answer):
    rows = [s.split() for s in answer['lines'] if s.startswith('DONE ')]
    if len(rows) != 1:
        raise ValueError('expected exactly one DONE')
    row = rows[0]
    prompt_ms, decode_ms = float(row[3]), float(row[4])
    if not all(math.isfinite(x) and x >= 0 for x in (prompt_ms, decode_ms)):
        raise ValueError('invalid timing')
    return {'produced': int(row[1]), 'prompt_tokens': int(row[2]),
            'prompt_ms': prompt_ms, 'decode_ms': decode_ms,
            'finish': row[5], 'reused': int(row[8]), 'prompt_read': int(row[14])}


def verify(plan, raw, state_files=None):
    r = parse(plan, raw)
    p = plan['prompt_tokens']
    full = r['full']['tokens']
    gates = {'actual_512_length': len(full) == 512 and done(r['full'])['finish'] == 'length',
             'full_committed_cursor': r['full_state']['state']['pos'] == p + len(full)}
    if plan.get('requires_decode_warmup'):
        warm = r['decode_warmup']
        gates['actual_decode_warmup_512'] = len(warm['tokens']) == 512 and done(warm)['finish'] == 'length'
    for n, chunk in [(128, 32), (256, 64), (512, 64)]:
        combined = []
        for k in range(n // chunk):
            a = r[f'seg{n}_{k}']
            combined += a['tokens']
            gates[f'seg{n}_{k}_actual_length'] = len(a['tokens']) == chunk and done(a)['finish'] == 'length'
            gates[f'seg{n}_{k}_cursor'] = r[f'seg{n}_{k}_state']['state']['pos'] == p + len(combined)
            gates[f'seg{n}_{k}_no_prefill_replay'] = done(a)['prompt_read'] == 0 and done(a)['reused'] == p + k * chunk
        gates[f'seg{n}_equals_single_prefix'] = len(full) >= n and combined == full[:n]
    singles = []
    for k in range(16):
        singles += r[f'one{k}']['tokens']
        gates[f'one{k}_actual_token_cursor'] = (len(r[f'one{k}']['tokens']) == 1 and
                                               r[f'one{k}_state']['state']['pos'] == p + k + 1)
    singles += r['one_tail']['tokens']
    gates['single_token_boundaries_equal_128'] = len(singles) == 128 and singles == full[:128]
    gates['single_token_final_cursor'] = r['one_final_state']['state']['pos'] == p + len(singles)
    gates['length_error_refused'] = any(s.startswith('ERR ') for s in r['length_error']['lines'])
    gates['length_error_state_unchanged'] = r['error_before']['state'] == r['error_after']['state']
    gates['long_save_prefix_actual_128'] = len(r['state_prefix']['tokens']) == 128 and r['state_prefix']['tokens'] == full[:128]
    for name, prefix in [('save', 'SAVED '), ('load', 'LOADED '), ('resave', 'SAVED '),
                         ('reload', 'LOADED '), ('resave2', 'SAVED ')]:
        gates[name + '_succeeded'] = any(s.startswith(prefix) for s in r[name]['lines'])
    gates['load_pure_refused'] = any('no logits' in s for s in r['load_pure']['lines'])
    gates['load_refusal_state_unchanged'] = r['load_before']['state'] == r['load_after']['state']
    gates['restored_append_actual_32'] = len(r['restored_append']['tokens']) == len(r['live_append']['tokens']) == 32
    gates['restored_append_equals_live'] = r['restored_append']['tokens'] == r['live_append']['tokens']
    gates['restored_append_cursor'] = (r['live_append_state']['state']['pos'] ==
                                      r['restored_append_state']['state']['pos'] == p + 128 + 4 + 32)
    canceled = r['cancel']['tokens']
    continued = canceled + r['after_cancel']['tokens']
    gates['actual_mid_decode_cancel'] = 8 <= len(canceled) < 512 and done(r['cancel'])['finish'] == 'cancel'
    gates['cancel_cursor'] = r['cancel_state']['state']['pos'] == p + len(canceled)
    gates['after_cancel_matches_single_prefix'] = len(full) >= len(continued) and continued == full[:len(continued)]
    gates['after_cancel_cursor'] = r['after_cancel_state']['state']['pos'] == p + len(continued)
    gates['fresh_after_error_cancel_matches_128'] = (len(r['fresh_after_error_cancel']['tokens']) == 128 and
                                                    r['fresh_after_error_cancel']['tokens'] == full[:128])
    gates['fresh_after_error_cancel_cursor'] = r['fresh_after_error_cancel_state']['state']['pos'] == p + 128
    gates['real_short_eos'] = done(r['short_eos'])['finish'] == 'stop' and bool(r['short_eos']['tokens'])
    gates['eos_committed_cursor'] = (r['short_eos_state']['state']['pos'] ==
                                    plan['short_prompt_tokens'] + len(r['short_eos']['tokens']))
    gates['no_unexpected_errors'] = all(not any(s.startswith('ERR ') for s in a['lines'])
                                      for name, a in r.items() if name not in {'length_error', 'load_pure'})
    container_checks = None
    if state_files is not None:
        exports_ok = all(gates[n + '_succeeded'] for n in ['save', 'load', 'resave', 'reload', 'resave2'])
        paths = [Path(p) for p in state_files]
        existing = len(paths) == 3 and all(p.is_file() for p in paths)
        blobs = [p.read_bytes() for p in paths] if exports_ok and existing else []
        gates['container_roundtrips_byte_equal'] = len(blobs) == 3 and blobs[0] == blobs[1] == blobs[2]
        container_checks = {str(p): hashlib.sha256(b).hexdigest() for p, b in zip(paths, blobs)}
    replay_names = {f'seg{n}_equals_single_prefix' for n in (128, 256, 512)} | {
        'single_token_boundaries_equal_128', 'long_save_prefix_actual_128',
        'after_cancel_matches_single_prefix', 'fresh_after_error_cancel_matches_128'}
    # Identical text across independent prefills is a useful numeric diagnostic,
    # not a product correctness requirement. Cursor/control/container checks stay
    # separate. Neither group certifies business quality or tensor equivalence.
    return {'registered_all_checks_pass': all(gates.values()),
            'structural_checks_pass': all(v for k, v in gates.items() if k not in replay_names),
            'exact_replay_diagnostics_pass': all(gates[k] for k in replay_names),
            'exact_replay_diagnostic_names': sorted(replay_names),
            'gates': gates, 'responses': r,
            'container_checks': container_checks,
            'scope': 'greedy structural checks and exact replay diagnostics; seeded append after LOAD; not numerical tensor equivalence or product quality',
            'container_gate_checked': state_files is not None}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan', required=True)
    cli.add_argument('--raw', required=True)
    cli.add_argument('--out', required=True)
    cli.add_argument('--state-file', action='append')
    cli.add_argument('--require-exact-replay', action='store_true',
                     help='also fail the exit status on exact replay differences; diagnostic opt-in')
    a = cli.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    raw = Path(a.raw).read_bytes()
    report = verify(plan, raw.decode('utf-8', 'replace'), a.state_file)
    report['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    report['plan_sha256'] = hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'structural_checks_pass': report['structural_checks_pass'],
                      'exact_replay_diagnostics_pass': report['exact_replay_diagnostics_pass'],
                      'registered_all_checks_pass': report['registered_all_checks_pass'],
                      'checks': len(report['gates']),
                      'container_gate_checked': report['container_gate_checked'],
                      'failed': [k for k, v in report['gates'].items() if not v]}))
    accepted = report['registered_all_checks_pass'] if a.require_exact_replay else report['structural_checks_pass']
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
