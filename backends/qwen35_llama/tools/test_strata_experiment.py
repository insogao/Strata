"""CPU-only budgets, decision rules and measured Profile lifecycle. No launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import strata_experiment as ex
import leased_native  # import/syntax must succeed offline, before any GPU dispatch
import strata_tune as st
from test_strata_cost_screen import local_fp, L


def arm(ms=1000, spread=0):
    return {'metrics': {s + str(i): {'e2e_observed_ms': ms + spread, 'prefill_ms': ms * .8,
                                    'n': n, 'finish': 'length'}
                        for s, n in [('P', 1), ('S', 16), ('L', 128)] for i in range(2)},
            'resources': {'gpu_peak_mib': 7000}}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / 'fake.bin'; self.engine.write_bytes(b'not executable')
        pack = self.root / 'pack'; pack.mkdir(); (pack / 'native_experts.txt').write_bytes(b'fixture')
        shapes = []
        for s, n, m in [('P', 3896, 1), ('S', 3896, 16), ('L', 3817, 128)]:
            p = self.root / (s + '.tokens'); p.write_text(','.join(['1'] * n))
            shapes.append({'id': s, 'tokens_file': str(p), 'sha256': st.file_sha(p),
                           'prompt_tokens': n, 'max_new': m})
        self.spec = {'schema': 'strata-single-gpu-experiment/1', 'cwd': str(self.root.resolve()),
            'engine': {'path': str(self.engine), 'sha256': st.file_sha(self.engine)},
            'model': {'path': 'fake.gguf', 'sha256': 'a' * 64, 'quant': 'fixture'}, 'pack': 'pack',
            'lease': {'tool': str(ex.TOOLS / 'gpu_lock.py'), 'lock': str(self.root / 'gpu.lock')},
            'runtime': {'id': 'fixture', 'cuda_bin': 'unused', 'env': {
                'STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE': '1', 'STRATA_QWEN35_PACKED_IDS': '1',
                'STRATA_QWEN35_EXPERT_ORDER': 'count_desc'}},
            'budget': {'max_native': 4, 'native_timeout_s': 60, 'window_s': 1200, 'wait_s': 1},
            'resources': {'max_gpu_used_mib': 8000, 'max_host_rss_mib': 24000,
                          'min_available_ram_mib': 256, 'baseline_gpu_estimate_mib': 7500,
                          'estimated_host_mib': 16000},
            'baseline': {'id': 'base', 'config': {'slots': 3200, 'threads': 8, 'ubatch': 4096,
                                                 'ctx': 4096, 'compact': False}},
            'candidates': [], 'shapes': shapes, 'policy': {'min_gain': .05, 'early_regression': .15}}
        self.fp = local_fp()
        self.fp['gpu'].update(uuid=L('GPU-fixture'), compute_capability=L('8.9'),
                             driver_version=L('fixture'), cuda_driver_max=L('12.8'))
        self.fp['cpu']['model_name'] = L('fixture CPU')
        self.fp['host'] = {'os': L('fixture OS')}
        self.fp['ram']['commit_avail_mib'] = L(48000)
        self.fp['model']['mtime_utc'] = L('fixture')
        self.fp['model']['gguf']['routed_expert_layer_bytes'] = L([300000000] * 40)

    def fake_runner(self, spec, config, directory, started):
        directory.mkdir()
        (directory / 'raw.log').write_bytes(b'CPU synthetic evidence, not hardware')
        (directory / 'result.json').write_text(json.dumps({'commands': [{'pid': 1}]}))
        return arm()

    def profile(self):
        return ex.execute(self.spec, self.root / 'record', smoke=True, runner=self.fake_runner, fp=self.fp)

    def test_profile_and_identity_roundtrip_never_installs_or_selects_baseline(self):
        p = self.profile()
        self.assertEqual(p['status'], 'measured')
        self.assertIsNone(p['selected_configuration'])
        self.assertFalse(p['auto_apply']); self.assertFalse(p['production_approved'])
        self.assertTrue(ex.check(p, self.spec, self.fp)['valid'])
        self.assertEqual(p['native_launches'], 1)

    def test_evidence_and_pack_mutation_invalidate(self):
        p = self.profile()
        (self.root / 'record/baseline-profile/raw.log').write_bytes(b'changed')
        self.assertFalse(ex.check(p, self.spec, self.fp)['valid'])
        (self.root / 'pack/native_experts.txt').write_bytes(b'changed')
        self.assertFalse(ex.check(p, self.spec, self.fp)['valid'])

    def test_model_runtime_workload_and_device_changes_invalidate(self):
        p = self.profile()
        for section, key, value in [('model', 'quant', 'other'), ('runtime', 'id', 'other')]:
            changed = copy.deepcopy(self.spec); changed[section][key] = value
            self.assertFalse(ex.check(p, changed, self.fp)['valid'])
        fp = copy.deepcopy(self.fp); fp['gpu']['uuid'] = L('other')
        self.assertFalse(ex.check(p, self.spec, fp)['valid'])

    def test_resealed_unmeasured_winner_is_refused(self):
        p = self.profile(); p['selected_configuration'] = self.spec['baseline']['config']
        p['status'] = 'validated'; p.pop('content_sha256'); p['content_sha256'] = st.digest(p)
        self.assertFalse(ex.check(p, self.spec, self.fp)['valid'])

    def test_memory_refusal_consumes_no_native(self):
        self.spec['resources']['baseline_gpu_estimate_mib'] = 9000
        calls = []
        p = ex.execute(self.spec, self.root / 'refused', smoke=True,
                       runner=lambda *x: calls.append(x), fp=self.fp)
        self.assertFalse(calls)
        self.assertEqual(p['native_attempts'], 0)
        self.assertEqual(p['native_launches'], 0)
        self.assertIsNone(p['selected_configuration'])
        self.assertIn('capacity refused', p['failure'])

    def test_failure_retained_without_retry(self):
        calls = []
        def fail(*args):
            calls.append(args); raise TimeoutError('fixture deadline')
        p = ex.execute(self.spec, self.root / 'failed', smoke=True, runner=fail, fp=self.fp)
        self.assertEqual(len(calls), 1); self.assertEqual(p['native_attempts'], 1)
        self.assertTrue((self.root / 'failed/failure.json').is_file())
        self.assertIsNone(p['selected_configuration'])

    def test_real_measurement_only_decision_and_drift(self):
        self.assertEqual(ex.compare([arm(), arm(900), arm(900), arm()], self.spec['policy'])['verdict'], 'PASS')
        for trials in ([arm(), arm(1100), arm(1100), arm()],
                       [arm(), arm(950), arm(850), arm()], [arm(), arm(900)]):
            self.assertEqual(ex.compare(trials, self.spec['policy'])['verdict'], 'NO-GO')

    def test_candidate_generation_small_single_variable_and_no2200(self):
        p = self.profile(); cs = ex.generate_candidates(self.spec, p)
        self.assertLessEqual(len(cs), 3)
        for c in cs:
            self.assertNotEqual(c['config']['slots'], 2200)
            self.assertEqual(sum(c['config'][k] != self.spec['baseline']['config'][k] for k in ex.CONFIG_KEYS), 1)

    def test_input_engine_and_budget_guards(self):
        ex.validate(self.spec)
        for k, value in [('max_native', 100), ('wait_s', 61), ('native_timeout_s', -1)]:
            bad = copy.deepcopy(self.spec); bad['budget'][k] = value
            with self.assertRaises(ValueError): ex.validate(bad)
        self.engine.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'engine SHA'): ex.validate(self.spec)

    def test_n1_has_no_decode_tps_or_tpot(self):
        a = {'tokens': [1], 'token_observation_s': [1], 'started_s': 0, 'finished_s': 2,
             'done': {'prompt_tokens': 3896, 'prompt_read': 3896, 'prompt_ms': 1000,
                      'decode_ms': 0, 'finish': 'length'}}
        m = ex.metrics(a)
        self.assertIsNone(m['decode_observed_tps']); self.assertIsNone(m['tpot_observed_ms'])


if __name__ == '__main__': unittest.main()
