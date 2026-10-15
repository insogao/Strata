"""CPU replay of real long-trajectory evidence, plus hostile evidence cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_session_long import done, verify


EVIDENCE = Path(__file__).resolve().parents[1] / 'tests/evidence/session_long_20261015'


class LongEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((EVIDENCE / 'warm-review-plan.json').read_text(encoding='utf-8'))
        cls.raw = (EVIDENCE / 'warm-raw.log').read_bytes().decode('utf-8', 'replace')
        cls.report = verify(cls.plan, cls.raw)

    def test_real_warm_structure_passes_but_numeric_replay_diff_is_retained(self):
        self.assertTrue(self.report['structural_checks_pass'])
        self.assertFalse(self.report['exact_replay_diagnostics_pass'])
        self.assertFalse(self.report['registered_all_checks_pass'])
        self.assertFalse(self.report['container_gate_checked'])
        self.assertEqual(len(self.report['responses']['full']['tokens']), 512)

    def test_requested_512_with_early_eos_is_not_actual_length(self):
        row = next(s for s in self.report['responses']['full']['lines'] if s.startswith('DONE '))
        corrupted = self.raw.replace(row, row.replace(' length ', ' stop '), 1)
        self.assertFalse(verify(self.plan, corrupted)['gates']['actual_512_length'])

    def test_done_count_cannot_claim_more_than_actual_token_rows(self):
        row = next(s for s in self.report['responses']['full']['lines'] if s.startswith('DONE '))
        bad = self.raw.replace(row, row.replace('DONE 512 ', 'DONE 513 '), 1)
        with self.assertRaises(ValueError):
            verify(self.plan, bad)

    def test_wrong_committed_cursor_is_detected(self):
        row = next(s for s in self.report['responses']['full_state']['lines'] if s.startswith('STATE '))
        state = json.loads(row[6:])
        state['pos'] -= 1
        bad = self.raw.replace(row, 'STATE ' + json.dumps(state), 1)
        self.assertFalse(verify(self.plan, bad)['gates']['full_committed_cursor'])

    def test_length_exit_does_not_prove_stop(self):
        row = next(s for s in self.report['responses']['cancel']['lines'] if s.startswith('DONE '))
        bad = self.raw.replace(row, row.replace(' cancel ', ' length '), 1)
        self.assertFalse(verify(self.plan, bad)['gates']['actual_mid_decode_cancel'])

    def test_failed_save_cannot_be_proved_by_preexisting_identical_files(self):
        row = next(s for s in self.report['responses']['save']['lines'] if s.startswith('SAVED '))
        bad = self.raw.replace(row, 'ERR session: denied', 1)
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / str(i) for i in range(3)]
            for p in paths:
                p.write_bytes(b'preexisting identical file')
            result = verify(self.plan, bad, paths)
        self.assertFalse(result['gates']['save_succeeded'])
        self.assertFalse(result['gates']['container_roundtrips_byte_equal'])
        self.assertFalse(result['structural_checks_pass'])

    def test_missing_terminal_response_fails_closed(self):
        with self.assertRaises(ValueError):
            verify(self.plan, self.raw[:self.raw.rfind('STATE ')])

    def test_nonfinite_timings_are_rejected(self):
        a = copy.deepcopy(self.report['responses']['full'])
        a['lines'] = [s.replace('DONE 512 0 ', 'DONE 512 0 nan ') if s.startswith('DONE ') else s
                      for s in a['lines']]
        with self.assertRaises(ValueError):
            done(a)

    def test_original_first_decode_mismatch_remains_failed(self):
        plan = json.loads((EVIDENCE / 'first-review-plan.json').read_text(encoding='utf-8'))
        raw = (EVIDENCE / 'first-raw.log').read_bytes().decode('utf-8', 'replace')
        result = verify(plan, raw)
        self.assertFalse(result['registered_all_checks_pass'])
        self.assertTrue(result['gates']['actual_512_length'])
        self.assertFalse(result['gates']['seg512_equals_single_prefix'])


if __name__ == '__main__':
    unittest.main()
