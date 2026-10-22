import copy
import json
import unittest

import deepinfer_emitter_regression as emitter
from strata_diagnostics import (
    counter_delta,
    decode_diag_gate,
    validate_decode_diag_record,
)


def valid_record():
    return {
        "schema": "strata-decode-diag-v1",
        "event": "request_end",
        "req": 2,
        "serve_req": 2,
        "kind": "gen",
        "cont": False,
        "session": False,
        "keep": False,
        "prompt_tokens": 3817,
        "max_new": 128,
        "base_pos": 0,
        "produced": 129,
        "finish": "stop",
        "prompt_ms": 2100.5,
        "decode_ms": 3200.25,
        "native_tier": True,
        "counter_source": "Qwen35BridgeStats cumulative snapshot delta at serve request boundaries",
        "scope": "prefill and decode are separate deltas",
        "prefill": {
            "ran": True,
            "counters_available": True,
            "counter_delta": {
                "prefill_layers": 40,
                "prefill_entries": 256,
                "decode_hits": 0,
                "prefill_stream_bytes": 1048576,
                "prefill_gpu_ms": 1800.0,
                "hit_overlap": 0,
                "ms_ids": 10.5,
            },
        },
        "decode": {
            "ran": True,
            "counters_available": True,
            "counter_delta": {
                "decode_layers": 40 * 129,
                "decode_hits": 3000,
                "decode_misses": 900,
                "jobs": 40 * 129,
                "hit_overlap": 4000,
                "ms_ids": 120.25,
                "ms_launch": 80.0,
                "ms_misses": 900.0,
                "ms_combine": 300.0,
                "expert_seconds": 1.25,
            },
        },
        "cache": {"slots": 3200, "resident": 3200, "bytes": 123, "payload_bytes": 100},
        "h2d": {"decode_ops": None, "decode_bytes": None, "available": False, "reason": "unavailable"},
        "legacy_done": {
            "hits": None, "lookups": None, "accepted": None, "offered": None,
            "reason": "placeholders",
        },
    }


class GateTests(unittest.TestCase):
    def test_exact_gate(self):
        self.assertTrue(decode_diag_gate("1"))
        for value in (None, "0", "true", "01", " 1"):
            self.assertFalse(decode_diag_gate(value))


class DeltaTests(unittest.TestCase):
    def test_prefill_decode_are_separate_deltas(self):
        base = {"hits": 10, "layers": 100}
        pre = {"hits": 12, "layers": 120}
        dec = {"hits": 17, "layers": 300}
        self.assertEqual(counter_delta(pre["hits"], base["hits"]), 2)
        self.assertEqual(counter_delta(pre["layers"], base["layers"]), 20)
        self.assertEqual(counter_delta(dec["hits"], pre["hits"]), 5)
        self.assertEqual(counter_delta(dec["layers"], pre["layers"]), 180)
        self.assertNotEqual(counter_delta(dec["hits"], base["hits"]), 5)
        self.assertEqual(counter_delta(dec["hits"], base["hits"]), 7)

    def test_continuation_baseline_is_not_carried(self):
        request1_end = 500
        request2_start = 500
        request2_end = 512
        self.assertEqual(counter_delta(request2_end, request2_start), 12)
        self.assertEqual(counter_delta(request2_end, request1_end), 12)

    def test_unavailable_and_reset_are_none(self):
        self.assertIsNone(counter_delta(None, 5))
        self.assertIsNone(counter_delta(5, None))
        self.assertIsNone(counter_delta(4, 5))
        self.assertIsNone(counter_delta(True, 0))


class SchemaTests(unittest.TestCase):
    def test_valid_record(self):
        self.assertEqual(validate_decode_diag_record(valid_record()), [])

    def test_json_roundtrip(self):
        rec = json.loads(json.dumps(valid_record()))
        self.assertEqual(validate_decode_diag_record(rec), [])

    def test_negative_nonfinite_and_placeholder_zero_are_rejected(self):
        rec = valid_record()
        rec["decode"]["counter_delta"]["ms_ids"] = -1
        self.assertTrue(validate_decode_diag_record(rec))
        rec = valid_record()
        rec["decode"]["counter_delta"]["ms_launch"] = float("nan")
        self.assertTrue(validate_decode_diag_record(rec))
        rec = valid_record()
        rec["legacy_done"]["hits"] = 0
        self.assertTrue(validate_decode_diag_record(rec))
        rec = valid_record()
        rec["h2d"]["decode_ops"] = 0
        self.assertTrue(validate_decode_diag_record(rec))

    def test_null_hit_probe_counters_are_unavailable_not_zero(self):
        rec = valid_record()
        for key in ("hit_overlap", "hit_late", "hit_inflight"):
            rec["decode"]["counter_delta"][key] = None
        self.assertEqual(validate_decode_diag_record(rec), [])
        rec["decode"]["counter_delta"]["hit_overlap"] = -1
        self.assertTrue(validate_decode_diag_record(rec))

    def test_unavailable_phase_must_not_carry_counters(self):
        rec = valid_record()
        rec["decode"] = {"ran": False, "counters_available": False}
        self.assertEqual(validate_decode_diag_record(rec), [])
        rec["decode"]["counter_delta"] = {"decode_hits": 0}
        self.assertTrue(validate_decode_diag_record(rec))

    def test_score_has_no_decode_delta(self):
        rec = valid_record()
        rec["kind"] = "score"
        rec["finish"] = "score"
        rec["produced"] = 0
        rec["decode"] = {"ran": False, "counters_available": False}
        self.assertEqual(validate_decode_diag_record(rec), [])


class EmitterExtractionTests(unittest.TestCase):
    def test_extract_exact_marker_block(self):
        text = "\n".join([
            "before",
            emitter.START_MARKER + " synthetic",
            "bool decode_diag_enabled() { return false; }",
            "struct RequestDiag {",
            "    ~RequestDiag() { emit(); }",
            "};",
            "after",
        ])
        found = emitter.extract_request_diag(text)
        self.assertEqual(found["start_line"], 2)
        self.assertEqual(found["end_line"], 6)
        self.assertIn("struct RequestDiag", found["block"])
        self.assertNotIn("after", found["block"])
        self.assertEqual(len(found["sha256"]), 64)

    def test_missing_and_duplicate_markers_rejected(self):
        with self.assertRaises(ValueError):
            emitter.extract_request_diag("struct RequestDiag {}")
        duplicated = "\n".join([emitter.START_MARKER, emitter.START_MARKER,
                                "bool decode_diag_enabled() {}",
                                "struct RequestDiag {", emitter.END_MARKER, "};"])
        with self.assertRaises(ValueError):
            emitter.extract_request_diag(duplicated)


class EmitterCaseTests(unittest.TestCase):
    def record(self, **changes):
        rec = {"schema": "strata-decode-diag-v1", "event": "request_end", "kind": "gen",
               "cont": False, "produced": 16, "max_new": 16, "finish": "length",
               "native_tier": True,
               "prefill": {"ran": True, "counters_available": True,
                           "counter_delta": {"prefill_layers": 40,
                                             "prefill_stream_bytes": 1048576}},
               "decode": {"ran": True, "counters_available": True,
                          "counter_delta": {"decode_layers": 5160, "jobs": 5160,
                                            "ms_ids": 109.75}},
               "cache": {"slots": 3200, "resident": 3200, "bytes": 123,
                         "payload_bytes": 100}}
        rec.update(changes)
        return rec

    def row(self, rec):
        return {"case": "gen", "exit": 0, "stderr": json.dumps(rec) + "\n", "stdout": ""}

    def test_valid_gen_and_off(self):
        self.assertEqual(emitter.validate_case("gen", self.row(self.record())), [])
        self.assertEqual(emitter.validate_case("off", {"case": "off", "exit": 0,
                                                       "stderr": "", "stdout": ""}), [])
        self.assertTrue(emitter.validate_case("off", {"case": "off", "exit": 0,
                                                      "stderr": "{}", "stdout": ""}))

    def test_produced_above_max_is_rejected(self):
        bad = self.record(produced=17)
        self.assertTrue(emitter.validate_case("gen", self.row(bad)))
        bad = self.record(produced=-1)
        self.assertTrue(emitter.validate_case("gen", self.row(bad)))

    def test_score_and_unavailable_shapes(self):
        score = self.record(kind="score", produced=0, max_new=0, finish="score",
                            decode={"ran": False, "counters_available": False})
        self.assertEqual(emitter.validate_case("score", self.row(score)), [])
        unavailable = self.record(prefill={"ran": False, "counters_available": False},
                                  decode={"ran": False, "counters_available": False},
                                  cache={"slots": None, "resident": None,
                                         "bytes": None, "payload_bytes": None})
        self.assertEqual(emitter.validate_case("unavailable", self.row(unavailable)), [])

    def test_cont_flag_must_be_true(self):
        bad = self.record(cont=False)
        self.assertTrue(emitter.validate_case("cont", self.row(bad)))


if __name__ == "__main__":
    unittest.main()
