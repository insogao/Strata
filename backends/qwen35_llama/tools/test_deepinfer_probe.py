"""Offline checks for the DI02 diagnostic probe; no engine, GPU or network."""
import copy
import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepinfer_diagnostic_probe as probe


def good_record(**overrides):
    rec = {
        "schema": "strata-decode-diag-v1", "event": "request_end",
        "req": 1, "serve_req": 1, "kind": "gen", "cont": False, "session": False,
        "keep": False, "prompt_tokens": 3896, "max_new": 1, "base_pos": 0,
        "produced": 1, "finish": "length", "prompt_ms": 10.0, "decode_ms": 1.0,
        "native_tier": True,
        "prefill": {"ran": True, "counters_available": True,
                    "counter_delta": {"prefill_layers": 1, "ms_ids": None}},
        "decode": {"ran": True, "counters_available": True,
                   "counter_delta": {"decode_layers": 1, "jobs": 1}},
        "cache": {"slots": 3200, "resident": 1, "bytes": 2, "payload_bytes": 1},
        "h2d": {"decode_ops": None, "available": False},
        "legacy_done": {"hits": None},
    }
    rec.update(overrides)
    return rec


class ProbeTests(unittest.TestCase):
    def test_clock_scope(self):
        before = time.monotonic()
        stamp = datetime.now(timezone.utc).isoformat()
        self.assertTrue(stamp.endswith("+00:00"))
        self.assertGreaterEqual(time.monotonic(), before)

    def test_parse_and_malformed(self):
        lines = ["PP 1 10", json.dumps(good_record()), "not json",
                 '{"schema":"strata-decode-diag-v1","event":', "DONE x"]
        records, malformed = probe.parse_diag_lines(lines)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["line"], 2)
        self.assertEqual(len(malformed), 1)
        self.assertEqual(probe.validate_records(records), [])

    def test_duplicate_and_order(self):
        records = [{"line": 1, "record": good_record(req=2)},
                   {"line": 2, "record": good_record(req=2)}]
        errors = probe.validate_records(records)
        self.assertTrue(any("duplicate" in e for e in errors))
        self.assertTrue(any("order" in e for e in errors))

    def test_missing_counter_is_not_zero(self):
        rec = good_record()
        rec["decode"] = {"ran": True, "counters_available": False}
        errors = probe.validate_decode_diag_record(rec)
        self.assertEqual(errors, [])
        rec["decode"] = {"ran": True, "counters_available": True}
        self.assertIn("decode.counter_delta", probe.validate_decode_diag_record(rec))

    def test_match(self):
        expected = {"kind": "gen", "prompt_tokens": 3896, "max_new": 1, "base_pos": 0,
                    "produced": 1, "finish": "length", "cont": False,
                    "session": False, "keep": False}
        self.assertEqual(probe.match_diag(good_record(), expected), [])
        self.assertTrue(probe.match_diag(good_record(max_new=2), expected))
        self.assertTrue(probe.match_diag(good_record(counters_available=True), expected) == [])

    def test_n_less_than_two_has_no_tps(self):
        row = {"n": 1, "token_observation_s": [10.0], "started_s": 9.0, "finished_s": 11.0}
        out = probe.timing_metrics(row)
        self.assertIsNone(out["decode_tps_observed"])
        self.assertIsNone(out["tpot_observed_ms"])
        self.assertEqual(out["itl_observed_ms"], [])
        self.assertAlmostEqual(out["ttft_observed_ms"], 1000.0)
        row2 = {"n": 3, "token_observation_s": [10.0, 10.1, 10.3],
                "started_s": 9.0, "finished_s": 11.0}
        out2 = probe.timing_metrics(row2)
        self.assertAlmostEqual(out2["decode_tps_observed"], 2 / 0.3)
        self.assertAlmostEqual(out2["tpot_observed_ms"], 150.0)
        self.assertEqual(len(out2["itl_observed_ms"]), 2)

    def test_context_margin(self):
        self.assertEqual(probe.context_margin(3817, 256, 4096), 23)
        self.assertEqual(probe.context_margin(3817, 257, 4096), 22)
        with self.assertRaises(ValueError):
            probe.context_margin(3817, 280, 4096)

    def test_graph_unknown_is_null(self):
        self.assertIsNone(probe.graph_evidence("no graph line")["reservation_observed"])
        found = probe.graph_evidence("sched_reserve: graph: nodes = 3049, splits = 82, "
                                     "input objects = 4, input tensors = 10")
        self.assertTrue(found["reservation_observed"])
        self.assertEqual(found["nodes"], 3049)
        self.assertIsNone(found["coverage"])

    def test_arm_env_only_adds_diag_for_on(self):
        base = {"engine": {"path": "x", "sha256": "0" * 64},
                "runtime": {"id": "x", "env": {"A": "1", "B": "2", "C": "3"}}}
        manifest = {"legacy_exe": "y", "legacy_exe_sha256": "0" * 64}
        off = probe.make_arm_spec(copy.deepcopy(base), {"engine": "new", "diag": False}, manifest)
        on = probe.make_arm_spec(copy.deepcopy(base), {"engine": "new", "diag": True}, manifest)
        self.assertEqual(set(off["runtime"]["env"]), {"A", "B", "C"})
        self.assertEqual(set(on["runtime"]["env"]), {"A", "B", "C", "STRATA_QWEN35_DECODE_DIAGNOSTICS"})
        self.assertEqual(on["runtime"]["env"]["STRATA_QWEN35_DECODE_DIAGNOSTICS"], "1")

    def test_telemetry_unknown_never_zero(self):
        telemetry = {"samples": [{"gpu": {"mem_used": 2 ** 30, "power": None},
                                  "available_ram_bytes": 4 * 2 ** 30, "processes": []}]}
        out = probe.telemetry_summary(telemetry)
        self.assertEqual(out["gpu_used_mib"]["max"], 1024.0)
        self.assertEqual(out["power_w"]["n"], 0)
        self.assertIsNone(out["power_w"]["max"])


BUDGETS = {"gpu_native_launches": 5,
           "max_internal_protocol_requests_per_native": 2,
           "max_internal_protocol_requests_total": 3}


def write_launch(native_dir, payload=None, ready=True):
    native_dir = Path(native_dir)
    (native_dir / "raw").mkdir(parents=True, exist_ok=True)
    result = {"commands": [{"argv": ["engine.exe"], "pid": 123, "started_at": "2026-10-21T00:00:00"}],
              "binary_sha256_actual": "b" * 64, "started_at": "2026-10-21T00:00:00"}
    if payload is not None:
        result.update(payload)
    (native_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (native_dir / "raw" / "child_00.log").write_text("READY 4096\n" if ready else "loading\n",
                                                     encoding="utf-8")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ledger = probe.NativeLedger(self.root / "ledger.json", BUDGETS, "2026-10-21T00:00:00+00:00")

    def tearDown(self):
        self.tmp.cleanup()

    def test_constructor_failure_after_launch_is_native_not_zero(self):
        native_dir = self.root / "arm" / "native"
        native_dir.mkdir(parents=True)
        write_launch(native_dir)
        record = self.ledger.start_arm(1, probe.ARM_ORDER[0], 5, native_dir)
        self.ledger.constructor_failed(record, native_dir, ValueError("commit floor"))
        self.assertEqual(self.ledger.totals["lease_attempts"], 1)
        self.assertEqual(self.ledger.totals["binary_launches"], 1)
        self.assertEqual(self.ledger.totals["model_ready"], 1)
        self.assertEqual(self.ledger.totals["successful_arms"], 0)
        self.assertEqual(self.ledger.totals["protocol_attempts"], 0)
        self.assertEqual(json.loads((self.root / "ledger.json").read_text())["totals"]["binary_launches"], 1)

    def test_constructor_failure_without_receipt_is_unknown_not_zero(self):
        native_dir = self.root / "arm" / "native"
        native_dir.mkdir(parents=True)
        record = self.ledger.start_arm(1, probe.ARM_ORDER[0], 5, native_dir)
        self.ledger.constructor_failed(record, native_dir, ValueError("lease refused"))
        self.assertEqual(self.ledger.totals["binary_launches"], 0)
        self.assertEqual(self.ledger.totals["binary_launch_unknown"], 1)

    def test_command_error_consumes_attempt_and_cap_refuses_pre_send(self):
        native_dir = self.root / "arm" / "native"
        native_dir.mkdir(parents=True)
        write_launch(native_dir)
        record = self.ledger.start_arm(1, probe.ARM_ORDER[0], 5, native_dir)
        self.ledger.arm_ready(record, None, native_dir)
        self.ledger.before_protocol(record, "P0", "GEN 1 1")
        self.ledger.protocol_failed(record, "P0", TimeoutError("send failed"))
        self.assertEqual(self.ledger.totals["protocol_attempts"], 1)
        self.assertEqual(self.ledger.totals["protocol_completed"], 0)
        self.ledger.before_protocol(record, "P1", "GEN 1 1")
        with self.assertRaises(ValueError):
            self.ledger.before_protocol(record, "P2", "GEN 1 1")
        self.assertEqual(self.ledger.totals["protocol_attempts"], 2)

    def test_total_protocol_budget_is_global(self):
        dirs = []
        for index in range(2):
            native_dir = self.root / ("arm%d" % index) / "native"
            native_dir.mkdir(parents=True)
            write_launch(native_dir)
            record = self.ledger.start_arm(index + 1, probe.ARM_ORDER[index], 5, native_dir)
            self.ledger.arm_ready(record, None, native_dir)
            dirs.append(record)
        self.ledger.before_protocol(dirs[0], "a", "STATE")
        self.ledger.before_protocol(dirs[0], "b", "STATE")
        self.ledger.before_protocol(dirs[1], "c", "STATE")
        with self.assertRaises(ValueError):
            self.ledger.before_protocol(dirs[1], "d", "STATE")
        self.assertEqual(self.ledger.totals["protocol_attempts"], 3)

    def test_native_budget_refuses_before_lease(self):
        for index in range(2):
            native_dir = self.root / ("arm%d" % index) / "native"
            native_dir.mkdir(parents=True)
            self.ledger.start_arm(index + 1, probe.ARM_ORDER[index % len(probe.ARM_ORDER)], 2, native_dir)
        with self.assertRaises(ValueError):
            self.ledger.start_arm(3, probe.ARM_ORDER[2], 2, self.root / "arm2" / "native")
        self.assertEqual(self.ledger.totals["lease_attempts"], 2)

    def test_duplicate_receipt_identity_is_not_double_counted(self):
        for name in ("a", "b"):
            native_dir = self.root / name / "native"
            native_dir.mkdir(parents=True)
            write_launch(native_dir)
        audit = probe.reconcile_receipts(self.root)
        self.assertEqual(audit["totals"]["receipts"], 2)
        self.assertEqual(audit["totals"]["unique_receipts"], 1)
        self.assertEqual(audit["totals"]["lease_attempts"], 1)
        self.assertEqual(audit["totals"]["binary_launches"], 1)
        self.assertIsNone(audit["totals"]["protocol_sent"])
        self.assertEqual(audit["totals"]["protocol_sent_unknown_receipts"], 1)

    def test_known_zero_protocol_stays_zero(self):
        native_dir = self.root / "arm" / "native"
        native_dir.mkdir(parents=True)
        write_launch(native_dir)
        (native_dir / "transcript.json").write_text(
            json.dumps({"commands": [], "responses": {}}), encoding="utf-8")
        audit = probe.reconcile_receipts(self.root)
        self.assertEqual(audit["totals"]["protocol_sent"], 0)
        self.assertEqual(audit["totals"]["protocol_completed"], 0)
        self.assertEqual(audit["totals"]["protocol_sent_unknown_receipts"], 0)
        self.assertEqual(audit["totals"]["protocol_completed_unknown_receipts"], 0)

    def test_missing_receipt_is_explicit_unknown(self):
        native_dir = self.root / "arm" / "native"
        (native_dir / "raw").mkdir(parents=True)
        (native_dir / "raw" / "child_00.log").write_text("READY 4096\n", encoding="utf-8")
        audit = probe.reconcile_receipts(self.root)
        self.assertEqual(audit["totals"]["receipts_missing"], 1)
        self.assertEqual(audit["totals"]["lease_attempts"], 0)
        self.assertEqual(audit["totals"]["binary_launch_unknown"], 1)
        self.assertIsNone(audit["totals"]["protocol_sent"])

    def test_generation_counts_keep_eos_and_max(self):
        answer = {"tokens": [7, 248046], "done": {"produced": 2, "finish": "stop"}}
        self.assertEqual(probe.validate_generation_counts(answer, 2), 2)
        with self.assertRaises(ValueError):
            probe.validate_generation_counts({"tokens": [7], "done": {"produced": 2,
                                                                     "finish": "length"}}, 2)
        with self.assertRaises(ValueError):
            probe.validate_generation_counts({"tokens": [7, 8], "done": {"produced": 2,
                                                                        "finish": "length"}}, 1)
        with self.assertRaises(ValueError):
            probe.validate_generation_counts({"tokens": [7], "done": {"produced": 1,
                                                                     "finish": "stop"}}, 2)

    def test_resource_refusal_creates_no_lease(self):
        spec = {"resources": {"estimated_host_mib": 14000, "max_gpu_used_mib": 8060,
                              "min_available_ram_mib": 1024, "max_host_rss_mib": 28000},
                "baseline": {"config": {}}}
        fp = {"model": {"size_bytes": {"value": 13676723168}, "sha256": {"value": "a" * 64}},
              "ram": {"commit_avail_mib": {"value": 500.0}}, "gpu": {}}
        monitor = type("M", (), {"violation": None, "sample": lambda self: {
            "gpu_used_mib": 1.0, "gpu_total_mib": 8187.0, "available_ram_mib": 4096.0,
            "commit_available_mib": 500.0}})()
        guard = probe.preflight_resource_guard(
            spec, {"model_size": 13676723168},
            fingerprint_fn=lambda _: fp,
            admission_fn=lambda *_: {"admitted": True, "reasons": []},
            monitor_factory=lambda _: monitor)
        self.assertFalse(guard["admitted"])
        with self.assertRaises(probe.ResourceRefused):
            probe.require_admitted(guard)
        self.assertEqual(self.ledger.totals["lease_attempts"], 0)

    def test_max_arms_is_validated_not_sliced(self):
        manifest = {"budgets": BUDGETS, "arm_order": [a["id"] for a in probe.ARM_ORDER]}
        probe.validate_arm_options(manifest, 5)
        for value in (0, -1, 6):
            with self.assertRaises(ValueError):
                probe.validate_arm_options(manifest, value)
        bad = copy.deepcopy(manifest)
        bad["arm_order"] = list(reversed(bad["arm_order"]))
        with self.assertRaises(ValueError):
            probe.validate_arm_options(bad, 1)


if __name__ == "__main__":
    unittest.main()
