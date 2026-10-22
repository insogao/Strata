"""Offline checks for the DI02 diagnostic probe; no engine, GPU or network."""
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepinfer_diagnostic_probe as probe

RECORDED_ROOT = Path(os.environ.get(
    "STRATA_DI02_RECORDED_ROOT", r"D:\work\Strata魔改\results\deepinfer-20261021"))
RECORDED_DATASETS = (
    ("di02-l256-retest/probe-run/arm-01-legacy-ea06-OFF", "legacy-ea06-OFF"),
    ("di02-four-arm-qualification/probe-run/arm-01-new-OFF-A1", "new-OFF-A1"),
)
MODEL_PATH = Path(os.environ.get(
    "STRATA_DI02_MODEL_PATH",
    r"D:\work\StrataUnified\models\Qwen3.6-35B-A3B-UD-IQ3_S.gguf"))


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

    def test_skip_probe_arm_registry_and_marker(self):
        base = {"engine": {"path": "x", "sha256": "0" * 64},
                "runtime": {"id": "x", "env": dict(probe.REGISTERED_BASE_ENV)}}
        manifest = {"legacy_exe": "y", "legacy_exe_sha256": "0" * 64}
        order = probe.select_arm_order({"experiment_kind": probe.SKIP_PROBE_KIND})
        self.assertEqual([arm["id"] for arm in order], list(probe.SKIP_PROBE_IDS))
        smoke = probe.select_arm_order({"experiment_kind": probe.SKIP_PROBE_KIND,
                                        "arm_order": list(probe.SKIP_PROBE_SMOKE_IDS)})
        self.assertEqual([arm["id"] for arm in smoke], list(probe.SKIP_PROBE_SMOKE_IDS))
        off_arm, on_arm = order[2], order[3]
        off = probe.make_arm_spec(copy.deepcopy(base), off_arm, manifest)
        on = probe.make_arm_spec(copy.deepcopy(base), on_arm, manifest)
        self.assertEqual(off["runtime"]["env"], dict(probe.REGISTERED_BASE_ENV))
        self.assertEqual(on["runtime"]["env"],
                         dict(probe.REGISTERED_BASE_ENV,
                              STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE="1"))
        self.assertEqual(probe.validate_feature_marker(off_arm, "plain"), 0)
        one = "x " + probe.SKIP_PROBE_MARKER + "\n"
        self.assertEqual(probe.validate_feature_marker(on_arm, one), 1)
        with self.assertRaises(ValueError):
            probe.validate_feature_marker(on_arm, one + probe.SKIP_PROBE_MARKER + "\n")
        with self.assertRaises(ValueError):
            probe.validate_feature_marker(off_arm, one)
        with self.assertRaises(ValueError):
            probe.validate_arm_definition({"id": "x", "engine": "new", "diag": False,
                                           "cofetch": False, "skip_probe": True})

    def test_threads_arm_registry_and_exact_config(self):
        base = {"engine": {"path": "x", "sha256": "0" * 64},
                "runtime": {"id": "x", "env": dict(probe.REGISTERED_BASE_ENV)},
                "baseline": {"config": {"slots": 3200, "threads": 8, "ubatch": 4096,
                                        "ctx": 4096, "compact": False}}}
        manifest = {"legacy_exe": "y", "legacy_exe_sha256": "0" * 64}
        order = probe.select_arm_order({"experiment_kind": probe.THREADS_KIND})
        self.assertEqual([arm["id"] for arm in order], list(probe.THREADS_IDS))
        off = probe.make_arm_spec(copy.deepcopy(base), order[2], manifest)
        on = probe.make_arm_spec(copy.deepcopy(base), order[3], manifest)
        smoke = probe.make_arm_spec(copy.deepcopy(base), order[1], manifest)
        self.assertEqual(off["baseline"]["config"]["threads"], 8)
        self.assertEqual(on["baseline"]["config"]["threads"], 10)
        self.assertEqual(off["runtime"]["env"], dict(probe.REGISTERED_BASE_ENV))
        self.assertEqual(on["runtime"]["env"], dict(probe.REGISTERED_BASE_ENV))
        self.assertEqual(smoke["runtime"]["env"],
                         dict(probe.REGISTERED_BASE_ENV,
                              STRATA_QWEN35_DECODE_DIAGNOSTICS="1"))
        self.assertEqual(smoke["baseline"]["config"]["threads"], 10)
        for bad in ({"id": "x", "engine": "new", "diag": False, "threads": 9, "phase": "formal"},
                    {"id": "x", "engine": "new", "diag": False, "threads": 10,
                     "skip_probe": True, "phase": "formal"}):
            with self.assertRaises(ValueError):
                probe.validate_arm_definition(bad)

    def test_telemetry_unknown_never_zero(self):
        telemetry = {"samples": [{"gpu": {"mem_used": 2 ** 30, "power": None},
                                  "available_ram_bytes": 4 * 2 ** 30, "processes": []}]}
        out = probe.telemetry_summary(telemetry)
        self.assertEqual(out["gpu_used_mib"]["max"], 1024.0)
        self.assertEqual(out["power_w"]["n"], 0)
        self.assertIsNone(out["power_w"]["max"])

    def test_close_artifacts_read_native_resources_not_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "arm-01-new-OFF-A1"
            native = out / "native"
            native.mkdir(parents=True)
            (out / "telemetry.json").write_text(json.dumps({"samples": [{"gpu": {}}]}),
                                                encoding="utf-8")
            (native / "resources.json").write_text(
                json.dumps({"violation": None, "samples": 1}), encoding="utf-8")
            telemetry, resources = probe.close_artifacts(out, native)
            self.assertEqual(len(telemetry["samples"]), 1)
            self.assertIsNone(resources["violation"])
            self.assertFalse((out / "resources.json").exists())


class PromptPlanTests(unittest.TestCase):
    def build(self, ids):
        spec = {"shapes": [{"id": key, "tokens_file": key, "prompt_tokens": len(value)}
                           for key, value in ids.items()]}
        with mock.patch.object(probe.ex, "token_ids", side_effect=lambda name: ids[name]):
            return probe.build_prompts(spec)

    def test_old_missing_key_now_uses_same_L_fixture(self):
        prompts = self.build({"P": [1], "S": [2], "L": [3, 4]})
        self.assertIs(prompts["L256"], prompts["L"])
        self.assertEqual(prompts["L256"]["ids"], [3, 4])
        self.assertEqual(prompts["L256"]["csv"], "3,4")
        probe.validate_prompt_plan(prompts)

    def test_old_map_without_L256_fails_before_native(self):
        prompts = {"P": {"ids": [1], "csv": "1"}, "S": {"ids": [2], "csv": "2"},
                   "L": {"ids": [3, 4], "csv": "3,4"}}
        with self.assertRaises(KeyError):
            prompts["L256"]
        with self.assertRaisesRegex(ValueError, "missing shape L256"):
            probe.validate_prompt_plan(prompts)

    def test_all_planned_request_keys_resolve(self):
        prompts = self.build({"P": [1], "S": [2], "L": [3, 4]})
        probe.validate_prompt_plan(prompts)
        planned = dict(probe.PLANNED_PROMPTS)
        for name in ("resetA", "SCORE0", "resetB", "SCORE1", "score-held",
                     "gen-held", "reset-eos", "score-eos", "eos"):
            self.assertIn(name, planned)
            self.assertIn(planned[name], prompts)

    def test_missing_or_empty_L_fails_before_native(self):
        spec = {"shapes": [{"id": "P", "tokens_file": "P", "prompt_tokens": 1},
                           {"id": "S", "tokens_file": "S", "prompt_tokens": 1}]}
        with mock.patch.object(probe.ex, "token_ids", side_effect=lambda name: [7]):
            with self.assertRaisesRegex(ValueError, "missing L fixture"):
                probe.build_prompts(spec)
        prompts = self.build({"P": [1], "S": [2], "L": [3, 4]})
        prompts["L"] = {"ids": [], "csv": ""}
        prompts["L256"] = prompts["L"]
        with self.assertRaisesRegex(ValueError, "empty shape L"):
            probe.validate_prompt_plan(prompts)


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
        with self.assertRaises(ValueError):
            probe.validate_arm_options(manifest, 4)

    def test_remaining_four_arm_order_is_selected(self):
        manifest = {"budgets": dict(BUDGETS, gpu_native_launches=4),
                    "arm_order": list(probe.REMAINING_ARM_IDS)}
        order, budgets = probe.validate_arm_options(manifest, 4)
        self.assertEqual([a["id"] for a in order], list(probe.REMAINING_ARM_IDS))
        self.assertEqual(budgets["gpu_native_launches"], 4)
        with self.assertRaises(ValueError):
            probe.validate_arm_options(manifest, 5)

    def test_remaining_three_arm_order_is_selected(self):
        manifest = {"budgets": dict(BUDGETS, gpu_native_launches=3),
                    "arm_order": list(probe.REMAINING3_ARM_IDS)}
        order, budgets = probe.validate_arm_options(manifest, 3)
        self.assertEqual([a["id"] for a in order], list(probe.REMAINING3_ARM_IDS))
        self.assertEqual(budgets["gpu_native_launches"], 3)
        with self.assertRaises(ValueError):
            probe.validate_arm_options(manifest, 4)

    def test_full_arm_order_is_default_and_explicit(self):
        for manifest in ({"budgets": BUDGETS},
                         {"budgets": BUDGETS,
                          "arm_order": [a["id"] for a in probe.ARM_ORDER]}):
            order, _ = probe.validate_arm_options(manifest, 5)
            self.assertEqual([a["id"] for a in order], [a["id"] for a in probe.ARM_ORDER])

    def test_partial_or_reordered_arm_order_is_rejected(self):
        bad_orders = (
            ["legacy-ea06-OFF", "new-ON-B1", "new-ON-B2", "new-OFF-A2"],
            ["new-OFF-A1", "new-ON-B2", "new-ON-B1", "new-OFF-A2"],
            ["new-OFF-A1", "new-ON-B1", "new-OFF-A2"],
            ["new-ON-B2", "new-ON-B1", "new-OFF-A2"],
            ["new-ON-B1", "new-ON-B2"],
            ["new-ON-B1", "new-ON-B9", "new-OFF-A2"],
        )
        for order in bad_orders:
            manifest = {"budgets": BUDGETS, "arm_order": order}
            with self.assertRaises(ValueError):
                probe.validate_arm_options(manifest, len(order))


def token_sha(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


class FinalizerTests(unittest.TestCase):
    ARM = {"id": "new-OFF-A1", "engine": "new", "diag": False}
    SPEC = {"engine": {"sha256": "a" * 64}, "runtime": {"env": {}}, "baseline": {"config": {}}}

    def test_absent_optional_diag_fields_become_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "derived"
            native = out / "native"
            (native / "raw").mkdir(parents=True)
            (out / "telemetry.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
            (native / "resources.json").write_text(
                json.dumps({"violation": None, "samples": 0}), encoding="utf-8")
            (native / "raw/child_00.log").write_text("READY 4096\n", encoding="utf-8")
            rows = [
                {"name": "P0", "command": "GEN 1 1", "kind": "gen", "n": 1,
                 "finish": "length", "prompt_tokens": 1, "prompt_read": 1, "reused": 0,
                 "done_prompt_ms": 1.0, "done_decode_ms": 1.0,
                 "started_s": 1.0, "finished_s": 2.0, "token_ids": [1],
                 "token_ids_sha256": token_sha([1]), "token_observation_s": [2.0]},
                {"name": "state", "command": "STATE", "kind": None, "n": 0,
                 "finish": "state", "started_s": 2.0, "finished_s": 2.1,
                 "token_ids": [], "token_ids_sha256": token_sha([]),
                 "token_observation_s": [], "state": {"pos": 1}},
            ]
            result = probe.finalize_arm(self.ARM, out, self.SPEC, rows, {},
                                        native_exit=0)
            self.assertEqual(result["summary"]["diag_overhead_after_done_ms"]["n"], 0)
            self.assertEqual(result["summary"]["diag_count"], 0)
            for row in result["requests"]:
                self.assertIn("diag_after_done_ms", row)
                self.assertIsNone(row["diag_after_done_ms"])
                self.assertIsNone(row["diag_index"])
                self.assertIsNone(row["diag"])
            bad = copy.deepcopy(rows)
            del bad[0]["token_ids_sha256"]
            with self.assertRaises(ValueError):
                probe.finalize_arm(self.ARM, out, self.SPEC, bad, {}, native_exit=0)

    def test_on_fixture_keeps_connected_optional_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "derived"
            native = out / "native"
            (native / "raw").mkdir(parents=True)
            (out / "telemetry.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
            (native / "resources.json").write_text(
                json.dumps({"violation": None, "samples": 0}), encoding="utf-8")
            record = good_record()
            (native / "raw/child_00.log").write_text(
                "READY 4096\n" + json.dumps(record) + "\n", encoding="utf-8")
            rows = [{"name": "P0", "command": "GEN 1 1", "kind": "gen", "n": 1,
                     "finish": "length", "prompt_tokens": 1, "prompt_read": 1, "reused": 0,
                     "done_prompt_ms": 1.0, "done_decode_ms": 1.0,
                     "started_s": 1.0, "finished_s": 2.0, "token_ids": [1],
                     "token_ids_sha256": token_sha([1]), "token_observation_s": [2.0],
                     "diag": record, "diag_index": 0, "diag_observed_s": 2.2,
                     "send_to_diag_ms": 1200.0, "diag_after_done_ms": 200.0}]
            result = probe.finalize_arm({"id": "new-ON-B1", "engine": "new", "diag": True},
                                        out, self.SPEC, rows, {}, native_exit=0)
            self.assertEqual(result["summary"]["diag_count"], 1)
            self.assertEqual(result["summary"]["diag_overhead_after_done_ms"]["n"], 1)
            self.assertEqual(result["summary"]["diag_overhead_after_done_ms"]["median"], 200.0)
            self.assertEqual(result["requests"][0]["diag"], record)


class RecordedOfflineTests(unittest.TestCase):
    def test_actual_19op_recordings_finalize_postprocess_compare(self):
        env_dir = os.environ.get("STRATA_DI02_DERIVED_DIR")
        tmp = None
        if env_dir:
            derived_root = Path(env_dir)
        else:
            tmp = tempfile.TemporaryDirectory()
            derived_root = Path(tmp.name)
        try:
            self.assertTrue(MODEL_PATH.is_file(), "recorded offline tokenizer model missing")
            results, provenance = [], {}
            with mock.patch.object(probe, "LeasedNative",
                                   side_effect=AssertionError("offline path created a native")), \
                    mock.patch.object(probe, "ResourceMonitor",
                                      side_effect=AssertionError("offline path sampled GPU")):
                for rel, arm_id in RECORDED_DATASETS:
                    dataset = RECORDED_ROOT / rel
                    self.assertTrue(dataset.is_dir(), "recorded dataset missing: " + str(dataset))
                    result = probe.finalize_recorded_arm(dataset, arm_id,
                                                         derived_root / arm_id)
                    results.append(result)
                    provenance[arm_id] = result["summary"]["offline_finalization"]
            for result in results:
                summary = result["summary"]
                rows = result["requests"]
                by_name = {row["name"]: row for row in rows}
                self.assertEqual(len(rows), 19)
                self.assertEqual(summary["request_count"], 19)
                self.assertEqual(summary["native_exit"], 0)
                self.assertIsNone(summary["load_ready_s"])
                self.assertIsNone(summary["load_to_first_done_s"])
                self.assertEqual(summary["diag_count"], 0)
                self.assertEqual(summary["diag_overhead_after_done_ms"]["n"], 0)
                for row in rows:
                    self.assertIsNone(row["diag"])
                    self.assertIn("diag_after_done_ms", row)
                    self.assertIsNone(row["diag_after_done_ms"])
                self.assertIsNone(by_name["P0"]["decode_tps_observed"])
                self.assertIsNone(by_name["P1"]["decode_tps_observed"])
                self.assertIsNone(by_name["P0"]["tpot_observed_ms"])
                self.assertEqual(by_name["eos"]["n"], 17)
                self.assertEqual(by_name["eos"]["finish"], "stop")
                self.assertEqual(by_name["eos"]["token_ids"][-1], 248046)
                self.assertEqual(by_name["state-held"]["state"]["pos"], 3817 + 128)
                self.assertEqual(by_name["state-eos"]["state"]["pos"], 3896 + 17)
                self.assertEqual(summary["offline_finalization"]["responses_source"],
                                 "native/transcript.json#responses")
                self.assertIsNotNone(
                    summary["offline_finalization"]["recorded_files"]["requests.json"])
            results = probe.postprocess(results, MODEL_PATH)
            for result in results:
                self.assertEqual(set(result["score_facts"]), {"SCORE0", "SCORE1"})
                for fact in result["score_facts"].values():
                    self.assertTrue(all(math.isfinite(item["logprob"])
                                        for item in fact["candidate_logprobs"]))
                    self.assertTrue(0 <= fact["label_mass"] <= 1.0001)
                s0 = next(row for row in result["requests"] if row["name"] == "S0")
                self.assertIsInstance(s0["json"]["text"], str)
            comparison = probe.compare(results)
            self.assertEqual(comparison["classes"]["legacy-OFF"], ["legacy-ea06-OFF"])
            self.assertEqual(comparison["classes"]["new-OFF"], ["new-OFF-A1"])
            self.assertEqual(comparison["shape_gen"]["new-OFF"]["L256"]["n"], [256, 256])
            self.assertTrue(comparison["independent_score_rank"]["SCORE0"]["consistent"])
            self.assertTrue(comparison["independent_score_rank"]["SCORE1"]["consistent"])
            self.assertIsNone(comparison["new_on_batch_first_send_to_last_diag_median_ms"])
            (derived_root / "offline-provenance.json").write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
            (derived_root / "offline-comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        finally:
            if tmp is not None:
                tmp.cleanup()


COFETCH_BUDGETS = {"gpu_native_launches": 6,
                   "max_internal_protocol_requests_per_native": 19,
                   "max_internal_protocol_requests_total": 114}
COFETCH_BASE = {"engine": {"path": "x", "sha256": "0" * 64},
                "runtime": {"id": "x", "env": dict(probe.REGISTERED_BASE_ENV)}}
COFETCH_MANIFEST = {"experiment_kind": "id-act-cofetch", "budgets": COFETCH_BUDGETS,
                    "legacy_exe": "y", "legacy_exe_sha256": "0" * 64}


def cofetch_arm(arm_id):
    return next(arm for arm in probe.COFETCH_ARM_ORDER if arm["id"] == arm_id)


def cofetch_env(arm):
    env = dict(probe.REGISTERED_BASE_ENV)
    if arm["diag"]:
        env[probe.DIAG_ENV] = "1"
    if arm["cofetch"]:
        env[probe.COFETCH_ENV] = "1"
    return env


def cofetch_rows(scale=1.0, eos_stop=True, held_pos_delta=0, json_complete=True):
    def gen(name, max_new, prompt_read, reused, n, finish, tokens, base_ms):
        return {"name": name, "kind": "gen", "n": n, "finish": finish,
                "requested_max_new": max_new, "prompt_read": prompt_read, "reused": reused,
                "token_ids": tokens, "token_ids_sha256": token_sha(tokens),
                "e2e_observed_ms": base_ms * scale,
                "ttft_observed_ms": base_ms * scale / 10.0,
                "prefill_tps_from_done": 100.0 / scale,
                "decode_tps_observed": (n - 1) / (base_ms * scale / 1000.0) if n > 1 else None,
                "tpot_observed_ms": base_ms * scale / (n - 1) if n > 1 else None,
                "send_to_done_ms": base_ms * scale}

    def score(name, prompt_read):
        return {"name": name, "kind": "score", "n": 0, "finish": "score",
                "requested_max_new": 0, "prompt_read": prompt_read, "reused": 0,
                "token_ids": [], "token_ids_sha256": token_sha([]),
                "e2e_observed_ms": 100.0 * scale, "ttft_observed_ms": None,
                "prefill_tps_from_done": 100.0 / scale, "decode_tps_observed": None,
                "tpot_observed_ms": None, "send_to_done_ms": 100.0 * scale}

    rows = [
        gen("P0", 1, 3896, 0, 1, "length", [7], 100.0),
        gen("P1", 1, 3896, 0, 1, "length", [7], 100.0),
        gen("S0", 16, 3896, 0, 16, "length", list(range(16)), 200.0),
        gen("S1", 16, 3896, 0, 16, "length", list(range(16)), 200.0),
        gen("L0", 128, 3817, 0, 128, "length", list(range(128)), 500.0),
        gen("L1", 128, 3817, 0, 128, "length", list(range(128)), 500.0),
        gen("L2560", 256, 3817, 0, 256, "length", list(range(256)), 900.0),
        gen("L2561", 256, 3817, 0, 256, "length", list(range(256)), 900.0),
        score("resetA", 1),
        score("SCORE0", 3896),
        score("resetB", 1),
        score("SCORE1", 3896),
        score("score-held", 3817),
        gen("gen-held", 128, 0, 3817, 128, "length", list(range(128)), 500.0),
    ]
    rows.append({"name": "state-held", "kind": None, "n": 0, "finish": "state",
                 "token_ids": [], "token_ids_sha256": token_sha([]),
                 "state": {"pos": 3817 + 128 + held_pos_delta}})
    rows += [score("reset-eos", 1), score("score-eos", 3896)]
    eos_tokens = list(range(16)) + [248046] if eos_stop else list(range(17))
    rows.append(gen("eos", 32, 0, 3896, len(eos_tokens),
                    "stop" if eos_stop else "length", eos_tokens, 80.0))
    rows.append({"name": "state-eos", "kind": None, "n": 0, "finish": "state",
                 "token_ids": [], "token_ids_sha256": token_sha([]),
                 "state": {"pos": 3896 + len(eos_tokens)}})
    for row in rows:
        if row["name"] in ("S0", "S1"):
            row["json"] = {"text": "{}", "json_complete": json_complete,
                           "json_visible": json_complete}
    return rows


def cofetch_fact(ranking=(15, 16, 17, 18), nan=False):
    logprob = {15: -1.0, 16: -2.0, 17: -3.0, 18: -4.0}
    for index, candidate in enumerate(ranking):
        logprob[candidate] = -1.0 - index
    if nan:
        logprob[15] = float("nan")
    return {"candidate_logprobs": [{"id": candidate, "logprob": logprob[candidate]}
                                   for candidate in (15, 16, 17, 18)],
            "label_mass": 0.5, "choice": ranking[0], "calibrated": False}


def cofetch_result(arm_id, scale=1.0, ranking=(15, 16, 17, 18), nan=False,
                   marker=None, eos_stop=True, held_pos_delta=0, json_complete=True,
                   gpu_peak=7500.0):
    arm = copy.deepcopy(cofetch_arm(arm_id))
    if marker is None:
        marker = 1 if arm["cofetch"] else 0
    summary = {"arm": arm, "runtime_env": cofetch_env(arm),
               "arm_features": {"diag": arm["diag"], "cofetch": arm["cofetch"],
                                "phase": arm["phase"]},
               "cofetch_marker_count": marker,
               "resources": {"violation": None, "gpu_peak_mib": gpu_peak,
                             "host_rss_peak_mib": 15000.0,
                             "available_ram_min_mib": 4000.0,
                             "samples": [{"commit_available_mib": 12000.0}]},
               "diag_overhead_after_done_ms": {"n": 0, "min": None, "median": None,
                                               "max": None}}
    return {"summary": summary,
            "requests": cofetch_rows(scale=scale, eos_stop=eos_stop,
                                     held_pos_delta=held_pos_delta,
                                     json_complete=json_complete),
            "score_facts": {"SCORE0": cofetch_fact(ranking, nan),
                            "SCORE1": cofetch_fact(ranking, nan)}}


class CofetchConfigurationTests(unittest.TestCase):
    def test_exact_six_arm_mapping_and_env(self):
        order, budgets = probe.validate_arm_options(COFETCH_MANIFEST, 6)
        self.assertEqual([arm["id"] for arm in order], list(probe.COFETCH_IDS))
        self.assertEqual(budgets["gpu_native_launches"], 6)
        for arm in order:
            spec = probe.make_arm_spec(copy.deepcopy(COFETCH_BASE), arm, COFETCH_MANIFEST)
            env = spec["runtime"]["env"]
            if arm["cofetch"]:
                self.assertEqual(env.get(probe.COFETCH_ENV), "1")
            else:
                self.assertNotIn(probe.COFETCH_ENV, env)
            if arm["diag"]:
                self.assertEqual(env.get(probe.DIAG_ENV), "1")
            else:
                self.assertNotIn(probe.DIAG_ENV, env)
            self.assertEqual(set(env) - set(probe.REGISTERED_BASE_ENV),
                             ({probe.DIAG_ENV} if arm["diag"] else set())
                             | ({probe.COFETCH_ENV} if arm["cofetch"] else set()))

    def test_exact_subset_orders_are_explicit(self):
        for ids, budget in ((probe.COFETCH_SMOKE_IDS, 2), (probe.COFETCH_FORMAL_IDS, 4)):
            manifest = {"experiment_kind": "id-act-cofetch",
                        "budgets": dict(COFETCH_BUDGETS, gpu_native_launches=budget),
                        "arm_order": list(ids)}
            order, _ = probe.validate_arm_options(manifest, budget)
            self.assertEqual([arm["id"] for arm in order], list(ids))

    def test_failclosed_kind_order_and_flags(self):
        with self.assertRaisesRegex(ValueError, "unknown experiment_kind"):
            probe.select_arm_order({"experiment_kind": "other"})
        bad_orders = (
            list(reversed(probe.COFETCH_IDS)),
            list(probe.COFETCH_IDS[:3]),
            ["cofetch-OFF-A1", "cofetch-ON-B1", "cofetch-ON-B2", "cofetch-OFF-A2",
             "cofetch-OFF-smoke", "cofetch-ON-smoke"],
        )
        for order in bad_orders:
            with self.assertRaises(ValueError):
                probe.select_arm_order({"experiment_kind": "id-act-cofetch",
                                        "arm_order": order})
        extra = copy.deepcopy(COFETCH_BASE)
        extra["runtime"]["env"]["STRATA_QWEN35_UNREGISTERED"] = "1"
        with self.assertRaisesRegex(ValueError, "registered3"):
            probe.make_arm_spec(extra, probe.COFETCH_ARM_ORDER[0], COFETCH_MANIFEST)
        bad_arm = copy.deepcopy(probe.COFETCH_ARM_ORDER[0])
        bad_arm["unregistered_flag"] = True
        with self.assertRaisesRegex(ValueError, "unknown arm feature"):
            probe.validate_arm_definition(bad_arm)

    def test_di02_mode_is_unchanged(self):
        order, _ = probe.validate_arm_options({"budgets": BUDGETS}, 5)
        self.assertEqual([arm["id"] for arm in order], [a["id"] for a in probe.ARM_ORDER])
        base = {"engine": {"path": "x", "sha256": "0" * 64},
                "runtime": {"id": "x", "env": {"A": "1", "B": "2", "C": "3"}}}
        on = probe.make_arm_spec(copy.deepcopy(base),
                                 {"id": "new-ON-B1", "engine": "new", "diag": True},
                                 COFETCH_MANIFEST)
        self.assertEqual(on["runtime"]["env"],
                         {"A": "1", "B": "2", "C": "3", probe.DIAG_ENV: "1"})


class CofetchMarkerTests(unittest.TestCase):
    def test_on_exactly_one_off_never(self):
        on, off = cofetch_arm("cofetch-ON-smoke"), cofetch_arm("cofetch-OFF-smoke")
        self.assertEqual(probe.validate_cofetch_marker(on, "x\n" + probe.COFETCH_MARKER + "\n"), 1)
        self.assertEqual(probe.validate_cofetch_marker(off, "x\n"), 0)
        self.assertIsNone(probe.validate_cofetch_marker(probe.ARM_ORDER[1], "x\n"))

    def test_wrong_marker_counts_fail_closed(self):
        on, off = cofetch_arm("cofetch-ON-smoke"), cofetch_arm("cofetch-OFF-smoke")
        with self.assertRaisesRegex(ValueError, "marker count"):
            probe.validate_cofetch_marker(on, "x\n")
        with self.assertRaisesRegex(ValueError, "marker count"):
            probe.validate_cofetch_marker(on, probe.COFETCH_MARKER + "\n" + probe.COFETCH_MARKER)
        with self.assertRaisesRegex(ValueError, "emitted"):
            probe.validate_cofetch_marker(off, probe.COFETCH_MARKER + "\n")


class CofetchSmokeGateTests(unittest.TestCase):
    def smoke(self, **overrides):
        off = cofetch_result("cofetch-OFF-smoke", **overrides.get("off", {}))
        on = cofetch_result("cofetch-ON-smoke", **overrides.get("on", {}))
        return [off, on]

    def test_smoke_gate_accepts_complete_structural_evidence(self):
        gate = probe.validate_cofetch_smoke(self.smoke())
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["score_rank"]["SCORE0"]["consistent"])
        self.assertIn("not broad quality", gate["quality_scope"])

    def test_smoke_gate_rejects_wrong_count_cursor_eos_json_nan_ranking(self):
        cases = []
        wrong_count = self.smoke()
        wrong_count[0]["requests"].pop()
        cases.append(wrong_count)
        cases.append(self.smoke(off={"held_pos_delta": 1}))
        cases.append(self.smoke(on={"eos_stop": False}))
        cases.append(self.smoke(off={"json_complete": False}))
        cases.append(self.smoke(on={"nan": True}))
        cases.append(self.smoke(on={"ranking": (16, 15, 17, 18)}))
        for results in cases:
            with self.assertRaises(ValueError):
                probe.validate_cofetch_smoke(results)

    def test_persisted_marker_and_env_identity_are_required(self):
        results = self.smoke()
        results[1]["summary"]["cofetch_marker_count"] = 0
        with self.assertRaisesRegex(ValueError, "marker count"):
            probe.validate_cofetch_smoke(results)
        results = self.smoke()
        results[0]["summary"]["runtime_env"]["UNREGISTERED"] = "1"
        with self.assertRaisesRegex(ValueError, "runtime env"):
            probe.validate_cofetch_smoke(results)


class CofetchComparisonTests(unittest.TestCase):
    def formal(self, scales, **overrides):
        return [cofetch_result(arm_id, scale=scales[arm_id],
                               **overrides.get(arm_id, {}))
                for arm_id in probe.COFETCH_FORMAL_IDS]

    def test_named_only_comparison_and_gain_beyond_off_drift(self):
        comparison = probe.compare_cofetch(self.formal({
            "cofetch-OFF-A1": 1.0, "cofetch-OFF-A2": 1.05,
            "cofetch-ON-B1": 0.85, "cofetch-ON-B2": 0.88}))
        self.assertTrue(comparison["formal_complete"])
        self.assertEqual(set(comparison["named"]), set(probe.COFETCH_COMPARE_NAMES))
        self.assertNotIn("shape_gen", comparison)
        self.assertTrue(comparison["held_eos_not_mixed_into_shape_medians"])
        self.assertTrue(comparison["no_unconditional_pass"])
        self.assertAlmostEqual(comparison["named"]["P0"]["off_endpoint_drift"]["fraction"],
                               0.05 / 1.025)
        self.assertTrue(comparison["named"]["P0"]["gain_ge_5pct"])
        self.assertTrue(comparison["named"]["P0"]["beyond_off_drift"])
        self.assertIn("P0", comparison["workloads_with_gain_ge_5pct_and_beyond_off_drift"])
        self.assertEqual(comparison["decision"], "REVIEW_ONLY_GAIN_OBSERVED")
        self.assertFalse(comparison["instrumentation"]["formal_diag"])

    def test_no_gain_and_regression_are_not_pass(self):
        same = probe.compare_cofetch(self.formal({
            "cofetch-OFF-A1": 1.0, "cofetch-OFF-A2": 1.0,
            "cofetch-ON-B1": 1.0, "cofetch-ON-B2": 1.0}))
        self.assertEqual(same["decision"], "NO_GO_NO_GAIN_BEYOND_DRIFT")
        slower = probe.compare_cofetch(self.formal({
            "cofetch-OFF-A1": 1.0, "cofetch-OFF-A2": 1.0,
            "cofetch-ON-B1": 1.25, "cofetch-ON-B2": 1.25}))
        self.assertEqual(slower["decision"], "NO_GO_REGRESSION")
        self.assertTrue(any(value.startswith("e2e_regression:")
                            for value in slower["unacceptable_regressions"]))
        for comparison in (same, slower):
            self.assertTrue(comparison["no_unconditional_pass"])
            self.assertNotEqual(comparison["decision"], "PASS")

    def test_formal_score_rank_mismatch_is_a_quality_regression(self):
        comparison = probe.compare_cofetch(self.formal({
            "cofetch-OFF-A1": 1.0, "cofetch-OFF-A2": 1.0,
            "cofetch-ON-B1": 0.85, "cofetch-ON-B2": 0.85},
            **{"cofetch-ON-B1": {"ranking": (16, 15, 17, 18)}}))
        self.assertTrue(any(value.startswith("quality_regression:SCORE0")
                            for value in comparison["unacceptable_regressions"]))
        self.assertEqual(comparison["decision"], "NO_GO_REGRESSION")

    def test_smoke_only_comparison_is_incomplete(self):
        comparison = probe.compare_cofetch([cofetch_result("cofetch-OFF-smoke"),
                                            cofetch_result("cofetch-ON-smoke")])
        self.assertFalse(comparison["formal_complete"])
        self.assertTrue(comparison["no_unconditional_pass"])


if __name__ == "__main__":
    unittest.main()
