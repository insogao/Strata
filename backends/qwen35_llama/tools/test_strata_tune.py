"""CPU-only profile lifecycle tests; synthetic host, fake binary, no GPU/engine probes."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import strata_tune as st
from test_strata_cost_screen import local_fp, L


class TuneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-tune-", dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / "engine.bin"
        self.engine.write_bytes(b"not an executable")
        self.fp = local_fp()
        self.fp["gpu"].update(uuid=L("GPU-fixture"), compute_capability=L("8.9"),
                              driver_version=L("fixture-driver"), cuda_driver_max=L("12.8"))
        self.fp["cpu"]["model_name"] = L("fixture CPU")
        self.fp["host"] = {"os": L("fixture OS")}
        self.fp["model"]["mtime_utc"] = L("2026-10-15T00:00:00Z")
        self.workload = {"context": 4096, "max_new": 128, "batch": 1,
                         "manifest_sha256": "a" * 64, "mtp_k": 1, "protocol": "diagnostic-cli"}
        self.baseline = {"slots": 2700, "compact": True, "threads": 8}

    def build(self, **kw):
        return st.build_profile(self.fp, self.engine, self.workload, self.baseline,
                                runtime_id="registered-runtime", **kw)

    def check(self, profile, **kw):
        return st.check_profile(profile, self.fp, self.engine, self.workload,
                                runtime_id="registered-runtime", **kw)

    def test_roundtrip_never_selects_or_applies(self):
        p = json.loads(json.dumps(self.build()))
        self.assertTrue(self.check(p)["valid"])
        self.assertIsNone(p["selected_configuration"])
        self.assertFalse(p["fallback"]["auto_apply"])
        self.assertFalse(self.check(p)["production_approved"])
        self.assertEqual(p["fallback"]["configuration"], self.baseline)

    def test_identity_changes_invalidate(self):
        original = copy.deepcopy(self.fp)
        p = self.build()
        for section, key, value in [("gpu", "uuid", "other-card"), ("gpu", "driver_version", "new"),
                                    ("gpu", "name", "another GPU"), ("cpu", "logical_cores", 8),
                                    ("ram", "total_mib", 16000), ("model", "sha256", "b" * 64),
                                    ("model", "mtime_utc", "new"), ("model", "size_bytes", 99)]:
            with self.subTest(section=section, key=key):
                self.fp = copy.deepcopy(original)
                self.fp[section][key] = L(value)
                self.assertFalse(self.check(p)["valid"])

    def test_quant_geometry_changes_invalidate(self):
        p = self.build()
        self.fp["model"]["gguf"]["expert_count"] = L(128)
        self.assertFalse(self.check(p)["valid"])

    def test_workload_and_policy_changes_invalidate(self):
        p = self.build()
        for key, value in [("max_new", 256), ("manifest_sha256", "b" * 64), ("mtp_k", 2),
                           ("protocol", "serve"), ("sampling", {"temperature": 0.7})]:
            old = copy.deepcopy(self.workload)
            self.workload[key] = value
            self.assertFalse(self.check(p)["valid"])
            self.workload = old
        self.assertFalse(self.check(p, mode="balanced")["valid"])
        self.assertFalse(self.check(p, kv_bytes=2)["valid"])

    def test_binary_and_runtime_changes_invalidate(self):
        p = self.build()
        self.engine.write_bytes(b"changed binary")
        self.assertFalse(self.check(p)["valid"])
        self.engine.write_bytes(b"not an executable")
        self.assertFalse(st.check_profile(p, self.fp, self.engine, self.workload,
                                         runtime_id="new-runtime")["valid"])

    def test_tool_version_change_invalidate(self):
        p = self.build()
        p["identity"]["tool_sha256"]["strata_plan.py"] = "b" * 64
        p["identity_sha256"] = st.digest(p["identity"])
        p["content_sha256"] = st.digest({k: v for k, v in p.items() if k != "content_sha256"})
        self.assertFalse(self.check(p)["valid"])

    def test_idle_free_change_not_identity_but_capacity_rechecked(self):
        p = self.build()
        self.fp["gpu"]["vram_free_mib"] = L(7000)
        result = self.check(p)
        self.assertTrue(result["valid"])
        self.assertEqual(result["changed_fields"], [])
        self.fp["gpu"]["vram_free_mib"] = L(1)
        result = self.check(p)
        self.assertFalse(result["valid"])
        self.assertEqual(result["changed_fields"], [])

    def test_unknown_identity_is_never_wildcard(self):
        self.fp["model"]["sha256"] = L(None, "unknown")
        p = self.build()
        self.assertTrue(p["missing_identity"])
        self.assertFalse(self.check(p)["valid"])

    def test_evidence_changed_or_missing(self):
        proof = self.root / "evidence.json"
        proof.write_text("{}", encoding="utf-8")
        p = self.build(evidence=[proof])
        self.assertTrue(self.check(p)["valid"])
        proof.write_text("[]", encoding="utf-8")
        self.assertFalse(self.check(p)["valid"])
        proof.unlink()
        self.assertFalse(self.check(p)["valid"])

    def test_corruption_and_unapproved_selection_fail(self):
        p = self.build()
        p["fallback"]["configuration"]["slots"] = 2200
        self.assertFalse(self.check(p)["valid"])
        p = self.build()
        p["selected_configuration"] = self.baseline
        p["content_sha256"] = st.digest({k: v for k, v in p.items() if k != "content_sha256"})
        self.assertFalse(self.check(p)["valid"])
        p["schema"] = "strata-tune-profile/999"
        self.assertFalse(self.check(p)["valid"])

    def test_all_modes_are_capacity_policies(self):
        for mode in st.sp.MODES:
            p = self.build(mode=mode)
            self.assertEqual(p["capacity_plan"]["mode"], mode)
            self.assertIsNone(p["selected_configuration"])

    def test_invalid_workload_refused(self):
        for key, value in [("batch", 2), ("context", -1), ("max_new", True),
                           ("max_new", 4096), ("manifest_sha256", "unknown")]:
            bad = {**self.workload, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                st.validate_workload(bad)

    def test_local_screen_has_counterevidence_no_selection(self):
        p = self.build(diagnostic_screen=True)
        s = p["diagnostic_screen"]
        self.assertEqual(s["pilot_counterevidence"]["verdict"], "NO-GO")
        self.assertIsNone(s["selected_configuration"])
        self.assertLessEqual(len(s["shortlist"]), 5)
        self.fp["gpu"]["name"] = L("another GPU")
        with self.assertRaises(ValueError):
            self.build(diagnostic_screen=True)

    def test_cli_roundtrip_stale_exit_and_no_overwrite(self):
        def put(name, obj):
            p = self.root / name
            p.write_text(json.dumps(obj), encoding="utf-8")
            return str(p)
        fp = put("fp.json", self.fp)
        args = ["--fingerprint", fp, "--engine", str(self.engine), "--runtime-id", "runtime",
                "--workload", put("workload.json", self.workload)]
        out = self.root / "profile.json"
        create = ["tune", *args, "--baseline", put("baseline.json", self.baseline), "--out", str(out)]
        self.assertEqual(st.main(create), 0)
        before = out.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(st.main(create), 2)
        self.assertEqual(out.read_bytes(), before)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(st.main(["check", *args, "--profile", str(out)]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "matching-research-record")
        self.engine.write_bytes(b"new binary")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(st.main(["check", *args, "--profile", str(out)]), 2)

    def test_nonfinite_json_refused(self):
        bad = self.root / "bad.json"
        bad.write_text('{"value": NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            st.read_json(bad)

    def test_profile_schema_fields_and_manual_fallback_required(self):
        for mutation in ("missing", "extra", "auto_apply"):
            p = self.build()
            if mutation == "missing":
                del p["fallback"]
            elif mutation == "extra":
                p["approved"] = True
            else:
                p["fallback"]["auto_apply"] = True
            p["content_sha256"] = st.digest({k: v for k, v in p.items() if k != "content_sha256"})
            self.assertFalse(self.check(p)["valid"])

    def test_schema_required_fields_match_writer(self):
        schema = st.read_json(st.TOOLS.parents[2] / "docs/backends/strata_tune_profile.schema.json")
        self.assertEqual(set(schema["required"]), st.PROFILE_FIELDS)
        self.assertEqual(set(self.build()), set(schema["required"]))


if __name__ == "__main__":
    unittest.main()
