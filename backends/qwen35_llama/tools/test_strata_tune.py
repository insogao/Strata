#!/usr/bin/env python3
"""0-GPU unit tests for P3-4: strata_tune.py (inspect/screen/plan/profile/check).

    python backends/qwen35_llama/tools/test_strata_tune.py -v

Everything is synthetic or in-memory: no GPU, no nvidia-smi requirement, no engine, no lease,
no build.  Fingerprints are the same recorded local-MTP facts the P3-2 tests use, extended with
the device fields (driver/compute capability) the profile schema binds.  Temp files live in a
directory beside this test (never %TEMP%).
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import strata_screen as ss  # noqa: E402
import strata_tune as st  # noqa: E402


def L(v, s="gguf-meta"):
    return {"value": v, "source": s}


def local_fp(**device_overrides):
    """Synthetic fingerprint shaped like strata_fingerprint.py output for the local MTP model."""
    dev = {"name": "NVIDIA GeForce RTX 4060 Laptop GPU", "vram_total_mib": 8188,
           "driver_version": "560.94", "compute_capability": "8.9"}
    dev.update(device_overrides)
    return {
        "schema": "strata-fingerprint/1",
        "gpu": {"name": L(dev["name"], "nvidia-smi"), "compute_capability": L(dev["compute_capability"], "nvidia-smi"),
                "vram_total_mib": L(dev["vram_total_mib"], "nvidia-smi"), "vram_free_mib": L(7297, "nvidia-smi"),
                "driver_version": L(dev["driver_version"], "nvidia-smi")},
        "cpu": {"physical_cores": L(16, "os-api"), "logical_cores": L(32, "os-api")},
        "ram": {"total_mib": L(32448.5, "os-api"), "avail_mib": L(16162.6, "os-api"),
                "commit_limit_mib": L(44810.0, "os-api")},
        "model": {"path": L("<synthetic>", "ask"), "size_bytes": L(15346432288, "fs"),
                  "sha256": L("ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f", "ask"),
                  "gguf": {"block_count": L(41), "context_length": L(262144),
                           "expert_count": L(256), "expert_used_count": L(8),
                           "full_attention_interval": L(4),
                           "attention": {"head_count": L(16), "head_count_kv": L(2),
                                         "key_length": L(256), "value_length": L(256)},
                           "ssm": {"conv_kernel": L(4), "state_size": L(128), "group_count": L(16),
                                   "time_step_rank": L(32), "inner_size": L(4096)},
                           "routed_expert_bytes_total": L(13201571840),
                           "non_expert_bytes_total": L(2133869056),
                           "routed_expert_layers": L(41), "has_mtp_block": L(True)}},
        "workload": {"context": L(4096, "ask"), "max_new": L(128, "ask"), "batch": L(1, "ask")},
    }


ARGV_OK = "slots=2700,ubatch=4096,threads=8,ctx=8192,compact=on"


class TuneTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-tune-", dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fp_file = self.root / "fp.json"
        self.fp_file.write_text(json.dumps(local_fp()), encoding="utf-8")

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = st.main(argv)
        return code, out.getvalue(), err.getvalue()

    def write_profile(self, **kw):
        path = kw.pop("out", self.root / "profile.json")
        argv = ["profile", "--fingerprint", str(kw.pop("fp", self.fp_file)),
                "--model-sha", kw.pop("sha", "ab639a7f"),
                "--quant", kw.pop("quant", "IQ3_S"),
                "--slots", "2700", "--ubatch", "4096", "--threads", "8", "--ctx", "8192",
                "--compact", "on", "--confidence", kw.pop("confidence", "ok"),
                "--out", str(path)]
        for e in kw.pop("evidence", ["results/opencode-execution/strata-unified-p3-3-autotune-pilot"]):
            argv += ["--evidence", e]
        if kw.pop("recommend", False):
            argv += ["--recommend"]
        if kw.pop("json", False):
            argv += ["--json"]
        assert not kw, kw
        return self.run_cli(argv), path

    def check_profile(self, path, **kw):
        argv = ["check", str(path), "--fingerprint", str(kw.pop("fp", self.fp_file)),
                "--quant", kw.pop("quant", "IQ3_S"), "--argv", kw.pop("argv", ARGV_OK)]
        if kw.pop("json", False):
            argv += ["--json"]
        assert not kw, kw
        return self.run_cli(argv)


class InspectTests(TuneTestBase):
    def test_inspect_json_from_fingerprint(self):
        code, out, _ = self.run_cli(["inspect", "--fingerprint", str(self.fp_file), "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["schema"], "strata-tune-inspect/1")
        self.assertEqual(doc["gpu"]["name"], "NVIDIA GeForce RTX 4060 Laptop GPU")
        self.assertEqual(doc["model"]["sha256"], "ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f")
        self.assertIn("vram", doc["pools"])
        self.assertIn("host", doc["pools"])
        self.assertTrue(doc["plan_candidates"])

    def test_inspect_human_has_device_model_pools(self):
        code, out, _ = self.run_cli(["inspect", "--fingerprint", str(self.fp_file),
                                     "--context", "4096", "--max-new", "128"])
        self.assertEqual(code, 0)
        self.assertIn("device", out)
        self.assertIn("model", out)
        self.assertIn("vram", out)
        self.assertIn("host", out)

    def test_inspect_requires_a_fingerprint_source(self):
        code, _, err = self.run_cli(["inspect"])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("--fingerprint", err)

    def test_inspect_missing_fingerprint_file_is_refused(self):
        code, _, err = self.run_cli(["inspect", "--fingerprint", str(self.root / "nope.json")])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("no such fingerprint file", err)


class ScreenTests(TuneTestBase):
    def test_screen_output_carries_positioning_and_holdout(self):
        code, out, _ = self.run_cli(["screen", "--fingerprint", str(self.fp_file)])
        self.assertEqual(code, 0)
        self.assertIn("只能可信拒绝、不足以选优", out)
        self.assertIn("9.70", out)
        self.assertIn("21.30", out)
        self.assertIn("ok=", out)
        self.assertIn("rejected=", out)

    def test_screen_classification_matches_p3_2_and_is_deterministic(self):
        code, out1, _ = self.run_cli(["screen", "--fingerprint", str(self.fp_file)])
        code2, out2, _ = self.run_cli(["screen", "--fingerprint", str(self.fp_file)])
        self.assertEqual((code, code2), (0, 0))
        self.assertEqual(out1, out2)
        fp = json.loads(self.fp_file.read_text(encoding="utf-8"))
        fp["_path"] = str(self.fp_file)
        direct = ss.screen(fp, [2200, 2700, 3200], [True, False], [8, 16], [1024, 2048, 4096],
                           [4096, 8192], 5, 512, 306, 4)
        cj = json.loads(self.run_cli(["screen", "--fingerprint", str(self.fp_file), "--json"])[1])
        self.assertEqual([c["id"] for c in cj["candidates"]],
                         [r["id"] for r in direct["candidates"]])
        self.assertEqual([s["id"] for s in cj["shortlist"]],
                         [s["id"] for s in direct["shortlist"]])
        self.assertLessEqual(len(cj["shortlist"]), 5)
        self.assertEqual(cj["positioning"]["statement"], st.SCREEN_POSITION)
        self.assertEqual(cj["positioning"]["holdout_mean_abs_error_pct"], 9.7017)
        self.assertEqual(cj["positioning"]["holdout_max_abs_error_pct"], 21.2994)

    def test_screen_rejects_bad_fingerprint(self):
        bad = self.root / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = self.run_cli(["screen", "--fingerprint", str(bad)])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("not readable JSON", err)

    def test_screen_shortlist_cap_is_enforced(self):
        code, _, err = self.run_cli(["screen", "--fingerprint", str(self.fp_file), "--shortlist-max", "9"])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("--shortlist-max", err)


class PlanTests(TuneTestBase):
    def test_plan_is_a_preregistered_checklist_not_executed(self):
        code, out, _ = self.run_cli(["plan", "--fingerprint", str(self.fp_file), "--json"])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertFalse(doc["executed"])
        self.assertEqual(len(doc["design"]["per_candidate_interleave"]), 6)
        self.assertEqual(doc["design"]["per_candidate_interleave"][:3], ["S-A1", "S-B1", "S-A2"])
        self.assertEqual([s["id"] for s in doc["shapes"]], ["S", "L"])
        self.assertIn("PASS", doc["decision_rule"])
        self.assertIn("NO-GO", doc["decision_rule"])
        self.assertTrue(any("窗口" in r for r in doc["window_requirements"]))
        self.assertTrue(any("gpu-lock" in r for r in doc["lease_requirements"]))
        self.assertIn("preregistration.json", doc["preregistration"])

    def test_plan_human_prints_requirements_and_no_native(self):
        code, out, _ = self.run_cli(["plan", "--fingerprint", str(self.fp_file)])
        self.assertEqual(code, 0)
        self.assertIn("executed=False", out)
        self.assertIn("no native/lease/engine", out)
        self.assertIn("decision", out)
        self.assertIn("window", out)
        self.assertIn("lease", out)

    def test_plan_negative_shortlist_cap(self):
        code, _, err = self.run_cli(["plan", "--fingerprint", str(self.fp_file), "--shortlist-max", "0"])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("--shortlist-max", err)


class ProfileTests(TuneTestBase):
    def test_profile_roundtrip_and_check_passes(self):
        (code, out, _), path = self.write_profile()
        self.assertEqual(code, 0)
        self.assertIn("wrote", out)
        doc = json.loads(path.read_text(encoding="utf-8"))
        for key in ("schema_version", "model_sha256_prefix", "quant", "device_fingerprint", "argv",
                    "evidence", "confidence", "generated_at", "invalidators", "recommendation"):
            self.assertIn(key, doc)
        self.assertEqual(doc["schema_version"], st.PROFILE_SCHEMA)
        self.assertEqual(doc["argv"], {"slots": 2700, "ubatch": 4096, "threads": 8, "ctx": 8192,
                                       "compact": True})
        self.assertFalse(doc["recommendation"])
        ccode, cout, _ = self.check_profile(path)
        self.assertEqual(ccode, st.EXIT_OK)
        self.assertIn("check OK", cout)

    def test_profile_without_evidence_refuses_calibrated_confidence(self):
        (code, _, err), _ = self.write_profile(evidence=[], confidence="ok")
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("uncalibrated", err)

    def test_profile_uncalibrated_is_written_but_never_a_recommendation(self):
        (code, out, _), path = self.write_profile(evidence=[], confidence="uncalibrated")
        self.assertEqual(code, 0)
        self.assertIn("不是推荐", out)
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["confidence"], "uncalibrated")
        self.assertFalse(doc["recommendation"])
        ccode, cout, _ = self.check_profile(path)
        self.assertEqual(ccode, st.EXIT_WARN_UNCALIBRATED)
        self.assertIn("W-UNCALIBRATED", cout)

    def test_profile_recommend_requires_ok_and_evidence(self):
        (code, _, err), _ = self.write_profile(confidence="uncalibrated", recommend=True)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("recommendation", err)
        (code, _, err), _ = self.write_profile(evidence=[], confidence="ok", recommend=True)
        self.assertEqual(code, st.EXIT_REFUSED)
        (code, _, _), path = self.write_profile(confidence="ok", recommend=True)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["recommendation"])

    def test_profile_bad_sha_prefix_is_refused(self):
        (code, _, err), _ = self.write_profile(sha="NOT-HEX")
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("--model-sha", err)


class CheckTests(TuneTestBase):
    def test_check_model_sha_mismatch_prints_both_sides(self):
        (_, _, _), path = self.write_profile(sha="55983c5a")
        code, out, _ = self.check_profile(path)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("MODEL_SHA_MISMATCH", out)
        self.assertIn("stored", out)
        self.assertIn("current", out)
        self.assertIn("55983c5a", out)
        self.assertIn("ab639a7f", out)

    def test_check_quant_mismatch_no_conversion(self):
        (_, _, _), path = self.write_profile(quant="IQ3_S")
        code, out, _ = self.check_profile(path, quant="Q4_K_XL")
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("QUANT_MISMATCH", out)
        self.assertIn("no cross-quantization conversion", out)

    def test_check_cross_card_is_refused_with_both_fingerprints(self):
        (_, _, _), path = self.write_profile()
        before = path.read_bytes()
        other = self.root / "fp-4090.json"
        other.write_text(json.dumps(local_fp(name="NVIDIA GeForce RTX 4090", vram_total_mib=24564,
                                             driver_version="552.22", compute_capability="8.9")),
                         encoding="utf-8")
        code, out, _ = self.check_profile(path, fp=other)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("DEVICE_MISMATCH", out)
        self.assertIn("GeForce RTX 4090", out)
        self.assertIn("GeForce RTX 4060", out)
        self.assertIn("不做任何换算", out)
        self.assertEqual(path.read_bytes(), before)

    def test_check_argv_mismatch_names_the_field(self):
        (_, _, _), path = self.write_profile()
        code, out, _ = self.check_profile(
            path, argv="slots=2700,ubatch=4096,threads=16,ctx=8192,compact=on")
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("ARGV_MISMATCH", out)
        self.assertIn("threads", out)
        self.assertIn("16", out)

    def test_check_unknown_current_sha_fails_closed(self):
        (_, _, _), path = self.write_profile()
        fp = local_fp()
        fp["model"]["sha256"] = {"value": None, "source": "unknown"}
        no_sha = self.root / "fp-nosha.json"
        no_sha.write_text(json.dumps(fp), encoding="utf-8")
        code, out, _ = self.check_profile(path, fp=no_sha)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("MODEL_SHA_MISMATCH", out)
        self.assertIn("fail closed", out)

    def test_check_rejects_corrupt_or_foreign_profile(self):
        bad = self.root / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = self.check_profile(bad)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("not readable JSON", err)
        foreign = self.root / "foreign.json"
        foreign.write_text(json.dumps({"schema_version": "something/else"}), encoding="utf-8")
        code, _, err = self.check_profile(foreign)
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn(st.PROFILE_SCHEMA, err)

    def test_check_json_mode_reports_machine_readable_status(self):
        (_, _, _), path = self.write_profile()
        code, out, _ = self.check_profile(path, json=True)
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["schema"], "strata-tune-check/1")
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["mismatches"], [])


class CliContractTests(TuneTestBase):
    def test_exit_code_semantics(self):
        code, _, _ = self.run_cli(["--help"])
        self.assertEqual(code, 0)
        code, _, err = self.run_cli([])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("usage", err.lower())

    def test_unknown_command_and_option_are_explicit_errors(self):
        code, _, err = self.run_cli(["frobnicate"])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("invalid choice", err)
        code, _, err = self.run_cli(["screen", "--bogus"])
        self.assertEqual(code, st.EXIT_REFUSED)
        self.assertIn("unrecognized arguments", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
