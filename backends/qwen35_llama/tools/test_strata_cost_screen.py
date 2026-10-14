#!/usr/bin/env python3
"""0-GPU unit tests for P3-2: strata_cost_model.py + strata_screen.py.

    python backends/qwen35_llama/tools/test_strata_cost_screen.py -v

Everything is synthetic or in-memory: no GPU, no nvidia-smi requirement, no engine, no lease,
no benchmark, no writes.  The screen tests use a synthetic fingerprint whose numbers are the
recorded local-MTP facts (same shape as fp-mtp-iq3s.json: 41 blocks, 256 experts, top-8,
routed experts 13,201,571,840 B, 8,188 MiB card) so no external file is needed.
"""
import copy
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import strata_cost_model as cm  # noqa: E402
import strata_screen as ss  # noqa: E402


def L(v, s="gguf-meta"):
    return {"value": v, "source": s}


def local_fp(**overrides):
    """Synthetic fingerprint shaped like strata_fingerprint.py output for the local MTP model."""
    fp = {
        "schema": "strata-fingerprint/1",
        "gpu": {"name": L("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia-smi"),
                "vram_total_mib": L(8188, "nvidia-smi"), "vram_free_mib": L(7297, "nvidia-smi")},
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
    for k, v in overrides.items():
        if k == "commit_limit_mib":
            fp["ram"]["commit_limit_mib"] = L(v, "os-api")
        elif k == "vram_total_mib":
            fp["gpu"]["vram_total_mib"] = L(v, "nvidia-smi")
        elif k == "workload_context":
            fp["workload"]["context"] = L(v, "ask")
        else:
            raise KeyError(k)
    return fp


def default_screen(**kw):
    args = dict(slots=[2200, 2700, 3200], compacts=[True, False], threads=[8, 16],
                ubatches=[1024, 2048, 4096], contexts=[4096, 8192], shortlist_max=5,
                shape_tokens=512, shape_rounds=306, kv_bytes=4)
    args.update(kw)
    return ss.screen(local_fp(), **args)


class CostModelTests(unittest.TestCase):
    def setUp(self):
        self.params = cm.fit(cm.RECORDED_CASES)

    def test_params_are_direct_ratios_of_recorded_fields(self):
        p = self.params
        self.assertAlmostEqual(p["cpu_miss_us_per_expert"]["values_per_case"]["N1"],
                               3572.0 / 77086 * 1000, places=4)
        self.assertAlmostEqual(p["id_readback_ms_per_round"]["values_per_case"]["N2"],
                               3912.7 / 306, places=4)
        self.assertAlmostEqual(p["gpu_prefix_ms_per_round"]["values_per_case"]["N4"],
                               948.9 / 306, places=4)
        self.assertAlmostEqual(p["combine_ms_per_round"]["values_per_case"]["N3"],
                               121.1 / 160, places=4)
        self.assertAlmostEqual(p["non_tier_ms_per_token"]["values_per_case"]["N1"],
                               32.1005 - 5772.2 / 256, places=3)

    def test_every_parameter_has_source_interval_and_spread(self):
        for name in cm.PARAMS:
            p = self.params[name]
            self.assertTrue(p["source"])
            self.assertEqual(len(p["values_per_case"]), 4)
            self.assertLessEqual(p["min"], p["mean"])
            self.assertLessEqual(p["mean"], p["max"])
            self.assertGreaterEqual(p["spread_pct_of_mean"], 0.0)

    def test_monotonic_higher_miss_cost_lowers_tps(self):
        base = cm.predict(self.params, 512, 306, misses=149500)
        bumped = copy.deepcopy(self.params)
        bumped["cpu_miss_us_per_expert"]["mean"] *= 1.1
        after = cm.predict(bumped, 512, 306, misses=149500)
        self.assertLess(after["tps"]["central"], base["tps"]["central"])

    def test_more_misses_lower_tps(self):
        few = cm.predict(self.params, 512, 306, misses=100000)
        many = cm.predict(self.params, 512, 306, misses=200000)
        self.assertLess(many["tps"]["central"], few["tps"]["central"])

    def test_interval_widens_when_miss_fraction_unknown(self):
        known = cm.predict(self.params, 512, 306, misses=149500)
        unknown = cm.predict(self.params, 512, 306, misses=None)
        self.assertEqual(unknown["status"], "wide")
        self.assertIn("miss_fraction", unknown["unknown_terms"])
        known_width = known["tps"]["interval"][1] - known["tps"]["interval"][0]
        unknown_width = unknown["tps"]["interval"][1] - unknown["tps"]["interval"][0]
        self.assertGreater(unknown_width, known_width)
        self.assertAlmostEqual(unknown["tps"]["interval"][1],
                               1000.0 / unknown["ms_per_token"]["interval"][0], places=3)

    def test_uncalibrated_threads_are_not_guessed(self):
        p = cm.predict(self.params, 512, 306, misses=149500, threads=16)
        self.assertEqual(p["status"], "uncalibrated")
        self.assertEqual(p["unknown_terms"], ["cpu_pool_threads"])
        self.assertNotIn("tps", p)
        self.assertNotIn("ms_per_token", p)

    def test_holdout_is_within_batch_and_reports_honestly(self):
        h = cm.holdout(cm.RECORDED_CASES)
        self.assertEqual(len(h["rows"]), 4)
        for row in h["rows"]:
            self.assertEqual(len(row["fit_from"]), 3)
            self.assertNotIn(row["held_out"], row["fit_from"])
            self.assertIsInstance(row["measured_in_interval"], bool)
        self.assertGreater(h["mean_abs_central_error_pct"], 0.0)
        self.assertIn("/4", h["interval_coverage"])

    def test_in_sample_residuals_and_interval_cover(self):
        rows = cm.in_sample_residuals(self.params)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(row["interval_covers_measured"])


class ScreenTests(unittest.TestCase):
    def test_default_space_covers_all_candidates(self):
        doc = default_screen()
        self.assertEqual(len(doc["candidates"]), 3 * 2 * 2 * 3 * 2)
        statuses = {c["status"] for c in doc["candidates"]}
        self.assertTrue({"ok", "wide", "uncalibrated"}.issubset(statuses))
        self.assertNotIn("rejected", statuses)

    def test_anchor_prediction_matches_cost_model(self):
        doc = default_screen()
        anchor = next(c for c in doc["candidates"] if c["id"] == "s2700-c-t8-u4096-x8192")
        self.assertEqual(anchor["status"], "ok")
        expected = cm.predict(cm.fit(cm.RECORDED_CASES), 512, 306, misses=ss.ANCHOR_MISSES_256)
        self.assertEqual(anchor["prediction"]["tps"], expected["tps"])
        self.assertEqual(anchor["evidence_class"], "measured-anchor")

    def test_shortlist_cap_and_content(self):
        doc = default_screen()
        sl = doc["shortlist"]
        self.assertLessEqual(len(sl), 5)
        ids = [s["id"] for s in sl]
        self.assertEqual(len(ids), len(set(ids)))
        for s in sl:
            self.assertTrue(s["rationale"] or s["question"])
            self.assertTrue(s["expected_inflection_hypothesis"])
            self.assertTrue(s["failure_criteria"])
            self.assertTrue(s["minimal_verification"])
            self.assertNotEqual(s["status"], "rejected")
        # one entry per open question, not five copies of the anchor
        self.assertEqual(len({s["shortlist_key"] for s in sl}), len(sl))
        # both slot-count extremes are in for verification (not because of a speed prior)
        keys = {s["shortlist_key"] for s in sl}
        self.assertIn("anchor", keys)
        self.assertIn("slots-2200", keys)
        self.assertIn("slots-3200", keys)

    def test_rejects_context_beyond_declared_length(self):
        doc = default_screen(contexts=[524288])
        self.assertTrue(all(c["status"] == "rejected" for c in doc["candidates"]))
        reasons = " ".join(r["reason"] for c in doc["candidates"] for r in c["rejections"])
        self.assertIn("context_length", reasons)

    def test_rejects_ubatch_above_context(self):
        doc = default_screen(ubatches=[4096], contexts=[2048])
        self.assertTrue(all(c["status"] == "rejected" for c in doc["candidates"]))
        reasons = " ".join(r["reason"] for c in doc["candidates"] for r in c["rejections"])
        self.assertIn("ubatch", reasons)

    def test_rejects_host_pool_overflow(self):
        args = dict(slots=[2700], compacts=[True], threads=[8], ubatches=[4096], contexts=[8192],
                    shortlist_max=5, shape_tokens=512, shape_rounds=306, kv_bytes=4)
        fp = local_fp(commit_limit_mib=20000.0)
        doc = ss.screen(fp, **args)
        rejected = [c for c in doc["candidates"] if c["status"] == "rejected"]
        self.assertTrue(rejected)
        reasons = " ".join(r["reason"] for c in rejected for r in c["rejections"])
        self.assertIn("host requirement", reasons)

    def test_rejects_vram_growth_beyond_recorded_peak_headroom(self):
        # ctx 32768 KV/GDN growth (~1056 MiB) exceeds the 569.6 MiB headroom to the 347 peak
        doc = default_screen(contexts=[32768], slots=[2700], compacts=[True], threads=[8],
                             ubatches=[4096])
        self.assertTrue(all(c["status"] == "rejected" for c in doc["candidates"]))
        reasons = " ".join(r["reason"] for c in doc["candidates"] for r in c["rejections"])
        self.assertIn("headroom", reasons)

    def test_unknown_slot_count_cannot_outrank_the_measured_anchor(self):
        doc = default_screen()
        order = doc["ranking"]["order"]
        anchor = order.index("s2700-c-t8-u4096-x8192")
        bigger = order.index("s3200-c-t8-u4096-x8192")
        smaller = order.index("s2200-c-t8-u4096-x8192")
        self.assertLess(anchor, bigger)
        self.assertLess(anchor, smaller)

    def test_uncalibrated_cpu_pool_candidate_marked_not_guessed(self):
        doc = default_screen()
        cand = next(c for c in doc["candidates"] if c["id"] == "s2700-c-t16-u4096-x8192")
        self.assertEqual(cand["status"], "uncalibrated")
        self.assertNotIn("tps", cand["prediction"])
        self.assertTrue(cand["minimal_verification"])
        self.assertIn("threads=16", cand["unverified_dimensions"])

    def test_two_pools_are_not_mixed(self):
        a = default_screen(slots=[2200], compacts=[True], threads=[8], ubatches=[4096],
                           contexts=[8192])
        b = default_screen(slots=[3200], compacts=[True], threads=[8], ubatches=[4096],
                           contexts=[8192])
        cap_a = a["candidates"][0]["capacity"]
        cap_b = b["candidates"][0]["capacity"]
        self.assertEqual(cap_a["vram_static_leftover_mib"], cap_b["vram_static_leftover_mib"])
        self.assertNotEqual(cap_a["slot_cache_mib"], cap_b["slot_cache_mib"])
        self.assertGreater(cap_b["host_requirement_mib"], cap_a["host_requirement_mib"])

    def test_cache_cost_records(self):
        c = ss.cache_cost_mib(2700, True)
        self.assertEqual(c["mib"], 3199.2)
        self.assertTrue(c["measured_point"])
        u = ss.cache_cost_mib(2700, False)
        self.assertEqual(u["mib"], 4536.3)
        scaled = ss.cache_cost_mib(3200, True)
        self.assertAlmostEqual(scaled["mib"], 3200 * 3199.2 / 2700, places=1)
        self.assertFalse(scaled["measured_point"])
        self.assertIn("policy assumption", scaled["source"])

    def test_rejections_are_explicit_and_never_silent(self):
        doc = default_screen(contexts=[524288])
        for c in doc["candidates"]:
            self.assertEqual(c["status"], "rejected")
            self.assertTrue(c["rejections"])
            self.assertIsNone(c["prediction"])
            for r in c["rejections"]:
                self.assertTrue(r["reason"])
                self.assertTrue(r["stage"])


if __name__ == "__main__":
    unittest.main()
