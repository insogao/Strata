#!/usr/bin/env python3
"""0-GPU unit tests for the P3-1 foundation: strata_fingerprint.py + strata_plan.py.

    python backends/qwen35_llama/tools/test_strata_autotune.py -v

Everything is synthetic or in-memory: no GPU, no nvidia-smi requirement, no engine, no lease,
no benchmark, no writes outside the repository (and normally none at all).  The plan tests use a
synthetic fingerprint whose numbers are the recorded official-pin facts (tasks 341/342):
UD-Q4_K_XL MTP, routed experts 20,122,173,440 B over 256 experts in 41 layers, 8,188 MiB card,
free 2,120 MiB at placement (2.07 GiB) -> 6 of 256 experts per layer = 0.44 GiB.
"""
import copy
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import strata_fingerprint as fg  # noqa: E402
import strata_plan as sp  # noqa: E402


class T:
    """TensorInfo stand-in: name, type_name, expected_bytes()."""

    def __init__(self, name, type_name, nbytes):
        self.name, self.type_name, self._bytes = name, type_name, nbytes

    def expected_bytes(self):
        return self._bytes


def official_pin_gguf():
    meta = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 41,
        "qwen35moe.context_length": 262144,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.expert_feed_forward_length": 512,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.time_step_rank": 32,
        "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.nextn_predict_layers": 1,
    }
    tensors = [T("token_embd.weight", "Q6_K", 100_000_000), T("output.weight", "Q6_K", 100_000_000)]
    per_layer = 490_784_000  # 20,122,173,440 / 41
    for layer in range(41):
        tensors.append(T(f"blk.{layer}.ffn_gate_exps.weight", "Q4_K", per_layer // 3))
        tensors.append(T(f"blk.{layer}.ffn_up_exps.weight", "Q4_K", per_layer // 3))
        tensors.append(T(f"blk.{layer}.ffn_down_exps.weight", "Q5_K", per_layer - 2 * (per_layer // 3)))
        tensors.append(T(f"blk.{layer}.ffn_gate_shexp.weight", "Q6_K", 1_000_000))
        tensors.append(T(f"blk.{layer}.attn_q.weight", "Q6_K", 2_000_000))
    tensors.append(T("blk.41.nextn.enorm.weight", "F32", 8192))
    return meta, tensors


def official_pin_fingerprint(**overrides):
    """A synthetic fingerprint shaped exactly like the tool's output, official-pin anchors."""
    def f(v, s="gguf-meta"):
        return {"value": v, "source": s}

    fp = {
        "schema": "strata-fingerprint/1",
        "gpu": {
            "name": f("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia-smi"),
            "vram_total_mib": f(8188, "nvidia-smi"),
            "vram_free_mib": f(7297, "nvidia-smi"),
            "memory_bandwidth_gbps": f(256, "table"),
        },
        "cpu": {"physical_cores": f(16, "os-api"), "logical_cores": f(32, "os-api")},
        "ram": {"total_mib": f(32446.4, "os-api"), "avail_mib": f(16160.0, "os-api"),
                "commit_limit_mib": f(44809.0, "os-api"), "commit_avail_mib": f(13076.0, "os-api")},
        "model": {
            "path": f("D:/models/official-5bc3e238/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "ask"),
            "size_bytes": f(22853663008, "fs"),
            "sha256": f("55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1", "ask"),
            "gguf": {
                "architecture": f("qwen35moe"),
                "block_count": f(41), "context_length": f(262144), "embedding_length": f(2048),
                "expert_count": f(256), "expert_used_count": f(8), "full_attention_interval": f(4),
                "attention": {"head_count": f(16), "head_count_kv": f(2), "key_length": f(256), "value_length": f(256)},
                "ssm": {"conv_kernel": f(4), "state_size": f(128), "group_count": f(16),
                        "time_step_rank": f(32), "inner_size": f(4096)},
                "nextn_predict_layers": f(1), "has_mtp_block": f(True),
                "tensor_count": f(753), "tensor_bytes_total": f(22842671616),
                "routed_expert_bytes_total": f(20122173440), "non_expert_bytes_total": f(2720498176),
                "routed_expert_layers": f(41), "routed_expert_layer_bytes": f([490_784_000] * 41),
            },
        },
        "workload": {"context": f(4096, "ask"), "max_new": f(128, "ask"), "batch": f(1, "ask")},
    }
    for path, value in overrides.items():
        node = fp
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return fp


def plan_of(fp, **opts):
    return sp.build_plan(fp, dict(opts))


class FingerprintUnitTests(unittest.TestCase):
    def test_unknown_is_not_guessed(self):
        rc = fg.probe_gpu(runner=lambda cmd: (None, ""))
        self.assertIsNone(rc["name"]["value"])
        self.assertEqual(rc["name"]["source"], "unknown")
        self.assertIsNone(rc["vram_total_mib"]["value"])
        self.assertIsNone(fg.bandwidth_for("Some Unlisted GPU")["value"])
        self.assertEqual(fg.bandwidth_for("Some Unlisted GPU")["source"], "unknown")
        self.assertEqual(fg.field(None, "nvidia-smi")["source"], "unknown")

    def test_nvidia_smi_parse(self):
        parsed = fg.parse_nvidia_smi_query("NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 8188, 7297, 591.74")
        self.assertEqual(parsed["name"], "NVIDIA GeForce RTX 4060 Laptop GPU")
        self.assertEqual(parsed["vram_total_mib"], 8188)
        self.assertEqual(parsed["compute_capability"], "8.9")
        self.assertEqual(fg.parse_nvidia_smi_query("no gpu here"), {})

        def runner(cmd):
            if any("--query-gpu" in c for c in cmd):
                return 0, "NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 8188, 7297, 591.74"
            return 0, "| NVIDIA-SMI 591.74  CUDA Version: 13.1 |"

        gpu = fg.probe_gpu(runner=runner)
        self.assertEqual(gpu["memory_bandwidth_gbps"], {"value": 256, "source": "table",
                                                         "note": "vendor nominal spec matched on 'RTX 4060 Laptop'; GB/s decimal, no measurement"})
        self.assertEqual(gpu["cuda_driver_max"]["value"], "13.1")

    def test_bandwidth_table_is_a_table_not_a_measurement(self):
        self.assertIn("no measurement", fg.bandwidth_for("RTX 4060 Laptop GPU")["note"])

    def test_summarize_metadata_experts_and_mtp(self):
        meta, tensors = official_pin_gguf()
        s = fg.summarize_metadata(meta, tensors)
        self.assertEqual(s["routed_expert_bytes_total"]["source"], "gguf-meta")
        self.assertEqual(s["routed_expert_layers"]["value"], 41)
        self.assertEqual(s["expert_count"]["value"], 256)
        self.assertTrue(s["has_mtp_block"]["value"])
        self.assertEqual(s["nextn_predict_layers"]["value"], 1)
        total = s["routed_expert_bytes_total"]["value"]
        self.assertEqual(total, sum(t.expected_bytes() for t in tensors if "_exps" in t.name))
        self.assertEqual(s["non_expert_bytes_total"]["value"],
                         s["tensor_bytes_total"]["value"] - total)
        self.assertEqual(s["role_quant"]["ffn_gate_exps"]["value"], ["Q4_K"])
        self.assertEqual(len(s["routed_expert_layer_bytes"]["value"]), 41)

    def test_summarize_metadata_missing_is_unknown(self):
        s = fg.summarize_metadata({}, [])
        self.assertIsNone(s["expert_count"]["value"])
        self.assertEqual(s["expert_count"]["source"], "unknown")
        self.assertIsNone(s["context_length"]["value"])
        self.assertFalse(s["has_mtp_block"]["value"])

    def test_every_fingerprint_leaf_has_a_known_source(self):
        def fake_probe(section):
            return lambda *args, **kw: section

        doc = fg.build_fingerprint(
            "X:/model.gguf", {"context": 4096, "max_new": 128, "batch": 1}, None,
            gpu_probe=fake_probe({"name": fg.unknown("no gpu"), "vram_total_mib": fg.unknown("no gpu")}),
            cpu_probe=fake_probe({"physical_cores": fg.unknown("no psutil")}),
            ram_probe=fake_probe({"total_mib": fg.field(1000, "os-api")}),
            disk_probe=fake_probe({"kind": fg.unknown("no path")}),
            model_probe=fake_probe({"sha256": fg.unknown("no registered sha")}))
        leaves = fg.walk_sources(doc)
        self.assertGreater(len(leaves), 5)
        for path, leaf in leaves:
            self.assertIn(leaf["source"], fg.SOURCE_LEGEND, path)
            if leaf["value"] is None:
                self.assertEqual(leaf["source"], "unknown", path)

    def test_model_sha_policy_never_recomputed(self):
        section = fg.probe_model(__file__, None)
        self.assertIsNone(section["sha256"]["value"])
        self.assertEqual(section["sha256"]["source"], "unknown")
        section = fg.probe_model(__file__, "AB" * 32)
        self.assertEqual(section["sha256"]["value"], "ab" * 32)
        self.assertEqual(section["sha256"]["source"], "ask")
        self.assertEqual(section["size_bytes"]["source"], "fs")


class PlanTests(unittest.TestCase):
    def test_static_chain_order_and_two_pools(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1)
        self.assertEqual(plan["status"], "ok")
        steps = [s["step"] for s in plan["vram"]["static_chain"]["steps"]]
        self.assertEqual(steps, ["start", "desktop_other_used", "non_expert_weights", "kv_gdn_state",
                                 "workspace_graph", "prefill_staging", "safety_margin"])
        term_names = [t["name"] for t in plan["host"]["terms"]]
        self.assertEqual(term_names, ["model_resident", "host_expert_cache", "prefill_staging", "working_allowance"])
        self.assertNotIn("host", json.dumps(plan["vram"]))
        self.assertNotIn("vram", json.dumps(plan["host"]))

    def test_budget_monotonic_in_context(self):
        fp = official_pin_fingerprint()
        quotas, kvs = [], []
        for ctx in (1024, 2048, 4096):
            plan = plan_of(fp, context=ctx, max_new=16, batch=1)
            quotas.append(plan["vram"]["expert_slot"]["quota_per_layer"])
            kvs.append(plan["vram"]["state"]["kv_mib"])
        self.assertLess(kvs[0], kvs[1])
        self.assertLess(kvs[1], kvs[2])
        self.assertGreaterEqual(quotas[0], quotas[1])
        self.assertGreaterEqual(quotas[1], quotas[2])

    def test_runtime_clamp_anchor_reproduces_0_44_gib(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1, free_mib=2120.0)
        slot = plan["vram"]["expert_slot"]
        self.assertEqual(slot["quota_per_layer"], 6)
        self.assertAlmostEqual(slot["cache_gib"], 0.4392, delta=0.01)
        self.assertEqual(plan["vram"]["effective"]["source"], "runtime-free")
        self.assertEqual(plan["vram"]["runtime_clamp"]["leftover_mib"], 482.0)

    def test_runtime_clamp_anchor_ctx8192_reproduces_0_22_gib(self):
        plan = plan_of(official_pin_fingerprint(), context=8192, max_new=128, batch=1, free_mib=1925.0)
        slot = plan["vram"]["expert_slot"]
        self.assertEqual(slot["quota_per_layer"], 3)
        self.assertAlmostEqual(slot["cache_gib"], 0.2196, delta=0.01)

    def test_static_plan_reports_the_upper_bound_not_the_clamp(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1)
        self.assertEqual(plan["vram"]["effective"]["source"], "static")
        self.assertGreater(plan["vram"]["expert_slot"]["quota_per_layer"], 6)
        self.assertIsNone(plan["vram"]["runtime_clamp"])

    def test_out_of_bounds_explicitly_rejected(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=4)
        self.assertEqual(plan["status"], "rejected")
        self.assertTrue(any("batch" in json.dumps(r) for r in plan["rejected"]))
        plan = plan_of(official_pin_fingerprint(), context=300000, max_new=128, batch=1)
        self.assertEqual(plan["status"], "rejected")
        self.assertTrue(any("context_length" in r["reason"] for r in plan["rejected"]))
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=8192, batch=1)
        self.assertTrue(any("max_new > context" in r["reason"] for r in plan["rejected"]))
        plain = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1)
        self.assertTrue(any("all 256 experts" in r["reason"] for r in plain["rejected"]))

    def test_pinned_host_rejected_with_the_recorded_fact(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1, host_mode="pinned")
        self.assertFalse(plan["host"]["fits"])
        self.assertEqual(plan["host"]["fallback"], "pageable")
        self.assertTrue(any("cudaHostAlloc" in r["reason"] for r in plan["rejected"]))
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1,
                       host_mode="pinned", pinned_cap_mib=1024.0)
        self.assertTrue(any("pinned-cap-mib" in r["reason"] for r in plan["rejected"]))

    def test_host_pool_fits_pageable_model_plus_staging(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1)
        self.assertTrue(plan["host"]["fits"])
        self.assertTrue(plan["host"]["pagefile_counted"])
        self.assertGreater(plan["host"]["requirement_mib"], sp.mib(22853663008 - 1))
        self.assertIn("pagefile", plan["host"]["notes"][0])
        local = official_pin_fingerprint(**{"model.size_bytes": {"value": 15346432288, "source": "fs"}})
        plan = plan_of(local, context=4096, max_new=128, batch=1)
        self.assertTrue(plan["host"]["fits"])
        self.assertGreater(plan["host"]["requirement_mib"], sp.mib(15346432288))
        self.assertLess(plan["host"]["requirement_mib"], plan["host"]["physical"]["total_mib"])

    def test_two_pools_are_not_mixed(self):
        fp = official_pin_fingerprint()
        base = plan_of(fp, context=4096, max_new=128, batch=1)
        fp2 = copy.deepcopy(fp)
        fp2["ram"]["commit_limit_mib"] = {"value": 20000.0, "source": "os-api"}
        fp2["ram"]["total_mib"] = {"value": 16000.0, "source": "os-api"}
        other = plan_of(fp2, context=4096, max_new=128, batch=1)
        self.assertEqual(base["vram"], other["vram"])
        fp3 = copy.deepcopy(fp)
        fp3["gpu"]["vram_total_mib"] = {"value": 24576, "source": "nvidia-smi"}
        fp3["gpu"]["vram_free_mib"] = {"value": 23000, "source": "nvidia-smi"}
        other3 = plan_of(fp3, context=4096, max_new=128, batch=1)
        self.assertEqual(base["host"], other3["host"])

    def test_modes_change_allocation_only(self):
        fp = official_pin_fingerprint()
        reference = None
        for mode in ("safe", "balanced", "latency", "throughput"):
            plan = plan_of(fp, context=4096, max_new=128, batch=1, mode=mode)
            self.assertEqual(plan["mode"], mode)
            self.assertEqual(plan["status"], "ok")
            invariants = (plan["vram"]["total_mib"], plan["vram"]["state"]["kv_mib"],
                          plan["vram"]["state"]["gdn_mib"], plan["vram"]["expert_slot"]["step_mib"])
            if reference is None:
                reference = invariants
            else:
                self.assertEqual(invariants, reference)
            self.assertGreaterEqual(min(c["config"]["safety_mib"] for c in plan["candidates"]), 512.0)
            for cand in plan["candidates"]:
                self.assertTrue(cand["reason"] and cand["risk"])
            self.assertTrue(all(c["fits"] for c in plan["candidates"]))
        default = plan_of(fp, context=4096, max_new=128, batch=1)
        self.assertEqual(default["mode"], "safe")
        self.assertNotIn("tok/s", json.dumps(default).lower())
        self.assertNotIn('"tps"', json.dumps(default).lower())

    def test_unknown_vram_refuses_to_plan(self):
        fp = official_pin_fingerprint()
        fp["gpu"]["vram_total_mib"] = fg.unknown("no nvidia-smi")
        with self.assertRaises(ValueError) as caught:
            plan_of(fp, context=4096, max_new=128, batch=1)
        self.assertIn("vram_total_mib", str(caught.exception))

    def test_zero_slot_budget_is_a_listed_rejection(self):
        plan = plan_of(official_pin_fingerprint(), context=4096, max_new=128, batch=1, staging_mib=8000.0)
        self.assertEqual(plan["vram"]["expert_slot"]["quota_per_layer"], 0)
        self.assertTrue(any("no whole expert step" in r["reason"] for r in plan["rejected"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
