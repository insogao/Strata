#!/usr/bin/env python3
"""strata_screen.py - offline candidate screening for the P3 autotuner (P3-2, 0 GPU).

    python backends/qwen35_llama/tools/strata_screen.py --fingerprint fp.json [--out screen.json] \
        [--slots 2200,2700,3200] [--compacts on,off] [--threads 8,16] \
        [--ubatches 1024,2048,4096] [--contexts 4096,8192] [--shortlist-max 5] \
        [--shape-tokens 512] [--shape-rounds 306] [--kv-bytes 4]

Screens a coarse-to-fine candidate space (expert-slot budget / layer quota with COMPACT on-off
-> CPU-pool threads -> prefill ubatch/context) through three stages, all offline:

1. capacity - the P3-1 planner (strata_plan.py, imported) rejects out-of-bounds combinations
   against the VRAM and host pools SEPARATELY, with the rejection reason listed verbatim; this
   tool adds two local-engine guards on top: prefill ubatch <= context, and a candidate's
   KV/GDN growth may not exceed the headroom to the highest recorded device VRAM peak of the
   347 batch (7618.4 MiB of 8188 MiB; the static chain under-describes where the local engine
   actually spends VRAM - see "unexplained").
2. cost - for the candidates that fit, the recorded-batch cost model (strata_cost_model.py)
   predicts a per-token time interval.  Only the measured point (2700 COMPACT, threads 8) has
   a recorded miss fraction; every other slot/thread point stays unknown and the interval
   widens to physical bounds (or is refused as uncalibrated) - never silently guessed.
3. shortlist - at most 5 candidates for the P3-3 GPU verification, one per open question, each
   with an expected-inflection hypothesis and a failure criterion.  Slot count is explicitly
   NOT a speed prior: more slots never automatically outrank a measured point.

0 GPU, no lease, no engine, no benchmark; no speed is measured or promised here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import strata_cost_model as cm  # noqa: E402
import strata_plan as sp  # noqa: E402

SCHEMA = "strata-screen/1"

# Host-side routed-expert cache cost, both records from the same 347 raw line and summary:
#   "strata: native expert tier (compact): 2700 slots, 3.12 GiB (uniform would be 4.43 GiB,
#    saved 1336 MiB)"  +  summary cache_bytes 3354632192 (= 3199.2 MiB, 2700 resident).
# Other slot counts scale linearly (policy assumption: average expert size does not change with
# the slot budget); the 2700 points are measured, everything else is derived and says so.
SLOT_CACHE_RECORDS = {
    True: {"label": "compact", "slots": 2700, "mib": 3199.2,
           "source": ("results/opencode-execution/phase3-60tps-local-long-baseline-347/raw-N*/child_00.log "
                      "'native expert tier (compact): 2700 slots, 3.12 GiB' + summary cache_bytes 3354632192")},
    False: {"label": "uniform", "slots": 2700, "mib": 4536.32,
            "source": ("results/opencode-execution/phase3-60tps-local-long-baseline-347/raw-N*/child_00.log "
                       "'uniform would be 4.43 GiB, saved 1336 MiB'")},
}

# Highest device VRAM peak of the 347 batch (N3: 7618.4 MiB; N1 7555.3, N2 7592.1, N4 7591.0)
# against a device_total of 8187.5 MiB.  This headroom is the only recorded bound on any
# candidate that would add VRAM beyond the recorded family.
RECORDED_PEAK_MIB = 7618.4
RECORDED_PEAK_SOURCE = ("results/opencode-execution/phase3-60tps-local-long-baseline-347/"
                        "raw-N*/child_00.log summary peak_device_vram_mib (max N3=7618.4; "
                        "device_total_mib 8187.5)")

# The recorded/batch anchor of the cost model (task 347 batch identity).
ANCHOR = {"slots": 2700, "compact": True, "threads": 8, "ubatch": 4096, "ctx": 8192}
RECORDED_DIMS = {"slots": {2700}, "compact": {True}, "threads": {8}, "ubatch": {4096},
                 "ctx": {4096, 8192}}
ANCHOR_MISSES_256 = 149500  # 347 N2/N4, the 256-output shape (512 tokens, 306 rounds)
SHAPE_NOTE = ("prediction unit = the 347 256-token long-output shape (512 tokens, 306 rounds, "
              "382.5 routed evaluations per token; misses 149500 = N2/N4 counters)")

L_CASE_INPUTS = ("l-risk-explain 3817 ids sha a920a130... / l-benign-qa 3791 ids sha ee12f567... "
                 "(results/20261012-qwen35-decode-m0-repair156/fixtures/)")


def cache_cost_mib(slots: int, compact: bool) -> dict:
    rec = SLOT_CACHE_RECORDS[compact]
    return {"slots": slots, "compact": compact,
            "mib": round(slots * rec["mib"] / rec["slots"], 1),
            "source": (f"{rec['label']} per-slot cost = {rec['mib']} MiB / {rec['slots']} slots "
                       f"({rec['source']}); linear scaling to {slots} slots is a policy assumption"),
            "measured_point": slots == rec["slots"]}


def candidate_id(c: dict) -> str:
    return f"s{c['slots']}-{'c' if c['compact'] else 'u'}-t{c['threads']}-u{c['ubatch']}-x{c['ctx']}"


def enumerate_space(slots, compacts, threads, ubatches, contexts) -> list:
    out = []
    for s, cp, t, ub, x in itertools.product(slots, compacts, threads, ubatches, contexts):
        out.append({"slots": s, "compact": cp, "threads": t, "ubatch": ub, "ctx": x})
    return out


def check_capacity(fp: dict, cand: dict, kv_bytes: int = 4) -> dict:
    """Planner-stage capacity check for one candidate; rejections are explicit and listed."""
    rejections = []
    if cand["ubatch"] > cand["ctx"]:
        rejections.append({"stage": "local", "config": {"prefill_ubatch": cand["ubatch"], "context": cand["ctx"]},
                           "reason": f"prefill ubatch {cand['ubatch']} > context {cand['ctx']}"})
    cache = cache_cost_mib(cand["slots"], cand["compact"])
    opts = {"context": cand["ctx"], "batch": 1, "mode": "safe", "kv_bytes": kv_bytes,
            "threads": cand["threads"], "host_extra_mib": cache["mib"]}
    try:
        plan = sp.build_plan(fp, opts)
    except ValueError as exc:
        rejections.append({"stage": "planner", "config": {}, "reason": f"planner refused: {exc}"})
        return {"rejections": rejections, "plan": None, "cache": cache}
    informational = []
    if plan.get("status") != "ok":
        for r in plan.get("rejected", []):
            rejections.append({"stage": "planner", "config": r.get("config"), "reason": r.get("reason")})
    else:
        # status ok: the planner's "rejected" list holds informational sub-combinations (e.g.
        # "all 256 experts of every layer" cannot fit) - those do not reject the candidate.
        informational = [{"config": r.get("config"), "reason": r.get("reason")}
                         for r in plan.get("rejected", [])]

    gpu = fp.get("gpu", {})
    total = sp.leaf(gpu, "vram_total_mib", default={}).get("value")
    gguf = (fp.get("model", {}).get("gguf") or {})
    state_now = sp.state_bytes(gguf, int(cand["ctx"]), kv_bytes)
    state_anchor_ctx = sp.state_bytes(gguf, 8192, kv_bytes)
    kv_gdn_now = state_now["kv_mib"] + state_now["gdn_mib"]
    kv_gdn_anchor = state_anchor_ctx["kv_mib"] + state_anchor_ctx["gdn_mib"]
    delta = kv_gdn_now - kv_gdn_anchor
    headroom = (total - RECORDED_PEAK_MIB) if total is not None else None
    if headroom is not None and delta > headroom:
        rejections.append({"stage": "local-vram-guard",
                           "config": {"context": cand["ctx"]},
                           "reason": (f"KV/GDN growth {round(delta,1)} MiB vs ctx8192 exceeds the "
                                      f"headroom {round(headroom,1)} MiB to the highest recorded 347 "
                                      f"device VRAM peak ({RECORDED_PEAK_MIB} MiB)")})

    host_ok = bool(plan["host"]["fits"]) if plan else False
    if plan and not host_ok:
        rejections.append({"stage": "host-pool", "config": {"slots": cand["slots"], "compact": cand["compact"]},
                           "reason": (f"host requirement {plan['host']['requirement_mib']} MiB exceeds the "
                                      f"pageable bound (commit limit minus OS reserve)")})
    return {
        "rejections": rejections, "plan": plan, "cache": cache,
        "informational": informational,
        "capacity": {
            "kv_gdn_mib": round(kv_gdn_now, 1),
            "kv_gdn_delta_vs_ctx8192_mib": round(delta, 1),
            "headroom_to_recorded_peak_mib": round(headroom, 1) if headroom is not None else None,
            "vram_static_leftover_mib": plan["vram"]["static_chain"]["leftover_mib"] if plan else None,
            "host_requirement_mib": plan["host"]["requirement_mib"] if plan else None,
            "host_commit_bound_mib": (plan["host"]["commit"]["limit_mib"] - plan["host"]["commit"]["os_reserve_mib"])
                                     if plan and plan["host"]["commit"]["limit_mib"] is not None else None,
            "host_fits": host_ok,
            "slot_cache_mib": cache["mib"],
        },
    }


def classify(cand: dict) -> str:
    anchored = (cand["slots"] in RECORDED_DIMS["slots"] and cand["compact"] in RECORDED_DIMS["compact"]
                and cand["threads"] in RECORDED_DIMS["threads"])
    if anchored:
        return "measured-anchor"
    if cand["threads"] not in RECORDED_DIMS["threads"]:
        return "cpu-pool-uncalibrated"
    return "miss-fraction-unknown"


def predict_candidate(cand: dict, shape_tokens: int, shape_rounds: int) -> dict:
    params = cm.fit(cm.RECORDED_CASES)
    if cand["threads"] != cm.CONFIG["threads"]:
        return cm.predict(params, shape_tokens, shape_rounds, misses=None, threads=cand["threads"])
    if cand["slots"] == 2700 and cand["compact"]:
        pred = cm.predict(params, shape_tokens, shape_rounds, misses=ANCHOR_MISSES_256)
        pred["evidence"] = ("miss fraction measured in the same batch (347 N2/N4 counters); the "
                            "prediction replays that measurement through the fitted parameters")
        return pred
    return cm.predict(params, shape_tokens, shape_rounds, misses=None)


def unverified_dims(cand: dict) -> list:
    out = []
    for dim, key in (("slots", "slots"), ("compact", "compact"), ("threads", "threads"),
                     ("ubatch", "ubatch"), ("ctx", "ctx")):
        if cand[key] not in RECORDED_DIMS[dim]:
            out.append(f"{dim}={cand[key]}")
    return out


def minimal_verification(cand: dict) -> dict:
    return {
        "binary": "same exe sha 03bd7068... for both arms (347 identity); a new build would need its own registered sha",
        "quantization": "same GGUF ab639a7f... (IQ3_S, 15,346,432,288 B); no cross-quant conversion",
        "input": L_CASE_INPUTS,
        "profile": "same profile fab192ff... (results/20261009-codex-b6d0bdef-capacity3700/calibration-expanded.profile)",
        "design": ("paired ABBA interleaved, one GPU owner, 2 requests/process, 256-token shape "
                   "(128-token shape if time allows; n>=2 natives per arm for a drift band)"),
        "holdout": "keep one process out of the refit (strata_cost_model holdout) when comparing to the model",
        "gate": ("EOS/JSON/tier counters + peak VRAM saved raw before the next case; existing drift "
                 "bands for reference: 346 2.35-5.29%, 358 4.248% (cross-batch reference only)"),
    }


def rank_key(c: dict):
    if c["status"] == "rejected":
        return (3, 0.0, 99, 99.0, 0, 0, c["id"])
    worst = c["prediction"]["tps"]["interval"][0] if c.get("prediction") and "tps" in c["prediction"] else 0.0
    return (0 if c["status"] == "ok" else (1 if c["status"] == "wide" else 2),
            -worst, len(c["unverified_dimensions"]), abs(c["config"]["slots"] - 2700),
            -c["config"]["ctx"], -c["config"]["ubatch"], c["id"])


SHORTLIST_QUESTIONS = [
    ("anchor", "re-baseline the measured point so later arms share its session drift",
     "同批重测 2700 COMPACT anchor，预期落在 347 记录的 256 形状 ms/token {36.65, 41.14}（四例区间 [32.10, 41.14]）；不假设拐点",
     "实测超出同形状已记录漂移带（346 2.35-5.29% / 358 4.248% 作为参照），或 EOS/JSON 守卫失败",
     {"slots": 2700, "compact": True, "threads": 8, "ubatch": 4096, "ctx": 8192}),
    ("slots-2200", "does the slot budget below 2700 lose measurable hit rate?",
     "若命中率损失换算的 TPS 降幅小于漂移带 => 槽预算在 <=2700 已饱和，2200 可省 0.59 GiB host 且不应更慢",
     "decode TPS 相对 anchor 的降幅超过已记录漂移带；或运行中 AvailPhys 触底/页文件代价可见",
     {"slots": 2200, "compact": True, "threads": 8, "ubatch": 4096, "ctx": 8192}),
    ("slots-3200", "does the slot budget above 2700 buy anything?",
     "若命中率增益低于漂移带 => '更多槽=更快' 被证伪，2700 已是拐点；增益超漂移带才支持更大预算（+0.59 GiB host）",
     "相对 anchor 无漂移带之外的增益，或 host 池余量恶化触发运行时压力",
     {"slots": 3200, "compact": True, "threads": 8, "ubatch": 4096, "ctx": 8192}),
    ("threads-16", "is the miss phase limited by the CPU pool size?",
     "若 tier_ms.misses/round 显著下降且 TPS 上移 => CPU 池是瓶颈；若 misses 相不变 => 瓶颈不在线程数",
     "decode TPS 因争用相对 anchor 下降超漂移带；或 misses 相下降但 TPS 不动（证伪线程假设）",
     {"slots": 2700, "compact": True, "threads": 16, "ubatch": 4096, "ctx": 8192}),
    ("compact-off", "does the uniform layout change anything but footprint?",
     "预期 hit/miss 计数与 COMPACT=on 相同（仅 host 足迹 +1337 MiB 量级）；此臂验证布局不改变语义",
     "同输入的 hits/misses 与 compact 臂不同（布局正确性问题），或 host 池不再适配",
     {"slots": 2700, "compact": False, "threads": 8, "ubatch": 4096, "ctx": 8192}),
    ("ubatch-1024", "prefill ubatch below the recorded 4096",
     "decode 预测不变（ubatch 不进 decode 模型）；拐点问题在 prefill 步数/压力，需 prefill t/s 实测",
     "decode 超漂移带，或 prefill_tokens_per_s 相对 347 的 750.9-1038.9 带明显恶化",
     {"slots": 2700, "compact": True, "threads": 8, "ubatch": 1024, "ctx": 8192}),
]


def build_shortlist(candidates: list, max_n: int) -> list:
    by_id = {c["id"]: c for c in candidates if c["status"] != "rejected"}
    picks, seen = [], set()
    for key, question, hypothesis, failure, wants in SHORTLIST_QUESTIONS:
        match = next((c for c in candidates if c["status"] != "rejected"
                      and all(c["config"][k] == v for k, v in wants.items())), None)
        if match is None or match["id"] in seen:
            continue
        seen.add(match["id"])
        picks.append({**match, "shortlist_key": key, "question": question, "rationale": question,
                      "expected_inflection_hypothesis": hypothesis, "failure_criteria": failure})
        if len(picks) >= max_n:
            break
    return picks


def screen(fp: dict, slots, compacts, threads, ubatches, contexts, shortlist_max,
           shape_tokens, shape_rounds, kv_bytes) -> dict:
    space = enumerate_space(slots, compacts, threads, ubatches, contexts)
    rows = []
    for cand in space:
        cid = candidate_id(cand)
        cc = check_capacity(fp, cand, kv_bytes=kv_bytes)
        if cc["rejections"]:
            rows.append({"id": cid, "config": cand, "status": "rejected",
                         "rejections": cc["rejections"],
                         "capacity": cc.get("capacity"),
                         "planner_informational": cc.get("informational") or None,
                         "evidence_class": None, "prediction": None,
                         "unverified_dimensions": unverified_dims(cand),
                         "minimal_verification": None})
            continue
        pred = predict_candidate(cand, shape_tokens, shape_rounds)
        status = ("ok" if pred.get("status") == "ok" else
                  "wide" if pred.get("status") == "wide" else "uncalibrated")
        rows.append({"id": cid, "config": cand, "status": status, "rejections": [],
                     "capacity": cc["capacity"],
                     "planner_informational": cc.get("informational") or None,
                     "evidence_class": classify(cand),
                     "prediction": pred, "unverified_dimensions": unverified_dims(cand),
                     "minimal_verification": minimal_verification(cand)})
    rows.sort(key=rank_key)
    shortlist = build_shortlist(rows, shortlist_max)
    vram_plan = next((r["capacity"] for r in rows if r["status"] != "rejected"), None)
    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "notice": ("capacity + diagnostic cost intervals only.  Predictions are NOT product "
                   "speeds; a slot count is never a speed prior.  No engine was run (0 GPU)."),
        "fingerprint": {
            "path": fp.get("_path"),
            "gpu_total_mib": sp.leaf(fp.get("gpu", {}), "vram_total_mib", default={}).get("value"),
            "model": sp.leaf(fp.get("model", {}), "path", default={}).get("value"),
            "model_bytes": sp.leaf(fp.get("model", {}), "size_bytes", default={}).get("value"),
        },
        "pools": {
            "vram": {"recorded_peak_mib": RECORDED_PEAK_MIB, "recorded_peak_source": RECORDED_PEAK_SOURCE,
                     "note": ("the static planner chain under-describes where the local engine actually "
                              "spends VRAM (recorded peak 7618.4 of 8188 MiB); candidates are guarded "
                              "against adding KV/GDN beyond the headroom to that peak"),
                     "example_leftover_mib": vram_plan["vram_static_leftover_mib"] if vram_plan else None},
            "host": {"slot_cache_records": SLOT_CACHE_RECORDS,
                     "note": ("slot cache is hosted by the pageable host arena (P3-1 doc section 2.3); "
                              "the planner's host pool counts the pagefile commit limit - the two pools "
                              "are never added together")},
        },
        "shape": {"tokens": shape_tokens, "rounds": shape_rounds, "note": SHAPE_NOTE},
        "cost_model": {"schema": cm.SCHEMA,
                       "configuration": cm.CONFIG,
                       "notice": "predictions replay the 347 batch calibration; see strata_cost_model.py"},
        "candidates": rows,
        "ranking": {
            "keys": ["status class (ok<wide<uncalibrated<rejected)",
                     "worst-case TPS (interval lower bound) desc",
                     "unverified dimensions asc", "|slots-2700| asc", "ctx desc", "ubatch desc", "id"],
            "order": [r["id"] for r in rows],
            "note": ("unknown candidates are ranked by their physical worst case, so a larger slot "
                     "count can never outrank a measured point without measurement"),
        },
        "shortlist": shortlist,
        "unexplained": [
            "miss fraction of any slot count other than 2700 COMPACT (no long-output routing trace in the evidence set)",
            "context effect on decode TPS (347 N3/N4 are non-monotonic, n=2, single batch)",
            "prefill ubatch effect on TTFT/prefill t/s (no recorded ubatch != 4096 for the L-case prompts)",
            "where the local engine's 7.6 GiB device VRAM peak is spent (static chain suggests ~3.5 GiB; not decomposed in the listed evidence)",
            "paging cost when AvailPhys bottoms out (14-367 MiB minima; pagefile absorbs, cost unquantified)",
            "the non_tier_ms_per_token residual composition (draft/verify GPU prefix/emit/sampler + overlap error)",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fingerprint", required=True, help="JSON from strata_fingerprint.py (P3-1)")
    ap.add_argument("--out")
    ap.add_argument("--slots", default="2200,2700,3200")
    ap.add_argument("--compacts", default="on,off")
    ap.add_argument("--threads", default="8,16")
    ap.add_argument("--ubatches", default="1024,2048,4096")
    ap.add_argument("--contexts", default="4096,8192")
    ap.add_argument("--shortlist-max", dest="shortlist_max", type=int, default=5)
    ap.add_argument("--shape-tokens", dest="shape_tokens", type=int, default=512)
    ap.add_argument("--shape-rounds", dest="shape_rounds", type=int, default=306)
    ap.add_argument("--kv-bytes", dest="kv_bytes", type=int, default=4, choices=(2, 4),
                    help="local engine keeps f32 device KV (P3-1 doc); 2 = official engine default")
    a = ap.parse_args()
    fp = json.loads(Path(a.fingerprint).read_text(encoding="utf-8"))
    fp["_path"] = a.fingerprint
    if a.shortlist_max < 1 or a.shortlist_max > 5:
        print("--shortlist-max must be within 1..5 (the task caps the shortlist at 5)", file=sys.stderr)
        sys.exit(2)
    slots = [int(x) for x in a.slots.split(",") if x.strip()]
    compacts = [x.strip() == "on" for x in a.compacts.split(",") if x.strip()]
    threads = [int(x) for x in a.threads.split(",") if x.strip()]
    ubatches = [int(x) for x in a.ubatches.split(",") if x.strip()]
    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]
    doc = screen(fp, slots, compacts, threads, ubatches, contexts, a.shortlist_max,
                 a.shape_tokens, a.shape_rounds, a.kv_bytes)
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)
    sys.exit(0)


if __name__ == "__main__":
    main()
