#!/usr/bin/env python3
"""strata_tune.py - `strata tune`: the P3-4 user-facing CLI over the phase-3 offline autotuner.

    python backends/qwen35_llama/tools/strata_tune.py inspect  --fingerprint fp.json [--json]
    python backends/qwen35_llama/tools/strata_tune.py screen   --fingerprint fp.json [--json]
    python backends/qwen35_llama/tools/strata_tune.py plan     --fingerprint fp.json [--json]
    python backends/qwen35_llama/tools/strata_tune.py profile  --fingerprint fp.json \
        --model-sha <prefix|sha256> --quant <q> --slots 2700 --ubatch 4096 --threads 8 \
        --ctx 8192 --compact on --confidence ok --evidence <path-or-commit> [...] \
        [--recommend] --out profile.json [--json]
    python backends/qwen35_llama/tools/strata_tune.py check    profile.json \
        --fingerprint fp.json --quant <q> --argv slots=2700,ubatch=4096,threads=8,ctx=8192,compact=on [--json]

Where `--fingerprint` is optional: when absent the device/model fingerprint is probed with
strata_fingerprint.py (P3-1); a saved fingerprint document makes every subcommand runnable with
no real machine at all (tests use synthetic documents).

Subcommands (all reuse the P3-1/P3-2 modules; nothing is reimplemented here):
  inspect  device/model fingerprint + the two-pool (VRAM / host) capacity budget (P3-1 plan).
  screen   candidate screen over the P3-2 space; prints each candidate's class
           (ok/wide/uncalibrated/rejected) and the <=5 shortlist.  The offline model can only
           credibly REJECT, never select a winner - the command always prints that positioning
           plus the measured hold-out error (mean 9.70% / max 21.30%).
  plan     pre-registered GPU verification plan for the shortlist (shapes, arms, interleave,
           decision rule, window and lease requirements).  Planning only - nothing is executed,
           no lease is taken, no engine is started.
  profile  write a profile JSON for a configuration that evidence supports.  A configuration
           without evidence can only be written as confidence=uncalibrated and can never be
           written as a recommendation.
  check    compare a profile against the current environment (model SHA / quant / device /
           argv) and fail closed with a non-zero exit code on any invalidation.

Profile invalidation (any one fails `check`):
  model_sha256_prefix mismatch; quant mismatch; device_fingerprint mismatch (GPU name / VRAM /
  driver / compute capability); argv mismatch (slots/ubatch/threads/ctx/COMPACT); confidence
  uncalibrated -> warning code W-UNCALIBRATED (still not a recommendation).  Cross-card and
  cross-quantization reuse is refused and no conversion is ever performed.

Exit codes (stable, asserted by test_strata_tune.py):
  0  success / `check` passed (identities match, confidence ok|wide)
  2  usage error, missing/invalid input, or a refused operation / `check` invalidation failure
  3  `check` passed the identity comparison but the profile is confidence=uncalibrated
     (explicit warning code W-UNCALIBRATED; the profile must not be used as a recommendation)

0 GPU: never starts an engine, never takes a GPU lease, never builds.  The only external
read any subcommand can make is the same read-only nvidia-smi / GGUF-header probe P3-1
already performs when no --fingerprint document is supplied.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import strata_cost_model as cm  # noqa: E402
import strata_fingerprint as fg  # noqa: E402
import strata_plan as sp  # noqa: E402
import strata_screen as ss  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_WARN_UNCALIBRATED = 3

PROFILE_SCHEMA = "strata-tune-profile/1"
SCREEN_POSITION = "只能可信拒绝、不足以选优"
_SHA_PREFIX_RE = re.compile(r"^[0-9a-f]{8,64}$")
_DEVICE_KEYS = ("gpu_name", "vram_total_mib", "driver_version", "compute_capability")

INVALIDATORS = [
    {"code": "MODEL_SHA_MISMATCH",
     "rule": "model_sha256_prefix 与当前模型的注册 SHA 前缀不匹配即失效（当前 SHA 未知时 fail closed）"},
    {"code": "QUANT_MISMATCH",
     "rule": "quant 与当前量化不一致即失效；禁止任何跨量化换算"},
    {"code": "DEVICE_MISMATCH",
     "rule": "device_fingerprint 的 GPU 名/显存/驱动/计算能力任一变化即失效；明确拒绝跨卡套用，不做任何换算"},
    {"code": "ARGV_MISMATCH",
     "rule": "slots/ubatch/threads/ctx/COMPACT 与运行配置不一致即失效；不静默沿用"},
    {"code": "UNCALIBRATED",
     "rule": "confidence=uncalibrated 的 profile 不是推荐；check 打印警告码 W-UNCALIBRATED（退出码 3）"},
]


class TuneUsage(Exception):
    """A refused operation or invalid input; the message is printed and exit code 2 is returned."""


def _leaf(node) -> object:
    return node.get("value") if isinstance(node, dict) else node


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_fp(a) -> dict:
    """A fingerprint document: --fingerprint FILE verbatim, else probed via P3-1 (read-only)."""
    path = getattr(a, "fingerprint", None)
    if path:
        if not os.path.isfile(path):
            raise TuneUsage(f"no such fingerprint file: {path}")
        try:
            fp = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TuneUsage(f"fingerprint {path} is not readable JSON ({exc})")
        fp.setdefault("_path", path)
        return fp
    model = getattr(a, "model", None)
    if not model:
        raise TuneUsage("either --fingerprint FILE or --model MODEL.gguf is required")
    if not os.path.isfile(model):
        raise TuneUsage(f"no such model file: {model}")
    workload = {"context": getattr(a, "context", None), "max_new": getattr(a, "max_new", None),
                "batch": 1}
    fp = fg.build_fingerprint(model, workload, getattr(a, "sha", None),
                              getattr(a, "gpu_index", 0) or 0)
    fp["_path"] = model
    return fp


def _device_from_fp(fp: dict) -> dict:
    gpu = fp.get("gpu", {})
    return {"gpu_name": _leaf(gpu.get("name")),
            "vram_total_mib": _leaf(gpu.get("vram_total_mib")),
            "driver_version": _leaf(gpu.get("driver_version")),
            "compute_capability": _leaf(gpu.get("compute_capability"))}


def _parse_kv_list(text: str) -> dict:
    """'slots=2700,ubatch=4096,compact=on' -> {'slots': 2700, ...} (ints/bools auto-parsed)."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise TuneUsage(f"bad --argv item {part!r}: expected key=value")
        key, value = part.split("=", 1)
        key, value = key.strip(), value.strip()
        if value.lower() in ("on", "true"):
            parsed = True
        elif value.lower() in ("off", "false"):
            parsed = False
        elif re.fullmatch(r"-?\d+", value):
            parsed = int(value)
        else:
            parsed = value
        out[key] = parsed
    return out


def _emit_json(doc: dict) -> None:
    sys.stdout.write(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# --------------------------------------------------------------------------------------- inspect

def cmd_inspect(a) -> int:
    fp = _load_fp(a)
    opts = {"context": getattr(a, "context", None), "max_new": getattr(a, "max_new", None),
            "batch": 1, "mode": "safe", "kv_bytes": a.kv_bytes}
    plan = sp.build_plan(fp, opts)
    gpu = {k: _leaf(fp.get("gpu", {}).get(k))
           for k in ("name", "compute_capability", "vram_total_mib", "vram_free_mib", "driver_version")}
    gguf = fp.get("model", {}).get("gguf", {})
    doc = {
        "schema": "strata-tune-inspect/1",
        "gpu": gpu,
        "model": {"path": _leaf(fp.get("model", {}).get("path")),
                  "size_bytes": _leaf(fp.get("model", {}).get("size_bytes")),
                  "sha256": _leaf(fp.get("model", {}).get("sha256")),
                  "block_count": _leaf(gguf.get("block_count")),
                  "expert_count": _leaf(gguf.get("expert_count")),
                  "expert_used_count": _leaf(gguf.get("expert_used_count"))},
        "workload": {"context": _leaf(fp.get("workload", {}).get("context")),
                     "max_new": _leaf(fp.get("workload", {}).get("max_new")),
                     "batch": _leaf(fp.get("workload", {}).get("batch"))},
        "pools": {
            "vram": {"mode": plan["mode"], "status": plan["status"],
                     "effective_leftover_mib": plan["vram"]["effective"]["leftover_mib"],
                     "effective_source": plan["vram"]["effective"]["source"],
                     "quota_per_layer": plan["vram"]["expert_slot"]["quota_per_layer"],
                     "cache_gib": plan["vram"]["expert_slot"]["cache_gib"]},
            "host": {"mode": plan["host"]["mode"], "requirement_mib": plan["host"]["requirement_mib"],
                     "fits": plan["host"]["fits"]},
        },
        "plan_candidates": [c["name"] for c in plan.get("candidates", [])],
        "plan_rejections": [r.get("reason") for r in plan.get("rejected", [])],
        "notice": "fingerprint + capacity budget only (P3-1); no speed claim, no engine run",
    }
    if a.json:
        _emit_json(doc)
        return EXIT_OK
    g, m, p, h = doc["gpu"], doc["model"], doc["pools"]["vram"], doc["pools"]["host"]
    print(f"device : {_fmt(g['name'])} | {_fmt(g['vram_total_mib'])} MiB total / "
          f"{_fmt(g['vram_free_mib'])} MiB free | driver {_fmt(g['driver_version'])} | "
          f"cc {_fmt(g['compute_capability'])}")
    print(f"model  : {_fmt(m['path'])} | {_fmt(m['size_bytes'])} B | sha {_fmt(m['sha256'])} | "
          f"{_fmt(m['block_count'])} blocks | experts {_fmt(m['expert_count'])}x"
          f"{_fmt(m['expert_used_count'])}")
    print(f"vram   : mode={p['mode']} status={p['status']} leftover={_fmt(p['effective_leftover_mib'])} MiB "
          f"({p['effective_source']}) quota={_fmt(p['quota_per_layer'])}/layer cache={_fmt(p['cache_gib'])} GiB")
    print(f"host   : mode={h['mode']} requirement={_fmt(h['requirement_mib'])} MiB fits={h['fits']}")
    print(f"plan   : candidates={doc['plan_candidates']} rejections={doc['plan_rejections']}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------- screen

def _space_from(a) -> dict:
    return {"slots": [int(x) for x in a.slots.split(",") if x.strip()],
            "compacts": [x.strip() == "on" for x in a.compacts.split(",") if x.strip()],
            "threads": [int(x) for x in a.threads.split(",") if x.strip()],
            "ubatches": [int(x) for x in a.ubatches.split(",") if x.strip()],
            "contexts": [int(x) for x in a.contexts.split(",") if x.strip()]}


def _run_screen(a) -> tuple:
    if a.shortlist_max < 1 or a.shortlist_max > 5:
        raise TuneUsage("--shortlist-max must be within 1..5 (the task caps the shortlist at 5)")
    fp = _load_fp(a)
    space = _space_from(a)
    doc = ss.screen(fp, space["slots"], space["compacts"], space["threads"], space["ubatches"],
                    space["contexts"], a.shortlist_max, a.shape_tokens, a.shape_rounds, a.kv_bytes)
    return fp, doc


def _positioning() -> dict:
    ho = cm.holdout(cm.RECORDED_CASES)
    return {
        "statement": SCREEN_POSITION,
        "holdout_mean_abs_error_pct": ho["mean_abs_central_error_pct"],
        "holdout_max_abs_error_pct": ho["max_abs_central_error_pct"],
        "holdout_interval_coverage": ho["interval_coverage"],
        "note": ("离线成本模型对候选只能可信拒绝（reject）并按区间排序；hold-out 误差如下，"
                 "screening 结果不足以选优（select a winner），不得当作推荐或产品速度。"),
    }


def cmd_screen(a) -> int:
    _, doc = _run_screen(a)
    counts = {}
    for row in doc["candidates"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    out = {
        "schema": "strata-tune-screen/1",
        "positioning": _positioning(),
        "classification": {k: counts.get(k, 0) for k in ("ok", "wide", "uncalibrated", "rejected")},
        "candidates": [{"id": r["id"], "status": r["status"], "evidence_class": r["evidence_class"],
                        "unverified_dimensions": r["unverified_dimensions"]}
                       for r in doc["candidates"]],
        "shortlist": [{"shortlist_key": s["shortlist_key"], "id": s["id"], "config": s["config"],
                       "question": s["question"], "expected_inflection_hypothesis": s["expected_inflection_hypothesis"],
                       "failure_criteria": s["failure_criteria"]} for s in doc["shortlist"]],
        "notice": ("deterministic wrapper over strata_screen.py (P3-2) + strata_cost_model.py; "
                   "0 GPU, no engine, no lease"),
    }
    if a.json:
        _emit_json(out)
        return EXIT_OK
    pos = out["positioning"]
    print(f"定位：{pos['statement']}（hold-out 均值绝对误差 {pos['holdout_mean_abs_error_pct']:.2f}% / "
          f"最大 {pos['holdout_max_abs_error_pct']:.2f}%，覆盖率 {pos['holdout_interval_coverage']}；"
          f"离线模型只能可信拒绝，不能选优）")
    c = out["classification"]
    print(f"classification: ok={c['ok']} wide={c['wide']} uncalibrated={c['uncalibrated']} rejected={c['rejected']}")
    for row in out["candidates"]:
        print(f"  {row['id']:<28} {row['status']:<13} {str(row['evidence_class']):<24} "
              f"unverified={len(row['unverified_dimensions'])}")
    print(f"shortlist ({len(out['shortlist'])}):")
    for i, s in enumerate(out["shortlist"], 1):
        print(f"  #{i} {s['shortlist_key']:<12} {s['id']:<28} {s['question']}")
    return EXIT_OK


# ------------------------------------------------------------------------------------------ plan

def build_verification_plan(fp: dict, shortlist: list, shape_tokens: int, shape_rounds: int) -> dict:
    """The pre-registered verification design, derived from the P3-3 pilot's registered protocol."""
    return {
        "schema": "strata-tune-plan/1",
        "notice": ("预登记 GPU 验证计划（planning artifact）。本命令不执行任何 native、不申请租约、"
                   "不启动引擎；执行必须由受监督实验按本清单冻结预登记后进行。"),
        "executed": False,
        "fingerprint": {"gpu": _leaf(fp.get("gpu", {}).get("name")),
                        "model": _leaf(fp.get("model", {}).get("path")),
                        "model_sha256": _leaf(fp.get("model", {}).get("sha256"))},
        "shapes": [
            {"id": "S", "case": "fixture-benign 3896→16", "ctx": 4096,
             "tokens": shape_tokens, "rounds": shape_rounds,
             "note": "短形状 16 可见 token；P3-3 实测 finish=length/eos=0（以原文为准）"},
            {"id": "L", "case": "l-risk-explain 3817→128", "ctx": 8192,
             "tokens": shape_tokens, "rounds": shape_rounds,
             "note": "长形状 128 token 长输出"},
        ],
        "anchor": dict(ss.ANCHOR),
        "candidates": [{"shortlist_key": s["shortlist_key"], "id": s["id"], "config": s["config"],
                        "question": s["question"], "failure_criteria": s["failure_criteria"]}
                       for s in shortlist],
        "design": {
            "per_candidate_interleave": ["S-A1", "S-B1", "S-A2", "L-A1", "L-B1", "L-A2"],
            "arms": {"A": "anchor（2700 COMPACT，见 anchor 字段）", "B": "候选（shortlist 条目）"},
            "natives_per_candidate": 6,
            "serial": "严格串行，一次一个 native；无并行、无重跑",
            "cap_note": ("P3-3 登记的 6-native 硬帽覆盖单个候选；若要验证整个短名单，须在启动前"
                         "显式登记扩展预算，不得事后加臂"),
        },
        "decision_rule": {
            "sample": "warm（同进程第 2 请求 rq2）；冷样（rq1）只记录不判定",
            "statistic": "每条候选与基线的 warm TPS 取中位；Δ＝候选−基线",
            "drift_band": "臂内极差最大值",
            "PASS": "Δ≥+5% 且 Δ>漂移带 且配对方向一致",
            "NO-GO": "Δ≤0 或 |Δ|≤漂移带",
            "otherwise": "INCONCLUSIVE",
            "source": "P3-3 预登记（retry2/prep/preregistration.json，native 之前冻结）",
        },
        "window_requirements": [
            "以 task.md mtime 为锚的总续步窗口（started_at 不重置）；每个 native 启动前程序化检查剩余秒数并记录进 argv-plan.json（P3-3 实际 512→396 s）",
            "剩余窗口不足以在本步上限内完成时不得启动新 native（P3-3 批末 219 s 即停）",
            "native 启动前清空 STRATA*/GGML*/LLAMA_GRAPH*/CUDA_LAUNCH_BLOCKING 再设 preregistration.json 中的 exact env",
            "失败留原件；不自动重试、不用新编号洗预算；额外修复须主管显式登记",
        ],
        "lease_requirements": [
            "单 GPU 唯一 owner：每 native 一次 tools/gpu-lock.py run（declared binary＝engine exe，--expect-job，exeSHA 绑定），严格串行",
            "租约有限等待 ≤60 s 且计入窗口；忙则允许轻只读或 checkpoint，不轮询",
            "异常 cleanup 围栏保留；禁 agent 自释放/绕过/自动过期；仅主管核实全子树零且 owner 退出后 release",
            "每 native 落盘 audit/result/ready/gpu-lock-run.log/raw/child log/argv-plan.json；批前/批后记录 RAM/GPU 用量与进程残留",
            "engine 参数与导出目录用 ASCII 相对路径＋CreateProcessW 宽 cwd；不得传绝对中文 engine argv；CUDA DLLPATH 须 prepend runtime/cuda-12.8.1/bin 并记录",
        ],
        "preregistration": ("第一个 native 之前把形状/臂/交错顺序/判定规则/窗口与租约要求冻结为 "
                            "preregistration.json；native 之后不得修改判定。"),
    }


def cmd_plan(a) -> int:
    fp, doc = _run_screen(a)
    plan = build_verification_plan(fp, doc["shortlist"], a.shape_tokens, a.shape_rounds)
    if a.json:
        _emit_json(plan)
        return EXIT_OK
    print(f"plan: {len(plan['candidates'])} candidate(s) from the shortlist; executed={plan['executed']} "
          f"(planning only, no native/lease/engine)")
    for c in plan["candidates"]:
        print(f"  {c['shortlist_key']:<12} {c['id']:<28} {c['question']}")
    print(f"  interleave: {' '.join(plan['design']['per_candidate_interleave'])} (per candidate; "
          f"{plan['design']['natives_per_candidate']} natives)")
    dr = plan["decision_rule"]
    print(f"  decision: PASS={dr['PASS']} | NO-GO={dr['NO-GO']} | else={dr['otherwise']}")
    print(f"  window: {plan['window_requirements'][0]}")
    print(f"  lease : {plan['lease_requirements'][0]}")
    print(f"  preregistration: {plan['preregistration']}")
    return EXIT_OK


# --------------------------------------------------------------------------------------- profile

def cmd_profile(a) -> int:
    fp = _load_fp(a)
    device = _device_from_fp(fp)
    missing = [k for k in _DEVICE_KEYS if device[k] is None]
    if missing:
        raise TuneUsage(f"device_fingerprint incomplete ({', '.join(missing)} unknown in the "
                        f"fingerprint); refusing to write a profile against an unidentified device")
    sha = (a.model_sha or "").strip().lower()
    if not _SHA_PREFIX_RE.fullmatch(sha):
        raise TuneUsage("--model-sha must be a lowercase hex prefix/prefix-or-full SHA (8..64 hex chars)")
    evidence = [e for e in (a.evidence or []) if e.strip()]
    if not evidence and a.confidence != "uncalibrated":
        raise TuneUsage(f"no evidence supplied: only confidence=uncalibrated may be written "
                        f"(got {a.confidence}); 无证据的配置只能标 uncalibrated")
    if a.recommend and (a.confidence != "ok" or not evidence):
        raise TuneUsage(f"refusing to write a recommendation: confidence={a.confidence}, "
                        f"evidence={len(evidence)} item(s); a recommendation needs confidence=ok "
                        f"and at least one evidence item")
    argv = {"slots": a.slots, "ubatch": a.ubatch, "threads": a.threads, "ctx": a.ctx,
            "compact": a.compact == "on"}
    doc = {
        "schema_version": PROFILE_SCHEMA,
        "model_sha256_prefix": sha,
        "quant": a.quant,
        "device_fingerprint": device,
        "argv": argv,
        "evidence": evidence,
        "confidence": a.confidence,
        "recommendation": bool(a.recommend),
        "generated_at": _now(),
        "invalidators": INVALIDATORS,
        "notes": ("evidence-backed configuration for the registered local host only; the CLI never "
                  "auto-applies it; `strata_tune.py check` re-validates it against the live environment"),
    }
    out = Path(a.out)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.json:
        _emit_json(doc)
    else:
        print(f"wrote {out}")
        print(f"  confidence={doc['confidence']} recommendation={doc['recommendation']} "
              f"evidence={len(doc['evidence'])} item(s)")
        if not doc["recommendation"]:
            print("  不是推荐（not a recommendation: only evidence + confidence=ok + --recommend is one）")
    return EXIT_OK


# ----------------------------------------------------------------------------------------- check

def cmd_check(a) -> int:
    ppath = Path(a.profile)
    if not ppath.is_file():
        raise TuneUsage(f"no such profile file: {ppath}")
    try:
        prof = json.loads(ppath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuneUsage(f"profile {ppath} is not readable JSON ({exc})")
    if not isinstance(prof, dict) or prof.get("schema_version") != PROFILE_SCHEMA:
        raise TuneUsage(f"{ppath} is not a {PROFILE_SCHEMA} profile "
                        f"(schema_version={prof.get('schema_version') if isinstance(prof, dict) else None!r})")
    for key in ("model_sha256_prefix", "quant", "device_fingerprint", "argv", "confidence"):
        if key not in prof:
            raise TuneUsage(f"profile {ppath} misses required field {key!r}")

    fp = _load_fp(a)
    current_sha = _leaf(fp.get("model", {}).get("sha256"))
    current_dev = _device_from_fp(fp)
    current_argv = _parse_kv_list(a.argv)

    mismatches = []
    stored_prefix = str(prof["model_sha256_prefix"]).lower()
    if not current_sha:
        mismatches.append({"code": "MODEL_SHA_MISMATCH",
                           "detail": "当前 fingerprint 无注册 SHA（--sha/--fingerprint 未含）→ fail closed，"
                                     "无法确认 model_sha256_prefix",
                           "stored": stored_prefix, "current": None})
    elif not str(current_sha).lower().startswith(stored_prefix):
        mismatches.append({"code": "MODEL_SHA_MISMATCH", "stored": stored_prefix,
                           "current": str(current_sha).lower(),
                           "detail": "model_sha256_prefix 不匹配；不做任何换算"})
    if a.quant != prof["quant"]:
        mismatches.append({"code": "QUANT_MISMATCH", "stored": prof["quant"], "current": a.quant,
                           "detail": "quant 不匹配；禁止跨量化换算（no cross-quantization conversion）"})
    dev_diffs = {}
    for key in _DEVICE_KEYS:
        stored, current = prof["device_fingerprint"].get(key), current_dev.get(key)
        if stored != current:
            dev_diffs[key] = {"stored": stored, "current": current}
    if dev_diffs:
        mismatches.append({"code": "DEVICE_MISMATCH", "fields": dev_diffs,
                           "stored": prof["device_fingerprint"], "current": current_dev,
                           "detail": ("设备指纹变化：拒绝把该 profile 用于当前设备（跨卡/跨驱动套用一律拒绝，"
                                      "不做任何换算）")})
    argv_diffs = {}
    for key, stored in prof["argv"].items():
        current = current_argv.get(key, "<missing>")
        if stored != current:
            argv_diffs[key] = {"stored": stored, "current": current}
    if argv_diffs:
        mismatches.append({"code": "ARGV_MISMATCH", "fields": argv_diffs,
                           "stored": prof["argv"], "current": current_argv,
                           "detail": "argv 与运行配置不一致；不静默沿用"})

    warnings = []
    if prof.get("confidence") == "uncalibrated":
        warnings.append({"code": "W-UNCALIBRATED",
                         "detail": "confidence=uncalibrated：该 profile 不是推荐，不得当作选优依据"})

    doc = {"schema": "strata-tune-check/1", "profile": str(ppath),
           "status": "invalidated" if mismatches else "ok",
           "mismatches": mismatches, "warnings": warnings,
           "stored": {"model_sha256_prefix": stored_prefix, "quant": prof["quant"],
                      "device_fingerprint": prof["device_fingerprint"], "argv": prof["argv"]},
           "current": {"model_sha256": current_sha, "quant": a.quant,
                       "device_fingerprint": current_dev, "argv": current_argv}}
    if a.json:
        _emit_json(doc)
    else:
        if mismatches:
            print(f"check FAILED: {len(mismatches)} invalidation(s) for {ppath}")
            for m in mismatches:
                print(f"  {m['code']}: {m['detail']}")
                if m.get("stored") is not None:
                    print(f"    stored : {json.dumps(m['stored'], ensure_ascii=False)}")
                if m.get("current") is not None:
                    print(f"    current: {json.dumps(m['current'], ensure_ascii=False)}")
        elif warnings:
            print(f"check: {ppath} identities match (confidence={prof.get('confidence')}) "
                  f"- warning code(s) below; NOT a recommendation")
        else:
            print(f"check OK: {ppath} matches the current model/quant/device/argv "
                  f"(confidence={prof.get('confidence')})")
        for w in warnings:
            print(f"  {w['code']}: {w['detail']}")
    if mismatches:
        return EXIT_REFUSED
    if warnings:
        return EXIT_WARN_UNCALIBRATED
    return EXIT_OK


# ------------------------------------------------------------------------------------------- CLI

def _add_fp_args(sp_, with_workload: bool) -> None:
    sp_.add_argument("--fingerprint", help="a P3-1 fingerprint JSON; when given, nothing is probed")
    sp_.add_argument("--model", help="a GGUF file to fingerprint (header only) when --fingerprint is absent")
    sp_.add_argument("--sha", help="a registered model sha256 for the probe (recorded, never recomputed)")
    sp_.add_argument("--gpu-index", dest="gpu_index", type=int, default=0)
    if with_workload:
        sp_.add_argument("--context", type=int)
        sp_.add_argument("--max-new", type=int, dest="max_new")


def _add_space_args(sp_) -> None:
    sp_.add_argument("--slots", default="2200,2700,3200")
    sp_.add_argument("--compacts", default="on,off")
    sp_.add_argument("--threads", default="8,16")
    sp_.add_argument("--ubatches", default="1024,2048,4096")
    sp_.add_argument("--contexts", default="4096,8192")
    sp_.add_argument("--shortlist-max", dest="shortlist_max", type=int, default=5)
    sp_.add_argument("--shape-tokens", dest="shape_tokens", type=int, default=512)
    sp_.add_argument("--shape-rounds", dest="shape_rounds", type=int, default=306)
    sp_.add_argument("--kv-bytes", dest="kv_bytes", type=int, default=4, choices=(2, 4))


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strata_tune.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("inspect", help="device/model fingerprint + two-pool budget (P3-1)")
    _add_fp_args(p, with_workload=True)
    p.add_argument("--kv-bytes", dest="kv_bytes", type=int, default=2, choices=(2, 4))
    p.add_argument("--json", action="store_true", help="emit one JSON document")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("screen", help="P3-2 candidate screen (can only credibly reject)")
    _add_fp_args(p, with_workload=False)
    _add_space_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("plan", help="pre-registered GPU verification plan for the shortlist (not executed)")
    _add_fp_args(p, with_workload=False)
    _add_space_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("profile", help="write an evidence-backed profile JSON")
    _add_fp_args(p, with_workload=False)
    p.add_argument("--model-sha", dest="model_sha", required=True,
                   help="model sha256 prefix (8..64 lowercase hex; a registered value, never recomputed)")
    p.add_argument("--quant", required=True, help="the model's quantization tag (e.g. IQ3_S)")
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--ubatch", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--compact", required=True, choices=("on", "off"))
    p.add_argument("--confidence", required=True, choices=("ok", "wide", "uncalibrated"))
    p.add_argument("--evidence", action="append", default=[],
                   help="an evidence directory/commit reference; repeatable; required unless uncalibrated")
    p.add_argument("--recommend", action="store_true",
                   help="mark the profile as a recommendation (needs confidence=ok + evidence)")
    p.add_argument("--out", required=True, help="profile output path (JSON)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("check", help="validate a profile against the current environment")
    p.add_argument("profile", help="the profile JSON from `profile`")
    _add_fp_args(p, with_workload=False)
    p.add_argument("--quant", required=True, help="the quantization actually in use")
    p.add_argument("--argv", required=True,
                   help="the runtime configuration actually in use, e.g. "
                        "slots=2700,ubatch=4096,threads=8,ctx=8192,compact=on")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    return ap


def main(argv=None) -> int:
    ap = _parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_REFUSED
    try:
        return a.func(a)
    except TuneUsage as exc:
        print(f"strata-tune: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
