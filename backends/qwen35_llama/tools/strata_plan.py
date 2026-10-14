#!/usr/bin/env python3
"""strata_plan.py - resource budget planner for the P3 autotuner foundation (P3-1, 0 GPU).

    python backends/qwen35_llama/tools/strata_plan.py --fingerprint fp.json \
        [--context 4096] [--max-new 128] [--batch 1] \
        [--mode safe|balanced|latency|throughput] \
        [--safety-mib M] [--staging-mib M] [--workspace-mib M] [--kv-bytes 2|4] \
        [--free-mib X] [--host-mode pageable|pinned] [--pinned-cap-mib M] [--out plan.json]

Takes the JSON of strata_fingerprint.py plus a workload and writes a capacity plan:
two SEPARATE pools (VRAM and host RAM, never added together), the VRAM chain in the engine's
own order - non-expert weights -> KV/GDN state -> workspace/graph buffers -> prefill staging ->
safety margin -> expert slots - ranked candidates with reasons and risks, and explicit
rejections for out-of-bounds combinations.  Modes only choose the allocation strategy of the
leftover; they do not promise speed.  This tool measures nothing and runs no engine.

--free-mib models the engine's runtime clamp: the official engine decides the expert-slot
budget from the free VRAM at placement time minus staging minus the safety margin (recorded:
free 2.07 GiB, staging 1.10 GiB, margin 512 MiB -> 6 of 256 experts per layer, 0.44 GiB,
tasks 341/342 on this host).  Without --free-mib the plan reports the static upper bound and
the difference is the engine's own runtime reservation, which a GGUF cannot describe.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

MIB = 1 << 20
MODES = {
    "safe":       {"safety_frac": 0.06, "staging_mib": 1126, "staging_source": "table", "preference": ("expert_max", "expert_balanced", "threads_headroom", "ubatch_light", "expert_min", "staging_boost")},
    "balanced":   {"safety_frac": 0.04, "staging_mib": 1126, "staging_source": "table", "preference": ("expert_balanced", "expert_max", "threads_headroom", "ubatch_light", "expert_min", "staging_boost")},
    "latency":    {"safety_frac": 0.06, "staging_mib": 512,  "staging_source": "policy", "preference": ("expert_max", "threads_headroom", "ubatch_light", "expert_balanced", "expert_min", "staging_boost")},
    "throughput": {"safety_frac": 0.06, "staging_mib": 1536, "staging_source": "policy", "preference": ("staging_boost", "ubatch_light", "expert_max", "expert_balanced", "threads_headroom", "expert_min")},
}


def mib(nbytes: float) -> float:
    return nbytes / MIB


def leaf(doc: dict, *keys, default=None):
    node = doc
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if isinstance(node, dict) and "value" in node else default


def need(leaf_dict, what: str) -> object:
    if leaf_dict is None or leaf_dict.get("value") is None:
        raise ValueError(f"fingerprint lacks {what} (source unknown); refusing to plan on a guessed value")
    return leaf_dict["value"]


def state_bytes(model: dict, context: int, kv_bytes: int) -> dict:
    """KV + GDN state from GGUF metadata.  KV: ctx x kv-heads x (k_len+v_len) x dtype x full-attention
    layers.  GDN: per recurrent layer, state_size^2 * time_step_rank * 4 B (f32) + conv_channels *
    (conv_kernel-1) * 4 B, conv_channels = 2*state_size*group_count + inner_size (the engine's own
    geometry, src/core/session.cpp and include/strata/core/family_layout.hpp)."""
    block = int(need(leaf(model, "block_count"), "model.gguf.block_count"))
    interval = int(need(leaf(model, "full_attention_interval"), "model.gguf.full_attention_interval"))
    kvh = int(need(leaf(model, "attention", "head_count_kv"), "model.gguf.attention.head_count_kv"))
    klen = int(need(leaf(model, "attention", "key_length"), "model.gguf.attention.key_length"))
    vlen = int(need(leaf(model, "attention", "value_length"), "model.gguf.attention.value_length"))
    ssm = model.get("ssm", {})
    d_state = int(need(leaf(ssm, "state_size"), "model.gguf.ssm.state_size"))
    k_heads = int(need(leaf(ssm, "group_count"), "model.gguf.ssm.group_count"))
    v_heads = int(need(leaf(ssm, "time_step_rank"), "model.gguf.ssm.time_step_rank"))
    conv = int(need(leaf(ssm, "conv_kernel"), "model.gguf.ssm.conv_kernel"))
    inner = int(need(leaf(ssm, "inner_size"), "model.gguf.ssm.inner_size"))
    n_attn = (block + interval - 1) // interval
    n_rec = block - n_attn
    kv_total = n_attn * context * kvh * (klen + vlen) * kv_bytes
    conv_channels = 2 * d_state * k_heads + inner
    gdn_per_layer = d_state * d_state * v_heads * 4 + conv_channels * (conv - 1) * 4
    return {
        "attn_layers": n_attn, "recurrent_layers": n_rec,
        "kv_bytes_per_token_per_layer": kvh * (klen + vlen) * kv_bytes,
        "kv_mib": mib(kv_total), "gdn_mib": mib(n_rec * gdn_per_layer),
        "gdn_per_layer_mib": mib(gdn_per_layer), "conv_channels": conv_channels,
        "kv_bytes": kv_bytes,
        "formula": ("kv = attn_layers x ctx x head_count_kv x (key_length+value_length) x dtype; "
                    "gdn = recurrent_layers x (state_size^2 x time_step_rank x 4 + conv_channels x (conv_kernel-1) x 4)"),
    }


def expert_step_mib(expert_total_bytes: int, expert_count: int) -> float:
    """One expert of one layer, averaged over the routed-expert layers (bytes per quota step)."""
    return mib(expert_total_bytes / expert_count)


def quota_for(leftover_mib: float, step_mib: float, expert_count: int) -> int:
    if leftover_mib <= 0 or step_mib <= 0:
        return 0
    return max(0, min(expert_count, int(math.floor(leftover_mib / step_mib))))


def _candidate(name, quota, staging, safety, threads, ubatch, expert_layers, step, reason, risk):
    return {"name": name,
            "config": {"expert_quota_per_layer": quota,
                       "expert_slots_total": quota * expert_layers,
                       "expert_cache_mib": round(quota * step, 1),
                       "staging_mib": staging, "safety_mib": round(safety, 1),
                       "cpu_threads": threads, "prefill_ubatch": ubatch},
            "reason": reason, "risk": risk, "fits": True}


def build_plan(fp: dict, opts: dict) -> dict:
    gpu = fp.get("gpu", {})
    model_sec = fp.get("model", {})
    gguf = model_sec.get("gguf", {}) if isinstance(model_sec.get("gguf"), dict) else {}
    if "error" in gguf:
        raise ValueError(f"model GGUF unreadable: {gguf['error'].get('value')}")
    workload = fp.get("workload", {})

    ctx = opts.get("context")
    if ctx is None:
        ctx = leaf(workload, "context", default={}).get("value")
    max_new = opts.get("max_new")
    if max_new is None:
        max_new = leaf(workload, "max_new", default={}).get("value")
    batch = opts.get("batch")
    if batch is None:
        batch = leaf(workload, "batch", default={}).get("value") or 1

    context_max = need(leaf(gguf, "context_length"), "model.gguf.context_length")
    if ctx is None:
        raise ValueError("no workload context given (neither --context nor fingerprint.workload.context)")

    mode = opts.get("mode", "safe")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    mconf = MODES[mode]

    vram_total = need(leaf(gpu, "vram_total_mib"), "gpu.vram_total_mib")
    vram_free_leaf = leaf(gpu, "vram_free_mib")
    vram_free = vram_free_leaf.get("value") if vram_free_leaf else None
    desktop_used = (vram_total - vram_free) if vram_free is not None else 0.0

    kv_bytes = int(opts.get("kv_bytes", 2))
    st = state_bytes(gguf, int(ctx), kv_bytes)
    expert_total = int(need(leaf(gguf, "routed_expert_bytes_total"), "model.gguf.routed_expert_bytes_total"))
    non_expert = int(need(leaf(gguf, "non_expert_bytes_total"), "model.gguf.non_expert_bytes_total"))
    expert_count = int(need(leaf(gguf, "expert_count"), "model.gguf.expert_count"))
    expert_layers = int(need(leaf(gguf, "routed_expert_layers"), "model.gguf.routed_expert_layers"))
    step = expert_step_mib(expert_total, expert_count)

    workspace = float(opts["workspace_mib"] if opts.get("workspace_mib") is not None else 1024)
    workspace_source = "policy" if opts.get("workspace_mib") is None else "ask"
    staging = float(opts["staging_mib"] if opts.get("staging_mib") is not None else mconf["staging_mib"])
    staging_source = "ask" if opts.get("staging_mib") is not None else mconf["staging_source"]
    safety = float(opts["safety_mib"] if opts.get("safety_mib") is not None
                   else max(512.0, mconf["safety_frac"] * vram_total))
    safety_source = "ask" if opts.get("safety_mib") is not None else "policy"

    static_steps = [
        {"step": "start", "mib": round(float(vram_total), 1), "source": leaf(gpu, "vram_total_mib")["source"]},
        {"step": "desktop_other_used", "mib": -round(desktop_used, 1),
         "source": "nvidia-smi" if vram_free is not None else "unknown",
         "note": "total - free at fingerprint time (idle); engines re-clamp at placement, see --free-mib" if vram_free is not None else "free reading unavailable; started from total"},
        {"step": "non_expert_weights", "mib": -round(mib(non_expert), 1), "source": "gguf-meta",
         "note": "attention/GDN/shared-expert/embed/head/nextn (always resident)"},
        {"step": "kv_gdn_state", "mib": -round(st["kv_mib"] + st["gdn_mib"], 1), "source": "gguf-meta",
         "note": f"KV {round(st['kv_mib'],1)} MiB + GDN {round(st['gdn_mib'],1)} MiB at ctx={ctx} (kv dtype {kv_bytes} B)"},
        {"step": "workspace_graph", "mib": -workspace, "source": workspace_source,
         "note": "engine compute/graph reservation placeholder - not derivable from a GGUF; override with --workspace-mib"},
        {"step": "prefill_staging", "mib": -staging, "source": staging_source,
         "note": "measured 1.10 GiB on the official engine, tasks 341/342" if staging_source == "table" else "mode allocation policy"},
        {"step": "safety_margin", "mib": -round(safety, 1), "source": safety_source,
         "note": ">= 512 MiB or mode fraction of total, whichever is larger"},
    ]
    static_leftover = float(vram_total) - desktop_used - mib(non_expert) - st["kv_mib"] - st["gdn_mib"] - workspace - staging - safety

    runtime = None
    if opts.get("free_mib") is not None:
        free = float(opts["free_mib"])
        runtime = {
            "free_mib": {"value": free, "source": "ask",
                         "note": "free VRAM at the engine's placement moment, measured by the caller (341/342: 2.07 GiB at ctx4096, 1.88 GiB at ctx8192)"},
            "steps": [{"step": "free_at_placement", "mib": free, "source": "ask"},
                      {"step": "prefill_staging", "mib": -staging, "source": staging_source},
                      {"step": "safety_margin", "mib": -round(safety, 1), "source": safety_source}],
            "leftover_mib": free - staging - safety,
        }
        leftover = runtime["leftover_mib"]
        effective_source = "runtime-free"
    else:
        leftover = static_leftover
        effective_source = "static"

    quota = quota_for(leftover, step, expert_count)
    cache_mib = quota / expert_count * mib(expert_total) if expert_count else 0.0

    # ---- validity rejections (explicit, never a silent clamp) ----
    rejected = []
    status = "ok"
    if int(ctx) <= 0:
        status = "rejected"
        rejected.append({"config": {"context": ctx}, "reason": "context must be positive"})
    if max_new is not None and int(max_new) <= 0:
        status = "rejected"
        rejected.append({"config": {"max_new": max_new}, "reason": "max_new must be positive"})
    if int(ctx) > int(context_max):
        status = "rejected"
        rejected.append({"config": {"context": int(ctx)}, "reason": f"exceeds the model's declared context_length {context_max}"})
    if max_new is not None and int(max_new) > int(ctx):
        status = "rejected"
        rejected.append({"config": {"max_new": int(max_new)}, "reason": f"max_new > context ({ctx})"})
    if int(batch) != 1:
        status = "rejected"
        rejected.append({"config": {"batch": int(batch)},
                         "reason": "the serve protocol's batching contract v1 is batch=1 (batching=false, P2-7); batch>1 is not supported"})

    # ---- out-of-budget rejections ----
    needed_all = mib(expert_total)
    if needed_all > leftover:
        rejected.append({"config": {"expert_quota_per_layer": expert_count},
                         "reason": f"all {expert_count} experts of every layer need {round(needed_all,1)} MiB; "
                                   f"the effective expert budget is {round(leftover,1)} MiB (over by {round(needed_all-leftover,1)} MiB)"})
    if quota == 0:
        rejected.append({"config": {"expert_quota_per_layer": 0},
                         "reason": "the effective expert-slot budget fits no whole expert step; the engine would stream every expert (staging only)"})
    if int(ctx) < 4096:
        rejected.append({"config": {"prefill_ubatch": 4096}, "reason": f"ubatch > context ({ctx})"})

    host_mode = opts.get("host_mode", "pageable")
    pinned_cap = opts.get("pinned_cap_mib")
    threads_base = opts.get("threads")
    if threads_base is None:
        cs = fp.get("cpu", {})
        threads_base = leaf(cs, "physical_cores", default={}).get("value") or leaf(cs, "logical_cores", default={}).get("value")

    candidates = []
    if status == "ok" and quota > 0:
        ubatch_cap = 2048
        ub_main = min(ubatch_cap, int(ctx))
        rows = [
            ("expert_max", quota, staging, safety, threads_base, ub_main,
             f"spend the whole effective budget ({round(leftover,1)} MiB) on expert residency: up to {quota} of {expert_count} experts per layer, {round(cache_mib,1)} MiB total",
             "no slack beyond the safety margin; later graph growth must fit in the margin"),
            ("expert_balanced", max(1, int(quota * 0.75)), staging, safety, threads_base, ub_main,
             "keep ~25% of the expert budget unallocated as room for runtime growth",
             "fewer resident experts than the budget allows"),
            ("threads_headroom", quota, staging, safety, max(1, (threads_base or 2) - 2), ub_main,
             "leave two physical cores to the OS/desktop alongside the host-side work",
             "fewer threads for the host-side copy/combine pool"),
            ("ubatch_light", quota, staging, safety, threads_base, min(1024, int(ctx)),
             "smaller prefill ubatch: less transient compute pressure per step",
             "longer prefill in step count (a scheduling tradeoff, not a speed claim)"),
            ("expert_min", max(1, quota // 2), staging, safety, max(1, (threads_base or 2) - 2), min(512, int(ctx)),
             "a deliberately conservative floor: smallest expert residency worth keeping plus the highest slack",
             "most expert traffic goes to the host/disk path"),
        ]
        staging_boost = staging + 512
        leftover_boost = leftover - 512
        quota_boost = quota_for(leftover_boost, step, expert_count)
        if quota_boost > 0 and quota_boost < quota:
            rows.append(("staging_boost", quota_boost, staging_boost, safety, threads_base, min(4096, int(ctx)),
                         "move 512 MiB from expert slots into the prefill cold-expert staging buffer",
                         "fewer resident experts while prefill streams more cold data at once"))
        order = {name: i for i, name in enumerate(mconf["preference"])}
        rows.sort(key=lambda r: order.get(r[0], 99))
        for rank, (name, q, stag, saf, thr, ub, reason, risk) in enumerate(rows, 1):
            cand = _candidate(name, q, stag, saf, thr, ub, expert_layers, step, reason, risk)
            cand["rank"] = rank
            candidates.append(cand)

    # ---- host RAM pool (separate; pageable budget counts the pagefile, pinned does not) ----
    ram = fp.get("ram", {})
    phys_total = leaf(ram, "total_mib", default={}).get("value")
    phys_avail = leaf(ram, "avail_mib", default={}).get("value")
    commit_limit = leaf(ram, "commit_limit_mib", default={}).get("value")
    model_bytes = leaf(model_sec, "size_bytes", default={}).get("value")
    host_working = float(opts.get("host_working_mib", 1024))
    host_extra = float(opts.get("host_extra_mib", 0) or 0)
    os_reserve = float(opts.get("os_reserve_mib", 2048))
    host_terms = [
        {"name": "model_resident", "mib": round(mib(model_bytes), 1) if model_bytes else None,
         "source": "fs" if model_bytes else "unknown",
         "note": "the model file as the host keeps it (pack/resident); mmap'd file bytes count against the commit limit"},
        {"name": "host_expert_cache", "mib": host_extra, "source": "ask" if host_extra else "policy",
         "note": "caller-added host-side expert cache; the local engine's COMPACT 2700-slot resident was 3199.2 MiB host-side (tasks 346/347)"},
        {"name": "prefill_staging", "mib": staging, "source": staging_source},
        {"name": "working_allowance", "mib": host_working, "source": "policy",
         "note": "copies/combine buffers and allocator slack; override with --host-working-mib"},
    ]
    requirement = (mib(model_bytes) if model_bytes else 0.0) + host_extra + staging + host_working
    os_reserve_note = (f"OS reserve {os_reserve} MiB is a policy subtraction from the commit limit"
                       if commit_limit is not None else "commit limit unknown; physical RAM used as the bound")
    fits_pageable = requirement <= ((commit_limit - os_reserve) if commit_limit is not None else (phys_total or 0))
    if host_mode == "pinned":
        if pinned_cap is None:
            pinned_fits = False
            pinned_reason = ("no pinned cap was given; the only recorded kernel-pinned attempt on this host "
                             "(cudaHostAlloc of the 18.75 GiB expert arena) failed (tasks 341/342), so pinned is refused "
                             "- pageable is the supported path")
        elif requirement <= float(pinned_cap):
            pinned_fits = True
            pinned_reason = f"within the caller's --pinned-cap-mib {pinned_cap}"
        else:
            pinned_fits = False
            pinned_reason = (f"requires {round(requirement,1)} MiB > --pinned-cap-mib {pinned_cap}; the recorded "
                             f"18.75 GiB cudaHostAlloc failure warns that pinning is size-sensitive on this host")
        rejected.append({"config": {"host_mode": "pinned", "requirement_mib": round(requirement, 1)},
                         "reason": pinned_reason + "; pageable fallback is planned"})
    host = {
        "mode": host_mode,
        "physical": {"total_mib": phys_total, "avail_mib": phys_avail},
        "commit": {"limit_mib": commit_limit, "os_reserve_mib": os_reserve},
        "pagefile_counted": host_mode == "pageable",
        "terms": host_terms,
        "requirement_mib": round(requirement, 1),
        "fits": fits_pageable if host_mode == "pageable" else (fits_pageable and pinned_fits),
        "notes": [
            "pageable: the arena may be backed by the pagefile, so the bound is the commit limit, not physical RAM (" + os_reserve_note + ")",
            "observed on this host: in-run AvailPhys minima of 14-367 MiB with the pagefile absorbing the peak (tasks 341/342/347) - acceptable, and a known disk-paging cost, not quantified here",
            "pinned: no paging, but kernel-pinned allocations are size-sensitive here (cudaHostAlloc 18.75 GiB failed, tasks 341/342); the planner refuses pinned without an explicit --pinned-cap-mib",
            "this pool is never added to the VRAM pool; the two are independent budgets",
        ],
        "fallback": "pageable" if host_mode == "pinned" and not (pinned_cap and requirement <= float(pinned_cap)) else None,
    }

    assumptions = [
        {"id": "kv_dtype", "statement": f"KV cache dtype is set to {kv_bytes} bytes/element by default; the official engine's default is f16, the local strata engine keeps f32 device KV (src/core/session.cpp)", "source": "policy / --kv-bytes"},
        {"id": "gdn_state", "statement": "GDN state = state_size^2 x time_step_rank x 4 B + conv_channels x (conv_kernel-1) x 4 B per recurrent layer, conv_channels = 2*state_size*group_count + inner_size", "source": "table (engine source: src/core/session.cpp, include/strata/core/family_layout.hpp)"},
        {"id": "workspace", "statement": f"{workspace} MiB engine compute/graph reservation placeholder; a GGUF cannot describe it and no measurement is folded in here", "source": "policy / --workspace-mib"},
        {"id": "staging", "statement": f"{staging} MiB prefill cold-expert staging", "source": staging_source},
        {"id": "safety", "statement": f"{round(safety,1)} MiB safety margin = max(512 MiB, {mconf['safety_frac']:.0%} of total VRAM in mode {mode})", "source": "policy"},
        {"id": "desktop", "statement": "desktop/other VRAM usage is total - free at fingerprint time", "source": "nvidia-smi"},
        {"id": "expert_step", "statement": f"one quota step = one expert of one layer, averaged: routed_expert_bytes_total / expert_count = {round(step,2)} MiB", "source": "gguf-meta"},
        {"id": "no_speed_claim", "statement": "this plan is capacity only; it makes no speed or benefit claim", "source": "policy"},
    ]

    vram = {
        "total_mib": leaf(gpu, "vram_total_mib"),
        "desktop_used_mib": {"value": round(desktop_used, 1) if vram_free is not None else None,
                             "source": "nvidia-smi" if vram_free is not None else "unknown"},
        "static_chain": {"steps": static_steps, "leftover_mib": round(static_leftover, 1)},
        "runtime_clamp": runtime,
        "effective": {"leftover_mib": round(leftover, 1), "source": effective_source},
        "expert_slot": {"step_mib": round(step, 2),
                        "quota_per_layer": quota, "slots_total": quota * expert_layers,
                        "cache_mib": round(cache_mib, 1), "cache_gib": round(cache_mib / 1024, 3)},
        "state": st,
        "expert_layers": expert_layers,
        "expert_count": expert_count,
    }

    return {
        "schema": "strata-plan/1",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "notice": "capacity plan only - no speed or benefit claim is made or implied",
        "status": status,
        "mode": mode,
        "inputs": {
            "model": {"path": leaf(model_sec, "path"), "size_bytes": leaf(model_sec, "size_bytes"),
                      "sha256": leaf(model_sec, "sha256")},
            "workload": {"context": ctx, "max_new": max_new, "batch": int(batch)},
            "context_max": context_max,
            "overrides": {k: opts.get(k) for k in ("safety_mib", "staging_mib", "workspace_mib", "kv_bytes",
                                                   "free_mib", "host_mode", "pinned_cap_mib", "threads")},
        },
        "assumptions": assumptions,
        "vram": vram,
        "host": host,
        "candidates": candidates,
        "rejected": rejected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fingerprint", required=True, help="JSON from strata_fingerprint.py")
    ap.add_argument("--context", type=int)
    ap.add_argument("--max-new", type=int, dest="max_new")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--mode", default="safe", choices=sorted(MODES))
    ap.add_argument("--safety-mib", type=float, dest="safety_mib")
    ap.add_argument("--staging-mib", type=float, dest="staging_mib")
    ap.add_argument("--workspace-mib", type=float, dest="workspace_mib")
    ap.add_argument("--kv-bytes", type=int, dest="kv_bytes", default=2, choices=(2, 4))
    ap.add_argument("--free-mib", type=float, dest="free_mib",
                    help="free VRAM at the engine's placement moment; enables the runtime-clamp reproduction")
    ap.add_argument("--host-mode", dest="host_mode", default="pageable", choices=("pageable", "pinned"))
    ap.add_argument("--pinned-cap-mib", type=float, dest="pinned_cap_mib")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--host-working-mib", type=float, dest="host_working_mib", default=1024)
    ap.add_argument("--host-extra-mib", type=float, dest="host_extra_mib", default=0)
    ap.add_argument("--os-reserve-mib", type=float, dest="os_reserve_mib", default=2048)
    ap.add_argument("--out")
    a = ap.parse_args()
    fp = json.loads(Path(a.fingerprint).read_text(encoding="utf-8"))
    opts = {k: getattr(a, k) for k in ("context", "max_new", "batch", "mode", "safety_mib", "staging_mib",
                                       "workspace_mib", "kv_bytes", "free_mib", "host_mode", "pinned_cap_mib",
                                       "threads", "host_working_mib", "host_extra_mib", "os_reserve_mib")}
    try:
        plan = build_plan(fp, opts)
    except ValueError as exc:
        print(json.dumps({"schema": "strata-plan/1", "error": str(exc)}, indent=1), file=sys.stderr)
        sys.exit(2)
    text = json.dumps(plan, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)
    sys.exit(0 if plan["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
