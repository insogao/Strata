#!/usr/bin/env python3
"""strata_cost_model.py - offline per-token cost model for the P3 autotuner (P3-2, 0 GPU).

    python backends/qwen35_llama/tools/strata_cost_model.py cases
    python backends/qwen35_llama/tools/strata_cost_model.py params [--out params.json]
    python backends/qwen35_llama/tools/strata_cost_model.py predict \
        --tokens 512 --rounds 306 (--misses 149500 | --miss-fraction 0.7634 | --miss-fraction unknown) \
        [--threads 8] [--out predict.json]
    python backends/qwen35_llama/tools/strata_cost_model.py holdout [--out holdout.json]

This is a RECOMPUTABLE cost model, not a benchmark: it runs no engine, takes no lease, and
makes no product-speed claim.  It is calibrated ONLY on the four recorded N1-N4 processes of
task 347 (one batch: same exe sha 03bd7068..., same GGUF ab639a7f..., same profile fab192ff...,
MTP K1, 2700 COMPACT, threads 8, ubatch 4096).  Every parameter is listed with the source file
and field it was derived from; numbers from other batches (346/358/359/367/341/342/343) are
NOT mixed into the fit - they are context only (see the doc).

Model (per token of the MTP K1 decode transaction):

    ms_per_token = rounds_per_token x (id_readback_ms_per_round
                                       + gpu_prefix_ms_per_round
                                       + combine_ms_per_round)
                 + misses_per_token x cpu_miss_us_per_expert / 1000
                 + non_tier_ms_per_token

  * rounds_per_token, misses_per_token  - caller input (engine counters for a measured run,
    or a declared miss fraction for a candidate; a missing miss fraction stays unknown and the
    interval widens to the physical bounds [0 miss, every routed evaluation a miss]).
  * id_readback_ms_per_round           - 347 summary tier_ms.ids / rounds of the same process.
  * gpu_prefix_ms_per_round            - 347 summary tier_ms.launch / rounds (the GPU grouped-
    MMQ prefix of the miss path).
  * combine_ms_per_round               - 347 summary tier_ms.combine / rounds.
  * cpu_miss_us_per_expert             - 347 summary tier_ms.misses / decode_misses.
  * non_tier_ms_per_token              - residual: flush ms/token (metrics-347.jsonl TPOT
    weighted by (n-1)) minus the tier sums above, / tokens.  It is an empirical residual (the
    transaction work outside the tier phases: draft/verify GPU prefix/emit/sampler/overlap
    error), not a physical decomposition, and it is the widest single term.

No parameter is a free variable fitted to a target TPS: each is a direct ratio of two recorded
fields, and hold-out leaves one 347 process out, refits from the other three and reports the
missed prediction honestly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

SCHEMA = "strata-cost-model/1"
TIER_PHASES = ("ids", "launch", "misses", "combine")
PARAMS = (
    "id_readback_ms_per_round",
    "gpu_prefix_ms_per_round",
    "combine_ms_per_round",
    "cpu_miss_us_per_expert",
    "non_tier_ms_per_token",
)

SOURCE_347_DIR = "results/opencode-execution/phase3-60tps-local-long-baseline-347"
SOURCE_347_DOC = "docs/phase3_60tps/LOCAL_LONG_BASELINE_347.md"

STATIC_SOURCES = {
    "tier_ms": f"{SOURCE_347_DIR}/raw-N*/child_00.log field \"event\":\"summary\".tier_ms "
               f"(.ids/.launch/.misses/.combine); table also in {SOURCE_347_DOC} section 2",
    "hits_misses": f"{SOURCE_347_DIR}/raw-N*/child_00.log field \"event\":\"summary\" "
                   f".decode_hits/.decode_misses",
    "rounds_tokens": f"{SOURCE_347_DIR}/raw-N*/child_00.log field "
                     f"\"event\":\"qwen35_mtp_txn_request_timing\".rounds/.emitted (2 requests/process)",
    "flush": f"{SOURCE_347_DIR}/metrics-347.jsonl fields TPS_true_flush/TPOT_true_flush/"
             f"eos_complete_request_seconds/ttft_flush_request_seconds; formula "
             f"(n-1)/(last-first flush emit) recorded in {SOURCE_347_DOC} section 1",
    "config": f"{SOURCE_347_DIR}/job-N*.json argv + manifest347.json + {SOURCE_347_DOC} section 0",
}

# One batch: the four N1-N4 processes of task 347 (2026-10-14).  Fixed identity:
CONFIG = {
    "exe_sha256": "03bd7068f284450cef90aa633bc248c72b7272ee143da9a54443d79db84dde5f",
    "model_sha256": "ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f",
    "model_bytes": 15346432288,
    "profile_sha256": "fab192ff3821d7ae4d3384ae57b4fe475153d22604dbe7eccf2b8bbc97ce754e",
    "cache_slots": 2700,
    "compact": True,
    "threads": 8,
    "ubatch": 4096,
    "mtp": "K1",
    "gu_multi": 0,
    "requests_per_process": 2,
    "source": STATIC_SOURCES["config"],
}

# Recorded per-process rows (verbatim from the raw summary events; no cross-batch arithmetic).
RECORDED_CASES = [
    {
        "case": "N1", "ctx": 8192, "fixture": "l-risk-explain (3817 ids)",
        "rounds": 160, "tokens": 256, "hits": 25314, "misses": 77086,
        "tier_ms": {"ids": 1708.3, "launch": 398.2, "misses": 3572.0, "combine": 93.7, "admit": 0.0},
        "tps_true_flush": [30.897735, 31.410667],
        "tpot_true_flush": [0.032365, 0.031836],
        "n_eff": [127, 127],
        "peak_device_vram_mib": 7555.3, "prefill_tokens_per_s": 1038.91,
    },
    {
        "case": "N2", "ctx": 8192, "fixture": "l-benign-qa (3791 ids)",
        "rounds": 306, "tokens": 512, "hits": 46340, "misses": 149500,
        "tier_ms": {"ids": 3912.7, "launch": 1232.2, "misses": 9235.8, "combine": 288.1, "admit": 0.0},
        "tps_true_flush": [22.345905, 26.642097],
        "tpot_true_flush": [0.044751, 0.037535],
        "n_eff": [255, 255],
        "peak_device_vram_mib": 7592.1, "prefill_tokens_per_s": 750.923,
    },
    {
        "case": "N3", "ctx": 4096, "fixture": "l-risk-explain (3817 ids)",
        "rounds": 160, "tokens": 256, "hits": 25314, "misses": 77086,
        "tier_ms": {"ids": 1748.8, "launch": 493.3, "misses": 4008.2, "combine": 121.1, "admit": 0.0},
        "tps_true_flush": [26.569924, 27.430521],
        "tpot_true_flush": [0.037637, 0.036456],
        "n_eff": [127, 127],
        "peak_device_vram_mib": 7618.4, "prefill_tokens_per_s": 1030.13,
    },
    {
        "case": "N4", "ctx": 4096, "fixture": "l-benign-qa (3791 ids)",
        "rounds": 306, "tokens": 512, "hits": 46340, "misses": 149500,
        "tier_ms": {"ids": 3626.4, "launch": 948.9, "misses": 8172.4, "combine": 233.6, "admit": 0.0},
        "tps_true_flush": [26.053789, 28.645245],
        "tpot_true_flush": [0.038382, 0.034910],
        "n_eff": [255, 255],
        "peak_device_vram_mib": 7591.0, "prefill_tokens_per_s": 1011.69,
    },
]

PARAM_UNITS = {
    "id_readback_ms_per_round": "ms/round",
    "gpu_prefix_ms_per_round": "ms/round",
    "combine_ms_per_round": "ms/round",
    "cpu_miss_us_per_expert": "us per missed expert evaluation",
    "non_tier_ms_per_token": "ms/token",
}

PARAM_SOURCES = {
    "id_readback_ms_per_round": STATIC_SOURCES["tier_ms"] + " (.ids / rounds)",
    "gpu_prefix_ms_per_round": STATIC_SOURCES["tier_ms"] + " (.launch / rounds)",
    "combine_ms_per_round": STATIC_SOURCES["tier_ms"] + " (.combine / rounds)",
    "cpu_miss_us_per_expert": STATIC_SOURCES["tier_ms"] + " (.misses) / " + STATIC_SOURCES["hits_misses"] + " x 1000",
    "non_tier_ms_per_token": (STATIC_SOURCES["flush"] + " minus the tier sums above, per token "
                              "(empirical residual of this model, not a physical phase)"),
}


def derive(case: dict) -> dict:
    tier_ms = sum(case["tier_ms"][p] for p in TIER_PHASES)
    ms_per_token_tier = tier_ms / case["tokens"]
    windows = [n * tpot for n, tpot in zip(case["n_eff"], case["tpot_true_flush"])]
    ms_per_token_flush = sum(windows) / sum(case["n_eff"]) * 1000.0
    evals = case["hits"] + case["misses"]
    return {
        "case": case["case"], "ctx": case["ctx"],
        "rounds_per_token": case["rounds"] / case["tokens"],
        "evaluations_per_token": evals / case["tokens"],
        "miss_fraction": case["misses"] / evals,
        "tier_ms_total": tier_ms,
        "tier_ms_per_token": ms_per_token_tier,
        "flush_ms_per_token": ms_per_token_flush,
        "residual_ms_per_token": ms_per_token_flush - ms_per_token_tier,
        "tps_true_flush_weighted": sum(case["n_eff"]) / sum(windows),
    }


def case_params(case: dict) -> dict:
    d = derive(case)
    return {
        "id_readback_ms_per_round": case["tier_ms"]["ids"] / case["rounds"],
        "gpu_prefix_ms_per_round": case["tier_ms"]["launch"] / case["rounds"],
        "combine_ms_per_round": case["tier_ms"]["combine"] / case["rounds"],
        "cpu_miss_us_per_expert": case["tier_ms"]["misses"] / case["misses"] * 1000.0,
        "non_tier_ms_per_token": d["residual_ms_per_token"],
    }


def fit(cases) -> dict:
    per_case = {c["case"]: case_params(c) for c in cases}
    out = {}
    for p in PARAMS:
        vals = {c["case"]: round(per_case[c["case"]][p], 6) for c in cases}
        xs = list(vals.values())
        out[p] = {
            "unit": PARAM_UNITS[p],
            "source": PARAM_SOURCES[p],
            "values_per_case": vals,
            "min": min(xs), "max": max(xs),
            "mean": sum(xs) / len(xs),
            "spread_pct_of_mean": (max(xs) - min(xs)) / (sum(xs) / len(xs)) * 100.0,
        }
    return out


def _central(params: dict, name: str) -> float:
    return params[name]["mean"]


def _bounds(params: dict, name: str):
    return params[name]["min"], params[name]["max"]


def predict(params: dict, tokens: int, rounds: int, misses=None, threads: int = 8,
            demand_per_round: float = 640.0) -> dict:
    """Predict per-token time and TPS for one 347-calibrated configuration.

    misses      explicit count, or None to let the caller declare a miss fraction; a missing
                fraction is NOT guessed: the miss term widens to [0, every evaluation].
    demand_per_round
                routed expert evaluations per K1 round recorded in 347 counters
                (2 token positions x 8 experts x 40 routed layers = 640).
    """
    unknown = []
    if threads != CONFIG["threads"]:
        return {
            "status": "uncalibrated",
            "reason": (f"every recorded tier parameter is for threads={CONFIG['threads']} (347); "
                       f"no recorded calibration exists at threads={threads}, so no number is fabricated"),
            "unknown_terms": ["cpu_pool_threads"],
            "measured_basis": f"threads={CONFIG['threads']} only ({STATIC_SOURCES['config']})",
        }
    rounds_per_token = rounds / tokens
    if misses is not None:
        miss_lo_pt = miss_hi_pt = float(misses) / tokens
        miss_note = "misses supplied by caller (measured counter or declared candidate)"
    else:
        miss_lo_pt, miss_hi_pt = 0.0, demand_per_round * rounds / tokens
        miss_note = ("miss fraction unknown (not guessed): interval widens to the physical bounds "
                     "[0, every routed evaluation a miss]")
        unknown.append("miss_fraction")

    ids_min, ids_max = _bounds(params, "id_readback_ms_per_round")
    gpu_min, gpu_max = _bounds(params, "gpu_prefix_ms_per_round")
    cmb_min, cmb_max = _bounds(params, "combine_ms_per_round")
    miss_min, miss_max = _bounds(params, "cpu_miss_us_per_expert")
    nt_min, nt_max = _bounds(params, "non_tier_ms_per_token")

    fixed_lo = (ids_min + gpu_min + cmb_min) * rounds_per_token
    fixed_hi = (ids_max + gpu_max + cmb_max) * rounds_per_token
    fixed_central = sum(_central(params, p) for p in
                        ("id_readback_ms_per_round", "gpu_prefix_ms_per_round", "combine_ms_per_round")) * rounds_per_token
    miss_lo_ms = miss_lo_pt * miss_min / 1000.0
    miss_hi_ms = miss_hi_pt * miss_max / 1000.0
    miss_central = ((miss_lo_pt + miss_hi_pt) / 2.0) * _central(params, "cpu_miss_us_per_expert") / 1000.0
    ms_lo = fixed_lo + miss_lo_ms + nt_min
    ms_hi = fixed_hi + miss_hi_ms + nt_max
    ms_central = fixed_central + miss_central + _central(params, "non_tier_ms_per_token")
    tps_lo, tps_hi = 1000.0 / ms_hi, 1000.0 / ms_lo
    status = "wide" if unknown else "ok"
    return {
        "status": status,
        "unknown_terms": unknown,
        "inputs": {"tokens": tokens, "rounds": rounds, "threads": threads,
                   "rounds_per_token": round(rounds_per_token, 6),
                   "misses": misses if misses is not None else "unknown",
                   "misses_interval": ([misses, misses] if misses is not None
                                       else [round(miss_lo_pt * tokens, 1), round(miss_hi_pt * tokens, 1)]),
                   "misses_per_token": round((miss_lo_pt + miss_hi_pt) / 2.0, 4),
                   "demand_per_round": demand_per_round,
                   "miss_note": miss_note},
        "terms_ms_per_token": {
            "tier_round_fixed": {"central": round(fixed_central, 4),
                                 "interval": [round(fixed_lo, 4), round(fixed_hi, 4)],
                                 "source": "ids+launch+combine per round x rounds_per_token"},
            "cpu_miss": {"central": round(miss_central, 4),
                         "interval": [round(miss_lo_ms, 4), round(miss_hi_ms, 4)],
                         "source": "misses x cpu_miss_us_per_expert"},
            "non_tier": {"central": round(_central(params, "non_tier_ms_per_token"), 4),
                         "interval": [round(nt_min, 4), round(nt_max, 4)],
                         "source": "347 residual (flush - tier) per token"},
        },
        "ms_per_token": {"central": round(ms_central, 4),
                         "interval": [round(ms_lo, 4), round(ms_hi, 4)]},
        "tps": {"central": round(1000.0 / ms_central, 4),
                "interval": [round(tps_lo, 4), round(tps_hi, 4)]},
        "notice": ("diagnostic interval prediction from the 347 batch - not a product speed and "
                   "not a promise; only the interval endpoints follow from the recorded ranges"),
    }


def sensitivity(params: dict, tokens: int, rounds: int, misses: float) -> list:
    base = predict(params, tokens, rounds, misses=misses)["ms_per_token"]["central"]
    base_tps = 1000.0 / base
    out = []
    for p in PARAMS:
        bumped = {k: dict(v) for k, v in params.items()}
        bumped[p]["mean"] = params[p]["mean"] * 1.1
        ms = predict(bumped, tokens, rounds, misses=misses)["ms_per_token"]["central"]
        out.append({"parameter": p, "delta_pct_on_parameter": 10.0,
                    "tps_change_pct": round((1000.0 / ms - base_tps) / base_tps * 100.0, 4)})
    return out


def holdout(cases) -> dict:
    rows = []
    for held in cases:
        rest = [c for c in cases if c["case"] != held["case"]]
        p = fit(rest)
        pred = predict(p, held["tokens"], held["rounds"], misses=held["misses"])
        measured_tps = derive(held)["tps_true_flush_weighted"]
        measured_ms = derive(held)["flush_ms_per_token"]
        central_ms = pred["ms_per_token"]["central"]
        lo, hi = pred["ms_per_token"]["interval"]
        rows.append({
            "held_out": held["case"],
            "fit_from": [c["case"] for c in rest],
            "measured_ms_per_token": round(measured_ms, 4),
            "predicted_central_ms_per_token": round(central_ms, 4),
            "predicted_interval_ms_per_token": [round(lo, 4), round(hi, 4)],
            "central_error_ms": round(central_ms - measured_ms, 4),
            "central_error_pct": round((central_ms - measured_ms) / measured_ms * 100.0, 4),
            "measured_tps": round(measured_tps, 4),
            "predicted_central_tps": pred["tps"]["central"],
            "predicted_interval_tps": pred["tps"]["interval"],
            "measured_in_interval": bool(lo <= measured_ms <= hi),
        })
    errs = [abs(r["central_error_pct"]) for r in rows]
    return {
        "protocol": ("leave-one-out within the 347 batch (same exe/model/profile/session family); "
                     "parameters refit from the remaining three processes; the held-out process is "
                     "predicted with its own recorded counters"),
        "rows": rows,
        "mean_abs_central_error_pct": round(sum(errs) / len(errs), 4),
        "max_abs_central_error_pct": round(max(errs), 4),
        "interval_coverage": f"{sum(1 for r in rows if r['measured_in_interval'])}/{len(rows)}",
        "caveat": ("these errors are the honest residual of the model; do not average them away "
                   "and do not convert cross-batch numbers to narrow them"),
    }


def in_sample_residuals(params: dict) -> list:
    rows = []
    for c in RECORDED_CASES:
        pred = predict(params, c["tokens"], c["rounds"], misses=c["misses"])
        measured = derive(c)["flush_ms_per_token"]
        rows.append({
            "case": c["case"],
            "measured_ms_per_token": round(measured, 4),
            "central_ms_per_token": pred["ms_per_token"]["central"],
            "residual_ms": round(pred["ms_per_token"]["central"] - measured, 4),
            "interval_covers_measured": bool(pred["ms_per_token"]["interval"][0] <= measured
                                             <= pred["ms_per_token"]["interval"][1]),
        })
    return rows


def cmd_cases(_a) -> None:
    doc = {"schema": SCHEMA, "config": CONFIG, "sources": STATIC_SOURCES,
           "cases": [{"case": c["case"], "ctx": c["ctx"], "fixture": c["fixture"],
                      "rounds": c["rounds"], "tokens": c["tokens"], "hits": c["hits"],
                      "misses": c["misses"], "tier_ms": c["tier_ms"],
                      "tps_true_flush": c["tps_true_flush"],
                      "peak_device_vram_mib": c["peak_device_vram_mib"],
                      "prefill_tokens_per_s": c["prefill_tokens_per_s"],
                      "derived": {k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in derive(c).items()}} for c in RECORDED_CASES]}
    _emit(doc, _a.out)


def cmd_params(a) -> None:
    p = fit(RECORDED_CASES)
    doc = {"schema": SCHEMA, "config": CONFIG, "sources": STATIC_SOURCES,
           "parameters": p,
           "in_sample_residuals": in_sample_residuals(p),
           "notice": ("every parameter is a ratio of two recorded 347 fields (see source); the "
                      "interval is the observed per-case min/max across the four processes - it is "
                      "an envelope of one batch, not a confidence interval")}
    _emit(doc, a.out)


def cmd_predict(a) -> None:
    tokens, rounds = a.tokens, a.rounds
    misses = a.misses
    if a.miss_fraction is not None and a.miss_fraction != "unknown":
        if misses is not None:
            print("give either --misses or --miss-fraction, not both", file=sys.stderr)
            sys.exit(2)
        misses = float(a.miss_fraction) * 640.0 * rounds
    p = fit(RECORDED_CASES)
    doc = {"schema": SCHEMA, "config": CONFIG,
           "calibration": {k: {"min": v["min"], "max": v["max"], "mean": v["mean"],
                               "unit": v["unit"], "source": v["source"]} for k, v in p.items()},
           "prediction": predict(p, tokens, rounds, misses=misses, threads=a.threads),
           "sensitivity_central": (sensitivity(p, tokens, rounds, misses) if misses is not None else
                                   "not computed with an unknown miss fraction (ranges only)"),
           "notice": "diagnostic cost interval - not a product speed"}
    _emit(doc, a.out)


def cmd_holdout(a) -> None:
    doc = {"schema": SCHEMA, "config": CONFIG, "holdout": holdout(RECORDED_CASES)}
    _emit(doc, a.out)


def _emit(doc: dict, out_path) -> None:
    doc["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n"
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        sys.stdout.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (("cases", cmd_cases, "the four recorded 347 processes + derived rows"),
                               ("params", cmd_params, "fit the five parameters + in-sample residuals"),
                               ("predict", cmd_predict, "predict ms/token and TPS interval"),
                               ("holdout", cmd_holdout, "leave-one-out check within the 347 batch")):
        sp_ap = sub.add_parser(name, help=helptext)
        sp_ap.add_argument("--out")
        if name == "predict":
            sp_ap.add_argument("--tokens", type=int, required=True)
            sp_ap.add_argument("--rounds", type=int, required=True)
            sp_ap.add_argument("--misses", type=int)
            sp_ap.add_argument("--miss-fraction", dest="miss_fraction",
                               help="declared fraction, or 'unknown' for the physical-bounds widening")
            sp_ap.add_argument("--threads", type=int, default=8)
        sp_ap.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
