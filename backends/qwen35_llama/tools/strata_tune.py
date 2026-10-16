#!/usr/bin/env python3
"""Opt-in, offline autotuner profile CLI. No engine launches or configuration writes.

  python backends/qwen35_llama/tools/strata_tune.py tune --help
  python backends/qwen35_llama/tools/strata_tune.py check --help

P3-1 capacity estimates and P3-2 diagnostic rankings are NOT measured winners. This
version deliberately emits no selected configuration. A profile is a reproducible
research record, never a production admission token. See docs/PHASE3_DELIVERY.md.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import strata_fingerprint as sf
import strata_plan as sp

SCHEMA = "strata-tune-profile/1"
TOOLS = Path(__file__).resolve().parent
MTP_SHA = "ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f"
TOOL_FILES = ("strata_tune.py", "strata_fingerprint.py", "strata_plan.py",
              "strata_screen.py", "strata_cost_model.py", "gguf_reader.py", "_paths.py",
              "strata_experiment.py", "leased_native.py", "gpu_lock.py")
PROFILE_FIELDS = {"schema", "created_at", "status", "selected_configuration", "identity",
                  "identity_sha256", "missing_identity", "fingerprint", "capacity_plan",
                  "diagnostic_screen", "fallback", "evidence", "confidence", "next_action",
                  "content_sha256"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def bad_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=bad_constant)


def values(node):
    """Strip acquisition timestamps/provenance, retaining all declared identity values."""
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            return values(node["value"])
        return {k: values(v) for k, v in node.items()}
    if isinstance(node, list):
        return [values(v) for v in node]
    return node


def validate_workload(workload):
    if not isinstance(workload, dict):
        raise ValueError("workload must be an object")
    for key in ("context", "max_new", "batch"):
        if type(workload.get(key)) is not int or workload[key] <= 0:
            raise ValueError(f"workload.{key} must be a positive integer")
    if workload["batch"] != 1 or workload["max_new"] >= workload["context"]:
        raise ValueError("requires batch=1 and max_new < context")
    # Names alone cannot distinguish changed input/template bytes. Callers register their
    # canonical workload manifest, including fixtures, sampling, MTP and runtime flags.
    if not re.fullmatch(r"[0-9a-f]{64}", str(workload.get("manifest_sha256", ""))):
        raise ValueError("workload.manifest_sha256 must identify the full workload manifest")


def binding(fp, engine, workload, mode, kv_bytes, runtime_id):
    validate_workload(workload)
    if not isinstance(fp, dict) or fp.get("schema") != "strata-fingerprint/1":
        raise ValueError("unsupported fingerprint schema")
    data = values(fp)
    model = data.get("model", {})
    gpu = data.get("gpu", {})
    result = {
        "hardware": {
            "gpu": {k: gpu.get(k) for k in ("index", "uuid", "name", "compute_capability",
                                            "vram_total_mib", "driver_version", "cuda_driver_max")},
            "cpu": data.get("cpu"),
            "ram": {k: data.get("ram", {}).get(k) for k in ("total_mib", "commit_limit_mib")},
            "os": data.get("host", {}).get("os"),
        },
        "model": {k: model.get(k) for k in ("sha256", "size_bytes", "mtime_utc", "gguf")},
        "engine_sha256": file_sha(engine),
        "runtime_id": runtime_id,
        "tool_sha256": {name: file_sha(TOOLS / name) for name in TOOL_FILES},
        "workload": copy.deepcopy(workload),
        "policy": {"mode": mode, "kv_bytes": kv_bytes},
    }
    return result


def missing_identity(identity):
    required = {
        "gpu.name": identity["hardware"]["gpu"].get("name"),
        "gpu.uuid": identity["hardware"]["gpu"].get("uuid"),
        "gpu.compute_capability": identity["hardware"]["gpu"].get("compute_capability"),
        "gpu.driver_version": identity["hardware"]["gpu"].get("driver_version"),
        "gpu.vram_total_mib": identity["hardware"]["gpu"].get("vram_total_mib"),
        "cpu.model_name": (identity["hardware"].get("cpu") or {}).get("model_name"),
        "cpu.logical_cores": (identity["hardware"].get("cpu") or {}).get("logical_cores"),
        "ram.total_mib": identity["hardware"]["ram"].get("total_mib"),
        "ram.commit_limit_mib": identity["hardware"]["ram"].get("commit_limit_mib"),
        "os": identity["hardware"].get("os"),
        "model.size_bytes": identity["model"].get("size_bytes"),
        "model.mtime_utc": identity["model"].get("mtime_utc"),
        "model.gguf": identity["model"].get("gguf"),
        "runtime_id": identity.get("runtime_id"),
    }
    missing = [key for key, value in required.items() if value is None or value == "" or value == {}]
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity["model"].get("sha256", ""))):
        missing.append("model.sha256 (registered; never recomputed here)")
    if "error" in (identity["model"].get("gguf") or {}):
        missing.append("model.gguf.error")
    return missing


def artifact(path):
    p = Path(path).resolve()
    if p.stat().st_size > 16 * (1 << 20):
        raise ValueError("evidence index accepts small artifacts up to 16 MiB; never pass model payloads")
    return {"path": str(p), "sha256": file_sha(p), "size_bytes": p.stat().st_size}


def capacity_usable(plan):
    # The P3-1 planner lists zero slots as an informational rejection while leaving
    # status=ok (stream-all). Do not mistake that for an admitted cached configuration.
    return (plan["status"] == "ok" and plan["host"]["fits"]
            and plan["vram"]["effective"]["leftover_mib"] >= 0
            and plan["vram"]["expert_slot"]["quota_per_layer"] > 0)


def build_profile(fp, engine, workload, baseline, mode="safe", kv_bytes=4,
                  runtime_id="", evidence=(), diagnostic_screen=False):
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("baseline must be a nonempty caller-supplied configuration object")
    identity = binding(fp, engine, workload, mode, kv_bytes, runtime_id)
    if mode not in sp.MODES or kv_bytes not in (2, 4):
        raise ValueError("unsupported planning policy")
    plan = sp.build_plan(fp, {**{k: workload[k] for k in ("context", "max_new", "batch")},
                              "mode": mode, "kv_bytes": kv_bytes})
    screen = None
    if diagnostic_screen:
        # P3-2 embeds this machine/model's empirical calibration. Do not silently export
        # that model to a different GPU/quantization. Even here it cannot select a winner.
        if (identity["model"]["sha256"] != MTP_SHA
                or identity["hardware"]["gpu"]["name"] != "NVIDIA GeForce RTX 4060 Laptop GPU"):
            raise ValueError("diagnostic screen calibration only covers the recorded 4060 Laptop / MTP IQ3_S")
        import strata_screen as ss
        screen = ss.screen(fp, [2200, 2700, 3200], [True, False], [8, 16],
                           [1024, 2048, 4096], [workload["context"]], 5, 512, 306, kv_bytes)
        screen["usage"] = "347 diagnostic 256-token cost model; NOT the requested workload's measured speed"
        screen["selected_configuration"] = None
        screen["calibration_build_matches_current"] = None  # full calibration binary SHA is not bound here
        screen["pilot_counterevidence"] = {
            "source": "docs/backends/qwen35_llama_autotune_pilot.md",
            "candidate": "s2200-c-t8-u4096-x8192", "verdict": "NO-GO",
            "short_delta_pct": -5.171143, "long_delta_pct": -17.524041,
            "scope": "c3780a6a pilot, S3896/16 + L3817/128; historical diagnostic, not a universal ban",
        }
    doc = {
        "schema": SCHEMA, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research-only", "selected_configuration": None,
        "identity": identity, "identity_sha256": digest(identity),
        "missing_identity": missing_identity(identity),
        "fingerprint": copy.deepcopy(fp), "capacity_plan": plan, "diagnostic_screen": screen,
        "fallback": {"configuration": copy.deepcopy(baseline), "source": "caller baseline file",
                     "confidence": "caller-supplied; not certified by this tool", "auto_apply": False},
        "evidence": [artifact(path) for path in evidence],
        "confidence": {"capacity": "estimate", "speed": "unverified", "quality": "unverified",
                       "cross_hardware": "not validated"},
        "next_action": "paired GPU + quality/EOS/resource validation in a separately registered window",
    }
    doc["content_sha256"] = digest(doc)
    return doc


def changed_fields(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                out.append(path)
            else:
                out.extend(changed_fields(old[key], new[key], path))
        return out
    return [] if old == new else [prefix]


def check_profile(profile, fp, engine, workload, mode="safe", kv_bytes=4, runtime_id=""):
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    reasons = []
    if set(profile) != PROFILE_FIELDS:
        reasons.append("profile fields do not match v1 schema")
    if profile.get("schema") != SCHEMA:
        reasons.append("unsupported profile schema")
    unsigned = {k: v for k, v in profile.items() if k != "content_sha256"}
    if profile.get("content_sha256") != digest(unsigned):
        reasons.append("profile content checksum mismatch")
    if profile.get("status") != "research-only" or profile.get("selected_configuration") is not None:
        reasons.append("this version accepts research-only profiles with no selected configuration")
    if not isinstance(profile.get("fallback"), dict) or profile["fallback"].get("auto_apply") is not False:
        reasons.append("profile must retain a manual fallback with auto_apply=false")
    identity = binding(fp, engine, workload, mode, kv_bytes, runtime_id)
    changes = changed_fields(profile.get("identity", {}), identity)
    if changes:
        reasons.append("identity changed; regenerate offline profile and revalidate measurements")
    if profile.get("identity_sha256") != digest(profile.get("identity", {})):
        reasons.append("identity checksum mismatch")
    missing = missing_identity(identity)
    if missing:
        reasons.append("incomplete current identity")
    for item in profile.get("evidence", []):
        try:
            if file_sha(item["path"]) != item["sha256"]:
                reasons.append(f"evidence changed: {item['path']}")
        except (OSError, KeyError):
            reasons.append("evidence missing or malformed")
    plan = sp.build_plan(fp, {**{k: workload[k] for k in ("context", "max_new", "batch")},
                              "mode": mode, "kv_bytes": kv_bytes})
    if not capacity_usable(plan):
        reasons.append("current resource estimate rejects workload")
    return {"schema": "strata-tune-check/1", "valid": not reasons,
            "status": "stale-or-incomplete" if reasons else "matching-research-record",
            "reasons": reasons, "changed_fields": changes, "missing_identity": missing,
            "current_capacity_plan": plan, "production_approved": False, "auto_apply": False,
            "notice": "A matching record is not a capacity reservation or a quality/performance approval."}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"inspect", "screen", "plan", "experiment", "check-experiment"}:
        from strata_experiment import main as experiment_main
        return experiment_main(args)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("tune", "check"):
        p = sub.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--fingerprint", help="saved fingerprint; check only compares the supplied snapshot")
        source.add_argument("--model", help="fresh read-only host + GGUF header probe (no payload hash)")
        p.add_argument("--model-sha", help="registered model SHA256 for --model; no full-file hashing")
        p.add_argument("--gpu-index", type=int, default=0)
        p.add_argument("--engine", required=True, help="hash this binary without executing it")
        p.add_argument("--runtime-id", required=True, help="registered CUDA/DLL/runtime version or manifest digest")
        p.add_argument("--workload", required=True, help="JSON: context/max_new/batch/manifest_sha256 + settings")
        p.add_argument("--mode", choices=sorted(sp.MODES), default="safe")
        p.add_argument("--kv-bytes", type=int, choices=(2, 4), default=4)
        p.add_argument("--out", help="new JSON file only; existing files are never overwritten")
        if name == "tune":
            p.add_argument("--baseline", required=True, help="JSON configuration to retain as a manual fallback")
            p.add_argument("--evidence", action="append", default=[], help="small evidence artifact to hash/index")
            p.add_argument("--diagnostic-screen", action="store_true")
        else:
            p.add_argument("--profile", required=True)
    a = ap.parse_args(argv)
    try:
        workload = read_json(a.workload)
        validate_workload(workload)
        fp = (read_json(a.fingerprint) if a.fingerprint else
              sf.build_fingerprint(a.model, workload, a.model_sha, a.gpu_index))
        if a.model:
            rc, uuid = sf._run(["nvidia-smi", f"--id={a.gpu_index}", "--query-gpu=uuid",
                                "--format=csv,noheader,nounits"])
            fp["gpu"]["uuid"] = (sf.field(uuid.strip(), "nvidia-smi") if rc == 0 and uuid.strip()
                                  else sf.unknown("GPU UUID unavailable"))
        common = (fp, a.engine, workload)
        opts = {"mode": a.mode, "kv_bytes": a.kv_bytes, "runtime_id": a.runtime_id}
        if a.command == "tune":
            doc = build_profile(*common, read_json(a.baseline), evidence=a.evidence,
                                diagnostic_screen=a.diagnostic_screen, **opts)
            code = 0 if not doc["missing_identity"] and capacity_usable(doc["capacity_plan"]) else 2
        else:
            doc = check_profile(read_json(a.profile), *common, **opts)
            code = 0 if doc["valid"] else 2
        text = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if a.out:
            with Path(a.out).open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            sys.stdout.write(text)
        return code
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
