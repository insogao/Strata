"""DI02 bounded diagnostic probe: reuses the unchanged lease, transport and telemetry.

Own tool only.  No engine build/source change.  One arm = one LeasedNative process.
The suite is fixed in this file and uses the registered fixtures/config; L256 is a
supplementary registered case built from the same 3817-token L fixture.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import strata_experiment as ex
from leased_native import LeasedNative, ResourceMonitor, save as save_json
from strata_diagnostics import Sampler, validate_decode_diag_record
from strata_workloads import probabilities

DIAG_MARKER = "strata-decode-diag-v1"
DIAG_REQUEST_PREFIXES = ("GEN ", "SCORE ")
DIAG_ENV = "STRATA_QWEN35_DECODE_DIAGNOSTICS"
COFETCH_ENV = "STRATA_QWEN35_ID_ACT_COFETCH"
COFETCH_MARKER = "strata qwen35: cached-tier router-id/activation cofetch enabled"
REGISTERED_BASE_ENV = {
    "STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE": "1",
    "STRATA_QWEN35_PACKED_IDS": "1",
    "STRATA_QWEN35_EXPERT_ORDER": "count_desc",
}
DI02_KIND = "di02-diagnostic-paired"
COFETCH_KIND = "id-act-cofetch"
SKIP_PROBE_ENV = "STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE"
SKIP_PROBE_MARKER = "strata qwen35: hit-overlap probe counters disabled"
SKIP_PROBE_KIND = "skip-hit-overlap-probe"

ARM_ORDER = [
    {"id": "legacy-ea06-OFF", "engine": "legacy", "diag": False},
    {"id": "new-OFF-A1", "engine": "new", "diag": False},
    {"id": "new-ON-B1", "engine": "new", "diag": True},
    {"id": "new-ON-B2", "engine": "new", "diag": True},
    {"id": "new-OFF-A2", "engine": "new", "diag": False},
]
REMAINING_ARM_IDS = ("new-OFF-A1", "new-ON-B1", "new-ON-B2", "new-OFF-A2")
REMAINING3_ARM_IDS = ("new-ON-B1", "new-ON-B2", "new-OFF-A2")
COFETCH_ARM_ORDER = [
    {"id": "cofetch-OFF-smoke", "engine": "new", "diag": True, "cofetch": False, "phase": "smoke"},
    {"id": "cofetch-ON-smoke", "engine": "new", "diag": True, "cofetch": True, "phase": "smoke"},
    {"id": "cofetch-OFF-A1", "engine": "new", "diag": False, "cofetch": False, "phase": "formal"},
    {"id": "cofetch-ON-B1", "engine": "new", "diag": False, "cofetch": True, "phase": "formal"},
    {"id": "cofetch-ON-B2", "engine": "new", "diag": False, "cofetch": True, "phase": "formal"},
    {"id": "cofetch-OFF-A2", "engine": "new", "diag": False, "cofetch": False, "phase": "formal"},
]
COFETCH_IDS = tuple(arm["id"] for arm in COFETCH_ARM_ORDER)
COFETCH_SMOKE_IDS = COFETCH_IDS[:2]
COFETCH_FORMAL_IDS = COFETCH_IDS[2:]
SKIP_PROBE_ARM_ORDER = [
    {"id": "skipprobe-OFF-smoke", "engine": "new", "diag": True, "skip_probe": False, "phase": "smoke"},
    {"id": "skipprobe-ON-smoke", "engine": "new", "diag": True, "skip_probe": True, "phase": "smoke"},
    {"id": "skipprobe-OFF-A1", "engine": "new", "diag": False, "skip_probe": False, "phase": "formal"},
    {"id": "skipprobe-ON-B1", "engine": "new", "diag": False, "skip_probe": True, "phase": "formal"},
    {"id": "skipprobe-ON-B2", "engine": "new", "diag": False, "skip_probe": True, "phase": "formal"},
    {"id": "skipprobe-OFF-A2", "engine": "new", "diag": False, "skip_probe": False, "phase": "formal"},
]
SKIP_PROBE_IDS = tuple(arm["id"] for arm in SKIP_PROBE_ARM_ORDER)
SKIP_PROBE_SMOKE_IDS = SKIP_PROBE_IDS[:2]
SKIP_PROBE_FORMAL_IDS = SKIP_PROBE_IDS[2:]
THREADS_KIND = "cpu-threads-config"
THREADS_ARM_ORDER = [
    {"id": "threads8-OFF-smoke", "engine": "new", "diag": True, "threads": 8, "phase": "smoke"},
    {"id": "threads10-ON-smoke", "engine": "new", "diag": True, "threads": 10, "phase": "smoke"},
    {"id": "threads8-OFF-A1", "engine": "new", "diag": False, "threads": 8, "phase": "formal"},
    {"id": "threads10-ON-B1", "engine": "new", "diag": False, "threads": 10, "phase": "formal"},
    {"id": "threads10-ON-B2", "engine": "new", "diag": False, "threads": 10, "phase": "formal"},
    {"id": "threads8-OFF-A2", "engine": "new", "diag": False, "threads": 8, "phase": "formal"},
]
THREADS_IDS = tuple(arm["id"] for arm in THREADS_ARM_ORDER)
THREADS_SMOKE_IDS = THREADS_IDS[:2]
THREADS_FORMAL_IDS = THREADS_IDS[2:]
FEATURE_PROFILES = {
    COFETCH_KIND: {
        "kind": COFETCH_KIND, "feature": True,
        "key": "cofetch", "env": COFETCH_ENV, "marker": COFETCH_MARKER,
        "marker_count_key": "cofetch_marker_count", "gate_prefix": "cofetch",
        "arms": COFETCH_ARM_ORDER, "ids": COFETCH_IDS,
        "smoke_ids": COFETCH_SMOKE_IDS, "formal_ids": COFETCH_FORMAL_IDS,
    },
    SKIP_PROBE_KIND: {
        "kind": SKIP_PROBE_KIND, "feature": True,
        "key": "skip_probe", "env": SKIP_PROBE_ENV, "marker": SKIP_PROBE_MARKER,
        "marker_count_key": "skip_probe_marker_count", "gate_prefix": "skipprobe",
        "arms": SKIP_PROBE_ARM_ORDER, "ids": SKIP_PROBE_IDS,
        "smoke_ids": SKIP_PROBE_SMOKE_IDS, "formal_ids": SKIP_PROBE_FORMAL_IDS,
    },
    THREADS_KIND: {
        "kind": THREADS_KIND, "feature": False,
        "key": "threads", "env": None, "marker": None,
        "marker_count_key": None, "gate_prefix": "threads",
        "arms": THREADS_ARM_ORDER, "ids": THREADS_IDS,
        "smoke_ids": THREADS_SMOKE_IDS, "formal_ids": THREADS_FORMAL_IDS,
    },
}
FEATURE_KEYS = tuple(profile["key"] for profile in FEATURE_PROFILES.values()
                     if profile.get("feature"))
COFETCH_COMPARE_NAMES = ("P0", "P1", "S0", "S1", "L0", "L1", "L2560", "L2561",
                         "SCORE0", "SCORE1", "gen-held", "eos")
COFETCH_GAIN_THRESHOLD = 0.05
COFETCH_REGRESSION_LIMIT = 0.15
LEDGER_SCHEMA = "strata-di02-native-ledger/1"
REGISTERED_HOST_MIB = 14000
COMMIT_FLOOR_MIB = 1024
PLANNED_PROMPTS = (("P0", "P"), ("P1", "P"), ("S0", "S"), ("S1", "S"),
                   ("L0", "L"), ("L1", "L"), ("L2560", "L256"), ("L2561", "L256"),
                   ("resetA", "P"), ("SCORE0", "P"), ("resetB", "P"), ("SCORE1", "P"),
                   ("score-held", "L"), ("gen-held", "L"), ("reset-eos", "P"),
                   ("score-eos", "P"), ("eos", "P"))
COFETCH_REQUIRED_NAMES = (
    "P0", "P1", "S0", "S1", "L0", "L1", "L2560", "L2561",
    "resetA", "SCORE0", "resetB", "SCORE1", "score-held", "gen-held", "state-held",
    "reset-eos", "score-eos", "eos", "state-eos",
)


class ResourceRefused(ValueError):
    """Preflight refusal: no lease, native process or protocol command may start."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _receipt_identity(result):
    """Stable identity of one lease result; copies of one launch share it."""
    commands = result.get("commands") or []
    fields = {
        "binary": result.get("binary_sha256_actual"),
        "argv": [c.get("argv") for c in commands],
        "pid": [c.get("pid") for c in commands],
        "started_at": [c.get("started_at") for c in commands],
        "lease_started": result.get("started_at"),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def _raw_ready(native_dir):
    raw = Path(native_dir) / "raw"
    for path in sorted(raw.glob("child_*.log")) if raw.is_dir() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?m)^READY\b", text):
            return True
    return False


def inspect_native_dir(native_dir):
    """Read one arm's raw lease receipt without trusting successful-arm lists."""
    native_dir = Path(native_dir)
    result_path = native_dir / "result.json"
    entry = {"native_dir": str(native_dir), "receipt_exists": result_path.is_file(),
             "receipt_identity": None, "binary_launch": None, "model_ready": None,
             "protocol_sent": None, "protocol_completed": None, "error": None}
    if not result_path.is_file():
        try:
            entry["model_ready"] = _raw_ready(native_dir)
        except Exception as exc:
            entry["error"] = "receipt unavailable: " + repr(exc)
        return entry
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        entry["error"] = "invalid receipt: " + repr(exc)
        return entry
    commands = result.get("commands") or []
    entry["receipt_identity"] = _receipt_identity(result)
    entry["binary_launch"] = bool(commands and any(
        c.get("pid") is not None or c.get("started_at") for c in commands))
    entry["model_ready"] = _raw_ready(native_dir)
    transcript = native_dir / "transcript.json"
    if transcript.is_file():
        try:
            data = json.loads(transcript.read_text(encoding="utf-8"))
            entry["protocol_sent"] = len(data.get("commands") or [])
            entry["protocol_completed"] = len(data.get("responses") or {})
        except Exception as exc:
            entry["error"] = "invalid transcript: " + repr(exc)
    return entry


def _aggregate_known(entries, key):
    """Sum a counter only when every unique entry reports it; a null stays unknown."""
    known = [e[key] for e in entries if e.get(key) is not None]
    unknown = len(entries) - len(known)
    return (None if unknown else sum(known)), sum(known), unknown


def reconcile_receipts(root):
    """Audit every unique raw native receipt below root; never overwrite originals."""
    entries = []
    for native_dir in sorted(path for path in Path(root).rglob("native") if path.is_dir()):
        entry = inspect_native_dir(native_dir)
        entry["receipt_path"] = str(native_dir / "result.json")
        entries.append(entry)
    unique, seen = [], set()
    for entry in entries:
        identity = entry.get("receipt_identity")
        if identity is not None and identity in seen:
            entry["duplicate_of"] = identity
            continue
        if identity is not None:
            seen.add(identity)
        unique.append(entry)
    sent, sent_bound, sent_unknown = _aggregate_known(unique, "protocol_sent")
    completed, completed_bound, completed_unknown = _aggregate_known(
        unique, "protocol_completed")
    totals = {
        "receipts": len(entries), "unique_receipts": len(unique),
        "receipts_missing": sum(1 for e in entries if not e.get("receipt_exists")),
        "receipts_without_identity": sum(1 for e in entries if e.get("receipt_exists")
                                         and e.get("receipt_identity") is None),
        "lease_attempts": sum(1 for e in unique if e.get("receipt_exists")),
        "binary_launches": sum(1 for e in unique if e.get("binary_launch") is True),
        "binary_launch_unknown": sum(1 for e in unique if e.get("binary_launch") is None
                                     and not e.get("duplicate_of")),
        "model_ready": sum(1 for e in unique if e.get("model_ready") is True),
        "protocol_sent": sent,
        "protocol_sent_known_lower_bound": sent_bound,
        "protocol_sent_unknown_receipts": sent_unknown,
        "protocol_completed": completed,
        "protocol_completed_known_lower_bound": completed_bound,
        "protocol_completed_unknown_receipts": completed_unknown,
    }
    return {"schema": "strata-di02-receipt-audit/1", "root": str(Path(root).resolve()),
            "utc": utc_now(), "entries": entries, "totals": totals}


class NativeLedger:
    """Persisted before/at every send; failed commands still consume an attempt."""

    def __init__(self, path, budgets, started_at):
        self.path = Path(path)
        self.budgets = budgets
        self.started_at = started_at
        self.records = []
        self.totals = {
            "lease_attempts": 0, "binary_launches": 0, "binary_launch_unknown": 0,
            "model_ready": 0, "successful_arms": 0, "protocol_attempts": 0,
            "protocol_completed": 0,
        }
        self.save()

    def save(self):
        save_json(self.path, {"schema": LEDGER_SCHEMA, "started_at": self.started_at,
                              "utc": utc_now(), "budgets": self.budgets,
                              "totals": self.totals, "records": self.records})

    def start_arm(self, index, arm, max_native, native_dir):
        if index < 1:
            raise ValueError("arm index must be positive")
        if self.totals["lease_attempts"] >= max_native:
            raise ValueError("registered native budget exhausted before lease attempt")
        record = {"arm": arm["id"], "index": index, "native_dir": str(native_dir),
                  "lease_attempt": True, "binary_launch": None, "model_ready": None,
                  "successful_arm": False, "protocol_attempts": 0,
                  "protocol_completed": 0, "constructor_error": None,
                  "receipt_identity": None, "started_utc": utc_now()}
        self.records.append(record)
        self.totals["lease_attempts"] += 1
        self.save()
        return record

    def constructor_failed(self, record, native_dir, error):
        receipt = inspect_native_dir(native_dir)
        record["constructor_error"] = repr(error)
        record["binary_launch"] = receipt["binary_launch"]
        record["model_ready"] = receipt["model_ready"]
        record["receipt_identity"] = receipt["receipt_identity"]
        if receipt["binary_launch"] is True:
            self.totals["binary_launches"] += 1
        elif receipt["binary_launch"] is None:
            self.totals["binary_launch_unknown"] += 1
        if receipt["model_ready"] is True:
            self.totals["model_ready"] += 1
        self.save()

    def arm_ready(self, record, native, native_dir):
        receipt = inspect_native_dir(native_dir)
        record["binary_launch"] = True
        record["model_ready"] = True
        record["receipt_identity"] = receipt["receipt_identity"]
        self.totals["binary_launches"] += 1
        self.totals["model_ready"] += 1
        self.save()

    def before_protocol(self, record, name, command):
        per_native = self.budgets["max_internal_protocol_requests_per_native"]
        total = self.budgets["max_internal_protocol_requests_total"]
        if record["protocol_attempts"] >= per_native:
            raise ValueError("per-native protocol limit exceeded before send: " + name)
        if self.totals["protocol_attempts"] >= total:
            raise ValueError("total protocol limit exceeded before send: " + name)
        record["protocol_attempts"] += 1
        self.totals["protocol_attempts"] += 1
        record.setdefault("commands", []).append({"name": name, "command": command,
                                                  "utc": utc_now(), "state": "sent"})
        self.save()

    def protocol_failed(self, record, name, error):
        for row in reversed(record.get("commands", [])):
            if row["name"] == name:
                row["state"] = "failed"
                row["error"] = repr(error)
                break
        self.save()

    def protocol_completed(self, record, name):
        record["protocol_completed"] += 1
        self.totals["protocol_completed"] += 1
        for row in reversed(record.get("commands", [])):
            if row["name"] == name:
                row["state"] = "completed"
                break
        self.save()

    def arm_complete(self, record):
        record["successful_arm"] = True
        self.totals["successful_arms"] += 1
        self.save()


def validate_generation_counts(answer, max_new):
    done = answer["done"]
    n = len(answer["tokens"])
    if done.get("produced") != n:
        raise ValueError("GEN produced %r != observed %d" % (done.get("produced"), n))
    if not 0 <= n <= max_new:
        raise ValueError("GEN token count %d outside 0..%d" % (n, max_new))
    if done.get("finish") == "stop" and (not answer["tokens"] or answer["tokens"][-1] != 248046):
        raise ValueError("EOS/finish mismatch")
    return n


def experiment_kind(manifest):
    kind = manifest.get("experiment_kind", DI02_KIND)
    if kind not in (DI02_KIND,) + tuple(FEATURE_PROFILES):
        raise ValueError("unknown experiment_kind %r" % (kind,))
    return kind


def feature_profile(arm):
    """The exact registered feature profile of one arm, or None for DI02/legacy arms."""
    keys = [key for key in FEATURE_KEYS if key in arm]
    if len(keys) > 1:
        raise ValueError("arm %r mixes registered feature flags" % (arm.get("id"),))
    if not keys:
        return None
    for profile in FEATURE_PROFILES.values():
        if profile["key"] == keys[0]:
            return profile
    raise ValueError("unknown feature key %r" % (keys[0],))


def manifest_profile(manifest):
    return FEATURE_PROFILES.get(experiment_kind(manifest))


def validate_arm_definition(arm):
    allowed = {"id", "engine", "diag"}
    profile = feature_profile(arm)
    if profile is not None:
        allowed |= {profile["key"], "phase"}
        if type(arm["diag"]) is not bool or type(arm[profile["key"]]) is not bool:
            raise ValueError("feature arm diagnostic/feature flags must be booleans")
        if arm["engine"] != "new":
            raise ValueError("feature arms must use the registered new binary")
        if arm.get("phase") not in ("smoke", "formal"):
            raise ValueError("feature arm phase must be smoke or formal")
    if "threads" in arm:
        if profile is not None:
            raise ValueError("config arm must not mix a feature gate")
        allowed |= {"threads", "phase"}
        if type(arm["threads"]) is not int or arm["threads"] not in (8, 10):
            raise ValueError("threads arm must be exactly 8 or 10")
        if type(arm["diag"]) is not bool:
            raise ValueError("threads arm diagnostic flag must be boolean")
        if arm["engine"] != "new":
            raise ValueError("threads arms must use the registered new binary")
        if arm.get("phase") not in ("smoke", "formal"):
            raise ValueError("threads arm phase must be smoke or formal")
    unknown = set(arm) - allowed
    if unknown:
        raise ValueError("unknown arm feature flags: " + ", ".join(sorted(unknown)))
    return arm


def select_arm_order(manifest):
    profile = manifest_profile(manifest)
    if profile is not None:
        by_id = {arm["id"]: arm for arm in profile["arms"]}
        order = manifest.get("arm_order")
        registered = (list(profile["ids"]), list(profile["smoke_ids"]),
                      list(profile["formal_ids"]))
        if order is None:
            order = list(profile["ids"])
        if order not in registered:
            raise ValueError("manifest arm_order is neither the full six, exact smoke-two "
                             "nor exact formal-four registered feature order")
        return [copy.deepcopy(by_id[arm_id]) for arm_id in order]
    by_id = {arm["id"]: arm for arm in ARM_ORDER}
    order = manifest.get("arm_order")
    if order is None:
        return list(ARM_ORDER)
    registered = ([a["id"] for a in ARM_ORDER], list(REMAINING_ARM_IDS),
                  list(REMAINING3_ARM_IDS))
    if order not in registered:
        raise ValueError("manifest arm_order is neither the full, remaining-four nor remaining-three "
                         "registered order")
    return [by_id[arm_id] for arm_id in order]


def validate_arm_options(manifest, max_arms):
    budgets = manifest.get("budgets") or {}
    registered = budgets.get("gpu_native_launches")
    if not isinstance(registered, int) or registered < 1:
        raise ValueError("manifest has no positive registered native budget")
    order = select_arm_order(manifest)
    if not isinstance(max_arms, int) or max_arms < 1:
        raise ValueError("--max-arms must be positive")
    if max_arms > registered:
        raise ValueError("--max-arms %d exceeds registered native budget %d" % (max_arms, registered))
    if max_arms != len(order):
        raise ValueError("--max-arms %d differs from selected arm order count %d" % (max_arms, len(order)))
    for key in ("max_internal_protocol_requests_per_native", "max_internal_protocol_requests_total"):
        if not isinstance(budgets.get(key), int) or budgets[key] < 1:
            raise ValueError("manifest budget %s must be positive" % key)
    return order, budgets


def preflight_resource_guard(spec, manifest, fingerprint_fn=None, admission_fn=None,
                             monitor_factory=None):
    """Read-only admission; every refusal is fail-closed before any lease exists."""
    fingerprint_fn = ex.fingerprint if fingerprint_fn is None else fingerprint_fn
    admission_fn = ex.admission if admission_fn is None else admission_fn
    monitor_factory = ResourceMonitor if monitor_factory is None else monitor_factory
    fp = fingerprint_fn(spec)
    model = fp.get("model", {})
    model_size = (model.get("size_bytes") or {}).get("value")
    reasons = []
    registered_size = manifest.get("model_size")
    if registered_size is not None and model_size is not None and model_size != registered_size:
        reasons.append("model size %r != registered %r" % (model_size, registered_size))
    if manifest.get("model_sha256") is not None and \
            (model.get("sha256") or {}).get("value") != manifest.get("model_sha256"):
        reasons.append("fingerprint model SHA differs from registered SHA")
    admission = admission_fn(spec, fp, spec["baseline"]["config"])
    reasons.extend(admission.get("reasons") or [])
    monitor = monitor_factory(spec["resources"])
    try:
        sample = monitor.sample()
    except Exception as exc:
        sample = None
        reasons.append("resource monitor unavailable: " + repr(exc))
    if monitor.violation:
        reasons.append(monitor.violation)
    ram = fp.get("ram", {})
    commit = (ram.get("commit_avail_mib") or {}).get("value")
    registered_host = max(spec["resources"].get("estimated_host_mib", REGISTERED_HOST_MIB),
                          REGISTERED_HOST_MIB)
    floor = registered_host + COMMIT_FLOOR_MIB
    if commit is None:
        reasons.append("commit headroom unavailable")
    elif commit < floor:
        reasons.append("commit headroom %.1fMiB < %.0fMiB floor" % (commit, floor))
    if sample is not None:
        for key in ("gpu_used_mib", "gpu_total_mib", "available_ram_mib", "commit_available_mib"):
            if sample.get(key) is None:
                reasons.append("monitor %s unavailable" % key)
    return {"admitted": not reasons, "reasons": reasons, "admission": admission,
            "monitor": {"violation": monitor.violation, "sample": sample},
            "fingerprint": {"ram": ram, "gpu": fp.get("gpu"), "model_size": model_size,
                            "model_sha256": fp.get("model", {}).get("sha256", {}).get("value")},
            "floor_mib": floor}


def require_admitted(guard):
    if not guard.get("admitted"):
        raise ResourceRefused("resource admission refused: " + "; ".join(guard.get("reasons") or ["unknown"]))


def parse_diag_lines(lines):
    """Return (records, malformed) from already-split text lines."""
    records, malformed = [], []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if DIAG_MARKER not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:
            malformed.append({"line": number, "error": repr(exc), "text": line[:2000]})
            continue
        records.append({"line": number, "record": rec})
    return records, malformed


def validate_records(records):
    errors = []
    seen = set()
    for item in records:
        rec = item["record"]
        for problem in validate_decode_diag_record(rec):
            errors.append("line %s: %s" % (item["line"], problem))
        req = rec.get("req")
        if req in seen:
            errors.append("duplicate req %r" % (req,))
        seen.add(req)
    reqs = [item["record"].get("req") for item in records]
    if reqs != list(range(1, len(reqs) + 1)):
        errors.append("req order is %r, expected 1..%d" % (reqs, len(records)))
    return errors


def match_diag(record, expected):
    """Compare one real diagnostic record with the observed DONE metadata."""
    errors = validate_decode_diag_record(record)
    for key in ("kind", "prompt_tokens", "max_new", "base_pos", "produced", "finish",
                "cont", "session", "keep"):
        if record.get(key) != expected[key]:
            errors.append("%s %r != expected %r" % (key, record.get(key), expected[key]))
    return errors


def timing_metrics(row):
    """Observed file-reader timestamps; n < 2 never yields a TPS/TPOT."""
    n = row["n"]
    stamps = row["token_observation_s"]
    itl = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])] if n > 1 else []
    span = (stamps[-1] - stamps[0]) if n > 1 else None
    return {
        "ttft_observed_ms": (stamps[0] - row["started_s"]) * 1000.0 if n else None,
        "itl_observed_ms": itl,
        "tpot_observed_ms": span * 1000.0 / (n - 1) if span and span > 0 else None,
        "decode_tps_observed": (n - 1) / span if span and span > 0 else None,
        "e2e_observed_ms": (row["finished_s"] - row["started_s"]) * 1000.0,
        "last_visible_observed_ms": (stamps[-1] - row["started_s"]) * 1000.0 if n else None,
        "timing_scope": "childlog file observation, 5 ms polling after engine fflush; not exact flush",
    }


def context_margin(prompt_tokens, max_new, ctx):
    margin = ctx - prompt_tokens - max_new
    if margin < 0:
        raise ValueError("prompt+max_new exceeds context: %d+%d>%d" % (prompt_tokens, max_new, ctx))
    return margin


class DiagTail:
    """Independent append-only reader of the merged child log."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.partial = b""
        self.records = []
        self.observed = []
        self.malformed = []
        self.live_errors = []

    def poll(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
            self.offset += len(data)
        parts = (self.partial + data).split(b"\n")
        self.partial = parts.pop()
        found = []
        for raw in parts:
            line = raw.rstrip(b"\r").decode("utf-8", "replace").strip()
            if DIAG_MARKER not in line:
                continue
            observed = time.monotonic()
            try:
                rec = json.loads(line)
            except Exception as exc:
                self.malformed.append({"error": repr(exc), "text": line[:2000]})
                found.append(None)
                continue
            self.records.append(rec)
            self.observed.append(observed)
            found.append((rec, observed))
        return found

    def wait_next(self, expected_count, deadline):
        while time.monotonic() < deadline:
            batch = self.poll()
            for item in batch:
                if item is None:
                    raise ValueError("malformed diagnostic JSON in live child log")
            if self.malformed:
                raise ValueError("malformed diagnostic JSON in live child log")
            if len(self.records) >= expected_count:
                return self.records[expected_count - 1], self.observed[expected_count - 1]
            time.sleep(0.005)
        raise TimeoutError("diagnostic record missing before bounded operation deadline")


def _series(samples, getter):
    out = []
    for sample in samples:
        try:
            value = getter(sample)
        except (KeyError, TypeError):
            value = None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out.append(float(value))
    return out


def _summary(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def telemetry_summary(telemetry):
    samples = telemetry.get("samples", [])
    return {
        "samples": len(samples),
        "gpu_used_mib": _summary(_series(samples, lambda s: s["gpu"]["mem_used"] / 2 ** 20)),
        "gpu_util_pct": _summary(_series(samples, lambda s: s["gpu"].get("util"))),
        "power_w": _summary(_series(samples, lambda s: s["gpu"].get("power"))),
        "power_limit_w": _summary(_series(samples, lambda s: s["gpu"].get("power_limit"))),
        "temp_c": _summary(_series(samples, lambda s: s["gpu"].get("temp"))),
        "sm_mhz": _summary(_series(samples, lambda s: s["gpu"].get("sm_mhz"))),
        "memory_mhz": _summary(_series(samples, lambda s: s["gpu"].get("memory_mhz"))),
        "cpu_mhz_proxy": _summary(_series(samples, lambda s: s.get("cpu_mhz_proxy"))),
        "available_ram_mib": _summary(_series(samples, lambda s: s["available_ram_bytes"] / 2 ** 20)),
        "host_rss_mib": _summary(_series(samples, lambda s: sum(p["memory"]["rss"] for p in s.get("processes", [])) / 2 ** 20)),
        "pstate_seen": sorted({int(s["gpu"]["pstate"]) for s in samples if s["gpu"].get("pstate") is not None}),
        "throttle_reasons_nonzero": sorted({int(s["gpu"]["throttle_reasons"]) for s in samples
                                            if s["gpu"].get("throttle_reasons") not in (None, 0)}),
        "errors": telemetry.get("errors", []),
    }


def graph_evidence(raw_text):
    match = re.search(r"sched_reserve: graph: nodes = (\d+), splits = (\d+), input objects = (\d+), input tensors = (\d+)",
                      raw_text)
    if not match:
        return {"reservation_observed": None, "coverage": None,
                "reason": "no sched_reserve graph line in the captured child log"}
    return {"reservation_observed": True, "nodes": int(match.group(1)), "splits": int(match.group(2)),
            "input_objects": int(match.group(3)), "input_tensors": int(match.group(4)),
            "coverage": None, "coverage_reason": "graph node reservation is not per-request coverage"}


def build_prompts(spec):
    prompts = {}
    for shape in spec["shapes"]:
        ids = ex.token_ids(shape["tokens_file"])
        if len(ids) != shape["prompt_tokens"]:
            raise ValueError("fixture count mismatch " + shape["id"])
        prompts[shape["id"]] = {"ids": ids, "csv": ",".join(map(str, ids))}
    if "L" not in prompts:
        raise ValueError("prompt plan missing L fixture")
    prompts["L256"] = prompts["L"]
    return prompts


def validate_prompt_plan(prompts):
    for name, shape in PLANNED_PROMPTS:
        if shape not in prompts:
            raise ValueError("prompt plan missing shape %s for %s" % (shape, name))
        if not prompts[shape]["ids"] or not prompts[shape]["csv"]:
            raise ValueError("prompt plan empty shape %s for %s" % (shape, name))


def make_arm_spec(base, arm, manifest):
    validate_arm_definition(arm)
    profile = feature_profile(arm)
    spec = copy.deepcopy(base)
    if arm["engine"] == "legacy":
        spec["engine"] = {"path": manifest["legacy_exe"], "sha256": manifest["legacy_exe_sha256"]}
        spec["runtime"]["id"] = "legacy-ea06-compat"
        if sha_file(spec["engine"]["path"]) != spec["engine"]["sha256"]:
            raise ValueError("legacy engine SHA mismatch")
    if profile is not None:
        if spec["runtime"]["env"] != REGISTERED_BASE_ENV:
            raise ValueError("feature base runtime env must be exactly the registered3 flags")
        env = dict(REGISTERED_BASE_ENV)
        if arm["diag"]:
            env[DIAG_ENV] = "1"
        if arm[profile["key"]]:
            env[profile["env"]] = "1"
        spec["runtime"]["env"] = env
    elif "threads" in arm:
        if spec["runtime"]["env"] != REGISTERED_BASE_ENV:
            raise ValueError("threads base runtime env must be exactly the registered3 flags")
        env = dict(REGISTERED_BASE_ENV)
        if arm["diag"]:
            env[DIAG_ENV] = "1"
        spec["runtime"]["env"] = env
        spec["baseline"]["config"]["threads"] = arm["threads"]
    elif arm["diag"]:
        spec["runtime"]["env"] = dict(spec["runtime"]["env"],
                                      STRATA_QWEN35_DECODE_DIAGNOSTICS="1")
    return spec


def validate_feature_marker(arm, raw_text):
    """Persisted feature identity: ON once, OFF never; any other count is unsafe."""
    profile = feature_profile(arm)
    if profile is None:
        return None
    count = raw_text.count(profile["marker"])
    if arm[profile["key"]]:
        if count != 1:
            raise ValueError("%s ON marker count %d != 1" % (profile["key"], count))
    elif count != 0:
        raise ValueError("%s OFF emitted %d marker(s)" % (profile["key"], count))
    return count


validate_cofetch_marker = validate_feature_marker


def close_artifacts(out, native_dir, telemetry_path=None):
    telemetry_path = Path(out) / "telemetry.json" if telemetry_path is None else Path(telemetry_path)
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    resources = json.loads((Path(native_dir) / "resources.json").read_text(encoding="utf-8"))
    return telemetry, resources


OPTIONAL_DIAG_FIELDS = ("diag", "diag_index", "send_to_diag_ms", "diag_observed_s",
                        "diag_after_done_ms")
REQUIRED_ROW_FIELDS = ("name", "command", "kind", "n", "started_s", "finished_s",
                       "token_ids", "token_ids_sha256", "token_observation_s")


def validate_recorded_rows(requests):
    """Null optional diagnostics; missing required timing/token fields still fail."""
    for row in requests:
        for key in OPTIONAL_DIAG_FIELDS:
            row.setdefault(key, None)
        missing = [key for key in REQUIRED_ROW_FIELDS if key not in row]
        if missing:
            raise ValueError("request %r is missing required field(s) %s"
                             % (row.get("name"), ", ".join(missing)))
        if row["kind"] in ("gen", "score"):
            for key in ("finish", "prompt_tokens", "prompt_read", "reused",
                        "done_prompt_ms", "done_decode_ms"):
                if key not in row:
                    raise ValueError("request %r is missing required field %s"
                                     % (row["name"], key))
        if len(row["token_ids"]) != row["n"]:
            raise ValueError("request %r token count differs from n" % row["name"])
        if len(row["token_observation_s"]) != row["n"]:
            raise ValueError("request %r observation count differs from n" % row["name"])
        digest = hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest()
        if digest != row["token_ids_sha256"]:
            raise ValueError("request %r token SHA mismatch" % row["name"])
    return requests


def finalize_arm(arm, out, spec, requests, responses, *, native_dir=None, telemetry_path=None,
                 raw_text=None, load_ready_s=None, native_started=None, native_exit=None,
                 live_diag_rows=None, provenance=None):
    """Shared live/offline finalization: raw replay, artifacts, honest summary rows."""
    out = Path(out)
    native_dir = out / "native" if native_dir is None else Path(native_dir)
    if raw_text is None:
        raw_text = (native_dir / "raw/child_00.log").read_text(encoding="utf-8",
                                                               errors="replace")
    cofetch_marker_count = validate_feature_marker(arm, raw_text)
    validate_recorded_rows(requests)
    if not requests:
        raise ValueError("no recorded requests to finalize")
    if not isinstance(native_exit, int):
        raise ValueError("native exit code unavailable")
    lines = raw_text.splitlines()
    records, malformed = parse_diag_lines(lines)
    validation = validate_records(records)
    if arm["diag"]:
        if malformed:
            raise ValueError("malformed diagnostic JSON in raw child log")
        expected = len(live_diag_rows) if live_diag_rows is not None else \
            sum(1 for row in requests if row.get("diag") is not None)
        if len(records) != expected:
            raise ValueError("raw diagnostic row count %d != expected %d"
                             % (len(records), expected))
        if live_diag_rows is not None:
            for live, final in zip(live_diag_rows, [item["record"] for item in records]):
                if live != final:
                    raise ValueError("live diagnostic record differs from raw replay")
        if validation:
            raise ValueError("raw diagnostic validation: " + "; ".join(validation))
    else:
        if records or malformed:
            raise ValueError("OFF arm emitted diagnostic JSON")
    errs = [line for line in lines if line.startswith("ERR ")]
    if errs:
        raise ValueError("engine ERR line in child log: " + errs[0])
    telemetry, resources = close_artifacts(out, native_dir, telemetry_path)
    if "violation" not in resources:
        raise ValueError("resources.json lacks the required violation field")
    if resources.get("violation"):
        raise ValueError(resources["violation"])
    sends = [r["started_s"] for r in requests]
    dones = [r["finished_s"] for r in requests]
    diag_times = [r["diag_observed_s"] for r in requests if r["diag_observed_s"] is not None]
    summary = {
        "arm": arm, "engine_sha256": spec["engine"]["sha256"],
        "runtime_env": spec["runtime"]["env"], "config": spec["baseline"]["config"],
        "load_ready_s": load_ready_s,
        "load_to_first_done_s": dones[0] - native_started if native_started is not None else None,
        "batch_first_send_to_last_done_ms": (dones[-1] - sends[0]) * 1000.0,
        "batch_first_send_to_last_diag_ms": (diag_times[-1] - sends[0]) * 1000.0 if diag_times else None,
        "batch_wall_including_tailflush_ms": ((max(dones[-1], diag_times[-1] if diag_times else dones[-1]))
                                              - sends[0]) * 1000.0,
        "request_count": len(requests), "diag_count": len(records),
        "diag_overhead_after_done_ms": _summary([r.get("diag_after_done_ms") for r in requests
                                                 if r.get("diag_after_done_ms") is not None]),
        "resources": resources, "telemetry": telemetry_summary(telemetry),
        "graph": graph_evidence(raw_text),
        "native_exit": native_exit,
        "raw_childlog_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    profile = feature_profile(arm)
    if profile is not None:
        summary["arm_features"] = {"diag": arm["diag"], profile["key"]: arm[profile["key"]],
                                   "phase": arm["phase"]}
        summary[profile["marker_count_key"]] = cofetch_marker_count
    if provenance is not None:
        summary["offline_finalization"] = provenance
    save_json(out / "diag-records.json", {"records": [item["record"] for item in records],
                                          "malformed": malformed, "validation_errors": validation})
    save_json(out / "measurements.json", {"summary": summary, "requests": requests})
    save_json(out / "arm-summary.json", summary)
    return {"summary": summary, "requests": requests, "out": str(out), "responses": responses}


def recorded_responses(native_dir, requests, raw_text):
    """Recover responses from transcript.json, or ordered CANDLOGPROBS rows in raw."""
    transcript = Path(native_dir) / "transcript.json"
    if transcript.is_file():
        payload = json.loads(transcript.read_text(encoding="utf-8"))
        responses = payload.get("responses")
        if not isinstance(responses, dict):
            raise ValueError("transcript.json has no responses map")
        return responses, "native/transcript.json#responses"
    names = [row["name"] for row in requests if "cand=" in row["command"]]
    cand_lines = [line for line in raw_text.splitlines() if line.startswith("CANDLOGPROBS ")]
    if len(cand_lines) != len(names):
        raise ValueError("cannot recover %d candidate rows from raw log (found %d)"
                         % (len(names), len(cand_lines)))
    return ({name: {"lines": [line], "tokens": [], "token_observation_s": [],
                    "done": {"produced": 0, "finish": "score"}}
             for name, line in zip(names, cand_lines)},
            "raw child_00.log CANDLOGPROBS lines in request order")


def _recorded_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    return {"sha256": sha_file(path), "bytes": path.stat().st_size}


def finalize_recorded_arm(dataset_dir, arm_id, out_dir):
    """Offline full finalization of one already-recorded arm; no lease/GPU/model payload."""
    dataset = Path(dataset_dir)
    native_dir = dataset / "native"
    arm = next((item for item in ARM_ORDER if item["id"] == arm_id), None)
    if arm is None:
        raise ValueError("unknown recorded arm id " + arm_id)
    requests = json.loads((dataset / "requests.json").read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise ValueError("requests.json is not a request list")
    raw_text = (native_dir / "raw/child_00.log").read_text(encoding="utf-8", errors="replace")
    responses, responses_source = recorded_responses(native_dir, requests, raw_text)
    result = json.loads((native_dir / "result.json").read_text(encoding="utf-8"))
    job = json.loads((native_dir / "job.json").read_text(encoding="utf-8"))
    config_path = native_dir / "actual-configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if not job.get("binary_sha256"):
        raise ValueError("job.json lacks the engine SHA")
    recorded = {name: _recorded_file(dataset / name) for name in
                ("requests.json", "telemetry.json", "native/resources.json",
                 "native/result.json", "native/job.json", "native/actual-configuration.json",
                 "native/transcript.json", "native/raw/child_00.log")}
    provenance = {
        "label": "derived-recovery",
        "source_dataset": str(dataset),
        "arm_id": arm_id,
        "recorded_files": recorded,
        "responses_source": responses_source,
        "recorded_exit_code": result.get("exit_code"),
        "load_ready_s": None,
        "load_ready_reason": "native monotonic load timing was not persisted",
        "native_started": None,
        "native_started_reason": "native monotonic start was not persisted; batch metrics use row times",
        "runtime_env": None,
        "runtime_env_reason": "runtime env was not persisted; recording job command line only",
        "original_postprocess_failure": "KeyError diag_after_done_ms after 19 completed ops; "
                                        "original requests/raw are untouched",
    }
    spec = {"engine": {"path": job.get("binary"), "sha256": job["binary_sha256"]},
            "runtime": {"env": None},
            "baseline": {"config": config.get("requested") or {}}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return finalize_arm(arm, out, spec, requests, responses, native_dir=native_dir,
                        telemetry_path=dataset / "telemetry.json", raw_text=raw_text,
                        load_ready_s=None, native_started=None,
                        native_exit=result.get("exit_code"), live_diag_rows=None,
                        provenance=provenance)


def run_arm(out_root, index, arm, base_spec, manifest, started, prompts,
            ledger=None, native_factory=None):
    native_factory = LeasedNative if native_factory is None else native_factory
    out = Path(out_root) / ("arm-%02d-%s" % (index, arm["id"]))
    native_dir = out / "native"
    out.mkdir(parents=True, exist_ok=False)
    spec = make_arm_spec(base_spec, arm, manifest)
    clock = {"utc": utc_now(), "monotonic": time.monotonic(),
             "remaining_window_s": base_spec["budget"]["window_s"] - (time.time() - started),
             "arm": arm["id"], "diag": arm["diag"]}
    save_json(out / "clock-before-native.json", clock)
    if clock["remaining_window_s"] < base_spec["budget"]["native_timeout_s"] + base_spec["budget"]["wait_s"]:
        raise ValueError("continuation window insufficient; no native started")
    record = ledger.start_arm(index, arm, manifest["budgets"]["gpu_native_launches"],
                              native_dir) if ledger is not None else None
    try:
        native = native_factory(spec, spec["baseline"]["config"], native_dir, started)
    except BaseException as exc:
        if ledger is not None:
            ledger.constructor_failed(record, native_dir, exc)
        raise
    if ledger is not None:
        ledger.arm_ready(record, native, native_dir)
    sampler = Sampler(native)
    sampler.start()
    tail = DiagTail(native_dir / "raw/child_00.log") if arm["diag"] else None
    requests = []
    diag_rows = []
    failed = False
    session_active, cursor = False, 0
    last_done = None
    try:
        def ask(name, command, shape_id, gen_max=None, cont=False, keep=False, kind=None):
            nonlocal session_active, cursor, last_done
            if ledger is not None:
                ledger.before_protocol(record, name, command)
            try:
                answer = native.ask(name, command)
            except BaseException as exc:
                if ledger is not None:
                    ledger.protocol_failed(record, name, exc)
                raise
            if ledger is not None:
                ledger.protocol_completed(record, name)
            done = answer["done"]
            n = len(answer["tokens"])
            if kind == "gen":
                validate_generation_counts(answer, gen_max)
            elif kind == "score":
                if n != 0 or done.get("produced") != 0:
                    raise ValueError("SCORE completed with produced tokens " + name)
            row = {
                "name": name, "command": command, "shape": shape_id,
                "kind": kind, "n": n, "finish": done["finish"],
                "requested_max_new": gen_max, "prompt_tokens": done["prompt_tokens"],
                "prompt_read": done["prompt_read"], "reused": done["reused"],
                "done_prompt_ms": done["prompt_ms"], "done_decode_ms": done["decode_ms"],
                "prefill_tps_from_done": done["prompt_read"] * 1000.0 / done["prompt_ms"]
                if done["prompt_ms"] > 0 else None,
                "started_s": answer["started_s"], "finished_s": answer["finished_s"],
                "send_to_done_ms": (answer["finished_s"] - answer["started_s"]) * 1000.0,
                "token_ids": answer["tokens"],
                "token_ids_sha256": hashlib.sha256(json.dumps(answer["tokens"]).encode()).hexdigest(),
                "token_observation_s": answer["token_observation_s"],
                "diag": None, "send_to_diag_ms": None, "diag_observed_s": None,
                "diag_index": None, "diag_after_done_ms": None,
            }
            row.update(timing_metrics(row))
            if last_done is not None:
                row["gap_from_previous_done_ms"] = (answer["started_s"] - last_done) * 1000.0
            else:
                row["gap_from_previous_done_ms"] = None
            if kind:
                expected = {
                    "kind": kind, "prompt_tokens": done["prompt_read"], "max_new": gen_max or 0,
                    "base_pos": done["reused"], "produced": n, "finish": done["finish"],
                    "cont": bool(cont), "keep": bool(keep),
                    "session": bool(session_active) if not (kind == "score" and not keep) else False,
                }
                if kind == "gen":
                    if done["reused"] != (cursor if session_active else 0):
                        raise ValueError("GEN cursor/reused mismatch " + name)
                    if done["prompt_read"] != (0 if cont else len(prompts[shape_id]["ids"])):
                        raise ValueError("GEN prompt_read mismatch " + name)
                else:
                    expected["session"] = bool(keep and session_active)
                    if done["reused"] != (cursor if expected["session"] else 0):
                        raise ValueError("SCORE cursor/reused mismatch " + name)
                    if done["prompt_read"] != len(prompts[shape_id]["ids"]) and not (
                            name.startswith("reset") and command == "SCORE 0 topk=1"):
                        raise ValueError("SCORE prompt_read mismatch " + name)
                if arm["diag"]:
                    expected_count = len(diag_rows) + 1
                    deadline = min(answer["started_s"] + 60.0,
                                   native.started + spec["budget"]["native_timeout_s"])
                    rec, observed = tail.wait_next(expected_count, deadline)
                    errs = match_diag(rec, expected)
                    if errs:
                        raise ValueError("diagnostic mismatch %s: %s" % (name, "; ".join(errs)))
                    row["diag"] = rec
                    row["diag_index"] = len(diag_rows)
                    row["diag_observed_s"] = observed
                    row["send_to_diag_ms"] = (observed - answer["started_s"]) * 1000.0
                    row["diag_after_done_ms"] = (observed - answer["finished_s"]) * 1000.0
                    diag_rows.append(rec)
                if kind == "gen":
                    if done["finish"] not in {"length", "stop"}:
                        raise ValueError("GEN finish invalid " + name)
                    if done["finish"] == "length" and n != gen_max:
                        raise ValueError("length finish without max_new " + name)
                    if done["finish"] == "stop" and (not answer["tokens"] or answer["tokens"][-1] != 248046):
                        raise ValueError("EOS/finish mismatch " + name)
                    if session_active:
                        cursor = done["reused"] + done["prompt_read"] + n
                else:
                    if done["produced"] != 0 or done["finish"] != "score":
                        raise ValueError("SCORE completed with produced tokens " + name)
                    if keep:
                        cursor = done["reused"] + done["prompt_read"]
                        session_active = True
                    else:
                        cursor, session_active = 0, False
            last_done = answer["finished_s"]
            requests.append(row)
            save_json(out / "requests.json", requests)
            return answer, row

        def state(name, expected_pos):
            nonlocal last_done
            if ledger is not None:
                ledger.before_protocol(record, name, "STATE")
            try:
                answer = native.ask(name, "STATE")
            except BaseException as exc:
                if ledger is not None:
                    ledger.protocol_failed(record, name, exc)
                raise
            if ledger is not None:
                ledger.protocol_completed(record, name)
            payload = json.loads(answer["lines"][-1][6:])
            if payload.get("pos") != expected_pos:
                raise ValueError("%s cursor mismatch: %r != %r" % (name, payload.get("pos"), expected_pos))
            row = {"name": name, "command": "STATE", "shape": None, "kind": None,
                   "n": 0, "finish": "state", "requested_max_new": None,
                   "prompt_tokens": None, "prompt_read": None, "reused": None,
                   "done_prompt_ms": None, "done_decode_ms": None,
                   "prefill_tps_from_done": None,
                   "started_s": answer["started_s"], "finished_s": answer["finished_s"],
                   "send_to_done_ms": (answer["finished_s"] - answer["started_s"]) * 1000.0,
                   "token_ids": [], "token_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
                   "token_observation_s": [], "diag": None, "send_to_diag_ms": None,
                   "diag_observed_s": None, "diag_index": None,
                   "diag_after_done_ms": None, "state": payload,
                   "gap_from_previous_done_ms": (answer["started_s"] - last_done) * 1000.0
                   if last_done is not None else None}
            row.update(timing_metrics(row))
            last_done = answer["finished_s"]
            requests.append(row)
            save_json(out / "requests.json", requests)
            return payload

        P, S, L = prompts["P"], prompts["S"], prompts["L"]
        ctx = spec["baseline"]["config"]["ctx"]
        context_margin(len(L["ids"]), 256, ctx)

        ask("P0", "GEN 1 " + P["csv"], "P", gen_max=1, kind="gen")
        ask("P1", "GEN 1 " + P["csv"], "P", gen_max=1, kind="gen")
        ask("S0", "GEN 16 " + S["csv"], "S", gen_max=16, kind="gen")
        ask("S1", "GEN 16 " + S["csv"], "S", gen_max=16, kind="gen")
        ask("L0", "GEN 128 " + L["csv"], "L", gen_max=128, kind="gen")
        ask("L1", "GEN 128 " + L["csv"], "L", gen_max=128, kind="gen")
        ask("L2560", "GEN 256 " + L["csv"], "L256", gen_max=256, kind="gen")
        ask("L2561", "GEN 256 " + L["csv"], "L256", gen_max=256, kind="gen")

        ask("resetA", "SCORE 0 topk=1", "P", gen_max=0, kind="score")
        ask("SCORE0", "SCORE " + P["csv"] + " topk=0 cand=15,16,17,18", "P", gen_max=0, kind="score")
        ask("resetB", "SCORE 0 topk=1", "P", gen_max=0, kind="score")
        ask("SCORE1", "SCORE " + P["csv"] + " topk=0 cand=15,16,17,18", "P", gen_max=0, kind="score")
        ask("score-held", "SCORE " + L["csv"] + " topk=1 keep=1", "L", gen_max=0, keep=True, kind="score")
        ask("gen-held", "GEN 128 cont=1", "L", gen_max=128, cont=True, kind="gen")
        state("state-held", 3817 + requests[-1]["n"])
        ask("reset-eos", "SCORE 0 topk=1", "P", gen_max=0, kind="score")
        ask("score-eos", "SCORE " + P["csv"] + " topk=1 keep=1", "P", gen_max=0, keep=True, kind="score")
        ask("eos", "GEN 32 cont=1", "P", gen_max=32, cont=True, kind="gen")
        state("state-eos", 3896 + requests[-1]["n"])

        if arm["diag"]:
            if len(diag_rows) != 17:
                raise ValueError("ON expected 17 GEN/SCORE diagnostic rows, got %d" % len(diag_rows))
        elif tail is not None:
            raise AssertionError("OFF arm cannot have a diagnostic tail")
        if ledger is not None:
            if record["protocol_attempts"] != len(requests):
                raise ValueError("ledger attempts %d != observed requests %d"
                                 % (record["protocol_attempts"], len(requests)))
            if record["protocol_completed"] != len(requests):
                raise ValueError("ledger completions %d != observed requests %d"
                                 % (record["protocol_completed"], len(requests)))
    except BaseException as exc:
        failed = True
        save_json(out / "failure.json", {"error": repr(exc), "retry": False, "utc": utc_now()})
        raise
    finally:
        sampler.finish(out / "telemetry.json")
        native.close(abort=failed)
    receipt = native.receipt()
    result = finalize_arm(arm, out, spec, requests, native.responses,
                          native_dir=native_dir, load_ready_s=native.load_ready_s,
                          native_started=native.started,
                          native_exit=receipt["commands"][0]["exit_code"],
                          live_diag_rows=diag_rows)
    if ledger is not None:
        ledger.arm_complete(record)
    return result


def score_facts(requests, responses):
    facts = {}
    for row in requests:
        if row["kind"] != "score" or "cand=" not in row["command"]:
            continue
        answer = responses[row["name"]]
        prob = probabilities(answer, [15, 16, 17, 18])
        if any(not math.isfinite(p["logprob"]) for p in prob["candidate_logprobs"]):
            raise ValueError("nonfinite candidate logprob " + row["name"])
        facts[row["name"]] = prob
        if not math.isfinite(prob["label_mass"]) or not (0 <= prob["label_mass"] <= 1.0001):
            raise ValueError("nonfinite candidate probability " + row["name"])
    return facts


def json_visible(tokenizer, tokens):
    try:
        text = tokenizer.decode(tokens)
    except Exception as exc:
        return {"text": None, "json_complete": False, "json_visible": False, "error": repr(exc)}
    payload = None
    try:
        payload = json.loads(text)
    except Exception:
        pass
    complete = isinstance(payload, dict) and type(payload.get("risk")) is int and \
        payload["risk"] in range(4) and type(payload.get("violation")) is bool
    return {"text": text, "json_complete": complete,
            "json_visible": text.strip().startswith("{") and text.strip().endswith("}")}


def postprocess(results, model_path):
    import strata_tokenizer
    tokenizer = strata_tokenizer.Tokenizer.from_gguf(str(model_path))
    for result in results:
        responses = result.pop("responses", None)
        if responses is None:
            continue
        result["score_facts"] = score_facts(result["requests"], responses)
        for row in result["requests"]:
            if row["shape"] == "S":
                row["json"] = json_visible(tokenizer, row["token_ids"])
            if row["kind"] == "gen":
                row["token_norms"] = None
    return results


def compare(results):
    by_id = {r["summary"]["arm"]["id"]: r for r in results}
    classes = {}
    for result in results:
        arm = result["summary"]["arm"]
        key = ("legacy" if arm["engine"] == "legacy" else "new") + ("-ON" if arm["diag"] else "-OFF")
        classes.setdefault(key, []).append(result)
    shape_metrics = {}
    for key, arms in classes.items():
        shape_metrics[key] = {}
        for shape in ("P", "S", "L", "L256"):
            rows = []
            for arm in arms:
                rows += [r for r in arm["requests"] if r["shape"] == shape and r["kind"] == "gen"]
            shape_metrics[key][shape] = {
                "requests": len(rows),
                "n": [r["n"] for r in rows],
                "finish": [r["finish"] for r in rows],
                "send_to_done_ms_median": statistics.median([r["send_to_done_ms"] for r in rows]) if rows else None,
                "decode_tps_observed_median": statistics.median([r["decode_tps_observed"] for r in rows
                                                                  if r["decode_tps_observed"] is not None])
                if any(r["decode_tps_observed"] is not None for r in rows) else None,
                "token_hashes": [r["token_ids_sha256"] for r in rows],
            }
    independent = {}
    for result in results:
        facts = result.get("score_facts", {})
        independent[result["summary"]["arm"]["id"]] = {
            name: {"choice": facts[name]["choice"], "label_mass": facts[name]["label_mass"]}
            for name in ("SCORE0", "SCORE1") if name in facts}
    ranks = {}
    for name in ("SCORE0", "SCORE1"):
        choices = {arm: value[name]["choice"] for arm, value in independent.items() if name in value}
        ranks[name] = {"choices": choices, "consistent": len(set(choices.values())) <= 1}
    on = [r for r in results if r["summary"]["arm"]["diag"]]
    off = [r for r in results if r["summary"]["arm"]["engine"] == "new" and not r["summary"]["arm"]["diag"]]
    diag_overhead = [r["summary"]["diag_overhead_after_done_ms"] for r in on]
    batch_walls = {r["summary"]["arm"]["id"]: r["summary"]["batch_wall_including_tailflush_ms"]
                   for r in results}
    off_med = statistics.median([r["summary"]["batch_first_send_to_last_done_ms"] for r in off]) if off else None
    on_med = statistics.median([r["summary"]["batch_first_send_to_last_diag_ms"] for r in on
                                if r["summary"]["batch_first_send_to_last_diag_ms"] is not None]) if on else None
    return {
        "classes": {key: [r["summary"]["arm"]["id"] for r in arms] for key, arms in classes.items()},
        "shape_gen": shape_metrics,
        "independent_score_rank": ranks,
        "on_diag_overhead_after_done_ms": diag_overhead,
        "batch_wall_including_tailflush_ms": batch_walls,
        "new_off_batch_first_send_to_last_done_median_ms": off_med,
        "new_on_batch_first_send_to_last_diag_median_ms": on_med,
        "no_speed_claim": "single host, small n, n=1 per OFF/ON sequence; medians descriptive only",
    }


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = [value for value in values if _finite_number(value)]
    return sum(values) / len(values) if values else None


def _score_fact_rank(fact, name):
    pairs = (fact or {}).get("candidate_logprobs") or []
    if [pair.get("id") for pair in pairs] != [15, 16, 17, 18]:
        raise ValueError("%s candidate order/count mismatch" % name)
    if any(not _finite_number(pair.get("logprob")) for pair in pairs):
        raise ValueError("%s nonfinite candidate logprob" % name)
    mass = fact.get("label_mass")
    if not _finite_number(mass) or not 0 <= mass <= 1.0001:
        raise ValueError("%s invalid candidate probability mass" % name)
    ranking = [pair["id"] for pair in sorted(pairs, key=lambda pair: pair["logprob"],
                                             reverse=True)]
    if fact.get("choice") != ranking[0]:
        raise ValueError("%s candidate choice is not the ranking maximum" % name)
    return {"choice": fact["choice"], "ranking": ranking, "label_mass": mass}


def _cofetch_arm_check(result, require_json=True, profile=None):
    summary = result.get("summary") or {}
    arm = summary.get("arm") or {}
    arm_profile = feature_profile(arm)
    if profile is None:
        profile = arm_profile
    if profile is None:
        raise ValueError("exact gate received an unregistered arm")
    validate_arm_definition(arm)
    expected_env = dict(REGISTERED_BASE_ENV)
    if arm["diag"]:
        expected_env[DIAG_ENV] = "1"
    if profile.get("feature"):
        if arm_profile is None or arm_profile["key"] != profile["key"]:
            raise ValueError("%s arm does not match the registered feature" % arm.get("id"))
        if arm[profile["key"]]:
            expected_env[profile["env"]] = "1"
    elif "threads" not in arm:
        raise ValueError("%s config arm has no registered threads value" % arm.get("id"))
    if summary.get("runtime_env") != expected_env:
        raise ValueError("%s runtime env differs from the exact per-arm gate set" % arm["id"])
    if profile.get("feature"):
        features = summary.get("arm_features") or {}
        if features != {"diag": arm["diag"], profile["key"]: arm[profile["key"]],
                        "phase": arm["phase"]}:
            raise ValueError("%s persisted arm feature identity mismatch" % arm["id"])
        marker = summary.get(profile["marker_count_key"])
        if type(marker) is not int:
            raise ValueError("%s feature marker count is not persisted" % arm["id"])
        if arm[profile["key"]] and marker != 1:
            raise ValueError("%s feature ON marker count %d != 1" % (arm["id"], marker))
        if not arm[profile["key"]] and marker != 0:
            raise ValueError("%s feature OFF emitted %d marker(s)" % (arm["id"], marker))
    else:
        actual_threads = (summary.get("config") or {}).get("threads")
        if type(actual_threads) is not int or actual_threads != arm["threads"]:
            raise ValueError("%s persisted config threads %r != registered %r"
                             % (arm["id"], actual_threads, arm["threads"]))
    rows = result.get("requests") or []
    names = [row.get("name") for row in rows]
    if names != list(COFETCH_REQUIRED_NAMES):
        raise ValueError("%s operation count/order mismatch: %r" % (arm["id"], names))
    by_name = {row["name"]: row for row in rows}
    gen_max = {"P0": 1, "P1": 1, "S0": 16, "S1": 16, "L0": 128, "L1": 128,
               "L2560": 256, "L2561": 256, "gen-held": 128, "eos": 32}
    for name, expected_max in gen_max.items():
        row = by_name[name]
        if row.get("kind") != "gen":
            raise ValueError("%s %s is not a GEN row" % (arm["id"], name))
        if row.get("requested_max_new") != expected_max:
            raise ValueError("%s %s requested max_new mismatch" % (arm["id"], name))
        n = row.get("n")
        tokens = row.get("token_ids") or []
        if type(n) is not int or not 0 <= n <= expected_max or len(tokens) != n:
            raise ValueError("%s %s token count is illegal" % (arm["id"], name))
        finish = row.get("finish")
        if finish not in ("length", "stop"):
            raise ValueError("%s %s finish %r is illegal" % (arm["id"], name, finish))
        if finish == "length" and n != expected_max:
            raise ValueError("%s %s length finish without max_new" % (arm["id"], name))
        if finish == "stop" and (not tokens or tokens[-1] != 248046):
            raise ValueError("%s %s stop finish lacks a real EOS token" % (arm["id"], name))
    if by_name["eos"]["finish"] != "stop":
        raise ValueError("%s eos fixture did not finish at a real EOS" % arm["id"])
    for name in ("resetA", "SCORE0", "resetB", "SCORE1", "score-held",
                 "reset-eos", "score-eos"):
        row = by_name[name]
        if row.get("kind") != "score" or row.get("n") != 0 or row.get("finish") != "score":
            raise ValueError("%s %s is not a completed zero-token SCORE" % (arm["id"], name))
    if by_name["score-held"]["prompt_read"] != by_name["L0"]["prompt_read"]:
        raise ValueError("%s score-held prompt_read differs from the L fixture" % arm["id"])
    if by_name["gen-held"]["prompt_read"] != 0 or \
            by_name["gen-held"]["reused"] != by_name["score-held"]["prompt_read"]:
        raise ValueError("%s gen-held continuation is not prompt_read=0 at the held cursor"
                         % arm["id"])
    if by_name["score-eos"]["prompt_read"] != by_name["P0"]["prompt_read"]:
        raise ValueError("%s score-eos prompt_read differs from the P fixture" % arm["id"])
    if by_name["eos"]["prompt_read"] != 0 or \
            by_name["eos"]["reused"] != by_name["score-eos"]["prompt_read"]:
        raise ValueError("%s eos continuation is not prompt_read=0 at the held cursor"
                         % arm["id"])
    state_held = (by_name["state-held"].get("state") or {}).get("pos")
    if state_held != by_name["score-held"]["prompt_read"] + by_name["gen-held"]["n"]:
        raise ValueError("%s held STATE cursor discontinuity" % arm["id"])
    state_eos = (by_name["state-eos"].get("state") or {}).get("pos")
    if state_eos != by_name["score-eos"]["prompt_read"] + by_name["eos"]["n"]:
        raise ValueError("%s EOS STATE cursor discontinuity" % arm["id"])
    if require_json:
        for name in ("S0", "S1"):
            visible = by_name[name].get("json")
            if not isinstance(visible, dict) or not visible.get("json_complete"):
                raise ValueError("%s %s is not a complete parseable JSON fixture" % (arm["id"], name))
    facts = result.get("score_facts") or {}
    ranks = {}
    for name in ("SCORE0", "SCORE1"):
        if name not in facts:
            raise ValueError("%s missing %s candidate facts" % (arm["id"], name))
        ranks[name] = _score_fact_rank(facts[name], name)
    return {"arm": arm, "rows": len(rows), "score": ranks, "by_name": by_name}


def _cofetch_token_hash_diagnostic(arms, names):
    hashes = {}
    for name in names:
        values = {arm["arm"]["id"]: arm["by_name"][name].get("token_ids_sha256") for arm in arms}
        hashes[name] = {"hashes": values, "equal": len(set(values.values())) == 1}
    return hashes


def validate_cofetch_smoke(results, profile=None):
    profile = FEATURE_PROFILES[COFETCH_KIND] if profile is None else profile
    ids = tuple(profile["smoke_ids"])
    by_id = {result["summary"]["arm"]["id"]: result for result in results}
    if len(results) != len(ids) or set(by_id) != set(ids):
        raise ValueError("smoke gate requires exactly the registered smoke arms")
    checks = {arm_id: _cofetch_arm_check(by_id[arm_id], require_json=True, profile=profile)
              for arm_id in ids}
    for arm_id, check in checks.items():
        if not check["arm"]["diag"]:
            raise ValueError("%s smoke arm must run diagnostic ON" % arm_id)
    rank = {}
    for name in ("SCORE0", "SCORE1"):
        off = checks[ids[0]]["score"][name]
        on = checks[ids[1]]["score"][name]
        if off["ranking"] != on["ranking"]:
            raise ValueError("%s candidate ranking differs between smoke OFF/ON" % name)
        rank[name] = {"off": off, "on": on, "consistent": True}
    return {
        "passed": True,
        "arms": list(ids),
        "score_rank": rank,
        "token_hash_diagnostic": _cofetch_token_hash_diagnostic(
            [checks[arm_id] for arm_id in ids],
            ("L0", "L1", "L2560", "L2561", "gen-held", "eos")),
        "quality_scope": "structural smoke only; not broad quality or production approval",
    }


def validate_cofetch_formal(results, profile=None):
    profile = FEATURE_PROFILES[COFETCH_KIND] if profile is None else profile
    ids = tuple(profile["formal_ids"])
    by_id = {result["summary"]["arm"]["id"]: result for result in results}
    if len(results) != len(ids) or set(by_id) != set(ids):
        raise ValueError("formal gate requires exactly the registered formal arms")
    checks = {arm_id: _cofetch_arm_check(by_id[arm_id], require_json=True, profile=profile)
              for arm_id in ids}
    for arm_id, check in checks.items():
        if check["arm"]["diag"] or check["arm"]["phase"] != "formal":
            raise ValueError("%s formal arm must run diagnostic OFF" % arm_id)
    rank = {}
    for name in ("SCORE0", "SCORE1"):
        values = {arm_id: checks[arm_id]["score"][name]["ranking"]
                  for arm_id in ids}
        if len(set(tuple(value) for value in values.values())) != 1:
            raise ValueError("%s candidate ranking differs between formal arms" % name)
        rank[name] = {"rankings": values, "consistent": True}
    return {
        "passed": True,
        "arms": list(ids),
        "score_rank": rank,
        "token_hash_diagnostic": _cofetch_token_hash_diagnostic(
            [checks[arm_id] for arm_id in ids],
            ("L0", "L1", "L2560", "L2561", "gen-held", "eos")),
        "instrumentation": {"formal_diag": False,
                            "diag_after_done_available": False,
                            "reason": "formal arms use diagnostic OFF"},
        "quality_scope": "structural formal only; no unconditional PASS",
    }


def _resource_facts(summary):
    resources = summary.get("resources") or {}
    samples = resources.get("samples") or []
    commits = [sample.get("commit_available_mib") for sample in samples
               if _finite_number(sample.get("commit_available_mib"))]
    return {
        "gpu_peak_mib": resources.get("gpu_peak_mib"),
        "host_rss_peak_mib": resources.get("host_rss_peak_mib"),
        "available_ram_min_mib": resources.get("available_ram_min_mib"),
        "commit_available_min_mib": min(commits) if commits else None,
    }


def _metric_pair(row):
    if not row:
        return None
    return {
        "kind": row.get("kind"), "n": row.get("n"), "finish": row.get("finish"),
        "e2e_ms": row.get("e2e_observed_ms"), "ttft_ms": row.get("ttft_observed_ms"),
        "prefill_tps": row.get("prefill_tps_from_done"),
        "decode_tps": row.get("decode_tps_observed"),
        "tpot_ms": row.get("tpot_observed_ms"),
        "send_to_done_ms": row.get("send_to_done_ms"),
        "eos": row.get("finish") == "stop" and bool(row.get("token_ids") or [])
        and row["token_ids"][-1] == 248046,
    }


def _gain_fraction(off_values, on_values, higher_is_better=False):
    off_mean, on_mean = _mean(off_values), _mean(on_values)
    if off_mean is None or on_mean is None or off_mean == 0:
        return None
    return (on_mean - off_mean) / off_mean if higher_is_better else (off_mean - on_mean) / off_mean


def compare_cofetch(results, profile=None):
    profile = FEATURE_PROFILES[COFETCH_KIND] if profile is None else profile
    ids = tuple(profile["formal_ids"])
    off_ids, on_ids = (ids[0], ids[3]), (ids[1], ids[2])
    by_id = {result["summary"]["arm"]["id"]: result for result in results}
    formal = [by_id[arm_id] for arm_id in ids if arm_id in by_id]
    if len(formal) != len(ids):
        return {
            "experiment_kind": profile["kind"],
            "formal_complete": False,
            "arms": sorted(by_id),
            "named_only": True,
            "note": "formal four arms not complete; no end-to-end feature comparison",
            "no_unconditional_pass": True,
        }
    by_arm = {}
    for result in formal:
        arm_id = result["summary"]["arm"]["id"]
        by_arm[arm_id] = {row["name"]: row for row in result["requests"]}
    named = {}
    for name in COFETCH_COMPARE_NAMES:
        metrics = {arm_id: _metric_pair(by_arm[arm_id].get(name))
                   for arm_id in ids}
        off_e2e = [metrics[arm_id]["e2e_ms"] for arm_id in off_ids]
        on_e2e = [metrics[arm_id]["e2e_ms"] for arm_id in on_ids]
        off_mean, on_mean = _mean(off_e2e), _mean(on_e2e)
        drift_abs = abs(off_e2e[1] - off_e2e[0]) if all(_finite_number(v) for v in off_e2e) else None
        drift_mean = _mean(off_e2e)
        drift_fraction = drift_abs / drift_mean if drift_abs is not None and drift_mean else None
        gain = _gain_fraction(off_e2e, on_e2e)
        named[name] = {
            "off_A1_metrics": metrics[off_ids[0]],
            "off_A2_metrics": metrics[off_ids[1]],
            "on_B1_metrics": metrics[on_ids[0]],
            "on_B2_metrics": metrics[on_ids[1]],
            "e2e_off_mean_ms": off_mean,
            "e2e_on_mean_ms": on_mean,
            "e2e_gain_fraction": gain,
            "off_endpoint_drift": {"abs_ms": drift_abs, "mean_ms": drift_mean,
                                   "fraction": drift_fraction},
            "gain_ge_5pct": gain is not None and gain >= COFETCH_GAIN_THRESHOLD,
            "beyond_off_drift": gain is not None and drift_fraction is not None
            and gain > drift_fraction,
            "delta": {
                "ttft_gain_fraction": _gain_fraction(
                    [metrics[arm_id]["ttft_ms"] for arm_id in off_ids],
                    [metrics[arm_id]["ttft_ms"] for arm_id in on_ids]),
                "prefill_tps_gain_fraction": _gain_fraction(
                    [metrics[arm_id]["prefill_tps"] for arm_id in off_ids],
                    [metrics[arm_id]["prefill_tps"] for arm_id in on_ids],
                    higher_is_better=True),
                "decode_tps_gain_fraction": _gain_fraction(
                    [metrics[arm_id]["decode_tps"] for arm_id in off_ids],
                    [metrics[arm_id]["decode_tps"] for arm_id in on_ids],
                    higher_is_better=True),
                "tpot_gain_fraction": _gain_fraction(
                    [metrics[arm_id]["tpot_ms"] for arm_id in off_ids],
                    [metrics[arm_id]["tpot_ms"] for arm_id in on_ids]),
                "output_counts": {arm_id: metrics[arm_id]["n"] for arm_id in ids},
                "eos": {arm_id: metrics[arm_id]["eos"] for arm_id in ids},
            },
        }
    workloads = [name for name, value in named.items()
                 if value["gain_ge_5pct"] and value["beyond_off_drift"]]
    resources = {result["summary"]["arm"]["id"]: _resource_facts(result["summary"])
                 for result in formal}

    resource_delta = {}
    for key in ("gpu_peak_mib", "host_rss_peak_mib", "available_ram_min_mib",
                "commit_available_min_mib"):
        off_mean = _mean([resources[arm_id][key] for arm_id in off_ids])
        on_mean = _mean([resources[arm_id][key] for arm_id in on_ids])
        delta = (on_mean - off_mean) / off_mean if off_mean and on_mean is not None else None
        resource_delta[key] = {"off_mean": off_mean, "on_mean": on_mean,
                               "on_vs_off_fraction": delta}
    regressions = []
    for name, value in named.items():
        gain = value["e2e_gain_fraction"]
        if gain is not None and gain < -COFETCH_REGRESSION_LIMIT:
            regressions.append("e2e_regression:" + name)
    for key in ("gpu_peak_mib", "host_rss_peak_mib"):
        delta = resource_delta[key]["on_vs_off_fraction"]
        if delta is not None and delta > COFETCH_REGRESSION_LIMIT:
            regressions.append("resource_regression:" + key)
    for name in ("SCORE0", "SCORE1"):
        rankings = []
        for result in formal:
            fact = (result.get("score_facts") or {}).get(name)
            try:
                rankings.append(tuple(_score_fact_rank(fact, name)["ranking"]))
            except ValueError as exc:
                regressions.append("quality_failure:%s:%s" % (name, exc))
        if rankings and len(set(rankings)) != 1:
            regressions.append("quality_regression:" + name)
    for result in formal:
        for row in result["requests"]:
            if row["name"] in ("S0", "S1") and not (row.get("json") or {}).get("json_complete"):
                regressions.append("json_incomplete:%s:%s"
                                   % (result["summary"]["arm"]["id"], row["name"]))
    if regressions:
        decision = "NO_GO_REGRESSION"
    elif not workloads:
        decision = "NO_GO_NO_GAIN_BEYOND_DRIFT"
    else:
        decision = "REVIEW_ONLY_GAIN_OBSERVED"
    return {
        "experiment_kind": profile["kind"],
        "formal_complete": True,
        "arms": list(ids),
        "named_only": True,
        "named": named,
        "workloads_with_gain_ge_5pct_and_beyond_off_drift": workloads,
        "resources": resources,
        "resource_delta": resource_delta,
        "unacceptable_regressions": regressions,
        "decision": decision,
        "instrumentation": {"formal_diag": False,
                            "diag_after_done_available": False,
                            "reason": "formal arms use diagnostic OFF, so diagnostic-tail "
                                      "overhead cannot support the business timing argument"},
        "held_eos_not_mixed_into_shape_medians": True,
        "sample_scope": "single host, small n, descriptive paired rows; not reliable p95",
        "no_unconditional_pass": True,
    }


def preflight(manifest_path, task_dir, base_spec, fingerprint_fn=None,
              admission_fn=None, monitor_factory=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    now_utc = utc_now()
    started = datetime.fromisoformat(manifest["started_at"]).timestamp()
    remaining = base_spec["budget"]["window_s"] - (time.time() - started)
    ex.validate(base_spec)
    if manifest.get("runtime_env_OFF") is not None and \
            base_spec["runtime"]["env"] != manifest["runtime_env_OFF"]:
        raise ValueError("spec runtime env differs from registered OFF production flags")
    if "STRATA_QWEN35_DECODE_DIAGNOSTICS" in base_spec["runtime"]["env"]:
        raise ValueError("base spec must not arm the diagnostic gate")
    checks = {"utc": now_utc, "started_at": manifest["started_at"],
              "remaining_window_s": remaining, "spec_validated": True}
    engine_sha = sha_file(base_spec["engine"]["path"])
    if engine_sha != manifest["registered_exe_sha256"]:
        raise ValueError("registered new engine SHA mismatch")
    legacy_sha = sha_file(manifest["legacy_exe"])
    if legacy_sha != manifest["legacy_exe_sha256"]:
        raise ValueError("legacy engine SHA mismatch")
    source = {}
    for rel, expected in manifest["source_sha256"].items():
        actual = sha_file(Path(manifest["source_root"]) / rel.replace("/", "\\"))
        if actual != expected:
            raise ValueError("source preimage SHA mismatch: " + rel)
        source[rel] = actual
    fixtures = {}
    for shape in base_spec["shapes"]:
        actual = sha_file(shape["tokens_file"])
        if actual != shape["sha256"]:
            raise ValueError("fixture SHA mismatch " + shape["id"])
        fixtures[shape["id"]] = actual
    model = Path(base_spec["cwd"]) / base_spec["model"]["path"]
    pack = Path(base_spec["cwd"]) / base_spec["pack"]
    cuda = Path(base_spec["runtime"]["cuda_bin"])
    if not model.is_file() or model.stat().st_size <= 0:
        raise ValueError("registered model file missing")
    if manifest.get("model_size") is not None and model.stat().st_size != manifest["model_size"]:
        raise ValueError("registered model size mismatch")
    if not pack.is_dir():
        raise ValueError("pack directory missing")
    if not cuda.is_dir():
        raise ValueError("CUDA DLL directory missing")
    checks.update({"engine_sha256": engine_sha, "legacy_exe_sha256": legacy_sha,
                   "source_sha256": source, "fixture_sha256": fixtures,
                   "model_size": model.stat().st_size,
                   "model_sha_trust": "registered; header only; not rehashed",
                   "pack": str(pack), "cuda_bin": str(cuda),
                   "budgets": manifest["budgets"], "config": base_spec["baseline"]["config"],
                   "arm_order": [a["id"] for a in select_arm_order(manifest)]})
    try:
        guard = preflight_resource_guard(base_spec, manifest, fingerprint_fn,
                                         admission_fn, monitor_factory)
    except BaseException as exc:
        checks["resource_admission"] = {"admitted": False, "error": repr(exc),
                                        "reasons": ["fingerprint/monitor failed before lease"]}
        checks["admitted"] = False
        save_json(Path(task_dir) / "preflight.json", checks)
        raise ResourceRefused("resource preflight failed: " + repr(exc))
    checks["resource_admission"] = guard
    checks["admitted"] = guard["admitted"]
    save_json(Path(task_dir) / "preflight.json", checks)
    require_admitted(guard)
    return manifest, started, checks


def _checkpoint(out_root, ledger, status, error, arms_done, manifest, checks, started_at=None):
    totals = ledger.totals if ledger is not None else {
        "lease_attempts": 0, "binary_launches": 0, "binary_launch_unknown": 0,
        "model_ready": 0, "successful_arms": 0, "protocol_attempts": 0,
        "protocol_completed": 0}
    value = {
        "task": (manifest or {}).get("task_id", "di02-diagnostic-paired"),
        "status": status, "error": error,
        "utc": utc_now(), "started_at": started_at or (manifest or {}).get("started_at"),
        "arms_done": arms_done,
        "native_launches": totals["binary_launches"],
        "native_launch_unknown": totals["binary_launch_unknown"],
        "lease_attempts": totals["lease_attempts"],
        "model_ready": totals["model_ready"],
        "successful_arms": totals["successful_arms"],
        "protocol_requests": totals["protocol_attempts"],
        "protocol_completed": totals["protocol_completed"],
        "budget": (manifest or {}).get("budgets"),
        "preflight_admitted": (checks or {}).get("admitted"),
        "no_retry": True,
        "evidence": str(out_root),
    }
    save_json(Path(out_root) / "checkpoint.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description="DI02 own diagnostic probe")
    parser.add_argument("--receipt-audit", help="read-only raw receipt audit mode (no lease/GPU/model)")
    parser.add_argument("--receipt-audit-out")
    parser.add_argument("--manifest")
    parser.add_argument("--task-dir")
    parser.add_argument("--spec")
    parser.add_argument("--output")
    parser.add_argument("--max-arms", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.receipt_audit:
        audit = reconcile_receipts(args.receipt_audit)
        out = Path(args.receipt_audit_out) if args.receipt_audit_out else \
            Path(args.receipt_audit) / "receipt-audit.json"
        save_json(out, audit)
        print(json.dumps({"status": "AUDITED", "totals": audit["totals"], "out": str(out)}), flush=True)
        return
    missing = [name for name in ("manifest", "task_dir", "spec", "output") if not getattr(args, name)]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    base_spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    started = datetime.fromisoformat(manifest["started_at"]).timestamp()
    ledger = NativeLedger(out_root / "native-ledger.json", manifest["budgets"],
                          manifest["started_at"])
    status, error, checks, results = "FAILED", None, None, []
    kind, smoke_gate, formal_gate = DI02_KIND, None, None
    profile = None
    try:
        kind = experiment_kind(manifest)
        profile = FEATURE_PROFILES.get(kind)
        arm_order, _ = validate_arm_options(manifest, args.max_arms)
        save_json(out_root / "clock-before-preflight.json", {
            "utc": utc_now(), "monotonic": time.monotonic(),
            "remaining_window_s": base_spec["budget"]["window_s"] - (time.time() - started)})
        manifest, started, checks = preflight(args.manifest, args.task_dir, base_spec)
        save_json(out_root / "preflight.json", checks)
        comparison = None
        if args.preflight_only:
            status = "PREFLIGHT_ONLY_ADMITTED"
            error = None
        else:
            prompts = build_prompts(base_spec)
            validate_prompt_plan(prompts)
            model_path = Path(base_spec["cwd"]) / base_spec["model"]["path"]
            status, error = "COMPLETE", None
            for index, arm in enumerate(arm_order, 1):
                clock = {"utc": utc_now(), "monotonic": time.monotonic(),
                         "remaining_window_s": base_spec["budget"]["window_s"] - (time.time() - started)}
                save_json(out_root / ("clock-before-arm-%02d.json" % index), clock)
                if clock["remaining_window_s"] < base_spec["budget"]["native_timeout_s"] + base_spec["budget"]["wait_s"]:
                    status = "WINDOW_EXHAUSTED"
                    error = "remaining %.1fs below bound+wait; no new native" % clock["remaining_window_s"]
                    save_json(out_root / "clock-stop.json", clock)
                    break
                try:
                    results.append(run_arm(out_root, index, arm, base_spec, manifest, started, prompts,
                                           ledger=ledger))
                except BaseException as exc:
                    status = "FAILED"
                    error = repr(exc)
                    save_json(out_root / "failure.json", {"arm": arm["id"], "error": error,
                                                          "utc": utc_now(), "retry": False})
                    break
                save_json(out_root / "progress.json", {"arms_done": [r["summary"]["arm"]["id"] for r in results],
                                                       "utc": utc_now()})
                if profile is not None and arm["id"] == profile["smoke_ids"][-1]:
                    smoke = [item for item in results
                             if item["summary"]["arm"].get("phase") == "smoke"]
                    gate_name = profile["gate_prefix"] + "-smoke-gate.json"
                    try:
                        postprocess(smoke, model_path)
                        smoke_gate = validate_cofetch_smoke(smoke, profile)
                        smoke_gate["utc"] = utc_now()
                        save_json(out_root / gate_name, smoke_gate)
                    except BaseException as exc:
                        status = "SMOKE_REJECTED"
                        error = profile["gate_prefix"] + " smoke gate: " + repr(exc)
                        save_json(out_root / gate_name,
                                  {"passed": False, "error": error, "utc": utc_now(),
                                   "retry": False})
                        break
            try:
                if results:
                    results = postprocess(results, model_path)
            except BaseException as exc:
                status = "FAILED" if status == "COMPLETE" else status
                error = (error + "; " if error else "") + "postprocess: " + repr(exc)
            if profile is not None and status == "COMPLETE":
                formal = [item for item in results
                          if item["summary"]["arm"].get("phase") == "formal"]
                gate_name = profile["gate_prefix"] + "-formal-gate.json"
                if formal:
                    try:
                        formal_gate = validate_cofetch_formal(formal, profile)
                        formal_gate["utc"] = utc_now()
                        save_json(out_root / gate_name, formal_gate)
                    except BaseException as exc:
                        status = "PARTIAL"
                        error = profile["gate_prefix"] + " formal gate: " + repr(exc)
                        save_json(out_root / gate_name,
                                  {"passed": False, "error": error, "utc": utc_now(),
                                   "retry": False})
                elif [item["summary"]["arm"]["id"] for item in results] == list(profile["smoke_ids"]):
                    status = "SMOKE_PASSED"
            if results and profile is not None:
                comparison = compare_cofetch(results, profile)
            elif results:
                comparison = compare(results)
            if comparison and profile is None and \
                    not all(v["consistent"] for v in comparison["independent_score_rank"].values()):
                if status == "COMPLETE":
                    status = "PARTIAL"
                    error = (error + "; " if error else "") + "independent SCORE ranking differs between same-state arms"
        report = {
            "task": manifest.get("task_id", DI02_KIND),
            "experiment_kind": kind,
            "status": status,
            "error": error,
            "started_at": manifest["started_at"],
            "finished_at": utc_now(),
            "arms_done": [r["summary"]["arm"]["id"] for r in results],
            "preflight": checks,
            "comparison": comparison,
            "smoke_gate": smoke_gate,
            "formal_gate": formal_gate,
            "ledger": ledger.totals,
            "results": [{"arm": r["summary"], "requests": r["requests"], "score_facts": r["score_facts"]}
                        for r in results],
            "scope": "diagnostic observation; no engine change/build; no product speed claim",
        }
        save_json(out_root / "result.json", report)
    except ResourceRefused as exc:
        status, error = "RESOURCE_REFUSED", str(exc)
    except BaseException as exc:
        status = "PREFLIGHT_FAILED" if not results else "FAILED"
        error = repr(exc)
    if checks is None:
        saved = Path(args.task_dir) / "preflight.json"
        if saved.is_file():
            try:
                checks = json.loads(saved.read_text(encoding="utf-8"))
                save_json(out_root / "preflight.json", checks)
            except Exception:
                checks = None
    checkpoint = _checkpoint(out_root, ledger, status, error,
                             [r["summary"]["arm"]["id"] for r in results],
                             manifest, checks, manifest.get("started_at"))
    print(json.dumps({"status": status, "error": error, "arms": checkpoint["arms_done"],
                      "ledger": ledger.totals}, ensure_ascii=False), flush=True)
    if status in {"FAILED", "PREFLIGHT_FAILED", "RESOURCE_REFUSED", "SMOKE_REJECTED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
