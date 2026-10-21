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

ARM_ORDER = [
    {"id": "legacy-ea06-OFF", "engine": "legacy", "diag": False},
    {"id": "new-OFF-A1", "engine": "new", "diag": False},
    {"id": "new-ON-B1", "engine": "new", "diag": True},
    {"id": "new-ON-B2", "engine": "new", "diag": True},
    {"id": "new-OFF-A2", "engine": "new", "diag": False},
]
LEDGER_SCHEMA = "strata-di02-native-ledger/1"
REGISTERED_HOST_MIB = 14000
COMMIT_FLOOR_MIB = 1024


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


def validate_arm_options(manifest, max_arms):
    budgets = manifest.get("budgets") or {}
    registered = budgets.get("gpu_native_launches")
    if not isinstance(registered, int) or registered < 1:
        raise ValueError("manifest has no positive registered native budget")
    if not isinstance(max_arms, int) or max_arms < 1:
        raise ValueError("--max-arms must be positive")
    if max_arms > registered:
        raise ValueError("--max-arms %d exceeds registered native budget %d" % (max_arms, registered))
    if max_arms > len(ARM_ORDER):
        raise ValueError("--max-arms %d exceeds implemented arm order" % max_arms)
    order = manifest.get("arm_order")
    expected = [a["id"] for a in ARM_ORDER]
    if order is not None and order != expected:
        raise ValueError("manifest arm_order differs from the implemented registered order")
    for key in ("max_internal_protocol_requests_per_native", "max_internal_protocol_requests_total"):
        if not isinstance(budgets.get(key), int) or budgets[key] < 1:
            raise ValueError("manifest budget %s must be positive" % key)
    return registered, budgets


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
    return prompts


def make_arm_spec(base, arm, manifest):
    spec = copy.deepcopy(base)
    if arm["engine"] == "legacy":
        spec["engine"] = {"path": manifest["legacy_exe"], "sha256": manifest["legacy_exe_sha256"]}
        spec["runtime"]["id"] = "legacy-ea06-compat"
        if sha_file(spec["engine"]["path"]) != spec["engine"]["sha256"]:
            raise ValueError("legacy engine SHA mismatch")
    if arm["diag"]:
        spec["runtime"]["env"] = dict(spec["runtime"]["env"],
                                      STRATA_QWEN35_DECODE_DIAGNOSTICS="1")
    return spec


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
                   "diag_observed_s": None, "state": payload,
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
    raw_text = (native_dir / "raw/child_00.log").read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()
    records, malformed = parse_diag_lines(lines)
    validation = validate_records(records)
    if arm["diag"]:
        if malformed:
            raise ValueError("malformed diagnostic JSON in raw child log")
        if len(records) != len(diag_rows):
            raise ValueError("raw diagnostic row count %d != live %d" % (len(records), len(diag_rows)))
        for live, final in zip(diag_rows, [item["record"] for item in records]):
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
    telemetry = json.loads((out / "telemetry.json").read_text(encoding="utf-8"))
    resources = json.loads((out / "resources.json").read_text(encoding="utf-8"))
    if resources.get("violation"):
        raise ValueError(resources["violation"])
    sends = [r["started_s"] for r in requests]
    dones = [r["finished_s"] for r in requests]
    diag_times = [r["diag_observed_s"] for r in requests if r["diag_observed_s"] is not None]
    summary = {
        "arm": arm, "engine_sha256": spec["engine"]["sha256"],
        "runtime_env": spec["runtime"]["env"], "config": spec["baseline"]["config"],
        "load_ready_s": native.load_ready_s,
        "load_to_first_done_s": dones[0] - native.started,
        "batch_first_send_to_last_done_ms": (dones[-1] - sends[0]) * 1000.0,
        "batch_first_send_to_last_diag_ms": (diag_times[-1] - sends[0]) * 1000.0 if diag_times else None,
        "batch_wall_including_tailflush_ms": ((max(dones[-1], diag_times[-1] if diag_times else dones[-1]))
                                              - sends[0]) * 1000.0,
        "request_count": len(requests), "diag_count": len(diag_rows),
        "diag_overhead_after_done_ms": _summary([r["diag_after_done_ms"] for r in requests
                                                 if r["diag_after_done_ms"] is not None]),
        "resources": resources, "telemetry": telemetry_summary(telemetry),
        "graph": graph_evidence(raw_text),
        "native_exit": receipt["commands"][0]["exit_code"],
        "raw_childlog_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    save_json(out / "diag-records.json", {"records": [item["record"] for item in records],
                                          "malformed": malformed, "validation_errors": validation})
    save_json(out / "measurements.json", {"summary": summary, "requests": requests})
    save_json(out / "arm-summary.json", summary)
    if ledger is not None:
        ledger.arm_complete(record)
    return {"summary": summary, "requests": requests, "out": str(out), "responses": native.responses}


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
        responses = result.pop("responses")
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
                   "arm_order": [a["id"] for a in ARM_ORDER]})
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
        "task": "di02-diagnostic-paired", "status": status, "error": error,
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
    try:
        validate_arm_options(manifest, args.max_arms)
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
            status, error = "COMPLETE", None
            for index, arm in enumerate(ARM_ORDER[:args.max_arms], 1):
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
            try:
                if results:
                    results = postprocess(results, Path(base_spec["cwd"]) / base_spec["model"]["path"])
            except BaseException as exc:
                status = "FAILED" if status == "COMPLETE" else status
                error = (error + "; " if error else "") + "postprocess: " + repr(exc)
            comparison = compare(results) if results else None
            if comparison and not all(v["consistent"] for v in comparison["independent_score_rank"].values()):
                if status == "COMPLETE":
                    status = "PARTIAL"
                    error = (error + "; " if error else "") + "independent SCORE ranking differs between same-state arms"
        report = {
            "task": "di02-diagnostic-paired",
            "status": status,
            "error": error,
            "started_at": manifest["started_at"],
            "finished_at": utc_now(),
            "arms_done": [r["summary"]["arm"]["id"] for r in results],
            "preflight": checks,
            "comparison": comparison,
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
    if status in {"FAILED", "PREFLIGHT_FAILED", "RESOURCE_REFUSED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
