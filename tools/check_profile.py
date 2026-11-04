#!/usr/bin/env python3
"""Check one strata profile against a caller-supplied fingerprint.

Standard library only, CPU only, zero side effects: this tool never touches the
GPU, never starts the engine, never hashes a model file and never writes
anything. The fingerprint (device, model SHA, runtime argv, measured headroom)
is supplied by the caller; the tool only compares it against the profile's own
``invalidators`` list and prints ``OK`` (exit 0) or ``FAIL: <reason>`` (exit 1).

Rule dispatch is driven by the profile: every ``invalidators[].code`` in the
file must be implemented here, otherwise the check fails closed (a profile is
never trusted because the checker did not understand one of its rules).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

SCHEMA = "strata-profile/1"
TUNE_SCHEMA = "strata-tune-profile/1"
PROFILE_NAME = "long-output-4060-4096.json"

EXIT_OK = 0
EXIT_FAIL = 1

# Engine flags whose value was part of the calibration. Other argv items
# (--qwen35-model, --qwen35-pack, --serve) are covered by MODEL_SHA_MISMATCH and
# by the deployment itself, so they are not compared here.
CALIBRATED_FLAGS = (
    ("--qwen35-cache-slots", "slots"),
    ("--qwen35-threads", "threads"),
    ("--qwen35-ubatch", "ubatch"),
    ("--max-context", "ctx"),
)

_SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")

DEFAULT_MIN_HEADROOM_MIB = 300.0


class Usage(Exception):
    """Bad input; printed as FAIL with exit code 1 (never a silent pass)."""


def _leaf(node):
    """Unwrap a P3-1 fingerprint leaf ({"value": .., "source": ..}) to its value."""
    seen = 0
    while isinstance(node, dict) and "value" in node and seen < 8:
        node = node["value"]
        seen += 1
    return node


def _first(*candidates):
    for cand in candidates:
        if cand is not None:
            return cand
    return None


def _as_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def _as_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        return float(value)
    return None


def _as_text(value):
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value).strip()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "on", "true", "yes", "enabled"):
            return True
        if low in ("0", "off", "false", "no", "disabled", ""):
            return False
    return None


# --------------------------------------------------------------------------- profile

def load_profile(path):
    if not os.path.isfile(path):
        raise Usage(f"no such profile file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Usage(f"profile {path} is not readable JSON ({exc})")
    if not isinstance(doc, dict):
        raise Usage(f"profile {path} is not a JSON object")
    schema = doc.get("schema_version")
    if schema == TUNE_SCHEMA:
        raise Usage(f"{path} is a {TUNE_SCHEMA} file; check it with "
                     f"backends/qwen35_llama/tools/strata_tune.py check (this tool checks "
                     f"{SCHEMA})")
    if schema != SCHEMA:
        raise Usage(f"{path} is not a {SCHEMA} profile (schema_version={schema!r})")
    missing = [key for key in ("model", "engine", "device", "argv", "evidence",
                               "confidence", "invalidators", "generated_at")
               if key not in doc]
    if missing:
        raise Usage(f"profile {path} misses required field(s): {', '.join(missing)}")
    rules = doc["invalidators"]
    if not isinstance(rules, list) or not rules:
        raise Usage(f"profile {path}: invalidators must be a non-empty list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not rule.get("code"):
            raise Usage(f"profile {path}: invalidators[{index}] has no code")
    stored_sha = _as_text(doc["model"].get("sha256_prefix")) or ""
    if not _SHA_RE.fullmatch(stored_sha.lower()):
        raise Usage(f"profile {path}: model.sha256_prefix {stored_sha!r} is not a "
                     f"lowercase hex prefix (8..64 hex chars)")
    return doc


def _profile_argv(profile):
    """Profile argv (engine flags) -> {long flag: value string}."""
    argv = profile.get("argv")
    values = {}
    if isinstance(argv, dict):
        for flag, alias in CALIBRATED_FLAGS:
            got = _first(argv.get(flag), argv.get(alias))
            if got is not None:
                values[flag] = str(_as_int(got) if _as_int(got) is not None else _as_text(got))
        return values
    if isinstance(argv, list):
        items = [str(item) for item in argv]
        index = 0
        while index < len(items):
            item = items[index]
            if item.startswith("--") and "=" in item:
                flag, value = item.split("=", 1)
                values[flag] = value
            elif item in ("--qwen35-cache-slots", "--qwen35-threads", "--qwen35-ubatch",
                          "--max-context") and index + 1 < len(items):
                values[item] = items[index + 1]
                index += 1
            index += 1
        return values
    raise Usage("profile argv must be a list of engine flags or an object of values")


# ----------------------------------------------------------------------- fingerprint

def load_fingerprint(path):
    if not os.path.isfile(path):
        raise Usage(f"no such fingerprint file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Usage(f"fingerprint {path} is not readable JSON ({exc})")
    if not isinstance(raw, dict):
        raise Usage(f"fingerprint {path} is not a JSON object")
    return normalize_fingerprint(raw)


def normalize_fingerprint(raw):
    """Accept the documented minimal fingerprint and the P3-1 strata_fingerprint.py shape."""
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    device = raw.get("device") if isinstance(raw.get("device"), dict) else {}
    gpu = raw.get("gpu") if isinstance(raw.get("gpu"), dict) else {}

    sha = _first(_as_text(_leaf(model.get("sha256"))),
                 _as_text(_leaf(model.get("sha256_prefix"))),
                 _as_text(_leaf(raw.get("model_sha256"))),
                 _as_text(_leaf(raw.get("model_sha256_prefix"))))
    if sha:
        sha = sha.lower()
        if sha.startswith("sha256:"):
            sha = sha[len("sha256:"):]

    quant = _first(_as_text(_leaf(raw.get("quant"))),
                   _as_text(_leaf(model.get("quant"))),
                   _as_text(_leaf(model.get("quantization"))),
                   _as_text(_leaf(device.get("quant"))))

    gpu_name = _first(_as_text(_leaf(device.get("gpu"))),
                      _as_text(_leaf(device.get("name"))),
                      _as_text(_leaf(gpu.get("name"))),
                      _as_text(_leaf(raw.get("gpu_name"))))
    vram = _first(_as_int(_leaf(device.get("vram_total_mib"))),
                  _as_int(_leaf(gpu.get("vram_total_mib"))),
                  _as_int(_leaf(raw.get("vram_total_mib"))))
    driver = _first(_as_text(_leaf(device.get("driver"))),
                    _as_text(_leaf(device.get("driver_version"))),
                    _as_text(_leaf(gpu.get("driver_version"))),
                    _as_text(_leaf(raw.get("driver"))))
    cap = _first(_as_text(_leaf(device.get("compute_capability"))),
                 _as_text(_leaf(gpu.get("compute_capability"))),
                 _as_text(_leaf(raw.get("compute_capability"))))

    headroom = _first(_as_float(_leaf(raw.get("headroom_mib"))),
                      _as_float(_leaf(device.get("headroom_mib"))))
    if headroom is None:
        peak = _first(_as_float(_leaf(raw.get("gpu_peak_mib"))),
                      _as_float(_leaf(device.get("gpu_peak_mib"))))
        if peak is not None and vram is not None:
            headroom = float(vram) - peak

    output_tokens = _first(_as_int(_leaf(raw.get("output_tokens"))),
                           _as_int(_leaf(raw.get("max_new"))))

    gates_raw = raw.get("gates") if isinstance(raw.get("gates"), dict) else {}
    env = raw.get("env") if isinstance(raw.get("env"), dict) else {}
    gates = {key: _as_bool(_leaf(gates_raw.get(key))) for key in
             ("compact_slots", "compact", "mtp", "mtp_txn")}
    env_names = {
        "compact_slots": ("STRATA_QWEN35_COMPACT_SLOTS", "COMPACT_SLOTS"),
        "mtp": ("MTP",),
        "mtp_txn": ("MTP_TXN",),
    }
    for key, names in env_names.items():
        if gates.get(key) is None:
            for name in names:
                if name in env:
                    gates[key] = _as_bool(_leaf(env.get(name)))
                    break

    return {
        "model_sha256": sha,
        "quant": quant,
        "gpu": gpu_name,
        "vram_total_mib": vram,
        "driver": driver,
        "compute_capability": cap,
        "headroom_mib": headroom,
        "output_tokens": output_tokens,
        "argv": _fingerprint_argv(raw),
        "gates": gates,
    }


def _fingerprint_argv(raw):
    argv = raw.get("argv")
    values = {}
    if isinstance(argv, dict):
        for flag, alias in CALIBRATED_FLAGS:
            got = _first(argv.get(flag), argv.get(alias))
            if got is not None:
                number = _as_int(got)
                values[flag] = str(number) if number is not None else _as_text(got)
        return values
    if isinstance(argv, str):
        argv = argv.split()
    if isinstance(argv, list):
        items = [str(item) for item in argv]
        index = 0
        while index < len(items):
            item = items[index]
            if "=" in item and item.startswith("--"):
                flag, value = item.split("=", 1)
                values[flag] = value
            elif item in [flag for flag, _ in CALIBRATED_FLAGS] and index + 1 < len(items):
                values[item] = items[index + 1]
                index += 1
            index += 1
        return values
    return values


# ---------------------------------------------------------------------------- rules

def _rule_model_sha(profile, fingerprint, _options):
    stored = str(profile["model"]["sha256_prefix"]).lower()
    current = fingerprint["model_sha256"]
    if not current:
        return [f"fingerprint carries no model SHA -> fail closed; cannot confirm "
                f"model.sha256_prefix {stored}"]
    if not current.startswith(stored):
        return [f"model SHA prefix differs: profile {stored}, current {current[:16]}"]
    return []


def _rule_quant(profile, fingerprint, _options):
    stored = _as_text(profile["model"].get("quant"))
    current = fingerprint["quant"]
    if stored is None:
        return ["profile carries no model.quant; cannot confirm the quantization"]
    if not current:
        return [f"fingerprint carries no quant -> fail closed; profile quant {stored}"]
    if current.strip().lower() != stored.strip().lower():
        return [f"quant differs: profile {stored}, current {current} (no cross-quantization conversion)"]
    return []


def _rule_device(profile, fingerprint, _options):
    stored = profile["device"]
    problems = []
    pairs = (("gpu", "GPU name"), ("driver", "driver version"),
             ("compute_capability", "compute capability"))
    for key, label in pairs:
        want = _as_text(stored.get(key))
        got = fingerprint[key]
        if want is None:
            if got is not None:
                problems.append(f"profile has no device.{key} but the host reports {got}")
            continue
        if not got:
            problems.append(f"fingerprint carries no {label} -> fail closed; profile {want}")
            continue
        if got != want:
            problems.append(f"{label} differs: profile {want}, current {got}")
    want_vram = _as_int(stored.get("vram_total_mib"))
    got_vram = fingerprint["vram_total_mib"]
    if want_vram is None:
        problems.append("profile carries no device.vram_total_mib")
    elif got_vram is None:
        problems.append(f"fingerprint carries no VRAM total -> fail closed; profile {want_vram} MiB")
    elif got_vram != want_vram:
        problems.append(f"VRAM total differs: profile {want_vram} MiB, current {got_vram} MiB")
    if problems:
        problems.append("cross-card reuse refused: no scaling or conversion is performed")
    return problems


def _rule_argv(profile, fingerprint, _options):
    stored = _profile_argv(profile)
    if not stored:
        return ["profile argv carries none of the calibrated flags "
                "(--qwen35-cache-slots/--qwen35-threads/--qwen35-ubatch/--max-context)"]
    current = fingerprint["argv"]
    problems = []
    for flag, _alias in CALIBRATED_FLAGS:
        if flag not in stored:
            continue
        got = current.get(flag)
        if got is None:
            problems.append(f"{flag}: runtime value missing from the fingerprint "
                            f"(profile {stored[flag]})")
        elif str(_as_int(got) if _as_int(got) is not None else got) != str(
                _as_int(stored[flag]) if _as_int(stored[flag]) is not None else stored[flag]):
            problems.append(f"{flag}: profile {stored[flag]}, runtime {got}")
    return problems


def _rule_gates(profile, fingerprint, _options):
    stored = profile.get("gates") if isinstance(profile.get("gates"), dict) else {}
    expected_off = [name for name in ("compact_slots", "mtp", "mtp_txn")
                    if _as_bool(stored.get(name)) is False]
    if not expected_off:
        return ["profile declares no gate to keep off (gates.compact_slots/mtp/mtp_txn)"]
    problems = []
    for name in expected_off:
        state = fingerprint["gates"].get(name)
        if state is None:
            problems.append(f"gate {name}: fingerprint does not report it -> fail closed "
                            f"(profile requires it OFF)")
        elif state:
            problems.append(f"gate {name} is ON at runtime; the profile was measured with it OFF")
    return problems


def _rule_output_tokens(profile, fingerprint, options):
    minimum = _as_int(_leaf((profile.get("applies_to") or {}).get("output_tokens_min")))
    if minimum is None:
        return ["profile declares no applies_to.output_tokens_min"]
    want = options.get("output_tokens")
    have = _first(want, fingerprint["output_tokens"])
    if have is None:
        return [f"requested output length unknown -> fail closed; the profile only applies "
                f"at >= {minimum} output tokens"]
    if have < minimum:
        return [f"requested output {have} token(s) < {minimum}: the profile does not apply "
                f"(short JSON runs slower with it); keep the 3200-slot default"]
    return []


def _rule_headroom(profile, fingerprint, options):
    rule = next((item for item in profile["invalidators"]
                 if item.get("code") == "HEADROOM_BELOW_MIN"), {})
    minimum = _as_float(_first(_leaf(options.get("min_headroom_mib")),
                               _leaf(rule.get("min_headroom_mib")),
                               DEFAULT_MIN_HEADROOM_MIB))
    have = fingerprint["headroom_mib"]
    if have is None:
        return [f"fingerprint carries no measured headroom (headroom_mib, or "
                f"vram_total_mib - gpu_peak_mib) -> fail closed; minimum {minimum:.0f} MiB"]
    if have < minimum:
        return [f"measured headroom {have:.1f} MiB < {minimum:.0f} MiB minimum"]
    return []


RULES = {
    "MODEL_SHA_MISMATCH": _rule_model_sha,
    "QUANT_MISMATCH": _rule_quant,
    "DEVICE_MISMATCH": _rule_device,
    "ARGV_MISMATCH": _rule_argv,
    "GATE_TOGGLED": _rule_gates,
    "OUTPUT_TOKENS_BELOW_MIN": _rule_output_tokens,
    "HEADROOM_BELOW_MIN": _rule_headroom,
}


def check(profile, fingerprint, options):
    """Run every invalidator the profile declares; return [(code, [messages])]."""
    results = []
    for rule in profile["invalidators"]:
        code = str(rule["code"]).strip()
        handler = RULES.get(code)
        if handler is None:
            results.append((code, [f"unsupported invalidator code {code!r} in the profile "
                                   f"-> fail closed (this checker implements: "
                                   f"{', '.join(sorted(RULES))})"]))
            continue
        results.append((code, handler(profile, fingerprint, options)))
    return results


# ------------------------------------------------------------------------------ cli

def _default_profile():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "profiles", PROFILE_NAME)


def _configure_stdio():
    """UTF-8 stdout/stderr: profile and fingerprint paths may be non-ASCII (e.g. a
    Chinese work-tree root), and the default Windows console code page cannot print them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def build_parser():
    parser = argparse.ArgumentParser(
        prog="check_profile.py",
        description=("Check one strata profile against a caller-supplied fingerprint "
                     "(CPU only, stdlib only, no GPU access)."),
        epilog=(
            "fingerprint JSON, minimal form:\n"
            '  {"model_sha256": "<hex>", "quant": "UD-IQ3_S",\n'
            '   "device": {"gpu": "...", "vram_total_mib": 8188, "driver": "591.74",\n'
            '              "compute_capability": "8.9"},\n'
            '   "argv": ["--qwen35-cache-slots", "4096", "--qwen35-threads", "8",\n'
            '            "--qwen35-ubatch", "4096", "--max-context", "4096"],\n'
            '   "headroom_mib": 467.4, "output_tokens": 128,\n'
            '   "gates": {"compact_slots": false, "mtp": false, "mtp_txn": false}}\n\n'
            "A tools/strata_fingerprint.py document ({\"value\",\"source\"} leaves, "
            "\"gpu\" instead of \"device\") is accepted as well; headroom_mib is then "
            "computed from vram_total_mib - gpu_peak_mib when both are present.\n\n"
            "exit codes: 0 = OK, 1 = FAIL (any triggered rule, unreadable input, or a "
            "fingerprint field too sparse to decide -> fail closed)."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", default=_default_profile(),
                        help=f"profile to check (default: {PROFILE_NAME} next to this tool's tree)")
    parser.add_argument("--fingerprint-json", required=True, metavar="FILE",
                        help="fingerprint of the target host/runtime (never probed by this tool)")
    parser.add_argument("--output-tokens", type=int, default=None, metavar="N",
                        help="requested output length of the run to check "
                             "(overrides the fingerprint's output_tokens)")
    parser.add_argument("--min-headroom-mib", type=float, default=None, metavar="MIB",
                        help=f"override the headroom floor (profile default: "
                             f"{DEFAULT_MIN_HEADROOM_MIB:.0f} MiB)")
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable report instead of OK/FAIL lines")
    return parser


def main(argv=None):
    _configure_stdio()
    args = build_parser().parse_args(argv)
    options = {"output_tokens": args.output_tokens,
               "min_headroom_mib": args.min_headroom_mib}
    try:
        profile = load_profile(os.path.abspath(args.profile))
        fingerprint = load_fingerprint(os.path.abspath(args.fingerprint_json))
    except Usage as exc:
        print(f"FAIL: {exc}")
        return EXIT_FAIL

    results = check(profile, fingerprint, options)
    failed = [(code, messages) for code, messages in results if messages]
    status = "ok" if not failed else "invalidated"

    if args.json:
        payload = {
            "schema": "strata-profile-check/1",
            "profile": os.path.abspath(args.profile),
            "fingerprint": os.path.abspath(args.fingerprint_json),
            "status": status,
            "checks": [{"code": code, "ok": not messages, "messages": messages}
                       for code, messages in results],
            "stored": {"model_sha256_prefix": profile["model"]["sha256_prefix"],
                       "quant": profile["model"].get("quant"),
                       "device": profile.get("device"),
                       "argv": _profile_argv(profile)},
            "current": fingerprint,
        }
        print(json.dumps(payload, indent=1, ensure_ascii=False))
        return EXIT_OK if not failed else EXIT_FAIL

    if not failed:
        print(f"OK: {args.profile} applies to {args.fingerprint_json} "
              f"({len(results)} invalidation rule(s) checked, 0 triggered; "
              f"confidence: {profile.get('confidence')})")
        return EXIT_OK

    for code, messages in failed:
        for message in messages:
            print(f"FAIL: {code}: {message}")
    print(f"FAIL: {len(failed)} of {len(results)} invalidation rule(s) triggered; "
          f"do not apply {os.path.basename(args.profile)} to this runtime")
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
