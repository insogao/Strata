#!/usr/bin/env python3
"""Recompute a pre-registered session regression from gpu-lock's raw child log.

Does not launch an engine. `--plan` contains ordered {name, command} entries (the
exact stdin manifest), model/input/build identity and `arm`=old|fixed. Native raw
and the plan are kept separately; comparisons below use token IDs, not summaries.
"""
import argparse
import hashlib
import json
from pathlib import Path


def parse(plan, text):
    responses = {}
    commands = iter(plan["commands"])
    current = None
    for line in text.splitlines():
        if not line.startswith(("CAPS ", "STATE ", "SAVED ", "LOADED ", "PP ", "T ",
                                "LOGPROBS ", "CANDLOGPROBS ", "DONE ", "ERR ")):
            continue
        if current is None:
            current = next(commands, None)
            if current is None or current["command"] == "QUIT":
                raise ValueError("unmatched protocol response: " + line[:80])
            response = {"tokens": [], "lines": []}
        response["lines"].append(line)
        if line.startswith("T "):
            response["tokens"].append(int(line.split()[1]))
        if line.startswith("DONE "):
            fields = line.split()
            response["done"] = {"produced": int(fields[1]), "finish": fields[5], "reused": int(fields[8])}
            if response["done"]["produced"] != len(response["tokens"]):
                raise ValueError("DONE token count mismatch")
        if line.startswith("STATE "):
            response["state"] = json.loads(line[6:])
        if line.startswith(("CAPS ", "STATE ", "SAVED ", "LOADED ", "DONE ", "ERR ")):
            responses[current["name"]] = response
            current = None
    if current is not None or [c for c in commands if c["command"] != "QUIT"]:
        raise ValueError("missing protocol response(s)")
    return responses


def verify(plan, raw):
    r = parse(plan, raw)
    n = plan["prompt_tokens"]
    if plan["arm"] == "state":
        gates = state_gates(plan, r)
        return {"arm": "state", "gates": gates, "pass": all(gates.values()), "responses": r}
    full = r["full"]["tokens"]
    split = r["split1"]["tokens"] + r["split2"]["tokens"]
    gates = {"split_equals_single": split == full,
             "full_committed_cursor": r["full_state"]["state"]["pos"] == n + len(full),
             "split_committed_cursor": r["split_state"]["state"]["pos"] == n + len(split)}
    if plan["arm"] == "fixed":
        singles = sum((r[f"one{i}"]["tokens"] for i in range(16)), []) + r["one_tail"]["tokens"]
        gates["single_token_segments_equal_single"] = singles == full
        for i in range(16):
            gates[f"single_token_cursor_{i}"] = r[f"one_state{i}"]["state"]["pos"] == n + i + 1
        gates["single_token_final_cursor"] = r["one_final_state"]["state"]["pos"] == n + len(singles)
        for label in ("llmax", "overflow", "decimal", "suffix"):
            gates[f"{label}_rejected"] = any(s.startswith("ERR ") for s in r[label]["lines"])
            gates[f"{label}_state_unchanged"] = r[label + "_before"]["state"] == r[label + "_after"]["state"]
        gates.update(state_gates(plan, r))
        gates["reaches_real_eos"] = r["full"]["done"]["finish"] == "stop"
        expected_errors = {"llmax", "overflow", "decimal", "suffix", "load_pure"}
        gates["no_unexpected_errors"] = all(not any(s.startswith("ERR ") for s in answer["lines"])
                                           for name, answer in r.items() if name not in expected_errors)
    return {"arm": plan["arm"], "gates": gates,
            "pass": all(gates.values()) if plan["arm"] == "fixed" else not gates["split_equals_single"],
            "notice": "correctness regression; no throughput claim; LOAD pure continuation remains unsupported",
            "full_tokens": full, "split_tokens": split, "responses": r}


def state_gates(plan, r):
    files_exist = all(Path(p).is_file() for p in plan["state_files"])
    exports_succeeded = (any(s.startswith("SAVED ") for s in r["save16"]["lines"])
                         and any(s.startswith("LOADED ") for s in r["load16"]["lines"])
                         and any(s.startswith("SAVED ") for s in r["resave16"]["lines"]))
    return {
        "state_exports_succeeded": exports_succeeded,
        "state_files_exist": exports_succeeded and files_exist,
        "save_load_save_bytes_equal": exports_succeeded and files_exist and (Path(plan["state_files"][0]).read_bytes() ==
                                                         Path(plan["state_files"][1]).read_bytes()),
        "load_has_no_logits_refused": any("no logits" in s for s in r["load_pure"]["lines"]),
        "load_failed_pure_state_unchanged": r["load_before"]["state"] == r["load_after"]["state"],
        "load_append_token_parity": r["live_append"]["tokens"] == r["load_append"]["tokens"],
        "load_append_cursor_parity": (r["live_append_state"]["state"]["pos"] ==
                                      r["load_append_state"]["state"]["pos"]),
        "state_no_unexpected_errors": all(not any(s.startswith("ERR ") for s in answer["lines"])
                                         for name, answer in r.items()
                                         if name not in {"load_pure", "llmax", "overflow", "decimal", "suffix"}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    raw = Path(a.raw).read_bytes()
    result = verify(plan, raw.decode("utf-8", "replace"))
    result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    result["plan_sha256"] = hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
    result["state_sha256"] = {s: hashlib.sha256(Path(s).read_bytes()).hexdigest()
                              if result["gates"].get("state_exports_succeeded") and Path(s).is_file() else None
                              for s in plan.get("state_files", [])}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm": result["arm"], "pass": result["pass"], "gates": result["gates"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
