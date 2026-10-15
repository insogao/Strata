#!/usr/bin/env python3
"""stub_engine.py - TEST-ONLY stub of the qwen35_llama serve line protocol.

This is not the engine and is never used by score_api.py in a real run: it is a small
stand-in that answers `CAPS` / `SCORE` in the exact line grammar of
docs/backends/qwen35_llama_serve_protocol.md, so the service's 0-GPU unit tests can drive a
real child process over real pipes. Scores are deterministic: logit(id) = -0.5 * id plus a
prompt-dependent offset, over a configurable vocabulary size.

    python stub_engine.py --vocab 64        # speaks READY on stdout, answers on stdin

P2R-3c additions (test-only, same grammar as the engine's P2R-3a contract): `SCORE ... keep=1`
keeps/arms the session, `GEN <max_new> cont=1 [<ids>]` continues an armed session and answers
deterministic `T` lines, `STATE` answers one JSON line of session bookkeeping. No tokenizer,
no timings, no real model.
"""
import argparse
import json
import math
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--drop-scores", type=int, default=0,
                        help="silently drop the first N SCORE commands (timeout tests)")
    parser.add_argument("--score-delay-ms", type=float, default=0.0,
                        help="sleep this long before answering a SCORE (busy tests)")
    parser.add_argument("--cand-mode", choices=("normal", "count-bump", "unsolicited", "twice", "swap"),
                        default="normal",
                        help="test-only CANDLOGPROBS distortion for protocol-violation tests")
    parser.add_argument("--serve", action="store_true", help="accepted for CLI parity, ignored")
    args = parser.parse_args()

    out = sys.stdout.buffer

    def emit(line: str) -> None:
        out.write(line.encode("utf-8") + b"\n")
        out.flush()

    caps = {
        "protocol": "strata-serve-v1",
        "continuous_requests": True,
        "cancel": True,
        "state_isolation": True,
        "prefix_cache": False,
        "batching": False,
        "mtp": {"available": False, "enabled": False, "default": False, "k": None},
        "expert_formats": ["Q4_0", "Q5_0", "IQ3_S"],
        "context_max": args.context,
        "logits_read": True,
        "score_explicit_candidate_ids": True,
        "score_max_candidate_ids": 2048,
        "state_keep_score": True,
        "session_continue": True,
        "state_export": False,
        "disk_kv": False,
        "vision": False,
    }
    emit("READY %d stop" % args.context)
    dropped = 0
    session = {"armed": False, "origin": None, "pos": 0}

    def state_json():
        # Engine-shaped STATE subset (docs/backends/qwen35_llama_session_state.md): `session` is
        # the arming fact; `armed` is carried as the same-value alias mentioned by the task spec.
        return {"loaded": False, "pos": session["pos"], "bytes": 2216, "path": None,
                "covers": ["attn_kv", "gdn_state", "conv", "pos"],
                "session": session["armed"], "origin": session["origin"], "file_bytes": None,
                "max_context": args.context,
                "not_covered": ["logits", "sampler_rng_state", "expert_cache", "bridge_counters",
                                "rollback_history"],
                "armed": session["armed"]}

    def log_softmax(values):
        top = max(values)
        total = top + math.log(sum(math.exp(value - top) for value in values))
        return [value - total for value in values]

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.strip():
            continue
        head, _, rest = line.partition(" ")
        if head == "CAPS" and not rest.strip():
            emit("CAPS " + json.dumps(caps, separators=(",", ":")))
            continue
        if head == "QUIT":
            return 0
        if head == "STOP":
            continue
        if head == "SCORE":
            if dropped < args.drop_scores:
                dropped += 1
                continue
            ids = None
            topk = 5
            cand = None
            cand_seen = False
            keep = False
            keep_seen = False
            problem = None
            for token in rest.split():
                if token.startswith("topk="):
                    if not token[5:].isdigit():
                        problem = "topk must be a non-negative decimal integer"
                        break
                    topk = int(token[5:])
                elif token.startswith("keep="):
                    if keep_seen:
                        problem = "duplicate keep key"
                        break
                    keep_seen = True
                    if token != "keep=1":
                        problem = "keep must be the literal 1"
                        break
                    keep = True
                elif token.startswith("cand="):
                    if cand_seen:
                        problem = "duplicate cand key"
                        break
                    cand_seen = True
                    value = token[5:]
                    if not value or not all(part.isdigit() for part in value.split(",")):
                        problem = "cand must be decimal ids"
                        break
                    cand = [int(part) for part in value.split(",")]
                    if len(cand) > 2048:
                        problem = "cand has more than 2048 ids"
                        break
                    if len(set(cand)) != len(cand):
                        problem = "duplicate cand id"
                        break
                elif token and token.replace(",", "").isdigit():
                    if ids is not None:
                        problem = "two id lists"
                        break
                    ids = [int(part) for part in token.split(",")]
                else:
                    problem = "unknown key %r" % token
                    break
            if problem is None and ids is None:
                problem = "missing id list"
            if problem is None and topk == 0 and cand is None:
                problem = "topk must be positive without cand"
            if problem is not None:
                emit("ERR bad request: %s" % problem)
                continue
            if any(token_id < 0 or token_id >= args.vocab for token_id in ids + (cand or [])):
                emit("ERR a token id is outside the vocabulary")
                continue
            if args.score_delay_ms:
                time.sleep(args.score_delay_ms / 1000.0)
            # P2R-3a/P2R-3c session semantics: keep=1 appends at an armed cursor and stays
            # armed (origin "score-keep" when this request armed it); without keep=1 the (b)
            # reset clears the session before and after the answer.
            if keep:
                reused = session["pos"] if session["armed"] else 0
                if not session["armed"]:
                    session["armed"] = True
                    session["origin"] = "score-keep"
                session["pos"] = reused + len(ids)
            else:
                reused = 0
                session["armed"] = False
                session["origin"] = None
                session["pos"] = 0
            salt = sum(ids) + len(ids)
            logprobs = log_softmax([-(0.5 * i) - 0.125 * (salt % 7) for i in range(args.vocab)])
            pos = len(ids) - 1
            if topk > 0:
                count = min(topk, args.vocab)
                order = sorted(range(args.vocab), key=lambda i: (-logprobs[i], i))[:count]
                emit("LOGPROBS %d %d %s" % (pos, count,
                                            " ".join("%d:%.4f" % (i, logprobs[i]) for i in order)))
            if cand is not None:
                pairs = ["%d:%.4f" % (c, logprobs[c]) for c in cand]
                if args.cand_mode == "swap" and len(pairs) > 1:
                    pairs = list(reversed(pairs))
                declared = len(pairs) + (1 if args.cand_mode == "count-bump" else 0)
                line = "CANDLOGPROBS %d %d %s" % (pos, declared, " ".join(pairs))
                emit(line)
                if args.cand_mode == "twice":
                    emit(line)
            elif args.cand_mode == "unsolicited":
                emit("CANDLOGPROBS %d 1 0:%.4f" % (pos, logprobs[0]))
            emit("DONE 0 %d 1.0 0.0 score 0 0 %d 0 0 0 0 0.0 %d 0"
                 % (len(ids), reused, len(ids)))
            continue
        if head == "STATE" and not rest.strip():
            emit("STATE " + json.dumps(state_json(), separators=(",", ":")))
            continue
        if head == "GEN":
            max_new = None
            cont = False
            cont_seen = False
            ids = None
            problem = None
            for token in rest.split():
                if token == "cont=1":
                    if cont_seen:
                        problem = "duplicate cont key"
                        break
                    cont_seen = True
                    cont = True
                elif token.startswith("cont="):
                    problem = "cont must be the literal 1"
                    break
                elif token and token.replace(",", "").isdigit() and max_new is None:
                    max_new = int(token)
                elif token and token.replace(",", "").isdigit():
                    if ids is not None:
                        problem = "two id lists"
                        break
                    ids = [int(part) for part in token.split(",")]
                else:
                    problem = "unknown key %r" % token
                    break
            if problem is None and max_new is None:
                problem = "missing max_new"
            if problem is not None:
                emit("ERR bad request: %s" % problem)
                continue
            if not cont:
                # This stub implements only the explicit-continuation form the P2R-3c
                # service drives; a plain GEN is out of scope here.
                emit("ERR bad request: cont is required in this stub (only cont=1 is implemented)")
                continue
            if not session["armed"]:
                emit("ERR cont=1 needs an armed session")
                continue
            appended = ids or []
            if any(token_id < 0 or token_id >= args.vocab for token_id in appended):
                emit("ERR a token id is outside the vocabulary")
                continue
            cursor = session["pos"]
            for index in range(max_new):
                emit("T %d" % (index % args.vocab))
            emit("DONE %d %d 0.0 1.0 length 0 0 %d 0 0 0 0 0.0 %d 0"
                 % (max_new, len(appended), cursor, len(appended)))
            session["pos"] = cursor + len(appended) + max_new
            continue
        emit("ERR expected: CAPS | STATE | SCORE <id,id,...> [topk=N] [cand=id,id,...] "
             "[keep=1] | GEN <max_new> cont=1 [<id,id,...>]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
