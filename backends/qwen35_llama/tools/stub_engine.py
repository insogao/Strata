#!/usr/bin/env python3
"""stub_engine.py - TEST-ONLY stub of the qwen35_llama serve line protocol.

This is not the engine and is never used by score_api.py in a real run: it is a small
stand-in that answers `CAPS` / `SCORE` in the exact line grammar of
docs/backends/qwen35_llama_serve_protocol.md, so the service's 0-GPU unit tests can drive a
real child process over real pipes. Scores are deterministic: logit(id) = -0.5 * id plus a
prompt-dependent offset, over a configurable vocabulary size.

    python stub_engine.py --vocab 64        # speaks READY on stdout, answers on stdin

Only what the P2-3 tests need exists here: no GEN, no tokenizer, no real model.
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
        "state_export": False,
        "disk_kv": False,
        "vision": False,
    }
    emit("READY %d stop" % args.context)
    dropped = 0

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
            problem = None
            for token in rest.split():
                if token.startswith("topk="):
                    if not token[5:].isdigit():
                        problem = "topk must be a non-negative decimal integer"
                        break
                    topk = int(token[5:])
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
            emit("DONE 0 %d 1.0 0.0 score 0 0 0 0 0 0 0 0.0 %d 0" % (len(ids), len(ids)))
            continue
        emit("ERR expected: CAPS | SCORE <id,id,...> [topk=N] [cand=id,id,...] | "
             "GEN <max_new> [key=value ...] <id,id,...>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
