#!/usr/bin/env python3
"""stub_engine_official_q36.py - TEST-ONLY stub of the official q36 serve line protocol.

This is not the official engine and is never used by score_api.py in a real run: it is a
small stand-in that speaks the official protocol's minimal surface the way `serve/server.py`
drives the engine, so the backend-adapter unit tests can run a real child process over real
pipes without a GPU:

    engine -> client:  READY <ctx> stop | PP <done> <total> <ms> <tok/s> | T <id> |
                       DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> ...
    client -> engine:  GEN <max_new> [key=value ...] <id,id,...> | STOP | QUIT

It deliberately does NOT implement CAPS / SCORE / SAVE / LOAD / STATE (the qwen35_llama
additions) and answers them with ERR, so a test can prove that this protocol has no such
commands. Generated tokens are deterministic: token i = (sum(prompt ids) + i) mod vocab,
finish=length (no EOS). STOP between requests is stale and dropped, like the engine's
stale-STOP rule. `INFO batch_slots` is never printed.

    python stub_engine_official_q36.py --vocab 64
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--serve", action="store_true", help="accepted for CLI parity, ignored")
    args = parser.parse_args()

    out = sys.stdout.buffer

    def emit(line: str) -> None:
        out.write(line.encode("utf-8") + b"\n")
        out.flush()

    emit("READY %d stop" % args.context)

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.strip():
            continue
        head, _, rest = line.partition(" ")
        if head == "QUIT":
            return 0
        if head == "STOP":
            continue
        if head == "GEN":
            tokens = rest.split()
            ids = None
            max_new = None
            problem = None
            if len(tokens) < 2:
                problem = "missing max_new and id list" if not tokens else "missing id list"
            elif not tokens[0].isdigit() or int(tokens[0]) < 1:
                problem = "max_new must be a positive decimal integer"
            elif not tokens[-1].replace(",", "").isdigit():
                problem = "missing id list"
            else:
                max_new = int(tokens[0])
                ids = [int(part) for part in tokens[-1].split(",")]
            if problem is None and any(token_id < 0 or token_id >= args.vocab for token_id in ids):
                problem = "a token id is outside the vocabulary"
            if problem is not None:
                emit("ERR bad request: %s" % problem)
                continue
            prompt_tokens = len(ids)
            emit("PP %d %d 1.0 100.0" % (prompt_tokens, prompt_tokens))
            salt = sum(ids)
            for index in range(max_new):
                emit("T %d" % ((salt + index) % args.vocab))
            emit("DONE %d %d 1.0 %.1f length 0 0 0 0 0 0 0 0.0 %d 0"
                 % (max_new, prompt_tokens, 0.5 * max_new, prompt_tokens))
            continue
        emit("ERR expected: GEN <max_new> [key=value ...] <id,id,...> | STOP | QUIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
