# qwen35_llama serve protocol (M2, official server line protocol)

`strata generate --qwen35-model <gguf> --qwen35-pack <dir> [qwen35 options] --serve` runs the Qwen35 path as a
**resident engine for the official server** (`serve/server.py`, `StrataEngine`): the model is loaded once,
`READY <ctx> stop` is printed, and the process then serves any number of requests from stdin until `QUIT` or
EOF. The line protocol is the official engine's own - `serve/server.py` parses exactly these lines and needs no
changes. **P2-1** adds two commands the current server does not send yet: `CAPS` (the capability table) and
`SCORE` (read-only candidate logits); their contract is
`docs/backends/qwen35_llama_capabilities_and_score.md`.

Implementation: `src/program/qwen35_serve.cpp` (wired from `src/program/generate.cpp`; the one-shot CLI path
without `--serve` is untouched). This is a **phase-1 protocol shell**: it is the substrate for the later
server E2E work, not a performance claim.

## How to drive it from the official server

The server starts the engine from its JSON config (`exe` / `args` / `cwd` / `env` / `log`):

```json
{
  "exe": "engine/strata.exe",
  "cwd": "engine",
  "args": [
    "--qwen35-model", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    "--qwen35-pack", "models/qwen36-pack",
    "--max-context", "4096",
    "--qwen35-threads", "8",
    "--qwen35-cache-slots", "0",
    "--qwen35-ubatch", "128",
    "--serve"
  ],
  "env": {},
  "log": "logs/engine.log"
}
```

```text
python -m serve.server --engine strata --config strata.json --port 8080
```

Notes:

- `args` must **not** carry a prompt: serve mode needs no `--tokens`/`--qwen35-prompt`; `--max-new` is per
  request (`GEN <max_new> ...`), not a start-up option.
- `--qwen35-profile <file>` and `--qwen35-cache-slots N` work as in the CLI (expert profile / GPU expert
  cache); nothing about the Qwen35 options changes in serve mode.
- The engine prints **no `INFO ... batch_slots=N`**: the official server therefore stays on its single-request
  path (FIFO-serialized `GEN`), which is exactly this phase's contract - requests are serialized, one at a
  time. Continuous batching is not implemented.
- A relative `exe` resolves against `cwd` (the server makes it absolute; see `serve/server.py`).

## Line protocol (implemented subset)

### Engine -> client

| Line | Meaning |
|---|---|
| `READY <ctx> stop` | printed once after the model + context + expert service are up; `<ctx>` is the actual context requests are checked against; `stop` declares this engine honours `STOP` |
| `PP <done> <total> <ms> <tok/s>` | once per prefill chunk: tokens read so far, total prompt tokens, ms since the prompt read started, this chunk's fresh tokens/s |
| `T <id>` | one generated token; an EOS token is printed too, then the request ends (official EOS counting) |
| `LOGPROBS <pos> <k> <id>:<logprob> <id>:<logprob> ...` | **P2-1**: `SCORE`'s answer - the top-k of the full-vocabulary log-softmax of the last prompt position (`pos` = 0-based last prompt position, `logprob = logit - logsumexp(all logits)` at `%.4f`). One line per `SCORE`, always followed by its `DONE ... score` |
| `DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>` | one per completed request; field order is the official engine's (`serve/server.py::_parse_done`) |
| `CAPS <json>` | **P2-1**: the answer to `CAPS` - the backend capability table, the same one-line JSON `--capabilities` prints (see `docs/backends/qwen35_llama_capabilities_and_score.md`) |
| `ERR <message>` | request-level refusal or failure; **no `DONE` follows**, the process keeps serving |

Values in `DONE` for this phase: `accepted/offered/reused/hits/lookups/ram_blobs/file_blobs/offloaded` are `0`
and `file_mb` is `0.0` (no MTP, no prefix reuse, no cache accounting); `prompt_read` is the number of prompt
tokens actually read - fewer than `prompt_tokens` when a `STOP` ended the read part-way. A `SCORE` request
prints `produced=0` and `finish=score` (and no `T` line).

`finish` values: `stop` (EOS hit), `length` (max_new reached), `cancel` (`STOP`), `nonfinite` (a NaN in the
head logits - the request ends before emitting that token, same as the official `#879` behavior), `score`
(the P2-1 `SCORE` request completed; its answer is the `LOGPROBS` line before the `DONE`).

### Engine diagnostics on stderr (outside the line protocol)

After every completed request - the same point the `DONE` line is printed on stdout - the engine writes one
line to **stderr**:

```text
serve: req <i> finish=<stop|length|cancel|nonfinite|score> produced=<n> prompt_ms=<..> decode_ms=<..>
```

`<i>` is the 1-based index of completed requests of this process, in the same order as the `DONE` lines; the
other fields repeat the `DONE` values. A `SCORE` line appends its actual `k` as `topk=<k>`. The official
server does not read engine stderr, so this line survives server-side stdout handling - it is the engine's own
statement of why each request ended (`cancel` = a live `STOP` ended the request; a `STOP` read before the
request started stays stale and never reaches this line). Added by task 365b; stdout protocol and every
judgement are unchanged (P2-1 added `score`/`topk=<k>` for the new command).

### Client -> engine

| Line | Meaning |
|---|---|
| `GEN <max_new> [key=value ...] <id,id,...>` | one request. Keys implemented: `temperature`, `top_p`, `top_k`, `min_p`, `seed`. `temperature` absent/0 = greedy. **A non-default `penalty_*` or a `logit_bias` is REFUSED with `ERR`** (`penalty_last_n`/`penalty_repeat`/`penalty_freq`/`penalty_present`), never silently ignored. Unknown keys are skipped like the official engine skips them (the conversation-cache keys `ckpt`/`pin`/`cvec`/`pcie_frac`/`spec_min_p`/`aux_cpus`/`img_keys` have no effect here). |
| `SCORE <id,id,...> [topk=N]` | **P2-1**: read-only candidate logits. The prompt is read through the same prefill path as `GEN` (PP lines included) and **no token is generated**; the answer is one `LOGPROBS <pos> <k> <id>:<logprob> ...` line (full-vocabulary log-softmax, default `topk` 5) followed by `DONE ... score` with `produced=0`. Every key other than `topk` is REFUSED with `ERR`. State semantics **(b) independent**: the session is reset before the request and again after it, so the next request starts fresh. Full contract: `docs/backends/qwen35_llama_capabilities_and_score.md`. |
| `CAPS` | **P2-1**: answer the capability table as `CAPS <json>` (the `--capabilities` JSON) and keep serving. |
| `STOP` | cancel the running request: the decode loop checks the flag before every token (and between prefill chunks); the engine answers `DONE ... cancel` and keeps serving (a `SCORE` cancelled during the read answers `DONE ... cancel` with no `LOGPROBS`). A `STOP` that arrives between requests is stale and is dropped at the next request's start. |
| `QUIT` | exit 0 (the server's shutdown path; stdin EOF also exits 0) |

Empty lines are ignored. Any other line (e.g. `BGEN`, `SAVE`, `RESTORE`, `VRAM`, `ENC`, `GENI`) answers
`ERR expected: CAPS | SCORE <id,id,...> [topk=N] | GEN <max_new> [key=value ...] <id,id,...>` and the loop
continues - unknown input never exits or crashes the engine.

## Request lifecycle and state isolation

1. Parse/validate: `max_new >= 1`, token ids in the model's vocabulary, `prompt + max_new <= ctx`; a bad
   request answers `ERR` and the engine stays idle.
2. **State reset**: `llama_memory_clear(mem, true)` - KV cells plus the recurrent (GDN conv/SSM/rollback)
   planes, the same proven reset the two-request diagnostic uses. Positions restart at 0 for every request.
3. Prefill in `n_ubatch`-sized chunks (the context's actual micro-batch, default 128), `PP` after each chunk,
   then the prompt -> decode transition (`qwen35_bridge_finish_prompt()`, the expert-cache workspace loan
   return).
4. Decode: greedy, or `top_k -> top_p -> min_p -> temp -> dist(seed)` when `temperature > 0`; NaN scan before
   every sample; `T` per token; EOS is emitted as a `T` and counted in `produced` (`finish=stop`).
5. `DONE`. The next request repeats from step 2.

Semantics this phase promises:

- **No cross-request prefix reuse**: every prompt is read from token 0 and `reused` is honestly `0`. A later
  phase may add prefix reuse; until then the correctness-first behavior is specified and reported.
- **Cancel leaves no dirty state**: after a cancel in prefill or decode, the next request still resets and runs
  correctly (the reset in step 2 is unconditional).
- **EOS**: `llama_vocab_is_eog` decides the stop; the EOS token is printed and counted, matching the official
  engine.
- **No request-count limit**: any number of `GEN`s are served by one process (the CLI's
  `STRATA_QWEN35_TWO_REQUEST` diagnostic is a CLI mode and is refused under `--serve`).

## Not implemented (phase 1) - answers are refusals, never silent substitutions

| Not implemented | Behavior |
|---|---|
| penalties (`penalty_*` non-default), `logit_bias` | `ERR ...` (explicit refusal) |
| prefix reuse / conversation cache (`ckpt`, `pin`, `reuse` accounting) | requests always read the full prompt; `reused=0`; the keys are skipped |
| continuous batching (`batch_slots`, `BGEN`, slots) | `INFO batch_slots` is never printed, so the server stays single-request; `BGEN` -> `ERR` |
| vision (`GENI`, image embeddings) | `ERR` (unknown directive for this engine) |
| disk sessions (`SAVE`/`RESTORE`) | `ERR` |
| MTP draft/verify/transaction (K1) | **refused at start-up** (non-zero exit, message on stderr) when any `STRATA_QWEN35_MTP*` gate that changes the decode path is set: `STRATA_QWEN35_MTP`, `_SELFTEST`, `_DRAFT`, `_TXN`, `_T3_VERIFY_ORACLE`, `_T3_ATTN_SLICE`, `_T3_FA_INPUTS` |
| `--vram-elastic` / `VRAM` command, expert-cache adaptivity | not part of this engine; `VRAM` -> `ERR` |
| native-engine diagnostics (`STRATA_QWEN35_GRAPH_STATS`, DECODE_TIMESTAMPS, host timeline, ...) | inert in serve mode (never default-on; no effect on the protocol) |

## Default-off discipline

Serve mode opens **no** experimental gate by itself: MTP (`STRATA_QWEN35_MTP*`), COMPACT slot variants, the
two-request diagnostic and the layer-replay diagnostic are all still `env`-controlled and default OFF; the ones
that would change what the protocol means are refused (table above) instead of being half-supported. `--serve`
itself only changes the process shape (resident engine vs. one-shot runner).

## Build (standalone backend)

This backend is its own CMake project (see `docs/backends/qwen35_llama.md`, "Build isolation"):

```text
powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1
```

Artifact: `backends/qwen35_llama/build/strata.exe` (target `strata`, built with `-DSTRATA_QWEN35=ON`).
`--serve` requires `-DSTRATA_QWEN35=ON`, like `--qwen35-model` itself.
