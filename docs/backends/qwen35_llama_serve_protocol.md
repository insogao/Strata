# qwen35_llama serve protocol (M2, official server line protocol)

`strata generate --qwen35-model <gguf> --qwen35-pack <dir> [qwen35 options] --serve` runs the Qwen35 path as a
**resident engine for the official server** (`serve/server.py`, `StrataEngine`): the model is loaded once,
`READY <ctx> stop` is printed, and the process then serves any number of requests from stdin until `QUIT` or
EOF. The line protocol is the official engine's own - `serve/server.py` parses exactly these lines and needs no
changes. **P2-1** adds two commands the current server does not send yet: `CAPS` (the capability table) and
`SCORE` (read-only candidate logits); their contract is
`docs/backends/qwen35_llama_capabilities_and_score.md`. **P2-5** adds the single-session state
export/import commands `SAVE <path>` / `LOAD <path>` / `STATE` (the full contract, boundaries and the P2-6
GPU parity plan: `docs/backends/qwen35_llama_session_state.md`). **P2R-1** extends `SCORE` with the optional
`cand=<id,...>` key (explicit candidate ids, answered whatever their top-k rank) and `topk=0`
(candidates only, no `LOGPROBS` line); a request without `cand=` is answered byte-identically to the P2-1
engine, except for the `ERR expected:` hint spelled out below.

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
| `LOGPROBS <pos> <k> <id>:<logprob> <id>:<logprob> ...` | **P2-1**: `SCORE`'s answer - the top-k of the full-vocabulary log-softmax of the last prompt position (`pos` = 0-based last prompt position, `logprob = logit - logsumexp(all logits)` at `%.4f`). One line per `SCORE`, followed by its `DONE ... score`; **P2R-1**: absent when the request asked `topk=0` (explicit candidates only) |
| `CANDLOGPROBS <pos> <n> <id>:<logprob> <id>:<logprob> ...` | **P2R-1**: `SCORE`'s `cand=` answer - the requested ids, **in request order** (no sort, no dedup) with `logprob = logit - logsumexp`, printed at `%.4f` from the **same last-position row and the same full-vocab lse** as `LOGPROBS` (<pos> is the same; no second forward/prefill). One line per `SCORE` that carried `cand=`, after `LOGPROBS` (if any) and before `DONE ... score` |
| `DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>` | one per completed request; field order is the official engine's (`serve/server.py::_parse_done`) |
| `CAPS <json>` | **P2-1**: the answer to `CAPS` - the backend capability table, the same one-line JSON `--capabilities` prints (see `docs/backends/qwen35_llama_capabilities_and_score.md`) |
| `SAVED <bytes> <pos> <ms>` | **P2-5**: `SAVE`'s answer - the state blob size, the cursor (`pos`, occupied positions of sequence 0) and the elapsed ms |
| `LOADED <bytes> <pos> <ms>` | **P2-5**: `LOAD`'s answer after a fully successful import (same fields, measured on the imported state) |
| `STATE <json>` | **P2-5**: the answer to `STATE` - one JSON line (`loaded`/`pos`/`bytes`/`path`/`covers`/`session`/`origin`/`file_bytes`/`max_context`/`not_covered`; fields in `docs/backends/qwen35_llama_session_state.md`) |
| `ERR <message>` | request-level refusal or failure; **no `DONE` follows**, the process keeps serving |

Values in `DONE` for this phase: `accepted/offered/hits/lookups/ram_blobs/file_blobs/offloaded` are `0` and
`file_mb` is `0.0` (no MTP, no cache accounting); `reused` is `0` for an isolated request and the carried
session cursor for a P2-5 continuation (`GEN` after `SAVE`/`LOAD`); `prompt_read` is the number of prompt
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
other fields repeat the `DONE` values. A `SCORE` line appends its actual `k` as `topk=<k>`; when the request
carried `cand=`, one more field `cand=<n>` is appended (`... topk=5 cand=4`, and `... topk=0 cand=4` for a
candidates-only request), and without `cand=` the line is byte-identical to the P2-1 engine. The official
server does not read engine stderr, so this line survives server-side stdout handling - it is the engine's own
statement of why each request ended (`cancel` = a live `STOP` ended the request; a `STOP` read before the
request started stays stale and never reaches this line). Added by task 365b; stdout protocol and every
judgement are unchanged (P2-1 added `score`/`topk=<k>` for the new command; P2R-1 added `cand=<n>`).

### Client -> engine

| Line | Meaning |
|---|---|
| `GEN <max_new> [key=value ...] <id,id,...>` | one request. Keys implemented: `temperature`, `top_p`, `top_k`, `min_p`, `seed`. `temperature` absent/0 = greedy. **A non-default `penalty_*` or a `logit_bias` is REFUSED with `ERR`** (`penalty_last_n`/`penalty_repeat`/`penalty_freq`/`penalty_present`), never silently ignored. Unknown keys are skipped like the official engine skips them (the conversation-cache keys `ckpt`/`pin`/`cvec`/`pcie_frac`/`spec_min_p`/`aux_cpus`/`img_keys` have no effect here). |
| `SCORE <id,id,...> [topk=N] [cand=id,id,...]` | **P2-1 / P2R-1**: read-only logits. The prompt is read through the same prefill path as `GEN` (PP lines included) and **no token is generated**; the answer is `LOGPROBS <pos> <k> <id>:<logprob> ...` (full-vocabulary log-softmax, default `topk` 5) followed by `DONE ... score` with `produced=0`. Keys other than `topk`/`cand` are REFUSED with `ERR`. **P2R-1** — `cand=id,id,...`: comma-separated **explicit candidate ids** answered from the same row/lse as the top-k, whatever their rank, as one `CANDLOGPROBS <pos> <n> <id>:<logprob> ...` line in request order. Rules (all refusals `ERR`, the engine keeps serving): ids must be decimal and non-negative; **duplicates are refused** (never silently deduplicated); **more than 2048 ids are refused** (use `topk=` for a wide distribution); a **repeated `cand=` key is refused** (unlike `topk`, which keeps last-value-wins); an id outside `[0, n_vocab)` answers the same `ERR a token id is outside the vocabulary` as a prompt id. `topk=0` is now legal **only together with `cand=`**: it suppresses `LOGPROBS` and answers the candidates alone; `topk=0` without `cand=` is refused (`ERR bad request: topk`, the legacy text). A request without `cand=` is answered **byte-identically** to the P2-1 engine (stdout, `DONE` and the stderr diagnostic). State semantics **(b) independent**: the session is reset before the request and again after it, so the next request starts fresh. Full contract: `docs/backends/qwen35_llama_capabilities_and_score.md`; the GPU byte-compat cross-check against the P2-1 exe was **run by P2R-2 (2026-10-15) and passed** (same-position `LOGPROBS` byte-identical; evidence in `results/opencode-execution/strata-unified-p2r2-candidate-gpu/`, write-up in `docs/backends/qwen35_llama_score_api_e2e.md` §6). |
| `CAPS` | **P2-1**: answer the capability table as `CAPS <json>` (the `--capabilities` JSON) and keep serving. |
| `SAVE <path>` | **P2-5**: export the current sequence-0 session state (attention KV + GDN conv/SSM planes + positions, the raw `llama_state_seq_get_data` bytes) to `<path>` (absolute, printable ASCII, no `..`; optional allow-list `STRATA_QWEN35_SESSION_ROOTS`). Replaces an existing file atomically (`<path>.tmp` + replace). On success the session is armed and the next `GEN` continues it. Full contract: `docs/backends/qwen35_llama_session_state.md`. |
| `LOAD <path>` | **P2-5**: import a state file. The blob is imported into a TEMPORARY context first and swapped in only after the import fully succeeded, so a bad path/file/foreign blob answers `ERR` and leaves the live session untouched. On success the session is armed (`STATE`: `loaded=true`). |
| `STATE` | **P2-5**: answer one JSON line (`STATE <json>`) describing the session bookkeeping; never touches the state. |
| `STOP` | cancel the running request: the decode loop checks the flag before every token (and between prefill chunks); the engine answers `DONE ... cancel` and keeps serving (a `SCORE` cancelled during the read answers `DONE ... cancel` with neither `LOGPROBS` nor `CANDLOGPROBS`). A `STOP` that arrives between requests is stale and is dropped at the next request's start. |
| `QUIT` | exit 0 (the server's shutdown path; stdin EOF also exits 0) |

Empty lines are ignored. Any other line (e.g. `BGEN`, `RESTORE`, `VRAM`, `ENC`, `GENI`) answers
`ERR expected: CAPS | STATE | SAVE <path> | LOAD <path> | SCORE <id,id,...> [topk=N] [cand=id,id,...] |
GEN <max_new> [key=value ...] <id,id,...>` and the loop continues - unknown input never exits or crashes the
engine. (P2R-1 added the `[cand=id,id,...]` hint to this fallback line - the **only** pre-existing text this
task changes; every other old-client answer stays byte-identical. `RESTORE` is the official engine's own
session command and is **not** implemented here; P2-5's `LOAD` is the command of this engine - see the
session-state doc.)

## Request lifecycle and state isolation

1. Parse/validate: `max_new >= 1`, token ids in the model's vocabulary, `start + prompt + max_new <= ctx`
   (`start` is 0 unless a P2-5 session is armed); a bad request answers `ERR` and the engine stays idle.
2. **State reset**: `llama_memory_clear(mem, true)` - KV cells plus the recurrent (GDN conv/SSM/rollback)
   planes, the same proven reset the two-request diagnostic uses. Positions restart at 0 for every request -
   unless a `SAVE`/`LOAD` armed the session (P2-5), in which case this step is skipped and the prefill starts
   at the session cursor instead.
3. Prefill in `n_ubatch`-sized chunks (the context's actual micro-batch, default 128), `PP` after each chunk,
   then the prompt -> decode transition (`qwen35_bridge_finish_prompt()`, the expert-cache workspace loan
   return).
4. Decode: greedy, or `top_k -> top_p -> min_p -> temp -> dist(seed)` when `temperature > 0`; NaN scan before
   every sample; `T` per token; EOS is emitted as a `T` and counted in `produced` (`finish=stop`).
5. `DONE`. The next request repeats from step 2 (a session `GEN` it appends to the session; `reused` reports
   the carried cursor).

Semantics this phase promises:

- **No cross-request prefix reuse by default**: without `SAVE`/`LOAD` every prompt is read from token 0 and
  `reused` is honestly `0`. A later phase may add prefix reuse; until then the correctness-first behavior is
  specified and reported.
- **Session continuation is explicit (P2-5)**: after a successful `SAVE` or `LOAD` the session is armed; the
  next `GEN` prefills its prompt at the current cursor and `reused` reports the carried positions. `SCORE`'s
  (b) reset (or a process restart) clears the session. Details: `docs/backends/qwen35_llama_session_state.md`.
- **Cancel leaves no dirty state**: after a cancel in prefill or decode, the next request still resets (or,
  in session mode, continues from the self-consistent partial state) and runs correctly.
- **EOS**: `llama_vocab_is_eog` decides the stop; the EOS token is printed and counted, matching the official
  engine.
- **No request-count limit**: any number of `GEN`s are served by one process (the CLI's
  `STRATA_QWEN35_TWO_REQUEST` diagnostic is a CLI mode and is refused under `--serve`).

## Not implemented (phase 1) - answers are refusals, never silent substitutions

| Not implemented | Behavior |
|---|---|
| penalties (`penalty_*` non-default), `logit_bias` | `ERR ...` (explicit refusal) |
| prefix reuse / conversation cache (`ckpt`, `pin`, `reuse` accounting) | requests always read the full prompt; `reused=0` except for a P2-5 session continuation (the carried cursor); the keys are skipped |
| continuous batching (`batch_slots`, `BGEN`, slots) | `INFO batch_slots` is never printed, so the server stays single-request; `BGEN` -> `ERR` |
| vision (`GENI`, image embeddings) | `ERR` (unknown directive for this engine) |
| the official engine's disk-session grammar (`RESTORE`, `SERR`/`SAVED <tokens> <bytes> <ms>` responses) | not implemented; this engine's own single-session commands are `SAVE`/`LOAD`/`STATE` (P2-5), with the answers documented above |
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
