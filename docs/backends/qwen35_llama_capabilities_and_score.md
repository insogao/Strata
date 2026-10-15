# qwen35_llama capabilities table and SCORE (P2-1; `cand=`/`topk=0` added by P2R-1)

Phase-2 backend-interface work, task **P2-1** (branch `phase2/backend-interface`), extended by **P2R-1**
(branch `phase2r/review-gaps`). Two additions to the Qwen35 path, both implemented in
`backends/qwen35_llama/**` only:

1. **Capability table** - a machine-readable statement of what this backend actually does, available as
   `--capabilities` (CLI, one JSON line, exit 0, no model load, no GPU) and as the `CAPS` command of the
   serve line protocol (same JSON, process keeps serving).
2. **`SCORE`** - a read-only candidate-logits command: prefill the given ids through the **same** path as
   `GEN`, generate **no** token, and answer the last position's top-k log-softmax over the **full**
   vocabulary, then `DONE ... score`. **P2R-1** adds the optional `cand=<id,...>` key (explicit candidate
   ids answered from the same row/lse, whatever their top-k rank) and allows `topk=0` (candidates only,
   no `LOGPROBS` line). The GPU cross-check of the additions **ran in P2R-2 (2026-10-15) and passed**
   (evidence: `results/opencode-execution/strata-unified-p2r2-candidate-gpu/`; write-up: "P2R-2: GPU
   validation of the P2R-1 additions" in `docs/backends/qwen35_llama_score_api_e2e.md`).

The field values below are the implementation's own values, each with its source. Nothing in this table is a
product claim; features that do not exist are `false` and listed in "Not implemented" (a wrong `true` would be
worse than a `false`).

## CLI: `--capabilities`

```text
strata generate --capabilities [--qwen35-model <gguf>] [--max-context N]
```

- Prints **one JSON line** (no prefix) to stdout and exits 0.
- It **loads no model and touches no GPU**. Without `--qwen35-model`, `mtp.available` is answered `false`
  (the availability is simply not determined; nothing is fabricated).
- With `--qwen35-model`, only the **GGUF metadata header** is read (never a tensor payload), through the same
  `strata::GgufModel` reader the MTP probe uses (`src/core/qwen35_mtp.cpp`), to answer `mtp.available`. An
  unreadable file also answers `false`.
- `context_max` is the `--max-context` value (default 4096) - the context a serve process with these argv
  would actually check requests against. `--capabilities` is inert unless the build has
  `-DSTRATA_QWEN35=ON` (without it: a one-line error, exit 2).

## serve: `CAPS`

After `READY`, a `CAPS` line on stdin answers:

```text
CAPS {"protocol":"strata-serve-v1", ... }
```

The JSON is the **same string** `--capabilities` prints (one builder, `qwen35_capabilities_json` in
`src/program/qwen35_serve.cpp`), except that a serving process passes the **loaded model's actual values**:
`context_max` is the live `llama_n_ctx` (the value `READY` advertised), and `mtp.available` comes from the
loaded model's metadata. `CAPS` never disturbs a request: it is answered between requests, like any queued
line.

## Field-by-field values and sources

Example (production model, no MTP layers; `STRATA_Q6K_EXPERTS=ON` build - see `build.ps1`; the two
`score_*` fields are P2R-1, `state_keep_score`/`session_continue` are P2R-3a):

```json
{"protocol":"strata-serve-v1","continuous_requests":true,"cancel":true,"state_isolation":true,"prefix_cache":false,"batching":false,"mtp":{"available":false,"enabled":false,"default":false,"k":null},"expert_formats":["Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q4_K","Q5_K","Q6_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS","IQ1_M","Q2_0"],"context_max":4096,"logits_read":true,"score_explicit_candidate_ids":true,"score_max_candidate_ids":2048,"state_keep_score":true,"session_continue":true,"state_export":true,"disk_kv":false,"vision":false}
```

| Field | Value | Source / evidence |
|---|---|---|
| `protocol` | `"strata-serve-v1"` | The serve line protocol: `READY` / `GEN` / `PP` / `T` / `DONE` / `STOP` / `QUIT` (task 362), `CAPS` and `SCORE` (P2-1), `SAVE` / `LOAD` / `STATE` (P2-5). `src/program/qwen35_serve.cpp`, header comment. |
| `continuous_requests` | `true` | One resident process serves any number of requests (tasks 363d/364c: same process, consecutive requests, no request-count limit). |
| `cancel` | `true` | `STOP` ends an in-flight request at its next prefill chunk / decode token with engine-level `DONE ... cancel` (task 366). |
| `state_isolation` | `true` | Every request begins with `llama_memory_clear(mem, true)` - KV cells plus the recurrent GDN planes; `reused` is 0, no prefix is carried over (tasks 363d/364c). `reset_session()` in `qwen35_serve.cpp`. P2-5 qualifier: this holds for request streams with no armed session; after an explicit `SAVE`/`LOAD` the next `GEN` continues the session at the cursor and `reused` reports the carried positions (see `docs/backends/qwen35_llama_session_state.md`). |
| `prefix_cache` | `false` | No automatic cross-request prefix reuse exists; every prompt is read from token 0 (`reuse=0` in every `DONE`, except the explicit P2-5 session continuation). The conversation-cache keys (`ckpt`/`pin`/...) are skipped by `GEN` and have no effect. |
| `batching` | `false` | No `INFO batch_slots` is ever printed and requests are FIFO-serialized (`qwen35_serve.cpp`); a server therefore stays on its single-request path. `BGEN` is answered `ERR`. |
| `mtp.available` | bool | The model's declared MTP (NextN) layer count: GGUF key `qwen35moe.nextn_predict_layers` > 0 (declared `<arch>.nextn_predict_layers` at `vendor/llama.cpp/src/llama-arch.cpp:229`, loaded into `hparams.n_layer_nextn` at `llama-model.cpp:1341`). Measured here: the production `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` has no such key (`false`); `models/experimental/mtp-qwen36-5bc3e238/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` declares `1` (`true`). Without a model path the CLI answers `false` (undetermined, not fabricated). |
| `mtp.enabled` | bool | Exact env gate: `STRATA_QWEN35_MTP=1` (`src/core/qwen35_mtp.cpp:263-267`). `default` is always `false`. In a **serving** process this is `false` by construction: the STRATA_QWEN35_MTP* gates are refused before the model load (`qwen35_serve.cpp`), so a process that answers `CAPS` cannot have it set. |
| `mtp.default` | `false` | MTP is opt-in everywhere; nothing enables it by itself. |
| `mtp.k` | `null` | There is **no** runtime draft-length parameter in this phase: the MTP draft/verify/commit machinery is not implemented (the probe is schema-only), so no honest integer exists. `null`, not an invented number. |
| `expert_formats` | array | The compile-time grouped-expert kernel whitelist of the GPU native-expert path: the union of the gate/up and down role tables `STRATA_GU_FMTS` / `STRATA_D_FMTS` (`vendor`-independent source: `src/kernels/cuda/iq_kernels.cu:614-621`), exported by `kernels::iq_expert_role_format` (`iq_kernels.cu:2722`, declared `include/strata/kernels/iq_kernels.hpp:56`). A build without `STRATA_Q6K_EXPERTS` answers without `Q6_K`. This is the binding set: `qwen35_bridge_init` refuses a pack at startup unless every layer pair passes `kernels::native_expert_supported` (`src/core/qwen35_bridge.cpp:1848-1855`). See "expert_formats roles" below. |
| `context_max` | int | CLI: the `--max-context` argv value (default 4096) - what this invocation would check requests against. serve: the live `llama_n_ctx` (identical to the `READY <ctx>` value). A request whose prompt exceeds it is answered `ERR` and the process keeps serving. |
| `logits_read` | `true` | This task's `SCORE` command (see below); no other logits export exists. |
| `score_explicit_candidate_ids` | `true` | **P2R-1**: `SCORE` accepts the optional `cand=<id,...>` key and answers every requested id from the same last-position row/lse, whatever its top-k rank (`CANDLOGPROBS` line; see below). Requests without `cand=` remain byte-identical to the P2-1 engine. |
| `score_max_candidate_ids` | `2048` | **P2R-1**: the hard cap of explicit `cand=` ids per `SCORE`; above it the request is refused `ERR bad request: cand has more than 2048 ids (use topk= for a wide distribution)`. A constant of the implementation (`kScoreMaxCand` in `qwen35_serve.cpp`), not a tunable. |
| `state_keep_score` | `true` | **P2R-3a**: `SCORE` accepts the optional `keep=1` key - the same read-only logprobs answer, but no reset: the prompt is appended at the armed session's cursor (from 0 when none was armed), `DONE`'s `reused` reports the carried cursor and the session stays armed (`STATE` `origin="score-keep"` when this request armed it). A request without `keep=` is byte-identical to the pre-P2R-3a engine. |
| `session_continue` | `true` | **P2R-3a**: `GEN` accepts the optional `cont=1` key - an explicit continuation that requires an armed session (`ERR cont=1 needs an armed session` otherwise) and allows an **empty id list** (pure continuation). `DONE` reports `prompt_tokens`/`prompt_read` = the tokens actually appended/read and `reused` = the cursor before the append; the session stays armed. A request without `cont=` is byte-identical to the pre-P2R-3a engine. |
| `state_export` | `true` | P2-5: single-session state export/import exists - `SAVE <path>` / `LOAD <path>` / `STATE` (attention KV + GDN conv/SSM planes + positions, the raw `llama_state_seq` bytes). The official engine's own `RESTORE` command and its `SAVED <tokens> ...`/`SERR` grammar are **not** implemented; contract, boundaries (no cross-machine/cross-build guarantee; same-process and cross-process parity on this machine/build verified by P2-6, section 11 of the session-state doc) and failure semantics: `docs/backends/qwen35_llama_session_state.md`. Value flipped from `false` in P2-5 (the P2-2 captures below still show the old value - they are verbatim history of that build). |
| `disk_kv` | `false` | No disk KV; a session file is an explicit export, nothing is loaded, cached or reused implicitly, and nothing survives the process without `SAVE`. |
| `vision` | `false` | The Qwen35 serve path implements no vision (`GENI` is answered `ERR`); the native engine's `--vision` is a different engine and not reachable from `--qwen35-model`. |

P2R-3a note: state-keeping score (`SCORE ... keep=1` / `state_keep_score`) and conditional continuation
(`GEN ... cont=1` / `session_continue`) are implemented in the engine (contracts:
`docs/backends/qwen35_llama_session_state.md` §3b and `docs/backends/qwen35_llama_serve_protocol.md`).
**GPU evidence: P2R-3b ran 2026-10-15** (one native, new exe `7c441a94…`, rc=0) and confirmed the flow's
load-bearing facts: continuation `prompt_read=0`/`prompt_ms=0` (the 4K prefix is not re-prefilled), the
`origin="score-keep"` bookkeeping, `ERR cont=1 needs an armed session` after a plain `SCORE`, the fresh
(`reused=0`) path after a session clear, and the `keep=1` append context guard (`ERR session (...) +
prompt (...) exceeds the context`); in the same run the kept state reproduced a fresh full-prefill
reference **token-for-token 16/16**. The pre-registered reference schedule itself (identity vs the
*immediately following* `GEN <P_long>`) is **not executable at ctx4096** - a plain `GEN` continues an
armed session and its P_long append overflows; those pairwise comparisons are recorded as FAIL-as-written
in `qwen35_llama_score_api_e2e.md` §P2R-3b, and a corrected-sequence re-check is a separate dispatch. The
service exposure is P2R-3c.

### expert_formats roles

The two kernel roles are not interchangeable - a layer's gate/up type and down type have separate tables:

| Role | Formats (this build, `STRATA_Q6K_EXPERTS=ON`) |
|---|---|
| gate/up | Q5_0, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q2_0 |
| down | Q5_0, Q4_0, Q4_1, Q5_1, Q8_0, Q6_K, IQ4_NL, IQ3_S, IQ4_XS, Q2_0 |

`expert_formats` in the JSON is the **union** (the schema has one array): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K,
Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q2_0. The production pack's expert
lines are `gu_type 22 / d_type 21` = IQ2_S gate/up + IQ3_S down, both listed.

Two honest limits of this field:

- It is the **accelerated-kernel** whitelist, taken from the compile-time tables - it is not a CPU/ISA probe.
  A machine whose CPU lacks AVX2/AVX-512 has the same list (the kernels' runtime dispatch is a separate
  question; the GPU kernels are the ones this backend requires).
- The CPU fallback (`native_fmt`, ggml-cpu's generic `vec_dot`) can compute other types on their own, but a
  pack using one is refused at startup by `native_expert_supported`, so it is not a format this backend runs.
  Not listed here on purpose.

## `SCORE` - read-only candidate logits

### Syntax

```text
SCORE <id,id,...> [topk=N] [cand=id,id,...]
```

- `<id,id,...>`: comma-separated decimal token ids of the prompt (the same grammar `GEN` uses; spaces inside
  the list are rejected). The id list may appear before or after the keys; exactly one id list per command.
- `topk=N`: optional, default **5**; `N` must be a **non-negative** decimal integer (a repeated key takes the
  last value, unchanged). `topk=0` is legal **only together with `cand=`**: it suppresses the `LOGPROBS` line
  and answers the explicit candidates alone; `topk=0` without `cand=` is refused (`ERR bad request: topk` -
  an empty answer is no answer).
- `cand=id,id,...` (**P2R-1**): optional; comma-separated decimal token ids to answer explicitly. Rules
  (each violation is an `ERR` and the engine keeps serving):
  - non-decimal / negative / empty list -> `ERR bad request: cand`;
  - **duplicate id -> `ERR bad request: duplicate cand id <id>`** - explicit answer only, never silently
    deduplicated (so the answer always mirrors the question one-to-one);
  - **more than 2048 ids -> `ERR bad request: cand has more than 2048 ids (use topk= for a wide
    distribution)`** (the cap is also advertised as `score_max_candidate_ids`);
  - **a repeated `cand=` key -> `ERR bad request: duplicate cand key`** - deliberately NOT `topk`'s
    last-value-wins: a silently dropped candidate list would answer a different question;
  - an id outside `[0, n_vocab)` is refused by the same check as the prompt ids, with the same text
    `ERR a token id is outside the vocabulary`.
- Any other key is refused with `ERR` - a silently ignored key typo would answer a different question than
  the one asked. (Unlike `GEN`, which skips the conversation-cache keys, `SCORE` has no cache keys to skip.)
- Bad syntax, an out-of-vocabulary id, or a prompt longer than the context is answered `ERR ...` and the
  engine keeps serving - the same contract as `GEN`.
- **Backward compatibility (hard criterion)**: a request **without** `cand=` is answered **byte-for-byte
  identically** to the P2-1 engine - the `LOGPROBS` line, the `DONE` fields and the stderr diagnostic. The
  only pre-existing text P2R-1 changes is the fallback `ERR expected: ...` line, which now shows
  `SCORE <id,id,...> [topk=N] [cand=id,id,...]`. The same-request cross-check against the old exe was
  **run in P2R-2 (2026-10-15)**: at three same-position requests (cold P_short, warm#1/warm#2 P_long) the
  old (`6dee0c66...`) and new (`d0321a98...`) exes printed **byte-identical `LOGPROBS` lines**; `DONE`
  and the stderr diagnostic are byte-identical outside their timing fields (`prompt_ms`/`decode_ms`).

### Semantics (implemented exactly as listed)

1. The prompt is read through the **same** prefill path as `GEN` (`run_prefill` in `qwen35_serve.cpp`):
   `n_ubatch`-sized chunks, one `PP` line per chunk, then the prompt -> decode transition
   (`qwen35_bridge_finish_prompt`). A `STOP` during the read cancels it (see below). **No token is
   generated**: the sampler is not created nor advanced.
2. The **last position's logits** are taken (`llama_get_logits_ith(ctx, -1)`; the context runs with
   `n_outputs_max = 1`). The row is the distribution for the token that would follow the last prompt token.
   **One** log-softmax over the full vocabulary is computed from it; everything below reads that row and
   that lse - there is no second forward/prefill for any consumer.
3. Unless `topk=0` was requested, one line is printed:

   ```text
   LOGPROBS <pos> <k> <id>:<logprob> <id>:<logprob> ...
   ```

   - `<pos>`: the **0-based position of the last prompt token** (`prompt_tokens - 1`); positions restart at 0
     for every request (the state reset). With a 100-token prompt, `pos` is 99.
   - `<k>`: the number of pairs actually returned = `min(topk, n_vocab)` - always exactly `topk` unless `topk`
     exceeds the vocabulary size (151936 for this model).
   - `<logprob>`: the **full-vocabulary log-softmax**,
     `logprob(i) = logits[i] - (max(logits) + log(sum_j exp(logits[j] - max(logits))))`, printed with 4
     decimals (`%.4f`). The denominator is **every** one of the `n_vocab` logits - the top-k never
     renormalizes, so `sum` of any selection need not be 1. The selection itself is the `k` largest logits,
     ties broken by ascending id (deterministic order).
3b. If the request carried `cand=`, one line is printed after `LOGPROBS` (if any) and before `DONE`:

   ```text
   CANDLOGPROBS <pos> <n> <id>:<logprob> <id>:<logprob> ...
   ```

   - `<n>`: the number of requested candidates (1..2048, `len(cand)` of the command).
   - order: **exactly the request's `cand=` order** - no sort, no dedup (duplicates were refused at parse
     time), so pair `j` in the answer corresponds to candidate `j` in the command.
   - `<pos>` and each `<logprob>`: the same `pos`, the same row and the **same full-vocab lse** as the
     `LOGPROBS` line (`logprob = logits[id] - lse`, `%.4f`) - a requested id is answered whatever its top-k
     rank, including ids far below the top-k; the value is not renormalized over the candidate set.
4. Then the request's completion line, same field order as `GEN` (`serve/server.py::_parse_done`):

   ```text
   DONE 0 <prompt_tokens> <prompt_ms> 0.0 score 0 0 0 0 0 0 0 0.0 <prompt_read> 0
   ```

   `produced=0` (nothing was generated), `decode_ms=0.0`, `finish=score`; the cache counters are 0 (nothing
   is reused) and `prompt_read` is the number of prompt tokens read (fewer than `prompt_tokens` only when a
   `STOP` cancelled the read; a cancelled read answers `finish=cancel` and **no** `LOGPROBS`, because there is
   no complete last-position row to score).
5. **State semantics: (b) independent.** By construction, not by convention: `SCORE` starts from the same
   `llama_memory_clear(mem, true)` as `GEN`, and when it is done (answer, cancel, or NaN) it **resets the
   session again**, so the engine is never left at the prompt's prefill state. The next request - `GEN` or
   `SCORE` - always starts fresh, and `reused` stays 0. A `SCORE` has no observable side effect beyond its
   own answer (it does not advance any sampler; there is none).
6. Error handling, all `ERR ...` and the service continues:
   - malformed line / two id lists / unknown key / negative `topk` -> `ERR bad request: ...`;
   - `topk=0` without `cand=` -> `ERR bad request: topk` (unchanged legacy text);
   - malformed `cand=` (empty, non-decimal, negative), duplicate cand id, duplicate `cand=` key,
     more than 2048 cand ids -> `ERR bad request: ...` (texts above);
   - an id outside `[0, n_vocab)` (prompt **or** `cand`) -> `ERR a token id is outside the vocabulary`;
   - more prompt tokens than the context -> `ERR prompt (...) exceeds the context (...)`;
   - prefill failure or an unreturned prompt workspace -> `ERR prefill failed` / `ERR the prompt workspace
     loan was not returned`;
   - a NaN anywhere in the head logits -> `ERR score: the head logits are not all finite` (the same
     finite-logits invariant `GEN` enforces as `DONE ... nonfinite`; `SCORE` does not emit a partial answer).
   A `STOP` during the read answers `DONE ... cancel` with neither `LOGPROBS` nor `CANDLOGPROBS`.

### Level of approximation

None: the answer is the exact full-vocabulary log-softmax of the last row (no top-k denominator, no
truncation of the sum). The only numerical caveat is the `%.4f` print precision and the row's own fp32
arithmetic; the NaN guard rejects non-finite distributions instead of printing them.

## Not implemented (fields that are `false`, and why)

| Not implemented | Field / behavior | Why |
|---|---|---|
| cross-request prefix reuse (automatic) | `prefix_cache=false` | every prompt is read from token 0; correctness-first phase-1 contract (documented in `qwen35_llama_serve_protocol.md`). The explicit P2-5 session continuation (`SAVE`/`LOAD` + `GEN`) is a separate, opt-in mechanism |
| continuous batching | `batching=false` | no `INFO batch_slots`; requests are serialized one at a time; `BGEN` -> `ERR` |
| the official disk-session grammar (`RESTORE`, `SERR`) | - | not implemented; the backend's own single-session commands are `SAVE`/`LOAD`/`STATE` (P2-5, `state_export=true`) |
| disk KV | `disk_kv=false` | no implicit disk cache; a session file only exists through an explicit `SAVE` |
| vision | `vision=false` | `GENI`/image embeddings -> `ERR` |
| MTP draft/verify/commit | `mtp.available` is only availability; the runtime is refused at startup when a `STRATA_QWEN35_MTP*` gate is set | the phase-1/2 decode path does not implement it; the probe is schema-only |

Writable penalties/logit_bias are also absent from `GEN` (non-default `penalty_*` / `logit_bias` are refused
with `ERR`) - they are request options, not capability fields.

## How a future `/v1/score` will use this (HTTP not implemented here)

No HTTP server is implemented in P2-1. The intended mapping for the documented subset of a future
`/v1/score` endpoint:

1. Request: `POST /v1/score` with `{"ids":[int,...], "topk": N}`. The server maps it to one line on the
   engine's stdin: `SCORE <ids> topk=N` (default `topk` 5 if omitted), after checking a `CAPS` handshake
   (`logits_read=true`, `context_max` for the size limit).
2. Response: the `LOGPROBS <pos> <k> <id>:<logprob> ...` line is the payload
   (`position=pos`, candidates `[{id, logprob}, ...]` with the full-vocabulary denominator, to be documented
   as such in the endpoint); the following `DONE ... score` line is the usage/timing envelope
   (`prompt_tokens`, `prompt_ms`, `prompt_read`). `ERR` lines map to a 4xx with the message.
3. State: (b) independent - the endpoint is stateless; a `/v1/score` call never affects a later request.

P2R-1 note: the service now exists (P2-3, `tools/score_api.py`). It maps `/v1/score` `candidates` to
`SCORE <ids> topk=N cand=...` and `/v1/decisions` to `SCORE <ids> topk=0 cand=...` (one prefill, no top-k
dependency); the payload semantics (`full_vocab_probability` / `conditional_probability` / always-present
`label_mass` / no business confidence) are specified in `docs/backends/qwen35_llama_score_api_design.md`.

## Build record (P2-1)

- Clean full build (build directory deleted, `build.ps1` configure + build): **exit 0, 384/384 steps**,
  ~527 s, MSVC vcvars64 + CUDA 12.8.1, `CMAKE_CUDA_ARCHITECTURES=89`, `-DSTRATA_QWEN35=ON -DSTRATA_NATIVE_EXPERTS=ON
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON` (the standalone backend's usual flags).
- Artifact: `backends/qwen35_llama/build/strata.exe`, **139,226,624 B**,
  `sha256=66a48ad1366a122ea166ce557e23db85d62332bd62a2b017c2117e1c7cb27745`.
- Functional verification on GPU/cancellation/quality is **P2-2**, not this task (P2-1 is 0 GPU: no exe run).

## GPU verification (P2-2)

Functional verification of the two additions above on the target GPU, executed 2026-10-14 23:21-23:22 (+08:00)
from the neutral build/run root (`cwd=D:/work/StrataUnified`), under `tools/gpu-lock.py run` (one lease per
case, cleanup fence written before launch and removed only on the fully clean path).

Harness facts:

- exe (declared == actual, `argv[0]` == binary): `backends/qwen35_llama/build/strata.exe`,
  `sha256=66a48ad1366a122ea166ce557e23db85d62332bd62a2b017c2117e1c7cb27745`, 139,226,624 B.
- Serve argv: `--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models/pack --serve
  --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096`. Env: only the three
  production gates (`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`, `STRATA_QWEN35_PACKED_IDS=1`,
  `STRATA_QWEN35_EXPERT_ORDER=count_desc`); every other `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*` and
  `CUDA_LAUNCH_BLOCKING` cleared; CUDA runtime DLLs front-loaded on `PATH` (`runtime/cuda-12.8.1/bin`).
- stdin (supervisor-staged, byte-unchanged): 5 lines,
  `sha256=978eb95ad5ef868467d46c6ee523ee61e79eda4603b0a42d6a7d058f9fc29094` -
  `CAPS` / `SCORE <P1: 3869 ids> topk=5` / `GEN 16 temperature=0 <P1>` / `SCORE 999999999,1` / `GEN 16 <P1>`
  (P1 is the same 3869-id prompt as phase-1 task 365b N1).

### N1 - serve path (raw transcript, verbatim; child log lines 189-211)

```text
READY 4096 stop
CAPS {"protocol":"strata-serve-v1","continuous_requests":true,"cancel":true,"state_isolation":true,"prefix_cache":false,"batching":false,"mtp":{"available":false,"enabled":false,"default":false,"k":null},"expert_formats":["Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q4_K","Q5_K","Q6_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS","IQ1_M","Q2_0"],"context_max":4096,"logits_read":true,"state_export":false,"disk_kv":false,"vision":false}
PP 3869 3869 4150 932.6
LOGPROBS 3868 5 18:-0.8937 15:-1.2617 17:-1.2855 16:-3.5211 19:-8.3055
DONE 0 3869 4152.4 0.0 score 0 0 0 0 0 0 0 0.0 3869 0
serve: req 1 finish=score produced=0 prompt_ms=4152.4 decode_ms=0.0 topk=5
PP 3869 3869 2717 1424.7
T 18
T 248046
DONE 2 3869 2718.9 177.0 stop 0 0 0 0 0 0 0 0.0 3869 0
serve: req 2 finish=stop produced=2 prompt_ms=2718.9 decode_ms=177.0
ERR a token id is outside the vocabulary
PP 3869 3869 2760 1402.1
T 18
T 248046
DONE 2 3869 2762.5 31.4 stop 0 0 0 0 0 0 0 0.0 3869 0
serve: req 3 finish=stop produced=2 prompt_ms=2762.5 decode_ms=31.4
```

Results, checked against this document:

1. **CAPS**: one JSON line; parsed field-by-field equal to the table above (mismatch list empty): the `false`
   items are `false`, `mtp.available` is derived (`false` for this production GGUF), `expert_formats` is the
   compile-time 17-format whitelist, `context_max=4096` equals the live `READY` context.
2. **SCORE, no token generated (hard criterion)**: between its `PP` and its `DONE` there is **no `T` line**
   (nothing between `LOGPROBS` and `DONE ... score` either); `produced=0`, `finish=score`; `pos=3868 =
   3869-1`, `k=5` negative full-vocabulary log-softmax values; `prompt_read=3869`.
3. **State semantics (b) holds**: the next `GEN 16` ran normally (2 tokens, EOS id `248046`, `finish=stop`,
   `reused=0`), and after the `ERR` the last `GEN 16` also ran normally - each request independent.
4. **Bad input**: `SCORE 999999999,1` answered `ERR a token id is outside the vocabulary`, no `DONE`, and the
   service kept serving.
5. **Resources/cleanup**: child exit 0, 18.11 s wall for all five lines; job `ActiveProcesses` confirmed 0;
   cleanup fence marker removed (clean path); RAM avail 17,094 -> 5,900 MiB during load, back to 17,217;
   GPU used 1,275 -> peak 7,694 -> 1,281 MiB (fallback after exit).

P2-5 note: the two verbatim captures in this section (N1 `CAPS` and N2 `--capabilities`) show
`"state_export":false` because they were taken from the P2-1/P2-2 build. P2-5 implemented `SAVE`/`LOAD`/`STATE`
and flipped that field to `true`; every other field of the string is unchanged. The P2-5 build's GPU check of
the new commands is P2-6 (the P2-2 checks above - `CAPS` schema, `SCORE`, state semantics (b), bad input - are
the parts of this evidence that stay in force).

### N2 - CLI `--capabilities` (raw stdout, verbatim)

```text
{"protocol":"strata-serve-v1","continuous_requests":true,"cancel":true,"state_isolation":true,"prefix_cache":false,"batching":false,"mtp":{"available":false,"enabled":false,"default":false,"k":null},"expert_formats":["Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q4_K","Q5_K","Q6_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS","IQ1_M","Q2_0"],"context_max":4096,"logits_read":true,"state_export":false,"disk_kv":false,"vision":false}
```

exit 0 in 0.141 s, one line, no `READY`, no model load, no GPU work; the string is identical to the serve
`CAPS` JSON above (one builder). Plain `--capabilities` (no `--qwen35-model`) answers `mtp.available=false` as
documented ("not determined").

### Evidence

Research repo `results/opencode-execution/strata-unified-p2-2-caps-score-gpu/`: `job/`, `audit/`, `result/`
(gpu-lock results, digests `digest-N1.json`/`digest-N2.json` with per-check verdicts), `raw/raw-N1/child_00.log`
(full transcript), `batch/samples-N1.jsonl` (10 s RAM/GPU sampling), `pre-env/`, `batch/state-*`.

P2R-1 note (2026-10-15, branch `phase2r/review-gaps`): the `cand=`/`topk=0` extension and the two
`score_*` capability fields are implemented in `src/program/qwen35_serve.cpp` (parse, `CANDLOGPROBS` block,
CAPS builder); the explicit candidates read the **same** last-position row and the **same** full-vocab lse the
`LOGPROBS` line uses - one prefill, no second forward. New exe `sha256=d0321a98ab1908ef3c0f0d91b1b6e6c42758d79594222c6def425bf406a4bc09`
(139,257,344 B). This task itself was **0 GPU**. **P2R-2 (2026-10-15) has since run the GPU cross-check and it
passed**: old-vs-new exe `LOGPROBS` lines byte-identical at equal positions (cold/warm1/warm2), `cand=`
values byte-identical to the same-process `LOGPROBS` line, explicit out-of-top-k ids finite and `<= 0` with
candidate-set mass `<= 1`, `SCORE` top-1 == the greedy `GEN 1` first token on both prompts, and the
`/v1/decisions` payload (always-present `label_mass`, three probability scopes, `calibrated:false`, no
confidence/threshold field) verified against the live engine. Evidence:
`results/opencode-execution/strata-unified-p2r2-candidate-gpu/`; write-up in
`docs/backends/qwen35_llama_score_api_e2e.md` ("P2R-2: GPU validation of the P2R-1 additions"). The
P2-1/P2-2 captures above were taken from pre-P2R-1 builds and remain verbatim history.
