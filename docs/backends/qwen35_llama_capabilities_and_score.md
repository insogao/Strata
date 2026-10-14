# qwen35_llama capabilities table and SCORE (P2-1)

Phase-2 backend-interface work, task **P2-1** (branch `phase2/backend-interface`). Two additions to the Qwen35
path, both implemented in `backends/qwen35_llama/**` only:

1. **Capability table** - a machine-readable statement of what this backend actually does, available as
   `--capabilities` (CLI, one JSON line, exit 0, no model load, no GPU) and as the `CAPS` command of the
   serve line protocol (same JSON, process keeps serving).
2. **`SCORE`** - a read-only candidate-logits command: prefill the given ids through the **same** path as
   `GEN`, generate **no** token, and answer the last position's top-k log-softmax over the **full**
   vocabulary, then `DONE ... score`.

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

Example (production model, no MTP layers; `STRATA_Q6K_EXPERTS=ON` build - see `build.ps1`):

```json
{"protocol":"strata-serve-v1","continuous_requests":true,"cancel":true,"state_isolation":true,"prefix_cache":false,"batching":false,"mtp":{"available":false,"enabled":false,"default":false,"k":null},"expert_formats":["Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q4_K","Q5_K","Q6_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS","IQ1_M","Q2_0"],"context_max":4096,"logits_read":true,"state_export":false,"disk_kv":false,"vision":false}
```

| Field | Value | Source / evidence |
|---|---|---|
| `protocol` | `"strata-serve-v1"` | The serve line protocol: `READY` / `GEN` / `PP` / `T` / `DONE` / `STOP` / `QUIT` (task 362) plus this task's `CAPS` and `SCORE`. `src/program/qwen35_serve.cpp`, header comment. |
| `continuous_requests` | `true` | One resident process serves any number of requests (tasks 363d/364c: same process, consecutive requests, no request-count limit). |
| `cancel` | `true` | `STOP` ends an in-flight request at its next prefill chunk / decode token with engine-level `DONE ... cancel` (task 366). |
| `state_isolation` | `true` | Every request begins with `llama_memory_clear(mem, true)` - KV cells plus the recurrent GDN planes; `reused` is always 0, no prefix is carried over (tasks 363d/364c). `reset_session()` in `qwen35_serve.cpp`. |
| `prefix_cache` | `false` | No cross-request prefix reuse exists; every prompt is read from token 0 (`reuse=0` in every `DONE`). The conversation-cache keys (`ckpt`/`pin`/...) are skipped by `GEN` and have no effect. |
| `batching` | `false` | No `INFO batch_slots` is ever printed and requests are FIFO-serialized (`qwen35_serve.cpp`); a server therefore stays on its single-request path. `BGEN` is answered `ERR`. |
| `mtp.available` | bool | The model's declared MTP (NextN) layer count: GGUF key `qwen35moe.nextn_predict_layers` > 0 (declared `<arch>.nextn_predict_layers` at `vendor/llama.cpp/src/llama-arch.cpp:229`, loaded into `hparams.n_layer_nextn` at `llama-model.cpp:1341`). Measured here: the production `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` has no such key (`false`); `models/experimental/mtp-qwen36-5bc3e238/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` declares `1` (`true`). Without a model path the CLI answers `false` (undetermined, not fabricated). |
| `mtp.enabled` | bool | Exact env gate: `STRATA_QWEN35_MTP=1` (`src/core/qwen35_mtp.cpp:263-267`). `default` is always `false`. In a **serving** process this is `false` by construction: the STRATA_QWEN35_MTP* gates are refused before the model load (`qwen35_serve.cpp`), so a process that answers `CAPS` cannot have it set. |
| `mtp.default` | `false` | MTP is opt-in everywhere; nothing enables it by itself. |
| `mtp.k` | `null` | There is **no** runtime draft-length parameter in this phase: the MTP draft/verify/commit machinery is not implemented (the probe is schema-only), so no honest integer exists. `null`, not an invented number. |
| `expert_formats` | array | The compile-time grouped-expert kernel whitelist of the GPU native-expert path: the union of the gate/up and down role tables `STRATA_GU_FMTS` / `STRATA_D_FMTS` (`vendor`-independent source: `src/kernels/cuda/iq_kernels.cu:614-621`), exported by `kernels::iq_expert_role_format` (`iq_kernels.cu:2722`, declared `include/strata/kernels/iq_kernels.hpp:56`). A build without `STRATA_Q6K_EXPERTS` answers without `Q6_K`. This is the binding set: `qwen35_bridge_init` refuses a pack at startup unless every layer pair passes `kernels::native_expert_supported` (`src/core/qwen35_bridge.cpp:1848-1855`). See "expert_formats roles" below. |
| `context_max` | int | CLI: the `--max-context` argv value (default 4096) - what this invocation would check requests against. serve: the live `llama_n_ctx` (identical to the `READY <ctx>` value). A request whose prompt exceeds it is answered `ERR` and the process keeps serving. |
| `logits_read` | `true` | This task's `SCORE` command (see below); no other logits export exists. |
| `state_export` | `false` | No export/import of session state (KV + GDN + ...). `SAVE`/`RESTORE` are answered `ERR` (serve protocol doc, "Not implemented"). |
| `disk_kv` | `false` | No disk KV; nothing survives the process. |
| `vision` | `false` | The Qwen35 serve path implements no vision (`GENI` is answered `ERR`); the native engine's `--vision` is a different engine and not reachable from `--qwen35-model`. |

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
SCORE <id,id,...> [topk=N]
```

- `<id,id,...>`: comma-separated decimal token ids of the prompt (the same grammar `GEN` uses; spaces inside
  the list are rejected). The id list may appear before or after the key; exactly one id list per command.
- `topk=N`: optional, default **5**; `N` must be a positive decimal integer (a repeated key takes the last
  value). Any other key is refused with `ERR` - a silently ignored `topk` typo would answer a different
  question than the one asked. (Unlike `GEN`, which skips the conversation-cache keys, `SCORE` has no cache
  keys to skip.)
- Bad syntax, an out-of-vocabulary id, or a prompt longer than the context is answered `ERR ...` and the
  engine keeps serving - the same contract as `GEN`.

### Semantics (implemented exactly as listed)

1. The prompt is read through the **same** prefill path as `GEN` (`run_prefill` in `qwen35_serve.cpp`):
   `n_ubatch`-sized chunks, one `PP` line per chunk, then the prompt -> decode transition
   (`qwen35_bridge_finish_prompt`). A `STOP` during the read cancels it (see below). **No token is
   generated**: the sampler is not created nor advanced.
2. The **last position's logits** are taken (`llama_get_logits_ith(ctx, -1)`; the context runs with
   `n_outputs_max = 1`). The row is the distribution for the token that would follow the last prompt token.
3. One line is printed:

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
   - malformed line / two id lists / unknown key / non-positive `topk` -> `ERR bad request: ...`;
   - an id outside `[0, n_vocab)` -> `ERR a token id is outside the vocabulary`;
   - more prompt tokens than the context -> `ERR prompt (...) exceeds the context (...)`;
   - prefill failure or an unreturned prompt workspace -> `ERR prefill failed` / `ERR the prompt workspace
     loan was not returned`;
   - a NaN anywhere in the head logits -> `ERR score: the head logits are not all finite` (the same
     finite-logits invariant `GEN` enforces as `DONE ... nonfinite`; `SCORE` does not emit a partial answer).

### Level of approximation

None: the answer is the exact full-vocabulary log-softmax of the last row (no top-k denominator, no
truncation of the sum). The only numerical caveat is the `%.4f` print precision and the row's own fp32
arithmetic; the NaN guard rejects non-finite distributions instead of printing them.

## Not implemented (fields that are `false`, and why)

| Not implemented | Field / behavior | Why |
|---|---|---|
| cross-request prefix reuse | `prefix_cache=false` | every prompt is read from token 0; correctness-first phase-1 contract (documented in `qwen35_llama_serve_protocol.md`) |
| continuous batching | `batching=false` | no `INFO batch_slots`; requests are serialized one at a time; `BGEN` -> `ERR` |
| session state export/import | `state_export=false` | no KV/GDN snapshot export; `SAVE`/`RESTORE` -> `ERR` |
| disk KV | `disk_kv=false` | nothing persists across processes |
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

## Build record (P2-1)

- Clean full build (build directory deleted, `build.ps1` configure + build): **exit 0, 384/384 steps**,
  ~527 s, MSVC vcvars64 + CUDA 12.8.1, `CMAKE_CUDA_ARCHITECTURES=89`, `-DSTRATA_QWEN35=ON -DSTRATA_NATIVE_EXPERTS=ON
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON` (the standalone backend's usual flags).
- Artifact: `backends/qwen35_llama/build/strata.exe`, **139,226,624 B**,
  `sha256=66a48ad1366a122ea166ce557e23db85d62332bd62a2b017c2117e1c7cb27745`.
- Functional verification on GPU/cancellation/quality is **P2-2**, not this task (P2-1 is 0 GPU: no exe run).
