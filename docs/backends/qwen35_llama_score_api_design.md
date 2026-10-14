# qwen35_llama score API - `/v1/score`, `/v1/capabilities` and a minimal `/v1/decisions` (P2-3 subset)

Phase-2 backend-interface work, task **P2-3** (branch `phase2/backend-interface`). This adds a **thin HTTP
service of our own** that drives one existing serve-protocol engine process (`CAPS` / `SCORE`) and exposes three
endpoints. It is a **documented subset, not a protocol-compatible server**: nothing here claims SGLang
compatibility, concurrency, streaming or batching.

Files (all new; official files, the engine and every frozen artifact are untouched):

| File | What it is |
|---|---|
| `backends/qwen35_llama/tools/score_api.py` | the service (stdlib only) |
| `backends/qwen35_llama/tools/stub_engine.py` | **test-only** stub of the serve line protocol (used by the unit tests; never by the service) |
| `backends/qwen35_llama/tools/test_score_api.py` | 0-GPU unit tests (stub engine over real pipes + real HTTP) |
| `docs/backends/qwen35_llama_score_api_design.md` | this document |

Status of verification: **0 GPU, no engine run, no lease.** Everything measured below was measured with the stub
engine on CPU. Running the service against the real engine (real model, GPU lease, neutral cwd) is **P2-4** and
has not been done yet.

## 1. Running the service

```text
# spawn the engine from the official server config (exe/cwd/args/env), then listen
python backends/qwen35_llama/tools/score_api.py --start-engine \
    --config backends/qwen35_llama/server-config.json --host 127.0.0.1 --port 8097

# or pass the engine argv directly (service options must come BEFORE --argv)
python backends/qwen35_llama/tools/score_api.py --start-engine --port 8097 \
    --engine backends/qwen35_llama/build/strata.exe \
    --argv --qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models/pack \
           --serve --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096

# attach to an engine someone else started (named pipes / FIFOs; used for tests)
python backends/qwen35_llama/tools/score_api.py --attach-stdin <pipe-in> --attach-stdout <pipe-out>
```

Notes, each of them a refusal rather than a silent guess:

- `--argv` is `argparse.REMAINDER`: **service options must be placed before it**; everything after goes to the
  engine. Without `--config`/`--argv` there is nothing to start; an engine argv without `--serve` is refused
  with a message (the service only drives the resident serve protocol).
- `--start-engine` waits for the engine's `READY` line (`--start-timeout`, default 600 s) and refuses to listen
  before it. `--attach-stdin/--attach-stdout` send a `CAPS` handshake instead, so attaching to a live engine
  that is already past `READY` works. Both attach flags are needed together.
- The engine's environment is the service's environment plus the config's `env` plus `--env KEY=VALUE`; the
  service does not filter (or add) any `STRATA_*`/`GGML_*` gate. Engine stderr goes to the service's stderr
  unless `--engine-log PATH` is given. Nothing here sanitizes the environment - that stays the caller's job
  (P2-4's neutral-run discipline is unchanged).
- String `prompt` input needs a tokenizer: `--tokenizer-model <gguf>` loads the repo's own
  `tools/strata_tokenizer.py` (`Tokenizer.from_gguf`, byte-level BPE). Without that flag - or if the file does
  not load - a `prompt` request is refused `501` with the reason, and `prompt_ids` is the working path.
- On a clean shutdown (Ctrl-C, or the process being terminated) the service sends `QUIT` to an engine it
  spawned. The bind address defaults to `127.0.0.1`; **there is no authentication and no TLS in this subset** -
  do not expose it beyond localhost.

## 2. Endpoints

All responses are `application/json; charset=utf-8`.

### `GET /v1/capabilities`

One `CAPS` round-trip to the engine; the engine's JSON is passed through field-for-field and the service adds
exactly two fields of its own:

```json
{"protocol": "strata-serve-v1", "continuous_requests": true, "cancel": true, "state_isolation": true,
 "prefix_cache": false, "batching": false,
 "mtp": {"available": false, "enabled": false, "default": false, "k": null},
 "expert_formats": ["Q4_0", "Q5_0", "IQ3_S"], "context_max": 4096, "logits_read": true,
 "state_export": false, "disk_kv": false, "vision": false,
 "service": "score_api",
 "engine_sha256": "ec5460cac4f5f3f04696dcae2b9a5b645c51b1779fdb3670666c6eeb1f46b1e8"}
```

(the `expert_formats` array above is the **stub's** three-format placeholder from the sample capture; the real
engine answers its own 17-format whitelist, see
`docs/backends/qwen35_llama_capabilities_and_score.md`). `engine_sha256` is the SHA-256 of the engine file:
in start mode it is the executable the service spawned; in attach mode it is the `--engine` path **if one was
given**, otherwise `null`. The SHA is computed once and cached. Semantics of every passthrough field are the
P2-1/P2-2 contract; this endpoint adds nothing to them and never invents a value.

### `POST /v1/score`

Request (exactly one of `prompt_ids` / `prompt`):

| Field | Type | Meaning |
|---|---|---|
| `prompt_ids` | list of int | the prompt as token ids (the same grammar `GEN`/`SCORE` use) |
| `prompt` | string | text prompt; needs `--tokenizer-model`, otherwise refused `501` |
| `topk` | int > 0 | how many distribution entries to return; default 5 (the engine's own default) |

The service sends exactly one engine command, `SCORE <ids> topk=<topk>`, waits for `LOGPROBS` + `DONE ... score`
(ignoring the engine's `PP` progress lines), and answers:

```json
{"logprobs": [{"id": 0, "logprob": -0.9328}, {"id": 1, "logprob": -1.4328}], "pos": 0,
 "full_vocab_softmax": true, "generated_tokens": 0,
 "usage": {"prompt_tokens": 1, "prompt_ms": 1.0, "prompt_read": 1}}
```

- `logprobs`: the engine's `LOGPROBS` line, unchanged: the `k` largest logits of the last prompt position, and
  `logprob = logit - logsumexp(all logits)` over the **full vocabulary** (`%.4f` as printed by the engine; no
  top-k renormalization). `full_vocab_softmax: true` states that denominator, which P2-1/P2-2 verified.
- `pos`: 0-based position of the last prompt token (`prompt_tokens - 1`).
- `generated_tokens: 0`: hard state - SCORE generates no token (P2-2: no `T` line between `PP` and
  `DONE`, `produced=0`, `finish=score`).
- `usage`: the engine's `DONE` envelope (`prompt_tokens`, `prompt_ms` = prefill milliseconds, `prompt_read`;
  `prompt_read < prompt_tokens` only when a cancel ended the read part-way - this service never sends `STOP`
  except after a timeout, see 5).
- No service-side cap on `topk`: it is passed to the engine, whose `k = min(topk, n_vocab)` still applies.
- An out-of-vocabulary id or an over-context prompt arrives as the engine's own `ERR` and is answered `422`
  with the message verbatim, `"error_source": "engine"` (sample capture: `{"error": "a token id is outside the
  vocabulary", "error_source": "engine"}`).

### `POST /v1/decisions` (minimal subset)

Request:

| Field | Type | Meaning |
|---|---|---|
| `kind` | string | `"choice"`, `"score"` or `"yes_no"`; anything else is refused `422` (no business kinds, no thresholds anywhere) |
| `prompt_ids` | list of int | the prompt as token ids (there is no string `prompt` on this endpoint) |
| `candidates` | list of lists of int | the token ids to score, in the caller's order (the caller owns labels, rendering and prompt construction) |
| `topk` | int > 0 | optional; must be at least `len(candidates)` |

The service sends **one** `SCORE` with `topk = max(5, len(candidates), request.topk)` so every candidate shares
the same full-vocabulary denominator, then answers (values from the sample capture):

```json
{"kind": "choice",
 "candidates": [{"token_ids": [0], "logprob": -0.9328, "probability": 0.393450505...},
                {"token_ids": [1], "logprob": -1.4328, "probability": 0.238639794...},
                {"token_ids": [2], "logprob": -1.9328, "probability": 0.144742351...}],
 "pos": 1, "full_vocab_softmax": true, "generated_tokens": 0,
 "usage": {"prompt_tokens": 2, "prompt_ms": 1.0, "prompt_read": 2},
 "label_mass_omitted": "the candidate set is not a closed, mutually exclusive cover of the next-token
   distribution (total candidate probability 0.776833, tolerance 0.001); the per-candidate probabilities are
   returned un-renormalized"}
```

Validation (all refusals name the offending index/id; nothing is silently dropped):

- every candidate must be **exactly one token id** ("single-token uniqueness"); a multi-token candidate is
  refused `422` with an explanation (`candidates[0] has 2 tokens; multi-token candidates are not implemented in
  this subset - pass exactly one token id per candidate`). Token-sequence-to-id ambiguity cannot even arise for
  a single id, and there is no tokenizer in this path to re-encode text against;
- candidate ids must be pairwise **distinct** (mutual exclusivity); duplicates are refused `422`;
- `choice`/`score` need at least two candidates; `yes_no` needs exactly two (the caller's yes/no token pair -
  the service never hardcodes a word or language);
- every candidate must appear in the engine's returned top-k, otherwise `422` with the reason (the `SCORE`
  command has no explicit-id scoring; raise `topk` if the ids are plausible). This is the one **hard engine
  limitation** of the subset and it is a refusal, never an invented value.

Per candidate the response carries `logprob` (full-vocab log-softmax) and `probability = exp(logprob)` -
**un-renormalized**; renormalizing over the labels is left to the caller (a one-line sum).

#### `label_mass`

`label_mass` is emitted only when it is a **verified quantity** at this layer:

1. the candidates are mutually exclusive (enforced: distinct single-token ids), and
2. they **cover the full next-token distribution**: `sum(candidate probabilities) = 1` within a tolerance of
   `1e-3` (the engine prints logprobs at `%.4f`, so a full cover can only be read back within that precision).

When both hold, `label_mass` is that sum (sample capture, stub vocabulary of 4: `1.0000386724460892`). When they
do not - the normal case for real decisions - the field is **omitted** and `label_mass_omitted` explains why
with the computed partial mass. The service cannot verify that a partial candidate list is the complete set of
a decision, and it does not pretend to: a partial mass is conveyed as text in the omission reason, not as a
field with `label_mass`'s name.

The three kinds share this one scoring core. `choice` and `score` are accepted but not differentiated beyond
validation: the distinction in SGLang comes from server-side rendering and label aggregation (argmax option,
probability-weighted level), neither of which exists here. `yes_no` is the same two-candidate scoring with a
fixed arity.

## 3. Errors

| Status | `error_source` | When |
|---|---|---|
| 400 | service | body not a JSON object / invalid JSON / wrong field type / unknown field / both or neither of `prompt_ids`,`prompt` / empty or negative ids / boolean ids |
| 413 | service | body larger than 1 MiB |
| 422 | service | semantic refusals: unknown `kind`; multi-token or duplicate candidates; `yes_no` arity; `topk` smaller than the candidate count; candidate outside the returned top-k |
| 422 | engine | the engine answered `ERR <message>` - the message is passed through **verbatim** |
| 501 | service | string `prompt` without a usable tokenizer |
| 502 | service | the engine link is dead (process exited, pipes closed) or violated the line protocol; after a failed timeout drain the service stays dead until restarted |
| 503 | service | a request is already in flight (single engine, no queue) |
| 504 | service | the engine did not answer within `--timeout` (default 600 s) |

Failures are never silent: there is no path that answers an empty result with status 200. Unknown fields are
refused (`400`) rather than ignored, so an SGLang request body fails loudly instead of being half-understood.

## 4. Concurrency model: 503, never a queue

The service drives **one** engine and keeps **one** request in flight. A second request that arrives while one
is running (including `/v1/capabilities`) is answered `503` with `Retry-After: 1`; it is not queued, so a
caller always knows when nothing was done on its behalf. `POST` handlers run on `ThreadingHTTPServer` threads
so connections are accepted concurrently, but engine access is serialized by a non-blocking lock. This is the
processing model SGLang's `batching=false` analog at our layer; continuous batching stays on the engine's
unimplemented list.

## 5. Process lifecycle and timeouts

- Start: spawn, then wait for `READY` (`--start-timeout`). Attach: `CAPS` handshake.
- Request: write one command; read lines until `LOGPROBS` + `DONE ... score` (or `ERR`). `PP` lines are
  progress and are ignored.
- Timeout (`--timeout`): answer `504`, then send `STOP` and drain to the engine's next `DONE`/`ERR` (bounded by
  an internal 10 s drain timeout) so the next request starts synchronized. If the drain also fails, the link is
  marked dead: every later request answers `502` until the service is restarted. This is deliberately
  conservative - a wedged engine is never presented as healthy.
- A `T` line during a `SCORE` (the engine contract forbids generating a token) or a second `LOGPROBS` is a
  protocol violation and is answered `502`, not passed on.

## 6. State and latency semantics (P2-2 measured facts, reused as-is)

The service adds **no caching and no state**: every request is one engine `SCORE`, and the engine's own
contract (P2-1 implemented, P2-2 verified on GPU) is:

- **Read-only, semantics (b)**: `SCORE` reads the prompt through the same prefill path as `GEN` and generates
  nothing; it resets the session before and after, so the next request - `GEN` or `SCORE` - starts fresh and
  `reused` stays 0. A `/v1/score` or `/v1/decisions` call therefore never changes what a later request sees.
- P2-2 N1 raw transcript (source: `docs/backends/qwen35_llama_capabilities_and_score.md`, §GPU verification,
  2026-10-14 23:21-23:22 +08:00, exe `sha256=66a48ad1...`): `SCORE` on a 3869-token prompt answered
  `LOGPROBS 3868 5 18:-0.8937 15:-1.2617 17:-1.2855 16:-3.5211 19:-8.3055` then
  `DONE 0 3869 4152.4 0.0 score ...` with **no `T` line**, `pos = 3868 = 3869 - 1`; the following `GEN 16` ran
  normally (2 tokens, EOS `248046`, `finish=stop`, `reused=0`), and after an out-of-vocabulary `SCORE` was
  answered `ERR a token id is outside the vocabulary` the service kept serving. Child exit 0; 18.11 s wall for
  all five staged lines.
- Latency expectation at this layer: one prompt read per request (no cross-request prefix reuse), engine
  prefill measured in the same transcript as 4152.4 / 2717 / 2760 ms for the 3869-token prompt on that run.
  This service adds one process hop and JSON encoding on top; it has **not** been timed against the real engine
  (P2-4).

## 7. Differences from SGLang (explicit subset boundaries)

Reference: SGLang's native API and decision-model documentation (sgl-project/sglang, docs fetched 2026-10-14).
Both SGLang endpoints below accept far more than this subset; none of the differences are replicated, and no
field is accepted with a different meaning than documented here.

### `/v1/score`

| SGLang | This subset |
|---|---|
| `query` + `items` (text or token ids), items scored in a batch | one `prompt_ids` per request; no `query`/`items`, no batch |
| `label_token_ids` (one list per item) - arbitrary ids are scored | no explicit-id scoring; only the distribution's top-k is returned (a candidate outside it is `422`) |
| `apply_softmax` (renormalize over labels), `item_first`, `temperature`, `return_token_logprobs` | none of these fields; `logprob` is only ever the full-vocabulary log-softmax, and unknown fields are refused |
| `scores`: linear probabilities in label order | `logprobs`: `{id, logprob}` pairs of the top-k, in the engine's descending order; no probabilities in this endpoint |
| cross-encoder/rerank semantics, `model` selection | none; one engine, no model routing |

### `/v1/decisions`

| SGLang | This subset |
|---|---|
| `input` text + typed `questions` with `options`/`levels`/descriptions; server renders the chat template, chooses labels (A-Z, 0-9, yes/no), checks each label is one token at the answer position | the caller builds the prompt and passes `prompt_ids` + candidate token ids; no rendering, no labels, no questions text, no reasoning-block handling |
| requests are answered in one server-side batch of prefills; questions answered independently | one `kind` + one candidate list per request; no batch, no multi-question |
| `probabilities` renormalized over the labels (sum 1), `choice` (top option) and `score` (weighted mean level) aggregates | per-candidate `logprob` + un-renormalized `probability`; no aggregate fields |
| `label_mass` returned for text models whenever labels are scored | `label_mass` only when the candidate set is a **verified closed cover** (`sum = 1` within `1e-3`); otherwise omitted with the reason |
| `temperature` divides label logits; `prompt_format_version`, `return_prompt_token_ids`, `images`, `model` | none of these fields |
| errors 400 with per-field reasons; runs alongside generation traffic on the same server | 4xx/422 refusals as in §3; a single serialized engine, no chat endpoint at all |

**There is no `POST /v1/chat/completions` here**: ordinary chat stays the official server's job
(`serve/server.py`), unchanged.

## 8. Not implemented (and where it could go)

| Not implemented | Today's behavior | Possible later step |
|---|---|---|
| real-engine verification of this service | never run against the engine (0 GPU by design) | P2-4: GPU end-to-end on the real model under the usual lease/neutral-cwd discipline |
| explicit-id scoring (score ids outside the top-k) | refused `422` | engine-side `SCORE` extension (new key, P2-1 protocol precedent); removes the top-k candidate limitation |
| server-side prompt rendering / labels / question descriptions | caller owns prompt construction (`prompt_ids`) | a rendering layer *above* this service; the service stays content-free |
| streaming | none; one JSON response per request | incompatible with a single score answer; only relevant to future generation endpoints |
| batching / concurrent requests | `503` while busy | engine-side continuous batching (P2-1 `batching=false`), then a scheduler above it |
| session reuse / state export | every request resets; `reused=0` | engine `state_export` work (future phases); the service would gain a session key |
| auth, TLS, multi-engine routing | localhost bind only | deployment concern, deliberately out of scope for the subset |

## 9. Unit tests (0 GPU)

```text
cd <repo root>
python backends/qwen35_llama/tools/test_score_api.py -v
```

Result of the recorded run: **19 tests, OK** (`Ran 19 tests in 14.300s`, Python 3.11.9, Windows; raw log:
research repo `results/opencode-execution/strata-unified-p2-3-score-api/unit-tests.log`). Each test starts the
test-only stub (`stub_engine.py`) as a real child process over pipes and an in-process HTTP server on
`127.0.0.1:0`; no GPU, no lease, no engine, no writes outside the process pipes/sockets. Coverage:

- `/v1/score`: normal fields (`generated_tokens=0`, `pos`, `full_vocab_softmax`, logprobs order), `topk`
  default and override;
- `/v1/capabilities`: passthrough equality of engine fields plus `service`/`engine_sha256` (SHA of the declared
  engine file, verified against `hashlib`);
- engine `ERR` -> `422` + verbatim text + `error_source=engine`;
- `prompt` string without a tokenizer -> `501` + explanation, and with an unloadable `--tokenizer-model` the
  same refusal with the load error;
- bad bodies -> `400` (invalid JSON, both/neither of `prompt_ids`/`prompt`, unknown field, empty ids, boolean
  ids); unknown path -> `404`;
- `/v1/decisions`: all three kinds' normal paths, multi-token candidate refusal, duplicate-candidate refusal,
  unknown kind refusal (no business kinds), candidate outside top-k refusal, `label_mass` omitted on a partial
  set and present on a verified full cover (stub vocabulary 4, mass `1.00004`);
- concurrency: a second request during an in-flight score gets `503` + `Retry-After: 1`, and the first still
  completes `200`;
- timeout: a stub that never answers -> `504` within ~1 s (test budget), next request `502` (dead link);
- CLI: `--start-engine` with a real child process, `READY` + listening lines parsed, live `/v1/capabilities`
  and `/v1/score` against it, clean terminate.

The raw HTTP samples quoted in §2 are in `sample-responses.jsonl` (same directory; captured by
`p2-3-capture.py`).

## 10. Verification boundary of this task

- 0 GPU, no engine binary run, no lease, no model touched: every number above was produced by the stub engine
  on CPU, except the P2-2 transcript facts which are quoted with their source.
- Official files and engine source: zero edits (this task adds files only; the commit shows additions in
  `backends/qwen35_llama/tools/` and `docs/backends/`).
- No performance claim of any kind is made for the real engine; P2-4 owns that.
