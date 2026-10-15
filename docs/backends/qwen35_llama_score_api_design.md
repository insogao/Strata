# qwen35_llama score API - `/v1/score`, `/v1/capabilities` and a minimal `/v1/decisions` (P2-3 subset; P2R-1 payload semantics)

Phase-2 backend-interface work, task **P2-3** (branch `phase2/backend-interface`), extended by **P2R-1**
(branch `phase2r/review-gaps`). This adds a **thin HTTP service of our own** that drives one existing
serve-protocol engine process (`CAPS` / `SCORE`) and exposes three endpoints. It is a **documented subset,
not a protocol-compatible server**: nothing here claims SGLang compatibility, concurrency, streaming or
batching.

P2R-1 changes (2026-10-15): `/v1/score` gains an optional `candidates` field (explicit token-id scoring,
with no top-k dependency); `/v1/decisions` is re-specified around **three named probability scopes** - the
full-vocabulary probability, the conditional probability within the candidate set, and an explicit statement
that **no calibrated business confidence is provided** - and now **always returns `label_mass`** (the old
`label_mass_omitted` field and the "candidate outside the returned top-k -> 422" refusal are **deleted**).
The engine side is the P2R-1 `SCORE ... [cand=...]` / `topk=0` extension
(`docs/backends/qwen35_llama_capabilities_and_score.md`); GPU verification of the new paths was
**run 2026-10-15 by P2R-2 and passed** (evidence:
`results/opencode-execution/strata-unified-p2r2-candidate-gpu/`; write-up: "P2R-2: GPU validation of the
P2R-1 additions" in `docs/backends/qwen35_llama_score_api_e2e.md`). The payload-difference checklist vs
the P2-3/P2-4 payload is at the end of §2.

Files (all new; official files, the engine and every frozen artifact are untouched):

| File | What it is |
|---|---|
| `backends/qwen35_llama/tools/score_api.py` | the service (stdlib only) |
| `backends/qwen35_llama/tools/stub_engine.py` | **test-only** stub of the serve line protocol (used by the unit tests; never by the service) |
| `backends/qwen35_llama/tools/test_score_api.py` | 0-GPU unit tests (stub engine over real pipes + real HTTP) |
| `docs/backends/qwen35_llama_score_api_design.md` | this document |

Status of verification: the P2-3 baseline below was measured with the stub engine on CPU (0 GPU, no
engine run, no lease). Real-engine verification then happened in two steps: **P2-4** ran the thin service
against the real engine on GPU (`docs/backends/qwen35_llama_score_api_e2e.md`) with the pre-P2R-1 payload,
and **P2R-2 (2026-10-15)** validated the P2R-1 additions (explicit candidates, the always-present
`label_mass` payload, byte-compat of `cand=`-free requests) on the live engine - it **passed**; each
numeric claim below that originates from the stub capture remains labelled as such.

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
| `topk` | int >= 0 | how many distribution entries to return; default 5 (the engine's own default). `topk=0` is legal only together with `candidates` (and then no distribution is requested) |
| `candidates` | list of int | **P2R-1, optional**: explicit token ids to score (order preserved, `<= 2048`, pairwise distinct); orthogonal to `topk` |

The service sends exactly one engine command - `SCORE <ids> topk=<topk>` or, with candidates,
`SCORE <ids> topk=<topk> cand=<ids>` - waits for `LOGPROBS` + `CANDLOGPROBS` (only the lines that were
requested) + `DONE ... score` (ignoring the engine's `PP` progress lines), and answers:

```json
{"logprobs": [{"id": 0, "logprob": -0.9328}, {"id": 1, "logprob": -1.4328}], "pos": 0,
 "full_vocab_softmax": true, "generated_tokens": 0,
 "usage": {"prompt_tokens": 1, "prompt_ms": 1.0, "prompt_read": 1}}
```

- `logprobs`: the engine's `LOGPROBS` line, unchanged: the `k` largest logits of the last prompt position, and
  `logprob = logit - logsumexp(all logits)` over the **full vocabulary** (`%.4f` as printed by the engine; no
  top-k renormalization). `full_vocab_softmax: true` states that denominator, which P2-1/P2-2 verified.
  Empty `[]` when `topk=0` was requested (nothing was asked for; never an invented value).
- `candidates` (**P2R-1**, only when the request carried `candidates`): the engine's `CANDLOGPROBS` line as
  `[{"id":..., "logprob":...}, ...]`, **in the request's order**, values from the same last-position row and
  the same full-vocab lse as `logprobs` (the engine contract: one prefill, no second forward). Ids outside
  the distribution's top-k are answered like any other id.
- `pos`: 0-based position of the last prompt token (`prompt_tokens - 1`).
- `generated_tokens: 0`: hard state - SCORE generates no token (P2-2: no `T` line between `PP` and
  `DONE`, `produced=0`, `finish=score`).
- `usage`: the engine's `DONE` envelope (`prompt_tokens`, `prompt_ms` = prefill milliseconds, `prompt_read`;
  `prompt_read < prompt_tokens` only when a cancel ended the read part-way - this service never sends `STOP`
  except after a timeout, see 5).
- No service-side cap on `topk`: it is passed to the engine, whose `k = min(topk, n_vocab)` still applies.
- An out-of-vocabulary id (prompt or candidate) or an over-context prompt arrives as the engine's own `ERR`
  and is answered `422` with the message verbatim, `"error_source": "engine"` (sample capture: `{"error": "a
  token id is outside the vocabulary", "error_source": "engine"}`).
- Candidate validation (all refusals, never a silent fix): not a list / empty / non-integer / boolean /
  negative -> `400`; more than 2048 ids -> `400`; duplicated ids -> `400`; `topk=0` without `candidates` ->
  `422` (the engine has no answer to give); `candidates` with `topk>=1` is fine (both answers are returned).
- A malformed `CANDLOGPROBS` from the engine (short, bad pairs, declared-vs-actual count mismatch, appearing
  twice, appearing although no `candidates` were requested, or ids not in request order) is an
  `EngineProtocolError` -> `502`; a `LOGPROBS` line on a `topk=0` request is a `502` too. Lines are parsed
  strictly, never best-effort.

### `POST /v1/decisions` (minimal subset; P2R-1 payload)

Request:

| Field | Type | Meaning |
|---|---|---|
| `kind` | string | `"choice"`, `"score"` or `"yes_no"`; anything else is refused `422` (no business kinds, no thresholds anywhere) |
| `prompt_ids` | list of int | the prompt as token ids (there is no string `prompt` on this endpoint) |
| `candidates` | list of lists of int | the token ids to score, in the caller's order (the caller owns labels, rendering and prompt construction) |
| `topk` | int > 0 | optional, **deprecated for this endpoint**: still validated when present, but it no longer constrains or changes anything - scoring is explicit-id now (the old "topk must cover all candidates" refusal is deleted) |

The service sends **one** `SCORE <ids> topk=0 cand=<ids>` - one prefill, **no top-k dependency**, so a
candidate id is answered whatever its rank in the distribution - then answers (values from the stub capture;
`...` elides digits):

```json
{"kind": "choice",
 "candidates": [{"token_ids": [0], "logprob": -0.9328, "full_vocab_probability": 0.3934505...,
                 "conditional_probability": 0.50647...,
                 "probability": 0.3934505...},
                {"token_ids": [1], "logprob": -1.4328, "full_vocab_probability": 0.2386397...,
                 "conditional_probability": 0.30720...,
                 "probability": 0.2386397...},
                {"token_ids": [2], "logprob": -1.9328, "full_vocab_probability": 0.1447423...,
                 "conditional_probability": 0.186323...,
                 "probability": 0.1447423...}],
 "pos": 1, "full_vocab_softmax": true, "generated_tokens": 0,
 "usage": {"prompt_tokens": 2, "prompt_ms": 1.0, "prompt_read": 2},
 "label_mass": 0.7768326...,
 "label_mass_covers_distribution": false,
 "label_mass_tolerance": 0.001,
 "calibrated": false,
 "probability_scopes": {
   "full_vocab_probability": "exp(logprob) under the full-vocabulary softmax: comparable across requests; the sum over this candidate set is at most 1.",
   "conditional_probability": "full_vocab_probability / label_mass: meaningful only within this candidate set; not comparable across different candidate sets.",
   "business_confidence": "not provided: no calibration exists, so these numbers are not confidences; e.g. a label_mass of 0.31 means about 69% of the distribution mass lies outside the candidate set."
 }}
```

Validation (all refusals name the offending index/id; nothing is silently dropped):

- every candidate must be **exactly one token id** ("single-token uniqueness"); a multi-token candidate is
  refused `422` with an explanation (`candidates[0] has 2 tokens; multi-token candidates are not implemented in
  this subset - pass exactly one token id per candidate`). Token-sequence-to-id ambiguity cannot even arise for
  a single id, and there is no tokenizer in this path to re-encode text against;
- candidate ids must be pairwise **distinct** (mutual exclusivity); duplicates are refused `422`;
- `choice`/`score` need at least two candidates; `yes_no` needs exactly two (the caller's yes/no token pair -
  the service never hardcodes a word or language);
- more than 2048 candidates are refused `422` (the engine's explicit-candidate cap);
- an out-of-vocabulary id is the engine's own `ERR`, mapped to `422` with `"error_source": "engine"`;
- there is **no top-k membership requirement anymore** (P2R-1): the old refusal "candidate id(s) ... are outside
  the engine's returned top-N logprobs" is deleted, together with the `topk >= len(candidates)` check.

#### The three probability scopes (P2R-1, P0-2)

The payload distinguishes exactly three quantities and never presents one as another:

1. **Full-vocabulary probability** - `full_vocab_probability = exp(logprob)`, where `logprob` is the
   log-softmax over **all** `n_vocab` logits of the last prompt position (the same row/lse the engine's
   `LOGPROBS` line uses; one prefill). Group property: the sum over any candidate set is `<= 1`, and values
   are **comparable across requests** (same denominator, no renormalization).
2. **Candidate-set conditional probability** - `conditional_probability = full_vocab_probability /
   label_mass`. It answers "if the next token is one of these candidates, which one?" - meaningful **only
   within this candidate set** and **not comparable across different candidate sets** (a different set gets a
   different denominator). `null` when `label_mass` is 0 (nothing to condition on).
3. **Business confidence** - **not provided**. No calibration exists anywhere in this chain, so no field,
   threshold or verdict here is a confidence. `calibrated: false` and the `probability_scopes` text state
   this explicitly; a low `label_mass` (e.g. ~0.31 means about 69% of the mass is outside the candidate set)
   never triggers any "high confidence" field, and no confidence/threshold field exists to trigger.

`probability` is kept as a **deprecated alias** of `full_vocab_probability` (identical value; downstream
compatibility) and will be documented as such wherever the payload is described.

#### `label_mass`

`label_mass` is **always returned** (P2R-1; the old `label_mass_omitted` is deleted): it is the sum of the
candidates' `full_vocab_probability`, i.e. the share of the full next-token distribution the candidate set
covers - a **reported measurement**, not a gate. `label_mass_covers_distribution` is the explicit check
`|label_mass - 1| <= label_mass_tolerance` (`label_mass_tolerance = 1e-3`; the engine prints logprobs at
`%.4f`, so a full cover can only be read back within that precision), and `label_mass_tolerance` carries the
number itself. `label_mass = 1` within tolerance means the candidates are a closed cover; any other value
simply says how much mass lies elsewhere (the residual is not distributed to the candidates - the reported
per-candidate numbers are always un-renormalized full-vocab values).

The three kinds share this one scoring core. `choice` and `score` are accepted but not differentiated beyond
validation: the distinction in SGLang comes from server-side rendering and label aggregation (argmax option,
probability-weighted level), neither of which exists here. `yes_no` is the same two-candidate scoring with a
fixed arity.

#### Differences vs the P2-3/P2-4 decisions payload (what changed, field by field)

| Field | P2-3/P2-4 payload | P2R-1 payload |
|---|---|---|
| engine command | `SCORE <ids> topk=max(5, len(candidates), topk)` (candidates had to fall in the top-k) | `SCORE <ids> topk=0 cand=<ids>` (one prefill, explicit ids) |
| per candidate `logprob` | full-vocab log-softmax | unchanged (same row/lse semantics) |
| per candidate `probability` | `exp(logprob)`, un-renormalized | unchanged value, now documented as **deprecated alias** of `full_vocab_probability` |
| per candidate `full_vocab_probability` / `conditional_probability` | did not exist | new; conditional is `full/mass`, `null` at mass 0 |
| top-level `label_mass` | only when `abs(mass-1) <= 1e-3` | **always** returned |
| top-level `label_mass_omitted` | text explaining omission | **deleted** |
| top-level `label_mass_covers_distribution` / `label_mass_tolerance` | did not exist | new |
| top-level `calibrated` / `probability_scopes` | did not exist | new (`false` / three scoped strings) |
| `kind` / `pos` / `full_vocab_softmax` / `generated_tokens` / `usage` | | unchanged |
| refusal "candidate outside the returned top-k" (422) | existed | **deleted** (explicit scoring has no top-k dependency) |
| `topk` request field | must be >= `len(candidates)` | accepted (positive int) for compatibility; no effect |
| numeric equality for the same candidate ids | - | `logprob` and `probability` are the same values (same engine row; P2-4's 4dp parity claim is not re-derived here, and its GPU capture predates this payload) |

## 3. Errors

| Status | `error_source` | When |
|---|---|---|
| 400 | service | body not a JSON object / invalid JSON / wrong field type / unknown field / both or neither of `prompt_ids`,`prompt` / empty or negative ids / boolean ids / `candidates` not a list / non-integer / negative / duplicated / more than 2048 |
| 413 | service | body larger than 1 MiB |
| 422 | service | semantic refusals: unknown `kind`; multi-token or duplicate candidates; `yes_no` arity; more than 2048 candidates (`/v1/decisions`); `topk=0` without `candidates` (`/v1/score`) |
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
| `label_token_ids` (one list per item) - arbitrary ids are scored | `candidates` (P2R-1) scores one explicit id list per request, in request order; arbitrary ids are answered whatever their top-k rank; still no batch across items |
| `apply_softmax` (renormalize over labels), `item_first`, `temperature`, `return_token_logprobs` | none of these fields; `logprob` is only ever the full-vocabulary log-softmax, and unknown fields are refused; candidate answers are never renormalized |
| `scores`: linear probabilities in label order | `logprobs`: `{id, logprob}` pairs of the top-k, in the engine's descending order (`[]` for `topk=0`); optional `candidates`: `{id, logprob}` pairs in the caller's order; no probabilities in this endpoint |
| cross-encoder/rerank semantics, `model` selection | none; one engine, no model routing |

### `/v1/decisions`

| SGLang | This subset |
|---|---|
| `input` text + typed `questions` with `options`/`levels`/descriptions; server renders the chat template, chooses labels (A-Z, 0-9, yes/no), checks each label is one token at the answer position | the caller builds the prompt and passes `prompt_ids` + candidate token ids; no rendering, no labels, no questions text, no reasoning-block handling |
| requests are answered in one server-side batch of prefills; questions answered independently | one `kind` + one candidate list per request; no batch, no multi-question |
| `probabilities` renormalized over the labels (sum 1), `choice` (top option) and `score` (weighted mean level) aggregates | per-candidate `logprob`, `full_vocab_probability`, candidate-set `conditional_probability` and the deprecated `probability` alias; no aggregate fields, no renormalized probability field |
| `label_mass` returned for text models whenever labels are scored | `label_mass` **always** returned, with `label_mass_covers_distribution` stating whether it is a closed cover (P2R-1); never omitted, never gated |
| `temperature` divides label logits; `prompt_format_version`, `return_prompt_token_ids`, `images`, `model` | none of these fields |
| errors 400 with per-field reasons; runs alongside generation traffic on the same server | 4xx/422 refusals as in §3; a single serialized engine, no chat endpoint at all |

**There is no `POST /v1/chat/completions` here**: ordinary chat stays the official server's job
(`serve/server.py`), unchanged.

## 8. Not implemented (and where it could go)

| Not implemented | Today's behavior | Possible later step |
|---|---|---|
| real-engine verification of this service | never run against the engine (0 GPU by design) | P2-4: GPU end-to-end on the real model under the usual lease/neutral-cwd discipline |
| GPU verification of the P2R-1 additions | **done 2026-10-15 (P2R-2), passed**: explicit candidates byte-identical to the same-process `LOGPROBS`; out-of-top-k ids finite `<= 0` and the candidate-set union mass `<= 1`; `SCORE` top-1 == greedy `GEN 1` first token on both prompts; live `/v1/decisions` payload per §2 (always-present `label_mass`, three scopes, `calibrated:false`, no confidence field); old-vs-new exe `LOGPROBS` byte-identical at equal positions. P2R-2's remaining caveat (no observable near-tie in the top-5) was then **quantified by P2R-2b (2026-10-15)**: top-2000 class comparison - same-class repeats byte-identical, cross-class flips real for gaps up to 0.0383/0.0371 (band 0.0599/0.0713) | evidence: `results/opencode-execution/strata-unified-p2r2-candidate-gpu/` and `.../strata-unified-p2r2b-near-tie/`; write-up in `docs/backends/qwen35_llama_score_api_e2e.md` (§6 and §P2R-2b) |
| server-side prompt rendering / labels / question descriptions | caller owns prompt construction (`prompt_ids`) | a rendering layer *above* this service; the service stays content-free |
| streaming | none; one JSON response per request | incompatible with a single score answer; only relevant to future generation endpoints |
| batching / concurrent requests | `503` while busy | engine-side continuous batching (P2-1 `batching=false`), then a scheduler above it |
| session reuse / state export | every request resets; `reused=0` | engine `state_export` work (future phases); the service would gain a session key |
| auth, TLS, multi-engine routing | localhost bind only | deployment concern, deliberately out of scope for the subset |

## 9. Unit tests (0 GPU)

```text
cd <repo root>
python backends/qwen35_llama/tools/test_score_api.py -v          # 34 tests (P2R-1)
python backends/qwen35_llama/tools/test_backend_adapters.py -v   # 13 tests, unchanged and green
```

P2R-1 recorded run (2026-10-15): **34 tests, OK** (`Ran 34 tests in 22.879s`, Python 3.11.9, Windows; raw log
`results/opencode-execution/strata-unified-p2r1-candidate-scoring/unit-tests-score-api.log` in the research
repo), plus the P2-7 suite **13 tests, OK** (same directory, `unit-tests-backend-adapters.log`). Each test
starts the test-only stub (`stub_engine.py`) as a real child process over pipes and an in-process HTTP server
on `127.0.0.1:0`; no GPU, no lease, no engine, no writes outside the process pipes/sockets. Coverage (the
P2-3 baseline plus P2R-1):

- `/v1/score`: normal fields (`generated_tokens=0`, `pos`, `full_vocab_softmax`, logprobs order), `topk`
  default and override;
- `/v1/score` `candidates` (P2R-1): order preservation and value identity with the `LOGPROBS` row (same
  stub distribution), a candidate outside the top-k answered (old 422 regression -> 200), `topk=0` +
  candidates -> `logprobs: []` and candidates only, `topk=0` without candidates -> `422`, candidates+topk
  coexist, duplicate -> `400`, 2049 ids -> `400`, out-of-vocabulary -> engine `ERR` -> `422`;
- `CANDLOGPROBS` protocol discipline (P2R-1, stub distortion modes): declared-vs-actual count mismatch ->
  `502`, unsolicited line -> `502`, line twice -> `502`, swapped id order -> `502`;
- `/v1/capabilities`: passthrough equality of engine fields plus `service`/`engine_sha256` (SHA of the declared
  engine file, verified against `hashlib`);
- engine `ERR` -> `422` + verbatim text + `error_source=engine` (prompt id and candidate id paths);
- `prompt` string without a tokenizer -> `501` + explanation, and with an unloadable `--tokenizer-model` the
  same refusal with the load error;
- bad bodies -> `400` (invalid JSON, both/neither of `prompt_ids`/`prompt`, unknown field, empty ids, boolean
  ids); unknown path -> `404`;
- `/v1/decisions`: all three kinds' normal paths, multi-token candidate refusal, duplicate-candidate refusal,
  unknown kind refusal (no business kinds); candidate outside top-k **scored** (P2R-1);
- `/v1/decisions` payload (P2R-1): `label_mass` always present and equal to the probability sum,
  `label_mass_covers_distribution` false on a partial set / true on the closed cover (stub vocabulary 4,
  conditional probabilities summing to 1), `label_mass_tolerance`, `calibrated: false`,
  `probability_scopes` with exactly the three texts, deprecated `probability` == `full_vocab_probability`,
  `conditional_probability` == `full/mass`, low mass (~0.27) -> no confidence/threshold field anywhere;
- concurrency: a second request during an in-flight score gets `503` + `Retry-After: 1`, and the first still
  completes `200`;
- timeout: a stub that never answers -> `504` within ~1 s (test budget), next request `502` (dead link);
- CLI: `--start-engine` with a real child process, `READY` + listening lines parsed, live `/v1/capabilities`
  and `/v1/score` against it, clean terminate.

Changed P2-3-era tests (P2R-1 semantic changes only, listed in the task evidence): "candidate outside top-k ->
422" became "candidate outside top-k -> 200 scored", and the two `label_mass_omitted` assertions became
"`label_mass` always present" assertions (with the new fields checked). The P2-3 sample responses
(`sample-responses.jsonl` in the P2-3 evidence directory) predate the P2R-1 decisions payload and are kept as
history, not as the current contract.

## 10. Verification boundary of this task

- P2-3 baseline: 0 GPU, no engine binary run, no lease, no model touched: every number above was produced by
  the stub engine on CPU, except the P2-2 transcript facts which are quoted with their source.
- P2R-1 (2026-10-15): edited `backends/qwen35_llama/src/program/qwen35_serve.cpp` (the `cand=`/`topk=0`
  SCORE extension and the two `score_*` CAPS fields), `tools/score_api.py`, `tools/backend_adapters.py`
  (signature passthrough), `tools/stub_engine.py`, `tools/test_score_api.py`, and the four docs. The engine
  change is **compile-verified only** (build.ps1 exit 0; new exe
  `sha256=d0321a98ab1908ef3c0f0d91b1b6e6c42758d79594222c6def425bf406a4bc09`, 139,257,344 B) - no engine was
  started, no model loaded, no GPU touched. The same-request byte-compat check against the old exe
  (`6dee0c66...`) and the GPU identity check were carried out by **P2R-2 (2026-10-15) and passed**
  (evidence: `results/opencode-execution/strata-unified-p2r2-candidate-gpu/`, client/p2r2-compare.json).
- Official files and frozen artifacts: zero edits (P2R-1 touches only the files listed above; no `main`
  merge, no vendor change).
- P2R-2b (2026-10-15, no source edits): one native `gpu-lock run` lease of the same new exe
  `d0321a98…`, static stdin, top-2000 class tables; evidence
  `results/opencode-execution/strata-unified-p2r2b-near-tie/`. It quantifies the near-threshold order
  sensitivity the P2R-2 caveat left open: same-class repeat lines byte-equal; cross-class max|Δ|
  5.99e-2 / 7.13e-2 with real adjacent-pair flips for gaps up to 0.0383 / 0.0371 and min flipped gap
  0.0000 (scoped to this host / this IQ3_S quant / this exe).
- No performance claim of any kind is made for the real engine; P2-4 owns that.
