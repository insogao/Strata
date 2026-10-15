# Framework backend adapters: `qwen35_llama` and `official_q36` (P2-7)

Phase-2 closing item **P2-7** (branch `phase2/backend-interface`). This adds a
**framework-level backend adapter** so the same interface drives two backends over their own
line protocols, with capability tables that reflect each backend honestly:

| | ① `qwen35_llama` | ② `official_q36` |
|---|---|---|
| what it is | the in-repo migrated backend (`docs/backends/qwen35_llama_serve_protocol.md`) | the official q36 engine's own `--serve` protocol, the one `serve/server.py` drives |
| client -> engine | `CAPS` / `SCORE` / `GEN` / `SAVE` / `LOAD` / `STATE` / `STOP` / `QUIT` | `GEN` / `STOP` / `QUIT` |
| engine -> client | `READY` / `PP` / `T` / `LOGPROBS` / `DONE` / `CAPS` / `SAVED` / `LOADED` / `STATE` / `ERR` | `READY` / `PP` / `T` / `DONE` |
| capability table | engine `CAPS` passthrough + framework fields | **framework-synthesized** (no `CAPS` command exists) |
| `logits_read` / `state_export` | `true` / `true` (P2-2 / P2-5+P2-6, engine-reported) | **`false` / `false`** (no `SCORE`, no `SAVE`/`LOAD`/`STATE`) |
| `batching` | `false` | `false` (same value, same contract) |

Files (all in `backends/qwen35_llama/tools/`; official files, the engine source and every
frozen artifact are untouched):

| File | What it is |
|---|---|
| `backend_adapters.py` | the adapter layer: selection, capability synthesis, refusal discipline |
| `score_api.py` | the HTTP service, now driving the selected adapter (endpoint semantics unchanged) |
| `stub_engine.py` | **test-only** stub of the qwen35 protocol (existing) |
| `stub_engine_official_q36.py` | **test-only** stub of the official protocol (new) |
| `test_score_api.py` | the existing 0-GPU suite (19 tests at P2-7; 63 before P2R-5, 68 after; still green) |
| `test_backend_adapters.py` | the new 0-GPU suite (13 tests at P2-7; 22 after P2R-5's matrix tests; both stubs) |

Status of verification: **0 GPU, no engine run, no lease.** Everything below was verified
with the two stub engines on CPU. No real-engine run of the `official_q36` adapter has been
done in this phase (that would need the official engine binary and a GPU lease, and is not
part of this task).

## 1. Selecting a backend

```text
# qwen35_llama (default; identical to the pre-P2-7 behavior)
python backends/qwen35_llama/tools/score_api.py --backend qwen35_llama --start-engine ...

# official q36 engine (its own --serve protocol; no CAPS/SCORE/SAVE/LOAD)
python backends/qwen35_llama/tools/score_api.py --backend official_q36 --start-engine ...
```

- `--backend qwen35_llama|official_q36` on the command line, or the `"backend"` field of the
  `--config` JSON (the CLI flag wins when both are present). Default: `qwen35_llama`, so the
  pre-P2-7 command lines keep their meaning.
- Start mode is the same for both: spawn the engine, wait for `READY <ctx> stop`. The
  official adapter records `<ctx>` from the READY line (the only source of the context size
  in that protocol).
- Attach mode differs where the protocols differ: `qwen35_llama` sends `CAPS` as its
  handshake (pre-existing behavior); `official_q36` has no `CAPS`, so no handshake is sent
  and the service says so on stdout. Nothing is guessed on its behalf.
- An unknown backend is refused before anything starts: `--backend` is validated by
  argparse, a config-file value by `resolve_backend()` (`SystemExit` with the supported
  list).

## 2. Capability tables per backend (not equivalent)

`GET /v1/capabilities` always adds `service` and `engine_sha256`; the backend-specific part
is composed by the adapter.

**`qwen35_llama`** - the engine's own `CAPS` table, passed through unchanged; the framework
adds only `backend` and, while `batching` is false, `batching_contract`:

```json
{"protocol":"strata-serve-v1","continuous_requests":true,"cancel":true,"state_isolation":true,
 "prefix_cache":false,"batching":false,"batching_contract":"1","mtp":{...},"expert_formats":[...],
 "context_max":4096,"logits_read":true,"state_export":true,"disk_kv":false,"vision":false,
 "backend":"qwen35_llama"}
```

**`official_q36`** - synthesized by the framework, because the protocol has no `CAPS`
command and there is no engine-reported table to pass through:

```json
{"backend":"official_q36","protocol":"official-q36-serve",
 "capability_source":"framework-synthesized-from-protocol-facts",
 "generation":true,"continuous_requests":true,"cancel":true,
 "prefill_progress":true,"token_stream":true,
 "context_max":4096,"batching":false,"batching_contract":"1",
 "logits_read":false,"state_export":false}
```

Per-field basis (official table):

| Field | Value | Basis |
|---|---|---|
| `generation` | `true` | `GEN` is the protocol's request command (`serve/server.py` writes `GEN <max_new> [key=value ...] <ids>`) |
| `continuous_requests` | `true` | resident serve loop; the official server serializes repeated `GEN`s on one process (FIFO, one at a time - no concurrency implied) |
| `cancel` | `true` | `STOP` is part of the protocol and the server uses it to cancel; `READY ... stop` declares the engine honours it |
| `prefill_progress` | `true` | `PP` lines exist (`PP <done> <total> <ms> <tok/s>`) |
| `token_stream` | `true` | `T <id>` lines exist |
| `context_max` | `int` / `null` | from `READY <ctx> stop` when this process saw it (start mode); `null` when unknown (attach mode) - undetermined, not fabricated |
| `batching` | `false` | no `INFO batch_slots=N` in the driven surface; the official server batches only when the engine reports it. Contract: `docs/backends/qwen35_llama_batching_contract.md` |
| `batching_contract` | `"1"` | version of that contract |
| `logits_read` | `false` | **no `SCORE` command** in this protocol |
| `state_export` | `false` | **no `SAVE`/`LOAD`/`STATE` commands** in this protocol |

Fields the official protocol cannot report (`prefix_cache`, `state_isolation`, `mtp`,
`expert_formats`, `disk_kv`, `vision`, ...) are deliberately **absent**: absence is not a
claim in either direction. The official engine's own extensions (`GENI` image inputs,
`BGEN` batch slots, `RESTORE`/conversation cache, `INFO batch_slots`) are outside the
surface this adapter drives and are neither claimed nor used. In particular this table must
**not** be read as "the official engine cannot batch" - it says the framework does not
drive or claim batching for it (same stance as for `qwen35_llama`).

## 3. Endpoint behavior (unchanged semantics, per-backend support)

| Endpoint | `qwen35_llama` | `official_q36` |
|---|---|---|
| `GET /v1/capabilities` | 200, engine CAPS passthrough + framework fields | 200, synthesized table |
| `POST /v1/score` | 200 (unchanged) | **501** with the reason |
| `POST /v1/decisions` | 200 (unchanged) | **501** with the reason |

The 501 body is the service's normal error shape
(`{"error": ..., "error_source": "service"}`); the message names the endpoint, the backend
and the missing command, e.g.:

```text
/v1/score is not implemented on the official_q36 backend: its protocol (GEN/STOP/QUIT) has
no SCORE command, so next-token logprobs cannot be read. This service refuses explicitly
instead of substituting another operation (it never degrades /v1/score into generation).
Use the qwen35_llama backend for /v1/score.
```

No silent degradation exists anywhere in the path: the adapter raises
`UnsupportedBackendOperation`, the service maps it to `501`, and no engine command is sent
for a refused request.

## 4. Command surfaces as data

`backend_adapters.py` carries each surface explicitly (`client_commands`, `engine_lines`),
so the difference is testable rather than implicit: the official adapter's
`client_commands` is exactly `GEN/STOP/QUIT` - no `CAPS`, `SCORE`, `SAVE`, `LOAD`,
`STATE`. `QUIT` is part of both surfaces: the official server ends the engine with `QUIT`
before closing stdin (`serve/server.py`).

## 5. Verification (0 GPU)

```text
python backends/qwen35_llama/tools/test_score_api.py -v        # 68 tests, OK (regression)
python backends/qwen35_llama/tools/test_backend_adapters.py -v # 22 tests, OK (P2R-5 additions included)
```

The new suite drives both stubs: backend selection (CLI flag + config field + default +
unknown-value refusals), capability synthesis per backend (including the `false` items and
the contract version), `context_max` from the READY line, the explicit 501s with no
generation, a raw-pipe protocol-fidelity test of the official stub (`GEN` -> `PP`/`T`/
`DONE`, `CAPS`/`SCORE`/`SAVE`/`LOAD`/`STATE` -> `ERR`, stale `STOP` dropped, `QUIT` exits
0), and both CLI start-engine paths end to end. No GPU, no lease, no real engine, no
writes outside the repository.

## 6. Not done in this phase (explicitly)

- No generation endpoint is exposed by the HTTP service; the adapters' job here is
  selection, capability honesty and refusals.
- No batching/concurrency (contract only: `qwen35_llama_batching_contract.md`).
- No official-file or engine-source change; no real-engine run of the official backend.
- No claim that the two backends are equivalent, in either direction.

## 7. P2R-5: the declarative routing/capability matrix

PR#2 review (P1, "backend boundary"): the adapter layer had only two hand-written gates
(`require_score` / `require_backend_session`); there was no queryable contract of which
endpoint runs on which backend, how a refusal is shaped, or where the two-service split lives.
P2R-5 turns that into data in `backend_adapters.py`:

- `ROUTES` - `(endpoint, backend) -> cell`, one cell per pair (10 endpoint ids x 2 backends =
  **20 cells**, complete by construction and by test). Cell fields: `supported` (bool),
  `engine_command` (the protocol command line it maps to, or `null`), `refusal_status` (`501`
  for refused in-service cells, else `null`), `refusal_text_template` (`{path}`/`{backend}`/
  `{commands}`, else `null`), `requires` (the engine CAPS field the supported contract rests
  on, or `null`), `in_service` (whether score_api exposes it as an HTTP route at all) and
  `notes` (what the cell does and does not cover).
- `route(endpoint, backend)` - the cell (a copy); an unknown endpoint or backend raises
  `ValueError` (at HTTP level an unknown path stays `404`).
- `refusal_text(endpoint, backend)` - renders the cell's 501 text. On `official_q36` the four
  texts are **byte-identical** to the pre-P2R-5 hand-written ones (pinned by exact-string
  tests in both suites).
- `capabilities_for(backend)` - the static capability contract this layer declares (no engine):
  `score`, `score_explicit_candidate_ids`, `state_keep_score`, `session_continue`, `STATE`,
  `state_export`, `caps_handshake`, `protocol`, `client_commands`, plus each endpoint's
  `supported` flag. The declarative sibling of `BackendAdapter.capabilities()`, which needs a
  live link.
- `routes_payload()` - the whole matrix as JSON for `GET /v1/routes` (the 20 cells, each with
  its rendered `refusal_text` when refused).

`score_api.py` no longer decides anything by hand: every endpoint gate is
`route(endpoint, active_backend)`, and an unsupported in-service cell raises the cell's own
`NotImplementedFeature(refusal_text(...))`. The score family consults its three form cells
(`POST /v1/score` top-k, `:candidates`, `:keep`); none of the added checks is observable when
supported. `requires` documents the engine-side precondition (e.g. `keep` needs the engine's
`state_keep_score`); it is **not** a runtime check this round - the engine's own CAPS declaration
is the contract, and nothing is synthesized on the engine's behalf.

Matrix (pre-registered in the task and filled cell-by-cell; `in_service=false` rows are
boundary facts, not HTTP refusals):

| endpoint (matrix id) | `qwen35_llama` | `official_q36` |
|---|---|---|
| `POST /v1/score` (top-k) | supported - `SCORE <ids> topk=N` | **501** - no `SCORE` in the protocol |
| `POST /v1/score:candidates` | supported - `SCORE <ids> topk=N cand=<ids>` (requires `score_explicit_candidate_ids`) | **501** |
| `POST /v1/score:keep` | supported - `SCORE <ids> topk=N [cand=<ids>] keep=1` (requires `state_keep_score`) | **501** |
| `POST /v1/decisions` | supported - `SCORE <ids> topk=0 cand=<ids> [keep=1]` (requires `score_explicit_candidate_ids`; `keep` also `state_keep_score`) | **501** |
| `POST /v1/continue` | supported - `GEN <max_new> cont=1 [<ids>]` (requires `session_continue`) | **501** |
| `GET /v1/session` | supported - `STATE` | **501** |
| `GET /v1/capabilities` | supported - `CAPS` passthrough + framework fields | supported - framework-synthesized (no CAPS handshake) |
| `GET /v1/routes` | supported - framework-only, no engine command | supported - framework-only |
| engine `SAVE`/`LOAD` | supported - container v1 (`STRATAST`; not an HTTP route) | not supported - the official `RESTORE`/`SERR` grammar is outside this adapter and untouched |
| chat `/v1/chat/completions` | not in this service - official `serve/server.py` drives the same exe (single-entry recipe below) | not in this service - the official server is the built-in chat path |

## 8. Service discovery: `GET /v1/routes`

Read-only, rendered from the static matrix - **no engine command, no busy lock**, so it answers
the declared contract even on a dead/quarantined link:

```json
{"active_backend": "qwen35_llama",
 "backend_source": "cli | config | default (| unspecified when constructed directly)",
 "routes": [ {"endpoint": "POST /v1/score", "path": "/v1/score", "backend": "qwen35_llama",
              "in_service": true, "supported": true,
              "engine_command": "SCORE <id,id,...> topk=N", "requires": null,
              "refusal_status": null, "refusal_text": null, "notes": "..."} , ... (20 cells) ]}
```

`/v1/capabilities` now also carries `active_backend`, so the two discovery endpoints agree.
A caller decides "can I use endpoint X with backend B?" by reading the cell; the actual refusal
it would receive (501 text) is in the same cell. `GET /v1/routes` is 200 on both backends;
refused cells are marked `supported=false`, and the actual call answers 501 (spot-checked).

## 9. Single-entry deployment: recipe, boundaries, follow-up

**Stance (goal):** the user manages one deployment ("one config, one command, two health
checks"), not two unrelated services. This round delivers the **recipe**; the two listeners and
two engine processes still coexist (see the not-unified list - nothing here claims a merged
service, and no script is added this round: the seven-file scope of P2R-5 is code+docs only).

Recipe (documented, not scripted):

1. One config file in the shared shape (`exe` / `args` / `cwd` / `env` - the official server's
   `StrataEngine` block, which also tolerates the extra keys this service reads):

   ```json
   {"exe": "backends/qwen35_llama/build/strata.exe",
    "cwd": "backends/qwen35_llama",
    "args": ["--qwen35-model", "<model.gguf>", "--qwen35-pack", "<pack dir>",
             "--max-context", "4096", "--serve"],
    "backend": "qwen35_llama",
    "tokenizer": "<model.gguf>"}
   ```

   `args` must contain `--serve`; no prompt belongs in the args (serve mode).
2. Chat (official server, its built-in endpoint): `python -m serve.server --engine strata
   --config <config.json> --port 8080` - health: `GET /health`.
3. Score/decisions/session (this service, same config block): `python
   backends/qwen35_llama/tools/score_api.py --config <config.json> --start-engine
   --backend qwen35_llama --tokenizer-model <model.gguf> --port 8097` - health: the
   `READY <ctx> stop` line on stdout, `GET /v1/capabilities` (with `engine_sha256`), and
   `GET /v1/routes` for the matrix.

Engine-mode choice (fixed by this doc): the recipe uses `--start-engine` with the shared
config block. The other real mode, pipe attach (`--attach-stdin` / `--attach-stdout`), is for a
user who owns the engine's pipes; there is **no HTTP attach** in this round (the review text's
`--attach-url` option does not exist here). Boundary: each program drives its **own** engine
process; the official server does not share its engine with this service, and on the 8 GiB
target do not assume both engines can be resident at once (unvalidated; the memory tradeoff is
explicitly out of this task).

**Not unified in this round (explicit):**

- two listeners still coexist (official server for chat; this service for score/decisions/
  session); `GET /v1/routes` describes this service's contract, not a merged one;
- the official server's session grammar (`RESTORE`/`SERR`, conversation cache) is not merged
  into this service, and this service's container v1 (`SAVE`/`LOAD`/`STATE`) is not merged into
  the official protocol;
- the batching contract stays v1 and unimplemented on both services
  (`qwen35_llama_batching_contract.md`);
- chat is not an HTTP endpoint of this service (`404`); no shared engine process / no
  attach-url;
- the matrix's `requires` preconditions are engine-declared contracts, not runtime checks.

**Follow-up route (when the split may actually be merged):** move the score/decisions/session
endpoints into the official server once (a) the server's engine boundary can carry the
`CAPS`/`SCORE`/`GEN cont=1`/`STATE` commands without disturbing its request flow, (b) the two
session grammars are reconciled (container v1 vs `RESTORE`/`SERR`), and (c) batching contract
v1 is implemented. Until then, this matrix + recipe is the contract.
