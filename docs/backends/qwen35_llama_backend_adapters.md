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
| `test_score_api.py` | the existing 0-GPU suite (19 tests, untouched, still green) |
| `test_backend_adapters.py` | the new 0-GPU suite (13 tests, both stubs) |

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
python backends/qwen35_llama/tools/test_score_api.py -v        # 19 tests, OK (regression)
python backends/qwen35_llama/tools/test_backend_adapters.py -v # 13 tests, OK (new)
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
