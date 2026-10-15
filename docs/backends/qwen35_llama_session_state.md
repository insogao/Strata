# qwen35_llama single-session state export/import (P2-5, containerized by P2R-4a)

**Status**: implemented in `src/program/qwen35_serve.cpp` (P2-5, 0-GPU task; **P2R-4a** added the
self-describing file container - magic/version/lengths/CRC32/model fingerprint - around the unchanged raw
payload). This is the **single-session state boundary** for the serve engine: one explicit export/import of
the CURRENT session state; the payload itself still reuses llama's own sequence-state serialization
(`llama_state_seq_get_size/get_data/set_data`) - the container only describes and protects those bytes.
Functional/GPU verification (parity) ran as **P2-6** on 2026-10-15 against the **pre-container raw-blob
build** and passed on this machine/build (same-process and cross-process continuations byte-identical;
failure legs left the live session untouched) - results, raw lines and limits: section 11.

**P2R-4a scope**: the container format, the fixed LOAD validation order, the model fingerprint and the
`STATE` additions are implemented and documented here; the **real-engine matrix for the container
(corrupt/foreign/wrong-model/wrong-version/repeated-restore/isolation) ran as P2R-4c on 2026-10-15
(section 13: N1/N2 pass with documented sequence adaptations - pure continuations after LOAD cannot
sample, see there)**, and the follow-up **P2R-4d (section 14)** applied the supervisor decision on the
one doc-vs-build delta P2R-4c found: the v1 magic is now exactly the 8 ASCII bytes `STRATAST` in the
shipped build (early builds zero-filled the eighth byte and wrote `SRATAST\0`; those files are rejected
as bad magic), and it closed the `SAVE -> LOAD -> SAVE` byte-identity re-check. Before P2R-4a it was the
0-GPU build + unit tests of P2R-4a plus the (legacy-format) P2-6 parity record.

Scope rules this doc obeys: nothing here claims cross-machine portability; cross-process continuation
restores are verified for the same machine and build only (P2-6 N1/N2, section 11; container matrix:
section 13 - it includes a cross-process container restore: native2 LOADed the container written by
native1).

## 1. Commands

Added to the serve line protocol (client -> engine, one line each, processed between requests):

```text
SAVE <path>      export the current session state to <path>
LOAD <path>      import a session state from <path>; the session can then continue with GEN
STATE            print one JSON line describing the session bookkeeping
```

Answers (engine -> client, stdout):

| Command | Success | Failure |
|---|---|---|
| `SAVE` | `SAVED <bytes> <pos> <ms>` - **container file size** (P2R-4a; the pre-container build reported the raw payload size - see section 1b), sequence-0 cursor (occupied positions), elapsed ms | `ERR <reason>` |
| `LOAD` | `LOADED <bytes> <pos> <ms>` - the same fields, measured on the imported container file (container file size, cursor, elapsed) | `ERR <reason>` |
| `STATE` | `STATE <json>` (below) | never fails (a pure query) |

`<path>` is the rest of the line (leading/trailing spaces trimmed; paths with spaces are allowed). The path
gate (all checks textual; symbolic links/junctions are not resolved):

- non-empty, at most 1024 characters, **printable ASCII only** (it travels through the line protocol and the
  ANSI file APIs);
- **absolute** (drive-qualified `D:\...`/`D:/...` or UNC `\\server\share` on Windows; leading `/` elsewhere);
- no `..` component; must not end in a separator (a file name, not a directory);
- when the optional allow-list `STRATA_QWEN35_SESSION_ROOTS` is set (semicolon-separated absolute
  directories), the path must be inside one of the roots (normalized text comparison: `\`->`/`, ASCII
  lowercase on Windows, no trailing separator, component-boundary match). With the variable unset or empty,
  any structurally valid absolute path passes the shape gate and the filesystem decides readability/
  writability.

`STATE` JSON fields (one line, stable order; values below are illustrative):

```json
{"loaded":false,"pos":0,"bytes":2216,"path":null,"covers":["attn_kv","gdn_state","conv","pos"],
 "session":false,"origin":null,"file_bytes":null,"payload_bytes":null,"max_context":4096,
 "container_version":1,"model_fp_prefix":"0123456789abcdef",
 "not_covered":["logits","sampler_rng_state","expert_cache","pack","bridge_counters","rollback_history"]}
```

| Field | Meaning |
|---|---|
| `loaded` | `true` when the ACTIVE session came from a `LOAD` (imported file); `false` after a plain `SAVE`, before any session command, or after the session was cleared |
| `session` | `true` when a session is armed at all (`SAVE` or `LOAD`); this is the fact that makes the next `GEN` continue instead of resetting |
| `origin` | `"save"` / `"load"` / `"score-keep"` / `null` - which command armed the active session (`score-keep` = P2R-3a `SCORE keep=1` that found no armed session; continuing an armed session never changes it) |
| `pos` | current sequence-0 cursor: last occupied position + 1 (`0` when the sequence is empty). This is the position a continuation appends at |
| `bytes` | byte size of the state **as it is now** (the raw payload size `SAVE` would write at this moment). On a fresh/cleared engine this is the non-zero size of the empty state |
| `path` | file of the active session (`null` when none) |
| `file_bytes` | byte size of that file at the time of the last successful `SAVE`/`LOAD` - **the container size** (P2R-4a); `null` when no session file is involved (or after the session advanced the live state differs, see `bytes`) |
| `payload_bytes` | **P2R-4a**: the raw sequence-state payload size inside that file (`file_bytes` minus the 96-byte header for v1); `null` when no session file is involved |
| `covers` | what the exported payload covers - see section 2; this list is the implementation's own truth table |
| `not_covered` | state a restored session does NOT carry (see section 2) |
| `max_context` | the live `llama_n_ctx` (the same value `READY` advertises) |
| `container_version` | **P2R-4a**: `1` - the session-file container version this build writes and the only one it accepts (a build-level fact, like `covers`, independent of session state) |
| `model_fp_prefix` | **P2R-4a**: the first 8 bytes (16 lowercase hex chars) of the startup-cached model fingerprint (section 1b); `null` only in the corner case where the fingerprint could not be computed at startup (then `SAVE`/`LOAD` refuse) |

## 1b. The session file container (P2R-4a)

The file `SAVE` writes is no longer the bare blob: it is a **self-describing container** around the same
raw payload. Byte order is little-endian; the header is fixed and its length is a field, so v1 readers
require at least 96 bytes but accept longer headers (a writer may append fields):

| Offset | Length | Field | Meaning |
|---|---|---|---|
| 0 | 8 | `magic` | `"STRATAST"` (8 ASCII bytes, no NUL) |
| 8 | 4 | `version` | u32 = 1 |
| 12 | 4 | `header_bytes` | u32, total header length (v1 writers: 96) |
| 16 | 8 | `payload_bytes` | u64, raw `llama_state_seq_get_data` byte count |
| 24 | 8 | `file_bytes` | u64, `header_bytes + payload_bytes` (the whole file) |
| 32 | 4 | `payload_crc32` | u32, CRC-32/ISO-HDLC over the payload |
| 36 | 4 | `n_vocab` | u32, vocabulary size of the saving model |
| 40 | 4 | `max_context` | u32, context size of the saving engine |
| 44 | 8 | `pos` | u64, sequence cursor at `SAVE` (= `SAVED`'s `pos`) |
| 52 | 32 | `model_fp` | model fingerprint (see below) |
| 84 | 4 | `flags` | u32 = 0 (reserved) |
| 88 | 8 | reserved | zero (up to `header_bytes`) |
| `header_bytes` | `payload_bytes` | payload | the unchanged raw sequence-state bytes |

**The v1 magic is exactly the 8 ASCII bytes `STRATAST` (hex `53 54 52 41 54 41 53 54`), no NUL.** Early
P2R-4a..P2R-4c builds zero-filled the eighth byte of the constant and wrote `SRATAST\0` (hex
`53 52 41 54 41 53 54 00`), which disagreed with this table and with every `ERR` text; **P2R-4d
(section 14) fixed the constant** so writer, validator, error texts and this table all spell
`STRATAST`. The old-literal files were never released and there is **no migration burden**: the fixed
build refuses them with the bad-magic error above - they are neither containers of this version (byte 8
is NUL) nor legacy raw blobs (their first four bytes `53 52 41 54` are not llama's raw magic
`0xaf143cd8`).

`SAVE` writes the container to `<path>.tmp` and atomically replaces `<path>` (same discipline as P2-5),
and `SAVED <bytes>` reports the **container file size** (`file_bytes`; exactly 96 bytes more than the
pre-container build's count for the same state). `LOADED <bytes>` reports the container file size of the
imported file, so `SAVE` and `LOAD` of one file agree. `STATE` reports the split: `file_bytes` = container
size, `payload_bytes` = raw payload size.

`LOAD` validates the container in this **fixed order**, and every mismatch answers its own `ERR` and leaves
the live session (including an armed one) untouched - the import path is not entered, so no half-restore
is possible at this stage either:

1. **readable** - path gate, then file opened/read (whole file, 1 GiB cap - the cap now applies to the
   container; the payload is at most 999,999,904 bytes) and at least 96 bytes long;
2. **file size** - the file length equals the `file_bytes` field;
3. **magic** - `"STRATAST"`;
4. **version** - equals 1;
5. **header_bytes** - at least 96, within the file, and `payload_bytes == file size - header_bytes`;
6. **n_vocab** - equals the running model's vocabulary size;
7. **max_context** - equals the running engine's context (`llama_n_ctx`); equality, not ">=" - a state
   saved under a different context is a different context shape and the import requires the exact one;
8. **pos** - `pos <= max_context` (the payload's own cell metadata is validated by llama's reader inside
   the import, step 10 below);
9. **payload CRC32** - CRC-32/ISO-HDLC of the payload equals `payload_crc32`;
10. **model_fp** - the 32-byte fingerprint equals the engine's startup-cached fingerprint.

Only after all ten pass does the payload go through the P2-5 import discipline: a **temporary context**
(same model and parameters, fresh batch), `llama_state_seq_set_data` (llama's reader validates the raw
blob's own structure - the payload's `<u32 0xaf143cd8><i32 seq_id 0>` header included), and the swap-in
only on full success (section 4). Distinct error texts (all on stdout as `ERR session: ...`, the engine
keeps serving):

| Mismatch | `ERR` text (after `ERR session: `) |
|---|---|
| too short for the fixed header, not legacy | `'<path>' is too small to be a STRATAST session container (<n> bytes; the fixed header is 96)` |
| size field mismatch | `'<path>' size mismatch (the file is <n> bytes, the container declares <m>)` |
| magic missing, foreign file | `'<path>' is not a STRATAST session container (bad magic)` |
| **legacy raw blob** | `'<path>' is a legacy raw state blob (no STRATAST container); re-SAVE it with this build to migrate` |
| version | `'<path>' has container version <v>; this build reads version 1` |
| invalid header length fields | `'<path>' has an invalid container header length (header <h>, payload <p>, file <n>)` |
| vocab | `'<path>' was saved for vocab <v>; this model has <n>` |
| context | `'<path>' was saved under context <c>; this engine runs context <n>` |
| pos | `'<path>' declares position <p> beyond the context (<n>)` |
| CRC32 | `'<path>' payload CRC32 mismatch (container <x>, computed <y>)` |
| fingerprint | `'<path>' model fingerprint mismatch (container <pfx>..., this engine <pfx>...)` |

**Legacy raw blobs (pre-P2R-4a `SAVE` files)** are those whose first four bytes are llama's own raw
sequence-state magic `0xaf143cd8`. They are always **refused** with the dedicated text above (never
silently accepted); the migration is simply to `LOAD` them with the pre-container build (or any build that
still accepts them) and re-`SAVE` with this build - the engine no longer reads raw blobs at all. The
"legacy" classification is reachable both from the size check and from the magic check, so an old file is
always reported as an old file, not as a generic length error.

**Model fingerprint** (`model_fp`, `STATE`'s `model_fp_prefix`): `SHA-256( size_le64 || first min(1 MiB,
size) bytes || last min(1 MiB, size) bytes )` of the `--qwen35-model` file - exactly 2 MiB + 8 bytes are
ever read, never a full 13.6 GB hash. It is computed **once at startup** (right after the model loads) and
cached; `SAVE` stamps the cached value and `LOAD` compares against it. A same-path replacement of the model
file **after** startup does not change the weights the running engine holds, so the startup value remains
the correct compatibility reference and is deliberately not recomputed. What it is: a **compatibility
check** that tells models (architectures/quantizations/lengths) apart. What it is **not**: a security
measure (no keyed MAC, trivially forgeable by anyone editing the file; it detects accidents, not attacks).
Boundary: the **expert pack is not part of the fingerprint** - the pack (expert weight blobs) is not
sequence state and is not carried by the container (`not_covered` lists `pack`/`expert_cache`); a state
restored against the same model but a different pack restores as valid numbers of the same model with
different expert weights - the fingerprint would not detect it. If a future flow needs pack identity, it
must be added explicitly (e.g. as another fingerprint field).

## 2. What is exported: covers, exactly

The file contains the raw bytes of `llama_state_seq_get_data(ctx, ..., /*seq_id*/ 0)` - the engine's own
sequence-state blob for the single sequence this serve process runs. Its content is decided by the memory
module, and for the qwen35 (hybrid GDN + attention) architecture that is:

- `llama_memory_hybrid::state_write` -> `mem_attn->state_write` **plus** `mem_recr->state_write`
  (`vendor/llama.cpp/src/llama-memory-hybrid.cpp`).

| `covers` item | Backed by | What it is |
|---|---|---|
| `attn_kv` | `llama_kv_cache::state_write` | the attention layers' K/V cells (tensor data) and the cell metadata that owns them |
| `conv` | `llama_memory_recurrent::state_write`, the `r_l` rows | the GDN **convolution state** per recurrent layer (the same rows the graph reads through `ggml_ssm_conv`); the PLE conv-history row (`p_l`) travels with it when the model declares one |
| `gdn_state` | `llama_memory_recurrent::state_write`, the `s_l` rows | the GDN **recurrent (SSM) state** per recurrent layer |
| `pos` | both writers' cell metadata | per-cell positions and sequence ownership (the cursor a restored session resumes at) |
| *(blob header)* | `llama_context::state_seq_get_data/set_data` | `<u32 magic 0xaf143cd8><i32 seq_id>` written/checked by the engine itself |

So this is **not** an attention-KV-only snapshot: a restored session without the GDN planes could not
continue a recurrent model correctly, and both planes are in the blob by construction (the hybrid memory
writes/reads both, and a failure of the recurrent half aborts the whole import).

**Not covered** (`not_covered` in `STATE`, honest list):

- `logits` - the last step's output row is not part of the state; a continuation feeds tokens to produce the
  next row (a session can only be continued, not re-sampled at the same position without a new token);
- `sampler_rng_state` - the sampler chain is recreated per request; an explicit `seed` reproduces sampling;
- `expert_cache` - the expert residency cache is a separate subsystem (`qwen35_bridge`), untouched by the
  blob; a restored session simply starts with whatever residency it has, which affects speed, not values;
- `pack` - the expert weight pack directory is not sequence state and is not carried by the container;
  the model fingerprint covers the GGUF model file only (section 1b), so a state is not refused when the
  pack differs - restore the same pack the state was produced with if values must match;
- `bridge_counters` - the engine's diagnostics/statistics are not session state;
- `rollback_history` - the recurrent rollback snapshots (when the build has any) are not part of the blob;
  the import restores the logical current rows and resets the rollback index binding, so a restored session
  holds its current state, not the history that produced it.

## 3. Session semantics: what `GEN` does after SAVE/LOAD

The serve engine's documented default is per-request isolation (each `GEN`/`SCORE` starts from
`llama_memory_clear(mem, true)`). P2-5 adds one explicit exception:

1. A successful `SAVE` or `LOAD` **arms** the session (`session.active=true`).
2. The next `GEN` (and every `GEN` while the session stays armed) **does not reset**. It prefills its prompt
   at the current cursor (`pos`) and decodes from there - i.e. it is a **continuation**: the caller sends the
   tokens that extend the session (typically the next user turn; if the last generated token was not fed into
   the context, sending it first is the natural continuation).
3. The `DONE` field `reused` reports the carried cursor (the positions the request inherited); it stays `0`
   for requests with no session. All other fields are unchanged.
4. The session cursor and content evolve with each continuation; `STATE` always reports the live values
   (`bytes`/`pos`), while `path`/`file_bytes`/`payload_bytes` still describe the last file written/read
   (`file_bytes` = container size, `payload_bytes` = raw payload inside it).
5. The session is cleared (back to full isolation) by:
   - a `SCORE` request **without** `keep=1`, which performs its documented semantics-(b) reset - see
     section 6 (`SCORE ... keep=1` instead keeps/arms the session, and a failed/cancelled keep request
     clears it - section 3b);
   - process exit (nothing is implied to persist);
   - there is no separate "close session" command in this phase.
6. A failed `SAVE`/`LOAD` does **not** change the arming state: the previous session keeps working.
7. `CAPS`, `STOP`, `QUIT` and stdin EOF are unchanged by sessions.

The exported bytes are the state of the CURRENT sequence, whatever produced it: after a `GEN` it is that
request's final state (prompt + decoded tokens); after a `SCORE` it is the cleared state (see section 6);
right after `LOAD` it is the imported state. `SAVE` never fabricates or normalizes anything: it serializes
what the context holds.

## 3b. `SCORE keep=1` and `GEN cont=1`: scoring -> continuation without files (P2R-3a)

P2R-3a adds a **file-less** way to score and continue: `SCORE <ids> keep=1` answers the same read-only
`LOGPROBS`/`CANDLOGPROBS` payload while leaving the session armed, and `GEN <max_new> cont=1 [<ids>]`
continues an armed session explicitly (the multi-round continuation P2-5 only exposed through SAVE/LOAD).
**SAVE/LOAD are not a precondition of this flow and are not part of its acceptance** - the two commands
work with no session file anywhere; when a file-armed session exists they simply continue it.

Contract (`keep` and `cont` accept only the literal `1`; `keep=0`/`keep=true`/`keep=2`/`keep=` ->
`ERR bad request: keep`, `cont=` with any non-`1` value -> `ERR bad request: cont`; a repeated `keep=`/
`cont=` key -> `ERR bad request: duplicate keep key` / `ERR bad request: duplicate cont key`):

- `SCORE ... keep=1` with **no armed session**: the usual reset, the prompt is read from 0 (`reused=0`),
  the answer is printed exactly as a default `SCORE`, and after a **successful** answer the session is
  **armed** with cursor = tokens read; `STATE` then reports `origin="score-keep"`, `loaded=false`,
  `session=true`, no file fields (`path`/`file_bytes` null).
- `SCORE ... keep=1` with an **armed session**: **no reset** - the prompt is appended at the current
  cursor, `DONE`'s `reused` reports the cursor before the append, and the session stays armed with the
  cursor advanced by what was read. Continuing never re-arms: the session keeps its previous `origin` and
  file fields (a continuation of a `LOAD`ed session still reports `origin="load"`).
- A **failed or cancelled** `keep=1` request (any `ERR`, a NaN row, a `STOP`) resets the state and clears
  the session - a partial append is never left for a continuation to pick up; re-`LOAD` the checkpoint to
  continue from it. This matches the default `SCORE` failure behavior (it never leaves a session armed).
- `GEN <max_new> cont=1 [<ids>]`: **requires** an armed session, otherwise `ERR cont=1 needs an armed
  session`; the id list may be **empty** (pure continuation: append nothing, decode from the cursor) or
  non-empty (append, then decode). `DONE` reports `prompt_tokens`/`prompt_read` = the tokens actually
  appended/read (0 for a pure continuation) and `reused` = the cursor before the append; after the decode
  the session stays armed (multi-round). Without `cont=` a session `GEN` behaves exactly as in section 3
  (and an empty id list stays `ERR bad request: no token ids`).
- Context limits: `cursor + prompt (+ max_new)` must fit `max_context` as before; a `keep=1` append that
  would exceed the context answers `ERR session (...) + prompt (...) exceeds the context (...)` and leaves
  the armed session **untouched** (nothing was read, nothing was reset).
- The stdout protocol is the same shape as the default paths (`PP`/`LOGPROBS`/`CANDLOGPROBS`/`DONE ...
  score` for the score; `PP`/`T`/`DONE` for the continuation); the stderr diagnostic appends ` keep=1` /
  ` cont=1` for these requests.
- `CAPS` advertises the flow as `state_keep_score` / `session_continue` (both `true`).

Boundaries: the same single-session rules as the rest of this document (one sequence, same build/model);
`STATE`'s `covers`/`not_covered` are unchanged - `keep=1` adds no new state kind, it only changes when the
reset happens. Verification: **P2R-3b ran 2026-10-15** (one native, exe `7c441a94…`): the flow's
load-bearing facts were confirmed on the real engine (continuation `prompt_read=0`/`prompt_ms=0` - no
re-prefill; `origin="score-keep"`; `ERR cont=1 needs an armed session` after a plain `SCORE`; fresh path
after a session clear; the `keep=1` append context guard; same-run kept-vs-fresh full-prefill reference
16/16 token identity), but the pre-registered pairwise identity against an *immediately following*
`GEN <P_long>` is not executable at ctx4096, because a plain `GEN` continues an armed session and its
P_long append overflows (`ERR session (...) + prompt (...) exceeds the context`) - those items are
recorded as FAIL-as-written in `docs/backends/qwen35_llama_score_api_e2e.md` §P2R-3b and a
corrected-sequence re-check is a separate dispatch; the service mapping (`/v1/score` with `keep`,
`/v1/continue`) is P2R-3c.

## 4. Failure semantics (no half-restore)

`SAVE`:

- path gate failures (`ERR session: ...`), a missing startup fingerprint (refused with its own `ERR`; the
  container must carry a fingerprint), state size/read failures, and file failures all answer `ERR` and
  leave the session (and any previously armed state) exactly as it was;
- the file write is atomic in effect: bytes go to `<path>.tmp`, then `MoveFileExA(..., REPLACE_EXISTING)`
  (POSIX `rename`) replaces `<path>`. On any failure the `.tmp` is removed and an existing `<path>` is left
  untouched - a failed save never truncates the previous file;
- `SAVE` replaces an existing file (documented, not a refusal).

`LOAD`:

- container failures happen in the fixed order of section 1b and before any engine change: bad path
  (`ERR session: ...`), unreadable file, file larger than the 1 GiB cap, file shorter than the 96-byte
  container header (legacy raw blob included), size/magic/version/header length mismatch, wrong vocabulary
  or context, illegal position, payload CRC mismatch, fingerprint mismatch, legacy raw blob - each with its
  own text (section 1b), all with the live session untouched;
- a structurally valid header does not prove a compatible body. The import therefore runs **into a temporary
  context first** (`llama_init_from_model` with the exact live context parameters + the fresh
  `llama_batch_ext`): `llama_state_seq_set_data` either fully succeeds there or fails (llama's reader
  validates the structure it knows - layer count, cell counts, tensor types/row sizes - and on failure clears
  the target and throws). The live context is **never touched** before that success; only then is the old
  context (and batch) freed and the imported one swapped in. A failed import brings the engine back to
  exactly the pre-`LOAD` session - never a half-restored state;
- cost note: during the import the process holds two contexts momentarily (peak memory ~ the old state + the
  new state buffers); accepted for an explicit control command.

Anything answered `ERR` keeps the engine serving - the same request-level contract as `GEN`/`SCORE`.

## 5. Compatibility and version boundaries (what this does NOT guarantee)

- **Same build, same model, same context parameters** is the only supported pairing. The container
  (P2R-4a) now detects much of what P2-5 could not: wrong format version, wrong vocabulary or context,
  corruption anywhere in the payload (CRC32), and a different model **file** (fingerprint over
  size + first/last 1 MiB). Known residual gaps, stated plainly: (a) the fingerprint is a 2 MiB+8 sample,
  so two same-size model files that differ only outside the sampled head/tail would not be told apart;
  (b) the pack is deliberately not fingerprinted (section 1b); (c) a container from a different
  build/architecture with the same model, vocab and context passes the container checks and is then
  caught (or not) by llama's structural validation inside the import - if it is accepted, the "values"
  are what that build wrote. Do not treat session files as portable artifacts.
- **No cross-machine guarantee**: nothing here promises that a file written on one GPU/machine restores
  identically on another. Numerical identity of a continuation is only expected within the same build and
  hardware; the actual parity check is P2-6 on this machine.
- **Cross-process continuation (same machine, same build)**: verified by P2-6 N2 on 2026-10-15 - a fresh
  process that `LOAD`s the file continued byte-identically to the uninterrupted session (section 11).
  Cross-machine portability remains unverified and is not claimed.
- Position limits: the continuation check `cursor + prompt + max_new <= context_max` uses the live context;
  importing a state whose cells exceed this context is rejected by llama's reader (`ERR`).
- File cap: 1 GiB (a size gate, not a measurement; chosen well above the expected state size for this model
  at ctx 4096 - the exact blob size is recorded in P2-6); the whole file is read into host memory.

## 6. Interaction with `SCORE` (semantics (b)) and the other commands

`SCORE`'s contract is semantics **(b) independent**: it resets the session before the request and again
after it, so the next request starts fresh. (`SCORE ... keep=1` is the P2R-3a state-keeping exception and
is documented in section 3b.) Consequences for P2-5, stated plainly:

- `SCORE` **clears an armed session**: the reset is a real reset, so the engine also clears the session
  bookkeeping (`STATE`: `session=false`, `loaded=false`, `origin=null`, `path=null`, `pos=0`, `bytes` = the
  empty-state size). A `GEN` after a `SCORE` behaves exactly like a fresh isolated request, even if a session
  was active before.
- `SAVE` **before** a `SCORE` stays valid: the file is on disk and can be `LOAD`ed later (that is the point
  of an explicit checkpoint). `SAVE` **after** a `SCORE` writes the cleared (empty) state - a valid,
  importable file that carries nothing; `LOAD`ing it arms a session whose cursor is 0, which behaves like a
  fresh request.
- The `SCORE` path itself is unchanged: it never reads or writes session files, and its `LOGPROBS`/`DONE`
  output is identical with or without an armed session. A `SCORE ... keep=1` (P2R-3a, section 3b) answers
  the same way but does not perform the (b) reset.

Other commands: `CAPS` is untouched (`state_export` is now `true`, see
`qwen35_llama_capabilities_and_score.md`). `STOP` during a continuation cancels at the next chunk/token like
any request; the session then holds a self-consistent partial append (cursor advanced by what was read) and
stays armed - exact reproduction requires re-`LOAD`ing the checkpoint. `QUIT`/EOF exit without writing
anything.

## 7. Four kinds of "cache" - which one is this

The roadmap distinguishes four different mechanisms that all sound like "keeping state". P2-5 implements
exactly the first one; the other three are not touched:

| Mechanism | What it is | This phase |
|---|---|---|
| **single-request state (this export)** | the live llama sequence-0 state of the running engine (attention KV + GDN conv/SSM + positions), serialized on demand by `SAVE` and restored by `LOAD` for the SAME session | **implemented (P2-5)**: explicit checkpoint/restore, no lookup, no eviction, no automatic reuse; arming the session makes the next `GEN` continue it |
| cross-request prefix cache | automatic detection that a new prompt shares a prefix with an existing context and reuse of those cells (`reused > 0` without an explicit command) | **not implemented** (`prefix_cache=false`); needs prefix matching + accounting the engine does not have |
| disk KV | a persistent server-side KV tier that pages state to/from disk transparently during and across requests | **not implemented** (`disk_kv=false`); a session file exists only because the caller explicitly `SAVE`d it |
| expert weights (expert cache/residency) | which routed expert weight blobs are resident in VRAM/RAM/disk - a WeightsCache, not activation state (the native engine's `--qwen35-cache-slots` world) | **orthogonal**: the blob never contains expert weights; a restored session reuses whatever residency the process has (speed only, not values) |

A session file is therefore not a "cache": nothing loads it implicitly, nothing looks anything up in it, and
it never affects a request unless the caller `LOAD`s it.

## 8. Same-process reentrancy

`SAVE -> LOAD -> SAVE` (and repeated cycles) is a supported operation sequence: `LOAD` swaps in a fresh
context built from the same model/parameters, so the following `SAVE` serializes that context. On the same
build the second `SAVE` is expected to produce a byte-identical file (same cells, same positions, same
tensor rows; the import packs cells from head 0). This expectation got its empirical check in P2-6 - on the
**legacy raw-blob format**; the two files were byte-identical (`sha256 7d34f3b9…368414`, section 11). For
the P2R-4a container the same expectation holds by construction (all header fields are deterministic for
an unchanged session; CRC32 of identical payload bytes is identical); its real-engine re-check did **not**
run in P2R-4c (the preregistered N1/N2 matrix has no SAVE->LOAD->SAVE leg - section 13) and remains open.

## 9. Not implemented / not claimed

- the official engine's disk-session grammar (`RESTORE <path>`, `SAVED <tokens> <bytes> <ms>`,
  `SERR <kind> <published>`, `SESSION`/`SWAIT` progress lines, `FATAL`) - this engine's commands are
  `SAVE`/`LOAD`/`STATE` with the answers above;
- token history: the file stores state only, not the token list (llama's `state_seq` blob has no tokens).
  A continuation works with token ids the caller supplies; there is no "resume the exact prompt" helper;
- multiple sessions, session ids, listing, deletion, quotas, compression, encryption, integrity beyond the
  container's CRC32 (no keyed MAC/cryptographic authentication - the fingerprint is a compatibility check,
  not a security measure), fsync/folder-flush guarantees beyond the replace;
- cross-machine/cross-build portability (section 5);
- GPU parity (section 10).

## 10. P2-6 plan (GPU parity: SAVE -> continuation vs LOAD -> continuation, byte-identical)

**Executed 2026-10-15: PASS** - outcome, raw lines, table and limits in section 11. The actual run used the
standard GPU discipline (gpu-lock leases, production gates, same model/pack as P2-2/P2-4) with context
**8192** (not 4096), a fixed 4-id continuation prompt `P2` preregistered in every leg, and one failure leg
each of missing/truncated `LOAD`; section 11 lists these deviations from the plan text below. The plan text
is kept as the preregistered design:

P2-6 runs on the real engine under the standard GPU discipline (gpu-lock lease, production gates, context
4096, the same model/pack as P2-2/P2-4) and checks, in order:

1. **Reference continuation**: `GEN <P1>` (produces tokens `A`), `SAVE <s>`, `STATE` (expect
   `session=true`, `origin="save"`, `pos` = cursor); then `GEN <P2>` where `P2` is a short continuation
   prompt (e.g. the last generated token plus a couple of ids) - this run APPENDS at the cursor and produces
   the reference tokens `X` (its `DONE` shows `reused=cursor`). Record `X` byte-for-byte.
2. **Restored continuation**: fresh process (or, for the same-process leg, the same process after `QUIT`
   does not matter - the file is the carrier): `LOAD <s>`, `STATE` (expect `loaded=true`, same `pos`,
   `file_bytes` = file size); `GEN <P2>` (same ids) - the continuation from the restored state. Require the
   `T` sequence to be **byte-identical** to `X`.
3. **Reentrancy**: `SAVE -> LOAD -> SAVE` in one process: the two files must be byte-identical (or the
   difference must be explained and quantified - the expectation is equality on the same build).
4. **Failure legs** (each must `ERR` and leave the next `GEN` working): a path outside
   `STRATA_QWEN35_SESSION_ROOTS` when set, an unreadable path, a truncated file, a foreign file (e.g. the
   GGUF or random bytes), and - if feasible - a state exported from a different context shape; after each,
   `STATE` must still report the previous session and a following plain `GEN` must run normally.
5. **Isolation control**: without any `SAVE`/`LOAD`, two consecutive `GEN`s of the same prompt must still be
   byte-identical to the P2-2-recorded behavior (`reused=0`, fresh state) - the P2-5 change must not disturb
   the verified phase-1/2 semantics.

Pass criteria: step 1-3 token streams byte-equal; step 4 all `ERR` + service continues; step 5 unchanged
from the recorded runs. Evidence (raw transcripts, state file SHAs and sizes, STATE lines, lease audit) goes
to the P2-6 task directory in the research repo; this document is updated with the outcome.

## 11. Parity - P2-6 result (executed 2026-10-15, this machine)

**Format note**: P2-6 ran on the **pre-P2R-4a raw-blob build** (`6dee0c66…` does not write or read
containers); the state files it validated are **legacy raw blobs** under the current format. The
container's own real-engine matrix (corrupt/foreign/wrong-model/wrong-version/repeated-restore/isolation)
is **P2R-4c** and had not run when this section was written (it ran afterwards - section 13).

**PASS.** Exe `6dee0c66c3e87c3020c6ddde3c9ebaa40bed3369745e8e1dbfb0a7ac963ad90a` (139,253,760 B), context 8192,
neutral model + pack; three strictly serial `gpu-lock` leases (one process each, `--stdin-file`, all exit 0,
no cleanup fence left). State blob: 145,215,172 B at cursor `pos = 3870` (saved after a 4-token `GEN` on the
3869-id neutral prompt; the last sampled token - EOS - is not fed into the state, so the continuation prompt
starts the next inputs at the cursor). Evidence (raw transcripts, state files, lease audit):
`results/opencode-execution/strata-unified-p2-6-save-load-parity/`.

| leg | sequence | ids (per-id compare) | result |
|---|---|---|---|
| N1 same-process | `GEN 4 P1` -> `SAVE s1` -> `GEN 8 P2` (A1) -> `LOAD s1` -> `SAVE s1b` -> `GEN 8 P2` (B1) | A1 `[18,248046]` vs B1 `[18,248046]` | equal; first divergence none |
| N2 cross-process | fresh process: `LOAD s1` -> `GEN 8 P2` (C1) | C1 vs A1 | equal |
| N3 no half-restore | `LOAD s1` -> `GEN 8 P2` (D0); `LOAD <missing path>` -> `ERR` -> `GEN` (D1); `LOAD <truncated file>` -> `ERR` -> `GEN` (D2) | D1/D2 vs D0 | equal; the `STATE` line is character-identical across each failed `LOAD` |
| reentrancy | `SAVE s1` -> `LOAD s1` -> `SAVE s1b` | s1.bin vs s1b.bin, sha256 `7d34f3b933d6a92cce9d45d254c832c84e7a03a79a6107923a6bd65844c368414` | byte-identical (145,215,172 B each) |

Raw lines (originals in the evidence directory, `nativeN/protocol-lines.txt`):
`SAVED 145215172 3870 116.7` / `LOADED 145215172 3870 709.8` /
`DONE 2 4 102.3 54.2 stop 0 0 3870 …` (A1) / `DONE 2 4 105.0 49.2 stop 0 0 3870 …` (B1) /
`DONE 2 4 807.0 118.1 stop 0 0 3870 …` (C1) / `ERR session: cannot open '…' for reading` /
`ERR session: the import failed (incompatible model/context or corrupt state); the live session is unchanged`.
Every continuation's `DONE` reports `reused = 3870` (the carried cursor) and `finish = stop`; `18` = "2",
`248046` = EOS.

Unexplained items / limits (as executed): every continuation ended at EOS after 2 tokens (the neutral
prompt's recorded behavior), so each parity leg compares the same 2 ids (added bit-exact evidence: the
reentrancy file equality above and the identical `reused` bookkeeping); P2 was a fixed 4-id prompt because a
preregistered `--stdin-file` cannot carry a runtime-dependent "last generated token"; verification covers
this build, this machine and context 8192 only - **no cross-machine guarantee**; long-run stability (many
save/load cycles, long sequences, repeated continuations) is not measured here. `state_export` remains `true`.

## 12. Implementation pointers

- Commands and session bookkeeping: `backends/qwen35_llama/src/program/qwen35_serve.cpp` (`SESSION` helpers:
  `SessionBook`, `state_cursor`, `state_blob_size`, `session_path_allowed`, `read_file_bytes`,
  `write_file_atomic`; the `STATE`/`SAVE`/`LOAD` branches in the command loop; session-aware `GEN`).
- P2R-4a container additions (same file): the `kContainer*` constants, the LE scalar helpers
  (`put_u32_le`/`put_u64_le`/`get_u32_le`/`get_u64_le`),
  `crc32_ieee`, the self-contained `Sha256`, `compute_model_fingerprint` + `fp_prefix_hex` (computed once
  after the model load, cached), `write_container_header`, `SessionBook::payload_bytes`, the container
  build inside `SAVE` and the fixed-order validation inside `LOAD`.
- Blob write/read: llama's own API - `llama_state_seq_get_size/get_data/set_data` ->
  `vendor/llama.cpp/src/llama-context.cpp` (`state_seq_*`) -> `llama-memory-hybrid.cpp` ->
  `llama-memory-recurrent.cpp` / `llama-kv-cache.cpp`.
- P2R-3a additions (same file): the `keep=`/`cont=` parse keys, the state-keeping `SCORE` branch and the
  `cont` gate in the `GEN` branch, `SessionBook::from_score_keep`, and the two CAPS fields
  (`state_keep_score` / `session_continue`).
- Protocol summary: `docs/backends/qwen35_llama_serve_protocol.md`; capability table:
  `docs/backends/qwen35_llama_capabilities_and_score.md`.

## 13. P2R-4c - real-engine hardening matrix (executed 2026-10-15, this machine)

**Status: N1 PASS and N2 PASS** (all preregistered items), **2 natives** on exe
`b07b2c46a757161d6be236abccef9c6702ea73836eb66860bc6ebc9b0f64115c` (139,267,072 B, re-verified before
launch), ctx4096 / 8 threads / 3200 slots / ubatch4096, model+pack as P2-2/P2-6. One `gpu-lock` lease per
native, strictly serial (rc=0 both, 32.4 s / 30.5 s), no cleanup fence at any point, lock released, no
`strata.exe` residue; session files only under the research-repo evidence `state/` directory (engine paths
through the sanctioned ASCII junction, `STRATA_QWEN35_SESSION_ROOTS` set to it and recorded). Evidence:
`results/opencode-execution/strata-unified-p2r4c-state-matrix/` (both stdins verbatim, full child logs,
per-request parsed JSON, `client/p2r4c-compare.json`, per-rejection STATE pairs, state1/state2 and the six
corrupt variants with SHA256 + exact generation method, pre/post host state).

**Sequence adaptations (preregistered and documented in the evidence `final-report.md`).** A **pure**
`GEN <k> cont=1` as the *first* request after `LOAD` cannot sample: a restored context holds no logits
(this document's own `not_covered: logits`: "*a session can only be continued, not re-sampled at the same
position without a new token*"). The preregistered restore/rejection continuations therefore carry the
fixed 4-id prompt **P4 = the last 4 ids of P_long** (`248068,271,248069,271`; the P2-6 `P2` rule) and are
compared against an **in-process twin run from the same state**; `A` (the warm pure continuation) and the
`R1`/`R2`/`R3` full-prefill references are exactly as preregistered. Raw facts behind the adaptation (all
in the evidence): the preregistered pure `GEN 16 cont=1` right after `SCORE keep=1` reproduces the
P2R-3b-2/P2R-3b-order stream `[90, 198, 1, 78151, …, 92]` (16/16), while any *consecutive* pure
continuation re-samples the pending last token first (the engine keeps the logits of the last fed token in
the live context - there is no logits store in the file).

### N1 - positive path: SAVE/LOAD, 128/256, multi-round, repeated restore, isolation

`state1` = container saved right after the warm 16-token continuation (`SAVED 146055932 3911 348.0`;
on-disk size 146,055,932 B - `SAVED.bytes` equals the file size). `state2` = independent short-prompt
container (`SAVED 66336380 23 173.5`; 66,336,380 B on disk).

| item (preregistered) | result | raw anchor |
|---|---|---|
| `A == R1` 16/16 incl. finish (`length`) | **PASS** | both `[90,198,1,78151,763,220,15,11,198,1,9676,33805,763,867,198,92]`; `DONE 16 0 0.0 … length … reused=3896` vs `DONE 16 3896 2743.7 … length … reused=0` |
| `C == R2` (128-cap request) 17/17 incl. EOS | **PASS** | both `A + [248046]`, `finish=stop`; same 17/17 as P2R-3b-2 |
| `D == R3` (256-cap request) 7/7 incl. EOG | **PASS** | both `[271,248068,271,248069,271,9405,248044]`, `finish=stop` |
| restore parity `P == B == E` 16/16 incl. finish | **PASS** (adapted, see above) | `[90,198,220,328,78151,763,220,15,11,198,220,328,9676,33805,763,867]`; `B` after **two consecutive** `LOAD state1`; `E` after the `P_short` episode + `LOAD state1` |
| multi-round after `SAVE` (`GEN 32 cont=1`) | **PASS** (executed; evidence) | `[867,198,92,248046]`, `finish=stop`, `reused=3930` |
| repeated restore: `LOADED` fields equal | **PASS** | `LOADED 146055932 3911 772.3` then `LOADED 146055932 3911 436.0` (bytes/pos equal; ms differs) |
| `STATE.pos` after each `LOAD` identical | **PASS** | both STATE lines byte-identical: `loaded=true`, `pos=3911`, `path=state1.bin`, `file_bytes=146055932`, `payload_bytes=146055836`, `origin=load` |
| `STATE` after `SAVE` fields | **PASS** | `session=true`, `container_version=1`, `model_fp_prefix=0d4160ef57f03a80`, `payload_bytes = file_bytes - 96`; `origin="save"` (the arming command; the task text's `score-keep` literal is the one delta - see open items), `pos=3911` |
| whole N1 run: 0 `ERR` | **PASS** | 25/25 requests answered, no `ERR` line in the child log |

### N2 - rejection matrix: six corrupt/foreign containers, each rejected, session never touched

The six variants were derived **host-side from the `state1` container** (original SHA256
`f5fe4123de36b5f872a393003c496242d270f398de8c61917ced004ef468e986`, unchanged before/after N2; variant
SHA256 and generation method in the evidence `raw/bad-files.json`):

| # | variant | engine answer (verbatim, path shortened) |
|---|---|---|
| 1 | payload byte flipped (middle) | `ERR session: '…bad-payload-flip.bin' payload CRC32 mismatch (container 19cbc5d5, computed c179c6c0)` |
| 2 | version field := 2 | `ERR session: '…bad-version-bump.bin' has container version 2; this build reads version 1` |
| 3 | last 4096 B dropped | `ERR session: '…bad-truncated.bin' size mismatch (the file is 146051836 bytes, the container declares 146055932)` |
| 4 | model_fp first byte flipped | `ERR session: '…bad-fp-wrong.bin' model fingerprint mismatch (container 0c4160ef57f03a80..., this engine 0d4160ef57f03a80...)` |
| 5 | header stripped (bare payload) | `ERR session: '…bad-legacy-raw.bin' is a legacy raw state blob (no STRATAST container); re-SAVE it with this build to migrate` |
| 6 | magic first byte flipped | `ERR session: '…bad-magic-bad.bin' is not a STRATAST session container (bad magic)` |

Six distinct classes, each its own text, in the fixed validation order (size -> magic -> version ->
header -> vocab -> ctx -> pos -> CRC -> fp); **every rejection left the live session untouched**:

- rejection #1 happened directly after the `SCORE keep=1` + warm-continuation session (origin
  `score-keep`, `pos=3911`, no file): the `STATE` line before and after it is **byte-identical**;
- for rejections #2..#6 the session was anchored on a successful `LOAD state1` and rewound to it before
  each rejection: every pre/post `STATE` pair is **byte-identical** (`loaded=true`, `origin=load`,
  `path=state1.bin`, `pos=3911`);
- after every rejection the same continuation `GEN 16 cont=1 P4` reproduced the reference `D0`
  **token-exact 16/16** (`reused=3911`, `prompt_read=4`, `finish=length`) - D1..D5 all equal D0;
- `A' == R'` 16/16 (the N2-internal warm-vs-full-prefill pair); `A'` is the same 16-token stream as N1's
  `A` (cross-native reference only, not a judged identity);
- **cross-process restore of the container is covered**: `state1` was written by native1 and validated +
  continued by native2 (all N2 `LOAD state1` anchors and the D0..D5 continuations).

### Unexplained / open items (as executed)

- **Magic literal (found here, resolved by P2R-4d - section 14)**: the P2R-4c build wrote `SRATAST\0`
  (7 ASCII + NUL, hex `53 52 41 54 41 53 54 00`) while section 1b and every relevant `ERR` text spell
  `STRATAST` (8 ASCII bytes, no NUL) - the constant-vs-document disagreement
  (`qwen35_serve.cpp` `kContainerMagic = {'S','R','A','T','A','S','T'}`) was left untouched in P2R-4c
  pending a supervisor decision. Decision: **fix the code literal to match the document, version stays
  1**. P2R-4d changed the initializer to `{'S','T','R','A','T','A','S','T'}` (one-line diff) and proved
  on the real engine that new-magic containers round-trip while an old-literal file is refused with the
  bad-magic `ERR`.
- `SAVE -> LOAD -> SAVE` container **byte-identity** (section 8): P2R-4c did not run it (no second
  `SAVE` in its preregistered shape). **Closed by P2R-4d - section 14**: `s1` (live save), `s2` (save
  right after the first `LOAD`) and `s3` (save after the second `LOAD`) are byte-identical
  (SHA256 `8d157ce3…b45135`, 145,748,372 B each, whole-file compare with zero divergences).
- `STATE.origin` after `SAVE` is `"save"` (the last arming command), not the task text's literal
  `score-keep`; this matches section 1's field contract and the P2-6 record - recorded as a
  task-text-vs-spec delta, not a defect.
- The preregistered pure `GEN cont=1` legs right after `LOAD` are not executable on this engine (no
  logits in a restored context, section 2 `not_covered`); the adapted legs above cover the same restore
  identity with input-carrying continuations.
- Process note (kept in the evidence, not an engine fact): the two-native batch launched with the
  task-window (task.md mtime + 20 min) already elapsed (`-267 s` / `-302 s` window-left at the two
  native starts, 12:37:33 / 12:38:07 vs mtime 12:13:06) after ~22 min of executor preparation; it was
  executed as one already-started atomic operation with all guards (exe SHA, lock, fence, RAM/GPU,
  process table) re-verified immediately before each launch, and the raw values are recorded in
  `native1/argv-plan.json`, `native2/argv-plan.json` and `client/p2r4c-orch-summary.json`.

## 14. P2R-4d - magic fix + SAVE->LOAD->SAVE byte identity (executed 2026-10-15, this machine)

**Status: M1-M5 PASS.** **1 native** (cap 2; one `gpu-lock` lease; rc=0; 28.1 s; no cleanup fence; lock
released; no `strata.exe` residue) on the **rebuilt exe**
`c3780a6a02be44a739a272c8f1cb39b76d9d92bb4f15d18b12cf026145812b0b` (139,267,072 B; supersedes
`b07b2c46…`, same size). **The only source change is the one-line magic-literal fix**
(`kContainerMagic` = `{'S','R','A','T','A','S','T'}` -> `{'S','T','R','A','T','A','S','T'}`; `git diff`
= 1 insertion / 1 deletion); `version` stays 1, the layout is untouched, no other behaviour changed.
`build.ps1 -SkipConfigure` exit 0; the two existing python suites (unmodified) stayed green: 63
(`test_score_api.py`) + 13 (`test_backend_adapters.py`). Evidence:
`results/opencode-execution/strata-unified-p2r4d-magic-roundtrip/` (stdin verbatim, full child log,
per-request parsed JSON, `client/p2r4d-compare.json` + the corrected `p2r4d-compare-v2.json`, verbatim
`SAVED`/`LOADED` lines, s1/s2/s3 and both pre-made bad files with SHA256, lease audit/result, pre/post
host state).

**Sequence** (task.md section 2 plus read-only `STATE` probes for M4 and the supplemental same-process
twin from the twin note): `SCORE <P_long> topk=5 keep=1` -> `SAVE s1` -> `LOAD s1` -> `SAVE s2` ->
`GEN 16 cont=1 867,198,92,90` (cont#1) -> `LOAD s1` -> `SAVE s3` -> `LOAD legacy-magic.bin` ->
`LOAD crc-bad.bin` -> `GEN 16 cont=1 867,198,92,90` (cont#2) -> `SCORE <P_long> topk=1` -> `GEN 16
<P_long>` (reference R) -> `SCORE <P_long> keep=1` -> `GEN 16 cont=1 867,198,92,90` (twin) -> `QUIT`;
the session cursor is 3896 and every demand (max 3916) is within ctx 4096. The fixed 4 ids are the
task text's `867,198,92,90` (distinct from P2R-4c's P4; same four on all three continuation legs). The
two bad files were **pre-made host-side from the P2R-4c `state1.bin`** (old magic; source SHA
`f5fe4123…`, unchanged): `legacy-magic.bin` = that container verbatim; `crc-bad.bin` = first 8 bytes
`STRATAST` + one payload bit flipped (generation method and predicted CRC in `raw/bad-files.json`).

| item | result | raw anchor |
|---|---|---|
| **M1** new-magic container accepted; old-literal refused with the documented text | **PASS** | `LOAD s1` -> `LOADED 145748372 3896 766.6`; `LOAD legacy-magic.bin` -> `ERR session: '…legacy-magic.bin' is not a STRATAST session container (bad magic)` (the exact section-1b text) |
| **M2** `s1`/`s2`/`s3` byte-identical (hard judgement) | **PASS** | all three 145,748,372 B, SHA256 `8d157ce35eacc2ea4fba3d5f1b806585e8efc9e4c4dec469c9da9d677eb45135`, header magic `STRATAST` (`53 54 52 41 54 41 53 54`), pos 3896; whole-file compare: zero divergences |
| **M3** continuations token-exact, `reused=3896`/`prompt_read=4`, session survives both rejections | **PASS** | cont#1 = cont#2 = twin = `[198,220,328,78151,763,220,15,11,198,220,328,9676,33805,763,867,198]` (16/16, `finish=length`); each `DONE 16 4 … length 0 0 3896 0 0 0 0 0.0 4 0`; `STATE` after each rejection byte-identical to the pre-rejection `STATE` |
| **M4** no side effects | **PASS** | `SAVE` keeps `pos=3896` (STATE `pos` before/after every SAVE equal); after each `LOAD`: `loaded=true`, `pos=3896`, `origin="load"` |
| **M5** discipline | **PASS** | 1 native / 1 lease / rc=0 / no fence / lock `\x00` after / 0 `strata.exe`; `binary_sha256_actual` = declared (`c3780a6a…`); `SAVED.bytes` == on-disk size for s1/s2/s3; the two `LOADED` lines equal in bytes+pos (766.6 vs 651.8 ms - timing field differs) |

**Raw lines (verbatim):**

```text
SCORE keep        DONE 0 3896 4560.7 0.0 score 0 0 0 0 0 0 0 0.0 3896 0
SAVE s1           SAVED 145748372 3896 354.2
LOAD s1 #1        LOADED 145748372 3896 766.6
SAVE s2           SAVED 145748372 3896 471.3
GEN cont#1        DONE 16 4 144.9 1075.4 length 0 0 3896 0 0 0 0 0.0 4 0
LOAD s1 #2        LOADED 145748372 3896 651.8
SAVE s3           SAVED 145748372 3896 443.0
LOAD legacy       ERR session: '…legacy-magic.bin' is not a STRATAST session container (bad magic)
LOAD crc-bad      ERR session: '…crc-bad.bin' payload CRC32 mismatch (container 19cbc5d5, computed c179c6c0)
GEN cont#2        DONE 16 4 79.1 644.2 length 0 0 3896 0 0 0 0 0.0 4 0
SCORE reset       DONE 0 3896 2770.4 0.0 score 0 0 0 0 0 0 0 0.0 3896 0
GEN R             DONE 16 3896 2710.7 497.4 length 0 0 0 0 0 0 0 0.0 3896 0
SCORE keep twin   DONE 0 3896 2774.8 0.0 score 0 0 0 0 0 0 0 0.0 3896 0
GEN cont twin     DONE 16 4 67.0 338.9 length 0 0 3896 0 0 0 0 0.0 4 0
```

**Judgement bookkeeping**: the full-prefill reference `R` (`DONE 16 3896 2710.7 … reused=0`) is
recorded but **not** compared to the cursor continuations (the 4 appended ids move the sampling
position; the supplemental in-process twin from a fresh `SCORE keep=1` is the independent reference) -
per the task's judgement discipline; no run was repeated. `crc-bad.bin` clears size/magic/version/
header/vocab/ctx/pos and fails exactly at the payload CRC32 (predicted computed CRC `c179c6c0` == engine
output). One evidence-harness correction is recorded, not hidden: the first compare JSON compared the
two full `LOADED` lines including the timing ms field; `client/p2r4d_compare_fix.py` re-derived the M5
check from the sealed `native1/requests.json` (bytes+pos equality) into `client/p2r4d-compare-v2.json`
with raw logs and containers untouched.
