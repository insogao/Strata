# qwen35_llama single-session state export/import (P2-5)

**Status**: implemented in `src/program/qwen35_serve.cpp` (P2-5, 0-GPU task). This is the **single-session
state boundary** for the serve engine: one explicit export/import of the CURRENT session state, reusing
llama's own sequence-state serialization (`llama_state_seq_get_size/get_data/set_data`) - **no new
serialization format**. Functional/GPU verification (parity) ran as **P2-6** on 2026-10-15 and passed on this
machine/build (same-process and cross-process continuations byte-identical; failure legs left the live
session untouched) - results, raw lines and limits: section 11.

Scope rules this doc obeys: nothing here claims cross-machine portability; cross-process continuation
restores are verified for the same machine and build only (P2-6 N1/N2, section 11).

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
| `SAVE` | `SAVED <bytes> <pos> <ms>` - state blob size, sequence-0 cursor (occupied positions), elapsed ms | `ERR <reason>` |
| `LOAD` | `LOADED <bytes> <pos> <ms>` - the same fields, measured on the imported state | `ERR <reason>` |
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
 "session":false,"origin":null,"file_bytes":null,"max_context":4096,
 "not_covered":["logits","sampler_rng_state","expert_cache","bridge_counters","rollback_history"]}
```

| Field | Meaning |
|---|---|
| `loaded` | `true` when the ACTIVE session came from a `LOAD` (imported file); `false` after a plain `SAVE`, before any session command, or after the session was cleared |
| `session` | `true` when a session is armed at all (`SAVE` or `LOAD`); this is the fact that makes the next `GEN` continue instead of resetting |
| `origin` | `"save"` / `"load"` / `null` - which command armed the active session |
| `pos` | current sequence-0 cursor: last occupied position + 1 (`0` when the sequence is empty). This is the position a continuation appends at |
| `bytes` | byte size of the state **as it is now** (what `SAVE` would write at this moment). On a fresh/cleared engine this is the non-zero size of the empty state |
| `path` | file of the active session (`null` when none); `file_bytes` its byte size at the time of the last successful `SAVE`/`LOAD` (may differ from `bytes` once the session has advanced) |
| `covers` | what the exported bytes cover - see section 2; this list is the implementation's own truth table |
| `not_covered` | state a restored session does NOT carry (see section 2) |
| `max_context` | the live `llama_n_ctx` (the same value `READY` advertises) |

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
   (`bytes`/`pos`), while `path`/`file_bytes` still describe the last file written/read.
5. The session is cleared (back to full isolation) by:
   - a `SCORE` request, which performs its documented semantics-(b) reset - see section 6;
   - process exit (nothing is implied to persist);
   - there is no separate "close session" command in this phase.
6. A failed `SAVE`/`LOAD` does **not** change the arming state: the previous session keeps working.
7. `CAPS`, `STOP`, `QUIT` and stdin EOF are unchanged by sessions.

The exported bytes are the state of the CURRENT sequence, whatever produced it: after a `GEN` it is that
request's final state (prompt + decoded tokens); after a `SCORE` it is the cleared state (see section 6);
right after `LOAD` it is the imported state. `SAVE` never fabricates or normalizes anything: it serializes
what the context holds.

## 4. Failure semantics (no half-restore)

`SAVE`:

- path gate failures (`ERR session: ...`), state size/read failures, and file failures all answer `ERR` and
  leave the session (and any previously armed state) exactly as it was;
- the file write is atomic in effect: bytes go to `<path>.tmp`, then `MoveFileExA(..., REPLACE_EXISTING)`
  (POSIX `rename`) replaces `<path>`. On any failure the `.tmp` is removed and an existing `<path>` is left
  untouched - a failed save never truncates the previous file;
- `SAVE` replaces an existing file (documented, not a refusal).

`LOAD`:

- path/capability/format failures happen before any engine change: bad path (`ERR session: ...`), unreadable
  file, file smaller than the 8-byte blob header, file larger than the 1 GiB cap, wrong magic (not a
  sequence-state blob), header `seq_id != 0` (this engine restores sequence 0 only);
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

- **Same build, same model, same context parameters** is the only supported pairing. The blob has a magic
  and a sequence id - **no version field, no model name, no checksum**. A blob from a different
  build/architecture/layout may be rejected by llama's structural validation (`ERR`), but a blob from a
  same-shaped, different-weight model would not be detected: it would restore as valid numbers of the wrong
  distribution. Do not treat session files as portable artifacts.
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
after it, so the next request starts fresh. Consequences for P2-5, stated plainly:

- `SCORE` **clears an armed session**: the reset is a real reset, so the engine also clears the session
  bookkeeping (`STATE`: `session=false`, `loaded=false`, `origin=null`, `path=null`, `pos=0`, `bytes` = the
  empty-state size). A `GEN` after a `SCORE` behaves exactly like a fresh isolated request, even if a session
  was active before.
- `SAVE` **before** a `SCORE` stays valid: the file is on disk and can be `LOAD`ed later (that is the point
  of an explicit checkpoint). `SAVE` **after** a `SCORE` writes the cleared (empty) state - a valid,
  importable file that carries nothing; `LOAD`ing it arms a session whose cursor is 0, which behaves like a
  fresh request.
- The `SCORE` path itself is unchanged: it never reads or writes session files, and its `LOGPROBS`/`DONE`
  output is identical with or without an armed session.

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
tensor rows; the import packs cells from head 0). This expectation got its empirical check in P2-6: the two
files were byte-identical (`sha256 7d34f3b9…368414`, section 11).

## 9. Not implemented / not claimed

- the official engine's disk-session grammar (`RESTORE <path>`, `SAVED <tokens> <bytes> <ms>`,
  `SERR <kind> <published>`, `SESSION`/`SWAIT` progress lines, `FATAL`) - this engine's commands are
  `SAVE`/`LOAD`/`STATE` with the answers above;
- token history: the file stores state only, not the token list (llama's `state_seq` blob has no tokens).
  A continuation works with token ids the caller supplies; there is no "resume the exact prompt" helper;
- multiple sessions, session ids, listing, deletion, quotas, compression, encryption, checksums beyond the
  blob's own magic, fsync/folder-flush guarantees beyond the replace;
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
- Blob write/read: llama's own API - `llama_state_seq_get_size/get_data/set_data` ->
  `vendor/llama.cpp/src/llama-context.cpp` (`state_seq_*`) -> `llama-memory-hybrid.cpp` ->
  `llama-memory-recurrent.cpp` / `llama-kv-cache.cpp`.
- Protocol summary: `docs/backends/qwen35_llama_serve_protocol.md`; capability table:
  `docs/backends/qwen35_llama_capabilities_and_score.md`.
