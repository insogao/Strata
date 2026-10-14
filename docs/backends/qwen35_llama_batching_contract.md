# Multi-sequence batching capability contract (version 1)

Phase-2 closing item **P2-7** (branch `phase2/backend-interface`). This document is a
**contract only**: multi-sequence batching is **not implemented** in this phase, on either
backend. The capability tables served by `backends/qwen35_llama/tools/score_api.py` report
`batching=false` on both backends and carry `batching_contract="1"` - the version of this
text. Nothing here claims concurrency, and no code path in this phase admits more than one
in-flight request.

Scope: this contract is written for the framework layer (`tools/backend_adapters.py`,
`tools/score_api.py`) and applies to both selectable backends, `qwen35_llama` and
`official_q36` (see `docs/backends/qwen35_llama_backend_adapters.md`).

## 1. Current facts (why `batching=false` is the honest value)

- **`qwen35_llama`**: the serve engine never prints `INFO batch_slots=N`; requests are
  FIFO-serialized one at a time and `BGEN` is answered `ERR` (phase-1/2 contract,
  `docs/backends/qwen35_llama_serve_protocol.md` "Not implemented" and
  `docs/backends/qwen35_llama_capabilities_and_score.md`). The service additionally answers
  a second concurrent request with `503` (no queue) - the single-request discipline is
  enforced twice, deliberately.
- **`official_q36`**: the official server enables its batch path **only when the engine
  reports `INFO batch_slots=N`** (`serve/server.py`: `self.batch = int(self.info.get(
  "batch_slots") or 0)`); with no such report it stays on the single-request FIFO path. The
  surface this framework drives (`READY/GEN/PP/T/DONE/STOP`) contains no slot commands, and
  the adapter does not drive or claim them. `batching=false` here means "not driven and not
  claimed by this adapter", exactly as with `qwen35_llama`.
- Both capability tables therefore report `batching=false` **plus** `batching_contract="1"`.
  The version ties the false value to this contract text - it is a state, not a claim that
  the engines can never batch.

## 2. Contract (what a future implementation must satisfy before `batching=true`)

A future implementation may flip `batching=true` only when **every** item below is
implemented, verified on the target machine/build, and reported truthfully. Each item is
per sequence slot.

1. **Multi-sequence state isolation.** Every slot owns its own KV cells, GDN conv/SSM
   planes, positions and (if armed) session cursor. No cross-slot state is readable or
   reusable; slot state is never observable from another request. Prefix reuse, if any, is
   explicit and opt-in, never a side effect of co-residency.
2. **Admission and resource budget.** A slot is admitted only after a worst-case footprint
   check against the VRAM/context/expert-cache budget (section 3). No overcommit; admission
   failure is an explicit refusal (e.g. `503`), never a silent downgrade of another request
   or an unbounded run.
3. **Cancel and rollback semantics.** `STOP`/cancel names its slot and affects only that
   slot. After a cancel in prefill or decode, the slot's state is rolled back to a
   self-consistent point; the next request on that slot sees no dirty state. A cancelled
   slot's accounting (`DONE ... cancel`) is per request, not per batch.
4. **Error isolation.** An engine `ERR`, a NaN in head logits, an OOM or a dead slot must
   not corrupt other slots; the failure is surfaced on its own request, the slot is
   quarantined/released explicitly, and the remaining slots continue. The existing
   NaN/`nonfinite`, OOM and transaction-rollback gates are not relaxed for batch mode.
5. **MTP transaction interaction.** A draft/verify/commit transaction is atomic per slot:
   batch scheduling must not interleave one slot's transaction steps with another slot's
   steps, and a verify window belongs to exactly one slot. MTP gates (`STRATA_QWEN35_MTP*`)
   remain default-off; the K1 quality gate (2-true-vs-fixed-label-3, unclosed) still blocks
   any business claim, batch or not.
6. **Observation and event surface.** Per-slot accounting in every `DONE` (produced,
   prompt/decode ms, `reused`, accepted/offered where MTP applies), slot-identified event
   lines the driver can attribute, and a truthful `INFO batch_slots=N` / capability report
   that the server can trust. The capability table and this document are updated in the
   same change as the implementation, with the contract version bumped.
7. **Refusal discipline stays.** Endpoints unsupported by a backend still answer explicit
   errors; batching must not create a path where `/v1/score` (or anything else) silently
   changes meaning.

## 3. Admission policy on the 8 GiB target (conservative)

Target context: 4060 Laptop, 8 GiB VRAM, this machine/build only. Numbers must be measured
here; nothing is converted from other GPUs or quantizations.

- **Budget base**: free VRAM *after* model weights, the context/workspace allocation and
  the expert cache are resident (the expert-cache slots of `--qwen35-cache-slots` are a
  different resource from sequence slots, but both consume VRAM and are budgeted together).
- **Per-slot worst case** = KV cells for the slot's maximum context + GDN conv/SSM planes +
  worst-case transient decode workspace. Measured on this machine/build; no estimate is
  acceptable for admission.
- **Admissible slots** = `min(requested, floor((free_vram - reserve) / per_slot_worst_case))`
  with a positive reserve. `reserve` protects the single-request path and the expert-cache
  working set.
- **Re-evaluate at request boundaries**: when free VRAM drops below the reserve, no new
  slot is admitted; already-running slots are unaffected (no preemption, no degradation).
- **Refuse, never degrade**: a request that cannot be admitted gets an explicit refusal
  (`503`-style) rather than sharing a slot, truncating context, or running unbounded.

## 4. What this phase does not do

- No concurrency, no slot commands, no `BGEN`, no `n_seq_max` change, no scheduler.
- No capability flip: `batching=false` + `batching_contract="1"` on both backends.
- No performance claim of any kind from this contract.

Revision rule: any future batching implementation must bump the contract version, update
this document (facts + policy as measured), and update both capability tables in the same
change; the version string in the tables identifies which text governs the `false` value.
