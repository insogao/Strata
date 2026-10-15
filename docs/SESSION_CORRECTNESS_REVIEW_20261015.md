# Session correctness follow-up (single GPU, 2026-10-15)

This PR fixes the four findings from the independent review of `de89bf97`. It is
based on the completed P2R supplement `f3610f82`, **not** the older phase-three
branch. Target: `phase2r/review-gaps`. No production/main merge, frozen model/exe
replacement, MTP optimization or default cache change.

Current user scope is **one GPU per laptop/PC**. No multi-GPU implementation or
second-card experiment is required for this increment. Actual hardware evidence
below is Windows / RTX4060 Laptop8GiB; other OS/vendor configurations remain
unverified rather than blocking this machine's scoped correctness review.

## 1. Changes and observable behavior

| Review finding | Change | Verification |
|---|---|---|
| F1: repeated boundary token | Armed GEN commits every emitted token, including the final length token and EOS, before reporting DONE. Isolated GEN keeps its old evaluation count. No pending-token container field is needed. | Real-model single32 == split16+16 == sixteen GEN1 + tail; each single-token cursor advances; EOS committed |
| F2: workflow cross-talk | HTTP keep returns an opaque session id + revision. Every subsequent mutating request must present both; compare, pipe operation and revision publication share one lock. | Independent HTTP callers cannot append/reset/continue another workflow; stale/simultaneous revision replay cannot both succeed |
| F3: illegal probabilities | Both row parsers reject NaN/±Inf/positive logprob, negative ids/positions, duplicates and impossible probability mass (existing 0.001 rounding tolerance). DONE timings/counters/finish and generated ids are validated. Protocol failure quarantines the link. JSON refuses non-finite numbers. | Parser negatives + real pipes/HTTP fault injection produce502/quarantine; later engine requests remain502 |
| F4: integer/context overflow | HTTP positive counts bounded to int32; GEN parser checks strtoll ERANGE and suffixes; native context admission uses guarded subtraction, never unchecked cursor+prompt+max_new addition. | LLONG_MAX, LLONG_MAX+1, long decimal and suffix rejected; real STATE identical before/after |

The native format remains STRATAST v1. Previously exported containers still mean
the cursor/state they actually contain; the engine cannot reconstruct an omitted
pending token from an old file. **LOAD still does not restore logits.** A first
pure continuation after LOAD is rejected; append actual token ids to continue.
This PR does not manufacture logits or relabel that limitation as fixed.

## 2. Stateful HTTP compatibility change

Stateless SCORE/decisions requests on an unowned engine retain their behavior.
Stateful clients must adopt the explicit service contract, advertised as
`service_session_contract="opaque-lease-revision/1"` in capabilities:

```text
POST /v1/score
{"prompt_ids":[1,2,3],"keep":true}
=> ... "session":{"id":"<opaque bearer value>","revision":1}

POST /v1/continue
{"max_new":16,"session":{"id":"<same value>","revision":1}}
=> ... "session":{"id":"<same value>","revision":2}
```

Use the returned revision for the next operation. Missing/wrong id, stale revision
or an expired handle gives **409 before any engine command**. The current owner
must also supply its handle for keep-score/decisions appends and stateless scoring
that resets/releases the session. A successful owned stateless score/decision
returns `session:null`; the next workflow receives a new id. A concurrently busy
pipe still returns503, so a client must not treat503 as proof that its old revision
remains usable after another operation has completed.

STATE is read-only and never discloses the bearer id. It is diagnostic metadata,
not a way to acquire ownership. An engine error/timeout during an owned mutation
makes that workflow uncertain: further keep/continue is refused until its owner
performs a stateless reset-score/decision or restarts the service. The same handle
is retained for recovery; strangers cannot steal the partially advanced state.
Protocol corruption instead kills/quarantines the whole link, as before.

No lease expiry/background scheduler is introduced. If a client abandons its
handle, restart this service to recover it. This is a **single-session workflow
guard**, not multi-sequence batching, durable authentication or user authorization;
keep deployment on the documented local/protected interface. The direct engine
line protocol remains a single-owner pipe without HTTP bearer fields.

## 3. Verification and independently checkable evidence

- **100 CPU tests PASS** (existing backend/service tests plus10 new hardening tests):
  `python -B -m unittest discover -s backends/qwen35_llama/tools -p "test_*.py" -v`.
  Existing single-owner continuation cases now explicitly carry the issued handle;
  new adversarial tests use independent raw HTTP requests and assert no pipe send
  on ownership rejection. Stub tests alone are not native numerical evidence.
- Fresh Windows CUDA12.8.1 / MSVC build:384 actions, independent build directory,
  executable139,308,544B, SHA256
  `ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619`.
  No CUDA/HIP kernels/vendor files changed. HIP/SYCL/Linux were not built on this
  Windows NVIDIA host; no portable-backend qualification is claimed.
- Three strictly serial native processes, one canonical gpu-lock lease each,
  all exit0 and exact binary SHA binding, all children exited/no retained fence.
  Production model registered SHA `66a3ca88…` read-only, no full model rehash;
  same3896-token fixture SHA `4e6b2529…`,ctx4096/t8/3200slots/ubatch4096.

| Case | Actual outcome |
|---|---|
| Old `c3780a6a…` control | Single output17 tokens; split output18, repeating id92; old cursor also omits the last emitted token. Expected regression reproduced. |
| Fixed core | Single and split both17 identical IDs, real EOS; cursor3913=3896+17. Sixteen GEN1 boundaries and final cursor pass. Four length refusals leave STATE unchanged.30 core checks pass. |
| Fixed state follow-up | SAVE→LOAD→SAVE files each146,076,436B, byte-identical; SHA256 both `a196f5eeff5fe5fffa3cf7496f6b622e2c934e776f29765662754622fde8a7b6`. Live/restored4-id append outputs agree; LOAD pure refusal leaves STATE unchanged.8 checks pass. |

Review bundle (no model/container payloads):
[`session_correctness_20261015.json`](../backends/qwen35_llama/tests/evidence/session_correctness_20261015.json).
It contains exact token arrays, SHA identities, core/state gates and lease receipts.
The adjacent `tests/evidence/session_correctness_20261015/` directory contains
byte-exact old/fixed/state raw child logs (their SHA matches that bundle) and
review-only response-index plans. Those plans withhold the full policy prompt,
identify it by SHA, and cannot be used as executable stdin. A reviewer can call
the verifier's `parse(plan, raw_text)` to regroup responses and independently
recompute tokens/cursors/errors. State-byte comparison still requires the local
container files; those145MB binary artifacts are not included in Git.
Full raw/commands/failed attempts remain under
`D:/work/Strata魔改/results/code-review/session-correctness-20261015/`.
The CPU-only `backends/qwen35_llama/tools/verify_session_native.py` recomputes
responses/gates from `--plan`, `--raw`, `--out`; the state arm also checks container
bytes on disk. No engine is started by this verifier.

## 4. Failures retained and repair budgets

Nothing was silently retried or counted as a fully green run:

1. First build invocation failed during configuration: ASCII batch writing mangled
   a Chinese CUDA toolkit path. One registered correction used the existing ASCII
   junction. No compiler/model run occurred in that first invocation.
2. The second invocation completed384 actions/link, then its wrapper failed because
   Windows PowerShell lacked Get-FileHash. `build.ps1` now streams .NET SHA256.
   One registered post-check invocation returned0 with **ninja: no work to do**;
   no extra compilation. All three wrapper logs are retained.
3. The first fixed native passed F1/F4 core checks but my SAVE/LOAD test plan used
   relative paths, correctly refused by the absolute-only session contract. Its
   full verdict remains **FAIL for the state harness**, not retrospectively green.
   The last registered native was a targeted state-only test using absolute ASCII
   paths under the same root allow-list. No implementation/runtime retry or extra
   build was used to hide that planning error. All raw/stdin/manifests remain.
4. The first new CPU assertion confused request-local LOGPROBS.pos with absolute
   STATE.pos. It was corrected to assert both actual semantics; implementation was
   not changed to satisfy it. The failed log is retained.

## 5. Limits and integration

This closes the scoped review regressions on one fixture/machine. It does not
certify business risk labels, arbitrary prompts/sampling, MTP state transactions,
all corrupted-container cases, latency/Prefill regressions or product speed.
No cross-state-class logprob comparison is used as numerical evidence. Historical
risk2/label3 and the broader product acceptance gates remain unchanged.

Review this follow-up against PR #3's head; do not discard the earlier supplement.
Stage-three tooling PR #4 is parallel work and is not inherited here. Production
switches and merges remain the user's decision after review and relevant quality/
Prefill acceptance. No OpenCode agent, multi-GPU work or automation was started.
