# External review follow-up: real long output and acceptance scope

This supplements [PR #5](https://github.com/insogao/Strata/pull/5) after the
independent PR #1–#5 review. It adds long-output evidence and CPU-only service
protocol guards, verifier checks, tests and documentation. No new engine build,
kernel/vendor edits, model replacement, default changes or production/main merge
are included. The long native evidence used `1e9e5780`; the P1 increment starts
from `d428b83f` and does not alter that native executable or its raw records.

## Acceptance boundaries

Natural-language answers from independent requests need not be identical for a
product to be correct. These tests use temperature0/greedy decoding to make exact
token replay a useful **numerical diagnostic**, not a release requirement.
Floating-point execution differences can change near-tied choices and compound
over a long answer. A different token stream alone does not establish damaged
KV/GDN state, a broken segment boundary or a regression introduced by PR #5.
Conversely, allowing different text does not establish business quality.

The evidence verifier reports structural/control checks, exact replay diagnostics,
and the original conjunction of all registered checks separately. The two original
strict verdict files remain failed; no assertion was removed from those records.
CLI exit status uses structural checks by default; `--require-exact-replay` opts
into the stricter diagnostic. Neither exit status means product acceptance.

### Scoped PASS and the two P1 follow-ups

The external re-review grants **Scoped PASS** for PR #5's token commits, ownership
and revision isolation, actual long-output coverage, cancellation and state
recovery evidence. This is not a Strata production-release decision. Business
accuracy, service integration, stability and formal performance keep their own
acceptance gates. Exact replay is not a prerequisite for phase-three single-GPU
optimization; no multi-card experiment or broad replay study is added here.

The CPU-only P1 increment strengthens two boundaries:

- `verify_session_long.py` now builds `structural_checks` and
  `exact_replay_diagnostics` independently. Actual128-token SAVE prefixes, the
  112-token tail/128-token total,128-token post-error output and120-token
  post-cancel output have explicit length checks; expected `length` finishes
  remain required. Separate comparisons only describe token replay. The legacy
  combined `gates` view remains available. Restored/live output equality is also
  numerical diagnostic; actual counts, cursors and container checks remain
  structural. A shorter actual stream with a matching lowered DONE counter
  still fails structural acceptance. Historical raw logs, summary snapshots and
  failed verdicts are unchanged; the current verifier adds checks rather than
  rewriting those older reports.
- `EngineLink.generate_continue()` accepts only `length`, `stop` or `cancel`,
  checks DONE.produced equals actual T rows and never exceeds max_new, requires
  `length` to reach max_new, and requires EOS `stop` to include a token. An extra
  T beyond max_new is rejected immediately without waiting for DONE. SCORE's
  generic `score` finish stays valid for SCORE. Violations use the existing
  EngineProtocolError →502/quarantine path; legal early EOS and cancellation
  (including cancellation with zero tokens) remain supported.

Fault injection covers operation-specific illegal finishes, count overflow,
short length completion and empty EOS through real stub pipes and localhost
HTTP, including refusal to send further engine commands after quarantine. No
native source, executable, model, defaults or main branch change in this P1
increment; it needs no rebuild or new GPU measurement.

Validation: **17 targeted tests pass in2.982s; the complete117-test CPU suite
passes in55.140s**. Eight new tests extend the prior109-test suite. They include
six shorter-output mutations with internally consistent DONE counts, separate
structural/diagnostic membership, valid length/EOS/cancel boundaries, count and
finish refusals, and five real-pipe/HTTP quarantine cases. Logs and preregistration
are retained at `D:/work/Strata魔改/results/code-review/session-p1-20261015/`.

## Real-model trajectory coverage

Windows / RTX4060 Laptop8GiB; registered production GGUF `66a3ca88…` (not rehashed),
fixed executable `ea06d394…`; **3200slots**,t8,ubatch4096. A3817-token risk-explanation
fixture uses ctx4608 because3817+512=4329. This is a legal long-output test; it
does not replace the frozen Prefill c4096/out1 baseline. Greedy generation is
natural: no EOS masking, synthetic draft or forced long completion.

Two direct engine processes ran through canonical gpu-lock, one lease each,
exit0 in105.435s and120.099s, with confirmed empty child jobs and no retained fence.
The second suite added an actual512-token decode warmup; the batch window was
not reset. All subsequent steps remained bounded and separately registered.

| Check | Actual result |
|---|---|
| Length coverage | Each full reference actually emits512 tokens with `finish=length`. Separate32/64-token segments reach128/256/512 actual tokens. |
| Commit cursor | Full512 cursor4329; segment and16 single-token boundaries advance by emitted counts; pure continuation reports zero prompt reread. |
| Real cancellation | STOP is sent through an interactive stdin pipe after observing8 T rows. Both runs finish `cancel` with8 tokens, cursor3825; continuation proceeds from that cursor. |
| Error handling | Oversized decimal GEN refuses; before/after STATE is identical. Fresh request after cancellation/error runs normally. |
| SAVE/LOAD | After128 actual tokens, two LOAD/SAVE roundtrips succeed. Second suite's three146,753,068B containers are byte-identical, SHA256 `1029ebb45902b31dbb47f6683b4081ed98e83704493072c929a44c3fab50acba`. |
| LOAD limitation | Pure GEN after LOAD is still refused because logits are not serialized. Live/restored4-id seeded append produces matching32 IDs/cursors. This is a state diagnostic, not natural pure LOAD continuation. |
| Real EOS | Separate3896-token short fixture emits17 IDs including actual EOS, `finish=stop`; EOS is committed. |
| Exact replay diagnostics | Seven prefix comparisons differ in each suite. Original strict94-check and follow-up95-check conjunctions therefore remain false. Structural/control subsets pass. |

In the first suite, full512 versus later full128 differs at zero-based token42.
In the second, its two repeated full512 runs differ at309, and its full512 versus
later full128 differs at65. Subsequent full128, segmented128 and single-token128
paths agree with one another in each process. The two process references also
differ. A decode warmup therefore does not close this numerical reproducibility
question. It is **unresolved**, not proof of random sampling (temperature is0),
not automatically a state corruption finding, and not a newly certified quality
result. The historical cold/warm probability caveat also remains open.

## Bounded long-output timing control

Four additional native processes ran **old/fixed/fixed/old**, one exclusive lease
per process. Each process first performed an unmeasured actual512-token warmup,
then two measured512-token requests, each after reset+full SCORE on the same
3817-token input. All twelve generation legs actually finished `length` at512;
all four processes exited0 with confirmed cleanup. Engine/model/configuration
identities are in the committed receipts. No builds or CPU test suite ran during
these timing processes. There was no automatic native retry.

| Process | Executable | Measured rq1 decode ms | Measured rq2 decode ms |
|---|---|---:|---:|
| A1 | old `c3780a6a…` | 15617.5 | 15591.3 |
| B1 | fixed `ea06d394…` | 15152.2 | 15467.1 |
| B2 | fixed `ea06d394…` | 16074.1 | 16139.9 |
| A2 | old `c3780a6a…` | 15734.1 | 16152.6 |

Median complete decode window: old15675.8ms, fixed15770.6ms, **+0.60% time**.
The corresponding **512/decode_seconds aggregate return rates** are32.662 and
32.465 tokens/s. These are **not** `(n-1)/(last-first)` post-flush TPS, TPOT,
HTTP latency or JSON-visible E2E. File-reader timestamps are diagnostic only.
The observed ranges overlap (old15591.3–16152.6ms; fixed15152.2–16139.9ms).
Four requests per arm come from only two processes per arm, not four independent
hardware trials. No steady-state, reliable p95 or statistically established
speedup/slowdown is claimed.

PR #5 intentionally evaluates512 committed tokens on the armed path where the
old implementation evaluated511. The measured windows include that final commit.
The eight measured output streams have five different token-array hashes, so
the same input/length does not imply identical expert paths. Together with run
variation, this prevents assigning the94.8ms median difference solely to the
last-token commit. This test finds no large observed latency penalty; it does
not precisely isolate the cost of one evaluation.

This serve experiment has **MTP off by contract** and is a session-correctness
follow-up, not an MTP acceleration measurement. No frozen c4096/out1 Prefill or
business-quality gate is replaced. No exact whole-card peak measurement was
taken; allocator logs and post-run cleanup are the available resource evidence.

## Independent replay

Compact original logs, review-only response plans and summary receipts live in
[`session_long_20261015`](../backends/qwen35_llama/tests/evidence/session_long_20261015/).
The raw bytes retain their SHA across Git line-ending handling. Plans withhold
the full policy/input fixture and only identify response boundaries; they are
not executable stdin. The summary contains the exact token arrays for timing
requests, per-native SHA/cleanup receipts and every trajectory check.

CPU-only inspection, without an engine or model:

```text
python backends/qwen35_llama/tools/verify_session_long.py --plan backends/qwen35_llama/tests/evidence/session_long_20261015/warm-review-plan.json --raw backends/qwen35_llama/tests/evidence/session_long_20261015/warm-raw.log --out long-review.json
```

The complete backend CPU suite passes **109 tests in52.379s**, including9 new
evidence regressions: actual length versus requested maximum, token-count/cursor
corruption, missing terminal responses, nonfinite timing, real cancel versus
length completion, stale files after failed SAVE, and retained numeric failures.
These are CPU evidence/parser tests, not109 additional GPU experiments.

This reports structural checks and numeric replay diagnostics separately. It
also explicitly reports that local container-byte checks were not run unless
three `--state-file` arguments provide the actual payload files. The payloads
are retained locally under the second suite's immutable native/state directory,
not committed to Git. Full commands, transport receipts, failed verdicts and
the retention incident remain in
`D:/work/Strata魔改/results/code-review/session-long-20261015/`.

## Evidence retention incident

The supervisor client mistakenly reused the same three state output names in
the two suites. The second overwrote the first payloads. First-suite raw logs,
pre-overwrite SHA receipts and its original failed verifier result survive;
its container-byte check can no longer be independently rerun from that payload.
The second suite's three originals are now copied to its own immutable evidence
directory and independently rehashed. Future native state roots are unique per
job. This is a harness evidence-retention failure; it is not hidden or treated
as an engine failure or a successful recheck of the lost first payloads.

## Remaining external review items

| Item | Disposition |
|---|---|
| One service entry and shared Chat/score workflow | Open. Current two listeners and distinct workflows are explicitly incomplete product integration. |
| Classification quality/calibration | Open. Low label mass, probability ranking sensitivity and risk2/label3 dispute prohibit automatic business approval. No label changes here. |
| Session identity/security | Open. Container fingerprint samples file edges plus size; it is not full model SHA authentication. Keep service local/protected; bearer lease is workflow isolation, not user authentication. |
| Abandoned leases, queue/backpressure | Open. Current recovery remains service restart; no hidden TTL or multi-sequence claim. |
| CPU CI / portable clean build | Open. Local test receipts are not GitHub CI or Linux/HIP/SYCL qualification. |
| Branch integration | Core dependency order is #1→#2→#3→#5. Research tooling #4 must be reconciled onto that core before integration; newer status documents must not be overwritten. Another supervision line also opened overlapping #6; do not merge both tool implementations blindly. |
| Autotuner | Research-only. No selected production configuration, automatic application or measured2200-slot improvement is claimed. |

Single-GPU laptop/PC remains the implementation scope. Another GPU/card is not
a prerequisite for this increment. The project still lacks the quality,
integration and deployment evidence needed for a production release.
