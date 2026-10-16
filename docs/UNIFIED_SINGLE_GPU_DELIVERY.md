# Single-GPU measured tuner and unified API delivery

2026-10-16. Branch `codex/single-gpu-autotune`, based on accepted PR #5
`47c4b262`. No merge into main, default changes, automatic Profile installation,
native rebuild or production frozen-file replacement.

## Milestones and scope

| Milestone | Result |
|---|---|
| P3-5 | One canonical implementation: PR #4's stricter Profile lifecycle, shared helpers retained once; PR #6's inspect/screen/plan operations retained through the same dispatcher. [Selection record](PHASE3_CANONICAL_SELECTION.md). |
| P3-6 | Explicit `experiment --execute` launches the actual SHA-bound engine under the canonical Windows Job Object GPU lease. Resource admission, live budget checks, timeout, cleanup and evidence archival exercised. |
| P3-7 | Three one-variable candidates, 12 serial native processes, 72 requests, ABBA per candidate. All candidates NO-GO; selected_configuration=null. |
| P3-8 | Final complete191-test CPU suite PASS in64.183s, including startup bind-failure cleanup. Earlier190-test suite and targeted8 API tests also passed. Real integration31 gates PASS. Performance release gate remains open. |
| P3-9 | Opt-in single HTTP entry for Chat, Score, Decisions and Session using one engine, one busy lock and one ownership/revision contract. Real GGUF chat template and API E2E passed. MVP limits below; not product deployment approval. |

No CUDA/HIP/SYCL, native session, legacy server or score_api.py source changed
relative to PR #5. The existing `ea06d394…` native binary is reused, not rebuilt.
The prior frozen `0e00af8a…`/M0573, production GGUF and production3200 default
are untouched. Independent long text exact replay was not expanded.

## Reproduce the opt-in workflow

Windows online dependencies are listed in
`backends/qwen35_llama/requirements-experiments.txt`. Offline planning imports
neither pywin32 nor a GPU runtime. Online execution fails before launch if its
optional packages are missing; it never installs them automatically.

Use an explicit manifest with schema `strata-single-gpu-experiment/1`:
engine absolute path/full SHA; registered model SHA/quant and relative ASCII
model/pack paths; working directory; canonical lease tool and shared absolute
lock; CUDA bin; exact runtime environment; P/S/L input files/counts/SHA;
baseline configuration; budgets; resource limits; decision policy.
An annotated starting example is at
`backends/qwen35_llama/tests/evidence/single_gpu_20261016/example-spec.json`.
Replace paths and identities for your own registered files; this file does not
authorize execution or install a configuration by itself.

```powershell
python tools/strata_tune.py inspect --spec spec.json
python tools/strata_tune.py screen --spec spec.json
python tools/strata_tune.py plan --spec spec.json
python tools/strata_tune.py experiment --spec spec.json --execute --smoke --out NEW_REFERENCE_DIR
python tools/strata_tune.py check-experiment --spec spec.json --profile NEW_REFERENCE_DIR/profile.json
# Add baseline_profile pointing to that measured reference. Empty candidates
# generate at most3 one-variable candidates; max_native must cover the ABBA runs.
python tools/strata_tune.py experiment --spec search-spec.json --execute --out NEW_SEARCH_DIR
python tools/strata_server.py --spec spec.json --out NEW_SERVICE_DIR --port 8080 --lifetime 300
```

`tune/check` retain the canonical offline CLI. `inspect/screen/plan` never
launch an engine and do not select a theoretical winner. Output directories and
Profiles must be new. Online transport currently supports Windows with exactly
one identifiable NVIDIA GPU; unsupported platforms fail before launching.
No multi-GPU, tensor parallelism or cross-machine scheduling was added.

The resource planner uses an explicit baseline memory estimate plus conservative
positive slot growth. Reduced cache/COMPACT does not grant speculative extra
capacity. Larger ubatch workspace is refused until separately measured. Runtime
NVML/RSS/available-RAM/commit checks remain mandatory. Budget crossings stop the
related experiment, preserve logs and yield NO-GO without automatic retry.

## Actual performance experiments

RTX4060 Laptop8188MiB, Ryzen9 7945HX, Windows, CUDA12.8.1 runtime, engine
`ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619`,
production model `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`,
UD-IQ3_S. **This serve experiment uses MTP OFF. It is not a new MTP ON result.**
Same binary/model/input/path within every ABBA comparison; baseline is3200 slots,
t8/u4096/context4096/COMPACT false. Every native receives P0/P1,S0/S1,L0/L1:
P3896→1, S3896→16, L3817→128. Each request performs its full prompt; reused=0.
Suffix1 means the second request of its shape in one resident process, not a
cache-hit shortcut. All performance requests ended length, eos=false; S JSON
was visibly closed at16, but complete EOS request time is a separate check.

| Candidate change | Weighted E2E gain | Observed drift band | P time change | L E2E change | Verdict |
|---|---:|---:|---:|---:|---|
| ubatch4096→2048 | −10.074% |2.310%|+22.76%|−2.92%| NO-GO |
| threads8→6 |+0.167%|1.450%|−1.83%|−1.81%| NO-GO; gain inside drift |
| slots3200→2700 |−2.540%|4.096%|+0.09%|+9.10%| NO-GO |

Objective is90% normalized S E2E plus10% normalized L E2E. PASS requires at
least5% gain exceeding the observed range band, both paired directions positive,
and P/L nonregression guards. These are local pilot rules, not statistical
confidence intervals or universal hardware thresholds. Small samples do not
establish steady state or reliable p95. Failed2200-slot research was not rerun.

Whole-card sampled peaks: baseline7569.52MiB; ubatch2048 7343.52MiB;
threads6 7569.52MiB; slots2700 6985.52MiB. Cache reduction saves about584MiB
here but worsens long E2E; memory tradeoff is not algorithmic speedup. All12
search natives exited0, one lease/native, no retry or retained cleanup fence.
Their native process lifetimes were31.12–35.11s, including load and six requests.

Final-tool identity refresh on2026-10-16 is a separate one-native reference:

| Request | Prefill TPS | Observed TTFT ms | Observed decode TPS | Observed TPOT ms | Observed E2E ms |
|---|---:|---:|---:|---:|---:|
| P1 /1 token |1372.75|2853.80|n/a|n/a|2853.84|
| S1 /16 tokens |1416.57|2762.11|22.79|43.88|3420.42|
| L1 /128 tokens |1361.66|2812.95|25.86|38.68|7724.99|

The first P0 in that process took9520.9ms prefill. S0 decode was7.50TPS and
L0 20.11TPS; cold/first and later requests must not be collapsed. This new
reference is slower in decode than the previous day's search baselines, despite
the same native binary. **The cause is not located; no cross-day improvement or
performance regression PASS is asserted.** Warm P timing is close to the earlier
range, but the old frozen-build Prefill release gate is not newly certified.
No CPU benchmark/build ran concurrently with formal timing.

TPS is `(n−1)/(last−first)` from file-reader observations after native fflush,
with5ms polling resolution. It is not exact native flush timing; n<2 has no TPS.
TTFT, last-visible and EOS/DONE observation times are kept separately in evidence.
VRAM is100ms sampled whole-card high-water, not exact allocator peak. Host RSS
and minimum available memory are recorded per native. Decode internal kernel
critical path, GPU clock/temperature attribution and COMPACT performance remain
unmeasured in this pilot. The coarse S-prefill/L-decode split is not a kernel
profile and is insufficient to justify rewriting a CUDA kernel.

## Profile lifecycle and failed attempts

Offline profiles are research-only. Measured profiles bind full registered model
SHA, engine hash, quant, device UUID/CPU/RAM, runtime and Python package versions,
tool sources, input bytes, configuration, pack index and payload file stat, and
evidence hashes. Pack/model payload identity trusts the already verified full
hash registration plus recorded size/mtime; this work does not rehash15GB files.
Content checksum detects mutation; it is not a signed attestation against a
malicious author. Workload-validated means measured performance plus structural
gates only. `production_approved` and `auto_apply` remain false throughout.

The search finished before transport Ctrl-C isolation and source/dependency
identity hardening. Its original Profile correctly fails current identity
validation; its measurements/raw remain review evidence, not an installable
current Profile. The final refreshed baseline passes current validation, with
selected_configuration=null. No historical Profile was silently resealed.

Preserved failures: first smoke had a pipe-string SyntaxError before native
launch; second lacked psutil before launch; the next native succeeded but its
CPU JSON inspection lacked regex. After installing the declared dependencies,
that same raw was postprocessed without another GPU run; the original failed
Profile and derived artifact are both retained. Later dependency preflight
prevents this failure from occurring after a new launch.

The first integration harness forgot to reset an existing armed session before
an unrelated long SCORE. Native correctly rejected3920+3817>4096; the harness
then incorrectly expected cursor3817+4. Original failure and raw are preserved.
A specifically registered repair reset between scenarios and checked SCORE's
completion before continuing. One additional native passed31 integration gates;
there was no native-code change, label change or automatic retry. Total actual
model launches for this delivery:16, including that failed harness run.

## Unified entry and real correctness evidence

One opt-in local product endpoint serves `/v1/chat/completions`, `/v1/score`,
`/v1/decisions`, `/v1/continue`, `/v1/session`, `/v1/models`, capabilities/routes.
It inherits PR #5's ScoreService busy lock, owner token/revision and quarantine
behavior. Chat uses the same executor via private SCORE/GEN/reset, declares
prefix_reused=false and never claims cross-engine state reuse. An active Score
owner blocks an unrelated Chat request409 before pipe mutation. Explicit current
owner reset releases the workflow. Stale revisions similarly fail before writes.

The same native24.911s integration run covers actual short EOS17 tokens,
16-token continuation cursor, SAVE→LOAD→SAVE equal SHA, LOAD no-logits refusal
without state mutation, fixed append recovery, STOP after4 observed tokens,
cancelled cursor, subsequent8-token continuation, oversized refusal unchanged
state, shared HTTP Chat/Score/Decisions, owner/revision refusals, private state
reset and actual GGUF/Jinja message-template encoding. Decisions is uncalibrated;
this proves transport/structure, not business risk accuracy. State containers
stay local; their hashes and native export rows are published.

MVP limits: loopback only; one registered model; greedy text-only nonstream Chat;
no authentication, tool/image handling, multi-sequence batching or eager HTTP
disconnect cancellation. Unsupported streaming returns501. Engine STOP and
recovery are tested separately. `--lifetime`30..900s is an explicit finite
foreground prototype, not an installed daemon or a replacement for every legacy
UI/Anthropic feature. Bind failure also closes the owned native/link; shutdown
sends STOP then QUIT. Busy/invalid/quarantined routes retain existing status
semantics. Main/default server entry is unchanged.

## Review bundle and open gates

Public raw logs, token observation arrays, identities, paired arithmetic,
resource summaries, receipts and preregistrations are under
`backends/qwen35_llama/tests/evidence/single_gpu_20261016/`. Recompute without GPU:

```powershell
python backends/qwen35_llama/tools/verify_single_gpu_evidence.py backends/qwen35_llama/tests/evidence/single_gpu_20261016
python -m unittest discover -s backends/qwen35_llama/tools -p "test_*.py" -v
```

Independent verifier:307 checks PASS (raw SHA/count/order/prefill, timestamp
arithmetic, paired ratios/gain and cleanup). Full fixtures and binary/state
payloads are not redistributed; original artifacts remain in the local
`D:/work/StrataUnified/experiments/single-gpu-20261015/` and research ledger.
Absolute historical paths in evidence are provenance, not portable launch defaults.

Still open: no reliable tuning winner, cross-day decode slowdown attribution,
kernel-level bottleneck evidence, COMPACT pilot, formal frozen Prefill regression,
business0–3 quality, production lifecycle/streaming/deployment and final release
approval. Next useful performance action is a separately budgeted narrow decode
profile with clock/load context, rather than repeating these three NO-GO cells.
No product speedup or final production acceptance is claimed.
