# P3-10…P3-14: compatible baselines, business workloads and cancellation

2026-10-16. [Review PR #8](https://github.com/insogao/Strata/pull/8).
Branch `codex/workload-decode-diagnostics`, based on PR #7
`7a30da89`. Single RTX4060 Laptop8GB; no main merge, production-default change,
MTP enablement, native/CUDA change, model download or frozen-file replacement.

## Results and acceptance boundaries

| Task | Result |
|---|---|
| P3-10 | Historical Profile integrity and current-search compatibility are separate checks. Incompatible engine/model/quant/hardware/runtime/input/benchmark/baseline/resources/quality-policy refuses before any model launch. |
| P3-11 | Two controlled process repetitions reproduce cold/hot Decode differences. Power limiting and cold memory activity are observed; full causal attribution remains open. |
| P3-12 | W1 SCORE-only, W2 conditional SCORE→Continue and W3 generation run on real4K inputs. Manifest weights, actual branches/counts/finish, probabilities and state/no-reprefill gates are saved. |
| P3-13 | One COMPACT hypothesis tested; **INVALID_CONFIGURATION**, not a valid COMPACT performance comparison. Native silently stayed uniform without an expert profile. All raw and mathematical NO-GO results preserved; preflight and actual-layout gates now prevent this. No performance winner. |
| P3-14 | Cooperative SSE, disconnect handling, response-lost Score ownership cleanup, request deadlines, busy503/no queue, graceful finite CLI lifecycle. Final20 real service gates pass. Prefill cancellation is at native boundaries, not CUDA kernel preemption. |

Final complete CPU regression: **206 PASS in72.797s**. New evidence verifier:
**667 checks PASS**; historical PR #7 verifier307 checks still PASS. These are
contract/arithmetic checks, not performance samples or causal proof. Results
and raw console logs are included in the public bundle. GitHub Actions CPU CI is added for Windows/Linux, Python3.11/3.13,
syntax, protocol/session/tuner tests and both evidence verifiers. All four combinations passed on functional commit `26bb231e` in
[CI run38846690807](https://github.com/insogao/Strata/actions/runs/38846690807).
No hosted GPU was used; later documentation/array-display commits are separate.

## 1. Safe reuse of a historical baseline

`profile_integrity()` checks the artifact's schema, content checksum, identity
checksum, completed measurement and unchanged evidence. It does not equate an
intact historical artifact with a suitable current benchmark.
`baseline_compatibility()` compares the current task against that artifact:

- Full engine/model identities and quant; GPU UUID/driver/CPU/RAM/OS, tool and
  package/runtime identity, pack identity and runtime flags/path.
- Exact fixture SHA/count/shape, baseline configuration, resource limits,
  objective/gain policy, optional expert profile and telemetry regime.
- Benchmark version/path/timing/quality contract. Legacy profiles explicitly map
  to `psl-90-10/1`; new profiles record their actual benchmark descriptor.

Run budgets, evidence output locations, baseline_profile pointer and the explicit
candidate list may differ. Existing validation restricts candidates to one
declared configuration variable. Changing a baseline setting is not a candidate
change. There is no silent data migration or reuse of incompatible values.

The actual PR #7 historical reference remains intrinsically intact while being
rejected against the current tools. Negative CPU tests also isolate changed
engine/input/benchmark/runtime/resource conditions with an otherwise valid
baseline. Historical profiles were not resealed. Tool-source changes during this
delivery correctly invalidate earlier runtime Profiles; their raw measurements
remain versioned review evidence. A future search needs a fresh compatible
reference, not manually relabelled old measurements.

## 2. Decode diagnosis: observed facts versus causes

Same engine `ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619`,
model `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`,
UD-IQ3_S, MTP OFF, COMPACT false,3200 slots/t8/u4096/context4096. Inputs and
sequence match PR #7's P0/P1/S0/S1/L0/L1. Each diagnostic process then adds
SCORE-kept L128 and a short EOS check. Full prompt reads versus held-state decode
are separated. No independent long text equality requirement was introduced.

| Observed Decode TPS | Process A | New process B |
|---|---:|---:|
| S0 /16 tokens |11.98|16.13|
| S1 /16 tokens |37.72|36.75|
| L0 /128 tokens |29.64|28.35|
| L1 /128 tokens |31.96|30.24|
| Held-state L128, no new Prefill |33.89|34.03|

TPS uses file observations after native fflush,5ms polling, `(n−1)/(last−first)`;
it is not exact native flush timing.250ms rich telemetry is diagnostic and is
not mixed with older uninstrumented product-speed claims. Small samples are not
steady state or reliable p95. P0 cold prefill was9760.3ms in A; later prefill was
about2.7–2.8s. Cold process, first Decode and later requests are distinct.

Decode-only windows, excluding Prefill, show:

- S0 median ITL76.1/64.8ms versus S1 26.5/28.2ms. The slowdown is distributed
  across the short sequence, not solely one startup timestamp.
- S0 sampled process page-fault deltas32858/28537; only one S1 telemetry sample
  per run, so its zero delta cannot establish zero faults. Windows totals do
  not distinguish soft from hard faults, mapped model pages from driver/code
  pages, or memory bandwidth.
- Long/held windows show roughly38–40W power draw, enforced limits near40W,
  variable SM clocks, and software power-cap events. NVIDIA defines flag0x4
  as software power limiting in its [NVML documentation](https://docs.nvidia.com/deploy/archive/R510/nvml-api/group__nvmlClocksThrottleReasons.html).
  This is observed limiting, not proof that it caused the historical regression.
- Sampled temperatures were about54–60°C; no thermal-throttle flags appeared in
  the sampled Decode windows. Unsampled intervals are not ruled out. S0 can be
  slower despite higher sampled clocks than S1, so clock reduction alone does
  not explain the within-process cold penalty.
- Some cold windows include system disk reads, other slow windows do not.
  Process read-byte counters stay zero; this cannot exclude memory-mapped I/O.
  CPU frequency is an OS proxy; per-core load and process CPU/RSS/I/O were saved,
  but direct RAM bandwidth and critical CPU wait time were not measured.
- Peak sampled whole-card VRAM7893.85/7912.50MiB, available RAM minima
  5169.02/5202.65MiB. No budget violation/OOM/CUDA failure.

Identity comparison rules out a new model, quant, binary or compact change
**between these controlled runs**. Current warm values differ from PR #7's
later22.79/25.86TPS reference, but yesterday's runs lack clock/power/fault traces.
Their exact cause cannot be reconstructed. Older compact=true/MTP diagnostic CLI
experiments are not interchangeable with this serve path.

Bottleneck ranking: (1) first-Decode initialization/residency/mapped-page activity
is the strongest remaining cold-path hypothesis; correlation is not attribution;
(2) power limiting and clock variation constrain the current operating regime;
(3) host/GPU expert execution and transfer waits remain unseparated. Next useful
measurement is real serve cache/miss/transfer counters or a small bounded trace
linked to the slow token windows. No layer cost is invented or overlapping times
added. `qwen35_serve.cpp` explicitly hardcodes DONE hit/lookup/tier fields to zero;
they are **unavailable metrics**, not evidence of zero misses. Prompt-loan refill
rows are Prefill evidence only, not Decode miss-service times.

## 3. Manifest-driven business workflows

Legacy P/S/L remains supported. An optional `workloads` list selects entries with
id, kind, S/L fixture, positive weight (sum1), structural quality version, explicit
score candidates and, when applicable, max_new/explain_if policy. `score`,
`conditional`, `chat` are the three kinds. `always`, `never` and `choice-in` are
explicit conditional policies. No low-label-mass rejection or automatic business
release policy was added. The pilot forces both explanation branches for
reproducible cost comparison; it does not claim a learned business policy.

The pilot assigns W1=.5, W2-explain=.2, W2-no-explain=.1, W3-short=.1,
W3-long=.1. Warm second-request reference results:

| Workload | Actual result | Observed native workflow ms |
|---|---|---:|
| W1 /3896 input |0 generated; finite explicit candidate probabilities |2843.09|
| W2 explain /3817 input |128 output,length; SCORE2774.16ms; Continue reads0 new prompt tokens |6164.08|
| W2 no explain /3817 input |0 generated; score-only branch |2706.36|
| W3 short /3896 input |17 output,real EOS |3286.36|
| W3 long /3817 input |128 output,length |6253.61|

Native workflow intervals include candidate parsing and explanation, excluding
out-of-band validation STATE/reset commands. The separately measured real HTTP
service run covers full4K Score and Decisions responses, including JSON/network
overhead; final20-gate run records those client observations. These single HTTP
samples are not a throughput load test. `requests_per_s` in the engine artifact
is inverse isolated latency, a serial-rate proxy, not saturated service RPS.
Different output lengths' raw TPS are never averaged into the objective.

Paired comparisons use manifest-weighted normalized workflow E2E and refuse
different branch/count/finish outcomes. P regression guards remain. Numeric
candidate0–3 probabilities preserve existing full-vocabulary semantics; the
registered historical policy prompt originally requests JSON, so these numeric
candidate probes are mechanism/performance evidence, **not a calibrated risk
classifier or labelled accuracy result**. Low mass alone is not declared failure.

Reproduce with an adapted `workload-spec.json` from the evidence directory:

```powershell
python tools/strata_tune.py experiment --spec workload-spec.json --execute --smoke --out NEW_REFERENCE
# Point baseline_profile to that reference; register a specific candidate.
python tools/strata_tune.py experiment --spec search-spec.json --execute --out NEW_ABBA
```

## 4. COMPACT experiment was not applied

Hypothesis preregistered from near-limit VRAM/cold memory activity: COMPACT might
reduce footprint without losing3200 slots. One ABBA attempt,4 native processes,
all exit0, was run with identical inputs, weights and250ms telemetry on both arms.
The arithmetic gain was−0.514%, inside9.195% observed drift, selected=null.
**Both requested B arms explicitly logged no profile/staying uniform.** This is
an invalid configuration experiment, not evidence that real COMPACT is slow or
ineffective. Apparent VRAM differences between those uniform runs are not savings.

The controller previously bound requested flags but did not verify actual layout.
It now rejects COMPACT before launch unless a model-bound expert profile path/SHA
is registered, passes `--qwen35-profile`, and checks the native startup layout and
actual slot count before sending any benchmark request. Empty/ignored profiles
or a different layout/slot count fail closed. No existing calibration file was
blindly installed to rescue this trial. A future valid comparison must use the
same registered profile in both uniform and compact arms and prove actual layout.
The original four runs/Profile remain unedited, separately adjudicated invalid.

## 5. Streaming, abandonment and service lifecycle

`strata_streaming.CooperativeLink` wraps the existing strict EngineLink parsers;
score_api.py remains byte-unchanged. Existing DONE/count/limit/ownership validation
still executes. Streaming validates tokens before emitting SSE; malformed DONE
quarantines the engine and produces an error event without a false `[DONE]`.
Incremental text decodes only complete UTF-8 content; model control tokens are
not rendered. Streaming and ordinary Chat detect peer EOF and issue STOP, drain
the response, reset private state and release the same busy lock. Request deadlines
bound the entire operation; busy requests receive503/Retry-After1, no hidden queue.

Score/Decisions/Continue use the same abandonment handling. In particular,
disconnecting during keep=1 cannot strand a newly issued, unreachable owner in
the tested path: reset and lease discard happen while still holding the request
lock. Response-write failures attempt an exact-revision reset; an advanced/busy
revision is never stolen. Application-level receipt of a TCP-delivered response
is not guaranteed without an acknowledgement protocol; this local MVP does not
claim that stronger guarantee. TCP peer EOF is treated as cancellation.

CLI `--lifetime` remains finite and explicit (30..86400s, default300), one model
and one canonical GPU lease. Request threads are joined before native STOP/QUIT;
SIGTERM/SIGBREAK and Ctrl-C enter cleanup. Real tests used90s limits and exited
earlier via SIGBREAK; a day-long soak has **not** been run. Loopback only, no auth,
no batching or shared KV across processes. An in-flight Prefill kernel/chunk may
complete before cancellation; reset/skip-generation is verified, not GPU preemption.

The first service test passed11 gates, then a2s harness cancellation deadline was
shorter than the2.738s Prefill. Shutdown exposed daemon HTTP handlers not being
joined before pipe close, producing a500 on the abandoned request. Failure/raw
preserved. A registered repair uses non-daemon handlers and a bounded5s recovery
check; its18 gates passed. A further explicitly registered ownership test adds
lost keep=1 response handling; final20 gates passed, real GEN cancel<128, next
requests work, port/Job/lease clear. These are separate real natives, not hidden retries.

## Evidence and remaining work

Public evidence: `backends/qwen35_llama/tests/evidence/workflows_20261016/`.
It contains10 native receipts/raw logs, diagnostics, response observations,
profiles' measured values, original failures, preregistrations and CPU logs.
Model/state payloads and full policy input token files remain local. Recompute:

```powershell
python -B -m unittest discover -s backends/qwen35_llama/tools -p "test_*.py" -v
python backends/qwen35_llama/tools/verify_workflow_evidence.py backends/qwen35_llama/tests/evidence/workflows_20261016
python backends/qwen35_llama/tools/verify_single_gpu_evidence.py backends/qwen35_llama/tests/evidence/single_gpu_20261016
python tools/strata_server.py --spec spec.json --out NEW_DIR --port 8080 --lifetime 3600
```

No reliable performance improvement is claimed. Open gates: causal Decode
attribution, real cache/transfer counters, a valid profile-bound COMPACT comparison,
business0–3 accuracy, formal frozen Prefill release regression and long service
soak/deployment security. Main and production frozen artifacts are unchanged.
PR dependency remains #1→#2→#3→#5→#7→this branch; #4/#6 stay closed.
