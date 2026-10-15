# qwen35_llama `/v1/score` + `/v1/decisions` GPU end-to-end (P2-4 / P2-4b / P2-4c)

Phase-2 backend-interface work, branch `phase2/backend-interface`. Task **P2-4**: prove that the P2-3
thin service (`backends/qwen35_llama/tools/score_api.py`, 0-GPU unit layer) behaves correctly on a **real
engine process**, with **line-level** evidence taken from the engine's own stdout — not merely "the
service returned numbers". No source, tool or frozen artifact was modified for this task.

Evidence lives in the research repo under
`results/opencode-execution/strata-unified-p2-4-score-e2e/` (P2-4b = service session; P2-4c = the two
CLI captures plus this document). The engine and model are untouched:
exe `66a48ad1366a122ea166ce557e23db85d62332bd62a2b017c2117e1c7cb27745` (recomputed pre-run),
model `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`, pack `models/pack`.

## 1. Startup and lease forms (with compensation evidence)

Common: cwd `D:\work\StrataUnified` (ASCII), model/pack passed as **relative** ASCII argv, exact three
production gates (`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`, `STRATA_QWEN35_PACKED_IDS=1`,
`STRATA_QWEN35_EXPERT_ORDER=count_desc`), `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING`
cleared, `runtime/cuda-12.8.1/bin` prepended to `PATH`. Engine argv (both natives identical):

```text
strata.exe --qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models/pack --serve
           --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096
```

**native1 — service session (P2-4b)**: `score_api.py --start-engine` spawns the engine as a **grandchild**;
`tools/gpu-lock.py run` cannot declare a Python interpreter as the binary (DENY_BASENAMES), so this
session ran under a **`hold` lease** (900 s): `hold/audit.json` owner pid 34940, acquired
`2026-10-15T00:19:26.380`; `hold/ready.json`; lock file `results/runtime/gpu.lock`. Service argv is in
`service/service-argv.json`; the engine's stderr was captured by the service (`--engine-log`,
`service/engine-stderr.log`). Engine stdout protocol lines are consumed by the service; only its stderr
summaries (`serve: req …`) persist. **The engine's numeric `LOGPROBS`/`ERR` lines are not in any service
log** — this is why the two CLI captures below are the needed originals.

**native2 — two CLI captures (P2-4c)**: two `gpu-lock.py run` leases, each declaring the **frozen exe**
itself as binary, with a pre-written stdin file (`--stdin-file`), one process per lease, serial:

- **N-A** (`job/job-p24c-NA.json`, `m-p24c-NA-01`, runtime limit 300 s) — stdin `nativeA-stdin.txt`:
  `CAPS` / `SCORE <P1 3869 ids> topk=5` / `SCORE 999999999,1`. Captures the engine's `CAPS`, `LOGPROBS`
  and `ERR` originals.
- **N-B** (`job/job-p24c-NB.json`, `m-p24c-NB-01`, runtime limit 420 s) — stdin `nativeB-stdin.txt`:
  `SCORE <P1> topk=5` / `SCORE <P1> topk=5` / `GEN 16 <P1>` / `SCORE <P1> topk=5` — one process, four
  serial steps (cold / hot / generation / post-generation).

Compensation evidence (P2-4c): `raw/preflight_exe_sha_p24c.json` (recompute before any GPU work)
matches the frozen SHA; each `run` lease re-hashes the binary **after** lock acquisition
(`binary_sha256_actual` matches); `READY` appears in each child log; both runs exited 0 with
`timed_out=false`, `child_cleanup.all_exited=true`, cleanup fence (`gpu.lock.cleanup-required.json`)
removed at `00:28:24.579` / `00:28:41.031` and the lock released at the same instants
(`nativeA/result.json`, `nativeB/result.json`). Post-batch state (`state/p24c-post.json`): lock byte
`\x00`, no fence, no `strata.exe` processes, GPU 1269 MiB / 0 %. P2-4b's own post-batch Python step
aborted on a localized-output decode error after the case batch (its `console.log`); the supervisor
verified resources clean, re-registered the window, and P2-4c re-verified **before** starting
(`state/p24c-pre.json`).

## 2. Case results (P2-4b service session, one pass per case)

| Case | Request | Result |
|---|---|---|
| C1 | `GET /v1/capabilities` | 200; `service="score_api"`, `engine_sha256` = frozen SHA; engine `CAPS` identity proven in §4 |
| C2 | `POST /v1/score` P1, `topk=5` | 200; `generated_tokens=0`, `pos=3868`, `full_vocab_softmax=true`; pairs **cold**: `[18,-0.8937],[15,-1.2617],[17,-1.2855],[16,-3.5211],[19,-8.3055]` |
| C2b | same request again | 200; pairs **hot**: `[18,-0.8927],[15,-1.2618],[17,-1.2878],[16,-3.5129],[19,-8.3166]` |
| C3a | `decisions yes_no`, candidates `[9542]`(`' yes'`)/`[874]`(`' no'`, ids from the model's tokenizer) | 200; `logprob -4.2394 / -8.7255`, `pos=22`; **no `label_mass`** — `label_mass_omitted`: *"the candidate set is not a closed, mutually exclusive cover of the next-token distribution (total candidate probability 0.014579, tolerance 0.001); the per-candidate probabilities are returned un-renormalized"* |
| C3b | `decisions score`, candidates `[18],[15],[17]` (C2 top-3) | 200; `logprob -0.8927 / -1.2618 / -1.2878` — the **hot** values (same process, same `pos=3868`), same reason text for `label_mass_omitted` (total 0.968569) |
| C3c | `decisions choice`, one candidate is 2 tokens | **422**, service-side, **whole request rejected at parse** (engine never called): *"candidates[0] has 2 tokens; multi-token candidates are not implemented in this subset - pass exactly one token id per candidate"* |
| C3d | `decisions choice`, two single-token candidates | 200; values as C3b (`-0.8927 / -1.2618`), `label_mass_omitted` (total 0.692692) |
| C4 | `POST /v1/score` `[999999999,1]` | **422**, `error_source="engine"`, `error`: `a token id is outside the vocabulary` |
| C5 | `POST /v1/score` P1 again | 200; pairs **identical to C2b** (all 5 at 4 dp) — scoring does not disturb the serving process for later requests |

New data (P2-4c, engine stdout originals — `nativeA/originals.json`, `nativeB/originals.json`):

| Capture | Step | Engine stdout (verbatim) | `prompt_ms` (engine stderr) |
|---|---|---|---|
| N-A | `SCORE` P1 (first inference of the process) | `LOGPROBS 3868 5 18:-0.8937 15:-1.2617 17:-1.2855 16:-3.5211 19:-8.3055` | 9704.5 |
| N-A | `SCORE 999999999,1` | `ERR a token id is outside the vocabulary` | — (no inference) |
| N-B | `SCORE` P1 #1 | `LOGPROBS 3868 5 18:-0.8937 15:-1.2617 17:-1.2855 16:-3.5211 19:-8.3055` | 3460.6 |
| N-B | `SCORE` P1 #2 | `LOGPROBS 3868 5 18:-0.8927 15:-1.2618 17:-1.2878 16:-3.5129 19:-8.3166` | 2696.5 |
| N-B | `GEN 16` P1 | `T 18` `T 248046`, `DONE 2 3869 2714.1 117.1 stop …` (2 tokens, EOS) | 2714.1 (+117.1 decode) |
| N-B | `SCORE` P1 #3 (after the generation) | `LOGPROBS 3868 5 18:-0.8927 15:-1.2618 17:-1.2878 16:-3.5129 19:-8.3166` | 2690.6 |

## 3. Line-level identity: service output ⇔ engine stdout (`client/p24c-compare.json`)

All comparisons are at the protocol's printed precision (`%.4f` in the engine's `LOGPROBS` line; the
service exposes exactly those values).

- **Same state class, exact**: N-A (engine, cold) **==** C2 (service, cold) **==** N-B#1 (engine, cold,
  another process) — 5/5 values identical at 4 dp, max |Δ| = 0.0000. N-B#2 (engine, hot) **==**
  C2b (service, hot) **==** C5 (service, hot) — 5/5 identical, max |Δ| = 0.0000.
- **Cold vs hot, differing**: N-A (cold) vs C2b/C5 (hot) — per-id |Δ|:
  `18: 0.0010, 15: 0.0001, 17: 0.0023, 16: 0.0082, 19: 0.0111` (max 0.0111). This difference is a
  property of the **engine state class**, not of the service: N-B reproduces both classes inside one
  engine process, and each class matches the service exactly.
- **Generation does not change values**: N-B#3 (after `GEN 16`) **==** N-B#2 — 5/5 identical, |Δ| max 0.
  ("Scoring after one generation" is answered: unchanged.)
- **CAPS identity**: engine `CAPS {...}` line from N-A vs the C1 service body minus its own fields
  (`service`, `engine_sha256`) — **equal** (`client/p24c-compare.json` → `caps_compare`), and
  `engine_sha256` = frozen SHA.
- **ERR identity**: engine original `ERR a token id is outside the vocabulary` vs C4 service error text
  (`error_source="engine"`) — equal after the 4-character prefix.

Conclusion (pass-through): within a given state class the service relays the engine's values exactly
(every field, 4 dp, no rounding, no renormalization on the `score`/`choice` path); the only differences
observed anywhere in this task are the engine's own cold-first vs stabilized-repeat difference.

## 4. Numerical-stability statement and precision warning

Observed on this host/engine/model (hall3k-risky P1, `pos=3868`, `topk=5`):

- The **first** `SCORE` of P1 in a fresh engine process ("cold") differed from **subsequent** scores of
  the same prompt in the same process ("hot") by up to **1.11e-2** in logprob
  (per-id 1e-4 … 1.11e-2; same values in the P2-4b service session and both P2-4c processes).
- Every **hot** repeat was **identical at printed precision** (C2b ≡ C5 in the service; N-B#2 ≡ N-B#3
  in a CLI process; N-B#2 ≡ C2b/C5 across processes).
- Every **cold** first score was also identical across three independent processes at printed precision
  (P2-2 historical line, P2-4b C2, P2-4c N-A / N-B#1).
- Cold `prompt_ms` **varied widely** (7.0 s / 9.7 s / 3.5 s in the three cold runs) while cold **values**
  stayed identical — timing variation and value stability are separate observations.
- The engine itself documents `SCORE`/`GEN` as session-resetting (`reset_session` = `llama_memory_clear`
  of KV + recurrent planes before and after each request; `reuse` is always 0, `prefix_cache=false` in
  `CAPS`), so the cold/hot difference is **not** cross-request prompt reuse.

**Warning for downstream consumers**: do not compare a cold-state score with a hot-state score at finer
than **~1e-2**; at this host/model/quantization the cold↔hot deltas reach 1.11e-2, which is real and
reproducible. Consistency checks between service and engine must be done within the same state class
(cold↔cold or hot↔hot), where they are exact at the protocol's `%.4f` resolution. This document does
**not** claim score values are globally deterministic, and nothing here should be read as "the score API
is fully deterministic" — the service is a faithful relay; the engine's first call after load differs.

## 5. Assumptions, hypotheses, and untouched items

- **Root cause of the cold/hot difference: NOT established.** Any explanation (first-run kernel or
  heuristic selection, lazy initialization, …) is a **hypothesis** only, left to P2-5+ or a dedicated
  investigation. No per-layer cost or mechanism claim is made.
- The engine's `LOGPROBS` line prints `%.4f`; all "identical" statements above are at that resolution.
- C3c multi-token candidates: service-layer **whole-request** rejection (implementation as designed in
  P2-3); the engine was never called; this is not per-candidate validation.
- `label_mass` was not emitted by any `decisions` call; each response carries `label_mass_omitted` with
  the reason quoted in §2 (candidate set not a closed cover; probabilities returned un-renormalized).
- Not tested here: concurrent clients, streaming, stop/filter parameters on the score path, other
  prompt classes (only P1 = han3k-risky and the 23-id yes/no prompt), models other than the
  IQ3_S production file, and finer precision than the protocol prints.
- Case hygiene: each case and each native ran exactly once; no retries; no re-runs to chase numbers.

## 6. P2R-2: GPU validation of the P2R-1 additions (2026-10-15, branch `phase2r/review-gaps`)

Task **P2R-2** ran the P2R-1 additions (`SCORE ... [cand=...]` / `topk=0` and the new `/v1/decisions`
payload) on the real engine and cross-checked the new exe against the old one. Evidence:
`results/opencode-execution/strata-unified-p2r2-candidate-gpu/` (preflight SHA, jobs/argv, raw child logs,
HTTP originals, `client/p2r2-compare.json`, `p2r2-run-console.log.md`). Exes, SHAs recomputed pre-run: old
`6dee0c66c3e87c3020c6ddde3c9ebaa40bed3369745e8e1dbfb0a7ac963ad90a` (139,253,760 B), new
`d0321a98ab1908ef3c0f0d91b1b6e6c42758d79594222c6def425bf406a4bc09` (139,257,344 B). cwd
`D:\work\StrataUnified`; same engine argv as §1 (ctx 4096 / t8 / slots 3200 / ubatch 4096), the three
production gates, CUDA `bin` prepended to `PATH`. Four natives, one `gpu-lock run` lease each declaring
the exe under test, strictly serial, `--stdin-file`; N4 used the same `hold` + compensation form as §1.
0 new fences; post-batch: lock `\x00`, no marker, no `strata.exe`, GPU 1323 MiB / 17 %.

Prompts (ASCII, ids from the repo tokenizer, `n_vocab=248320`): **P_short** = 23-id closed yes/no question;
**P_open** = 25-id open continuation; **P_cls** = 61-id "Answer with a single digit" classification;
**P_long** = the 3896-id benign fixture. Candidate ids: `' yes'`=9542, `' no'`=874, `'0'..'3'`=15..18;
spread ids `0,1,2,3,100,1000,10000,50000,100000,150000`.

### A - old/new byte-compat at equal positions

`SCORE P_short topk=5` (first score of the process), `SCORE P_long topk=5` (warm#1 = first score of that
prompt), same again (warm#2); one process per exe.

- **`LOGPROBS` byte-identical 3/3**, e.g. cold P_short (both exes):
  `LOGPROBS 22 5 271:-0.0553 198:-4.0315 9542:-4.2394 7179:-4.9887 695:-5.5802`; warm#2 P_long (both):
  `LOGPROBS 3895 5 90:-0.1160 4754:-2.2494 13215:-6.1471 1:-7.3827 5202:-8.6236`.
- `DONE` lines: all non-timing fields identical; raw bytes differ only in `prompt_ms`/`decode_ms`. The
  stderr diagnostic (no `cand=`) is byte-identical outside `prompt_ms` - the new exe appends nothing.
- **Classification of the one systematic difference**: N2's own warm#1 vs warm#2 differ by up to
  `|Δ| = 0.0327` (id 13215: `-6.1144` vs `-6.1471`), i.e. repeats of the first score of a prompt differ
  from the stabilized repeat - the §4 phenomenon, here on P_long and reaching **3.27e-2** (larger than the
  1.11e-2 §4 record). Because old vs new is byte-identical **within each class**, the difference is
  classified as **state-class drift, not P2R-1 incompatibility**.

### B1 - explicit candidates vs the same-process `LOGPROBS`

`SCORE P_long topk=0 cand=90,4754,13215,1,5202` (new exe, hot; ids taken from N1's warm#2 line) answered
`CANDLOGPROBS 3895 5 90:-0.1160 4754:-2.2494 13215:-6.1471 1:-7.3827 5202:-8.6236` - **5/5 values
byte-equal** to that process's own `LOGPROBS` warm#2 line.

### B2 - candidates outside the top-k, and the union mass

`SCORE P_long topk=5 cand=0,1,2,3,100,1000,10000,50000,100000,150000` answered
`CANDLOGPROBS 3895 10 0:-13.5132 1:-7.3827 2:-15.8887 3:-16.4839 100:-20.6537 1000:-22.4266
10000:-24.8607 50000:-27.0409 100000:-23.6128 150000:-25.7831` - all finite, all `<= 0`.
Union mass over `{top-5} ∪ {spread}`: `Σexp(logp) = 0.99888066 <= 1`. 4dp quantization note: the engine
prints `%.4f`, so each `exp()` term carries that rounding (relative error ~5e-5..1e-4); id 1 appears in
**both** tables with the identical value `-7.3827` and is counted once in the union. One honest caveat: of
the 10 requested ids, id 1 actually sat **inside** that top-5 (rank 4) - the other 9 were outside it; the
cross-table value identity is the strongest same-request check here.

### B3 - `SCORE` top-1 == greedy `GEN 1` first token

`GEN 1 P_short` -> `T 271`; the following `SCORE P_short topk=5` top-1 id is **271**. `GEN 1 P_long` ->
`T 90`; `SCORE P_long topk=5` top-1 id is **90**. Both `GEN 1` requests ended `finish=length` (max_new=1,
no EOS), so both prompts apply. (The `SCORE` answers were hot while the `GEN`s were the processes' first
inference; the argmax did not change across the classes - consistent with A/C.)

### C - repeats, drift, near-threshold sensitivity

Selection rule (preregistered before the natives): the adjacent pair in N1's P_long warm#1 line with the
smallest gap -> **ids 1 / 5202, gap 1.2125**. Three repeats of
`SCORE P_long topk=5 cand=1,5202` (new exe) answered `1:-7.3827 5202:-8.6236` **identically each time**;
per-id value range `0.0`; rank order unchanged; `LOGPROBS` top-1 (90) unchanged. Caveat, stated plainly:
the P_long top-5 has **no true near-tie** (adjacent gaps 2.116 / 3.880 / 1.268 / 1.213), so this does not
demonstrate rank-flip sensitivity at gaps below the drift; it does show hot repeats are stable at the
`%.4f` print precision. Side by side with the §4 figure: the warm#1-vs-warm#2 deltas reached **3.27e-2**
(larger than §4's 1.11e-2), while every same-class repeat was identical. Practical rule restated: compare
within one state class; cold-vs-stabilized comparisons on this host need a resolution coarser than
~3.3e-2. **Root cause not established** - no mechanism is claimed.

### D - live service payload (P0-2 real cases)

New exe under `score_api.py` (hold lease + compensation, §1 form). Raw JSON:
`native4/http-evidence.jsonl`; engine stderr: `native4/engine-stderr.log`.

| Case | Request | Result |
|---|---|---|
| D1 | `decisions yes_no` P_short, candidates `[9542]`/`[874]` | 200; `logprob -4.2394 / -8.7255`; `label_mass 0.014579`, `covers=false`; `conditional 0.9889 / 0.0111` (Σ=1 at 4dp) - identical at print precision to the P2-4b C3a values of §2 (old exe/service) |
| D2 | `decisions yes_no` P_open | 200; `logprob -12.8303 / -9.6064`; **`label_mass 6.9975e-05`** (`< 1`, far below 0.5); `conditional 0.0383 / 0.9617` (Σ=1 at 4dp) |
| D3 | `decisions choice` P_cls, candidates `[15],[16],[17],[18]` | 200; `logprob -9.6950 / -11.4668 / -11.2866 / -12.8001`; `label_mass 8.7363e-05`; `conditional 0.7050 / 0.1199 / 0.1435 / 0.0316` (Σ=1 at 4dp) |
| D4 | `/v1/score` P_long, `candidates` = spread ids | 200; top-5 and candidates in request order; common id 1 equal in both tables (`-7.384`); no `label_mass` on the score path |

Every decisions response carries `label_mass` + `label_mass_covers_distribution:false` +
`label_mass_tolerance:0.001` + `calibrated:false` + the three `probability_scopes` texts; per candidate
`logprob` / `full_vocab_probability` / `conditional_probability` and the deprecated `probability`
(== the full-vocab value; `probability == exp(logprob)`); **no** field whose name contains
confidence/threshold/verdict anywhere, including the two low-mass cases; `pos`/`prompt_tokens` consistent
(22/23, 24/25, 60/61, 3895/3896).

### Unexplained / not covered

- Root cause of the first-vs-stabilized difference: **not established** (§4 warning updated with the
  3.3e-2 observation; the 1.11e-2 §4 line remains as recorded history).
- No true near-tie existed in the measured top-5 (C caveat); testing rank-flip sensitivity below the
  drift scale needs crafted near-threshold candidates or another prompt (P2R-3+) - **quantified by
  P2R-2b below via a top-2000 class comparison**.
- One model/quantization (IQ3_S), no cross-quantization comparison, no speed claim of any kind.

### P2R-2b - near-threshold rank sensitivity (top-2000 class comparison)

P2R-2's C caveat (no true near-tie inside the top-5) is quantified here by one same-process top-2000
class table. Run 2026-10-15 on branch `phase2r/review-gaps`: **one native**, new exe `d0321a98…`, one
`gpu-lock run` lease, rc=0, 12.7 s; static stdin `SCORE <P_short> topk=2000` ×3, `SCORE <P_long>
topk=2000` ×2, `QUIT`; pos 22/22/22/3895/3895; no `ERR`. Evidence:
`results/opencode-execution/strata-unified-p2r2b-near-tie/` (`final-report.md`, `client/p2r2b-analysis.json`,
raw stdout under `native1/raw/`).

1. **Per-id drift** (4dp printed values, Δ = later − earlier): P_short cold→warm1 max|Δ| **0.0599**,
   median 0.0098, 1978/1989 ids nonzero; P_short warm1→warm2 max|Δ| **0.0000** - the two full `LOGPROBS`
   lines are byte-equal, extending P2R-2's same-class zero-drift from top-5 to top-2000; P_long
   first→repeat max|Δ| **0.0713**, median 0.0207, 1987/1990 nonzero.
2. **Rank changes** (order key (value desc, id asc); adjacency on the common-id order): P_short
   cold→warm1 - 9837 inverted pairs (τ 0.990049), **881 flipped adjacent pairs** (e.g. `2062/2091`,
   gap 0.0037: `-8.9135→-8.9171` vs `-8.9172→-8.8936`); warm1→warm2 - 0 inversions, 0 flips; P_long
   first→repeat - 13930 inverted pairs (τ 0.985923), **895 flipped adjacent** (e.g. `1143/2083`,
   gap 0.0371: `-11.7410→-11.7781` vs `-11.7839→-11.7776`).
3. **Threshold rule (this host / this IQ3_S quant / this exe only)**: same-class repeats are
   printed-identical (band 0.0000, so any nonzero gap is stable). Across state classes the measured
   drift band is max|Δ| **0.0599** (P_short cold→warm1) / **0.0713** (P_long first→repeat); inside the
   band (gap ≤ band) **881/1961** and **895/1966** adjacent pairs flipped (~45%); flips occurred for
   gaps up to **0.0383 / 0.0371** and none above; the smallest flipped gap is **0.0000** (printed ties
   can flip). Practical reading: below a gap of ≈**0.038** the order of two candidates across state
   classes must be treated as reversible (conservative bound = the 0.06/0.07 band itself).
4. **Closest calls**: the closest non-flipped call is `4777/14149`, gap 0.0000, margin_to_flip
   **+0.0015** (P_short cold→warm1); the closest flipped pair is `91/3701` (P_long), gap 0.0000,
   margin **−0.0284**. Top-k cutoff churn, recorded as measured: 11/11 ids (cold→warm1) and 10/10
   (first→repeat) change membership; the task's a-priori "identical id sets" assumption did not hold,
   so drift/rank stats use the intersection and churn is reported separately.
5. Side by side with prior records (facts only, no root cause attributed): P2-4 **1.11e-2**; P2R-2
   top-5 max **3.27e-2** with no observable near-tie (min top-5 gap 1.2125); P2R-2b top-2000
   same-class **0** vs cross-class maxima **5.99e-2 / 7.13e-2** with real flips up to gap 3.8e-2.
