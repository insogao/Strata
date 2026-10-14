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
