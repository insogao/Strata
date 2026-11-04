# DeepInfer-inspired single-GPU experiments: review bundle

Fixed RTX4060 Laptop8GB, registered66a3 GGUF,3200slots,u4096,c4096,MTP OFF. Same binary within each comparison; formal diagnostics OFF. This bundle is research-only and supports **NO-GO for changing defaults**.

| Scheme | Change and rationale | Observed result | Decision |
|---|---|---|---|
| C1 ID/activation cofetch | Enqueue activation with router IDs to avoid a second wait | Structural smoke passed; final OFF arm hit host commit guard at436.4MiB; incomplete control | No confirmed gain; no adoption, no rerun |
| C2 skip overlap probes | Opt-in omission of query-only CUDA event statistics | Warm128 E2E+0.44%; warm256+0.31%; primary cases below5% | NO-GO; defaultOFF |
| C3 threads8→10 | CPU/GPU resource balance, no algorithm flags | First128+4.89%; warm128+0.47%;256+3.24%/+3.68%; short/Score flat | NO-GO for default; directional small-sample lead only |

`measurements.json` contains independently recomputed E2E and(n−1)/(last−first)TPS, actual token counts/finish, resource facts, argv, exact feature identity, candidate score facts, native cleanup and original-logSHA. The12 `*-status.log` files are filtered numeric DONE/STATE/diagnostic excerpts; they deliberately omit prompts/generated text and are not full raw logs. Full immutable originals remain in the local research workspace under `results/deepinfer-20261021/di04-autonomous-long/`.

Twelve natives completed228/228 operations with exit0, empty child subtrees and no retainedGPU fence. Smoke covers real EOS, complete shortJSON, candidate-score ranking and held-state cursors; this is not dataset accuracy or production stability approval. Long independent-token equality is diagnostic only.

Each formal comparison has only two samples per setting; no steady-state/p95 claim. Prefill/Score, first/subsequent requests and held/EOS are separate. Earlier major improvement was caused by the user enabling the GPU's higher-power mode, not these changes.

Accounting correction: the OpenCode observer recorded134HTTP200 and1HTTP429, recovered on the same Go/model/max route. The original worker narrative claiming0new429 is superseded. CPU commands7>cap6 and harness compiles2>cap1 are preserved deviations, not retroactively declared compliant. GPU/build counts stayed within their registered caps. No automatic production/default/profile installation.

Supervisor verification2026-10-22: **296CPU tests passed in76.174s**, no additional native runs. See `validation.json`. Windows/CUDA12.8.1 only; HIP/SYCL not validated.
