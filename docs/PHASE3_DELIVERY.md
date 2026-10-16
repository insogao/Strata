# Phase 3 delivery: offline autotuner research tooling

Historical P3-1…P3-4 report. The canonical P3-5…P3-9 implementation, real
single-GPU experiment results and unified API limits are recorded in
[UNIFIED_SINGLE_GPU_DELIVERY.md](UNIFIED_SINGLE_GPU_DELIVERY.md). This older
offline completion statement is not acceptance of the online autotuner.

2026-10-15. **P3-1…P3-4 engineering increment delivered for review; full product
autotuning acceptance remains open.** No production default change or measured
speed improvement is claimed. Code is opt-in and CPU-only outside the earlier
explicitly registered P3-3 GPU pilot.

| Item | Delivered | Evidence / limit |
|---|---|---|
| P3-1 | Device/model fingerprint and two-pool resource planner | [Foundation](backends/qwen35_llama_autotuner_foundation.md); 19 CPU tests; capacity estimates, not local-engine admission |
| P3-2 | Offline cost intervals, screening and ≤5 shortlist | [Screen](backends/qwen35_llama_offline_screen.md); 20 CPU tests; hold-out mean absolute error 9.70%, max21.30%, coverage2/4; insufficient to select a winner |
| P3-3 | Held-out paired GPU pilot | [Pilot](backends/qwen35_llama_autotune_pilot.md); six serial natives, no retries, one lease each, same `c3780a6a…` binary and experimental MTP `ab639a7f…` model |
| P3-4 | Equivalent `strata tune` CLI, persistent research schema, identity invalidation and evidence index | [CLI](backends/qwen35_llama_tune_cli.md); modes, manual fallback, checksums, no engine invocation, no auto-apply |

## Measured outcome and independent check

Only candidate `s2200-c-t8-u4096-x8192` was tested; shortlist #1 matched the baseline
and was skipped. Six raw flush arrays were independently recomputed at handoff:

| Shape | A warm TPS (two natives) | B warm TPS (one native) | Delta vs A median | Drift | Verdict |
|---|---|---|---|---|---|
| S:3896→16,ctx4096 | 28.834293 / 30.488871 | 28.127739 | −5.171% | 5.578% | NO-GO; loss within drift band |
| L:3817→128,ctx8192 | 28.501461 / 27.298571 | 23.010806 | −17.524% | 4.311% | NO-GO; decisive loss |

TPS is `(n−1)/(last_visible_flush−first_visible_flush)`, **diagnostic CLI K1**, rq2
of two same-prompt requests/process. Both shapes ended `finish=length`, `eos=0`;
S JSON closed at16 but did not reach EOS. No steady-state/p95/product-E2E claim.
The 2200-slot saturation hypothesis is falsified for this pilot. The baseline is
retained as an experimental reference, not promoted to a production winner.

Local raw: `D:/work/Strata魔改/results/opencode-execution/strata-unified-p3-3-autotune-pilot/retry2/`.
Independent recomputation: `D:/work/Strata魔改/results/code-review/p3-4-tune-cli/p33-independent-recompute.json`.
P3-3 preregistered deviations (experimental model, CLI rather than serve, diagnostic
flags, engine peak plus pre/post sampling) remain part of the result.

## Acceptance still open

- No second GPU was available/used; cross-card validation is pending. Capacity
  formulas do not certify performance or portability to other vendors/quantizations.
- No validated winner across 2700/3200 and short EOS,128/256 long EOS,ordinary chat,
  business weights,quality,TTFT/E2E and tails. No automatic GPU search executor or
  online tuning is implemented. The CLI intentionally returns no selected config.
- Static planner workspace, slot averaging and local engine VRAM discrepancy remain
  estimates. COMPACT cache capacity and the planner's per-layer quota are different
  configuration models. Live GPU resource admission still requires the real lease
  and engine resource checks; a saved JSON file is not a lease.
- P3-3's overall slowdown relative to older batches and changed S finish behavior
  remain unexplained. Preserve raw and compare within the registered batch.
- P2R supplement remains a separate PR (#3→#2→#1); this branch was based on
  `phase2/backend-interface` at `954a88e4`, not on the completed P2R branch. No state,
  scoring,session or quality guarantee is inherited from that other branch.
- The separate 2026-10-15 code review reported pending-token continuation, shared
  workflow ownership, non-finite score parsing and request-length range concerns.
  Those findings need scoped fixes/revalidation; this tooling PR does not close them.
  Local review: `results/code-review/strata-unified-de89bf97/REVIEW_20261015.md` in the
  research repository. Historical risk2/label3 adjudication also remains open.

## Review and integration

Stage-three work is proposed in [PR #4](https://github.com/insogao/Strata/pull/4)
to `phase2/backend-interface` (P3-4 code commit `ae1a3512`);
no main merge, P2R cherry-pick, engine/kernel edit, model overwrite or production
profile installation. P3-4 changes only Python tooling and documentation. Existing
engine sources/default server/setup entry points remain identical to the handoff
head. GPU/build budget for P3-4 is zero; CPU regressions and a fresh read-only host
profile lifecycle provide its verification. Product qualification needs the open
acceptance work above, not relabeling this engineering delivery as complete.
