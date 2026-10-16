# P3-5: one canonical tuner

Development branch: `codex/single-gpu-autotune`, from PR #5 `47c4b262`.
Review PR: [#7](https://github.com/insogao/Strata/pull/7). PR #4/#6 are closed
as superseded, with their branches and historical evidence preserved.
PR #4 `c320d801` and PR #6 `917df419` remain historical review branches, not
two implementations to merge in sequence. This new branch supersedes their
overlapping phase-three changes; it does not merge main or alter their heads.

| Area | Source decision and evidence |
|---|---|
| Fingerprint/plan/cost/screen and shared tests | Six files have identical Git blob IDs in both PRs. Keep once from #4. |
| Profile lifecycle | #4 canonical: registered full model SHA, engine/runtime/tool/workload identity, UUID/CPU/RAM, content checksum, evidence hashes, exclusive file creation and fail-closed invalidation. |
| #6 profile/check | Do not import: accepts8-character SHA prefixes, arbitrary evidence strings and caller-labelled recommendations; no engine/tool/workload checksum binding and overwrites profile files. More CLI code is not stronger lifecycle protection. |
| #6 inspect/screen/plan | Retain these operations through canonical CLI dispatch and shared capacity/screen functions. Plans remain dry-run; predictions never select a winner. Do not copy its second profile schema/implementation. |
| Selection | Offline profiles retain selected_configuration=null and auto_apply=false. Measured experiments use a distinct versioned artifact and require raw paired evidence; production approval is always false. |
| Commands | Existing tune/check remain compatible. New inspect/screen/plan/experiment/check-experiment delegate to the single measured-experiment module. No stacked strata_tune.py copies. |

Reproduced locally, CPU-only: #4's56 tuning tests pass in0.464s; #6's25 CLI tests
pass in0.238s in an isolated source snapshot. Neither result proves GPU speed.
Exact source lists/blob comparison are in the local P3-5 preregistration at
`D:/work/Strata魔改/results/unified-single-gpu-20261015/p3-5-manifest.json`.

Retained from #4: strata_fingerprint.py, strata_plan.py, strata_cost_model.py,
strata_screen.py, strata_tune.py; their three test modules; tools/strata_tune.py;
profile schema and tuning/foundation/screen/pilot documentation. The old
PHASE3_DELIVERY.md remains a historical offline-tool report until the measured
delivery report replaces its current summary. PROJECT_STATUS.md and
HANDOFF_FOR_NEXT_AGENT.md were **not** copied from either old branch; current
PR #5 correctness and acceptance history is preserved. #6's alternative delivery
and profile lifecycle do not overwrite them.

Research-only, measured, workload-validated and production-approved are distinct
states. A failed candidate remains NO-GO; no winner is fabricated to complete a
stage. No cross-machine, multi-GPU, tensor-parallel or distributed-expert work is
part of this branch.
