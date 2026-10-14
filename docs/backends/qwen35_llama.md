# qwen35_llama backend (imported, standalone, research)

`backends/qwen35_llama/` is the Qwen3.6/Qwen3.8-Flash MTP research engine imported from the research repo's
`results/worktrees/phase2-mtp232` (mini Strata tree `v0.1.41` + the **live** patched `vendor/llama.cpp`).
It is an **independent CMake project** inside this repo: it is not wired into the official top-level
`CMakeLists.txt` and must be configured and built separately (see "Build isolation" below).

This backend is **research-only**. Nothing here is a production claim; the MTP line is EXPERIMENTAL and its
correctness gates are not closed.

## Provenance and pin→live evidence

- Import source (read-only): `results/worktrees/phase2-mtp232`, HEAD `9bb9b6a556376f003d563f3e07a5d1e6dc91757f`
  (research repo). Its working tree had 17 dirty files + 2 untracked txn files
  (`include/strata/core/qwen35_mtp_txn.hpp`, `src/core/qwen35_mtp_txn.cpp`); all 19 were imported and verified
  byte-equal to both the live tree and the 353 archive `unified-assets-20261014/source/mtp232-changed/`
  (19/19 SHA256, 2026-10-14).
- Upstream pins (`UPSTREAM_QWEN35.json` in this directory): Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`
  (`v0.1.41`); llama.cpp `988190680d5a89fce97de3c20df2c2813731fd61` (`b11476`).
- **vendor/llama.cpp is imported in its live state, not rebuilt from the archived patches**: the archived
  handcrafted patch (6 files) does not contain `include/llama.h` / `src/llama-batch.cpp`, while the live
  `include/llama.h` carries `llama_batch_ext` / `llama_process` (llama.h:1012-1091). The live tree is also the
  "9-file" surface of inventory 353 §4 (extra `ggml-cuda/gated_delta_net.{cu,cuh}`, `norm.cu`, all default-off
  diagnostics). The origin of `llama.h`/`llama-batch.cpp` relative to pure `b11476` remains UNKNOWN
  (353 §4/§9) - do not rebuild vendor from the archived patches alone.
- Key file SHA256 in this import:
  - `vendor/llama.cpp/include/llama.h`
    `121aef7c9aa5204d255a30951022ef249a0f657faec28d1d545e2da501c26abb`
  - `vendor/llama.cpp/src/llama-batch.cpp`
    `b1809b99544c1d63e3a0cf306666b26d2d79c7d86042e69765f9a06346b1c59b`
  - `vendor/llama.cpp/src/models/qwen35moe.cpp`
    `f617c8b012be90b7a3d3e9adb3131308e9f58ea5e57f3853631b1d5620e9d3b1`
  - `vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
    `b877b3260ff56555858e389cdca4f5e9339a9ac2480789c4e1ef9e0d46840336`

## Capability and boundary (one-page classification, 353 §7 wording)

- **CERTIFIED** (credible measured evidence, migratable as-is): Qwen3.6 MMQ Prefill
  (`include/strata/prefill/moe_mmq.hpp`, `src/prefill/moe_mmq.cu`, `cold_stager.*`, `qwen35_prefill.cu`),
  the CPU+GPU mixed expert service baseline (`expert_source.*`, `expert_cache.*`, `pool.*`,
  `qwen35_bridge.*`), Prefill loan/restore. The 2700-slot COMPACT result is a **resource result only**
  (arena/payload 3199.2 MiB), not a speed claim.
- **EXPERIMENTAL** (feature exists, benefit/correctness not fully proven; default off):
  the whole **MTP K1 chain** (draft/verify/replay/state/emit + txn; T3 and quality gates are not closed,
  the 2-true-vs-fixed-label-3 quality failure is unresolved) - never label this CERTIFIED;
  COMPACT slots as a product default (env-gated, default OFF); profile variants.
- **REJECTED** (disproven / reproduced with no benefit - do not re-package): CPU GU-reuse slice route
  (`STRATA_QWEN35_SLICE_GU_MULTI`, tasks 298/300), norms fusion (`STRATA_QWEN35_FUSE_ADD_RMS_NORM`, task 151),
  the CUDA-Graph toggle (task 151), the 3200-slot speed route, expert-cache "early promotion".
- **DIAGNOSTIC-ONLY** (one-shot probes; instrumentation is not product performance):
  `STRATA_QWEN35_DECODE_TIMESTAMPS`/`_STOP_EOG`, GDN events, step probes, `STRATA_QWEN35_TRACE`,
  routing-detail logs, K1 cost split, T3 FA audit, oracle/fault-inject/teacher-forced probes - all default off.
- **UNKNOWN**: vendor pin→live completeness for `include/llama.h` / `src/llama-batch.cpp` (needs a pure
  `b11476` checkout to compare; 353 §4/§9).

## Environment gates and defaults

Every switch below is **default OFF**; the default path is byte-identical when the variable is unset. Most gates
require the exact value `1`. Representative gates (file:line in this backend):

- `STRATA_QWEN35_COMPACT_SLOTS` - non-empty and not `0` enables compact static slot sizes
  (`src/core/expert_source.cpp:3437`); fails closed to the uniform layout when no profile is loaded.
- `STRATA_QWEN35_SMALL_T_NATIVE` / `_SMALL_T_MULTI` / `_SMALL_T_COMPARE`, `STRATA_QWEN35_SHARED_OVERLAP` -
  decode/prompt prototypes (`src/core/qwen35_bridge.cpp:163/175/186/153`).
- MTP transaction gates: `STRATA_QWEN35_MTP_TXN`, `_MTP_TXN_K`, `_MTP_FAULT_INJECT`, `_MTP_ROLLBACK_ORACLE`,
  `_MTP_T3_VERIFY_ORACLE`, `_MTP_DRAFT_STATE_ORACLE`, ... (`src/core/qwen35_mtp_txn.cpp:1141+`). The T3 verify
  oracle additionally requires `STRATA_QWEN35_SLICE_GU_MULTI=1` and `STRATA_QWEN35_COMPACT_SLOTS=1`
  (`qwen35_mtp_txn.cpp:1437-1441`).
- REJECTED experiment gates (kept for evidence only): `STRATA_QWEN35_SLICE_GU_MULTI`
  (`src/kernels/cpu/native_expert.cpp:144`), `STRATA_QWEN35_FUSE_ADD_RMS_NORM`
  (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4293`).
- Diagnostic gates: `STRATA_QWEN35_DECODE_TIMESTAMPS` / `_STOP_EOG` (`src/program/qwen35_model.cpp:977/985`),
  `_GRAPH_STATS` (`:891`), `_GDN_EVENTS` (`:958`), `STRATA_QWEN35_TRACE` (`src/core/qwen35_bridge.cpp:824`).

Do not turn any EXPERIMENTAL/REJECTED gate into a default. The source defaults are the contract defaults.

## Model / quantisation requirements (two lines)

1. **IQ3_S line (pin model of this tree)** - `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`,
   13,676,723,168 bytes, SHA256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`
   (recorded in `UPSTREAM_QWEN35.json`; the frozen production GGUF named in the 60 TPS contract).
2. **MTP experimental line (newly verified model)** - `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` under
   `models/experimental/mtp-qwen36-5bc3e238/` (research side), 15,346,432,288 bytes, SHA256
   `ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f`, with `pack-main40` ready.

The build recipe below (mirroring the old review232 configuration) enables `STRATA_MMQ_KQUANTS` and
`STRATA_Q6K_EXPERTS`; no new quality or speed claim about either model line is made by this import.

## Build isolation

The official top-level project and this backend define the **same target names** (`strata`, `strata_spec`,
`strata_kernels_cpu`, `strata-gguf`, `strata-dequant`, `strata-plan`, ...) and this backend's vendor is a
**patched** llama.cpp with its own ABI/CUDA-DLL expectations, while the official repo links a clean pinned
ggml. Pulling both into one CMake tree (e.g. `add_subdirectory`) would collide on target names and mix two
ggml/llama binaries in one process. Therefore the two are kept separate: configure/build this backend as its
own project into its own build directory and produce its own executable. The copied `CMakeLists.txt` is kept
byte-identical to the source tree (import fidelity); the separation is documented here instead of renaming
targets. Build the official project exactly as before; never add this directory to its build.

## Building

Entry point: `backends/qwen35_llama/build.ps1` (writes one `cmd.exe` batch in the build dir, calls
`vcvars64.bat`, then cmake configure + build). Toolchain is the recorded one for this engine: MSVC
2022 Build Tools (vcvars64, cl 14.44), CUDA 12.8.1 nvcc (V12.8.93), Ninja, `CMAKE_CUDA_ARCHITECTURES=89`,
`CMAKE_CUDA_RUNTIME_LIBRARY=Shared` (the tree documents the MSVC static-runtime LNK1169 at CMakeLists.txt:141-147;
the old build cache pinned the same value).

```
powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1                # configure + build
powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1 -SkipConfigure # rebuild only
# parameters: -BuildDir <dir> (default <this dir>\build), -CudaRoot <cuda-12.8.1>, -Jobs <n> (default 8)
```

Configure flags actually used for this import's verified build:

```
-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
-DCMAKE_CUDA_COMPILER=<cuda-12.8.1>\bin\nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=89
-DSTRATA_ENABLE_CUDA=ON -DSTRATA_QWEN35=ON -DSTRATA_GGML_DIR=<this dir>\vendor\llama.cpp
-DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
```

Artifact: `build/strata.exe`. Verified clean build (2026-10-14): exit 0, 383 ninja targets, wall 666 s
(8-way), size **139,208,192 bytes**, SHA256
`6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4`. Build attempts used: 3 (1 link failure
LNK1169 fixed by adding `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`; 1 incremental relink success; 1 clean full
success). MSVC `/DEBUG` embeds a PDB signature/timestamp, so exe bytes are not reproducible across build
sessions - compare SHAs only within one log. Runtime needs CUDA 12.8 `cudart64_12.dll` and `cublas64_12.dll`
(`...\cuda-12.8.1\bin` on `PATH`).

**This backend has not been executed in this repository**: no model load, no GPU smoke test, no performance
measurement. Running it is a separate task.

## Unfinished / known gaps (honest list)

- Batching / Session integration is not wired in this import.
- The MTP transaction chain is not closed: T3 verify oracle, quality gates and state return paths remain open;
  the 2-true-vs-fixed-label-3 quality failure is unresolved. Do not present it as usable.
- Vendor pin→live provenance for `include/llama.h` / `src/llama-batch.cpp` is unverified (needs pure `b11476`).
- No GPU smoke test, no EOS/full-JSON regression, no performance numbers for this build.
- The imported tree keeps its own docs/AGENTS.md from the mini repo; they describe the mini tree, not this
  repository's process.
