# Autotuner foundation: device/model fingerprint + resource budget planner (P3-1)

Phase-3 opening item **P3-1** (branch `phase3/autotuner`, from the phase-2 head `954a88e4`).
Two tools and this document; **0 GPU, no lease, no engine run, no benchmark, no performance
claim of any kind**. `tools/strata_fingerprint.py` writes one sourced JSON fingerprint of the
host, the GPU and the model; `tools/strata_plan.py` turns that fingerprint plus a workload into
a capacity plan with two separate pools, ranked candidates and explicit rejections. The tools
are read-only: they read OS APIs, `nvidia-smi`, and the GGUF **header**; nothing is hashed,
nothing is measured beyond act-of-query readings.

Files:

| File | What it is |
|---|---|
| `backends/qwen35_llama/tools/strata_fingerprint.py` | the fingerprint writer (`--model`, workload, `--out`) |
| `backends/qwen35_llama/tools/strata_plan.py` | the budget planner (`--fingerprint` + workload, `--mode`, `--out`) |
| `backends/qwen35_llama/tools/test_strata_autotune.py` | 0-GPU unit tests (19, synthetic fingerprints / text) |

```text
python backends/qwen35_llama/tools/strata_fingerprint.py --model <gguf> \
    [--context 4096] [--max-new 128] [--batch 1] [--sha <registered sha256>] [--out fp.json]

python backends/qwen35_llama/tools/strata_plan.py --fingerprint fp.json \
    [--context N] [--max-new N] [--batch 1] \
    [--mode safe|balanced|latency|throughput] [--safety-mib M] [--staging-mib M] \
    [--workspace-mib M] [--kv-bytes 2|4] [--free-mib X] \
    [--host-mode pageable|pinned] [--pinned-cap-mib M] [--out plan.json]
```

## 1. The fingerprint: fields, sources, and the no-guessing rule

Every leaf of the JSON is `{"value", "source"[, "note"]}`. The source vocabulary is carried in
the document itself (`source_legend`) and shown here:

| source | meaning |
|---|---|
| `nvidia-smi` | read from the NVIDIA driver CLI; a read-only query (name, sm, total/free VRAM, driver, the driver's max CUDA version) |
| `table` | a value in the tool's local table - currently the vendor-nominal memory bandwidth per GPU model; **no benchmark is run** |
| `gguf-meta` | read from the GGUF header; tensor bytes are computed from declared shapes and quantization block geometry (`gguf_reader.py`), never by reading the file's data |
| `os-api` | an OS API (ctypes `GlobalMemoryStatusEx` / `IsProcessorFeaturePresent` / `GetDriveTypeW`, `winreg`, `shutil`, `platform`); read-only |
| `fs` | a filesystem stat (size, mtime) - the file itself is never hashed here |
| `ask` | supplied by the caller (workload, registered SHA) |
| `policy` | a deliberate default of this toolchain, documented below; not a fact |
| `unknown` | not obtainable on this host/path - and deliberately **not guessed** |

Sections:

- **GPU**: `name`, `compute_capability`, `vram_total_mib`, `vram_free_mib` (flagged as an idle
  reading, not a booking), `driver_version`, `cuda_driver_max` (the driver's maximum supported
  CUDA version, explicitly not the runtime an engine links), `memory_bandwidth_gbps` (table).
  If `nvidia-smi` is absent or answers nothing, every field is `unknown`.
- **CPU**: `model_name` (registry `ProcessorNameString`), `physical_cores` (psutil; `unknown`
  if absent - the logical count is not divided to fake it), `logical_cores`, `isa`
  (`avx`/`avx2`/`avx512f` via `IsProcessorFeaturePresent`).
- **RAM**: total/available physical and the **commit limit** (= physical + pagefile) with its
  available part. The commit limit matters because the host pool may be pageable (below).
- **Disk**: the model volume's kind (`GetDriveTypeW`; rotational media is not probed),
  free/total - best effort, `unknown` with a note on failure.
- **Model**: path, `size_bytes`/`mtime_utc` (`fs`), `sha256` (only if a **registered** value was
  passed with `--sha`; otherwise `unknown` with the policy note - the tool never hashes), and
  the GGUF facts: architecture, `block_count`, `context_length`, `embedding_length`,
  `expert_count`, `expert_used_count` (top-k), expert and shared-expert feed-forward lengths,
  attention head/key/value geometry, `full_attention_interval`, the `ssm.*` GDN geometry,
  `nextn_predict_layers` and `has_mtp_block`, tensor count and bytes, routed-expert total and
  **per-layer expert bytes** (one entry per layer), non-expert total, and per-role quantization
  types and bytes (gate/up/down experts, shared experts, attention, GDN, embed/head).
- **Workload**: model path, context, max_new, batch (default 1 - the serve protocol's batching
  contract v1 is batch=1, P2-7) - all `ask`.

`gguf_reader.py`'s declared block geometry is the reference for tensor bytes; a tensor whose
shape or type the geometry table cannot cover would be reported with a zero-bytes note rather
than a guess (no such tensor exists in the two local files).

## 2. The plan: two pools, the VRAM chain, and the formulas

The plan treats **VRAM and host RAM as two independent budgets** - they are never added, and
the unit tests assert cross-changes do not leak (`test_two_pools_are_not_mixed`).

### 2.1 VRAM, in the engine's own order

The static chain, exactly as printed by the tool (all terms MiB):

```
start                       = vram_total                     (nvidia-smi)
- desktop_other_used        = vram_total - vram_free         (nvidia-smi, idle)
- non_expert_weights        = tensor_bytes_total - routed_expert_bytes_total   (gguf-meta)
- kv_gdn_state              (formulas below)                 (gguf-meta)
- workspace_graph           = 1024 by default                (policy, --workspace-mib)
- prefill_staging           = 1126 by default in safe/balanced (table: measured 1.10 GiB on
                              the official engine, tasks 341/342; mode defaults otherwise)
- safety_margin             = max(512, 6% of total) in safe   (policy; the task contract)
= leftover -> expert slots
```

**KV/GDN formulas** (`state_bytes()`):

```
attn_layers       = ceil(block_count / full_attention_interval)      # 41/4 -> 11 for this model
recurrent_layers  = block_count - attn_layers                        # 30
kv                = attn_layers x context x head_count_kv x (key_length + value_length) x kv_dtype
gdn per layer     = state_size^2 x time_step_rank x 4                # recurrent state (f32)
                  + conv_channels x (conv_kernel - 1) x 4            # conv state (f32)
conv_channels     = 2 x state_size x group_count + inner_size        # 8192 for this model
```

The GDN formula is the engine's own geometry, not an invention: see
`src/core/session.cpp` (`state_size^2 * ssm_v_heads` + `conv_channels * (d_conv-1)`) and
`include/strata/core/family_layout.hpp` (the `qwen35moe.*` key mapping, including
`conv_channels`). The KV dtype is an explicit assumption: default `--kv-bytes 2` (f16, the
official engine's default); **the local strata engine keeps f32 device KV** (`dalloc` float
`kc`/`vc` in `src/qwen35/q36_engine.cu:1231-1249`), so pass `--kv-bytes 4` for it. Both values
are tiny on this model (f16 at ctx4096: ~88 MiB; f32: ~176 MiB) - the formula is given so the
term is auditable, not because it dominates.

**Expert slots**: one quota step = one expert of one layer, averaged as
`routed_expert_bytes_total / expert_count` (official pin: 74.96 MiB/step; local IQ3_S:
49.18 MiB/step). `quota_per_layer = floor(leftover / step)`, clamped to `expert_count`; the
cache total is `quota/expert_count x routed_expert_bytes_total`. Per-layer expert blobs differ
slightly (gate/up/down mixes per layer - see `routed_expert_layer_bytes`); the planner uses the
average and says so.

### 2.2 The runtime clamp (`--free-mib`)

The official engine does not plan from `vram_total`; it decides the expert budget from the
**free VRAM at placement time** minus staging minus the safety margin, then floors the result
to whole experts per layer (recorded, tasks 341/342: free 2.07 GiB - 1.10 GiB staging - 512 MiB
=> ~0.47 GiB => 6 of 256 experts per layer = 0.44 GiB). `--free-mib X` reproduces exactly that
arithmetic (`free - staging - safety` -> quota), with `--staging-mib`/`--safety-mib` overrides,
and marks the effective budget source as `runtime-free`. Without `--free-mib` the output is the
**static upper bound**, and the difference between the two is the engine's own runtime
reservation, which no GGUF can describe (section 4).

### 2.3 Host RAM pool

Terms: `model_resident` (the model file's bytes - pack/resident; `fs`), `host_expert_cache`
(caller-added via `--host-extra-mib`; the local engine's COMPACT 2700-slot host-side resident
was **3199.2 MiB**, tasks 346/347), `prefill_staging`, `working_allowance` (1024 MiB policy).
`requirement = model_resident + host_expert_cache + staging + working_allowance`.

- **pageable (default)**: the arena may be backed by the pagefile, so the bound is the
  **commit limit** (physical + pagefile) minus an OS reserve (2048 MiB policy), not physical
  RAM. The pagefile **is counted** (`pagefile_counted: true`). Observed on this host: in-run
  `AvailPhys` minima of 14-367 MiB while the pagefile absorbed the peak (tasks 341/342/347) -
  survivable, and a known disk-paging cost that this planner does **not** quantify.
- **pinned**: no paging, but kernel-pinned allocations are size-sensitive on this host: the one
  recorded attempt, `cudaHostAlloc` of the 18.75 GiB expert arena, **failed** (tasks 341/342),
  which is why the official engine runs its host arena pageable. The planner therefore refuses
  `--host-mode pinned` unless the caller states an explicit `--pinned-cap-mib`, and lists the
  rejection with that recorded fact plus the pageable fallback.

### 2.4 Candidates, modes, and rejections

Candidates vary in the sensitivity order the task names - **expert quota -> threads/CPU pool ->
prefill ubatch/context** - and each carries a `reason` and a `risk`; they are ranked per the
mode's stated preference. Modes change only the allocation strategy of the leftover:

| mode | safety margin | staging | ranking preference |
|---|---|---|---|
| `safe` (default) | max(512, 6%) | 1126 MiB (measured) | expert slots first, balanced |
| `balanced` | max(512, 4%) | 1126 MiB | balanced first |
| `latency` | max(512, 6%) | 512 MiB | maximize resident experts |
| `throughput` | max(512, 6%) | 1536 MiB | staging/prefill first, then slots |

No mode promises speed; the differences are only what the leftover is spent on. Out-of-bounds
combinations are **rejected explicitly and listed** (never silently dropped): all-experts-per-
layer (with the over-budget MiB count), `batch>1` (batching contract v1), `context >
context_length`, `max_new > context`, a ubatch larger than the context, a zero-slot budget, and
pinned host mode (above). Example from the anchor run: "all 256 experts of every layer need
19190.0 MiB; the effective expert budget is 482.0 MiB (over by 18708.0 MiB)".

## 3. Self-check anchors (reproduced, not tuned)

Recorded facts used as anchors (local project ledger, tasks 341/342/346/347):

- Official pin (UD-Q4_K_XL MTP, `55983c5a...`, 22,853,663,008 B): 41 layers, 256 experts,
  routed experts 20,122,173,440 B (18.74 GiB), dense 2,720,498,176 B; on the 8,188 MiB card the
  engine clamped the expert cache to **6 of 256 experts per layer = 0.44 GiB at ctx4096** and
  **3 of 256 = 0.22 GiB at ctx8192** (free 2.07 / 1.88 GiB at placement, staging 1.10 GiB,
  margin 512 MiB).
- Local experimental MTP model (IQ3_S `ab639a7f...`, 15,346,432,288 B): runs with a pageable
  host arena (AvailPhys minima 14-367 MiB); COMPACT 2700-slot host-side resident 3199.2 MiB.
- `cudaHostAlloc` 18.75 GiB failed on this host => the arena is pageable.

Run on this host (fingerprint outputs and plan outputs saved under
`results/opencode-execution/strata-unified-p3-1-fingerprint/` in the main workspace):

```text
strata_fingerprint.py --model .../official-5bc3e238/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --sha 55983c5a...
strata_plan.py --fingerprint fp-official-pin.json --context 4096 --free-mib 2120
  -> expert_slot: quota_per_layer 6, cache 0.439 GiB   (recorded: 6/256, 0.44 GiB)
strata_plan.py --fingerprint fp-official-pin.json --context 8192 --free-mib 1925
  -> expert_slot: quota_per_layer 3, cache 0.220 GiB   (recorded: 3/256, 0.22 GiB)
strata_plan.py --fingerprint fp-mtp-iq3s.json --context 4096
  -> host: requirement 16,785.5 MiB <= 32,448.5 MiB physical, commit limit 44,810.0 MiB,
     fits pageable (the 15.3 GiB model + staging + working allowance)
```

The VRAM anchor matches the measurement's order of magnitude exactly (it reproduces the
recorded placement, 0.44 GiB, from the recorded free reading) and the host anchor fits the
15.3 GiB model + staging pageable. Without `--free-mib` the same official-pin fingerprint at
ctx4096 gives the static upper bound **25 experts/layer, 1.83 GiB**: the static chain
(start 8188 - desktop 891 - dense 2594.5 - KV/GDN 150.8 - workspace 1024 = 3527.7 MiB
"free-before-staging") sits ~1408 MiB above the recorded placement-time free (2120 MiB). That
difference is the engine's runtime consumption the static model cannot see; the planner's
1024 MiB workspace placeholder covers 1024 of it and the remaining ~384 MiB has no term
(compute buffer growth, MTP/draft buffers, allocator slack). No parameter was tuned to close
it: the recorded runtime free, when supplied, is authoritative, and the static term is
documented as an upper bound.

## 4. What this is NOT, and what is not implemented

- **容量自适应 != 性能自调优**: this foundation estimates *what fits* (capacity). It is not a
  search for the fastest configuration, and P3-1 produced no speed/benefit conclusion. The
  official `--expert-cache auto`, dynamic swap and "VRAM elastic" behaviors are *runtime
  capacity responses* of an engine to free memory; the planner is an **offline** capacity model
  that predicts and documents them. Neither is a performance autotuner; this tool does not
  even claim the predicted placement is faster than a smaller one.
- **Not implemented** (deliberately out of P3-1): a real offline search over configurations
  (P3-2), online/adaptive tuning during serving, and cross-GPU/cross-machine validation
  (all local numbers are for the single RTX 4060 Laptop 8 GiB host and are not converted to
  other GPUs or quantizations).
- The `workspace_graph` term is a policy placeholder, not a measurement; the compute buffer's
  true size depends on the engine, batch/ubatch and graph shape. Treat static expert-slot
  numbers as upper bounds and prefer `--free-mib` with a recorded placement reading.
- The planner models batch=1 (contract v1). Continuous batching would change both pools.

## 5. Verification

`python backends/qwen35_llama/tools/test_strata_autotune.py -v` -> **19 tests, all pass**
(0 GPU; synthetic fingerprints and text only). They cover: every fingerprint leaf has a
vocabulary source and `None` is never guessed; nvidia-smi parse + table bandwidth; GGUF
summarization (experts per layer, MTP flag, per-role quantization); the SHA policy (registered
value or unknown, never hashed); the static chain order; budget monotonicity in context
(KV up, slot budget down); the ctx4096/ctx8192 runtime-clamp anchors (quota 6 = 0.439 GiB and
quota 3 = 0.220 GiB); explicit rejections (all-experts, batch>1, over-context, max_new>context,
pinned host, zero-slot); the pageable host fit (including the 15.3 GiB model); the two pools
not mixing; mode changes affecting only allocation; and the refusal to plan with an unknown
VRAM total. Test output is saved beside the anchor JSONs in the evidence directory.
