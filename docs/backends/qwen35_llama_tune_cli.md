# Offline tune CLI and persistent profiles (P3-4)

The equivalent of `strata tune` is `python tools/strata_tune.py tune`. It combines
P3-1's capacity planner with an optional P3-2 diagnostic screen, saves a research
profile, and checks its identity before reuse. **No engine launch, GPU allocation,
benchmark, service restart, default change or automatic configuration application.**
The default installation/server paths do not import this tool.

## Usage

Prepare two UTF-8 JSON files. `workload.json` contains at least:

```json
{
  "context": 8192,
  "max_new": 128,
  "batch": 1,
  "manifest_sha256": "<64 lowercase hex characters from your workload manifest>",
  "protocol": "diagnostic-cli",
  "mtp_k": 1,
  "sampling": {"temperature": 0}
}
```

The external workload manifest must identify fixture token files, chat template,
sampling/stop settings, workload proportions, MTP mode/K, engine environment,
expert-profile/pack hashes and runtime dependencies. Compute its SHA once and pass
that value; it is **caller registered**, not independently verified by the CLI.
All additional workload JSON fields are also bound, so a change invalidates reuse.
This makes missing registration visible without reading model payloads or guessing
that two prompts with the same display name are identical.

`baseline.json` is your **existing** configuration object, retained as a manual
fallback (for example `{"slots":2700,"compact":true,"threads":8}`). This does not
certify that baseline or install it. Production 3200 defaults remain unchanged.

```text
python tools/strata_tune.py tune --model /path/model.gguf --model-sha REGISTERED_SHA256 \
  --engine /path/strata.exe --runtime-id CUDA_DLL_MANIFEST_SHA256 \
  --workload workload.json --baseline baseline.json --mode safe --kv-bytes 4 \
  --evidence small-compare.json --out research-profile.json

python tools/strata_tune.py check --profile research-profile.json \
  --model /path/model.gguf --model-sha REGISTERED_SHA256 --engine /path/strata.exe \
  --runtime-id CUDA_DLL_MANIFEST_SHA256 --workload workload.json --mode safe --kv-bytes 4
```

Line continuations above are schematic; pass as one line on PowerShell. `--model`
reads the GGUF header/stat and probes OS/`nvidia-smi` only. It never hashes the full
model. The engine file and small evidence artifacts are hashed, never executed.
Evidence entries are limited to 16 MiB to avoid accidentally indexing model payloads.
`--fingerprint fp.json` replaces `--model` for offline/synthetic work; that compares
only the supplied snapshot and **cannot attest to the current machine**. Old P3-1
fingerprints lack GPU UUID and require regeneration/enrichment, not a wildcard.

Modes `safe` (default), `balanced`, `latency`, `throughput` retain P3-1's allocation
policies. They are capacity preferences, not measured performance objectives.
`--kv-bytes 4` is the migrated backend's f32 KV; official engine f16 uses 2.
The planner's expert quotas are estimates; they are not direct local COMPACT-slot
settings and are not applied to the fallback. A zero-slot/negative budget or an
over-budget host plan is rejected even if P3-1's status field remains `ok`.

## Schema and invalidation

Schema: [strata_tune_profile.schema.json](strata_tune_profile.schema.json).

| Field | Meaning |
|---|---|
| `status`, `selected_configuration` | Always `research-only`, `null`; this version cannot certify a measured winner |
| `identity` | GPU UUID/name/architecture/VRAM/driver/CUDA-driver limit, CPU, RAM/commit limit, OS; model registered SHA/size/mtime/entire GGUF metadata; actual engine SHA; caller runtime ID; tool source hashes; workload and policy |
| `fingerprint`, `capacity_plan` | Full acquisition provenance and both independent memory budgets |
| `diagnostic_screen` | Optional historical local calibration, never automatic selection |
| `fallback` | Caller configuration, source/confidence, `auto_apply=false` |
| `evidence` | Absolute local paths + size/SHA of comparison/raw/manifest artifacts; external raw need not be in Git |
| `confidence` | Capacity=estimate; speed/quality=unverified; cross-hardware=not validated |
| `identity_sha256`, `content_sha256` | Canonical sorted UTF-8 JSON SHA256; content hash excludes its own field; corruption checks, **not signatures/authentication** |

`check` compares identity field by field, verifies checksums/evidence and recalculates
capacity from the supplied current fingerprint. Idle free VRAM/RAM and acquisition
timestamps are observations, not identity keys; resource changes still affect the
current capacity result. A changed GPU, driver, model, binary, tool source, workload,
runtime ID, mode or KV dtype invalidates the record. Unknown required identity fields
make reuse fail closed. A matching record remains research-only: it does not reserve
memory, validate state/quality/EOS or authorize a production switch.

Exit codes: **0** = record written/matching research identity and estimated capacity;
**2** = invalid input, incomplete/stale identity, rejected capacity, changed/missing
evidence or I/O error. `tune` still emits a research record for an incomplete/rejected
plan when the planner can describe it; check the exit code. `--out` creates a **new**
file exclusively; existing files are never overwritten, including the input profile.

## Historical screen is explicitly opt-in

`--diagnostic-screen` only accepts the recorded RTX 4060 Laptop + registered
`ab639a7f…` IQ3_S identity. It shows the fixed 347 **256-token diagnostic model**,
not a prediction validated for the user's requested output shape or another binary.
No calibration-build equivalence is asserted. Other devices/quantizations can use
the capacity planner; importing this calibration is refused.

The output carries P3-3 counterevidence: the 2200-slot candidate lost S/L paired
comparisons (−5.171% / −17.524%) on `c3780a6a…`, and is **NO-GO for that pilot**.
Those facts are not extrapolated to a new binary/device or converted into a new
performance estimate. P3-2 `ok` means model/capacity classification, **not GPU PASS**.
No mode selects the top-ranked prediction, and no profile is applied to serving.

## Verification

```text
python -B -m unittest discover -s backends/qwen35_llama/tools -p "test_strata_*.py" -v
```

CPU-only tests exercise profile persistence, corruption, every identity class,
unknown identity, changed/missing evidence, workload/mode changes, capacity changes,
CLI exits/no-overwrite, and refusal to transfer the local calibration. No new GPU
speed measurements were made for P3-4. See [phase-three delivery](../PHASE3_DELIVERY.md)
for the product acceptance gaps.
