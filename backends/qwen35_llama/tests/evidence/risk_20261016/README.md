# P4 development observations

Only anonymized statistics and original-file SHA indexes are public. No source
messages, original sample/conversation IDs or input-token lists are included.
The reference labels are AI-generated design labels, NOT human adjudicated gold.

`observations.json` contains the four-case paired measurements and twelve mixed
flow measurements. `results.json` keeps the actual observer values, native lease
receipts, memory sampling, fixed-P timings and the corrected empty-explanation
failure counts. All eight requested explanations ended with EOS only. The old
private driver summary incorrectly counted a parseable risk as full success;
that original is retained locally. The corrected result is NO-GO, zero completed
explanation flows, no throughput-gain claim.

Run `python backends/qwen35_llama/tools/verify_risk_evidence.py` with this directory.
It checks 46 arithmetic/SHA conditions. This verifies the public redacted
derivation, not private native logs or human accuracy. Private original locations
and the full limits are in `docs/PHASE4_LOGITS_QUALITY_AND_MIXED_LOAD.md`.
