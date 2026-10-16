# Balanced600 synthetic-reference comparison

600 fixed cases,150 per reference risk,1800 method records. Seed2026101609.
Reference labels are Jev `synthetic_design_label`; no human accuracy claim.
Anonymous `case_index` joins three measurements, never the original source ID.
No source messages, original case/conversation IDs, or input-token lists are public.

`observations.csv` contains actual counts, DONE finishes, observer E2E/native
Prefill/Decode, prompt hashes and predictions. `results.json` contains ordinal
matrices, paired conversation-cluster bootstrap, sampled resources, checkpoint
hash indexes and the matched580-case warm comparison. All600 measurements stay
in the primary tables; the warm subset excludes entire first-request triples
using the registered order-only rule.

Run `verify_risk_campaign_evidence.py` with this directory for independent
arithmetic/SHA checks. Original native logs remain local; public verification
does not claim to independently replay those private inputs or assess human risk.
Full report: `docs/P4_BALANCED600_COMPARISON_20261016.md`.

`explanation-repair.json` adds24 successful mixed flows with12 actual nonempty EOS explanations, real label commits and cursor checks. The verifier now checks127 conditions. Old inline NO-GO originals remain unchanged.
