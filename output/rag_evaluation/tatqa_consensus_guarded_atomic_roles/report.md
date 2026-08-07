# TAT-QA consensus-guarded atomic-role experiment (v46)

- Status: `TATQA_FULL_CONTEXT_POOL_INCONCLUSIVE`
- Stage: `POST_ACCESS_STRUCTURAL_CENSUS_PRE_QUERY`
- Registered source: official revision `870accc41953dcde885aabeb963d94aabdc0fbc3`
- License: MIT, checked before opening the registered data member
- Gate 2: `NO-GO/SHADOW`

## Why the experiment stopped

The protocol required every selected question to expose an exact official mapping to table-cell or paragraph candidates. The official raw dev split contains 1,668 questions but exposes no `mapping` field; the official TAGOP dev member also exposes no evidence mapping. Consequently, zero questions satisfy the registered gold-evidence contract.

The raw split contains answer, derivation, answer-source and relevant-paragraph annotations, but deriving table pseudo-gold from those fields would change the registered endpoint after data access and leak answer information into the evidence labels. The workflow therefore stopped before query generation, neural scoring, gold joining, metric computation or bootstrap.

## What this result does and does not mean

This is a dataset-contract failure, not a negative result for the consensus-guarded selector. No selector comparison exists. The v46 formula remains validated only by six synthetic invariance tests and must not be tuned or claimed effective from this closure.

The next admissible experiment must use a newly frozen public dataset with explicit gold evidence annotations. TAT-QA cases, answers and derivations may not be reused to retrofit the method or construct pseudo-gold. Gate 2 remains `NO-GO/SHADOW`.
