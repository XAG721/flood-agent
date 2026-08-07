# ContractNLI robust consensus development calibration (v51)

- Status: `CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN`
- Cases/documents: 240/60
- Train open authorized: `false`
- Gate 2: `NO-GO/SHADOW`

## Five-fold OOF result

- Ungated v49 utility F1: 0.154504
- OOF candidate utility F1: 0.191845
- OOF utility gain: 0.037341
- OOF evidence F1 drop: 0.037738
- OOF missing abstention accuracy: 0.187500
- OOF abstention rate: 0.141667
- Final threshold hex: `0x1.3c887b71d53bcp+3`

## Development checks

- `exact_cases_equals_240`: `true`
- `exact_cases_per_label_equals_80`: `true`
- `minimum_documents_met`: `true`
- `all_five_folds_nonempty`: `true`
- `oof_candidate_minus_ungated_v49_utility_at_least_0_02`: `true`
- `oof_not_mentioned_abstention_accuracy_at_least_0_2`: `false`
- `oof_evidence_bearing_f1_drop_at_most_0_03`: `false`
- `oof_candidate_abstention_rate_at_least_0_05`: `true`
- `oof_candidate_abstention_rate_at_most_0_8`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`

## Boundary

This supervised dev result is not confirmation, does not authorize selector adoption, and may open the previously untouched train split only when every registered development check passes.
