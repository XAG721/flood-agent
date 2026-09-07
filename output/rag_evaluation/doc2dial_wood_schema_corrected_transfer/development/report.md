# Doc2Dial wOOD schema-corrected transfer (development, v56)

- Status: `DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- Cases/documents/dialogues: 400/261/386
- No-answer subtypes: {"ood_act": 176, "other_empty": 24}
- Candidate utility F1: 0.517417
- Candidate answer F1: 0.159833
- Candidate answer recall: 0.123750
- No-answer abstention accuracy: 0.875000
- Answer-bearing gate pass rate: 0.650000
- No-answer gate pass rate: 0.125000
- Retrieval/gate/selector changed from v54: `false`
- Confirmation opened before gates: `false`
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `candidate_minus_strongest_shared_gate_non_frc`: -0.008827 (95% CI [-0.021324, +0.003586])
- `candidate_minus_same_gate_frc_ablation`: -0.000381 (95% CI [-0.004858, +0.003528])
- `candidate_minus_ungated_v49`: +0.414071 (95% CI [+0.360481, +0.469159])
- `candidate_minus_strongest_ungated_frozen_frc`: +0.409153 (95% CI [+0.353870, +0.465920])

## Support checks

- `exact_cases_equals_400`: `true`
- `exact_answer_state_balance`: `true`
- `minimum_dialogues_at_least_160`: `true`
- `minimum_documents_at_least_80`: `true`
- `dialogue_cap_at_most_2`: `true`
- `document_state_cap_at_most_4`: `true`
- `schema_exclusion_rate_at_most_0_01`: `true`
- `candidate_ceiling_complete_rate_at_least_0_99`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `false`
- `candidate_minus_same_gate_frc_ablation_point_at_least_0_005`: `false`
- `candidate_minus_same_gate_frc_ablation_ci_low_above_0`: `false`
- `candidate_minus_ungated_v49_point_at_least_0_01`: `true`
- `candidate_minus_ungated_v49_ci_low_above_0`: `true`
- `answer_macro_recall_drop_vs_ungated_v49_at_most_0_03`: `false`
- `no_answer_abstention_accuracy_at_least_0_5`: `true`
- `answer_bearing_gate_pass_rate_at_least_0_5`: `true`
- `candidate_abstention_rate_at_least_0_1`: `true`
- `candidate_abstention_rate_at_most_0_8`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_03`: `false`
- `deterministic_decoy_fallback_rate_equals_0`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_or_prediction_fallback_rate_equals_0`: `true`

## Boundary

v56 disclosed pre-outcome train schema access caused by v55. It changed only schema normalization and the official empty-reference interpretation; it is not an official shared-task result, SetR reproduction, flood-domain validation or rollout authorization.
