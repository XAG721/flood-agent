# QuAC anchor-safe consensus slot (development, v57)

- Status: `QUAC_V57_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION`
- Cases/documents/dialogues: 600/554/578
- Candidate utility F1: 0.114109
- Candidate answer F1: 0.161552
- Candidate answer recall: 0.424537
- No-answer abstention accuracy: 0.066667
- Single-slot trigger rate: 0.085000
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `candidate_minus_exact_anchor_ablation`: +0.000317 (95% CI [-0.001667, +0.002381])
- `candidate_minus_strongest_shared_gate_non_frc`: +0.000159 (95% CI [-0.002336, +0.002682])
- `candidate_minus_strongest_same_gate_frc`: +0.014943 (95% CI [-0.000381, +0.029848])

## Support checks

- `exact_cases_equals_600`: `true`
- `exact_answer_state_balance`: `true`
- `minimum_dialogues_at_least_250`: `true`
- `minimum_documents_at_least_200`: `true`
- `dialogue_cap_at_most_2`: `true`
- `document_state_cap_at_most_4`: `true`
- `schema_exclusion_rate_at_most_0_01`: `true`
- `candidate_ceiling_complete_rate_at_least_0_99`: `true`
- `single_slot_trigger_rate_at_least_0_1`: `false`
- `single_slot_trigger_rate_at_most_0_8`: `true`
- `candidate_minus_exact_anchor_ablation_point_at_least_0_005`: `false`
- `candidate_minus_exact_anchor_ablation_ci_low_above_0`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `false`
- `candidate_minus_strongest_same_gate_frc_point_at_least_0_005`: `true`
- `candidate_minus_strongest_same_gate_frc_ci_low_above_0`: `false`
- `answer_recall_drop_vs_exact_anchor_ablation_at_most_0_01`: `true`
- `no_answer_abstention_accuracy_at_least_0_5`: `false`
- `answer_bearing_gate_pass_rate_at_least_0_5`: `true`
- `candidate_abstention_rate_at_least_0_1`: `false`
- `candidate_abstention_rate_at_most_0_8`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_02`: `true`
- `deterministic_decoy_fallback_rate_equals_0`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_or_prediction_fallback_rate_equals_0`: `true`

## Boundary

This is a balanced QuAC evidence-unit selection mechanism experiment. It is not official QuAC answer extraction, hidden-test evaluation, SetR reproduction, selector adoption, or flood-domain validation.
