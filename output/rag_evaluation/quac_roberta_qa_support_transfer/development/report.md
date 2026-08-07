# QuAC RoBERTa QA support transfer (development, v62)

- Status: `QUAC_V62_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION`
- Cases/documents/dialogues: 600/566/587
- Verifier balanced accuracy: 0.590000
- Answer-bearing support pass rate: 0.686667
- No-answer rejection rate: 0.493333
- Candidate utility F1: 0.308444
- Candidate answer recall: 0.139722
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `gated_exact_anchor_minus_ungated_exact_anchor`: +0.219598 (95% CI [+0.184910, +0.255070])
- `candidate_minus_gated_exact_anchor`: -0.013077 (95% CI [-0.027091, +0.001271])
- `candidate_minus_strongest_shared_gate_non_frc`: -0.013077 (95% CI [-0.027091, +0.001271])
- `candidate_minus_strongest_same_gate_frc`: -0.013272 (95% CI [-0.026309, -0.000087])

## Support checks

- `exact_cases_equals_600`: `true`
- `exact_answer_state_balance`: `true`
- `selected_v57_commitment_overlap_equals_0`: `true`
- `minimum_dialogues_at_least_250`: `true`
- `minimum_documents_at_least_200`: `true`
- `dialogue_cap_at_most_2`: `true`
- `document_state_cap_at_most_4`: `true`
- `schema_exclusion_rate_at_most_0_01`: `true`
- `candidate_ceiling_complete_rate_at_least_0_99`: `true`
- `qa_invalid_rate_equals_0`: `true`
- `support_verifier_balanced_accuracy_at_least_0_75`: `false`
- `answer_bearing_support_pass_rate_at_least_0_75`: `false`
- `no_answer_rejection_rate_at_least_0_65`: `false`
- `candidate_answer_recall_at_least_0_70`: `false`
- `candidate_no_answer_abstention_accuracy_at_least_0_65`: `false`
- `candidate_abstention_rate_at_least_0_15`: `true`
- `candidate_abstention_rate_at_most_0_70`: `true`
- `candidate_minus_gated_exact_anchor_point_at_least_0_005`: `false`
- `candidate_minus_gated_exact_anchor_ci_low_above_0`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `false`
- `candidate_minus_strongest_same_gate_frc_point_at_least_minus_0_005`: `false`
- `gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_02`: `false`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_fallback_rate_equals_0`: `true`

## Boundary

This is an out-of-domain QuAC component-transfer and evidence-selection mechanism experiment. The QA model was trained on SQuAD2. This is not official QuAC answer extraction, official SQuAD2 evaluation, SetR reproduction, open-domain retrieval, selector adoption, or flood-domain validation.
