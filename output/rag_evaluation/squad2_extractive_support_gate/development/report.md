# SQuAD 2.0 extractive support gate (development, v59)

- Status: `SQUAD2_V59_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- Cases/articles/paragraphs: 600/309/592
- Verifier balanced accuracy: 0.663333
- Answer-bearing span pass rate: 0.426667
- No-answer rejection rate: 0.900000
- Invalid-output rate: 0.503333
- Candidate utility F1: 0.631667
- Candidate answer F1: 0.363333
- Candidate no-answer abstention: 0.900000
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `gated_exact_anchor_minus_ungated_exact_anchor`: +0.327259 (95% CI [+0.265413, +0.387175])
- `candidate_minus_gated_exact_anchor`: +0.087934 (95% CI [+0.068810, +0.107662])
- `candidate_minus_strongest_shared_gate_non_frc`: +0.080230 (95% CI [+0.061301, +0.099592])
- `candidate_minus_strongest_same_gate_frc`: +0.001389 (95% CI [+0.000000, +0.003252])

## Support checks

- `exact_cases_equals_600`: `true`
- `exact_answer_state_balance`: `true`
- `selected_v58_commitment_overlap_equals_0`: `true`
- `minimum_paragraphs_at_least_250`: `true`
- `minimum_articles_for_stage`: `true`
- `paragraph_cap_at_most_2`: `true`
- `article_state_cap_at_most_12`: `true`
- `schema_exclusion_rate_at_most_0_01`: `true`
- `candidate_ceiling_complete_rate_at_least_0_99`: `true`
- `support_verifier_balanced_accuracy_at_least_0_75`: `false`
- `answer_bearing_span_pass_rate_at_least_0_75`: `false`
- `no_answer_rejection_rate_at_least_0_65`: `true`
- `invalid_output_rate_at_most_0_02`: `false`
- `gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0`: `true`
- `candidate_minus_gated_exact_anchor_point_at_least_0_05`: `true`
- `candidate_minus_gated_exact_anchor_ci_low_above_0`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_05`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `true`
- `candidate_minus_strongest_same_gate_frc_point_at_least_minus_0_005`: `true`
- `answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1`: `false`
- `candidate_no_answer_abstention_accuracy_at_least_0_65`: `true`
- `candidate_abstention_rate_at_least_0_2`: `true`
- `candidate_abstention_rate_at_most_0_7`: `false`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_02`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_fallback_rate_equals_0`: `true`

## Boundary

This is a balanced closed-paragraph SQuAD 2.0 mechanism experiment on a v58-disjoint sample. It is not official answer-string scoring, hidden-test evaluation, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain validation.
