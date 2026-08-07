# SQuAD 2.0 dual support union (development, v61)

- Status: `SQUAD2_V61_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- Cases/articles/paragraphs: 600/309/590
- Verifier balanced accuracy: 0.748333
- Answer-bearing span pass rate: 0.723333
- No-answer rejection rate: 0.773333
- Invalid-output rate: 0.011667
- Candidate utility F1: 0.701389
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `gated_exact_anchor_minus_ungated_exact_anchor`: +0.325458 (95% CI [+0.274404, +0.375616])
- `candidate_minus_gated_exact_anchor`: +0.163624 (95% CI [+0.139499, +0.187655])
- `candidate_minus_strongest_shared_gate_non_frc`: +0.143495 (95% CI [+0.120378, +0.166998])
- `candidate_minus_strongest_same_gate_frc`: +0.007778 (95% CI [+0.003988, +0.012048])

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
- `invalid_output_rate_at_most_0_02`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0`: `true`
- `candidate_minus_gated_exact_anchor_point_at_least_0_05`: `true`
- `candidate_minus_gated_exact_anchor_ci_low_above_0`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_05`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `true`
- `candidate_minus_strongest_same_gate_frc_point_at_least_minus_0_005`: `true`
- `candidate_abstention_rate_at_most_0_7`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_02`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_fallback_rate_equals_0`: `true`
- `selected_prior_commitment_overlap_equals_0`: `true`
- `answer_bearing_union_pass_rate_at_least_0_8`: `false`
- `no_answer_union_rejection_rate_at_least_0_6`: `true`
- `binary_malformed_rate_equals_0`: `true`
- `structured_invalid_rate_at_most_0_02`: `true`
- `candidate_answer_recall_at_least_0_75`: `false`
- `candidate_no_answer_abstention_accuracy_at_least_0_6`: `true`
- `candidate_abstention_rate_at_least_0_15`: `true`

## Boundary

This is a third disjoint balanced SQuAD 2.0 mechanism sample. It is not official answer-string scoring, hidden-test evaluation, SetR reproduction, selector adoption, or flood-domain validation.
