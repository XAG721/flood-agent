# SQuAD 2.0 generative answerability gate (development, v58)

- Status: `SQUAD2_V58_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- Cases/articles/paragraphs: 600/299/590
- Verifier balanced accuracy: 0.710000
- Answer-bearing pass rate: 0.560000
- No-answer rejection rate: 0.860000
- Candidate utility F1: 0.672593
- Candidate answer F1: 0.485185
- Candidate no-answer abstention: 0.860000
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `gated_exact_anchor_minus_ungated_exact_anchor`: +0.331706 (95% CI [+0.276597, +0.387496])
- `candidate_minus_exact_anchor_gated_ablation`: +0.120569 (95% CI [+0.100411, +0.141741])
- `candidate_minus_strongest_shared_gate_non_frc`: +0.107759 (95% CI [+0.088160, +0.127872])
- `candidate_minus_strongest_same_gate_frc`: -0.004444 (95% CI [-0.007778, -0.001667])

## Support checks

- `exact_cases_equals_600`: `true`
- `exact_answer_state_balance`: `true`
- `minimum_paragraphs_at_least_250`: `true`
- `minimum_articles_for_stage`: `true`
- `paragraph_cap_at_most_2`: `true`
- `article_state_cap_at_most_12`: `true`
- `schema_exclusion_rate_at_most_0_01`: `true`
- `candidate_ceiling_complete_rate_at_least_0_99`: `true`
- `support_verifier_balanced_accuracy_at_least_0_75`: `false`
- `answer_bearing_support_pass_rate_at_least_0_75`: `false`
- `no_answer_support_rejection_rate_at_least_0_75`: `true`
- `malformed_or_fallback_rate_equals_0`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1`: `true`
- `gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0`: `true`
- `candidate_minus_exact_anchor_gated_ablation_point_at_least_0_005`: `true`
- `candidate_minus_exact_anchor_gated_ablation_ci_low_above_0`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `true`
- `candidate_minus_strongest_same_gate_frc_point_at_least_0_005`: `false`
- `candidate_minus_strongest_same_gate_frc_ci_low_above_0`: `false`
- `answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1`: `false`
- `candidate_no_answer_abstention_accuracy_at_least_0_75`: `true`
- `candidate_abstention_rate_at_least_0_25`: `true`
- `candidate_abstention_rate_at_most_0_75`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_02`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_fallback_rate_equals_0`: `true`

## Boundary

This is a balanced closed-paragraph SQuAD 2.0 mechanism experiment. It is not official answer-string scoring, hidden-test evaluation, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain validation.
