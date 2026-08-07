# CUAD Top-3 rank-concurrence role closure (development, v53)

- Status: `CUAD_V53_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TEST`
- Cases/contracts: 400/259
- Candidate utility F1: 0.046032
- Candidate answer F1: 0.087063
- Candidate answer recall: 0.130000
- No-answer abstention accuracy: 0.005000
- Abstention rate: 0.007500
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `candidate_minus_strongest_shared_gate_non_frc`: -0.023731 (95% CI [-0.040711, -0.006903])
- `candidate_minus_same_gate_frc_ablation`: +0.004032 (95% CI [-0.001032, +0.010042])
- `candidate_minus_ungated_v49`: +0.006532 (95% CI [+0.000050, +0.014858])
- `candidate_minus_strongest_ungated_frozen_frc`: +0.002264 (95% CI [-0.005495, +0.010778])

## Support checks

- `exact_cases_equals_400`: `true`
- `exact_answer_state_balance`: `true`
- `minimum_contracts_at_least_90`: `true`
- `candidate_ceiling_complete_rate_at_least_0_9`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `false`
- `candidate_minus_same_gate_frc_ablation_point_at_least_0_005`: `false`
- `candidate_minus_same_gate_frc_ablation_ci_low_above_0`: `false`
- `candidate_minus_ungated_v49_point_at_least_0_01`: `false`
- `candidate_minus_ungated_v49_ci_low_above_0`: `true`
- `answer_macro_recall_drop_vs_ungated_v49_at_most_0_03`: `true`
- `no_answer_abstention_accuracy_at_least_0_2`: `false`
- `candidate_abstention_rate_at_least_0_05`: `false`
- `candidate_abstention_rate_at_most_0_8`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_03`: `false`
- `deterministic_query_fallback_rate_equals_0`: `true`
- `score_or_prediction_fallback_rate_equals_0`: `true`

## Boundary

This is a balanced, full-given-contract CUAD clause-selection mechanism experiment. It is not an official CUAD leaderboard result, open-corpus retrieval, legal advice, SetR reproduction, selector adoption, or flood-domain expert validation.
