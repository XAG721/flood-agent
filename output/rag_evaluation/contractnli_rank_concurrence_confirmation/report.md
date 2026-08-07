# ContractNLI rank-concurrence confirmation (v52)

- Status: `CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED`
- Cases/documents: 600/319
- Candidate utility F1: 0.250304
- Candidate evidence F1: 0.170455
- Missing abstention accuracy: 0.410000
- Abstention rate: 0.373333
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Family comparisons

- `candidate_minus_strongest_shared_gate_non_frc`: +0.009038 (95% CI [-0.006814, +0.024925])
- `candidate_minus_ungated_v49`: +0.081783 (95% CI [+0.047778, +0.117474])
- `candidate_minus_strongest_ungated_frozen_frc`: +0.081783 (95% CI [+0.047778, +0.117474])
- `candidate_minus_native_zero_control`: +0.081783 (95% CI [+0.047778, +0.117474])

## Support checks

- `exact_cases_equals_600`: `true`
- `candidate_ceiling_complete_rate_on_evidence_cases_equals_1`: `true`
- `not_mentioned_official_empty_evidence_rate_equals_1`: `true`
- `candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01`: `false`
- `candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0`: `false`
- `candidate_minus_ungated_v49_point_at_least_0_01`: `true`
- `candidate_minus_ungated_v49_ci_low_above_0`: `true`
- `candidate_minus_native_zero_control_point_at_least_0_02`: `true`
- `candidate_minus_native_zero_control_ci_low_above_0`: `true`
- `evidence_bearing_macro_f1_drop_vs_ungated_v49_at_most_0_03`: `false`
- `not_mentioned_abstention_accuracy_at_least_0_2`: `true`
- `not_mentioned_improvement_vs_native_zero_at_least_0_2`: `true`
- `candidate_abstention_rate_at_least_0_05`: `true`
- `candidate_abstention_rate_at_most_0_8`: `true`
- `every_budget_and_supported_stratum_delta_at_least_minus_0_03`: `true`
- `deterministic_query_fallback_rate_equals_0`: `true`

## Boundary

This is a one-time, role-reversed ContractNLI train confirmation of a parameter-free full-contract span-selection guard. It is not an official leaderboard result, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain expert validation.
