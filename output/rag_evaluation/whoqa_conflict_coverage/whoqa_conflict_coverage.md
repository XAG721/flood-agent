# WhoQA independent conflict-viewpoint selection audit

- Status: `WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED`
- Cases: 5152
- Selection runs: 161742
- Every available public question template is scored and averaged within q_id.
- Gold answer mappings are joined only after selector context IDs are frozen.
- Gate 2 remains `NO-GO/SHADOW`.

## Aggregate selection metrics

| Method | Normalized viewpoint coverage | Raw viewpoint recall | Complete disclosure | >=2 viewpoints | Budget shortfall |
|---|---:|---:|---:|---:|---:|
| bm25_topk | 0.995203 | 0.929376 | 0.986548 | 0.999884 | 0.000000 |
| dense_topk | 0.994775 | 0.928974 | 0.985234 | 0.999845 | 0.000000 |
| hybrid_topk | 0.995144 | 0.929324 | 0.986573 | 0.999961 | 0.000000 |
| cross_encoder_topk | 0.994871 | 0.929078 | 0.985786 | 0.999767 | 0.000000 |
| coverage_greedy_proxy | 0.995029 | 0.929222 | 0.986071 | 0.999806 | 0.000000 |
| frc_select | 0.994919 | 0.929130 | 0.985984 | 0.999767 | 0.000000 |

## Frozen comparisons

- Observed strongest baseline: `bm25_topk`
- FRC minus strongest point: -0.000283
- Simultaneous 95% CI: [-0.001100, 0.000039]
- FRC minus coverage proxy: -0.000110, 95% CI [-0.000524, 0.000298]

## Decision checks

- FAIL — `frc_minus_strongest_point_gain_at_least_0_05`
- FAIL — `frc_minus_strongest_simultaneous_ci_low_above_zero`
- FAIL — `frc_minus_reference_ci_low_above_zero`
- PASS — `no_viewpoint_count_stratum_delta_below_minus_0_05`
- PASS — `no_property_stratum_delta_below_minus_0_05`
- PASS — `frc_budget_shortfall_not_above_strongest_baseline`
- PASS — `all_cases_scored`
- PASS — `gold_leakage_checks_pass`
- PASS — `official_viewpoint_count_alignment_pass`

## Interpretation boundary

This experiment tests evidence selection for naturally occurring same-name ambiguity. It does not confirm the v32 routing model, reproduce SetR, establish flood-domain effectiveness, or authorize production rollout.
