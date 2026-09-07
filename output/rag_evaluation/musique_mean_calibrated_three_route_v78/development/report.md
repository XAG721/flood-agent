# MuSiQue 均值校准三路路由实验（v78 development）

- 状态：`MUSIQUE_V78_MEAN_CALIBRATED_THREE_ROUTE_ADVANTAGE_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 候选证据 F1：`0.546382`
- 最强登记对照：`alternating_anchor_link_control` / `0.548082`
- 差值：`-0.001700`，95% CI [`-0.006865`, `+0.003572`]
- 路由分布：`{'alternating_anchor_link_control': 316, 'soft_title_link_support_path_closure_v74': 284}`
- 非劣包络：`True`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_600`
- PASS `exact_hop_quota`
- PASS `prior_calibration_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- FAIL `candidate_evidence_macro_f1_at_least_0_55`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_005`
- FAIL `candidate_minus_strongest_control_ci_low_above_0`
- FAIL `candidate_minus_frozen_v76_f1_at_least_0_005`
- FAIL `candidate_minus_frozen_v76_ci_low_above_0`
- FAIL `every_hop_delta_vs_best_control_at_least_minus_0_005`
- PASS `at_least_two_routes_each_cover_at_least_0_05`
- PASS `largest_route_fraction_at_most_0_9`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

v77 是显式目标域均值校准集，v78 只验证新样本泛化；这不是零目标调参、独立训练数据确认、官方榜单、真实 SetR、洪水专家或生产证据。Gate 2 保持 `NO-GO/SHADOW`。
