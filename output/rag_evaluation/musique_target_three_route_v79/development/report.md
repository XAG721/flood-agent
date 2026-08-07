# MuSiQue 目标域三路路由实验（v79 development）

- 状态：`MUSIQUE_V79_TARGET_THREE_ROUTE_ADVANTAGE_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 候选证据 F1：`0.553723`
- 最强登记对照：`alternating_anchor_link_control` / `0.550853`
- 差值：`+0.002871`，95% CI [`-0.002187`, `+0.007844`]
- 路由分布：`{'alternating_anchor_link_control': 164, 'cross_encoder_topk': 193, 'soft_title_link_support_path_closure_v74': 113}`
- 非劣包络：`True`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_470`
- PASS `exact_hop_quota`
- PASS `prior_training_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `candidate_evidence_macro_f1_at_least_0_55`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_005`
- FAIL `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_frozen_v78_f1_at_least_0_005`
- FAIL `candidate_minus_frozen_v78_ci_low_above_0`
- PASS `every_hop_delta_vs_best_control_at_least_minus_0_005`
- PASS `at_least_two_routes_each_cover_at_least_0_05`
- PASS `largest_route_fraction_at_most_0_9`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

v78 是显式目标域训练与模型选择集；v79 只检验同一数据集的新样本泛化。这不是零目标调参、独立数据集确认、官方榜单、真实 SetR、洪水专家或生产证据。Gate 2 保持 `NO-GO/SHADOW`。
