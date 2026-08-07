# MuSiQue 锚点默认三路路由终末留出实验（v80）

- 状态：`MUSIQUE_V80_ANCHOR_DEFAULT_TERMINAL_HOLDOUT_ADVANTAGE_NOT_ESTABLISHED`
- 候选证据 F1：`0.527817`
- 最强登记对照：`musique_target_trained_three_route_router_v79` / `0.526257`
- 差值：`+0.001561`，95% CI [`-0.002533`, `+0.005774`]
- 路由分布：`{'alternating_anchor_link_control': 273, 'cross_encoder_topk': 186, 'soft_title_link_support_path_closure_v74': 141}`
- 非劣包络：`True`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_600`
- PASS `exact_hop_quota`
- PASS `prior_training_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- FAIL `candidate_evidence_macro_f1_at_least_0_55`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_005`
- FAIL `candidate_minus_strongest_control_ci_low_above_0`
- FAIL `candidate_minus_frozen_v79_f1_at_least_0_005`
- FAIL `candidate_minus_frozen_v79_ci_low_above_0`
- FAIL `every_hop_delta_vs_best_control_at_least_minus_0_005`
- PASS `at_least_two_routes_each_cover_at_least_0_05`
- PASS `largest_route_fraction_at_most_0_9`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

v78+v79 是显式目标域训练与模型选择集；v80 是同一数据集剩余容量上的单次终末留出。它不是零目标调参、独立训练数据、官方榜单、真实 SetR、洪水专家或生产证据。Gate 2 保持 `NO-GO/SHADOW`。
