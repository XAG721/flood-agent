# 2Wiki v82 级联路由器development实验

- 状态：`2WIKI_V82_CASCADED_ROUTER_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP`
- 候选证据 F1：`0.544792`
- 最强登记对照：`twowiki_residual_three_route_router_v81` / `0.542649`
- 差值：`+0.002143`，95% CI [`+0.000357`, `+0.004286`]
- 直接比较识别平衡准确率：`0.920000`
- 动作分布：`{'direct_comparison_cross_override': 168, 'v75_default': 478, 'v81_safe_flip_override': 154}`
- 路由分布：`{'alternating_anchor_link_control': 155, 'cross_encoder_topk': 168, 'soft_title_link_support_path_closure_v74': 477}`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_800`
- PASS `exact_question_type_quota`
- PASS `prior_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `candidate_evidence_macro_f1_at_least_0_535`
- PASS `candidate_complete_evidence_recall_at_least_0_58`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_005`
- PASS `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_frozen_v75_f1_at_least_0_005`
- PASS `candidate_minus_frozen_v75_ci_low_above_0`
- PASS `every_question_type_delta_vs_best_control_at_least_minus_0_005`
- PASS `direct_comparison_balanced_accuracy_at_least_0_90`
- PASS `override_action_fraction_at_least_0_05`
- PASS `at_least_two_routes_each_cover_at_least_0_05`
- PASS `largest_route_fraction_at_most_0_9`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

v82 使用同一公开数据集内、与 4200 个既往 2Wiki 样本互斥的新样本。
它不是独立数据集、官方榜单、答案生成、真实 SetR、洪水专家验证或生产证据；
无论本阶段结果如何，Gate 2 都保持 `NO-GO/SHADOW`。
