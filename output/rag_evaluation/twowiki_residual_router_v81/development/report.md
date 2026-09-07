# 2Wiki v81 残差三路由器development实验

- 状态：`2WIKI_V81_RESIDUAL_ROUTER_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP`
- 候选证据 F1：`0.547195`
- 最强登记对照：`learned_question_routed_support_path_v75` / `0.542751`
- 差值：`+0.004444`，95% CI [`+0.001706`, `+0.007381`]
- 动作分布：`{'cross_override': 80, 'soft_anchor_flip_override': 202, 'v75_default': 518}`
- 路由分布：`{'alternating_anchor_link_control': 201, 'cross_encoder_topk': 80, 'soft_title_link_support_path_closure_v74': 519}`
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
- FAIL `candidate_minus_frozen_v75_f1_at_least_0_005`
- PASS `candidate_minus_frozen_v75_ci_low_above_0`
- FAIL `every_question_type_delta_vs_best_control_at_least_minus_0_005`
- PASS `override_action_fraction_at_least_0_05`
- PASS `at_least_two_routes_each_cover_at_least_0_05`
- PASS `largest_route_fraction_at_most_0_9`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

v81 使用同一公开数据集内、与 3400 个既往 2Wiki 样本互斥的新样本。
它不是独立数据集、官方榜单、答案生成、真实 SetR、洪水专家验证或生产证据；
无论本阶段结果如何，Gate 2 都保持 `NO-GO/SHADOW`。
