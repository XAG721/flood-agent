# 2Wiki 问题路由支持路径实验（v75 confirmation）

- 状态：`2WIKI_V75_QUESTION_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED`
- 路由平衡准确率：`0.998750`
- 候选证据 F1：`0.531013`
- 最强静态对照：`alternating_anchor_link_control` / `0.517441`
- 差值：`+0.013571`，95% CI [`+0.008214`, `+0.018929`]
- Gate 2：`NO-GO/SHADOW`

## 门槛

- PASS `exact_cases_equals_800`
- PASS `exact_question_type_balance`
- PASS `prior_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `router_balanced_accuracy_at_least_0_90`
- PASS `comparison_route_rate_between_0_35_and_0_65`
- PASS `candidate_evidence_macro_f1_at_least_0_53`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- PASS `candidate_minus_strongest_static_f1_at_least_0_01`
- PASS `candidate_minus_strongest_static_ci_low_above_0`
- PASS `candidate_minus_frc_select_f1_at_least_0_05`
- PASS `candidate_minus_frc_select_ci_low_above_0`
- PASS `every_question_type_delta_vs_best_static_at_least_minus_0_01`
- PASS `mean_selected_tokens_within_1_05_of_strongest_static`

该实验不是官方排行榜、答案生成、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。
