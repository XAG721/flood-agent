# 2Wiki v83 精度裁剪器confirmation实验

- 状态：`2WIKI_V83_PRECISION_TRIM_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED`
- 候选证据 F1：`0.549420`
- 最强登记对照：`twowiki_cascaded_style_residual_router_v82` / `0.542986`
- 差值：`+0.006434`，95% CI [`+0.002465`, `+0.010253`]
- bridge 分类平衡准确率：`0.989167`
- 动作分布：`{'bridge_aware_trim_to_four': 209, 'keep_frozen_v82_selection': 591}`
- 保留条数：`{'4': 209, '5': 591}`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_800`
- PASS `exact_question_type_quota`
- PASS `prior_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- FAIL `candidate_evidence_macro_f1_at_least_0_55`
- PASS `candidate_complete_evidence_recall_at_least_0_58`
- PASS `candidate_minus_strongest_control_f1_at_least_0_005`
- PASS `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_frozen_v82_f1_at_least_0_005`
- PASS `candidate_minus_frozen_v82_ci_low_above_0`
- PASS `every_question_type_delta_vs_best_control_at_least_minus_0_005`
- PASS `bridge_classifier_balanced_accuracy_at_least_0_95`
- PASS `trim_action_fraction_at_least_0_15`
- PASS `trim_action_fraction_at_most_0_35`
- PASS `both_retained_counts_each_cover_at_least_0_05`
- PASS `mean_selected_tokens_not_above_strongest_control`
- PASS `complete_evidence_recall_delta_vs_strongest_at_least_minus_0_03`
- PASS `evidence_macro_recall_delta_vs_strongest_at_least_minus_0_03`

v83 使用同一公开数据集内、与 5000 个既往 2Wiki 样本互斥的新样本。
它不是独立数据集、官方榜单、答案生成、真实 SetR、洪水专家验证或生产证据；
无论本阶段结果如何，Gate 2 都保持 `NO-GO/SHADOW`。
