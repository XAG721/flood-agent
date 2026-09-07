# HotpotQA v84 问题类型感知证据基数实验（development）

- 状态：`HOTPOT_V84_CARDINALITY_DEVELOPMENT_CONSTRAINED_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION`
- 候选证据 F1：`0.644873`
- 候选完整证据召回：`0.643333`
- 最强安全合格对照：`frozen_v76_prefix4_cardinality_control` / `0.615834`
- 约束内 F1 差值：`+0.029039`，95% CI [`+0.022617`, `+0.035716`]
- 原始 F1 最强对照：`frozen_v76_prefix3_cardinality_control` / `0.679231`（安全合格：`False`）
- comparison 分类平衡准确率：`0.940000`
- 动作分布：`{'predicted_bridge_retain_four': 373, 'predicted_comparison_retain_three': 227}`
- 保留条数：`{'3': 227, '4': 373}`
- Gate 2：`NO-GO/SHADOW`

## 严格门槛

- PASS `exact_cases_equals_600`
- PASS `exact_question_type_quota`
- PASS `prior_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `candidate_evidence_macro_f1_at_least_0_62`
- PASS `candidate_complete_evidence_recall_at_least_0_62`
- PASS `candidate_minus_strongest_safety_eligible_f1_at_least_0_005`
- PASS `candidate_minus_strongest_safety_eligible_ci_low_above_0`
- PASS `candidate_minus_fixed_prefix4_f1_at_least_0_005`
- PASS `candidate_minus_fixed_prefix4_ci_low_above_0`
- PASS `every_question_type_delta_vs_best_safety_eligible_at_least_minus_0_005`
- PASS `comparison_classifier_balanced_accuracy_at_least_0_88`
- PASS `comparison_action_fraction_at_least_0_2`
- PASS `comparison_action_fraction_at_most_0_5`
- PASS `both_actions_each_cover_at_least_0_1`
- PASS `mean_selected_tokens_not_above_strongest_safety_eligible`
- PASS `complete_evidence_recall_delta_vs_strongest_safety_eligible_at_least_minus_0_08`
- PASS `evidence_macro_recall_delta_vs_strongest_safety_eligible_at_least_minus_0_08`

该实验检验的是完整证据召回不低于 0.62 条件下的 F1 优势；原始 F1 最强但不满足安全下限的对照仍完整披露。它不是 HotpotQA 官方排行榜、答案生成、真实 SetR、洪水专家评测或生产放行证据，Gate 2 保持 `NO-GO/SHADOW`。
