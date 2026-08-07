# HotpotQA 图路由实验（v76 confirmation）

- 状态：`HOTPOT_V76_GRAPH_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED`
- 候选证据 F1：`0.570916`
- 最强登记对照：`cross_encoder_topk` / `0.564456`
- 差值：`+0.006460`，95% CI [`+0.002201`, `+0.010864`]
- 软闭包覆盖率：`0.236250`
- Gate 2：`NO-GO/SHADOW`

## 门槛

- PASS `exact_cases_equals_800`
- PASS `exact_question_type_balance`
- PASS `history_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `candidate_evidence_macro_f1_at_least_0_55`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- PASS `candidate_minus_strongest_control_f1_at_least_0_005`
- PASS `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_cross_encoder_f1_at_least_0_005`
- PASS `candidate_minus_cross_encoder_ci_low_above_0`
- PASS `every_question_type_delta_vs_best_control_at_least_minus_0_005`
- PASS `soft_override_rate_between_0_15_and_0_55`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

该实验不是 HotpotQA 官方排行榜、答案生成、真实 SetR、洪水专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。
