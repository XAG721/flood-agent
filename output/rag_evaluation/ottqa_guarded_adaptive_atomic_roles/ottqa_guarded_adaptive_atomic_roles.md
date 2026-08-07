# OTT-QA 有保护自适应证据基数实验（v44）

- 状态：`OTTQA_GUARDED_ADAPTIVE_SUPPORT_NOT_ESTABLISHED`
- 可评测样本：240
- 候选池答案证据上限率：0.995833
- 最强非 FRC 基线：`official_anchor_topk`
- v44 - v41 动态：+0.016093，95% CI [+0.001488, +0.029713]
- v44 - 最强非 FRC：-0.015061，95% CI [-0.058018, +0.014180]
- v44 - v43：+0.020596，95% CI [-0.000110, +0.042439]
- 相对动态方法平均少选证据单元：1.125000
- 相对动态方法答案证据召回下降：+0.066667
- Gate 2：`NO-GO/SHADOW`；该有界候选池实验不授权 CANARY/DEFAULT。

## 方法汇总

| 方法 | Evidence F1 | Precision | Answer recall | 平均单元数 |
|---|---:|---:|---:|---:|
| `official_anchor_topk` | 0.205021 | 0.132454 | 0.454167 | 4.231944 |
| `bm25_topk` | 0.122723 | 0.075718 | 0.325000 | 4.816667 |
| `dense_topk` | 0.120235 | 0.073634 | 0.327778 | 4.933333 |
| `hybrid_topk` | 0.146133 | 0.089769 | 0.393056 | 4.891667 |
| `cross_encoder_topk` | 0.186456 | 0.114699 | 0.498611 | 4.877778 |
| `cross_encoder_knapsack` | 0.186738 | 0.114676 | 0.502778 | 4.911111 |
| `static_threshold_frc_v39_replay` | 0.184594 | 0.113426 | 0.495833 | 4.883333 |
| `static_rank_coverage_frc_v41` | 0.182413 | 0.111644 | 0.498611 | 4.884722 |
| `dynamic_rank_coverage_frc_v41` | 0.173866 | 0.106435 | 0.475000 | 4.848611 |
| `adaptive_argmax_cardinality_frc_v43` | 0.169364 | 0.119329 | 0.291667 | 2.676389 |
| `guarded_adaptive_cardinality_frc_v44` | 0.189960 | 0.123773 | 0.408333 | 3.723611 |

## 支持检查

- PASS `minimum_total_cases_met`
- PASS `minimum_cases_per_source_mode_met`
- PASS `candidate_ceiling_complete_rate_at_least_0_60`
- PASS `query_parser_fallback_rate_at_most_0_05`
- PASS `guarded_minus_dynamic_point_at_least_0_005`
- PASS `guarded_minus_dynamic_ci_low_above_0`
- FAIL `guarded_minus_strongest_point_at_least_0_01`
- FAIL `guarded_minus_strongest_ci_low_above_0`
- FAIL `every_budget_and_supported_source_mode_delta_at_least_minus_0_02`
- PASS `mean_selected_unit_reduction_at_least_0_50`
- FAIL `answer_evidence_recall_drop_at_most_0_02`
- PASS `guarded_minus_v43_recall_at_least_0_02`
- PASS `guarded_minus_v43_f1_at_least_minus_0_005`

## 边界

该结果只比较官方 oracle table 及其 linked passages 内的答案承载证据选择，不是 OTT-QA 官方榜单、完整开放域检索或完整多跳证据链评测。
