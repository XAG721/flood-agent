# FEVEROUS 自适应原子角色选择器（v43）

- 状态：`FEVEROUS_BOUNDED_POOL_INCONCLUSIVE`
- 可评测样本：240
- 候选池完整证据组上限率：0.462500
- 最强非 FRC 基线：`cross_encoder_topk`
- 自适应减动态角色：-0.045874，95% CI [-0.079872, -0.016021]
- 自适应减重采样最强非 FRC：-0.036780，95% CI [-0.073102, -0.005646]
- 相对动态方法平均少选证据单元：2.227778
- 相对动态方法完整证据召回下降：+0.087500
- Gate 2：`NO-GO/SHADOW`；本实验不授权 CANARY/DEFAULT。

## 方法汇总

| 方法 | Evidence F1 | Precision | Complete recall | 平均单元数 |
|---|---:|---:|---:|---:|
| `official_baseline_topk` | 0.128629 | 0.137500 | 0.120833 | 4.995833 |
| `bm25_topk` | 0.159776 | 0.176667 | 0.145833 | 5.000000 |
| `dense_topk` | 0.176751 | 0.202292 | 0.156944 | 4.997222 |
| `hybrid_topk` | 0.188746 | 0.217569 | 0.166667 | 4.997222 |
| `cross_encoder_topk` | 0.209285 | 0.224722 | 0.195833 | 4.998611 |
| `cross_encoder_knapsack` | 0.209285 | 0.224722 | 0.195833 | 5.000000 |
| `static_threshold_frc_v39_replay` | 0.208681 | 0.223333 | 0.195833 | 4.998611 |
| `static_rank_coverage_frc_v41` | 0.209044 | 0.224167 | 0.195833 | 4.998611 |
| `dynamic_rank_coverage_frc_v41` | 0.218379 | 0.229444 | 0.208333 | 4.998611 |
| `adaptive_argmax_cardinality_frc_v43` | 0.172505 | 0.301389 | 0.120833 | 2.770833 |

## 支持检查

- PASS `minimum_total_cases_met`
- FAIL `candidate_ceiling_complete_rate_at_least_0_60`
- PASS `query_parser_fallback_rate_at_most_0_05`
- FAIL `adaptive_minus_dynamic_point_at_least_0_01`
- FAIL `adaptive_minus_dynamic_ci_low_above_0`
- FAIL `adaptive_minus_strongest_point_at_least_0_01`
- FAIL `adaptive_minus_strongest_ci_low_above_0`
- FAIL `every_budget_and_supported_challenge_delta_at_least_minus_0_02`
- PASS `mean_selected_unit_reduction_at_least_0_50`
- FAIL `complete_evidence_recall_drop_at_most_0_02`

## 边界

该结果只比较官方基线候选页内的证据单元选择，不是 FEVEROUS 官方榜单结果，也不证明全库检索、真实 SetR、防汛领域效果或生产可用性。
