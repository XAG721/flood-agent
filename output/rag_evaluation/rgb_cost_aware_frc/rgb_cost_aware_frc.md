# RGB 成本感知 FRC 盲评报告（v35）

- 状态：`RGB_COST_AWARE_FRC_SAFETY_REGRESSION`
- 有效案例：498；候选分块：14580；选择运行：13446
- 最强非 FRC 基线：`coverage_greedy_proxy`
- 成本感知 FRC - 旧 FRC：-0.043778，95% CI [-0.061536, -0.026205]
- 成本感知 FRC - 最强基线：-0.048719，同时 bootstrap 95% CI [-0.066307, -0.032655]
- 最差数据集/预算差值：-0.127688
- 错误正例选择率相对旧 FRC 变化：-0.022449

## 等权主指标

| 方法 | evidence F1 |
|---|---:|
| `bm25_topk` | 0.397261 |
| `dense_topk` | 0.498857 |
| `hybrid_topk` | 0.480014 |
| `cross_encoder_topk` | 0.545701 |
| `cross_encoder_density` | 0.501741 |
| `cross_encoder_knapsack` | 0.549726 |
| `coverage_greedy_proxy` | 0.556141 |
| `frc_select_v34` | 0.551199 |
| `frc_cost_aware_v35` | 0.507421 |

## 预注册判定

- FAIL `old_frc_point_at_least_0_01`
- FAIL `old_frc_ci_low_above_0`
- FAIL `strongest_point_at_least_0_01`
- FAIL `strongest_simultaneous_ci_low_above_0`
- FAIL `every_dataset_budget_delta_at_least_minus_0_02`
- PASS `positive_wrong_rate_increase_at_most_0_02`

## 边界

本实验只检验公开跨域数据上的证据选择可行性，不复现 SetR，不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、CANARY 或 DEFAULT 状态。
