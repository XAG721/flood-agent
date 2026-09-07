# MuSiQue 双资源 FRC 独立盲评（v36）

- 状态：`MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_NOT_ESTABLISHED`
- 案例：2417；候选分块：48656；选择运行：72510
- 最强非 FRC 基线：`cross_encoder_topk`
- 双资源 FRC - v35：+0.004402，95% CI [+0.002244, +0.006613]
- 双资源 FRC - 旧 FRC：-0.002256，95% CI [-0.004128, -0.000403]
- 双资源 FRC - 最强基线：-0.001854，同时 95% CI [-0.003784, -0.000020]
- 最差预算差值：-0.004570；最差 hop 差值：-0.004806

## 等权主指标

| 方法 | support evidence F1 |
|---|---:|
| `bm25_topk` | 0.361082 |
| `dense_topk` | 0.494453 |
| `hybrid_topk` | 0.448289 |
| `cross_encoder_topk` | 0.530399 |
| `cross_encoder_density` | 0.523163 |
| `cross_encoder_knapsack` | 0.529318 |
| `coverage_greedy_proxy` | 0.528341 |
| `frc_select_v34` | 0.530801 |
| `frc_cost_aware_v35` | 0.524143 |
| `frc_dual_resource_v36` | 0.528545 |

## 预注册判定

- FAIL `dual_minus_v35_point_at_least_0_01`
- PASS `dual_minus_v35_ci_low_above_0`
- FAIL `dual_minus_old_frc_point_at_least_0_01`
- FAIL `dual_minus_old_frc_ci_low_above_0`
- FAIL `dual_minus_strongest_point_at_least_0_01`
- FAIL `dual_minus_strongest_simultaneous_ci_low_above_0`
- PASS `every_budget_delta_at_least_minus_0_02`
- PASS `every_hop_delta_at_least_minus_0_02`

## 边界

本实验只检验公开跨域多跳数据上的证据选择，不复现 SetR，不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、CANARY 或 DEFAULT 状态。
