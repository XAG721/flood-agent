# HoVer 任务特定核验角色盲评（v39）

- 状态：`HOVER_VERIFICATION_ROLE_SUPPORT_NOT_ESTABLISHED`
- 案例：4000；候选文档：80000；候选分块：81224；选择运行：84000
- 任务特定角色 FRC - 通用角色 FRC：-0.000035，95% CI [-0.000187, +0.000092]
- 任务特定角色 FRC - 最强非 FRC：+0.000267，同时 95% CI [+0.000001, +0.000578]
- 最差预算差值：+0.000051
- 最差预注册分层差值：-0.000129

## 等权主指标

| 方法 | document evidence F1 |
|---|---:|
| `bm25_topk` | 0.373073 |
| `dense_topk` | 0.404287 |
| `hybrid_topk` | 0.414258 |
| `cross_encoder_topk` | 0.433019 |
| `cross_encoder_knapsack` | 0.429862 |
| `frc_generic_roles_v39` | 0.433321 |
| `frc_verification_roles_v39` | 0.433286 |

## 预注册判定

- FAIL `verification_minus_generic_point_at_least_0_01`
- FAIL `verification_minus_generic_ci_low_above_0`
- FAIL `verification_minus_strongest_point_at_least_0_01`
- PASS `verification_minus_strongest_simultaneous_ci_low_above_0`
- PASS `every_budget_delta_at_least_minus_0_02`
- PASS `every_supported_stratum_delta_at_least_minus_0_02`

## 边界

本实验只检验公开跨域多文档数据上的证据选择，不复现 SetR，不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、CANARY 或 DEFAULT 状态。
