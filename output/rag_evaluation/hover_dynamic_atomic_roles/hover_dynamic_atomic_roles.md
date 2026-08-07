# HoVer 动态原子核验角色确认（v41）

- 状态：`DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED`
- 确认样例：2000
- 动态 - 静态秩覆盖：+0.001593，95% CI [+0.000453, +0.002762]
- 动态 - bootstrap 最强非 FRC：+0.001463，95% CI [+0.000308, +0.002688]

## 方法主指标

| 方法 | Document Evidence F1 |
|---|---:|
| `bm25_topk` | 0.373976 |
| `dense_topk` | 0.406294 |
| `hybrid_topk` | 0.412885 |
| `cross_encoder_topk` | 0.427388 |
| `cross_encoder_knapsack` | 0.423923 |
| `static_threshold_frc_v39_replay` | 0.427499 |
| `static_rank_coverage_frc_v41` | 0.427258 |
| `dynamic_rank_coverage_frc_v41` | 0.428851 |

## 边界

本轮只是在 HoVer 同数据集家族的未分析 train 分区上确认。无论结果是否通过，都不复现 SetR、不证明真实防汛领域效果、不改变 Gate 2，也不授权 CANARY、DEFAULT 或选择器替换。
