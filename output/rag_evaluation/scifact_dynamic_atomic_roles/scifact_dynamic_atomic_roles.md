# SciFact 动态原子角色外部确认（v42）

- 状态：`SCIFACT_DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED`
- 评测 claim：187
- 动态 - 静态秩覆盖：+0.002821，95% CI [-0.007003, +0.012092]
- 动态 - bootstrap 最强非 FRC：-0.056707，95% CI [-0.078099, -0.036516]

## 方法主指标

| 方法 | Document Evidence F1 |
|---|---:|
| `bm25_topk` | 0.373394 |
| `dense_topk` | 0.460616 |
| `hybrid_topk` | 0.437297 |
| `cross_encoder_topk` | 0.442744 |
| `cross_encoder_knapsack` | 0.435839 |
| `static_threshold_frc_v39_replay` | 0.441734 |
| `static_rank_coverage_frc_v41` | 0.401088 |
| `dynamic_rank_coverage_frc_v41` | 0.403909 |

## 解释边界

这是冻结 v41 方法在一个外部公开数据集上的单次迁移确认。无论结果如何，都不授权选择器上线、不改变 Gate 2、不复现 SetR，也不证明洪水防汛领域效果；SciFact 此后不得用于方法调参。
