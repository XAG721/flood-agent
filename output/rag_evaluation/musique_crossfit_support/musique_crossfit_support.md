# MuSiQue case 隔离支持校准诊断（v38）

- 状态：`CROSSFIT_SUPPORT_SIGNAL_NOT_ESTABLISHED`
- 下一步：`STOP_THIS_CALIBRATION_FAMILY_ON_MUSIQUE`
- 案例：2417；候选分块：48656；五折选择运行：36255
- 完整模型 - cross-encoder top-k：+0.007383，95% CI [+0.003425, +0.011259]
- 完整模型 - 基础模型：+0.000548，95% CI [-0.000744, +0.001834]

## 方法均值

| 方法 | Evidence F1 | Precision | Recall |
|---|---:|---:|---:|
| `cross_encoder_topk` | 0.530399 | 0.422220 | 0.761389 |
| `frc_select_v34` | 0.530801 | 0.422827 | 0.761746 |
| `frc_dual_resource_v36` | 0.528545 | 0.418963 | 0.762596 |
| `crossfit_base_support_topk_v38` | 0.537234 | 0.426468 | 0.773261 |
| `crossfit_full_support_topk_v38` | 0.537782 | 0.427033 | 0.773882 |

## 候选级 held-out 诊断

- `base`：AUC 0.880769，AP 0.645178，Brier 0.124061
- `full`：AUC 0.881182，AP 0.651500，Brier 0.123131

## 边界

这是结果揭示后的 case 隔离发现研究。held-out 标签不参与对应折模型拟合，但方法族本身是在 MuSiQue 结果已知后登记，因此不能作为独立确认、采用证据或 Gate 2 放行依据。
