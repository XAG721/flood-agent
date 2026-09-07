# MuSiQue v69 单调交互链校准

- 案例：1600
- 状态：`MUSIQUE_V69_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.571250 | 0.536250 | 0.606250 |
| `calibrated_chain_bottleneck` | 0.656250 | 0.555000 | 0.757500 |
| `linear_nine_signal_control` | 0.708125 | 0.620000 | 0.796250 |
| `monotone_additive_spline_control` | 0.706250 | 0.608750 | 0.803750 |
| `monotone_interaction_candidate` | 0.704375 | 0.612500 | 0.796250 |

校准只冻结基函数、交互、模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
