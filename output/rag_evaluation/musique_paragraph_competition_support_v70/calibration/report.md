# MuSiQue v70 段落竞争稳定性校准

- 案例：1600
- 状态：`MUSIQUE_V70_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.586875 | 0.585000 | 0.588750 |
| `calibrated_chain_bottleneck` | 0.652500 | 0.576250 | 0.728750 |
| `linear_nine_signal_control` | 0.681250 | 0.572500 | 0.790000 |
| `monotone_additive_spline_control` | 0.689375 | 0.591250 | 0.787500 |
| `monotone_interaction_control` | 0.688750 | 0.586250 | 0.791250 |
| `paragraph_competition_only_control` | 0.598125 | 0.532500 | 0.663750 |
| `paragraph_competition_candidate` | 0.683750 | 0.586250 | 0.781250 |

校准只冻结模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
