# MuSiQue v71 桥接实体反事实依赖校准

- 案例：1600
- 状态：`MUSIQUE_V71_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.613125 | 0.591250 | 0.635000 |
| `calibrated_chain_bottleneck` | 0.655625 | 0.560000 | 0.751250 |
| `linear_nine_signal_control` | 0.690000 | 0.591250 | 0.788750 |
| `monotone_additive_spline_control` | 0.698125 | 0.605000 | 0.791250 |
| `monotone_interaction_control` | 0.696250 | 0.598750 | 0.793750 |
| `paragraph_competition_control` | 0.695000 | 0.593750 | 0.796250 |
| `bridge_counterfactual_only_control` | 0.587500 | 0.516250 | 0.658750 |
| `bridge_counterfactual_candidate` | 0.689375 | 0.587500 | 0.791250 |

校准只冻结模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
