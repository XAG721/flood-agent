# MuSiQue v72 域内竞争桥接实体校准

- 案例：1600
- 状态：`MUSIQUE_V72_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.578125 | 0.547500 | 0.608750 |
| `calibrated_chain_bottleneck` | 0.658750 | 0.595000 | 0.722500 |
| `linear_nine_signal_control` | 0.687500 | 0.591250 | 0.783750 |
| `monotone_additive_spline_control` | 0.688125 | 0.585000 | 0.791250 |
| `monotone_interaction_control` | 0.691250 | 0.595000 | 0.787500 |
| `paragraph_competition_control` | 0.688125 | 0.587500 | 0.788750 |
| `fixed_sentinel_control` | 0.683750 | 0.587500 | 0.780000 |
| `in_domain_rival_only_control` | 0.610000 | 0.451250 | 0.768750 |
| `in_domain_rival_candidate` | 0.687500 | 0.598750 | 0.776250 |

校准只冻结模型和阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
