# MuSiQue v73 上下文桥接实体擦除校准

- 案例：1600
- 状态：`MUSIQUE_V73_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.605625 | 0.590000 | 0.621250 |
| `calibrated_chain_bottleneck` | 0.685625 | 0.575000 | 0.796250 |
| `linear_nine_signal_control` | 0.711250 | 0.627500 | 0.795000 |
| `monotone_additive_spline_control` | 0.712500 | 0.607500 | 0.817500 |
| `monotone_interaction_control` | 0.714375 | 0.621250 | 0.807500 |
| `paragraph_competition_control` | 0.710000 | 0.603750 | 0.816250 |
| `fixed_sentinel_control` | 0.706250 | 0.602500 | 0.810000 |
| `in_domain_rival_control` | 0.707500 | 0.601250 | 0.813750 |
| `context_erasure_only_control` | 0.595625 | 0.430000 | 0.761250 |
| `context_erasure_candidate` | 0.716250 | 0.627500 | 0.805000 |

校准只冻结模型和阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
