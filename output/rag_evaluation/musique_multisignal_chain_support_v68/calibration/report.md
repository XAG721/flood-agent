# MuSiQue v68 多信号链校准

- 案例：1200
- 状态：`MUSIQUE_V68_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.611667 | 0.546667 | 0.676667 |
| `calibrated_chain_bottleneck` | 0.646667 | 0.523333 | 0.770000 |
| `two_signal_logistic_control` | 0.663333 | 0.586667 | 0.740000 |
| `multisignal_logistic_candidate` | 0.691667 | 0.596667 | 0.786667 |

校准只冻结模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。
