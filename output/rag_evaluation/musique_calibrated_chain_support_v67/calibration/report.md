# MuSiQue v67 链级阈值校准

- 案例：1000
- 状态：`MUSIQUE_V67_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`

| 方法 | 阈值 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |
| --- | ---: | ---: | ---: | ---: |
| `direct_composed_question` | 1.216796875 | 0.593000 | 0.594000 | 0.592000 |
| `oracle_plan_chain_bottleneck` | -1.070312500 | 0.662000 | 0.590000 | 0.734000 |

校准结果只冻结阈值并开放互斥开发集，不授权方法采用、检索评分或 Gate 2 放行。
