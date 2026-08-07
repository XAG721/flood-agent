# MuSiQue v68 多信号链支持门（development）

- 状态：`MUSIQUE_V68_MULTISIGNAL_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：600
- 最强公平基线：`calibrated_chain_bottleneck`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.580000 | 0.493333 | 0.666667 |
| `fixed_v66_full_chain` | 0.625000 | 0.410000 | 0.840000 |
| `calibrated_chain_bottleneck` | 0.638333 | 0.523333 | 0.753333 |
| `two_signal_logistic_control` | 0.633333 | 0.546667 | 0.720000 |
| `multisignal_logistic_candidate` | 0.675000 | 0.573333 | 0.776667 |

候选相对最强公平基线差值 +0.036667，95% CI [+0.010000,+0.063333]。

该实验只检验 oracle 计划多信号支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
