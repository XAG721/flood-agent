# MuSiQue v69 单调交互链支持门（development）

- 状态：`MUSIQUE_V69_MONOTONE_INTERACTION_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：800
- 最强公平基线：`monotone_additive_spline_control`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.577500 | 0.562500 | 0.592500 |
| `fixed_v66_full_chain` | 0.656250 | 0.450000 | 0.862500 |
| `calibrated_chain_bottleneck` | 0.665000 | 0.545000 | 0.785000 |
| `linear_nine_signal_control` | 0.683750 | 0.612500 | 0.755000 |
| `monotone_additive_spline_control` | 0.690000 | 0.615000 | 0.765000 |
| `monotone_interaction_candidate` | 0.687500 | 0.620000 | 0.755000 |

候选相对最强公平基线差值 -0.002500，95% CI [-0.010000,+0.005000]。
候选相对无交互分段控制差值 -0.002500，95% CI [-0.010000,+0.005000]。

该实验只检验 oracle 计划单调交互支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
