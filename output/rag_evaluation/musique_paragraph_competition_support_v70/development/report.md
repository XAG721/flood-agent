# MuSiQue v70 段落竞争支持门（development）

- 状态：`MUSIQUE_V70_PARAGRAPH_COMPETITION_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：800
- 最强公平基线：`monotone_interaction_control`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.595000 | 0.572500 | 0.617500 |
| `fixed_v66_full_chain` | 0.631250 | 0.417500 | 0.845000 |
| `calibrated_chain_bottleneck` | 0.651250 | 0.597500 | 0.705000 |
| `linear_nine_signal_control` | 0.666250 | 0.560000 | 0.772500 |
| `monotone_additive_spline_control` | 0.686250 | 0.600000 | 0.772500 |
| `monotone_interaction_control` | 0.691250 | 0.600000 | 0.782500 |
| `paragraph_competition_only_control` | 0.568750 | 0.462500 | 0.675000 |
| `paragraph_competition_candidate` | 0.665000 | 0.572500 | 0.757500 |

候选相对最强公平基线差值 -0.026250，95% CI [-0.048750,-0.003750]。
候选相对最强既有特征控制差值 -0.026250，95% CI [-0.048750,-0.004969]。
候选相对竞争信号单独控制差值 +0.096250，95% CI [+0.060000,+0.132500]。

该实验只检验 oracle 计划段落竞争支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
