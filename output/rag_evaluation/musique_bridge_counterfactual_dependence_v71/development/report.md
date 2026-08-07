# MuSiQue v71 桥接实体反事实支持门（development）

- 状态：`MUSIQUE_V71_BRIDGE_COUNTERFACTUAL_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：800
- 最强公平基线：`monotone_additive_spline_control`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.578750 | 0.585000 | 0.572500 |
| `fixed_v66_full_chain` | 0.643750 | 0.420000 | 0.867500 |
| `calibrated_chain_bottleneck` | 0.688750 | 0.597500 | 0.780000 |
| `linear_nine_signal_control` | 0.708750 | 0.622500 | 0.795000 |
| `monotone_additive_spline_control` | 0.722500 | 0.620000 | 0.825000 |
| `monotone_interaction_control` | 0.717500 | 0.617500 | 0.817500 |
| `paragraph_competition_control` | 0.705000 | 0.620000 | 0.790000 |
| `bridge_counterfactual_only_control` | 0.615000 | 0.562500 | 0.667500 |
| `bridge_counterfactual_candidate` | 0.706250 | 0.622500 | 0.790000 |

候选相对最强公平基线差值 -0.016250，95% CI [-0.033750,+0.002500]。
候选相对最强既有特征控制差值 -0.016250，95% CI [-0.035000,+0.002500]。
候选相对反事实信号单独控制差值 +0.091250，95% CI [+0.056250,+0.126250]。

该实验只检验 oracle 计划桥接实体反事实支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
