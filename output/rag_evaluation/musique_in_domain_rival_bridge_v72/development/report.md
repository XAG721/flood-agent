# MuSiQue v72 域内竞争桥接实体支持门（development）

- 状态：`MUSIQUE_V72_IN_DOMAIN_RIVAL_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：800
- 最强公平基线：`linear_nine_signal_control`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.562500 | 0.602500 | 0.522500 |
| `fixed_v66_full_chain` | 0.638750 | 0.415000 | 0.862500 |
| `calibrated_chain_bottleneck` | 0.671250 | 0.617500 | 0.725000 |
| `linear_nine_signal_control` | 0.707500 | 0.612500 | 0.802500 |
| `monotone_additive_spline_control` | 0.698750 | 0.607500 | 0.790000 |
| `monotone_interaction_control` | 0.706250 | 0.615000 | 0.797500 |
| `paragraph_competition_control` | 0.701250 | 0.602500 | 0.800000 |
| `fixed_sentinel_control` | 0.703750 | 0.600000 | 0.807500 |
| `in_domain_rival_only_control` | 0.593750 | 0.432500 | 0.755000 |
| `in_domain_rival_candidate` | 0.706250 | 0.617500 | 0.795000 |

候选相对最强公平基线差值 -0.001250，95% CI [-0.015000,+0.012500]。
候选相对最强既有特征控制差值 -0.001250，95% CI [-0.015000,+0.012500]。
候选相对域内竞争信号单独控制差值 +0.112500，95% CI [+0.080000,+0.146250]。

该实验只检验 oracle 计划域内竞争桥接实体支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
