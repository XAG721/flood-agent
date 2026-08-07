# MuSiQue v73 上下文桥接实体擦除支持门（development）

- 状态：`MUSIQUE_V73_CONTEXT_BRIDGE_ERASURE_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：800
- 平均依赖覆盖率：0.998750
- 最强公平基线：`in_domain_rival_control`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| `calibrated_direct_composed_question` | 0.578750 | 0.532500 | 0.625000 |
| `fixed_v66_full_chain` | 0.638750 | 0.425000 | 0.852500 |
| `calibrated_chain_bottleneck` | 0.638750 | 0.535000 | 0.742500 |
| `linear_nine_signal_control` | 0.692500 | 0.612500 | 0.772500 |
| `monotone_additive_spline_control` | 0.700000 | 0.625000 | 0.775000 |
| `monotone_interaction_control` | 0.697500 | 0.630000 | 0.765000 |
| `paragraph_competition_control` | 0.696250 | 0.612500 | 0.780000 |
| `fixed_sentinel_control` | 0.695000 | 0.590000 | 0.800000 |
| `in_domain_rival_control` | 0.702500 | 0.610000 | 0.795000 |
| `context_erasure_only_control` | 0.563750 | 0.375000 | 0.752500 |
| `context_erasure_candidate` | 0.702500 | 0.587500 | 0.817500 |

候选相对最强公平基线差值 +0.000000，95% CI [-0.013750,+0.013750]。
候选相对最强既有特征控制差值 +0.000000，95% CI [-0.013750,+0.013750]。
候选相对擦除信号单独控制差值 +0.138750，95% CI [+0.103750,+0.175000]。

该实验只检验 oracle 计划上下文桥接实体擦除支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
