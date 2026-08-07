# MuSiQue v67 校准链支持门（development）

- 状态：`MUSIQUE_V67_CALIBRATED_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 案例：600
- 最强公平基线：`fixed_v66_full_chain`

| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |
| --- | ---: | ---: | ---: |
| 校准直接门 | 0.568333 | 0.570000 | 0.566667 |
| 固定 v66 完整链 | 0.615000 | 0.393333 | 0.836667 |
| 校准链瓶颈门 | 0.626667 | 0.553333 | 0.700000 |

候选相对最强公平基线的配对正确性差值为 +0.011667，95% CI [-0.018333,+0.041667]。

该实验只检验 oracle 计划下的链级校准支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。
