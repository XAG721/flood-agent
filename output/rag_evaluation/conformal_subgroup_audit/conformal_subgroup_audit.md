# FRC-RAG 可观察子群充分性压力审计

- 状态：`RUN_PUBLIC_REAL_MODEL_OBSERVABLE_SUBGROUP_STRESS_AUDIT`
- 数据集：2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- 主 alpha：0.1
- 每数据集重复分组：10
- 子群最低支持：每次 30 个不完整 case，至少 7 次可评
- 总体结论：`SUBGROUP_INSTABILITY_DETECTED`
- Gate 2：`NO-GO/SHADOW`

## 数据集最坏可评子群

| 数据集 | 可评子群数 | 最坏维度 | 最坏取值 | 平均 case 风险 | 最大 case 风险 | alpha 内重复 |
|---|---:|---|---|---:|---:|---:|
| 2WikiMultiHopQA | 7 | candidate_count_bucket | lt_20 | 0.189738 | 0.378378 | 3/10 |
| ConditionalQA | 3 | candidate_count_bucket | lt_20 | 0.082348 | 0.151163 | 7/10 |
| HotpotQA | 5 | question_type | comparison | 0.179924 | 0.269231 | 1/10 |
| MultiHop-RAG | 7 | required_role_count | 2 | 0.134452 | 0.185065 | 2/10 |

## 理论边界

- 全局有限样本名义上界全部不高于 alpha：`True`
- 所有可评子群平均风险不高于 alpha：`False`
- 所有可评子群达到重复一致性目标：`False`
- 不声明分布无关的条件子群保证。

该审计只暴露全局校准在可观察子群上的异质性，不使用 gold 字段分组、不改变阈值，也不改变 `NO-GO/SHADOW`。
