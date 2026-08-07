# FRC-RAG 证据充分性重复分组稳健性审计

- 状态：`RUN_PUBLIC_REAL_MODEL_REPEATED_GROUP_SPLIT_ROBUSTNESS`
- 数据：ConditionalQA
- 预注册重复分组：10 组
- 冻结分数 SHA-256：`feb0d75957c1bb09f07985979bb0816abc761ce6fafc66b40d9d99b02c1374b5`
- 主分析 alpha：0.1
- Gate 2：`NO-GO/SHADOW`

## 风险—效用跨分组分布

| 判定器 | case 家族误放行率均值 [min, max] | 变体误放行率均值 | 拒答率均值 | 完整召回率均值 | 放行精度均值 |
|---|---:|---:|---:|---:|---:|
| 角色覆盖启发式 | 0.847827 [0.810811, 0.909091] | 0.596596 | 0.344626 | 1.000000 | 0.222159 |
| Split-conformal alpha=0.05 | 0.041632 [0.011765, 0.077922] | 0.016077 | 0.967870 | 0.123856 | 0.575431 |
| Split-conformal alpha=0.10 | 0.080422 [0.037037, 0.147727] | 0.032185 | 0.930263 | 0.289074 | 0.602499 |
| Split-conformal alpha=0.20 | 0.209263 [0.102564, 0.311688] | 0.095605 | 0.830487 | 0.608317 | 0.522162 |

## 主分析逐分组结果

| 分组版本 | evaluation case | AUC | case 家族误放行率 | 拒答率 | 完整召回率 | 放行精度 |
|---|---:|---:|---:|---:|---:|---:|
| `frc-conformal-sufficiency-split-v1` | 75 | 0.931552 | 0.040541 | 0.946667 | 0.295455 | 0.812500 |
| `frc-conformal-sufficiency-split-v1-repeat-01` | 81 | 0.912256 | 0.062500 | 0.959877 | 0.181818 | 0.615385 |
| `frc-conformal-sufficiency-split-v1-repeat-02` | 89 | 0.905417 | 0.147727 | 0.884831 | 0.375000 | 0.512195 |
| `frc-conformal-sufficiency-split-v1-repeat-03` | 78 | 0.840207 | 0.038462 | 0.974359 | 0.058824 | 0.375000 |
| `frc-conformal-sufficiency-split-v1-repeat-04` | 77 | 0.858150 | 0.103896 | 0.922078 | 0.277778 | 0.416667 |
| `frc-conformal-sufficiency-split-v1-repeat-05` | 83 | 0.881319 | 0.037037 | 0.930723 | 0.326923 | 0.739130 |
| `frc-conformal-sufficiency-split-v1-repeat-06` | 81 | 0.909239 | 0.088608 | 0.919753 | 0.326531 | 0.615385 |
| `frc-conformal-sufficiency-split-v1-repeat-07` | 80 | 0.902456 | 0.075949 | 0.950000 | 0.285714 | 0.625000 |
| `frc-conformal-sufficiency-split-v1-repeat-08` | 80 | 0.889638 | 0.115385 | 0.893750 | 0.423077 | 0.647059 |
| `frc-conformal-sufficiency-split-v1-repeat-09` | 85 | 0.893893 | 0.094118 | 0.920588 | 0.339623 | 0.666667 |

## 判定

- 主分析在 10/10 组中低于角色覆盖启发式。
- case 家族误放行率在 7/10 组中不高于预注册 alpha。
- 平均拒答率为 0.930263，平均完整召回率为 0.289074。

## 三态人工复核分层

| 状态 | 分数边界 | 平均占比 | 解释 |
|---|---|---:|---|
| `HIGH_CONFIDENCE_CANDIDATE` | alpha=0.10 严格阈值以上 | 0.069737 | 保持主安全边界；Gate 2 阻断自动完成，仍需人工复核 |
| `REVIEW_PRIORITY` | alpha=0.20 与主阈值之间 | 0.099776 | 只提高人工复核优先级，不自动放行 |
| `INSUFFICIENT` | 低于人工复核带 | 0.830487 | 维持证据不足/继续检索 |

人工复核优先带平均额外覆盖完整变体 `0.319242`，HIGH_CONFIDENCE_CANDIDATE 与 REVIEW_PRIORITY 合计覆盖完整变体 `0.608317`；复核带本身精度为 `0.465660`。

重复分组用于检查结论是否依赖单次划分，不用于挑选最有利分组、alpha 或模型。这些分组复用同一公开数据，彼此相关，不能解释为独立重复试验或置信区间。

- Repeated splits reuse the same ConditionalQA cases and are correlated descriptive checks, not confidence intervals.
- ConditionalQA is cross-domain and missing evidence is removed synthetically.
- No repeated-split result replaces double-expert flood-domain completeness labels.
- Evaluation outcomes do not select alpha, regularization, features or a preferred split.

系统继续保持 `NO-GO/SHADOW`；真实防汛双专家标注、独立行为评判和可接受的安全—效用折中仍未完成。
