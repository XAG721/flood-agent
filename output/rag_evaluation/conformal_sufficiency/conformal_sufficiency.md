# FRC-RAG 证据充分性 Split-Conformal 校准实验

- 状态：`RUN_PUBLIC_REAL_MODEL_SPLIT_CONFORMAL_ABSTENTION`
- 数据：ConditionalQA，261 个分组样本，1044 个缺失变体
- 冻结评分器：`BAAI/bge-large-en-v1.5` + `BAAI/bge-reranker-large`，校准 `per_case_minmax`
- 参考运行报告 SHA-256：`ae7cb9392be2bbfc0e08b0cdb7a9223d4dca247518db0924940e36d9b214bc65`
- 冻结分数来源 SHA-256：`feb0d75957c1bb09f07985979bb0816abc761ce6fafc66b40d9d99b02c1374b5`
- 主分析：alpha=0.1，严格放行阈值 `0.797548`
- Gate 2：`NO-GO/SHADOW`

## 分组切分

| 切分 | 样本 | 变体 | 完整 | 不完整 | AUC |
|---|---:|---:|---:|---:|---:|
| train | 110 | 440 | 67 | 373 | 0.904085 |
| calibration | 76 | 304 | 40 | 264 | 0.914205 |
| evaluation | 75 | 300 | 44 | 256 | 0.931552 |

## 主评估结果

| 判定器 | 放行率 | 不完整误放行率 | 缺源证据误放行率 | 完整召回率 | 放行精度 |
|---|---:|---:|---:|---:|---:|
| 角色覆盖阈值启发式 | 0.630000 | 0.566406 | 0.455882 | 1.000000 | 0.232804 |
| Split-conformal (alpha=0.1) | 0.053333 | 0.011719 | 0.014706 | 0.295455 | 0.812500 |

不完整误放行率的 split-conformal Wilson 95% 区间为 `[0.003993, 0.033882]`。
以 case 家族计，任一不完整变体被误放行的比例为 `0.040541` （3/74）。
配对 McNemar 精确检验：基线错误而校准正确 142 个，基线正确而校准错误 0 个，`p=3.587e-43`。

## Alpha 风险—效用敏感性

| Alpha | 阈值 | case 家族误放行率 | 变体误放行率 | 放行率 | 完整召回率 | 放行精度 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.842162 | 0.027027 | 0.007812 | 0.020000 | 0.090909 | 0.666667 |
| 0.10 | 0.797548 | 0.040541 | 0.011719 | 0.053333 | 0.295455 | 0.812500 |
| 0.20 | 0.733614 | 0.148649 | 0.066406 | 0.153333 | 0.659091 | 0.630435 |

alpha=0.05/0.10/0.20 均在 evaluation 结果揭示前预先固定；该表用于显示安全与效用权衡，不据 evaluation 重新选择主阈值。

## 不同缺失比例

| 目标缺失比例 | 基线误放行率 | 校准误放行率 | 基线放行率 | 校准放行率 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.000000 | 0.000000 | 1.000000 | 0.106667 |
| 0.25 | 0.692308 | 0.015385 | 0.733333 | 0.066667 |
| 0.50 | 0.464789 | 0.014085 | 0.493333 | 0.026667 |
| 0.75 | 0.283784 | 0.013514 | 0.293333 | 0.013333 |

## 解释边界

该实验只评估“是否应当拒答/转人工”的安全判定头，不证明检索质量提高。同一 case 的四个缺失变体始终位于同一分组，特征不读取 gold 字段；阈值只由 calibration 中每个 case 的最不利不完整变体确定，evaluation 不参与训练或选阈值。

- ConditionalQA is cross-domain rather than a real district flood corpus.
- Missingness is deterministic synthetic removal from a frozen candidate pool.
- Completeness labels are derived from benchmark evidence ids, not double-expert flood-domain judgments.
- The evaluation is a single grouped split and does not authorize CANARY or DEFAULT.

因此系统继续保持 `NO-GO/SHADOW`；本结果不能替代真实防汛领域的双专家完整性标注、独立 expected-behavior 评判或生产验收。
