# FRC-RAG 证据充分性跨数据集确认系列

- 状态：`AGGREGATE_PUBLIC_REAL_MODEL_CROSS_DATASET_CONFIRMATIONS`
- 参考数据：ConditionalQA
- 确认数据集数：3
- alpha：0.1
- 系列结论：`PARTIAL_CONFIRMATION`
- Gate 2：`NO-GO/SHADOW`

## 独立确认结果

| 数据集 | 结论 | case 风险均值 | 拒答率 | 完整召回率 | 安全改善分组 | 风险不高于 alpha 分组 |
|---|---|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | `FULL_CONFIRMATION` | 0.090575 | 0.927512 | 0.162305 | 10/10 | 7/10 |
| HotpotQA | `PARTIAL_CONFIRMATION` | 0.096653 | 0.884924 | 0.218803 | 10/10 | 6/10 |
| MultiHop-RAG | `FULL_CONFIRMATION` | 0.087890 | 0.884396 | 0.415231 | 10/10 | 8/10 |

完整确认 2/3；部分确认 1/3。

系列规则不允许用一个数据集的通过覆盖另一个数据集的失败。本报告不改变 `NO-GO/SHADOW`，也不能替代真实防汛领域双专家标注。
