# FRC-RAG 证据充分性 MultiHop-RAG 跨数据集确认

- 状态：`RUN_PUBLIC_REAL_MODEL_MULTIHOPRAG_CONFIRMATION`
- 参考数据：ConditionalQA
- 确认数据：MultiHop-RAG
- 预注册重复分组：10 组
- alpha：0.1
- Gate 2：`NO-GO/SHADOW`

## 跨数据集结果

| 指标 | ConditionalQA 参考 | MultiHop-RAG 确认 | 差值 |
|---|---:|---:|---:|
| case 家族误放行率均值 | 0.080422 | 0.087890 | +0.007468 |
| 拒答率均值 | 0.930263 | 0.884396 | -0.045867 |
| 完整召回率均值 | 0.289074 | 0.415231 | +0.126157 |

## 预注册确认检查

- `safety_reduction_repeats_meet_pre_registered_minimum`：`PASS`
- `alpha_control_repeats_meet_pre_registered_minimum`：`PASS`
- `mean_case_family_false_complete_rate_at_or_below_alpha`：`PASS`
- `confirmation_dataset_differs_from_reference`：`PASS`
- `split_versions_frozen_from_reference`：`PASS`

确认状态：`FULL_CONFIRMATION`；完整跨数据集确认：`True`。

MultiHop-RAG 上相对角色覆盖启发式的安全改善出现在 10/10 组；实测 case 风险不高于 alpha 的分组为 8/10。

该确认只检验固定安全协议是否跨公开数据复现，不进行候选选型，也不证明 FRC 检索优于公平基线。系统继续保持 `NO-GO/SHADOW`。

- MultiHop-RAG is a public QA corpus, not a district flood corpus.
- The model is refit and calibrated within the confirmation dataset grouped splits; this is protocol transfer, not zero-shot weight transfer.
- Missing evidence remains deterministic synthetic removal from frozen candidates.
- Cross-dataset confirmation does not replace flood-domain double-expert labels or independent behavior review.
