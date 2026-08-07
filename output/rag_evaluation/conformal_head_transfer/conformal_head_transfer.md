# FRC-RAG 跨数据集充分性头迁移审计

- 状态：`DO_NOT_ADOPT`
- 数据集：2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- 重复分组：10
- alpha：0.1
- 目标 train 标签/特征用于迁移头：`False`
- 目标 calibration 标签用于阈值：`True`
- 独立确认：`False`；零样本：`False`
- Gate 2：`NO-GO/SHADOW`

## 与目标集重新拟合头比较

| 数据集 | 目标拟合风险 | 迁移风险 | 目标拟合完整召回 | 迁移完整召回 | 目标拟合 AUC | 迁移 AUC | alpha 达标次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.090575 | 0.104410 | 0.162305 | 0.114892 | 0.748080 | 0.663910 | 5/10 |
| ConditionalQA | 0.080422 | 0.085829 | 0.289074 | 0.086731 | 0.892413 | 0.815340 | 6/10 |
| HotpotQA | 0.096653 | 0.102824 | 0.218803 | 0.152561 | 0.763211 | 0.706035 | 6/10 |
| MultiHop-RAG | 0.087890 | 0.089497 | 0.415231 | 0.064046 | 0.921522 | 0.679523 | 9/10 |

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| target_train_exclusion_and_source_train_only | `true` | `True` | `True` |
| target_threshold_uses_calibration_only | `true` | `True` | `True` |
| no_evaluation_selection | `true` | `True` | `True` |
| maximum_mean_case_family_false_complete_rate_each_dataset | `{"2WikiMultiHopQA": 0.10441, "ConditionalQA": 0.085829, "HotpotQA": 0.102824, "MultiHop-RAG": 0.089497}` | `<=0.1` | `False` |
| maximum_dataset_case_family_risk_increase_vs_target_fitted | `{"2WikiMultiHopQA": 0.013835, "ConditionalQA": 0.005406, "HotpotQA": 0.006172, "MultiHop-RAG": 0.001607}` | `<=0.01` | `False` |
| maximum_dataset_complete_recall_decrease_vs_target_fitted | `{"2WikiMultiHopQA": 0.047414, "ConditionalQA": 0.202343, "HotpotQA": 0.066243, "MultiHop-RAG": 0.351185}` | `<=0.05` | `False` |
| cross_dataset_mean_evaluation_auc_decrease_vs_target_fitted | `0.115104` | `<=0.02` | `False` |
| datasets_with_at_least_7_of_10_alpha_control_repeats | `{"count": 1, "per_dataset": {"2WikiMultiHopQA": 5, "ConditionalQA": 6, "HotpotQA": 6, "MultiHop-RAG": 9}}` | `>=3` | `False` |
| all_finite_sample_nominal_bounds_at_or_below_alpha | `true` | `True` | `True` |

## 结论边界

该审计只检验目标 train 标签隔离后的模型参数迁移；目标 calibration 仍使用标签设阈值，因此不是零样本。四个数据集结果均已在方法设计前可见，故结果属于事后开发，不能改变 Gate 2。

- 下一步：`STOP_TRANSFER_METHOD_SELECTION_WITHOUT_NEW_DATA`
