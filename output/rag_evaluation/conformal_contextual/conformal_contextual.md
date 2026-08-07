# FRC-RAG train-only 上下文充分性头事后方法开发

- 状态：`DO_NOT_ADOPT`
- 数据集：ConditionalQA, HotpotQA, MultiHop-RAG
- alpha：0.1
- 上下文：question_type, required_role_count, question_type_x_required_role_count；每类最低训练 case：10
- 阈值：仅 calibration 的全局阈值
- 独立确认：`False`
- Gate 2：`NO-GO/SHADOW`

## 与冻结全局充分性头的比较

| 数据集 | 基线 case 风险 | 上下文 case 风险 | 基线完整召回 | 上下文完整召回 | 最坏子群风险改善 |
|---|---:|---:|---:|---:|---:|
| ConditionalQA | 0.080422 | 0.091715 | 0.289074 | 0.314925 | -0.011564 |
| HotpotQA | 0.096653 | 0.094070 | 0.218803 | 0.219413 | -0.005628 |
| MultiHop-RAG | 0.087890 | 0.087812 | 0.415231 | 0.424655 | -0.002847 |

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| no_evaluation_selection | `true` | `True` | `True` |
| context_vocabulary_train_only | `true` | `True` | `True` |
| threshold_calibration_only | `true` | `True` | `True` |
| mean_complete_recall_gain | `0.011962` | `>=0.02` | `False` |
| maximum_dataset_complete_recall_decrease | `{"ConditionalQA": -0.025851, "HotpotQA": -0.00061, "MultiHop-RAG": -0.009424}` | `<=0.01` | `True` |
| maximum_dataset_overall_case_risk_increase | `{"ConditionalQA": 0.011292, "HotpotQA": -0.002583, "MultiHop-RAG": -7.8e-05}` | `<=0.01` | `False` |
| mean_worst_eligible_subgroup_risk_reduction | `-0.00668` | `>=0.02` | `False` |
| all_eligible_subgroup_mean_risks_at_or_below_alpha | `false` | `True` | `False` |
| all_eligible_subgroup_repeat_consistency_targets_met | `false` | `True` | `False` |

## 结论边界

该实验只检验 train-only 可观察上下文是否能改善充分性打分。它是在既有结果可见后开展的方法开发，不是独立确认；当前 Gate 2 继续保持 `NO-GO/SHADOW`。
