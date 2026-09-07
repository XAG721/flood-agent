# FRC-RAG 多阶段检索分数稳定性充分性头

- 状态：`DO_NOT_ADOPT`
- 数据集：2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- 特征：37 + 27 = 64
- 头拟合：每个目标数据集 train-only
- 阈值：每个目标数据集 calibration-only 全局阈值
- 独立确认：`False`
- Gate 2：`NO-GO/SHADOW`

## 与冻结 37 维充分性头比较

| 数据集 | 基线风险 | 稳定性风险 | 基线召回 | 稳定性召回 | AUC 增益 | 最坏子群风险改善 |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.090575 | 0.104750 | 0.162305 | 0.199632 | 0.008122 | 0.006077 |
| ConditionalQA | 0.080422 | 0.110814 | 0.289074 | 0.472047 | 0.015268 | -0.029127 |
| HotpotQA | 0.096653 | 0.106772 | 0.218803 | 0.266159 | 0.023048 | 0.067258 |
| MultiHop-RAG | 0.087890 | 0.093685 | 0.415231 | 0.510207 | 0.012398 | -0.002283 |

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| feature_schema_frozen_before_run | `true` | `True` | `True` |
| target_train_fitted_and_standardized | `true` | `True` | `True` |
| evaluation_not_used_for_selection | `true` | `True` | `True` |
| threshold_calibration_only | `true` | `True` | `True` |
| inference_features_exclude_gold_fields | `true` | `True` | `True` |
| minimum_mean_complete_recall_gain | `0.090658` | `>=0.02` | `True` |
| maximum_dataset_complete_recall_decrease | `{"2WikiMultiHopQA": -0.037326, "ConditionalQA": -0.182973, "HotpotQA": -0.047355, "MultiHop-RAG": -0.094976}` | `<=0.01` | `True` |
| maximum_dataset_case_risk_increase | `{"2WikiMultiHopQA": 0.014174, "ConditionalQA": 0.030392, "HotpotQA": 0.01012, "MultiHop-RAG": 0.005795}` | `<=0.01` | `False` |
| maximum_each_dataset_mean_case_risk | `{"2WikiMultiHopQA": 0.10475, "ConditionalQA": 0.110814, "HotpotQA": 0.106772, "MultiHop-RAG": 0.093685}` | `<=0.1` | `False` |
| minimum_mean_evaluation_auc_gain | `0.014709` | `>=0.01` | `True` |
| minimum_datasets_with_positive_auc_gain | `{"count": 4, "per_dataset": {"2WikiMultiHopQA": 0.008122, "ConditionalQA": 0.015268, "HotpotQA": 0.023048, "MultiHop-RAG": 0.012398}}` | `>=3` | `True` |
| minimum_datasets_with_at_least_7_of_10_alpha_control_repeats | `{"count": 0, "per_dataset": {"2WikiMultiHopQA": 6, "ConditionalQA": 3, "HotpotQA": 6, "MultiHop-RAG": 6}}` | `>=4` | `False` |
| minimum_mean_worst_eligible_subgroup_risk_reduction | `{"mean": 0.010481, "per_dataset": {"2WikiMultiHopQA": 0.006077, "ConditionalQA": -0.029127, "HotpotQA": 0.067258, "MultiHop-RAG": -0.002283}}` | `>=0.01` | `True` |
| maximum_dataset_worst_eligible_subgroup_risk_increase | `{"2WikiMultiHopQA": -0.006077, "ConditionalQA": 0.029127, "HotpotQA": -0.067258, "MultiHop-RAG": 0.002283}` | `<=0.02` | `False` |
| all_finite_sample_nominal_bounds_at_or_below_alpha | `{"count": 40, "maximum": 0.1}` | `True` | `True` |

## 结论边界

该实验只检验冻结多阶段检索分数的一致性是否为新的可观察充分性信号。它使用目标 train 拟合并在既有公开 evaluation 上事后开发，不是独立确认；无论结果如何，Gate 2 均保持 `NO-GO/SHADOW`。
