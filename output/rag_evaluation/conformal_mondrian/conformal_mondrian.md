# FRC-RAG 分层 Mondrian conformal 事后方法开发

- 状态：`DO_NOT_ADOPT`
- 数据集：ConditionalQA, HotpotQA, MultiHop-RAG
- alpha：0.1
- 回退顺序：joint → question_type → required_role_count → global；分组阈值最低校准 case 数：30
- 独立确认：`False`
- Gate 2：`NO-GO/SHADOW`

## 与全局阈值的冻结比较

| 数据集 | 全局 case 风险 | Mondrian case 风险 | 全局完整召回 | Mondrian 完整召回 | 最坏子群风险改善 | 分组阈值占比 |
|---|---:|---:|---:|---:|---:|---:|
| ConditionalQA | 0.080422 | 0.075427 | 0.289074 | 0.290268 | 0.005077 | 1.000000 |
| HotpotQA | 0.096653 | 0.093480 | 0.218803 | 0.201092 | 0.082985 | 1.000000 |
| MultiHop-RAG | 0.087890 | 0.074545 | 0.415231 | 0.348327 | 0.053930 | 1.000000 |

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| no_evaluation_selection | `true` | `True` | `True` |
| all_threshold_sources_calibration_only | `true` | `True` | `True` |
| mean_worst_eligible_subgroup_risk_reduction | `0.047331` | `>=0.02` | `True` |
| maximum_dataset_overall_case_risk_increase | `{"ConditionalQA": -0.004995, "HotpotQA": -0.003173, "MultiHop-RAG": -0.013345}` | `<=0.01` | `True` |
| maximum_dataset_complete_recall_decrease | `{"ConditionalQA": -0.001194, "HotpotQA": 0.017711, "MultiHop-RAG": 0.066904}` | `<=0.05` | `False` |
| group_specific_threshold_case_fraction | `1.0` | `>=0.5` | `True` |
| all_eligible_subgroup_mean_risks_at_or_below_alpha | `true` | `True` | `True` |
| all_eligible_subgroup_repeat_consistency_targets_met | `false` | `True` | `False` |

## 结论边界

该实验是在已查看子群不稳定结果后开展的方法开发，不是独立确认。只有冻结方法后在全新未触碰数据或独立防汛专家数据上复现，才可重新评审；当前继续保持 `NO-GO/SHADOW`。
