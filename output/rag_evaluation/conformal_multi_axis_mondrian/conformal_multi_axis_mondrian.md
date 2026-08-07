# FRC-RAG 多轴 Mondrian conformal 事后方法开发

- 状态：`DO_NOT_ADOPT`
- 数据集：2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- alpha：0.1
- 回退顺序：question_type_candidate_count_bucket_required_role_count → question_type_candidate_count_bucket → candidate_count_bucket_required_role_count → question_type_required_role_count → candidate_count_bucket → question_type → required_role_count → global
- 分组阈值最低校准 case 数：30
- 事后方法开发：`True`；独立确认：`False`
- Gate 2：`NO-GO/SHADOW`

## 与全局阈值的冻结比较

| 数据集 | 全局 case 风险 | 多轴 case 风险 | 全局完整召回 | 多轴完整召回 | 最坏子群风险改善 | 分组阈值占比 |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.090575 | 0.070206 | 0.162305 | 0.148165 | 0.100900 | 1.000000 |
| ConditionalQA | 0.080422 | 0.072862 | 0.289074 | 0.286422 | 0.007709 | 1.000000 |
| HotpotQA | 0.096653 | 0.099623 | 0.218803 | 0.212823 | 0.060318 | 1.000000 |
| MultiHop-RAG | 0.087890 | 0.074545 | 0.415231 | 0.348327 | 0.053930 | 1.000000 |

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| no_evaluation_selection | `true` | `True` | `True` |
| all_threshold_sources_calibration_only | `true` | `True` | `True` |
| mean_worst_eligible_subgroup_risk_reduction | `0.055714` | `>=0.02` | `True` |
| maximum_dataset_overall_case_risk_increase | `{"2WikiMultiHopQA": -0.020369, "ConditionalQA": -0.00756, "HotpotQA": 0.00297, "MultiHop-RAG": -0.013345}` | `<=0.01` | `True` |
| maximum_dataset_complete_recall_decrease | `{"2WikiMultiHopQA": 0.01414, "ConditionalQA": 0.002652, "HotpotQA": 0.00598, "MultiHop-RAG": 0.066904}` | `<=0.05` | `False` |
| group_specific_threshold_case_fraction | `1.0` | `>=0.5` | `True` |
| all_eligible_subgroup_mean_risks_at_or_below_alpha | `false` | `True` | `False` |
| all_eligible_subgroup_repeat_consistency_targets_met | `false` | `True` | `False` |

## 决策边界

该层级是在查看四数据集子群结果后冻结的事后开发规则，不能作为独立确认。只有全部采用检查通过才会下载并运行未触碰的 MuSiQue；否则保留负结果并停止扩张。

- 下一步：`STOP_WITHOUT_MUSIQUE_DOWNLOAD`
- 当前仍保持 `NO-GO/SHADOW`。
