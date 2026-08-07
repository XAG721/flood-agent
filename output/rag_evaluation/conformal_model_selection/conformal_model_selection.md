# FRC-RAG 证据充分性嵌套选型实验

- 状态：`RUN_PUBLIC_REAL_MODEL_NESTED_GROUP_SELECTION`
- 外层预注册分组：10 组
- 每组内层重复：3 次
- 候选配置：12 个
- alpha：0.1
- Gate 2：`NO-GO/SHADOW`

## Outer evaluation 汇总

| 指标 | 固定线性 L2=4 | train 内嵌套选型 | 配对变化 |
|---|---:|---:|---:|
| case 家族误放行率均值 | 0.080422 | 0.121352 | +0.040930 |
| 拒答率均值 | 0.930263 | 0.907726 | -0.022537 |
| 完整召回率均值 | 0.289074 | 0.354834 | +0.065759 |
| 放行精度均值 | 0.602499 | 0.564146 | -0.038353 |

## 每个 outer 分组的选型

| Outer 分组 | 选中配置 | 固定召回 | 选中召回 | 固定风险 | 选中风险 |
|---|---|---:|---:|---:|---:|
| `frc-conformal-sufficiency-split-v1` | `linear-l2-16` | 0.295455 | 0.363636 | 0.040541 | 0.040541 |
| `frc-conformal-sufficiency-split-v1-repeat-01` | `quadratic-l2-4` | 0.181818 | 0.181818 | 0.062500 | 0.075000 |
| `frc-conformal-sufficiency-split-v1-repeat-02` | `quadratic-l2-1` | 0.375000 | 0.410714 | 0.147727 | 0.136364 |
| `frc-conformal-sufficiency-split-v1-repeat-03` | `quadratic-l2-16` | 0.058824 | 0.098039 | 0.038462 | 0.089744 |
| `frc-conformal-sufficiency-split-v1-repeat-04` | `sqrt-l2-1` | 0.277778 | 0.611111 | 0.103896 | 0.259740 |
| `frc-conformal-sufficiency-split-v1-repeat-05` | `sqrt-l2-16` | 0.326923 | 0.384615 | 0.037037 | 0.086420 |
| `frc-conformal-sufficiency-split-v1-repeat-06` | `log1p-l2-1` | 0.326531 | 0.326531 | 0.088608 | 0.151899 |
| `frc-conformal-sufficiency-split-v1-repeat-07` | `linear-l2-1` | 0.285714 | 0.257143 | 0.075949 | 0.101266 |
| `frc-conformal-sufficiency-split-v1-repeat-08` | `sqrt-l2-1` | 0.423077 | 0.480769 | 0.115385 | 0.166667 |
| `frc-conformal-sufficiency-split-v1-repeat-09` | `sqrt-l2-16` | 0.339623 | 0.433962 | 0.094118 | 0.105882 |

## 预注册采用判据

- `mean_complete_recall_gain_at_least_0_05`：`PASS`
- `mean_abstention_reduction_at_least_0_05`：`FAIL`
- `mean_case_family_false_complete_increase_at_most_0_02`：`FAIL`
- `evaluation_does_not_select_candidate`：`PASS`

实验默认切换：`False`。

所有候选配置都只在 outer train 的三次内层 fit/calibration/validation case 分组上比较；内层 validation 不参与定阈值，outer calibration 只定最终阈值，outer evaluation 不参与选型。即使采用判据通过，系统仍保持 `NO-GO/SHADOW`。

- All outer repeats reuse the same cross-domain ConditionalQA cases.
- The candidate grid and three-way inner split are pre-registered; no outer evaluation result adds or removes a candidate.
- Synthetic missingness and benchmark evidence ids do not replace flood-domain double-expert labels.
- Passing the experimental adoption checks would not authorize CANARY or DEFAULT.
