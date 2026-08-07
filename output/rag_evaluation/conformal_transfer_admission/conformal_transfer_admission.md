# FRC-RAG 跨域充分性头 calibration-only 准入守卫

- 状态：`DO_NOT_ADOPT`
- 数据集：2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- 重复总数：40
- 证书使用 evaluation：`False`
- 证书使用目标 train：`False`
- 修改模型/阈值/声明：`False`
- Gate 2：`NO-GO/SHADOW`

## 证书结果

- 通过证书：3
- 数据集分布：`{"2WikiMultiHopQA": 0, "ConditionalQA": 2, "HotpotQA": 1, "MultiHop-RAG": 0}`
- 误准入（evaluation 风险 > alpha）：3
- 低效用准入（完整召回 < 0.10）：0
- 安全且有效证书精度：0.000000

## 预注册采用检查

| 检查 | 实测 | 目标 | 通过 |
|---|---|---|---|
| certificate_uses_calibration_only | `true` | `True` | `True` |
| minimum_total_certified_repeats | `3` | `>=8` | `False` |
| minimum_datasets_with_at_least_one_certificate | `{"count": 2, "per_dataset": {"2WikiMultiHopQA": 0, "ConditionalQA": 2, "HotpotQA": 1, "MultiHop-RAG": 0}}` | `>=2` | `True` |
| maximum_unsafe_certificate_count | `3` | `<=0` | `False` |
| maximum_low_utility_certificate_count | `0` | `<=0` | `True` |
| minimum_safe_and_useful_certificate_precision | `0.0` | `>=1.0` | `False` |

## 边界

该守卫只能拒绝跨域头，不能改善其排序能力或加强 conformal 保证。证书在已知迁移失败后设计，属于事后方法开发。

- 下一步：`REQUIRE_TARGET_FITTED_HEAD_STOP_GUARD_TUNING`
