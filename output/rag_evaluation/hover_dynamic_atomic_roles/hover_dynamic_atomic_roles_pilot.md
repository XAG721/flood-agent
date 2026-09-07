# HoVer 动态原子核验角色机制门槛（v41）

- 状态：`MECHANISM_ESTABLISHED_OPEN_CONFIRMATION`
- 样例：512（未连接 label、hop 或 gold）
- 静态/动态角色内部平均 Spearman：0.945471 / 0.521314
- 相关性下降：+0.424158
- 静态/动态平均不同角色首选数：1.154297 / 2.230469
- 静态与动态有序选择完全相同率：0.432292

## 预注册检查

- `generation_fallback_rate_at_most_0_05`：PASS
- `dynamic_spearman_reduction_at_least_0_05`：PASS
- `dynamic_distinct_argmax_gain_at_least_0_25`：PASS
- `ordered_selection_identity_at_most_0_90`：PASS
- `no_forbidden_field_leak`：PASS

## 边界

本报告只验证角色信号是否真正改变排序结构，不使用任何正确性标签，也不构成检索性能、真实防汛有效性、SetR 复现或 Gate 2 放行证据。
