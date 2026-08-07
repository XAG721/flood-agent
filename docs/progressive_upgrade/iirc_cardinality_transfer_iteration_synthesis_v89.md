# v85—v89 IIRC 证据基数路由迭代综合报告

## 1. 结论

v89 已在 IIRC 上建立“冻结 v88 模型、案例互斥、同数据集”的源级证据选择复制支持。开发集和一次性确认集分别为 800/1200 例，均在协议、模型、适配器、对照、门禁和目标 ID 承诺锁定后完成盲评分，两个阶段全部严格门禁通过。

| 阶段 | 候选 F1 | `frc_fixed2` F1 | 配对差值 | 95% CI | 最强观察对照 | 相对最强对照 | 完整证据召回 | 平均源数 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 开发 800 | 0.636527 | 0.606125 | +0.030402 | [+0.017976, +0.042557] | v87，0.620408 | +0.016119 | 0.427500 | 1.925000 |
| 确认 1200 | 0.627097 | 0.598195 | +0.028902 | [+0.017896, +0.039656] | v87，0.616161 | +0.010936 | 0.433333 | 1.930833 |

这支持以下窄结论：冻结的 154 维分数差特征和 ExtraTrees 基数策略，在新的 IIRC 问题上能够重复优于全部 13 个登记对照，并保持完整证据召回、平均证据数和动作覆盖门槛。

它不支持跨训练数据集迁移、答案生成质量、真实 SetR 复现、洪水领域有效性、选择器采用或生产放行。因此 Gate 2 继续为 `NO-GO/SHADOW`，CANARY、DEFAULT 与生产切流均未授权。

## 2. 迭代链

### v85：BeerQA 来源资格关闭

BeerQA 官方 dev 文件包含 8,132 条 SQuAD 来源和 5,989 条 HotpotQA 来源记录，没有官网描述的可独立评测 3+ hop 新标注子集。因为既有研究已使用 SQuAD 与 HotpotQA，BeerQA 不能作为独立新数据集验证；实验在模型评分和指标前以 `SOURCE_INADMISSIBLE_NO_INDEPENDENT_EVALUABLE_SUBSET` 关闭，没有替换样本或重标来源。

### v86：HotpotQA 基数模型零调参迁移到 IIRC

v86 首先登记 IIRC 原始 train/dev 与上下文归档的来源、大小和 SHA-256，然后把冻结的 HotpotQA 问题类型基数模型直接迁移到 400 条 IIRC 开发样本。候选 F1 为 0.616381，低于最强 `frc_fixed2` 的 0.620893，差值 -0.004512，95% CI [-0.018155, +0.001346]；动作 2 占 93.5%，5 项严格门失败，确认集保持未打开。

该结果建立了 IIRC 源级盲评管线，但否定了“HotpotQA 问题基数模型无需目标域训练即可优于固定二源 FRC”的假设。

### v87：IIRC 分数差路由器

v87 只使用已揭示的 v86 开发样本训练 154 维无 gold 分数差模型，并在新的 400 条 IIRC 样本上评测。候选 F1 0.616637，相对 `frc_fixed2` +0.007831，但 95% CI [-0.002405, +0.018010] 跨 0；相对全部控制的同时区间也跨 0，3 项严格门失败，确认未打开。

这说明目标域运行时分数具有正向路由信号，但 400 例不足以建立稳定优势。

### v88：汇聚两批开发证据

v88 将 v86/v87 共 800 条已揭示开发样本作为永久排除训练历史，冻结 154 维特征、ExtraTrees 150 棵树和动作偏置 `[-0.03, 0, 0.04, 0]`。交叉拟合候选 F1 0.639479，相对固定二源 +0.024630，95% CI [+0.011117, +0.037949]；两个训练队列互换训练/验证的点增益均为正。

新的 800 条 v88 开发集候选 F1 0.634559，相对 `frc_fixed2` +0.028773，95% CI [+0.015490, +0.041769]，全部 14 项严格门通过。但确认准备第 46 条样本只有主来源加两个有效链接来源，共 3 个候选；冻结适配器要求至少 4 个候选，因此在任何确认缓存或指标写出前失败关闭。该失败是适配覆盖问题，不是模型质量失败，整个 v88 确认承诺永久退出后续目标池。

### v89：低候选鲁棒适配与复制确认

v89 不重训、不调参，原样复用 v88 模型。唯一适配变化是：

- 候选源不少于 4 时，执行完全冻结的 v88 模型；
- 候选源为 1—3 时，选择所有真实可用源，不调用四源模型；
- 每个控制方法请求的前缀长度上限为实际可用唯一源数，并报告实际基数；
- 不丢弃、不替换样本，不制造占位源，也不读取答案或 gold。

v89 目标只从 v88 剩余承诺中抽取。开发集含 1 条三候选样本，适配覆盖率为 1.0；该单例只证明程序覆盖，不能证明低候选性能。确认集候选数范围为 8—43，因此全程走冻结模型。

开发与确认阶段的候选动作分布分别为 `1/2/3 = 277/306/217` 和 `412/459/329`，三种动作均得到实质使用。候选在两个阶段相对全部 13 个登记对照的点差均为正；相对主控制的配对区间下界均大于 0，全部严格门禁通过。

## 3. 公平性与泄漏边界

- 目标 ID、阶段规模、抽样盐值、模型、适配器、13 个对照、指标、bootstrap 种子和门禁均在目标内容访问前锁定。
- 开发/确认分别为 800/1200 个唯一 ID，彼此及与 v86—v88 已暴露目标零重叠。
- 盲缓存递归扫描未发现 `answer`、`answer_type`、`context`、`question_links` 或 `gold_evidence_sources` 字段。
- BGE 与 reranker 分数、各方法选择输出均在密封 gold 联结前完整写定。
- 开发集全部门禁通过后才创建确认开封凭证；确认内容、评分和指标此前均不存在。
- v89 没有使用确认结果选择模型、阈值、适配器、门禁或对照。

## 4. 仍未满足的 Gate 2 条件

v89 是重要的公开数据源选择证据，但不是完整 FRC-RAG 放行证据：

1. v86 的 HotpotQA 到 IIRC 零调参迁移失败，因此没有建立独立训练数据集迁移优势。
2. 当前指标只评价金证据来源选择，没有评价答案正确性、引用正确性、冲突处置或拒答质量。
3. 没有复现真实 SetR；论文权重未公开且本地不具备公平训练条件时，继续使用可复现的公开控制，不能把模拟控制写成 SetR。
4. IIRC 不是洪水业务数据，不能替代防汛规则、预案、案例和双专家盲评。
5. 没有生产流量、安全、时延、容量、回滚与机构 UAT 证据。

因此当前状态固定为：选择器不采用，Gate 2 `NO-GO/SHADOW`。后续优先级应是独立训练来源迁移、端到端回答与引用盲评，以及真实防汛双专家验证，而不是继续在已揭示 IIRC 目标上调参。

## 5. 关键可复现资产

- [v89 协议](iirc_low_candidate_robustness_protocol_v89.json)
- [v89 来源登记](iirc_low_candidate_robustness_source_registration_v89.json)
- [v89 实现锁](iirc_low_candidate_robustness_implementation_v89.json)
- [v89 开发结果](iirc_low_candidate_robustness_development_result_v89.json)
- [v89 确认开封凭证](iirc_low_candidate_robustness_confirmation_open_v89.json)
- [v89 确认结果](iirc_low_candidate_robustness_confirmation_result_v89.json)
- [v89 关闭记录](iirc_low_candidate_robustness_closure_v89.json)
- [开发报告](../../output/rag_evaluation/iirc_low_candidate_robustness_v89/iirc_low_candidate_robustness_development_v89.md)
- [确认报告](../../output/rag_evaluation/iirc_low_candidate_robustness_v89/iirc_low_candidate_robustness_confirmation_v89.md)

复跑时必须保留冻结模型 revision 与协议边界：

```powershell
D:\anaconda3\envs\rag_exp\python.exe scripts\run_iirc_low_candidate_robustness_transfer.py prepare --stage development --hf-home D:\RAG_test\.hf_cache
D:\anaconda3\envs\rag_exp\python.exe scripts\run_iirc_low_candidate_robustness_transfer.py score --stage development --hf-home D:\RAG_test\.hf_cache
D:\anaconda3\envs\rag_exp\python.exe scripts\run_iirc_low_candidate_robustness_transfer.py select --stage development
D:\anaconda3\envs\rag_exp\python.exe scripts\run_iirc_low_candidate_robustness_transfer.py evaluate --stage development
```

确认阶段只能在开发结果全门禁通过并创建开封凭证后执行；已揭示确认结果不得重新包装为新的独立确认。
