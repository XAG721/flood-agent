# QuAC RoBERTa QA 支持迁移实验综合结论（v62）

## 1. 研究问题

v62 检验一个比 v58—v61 更受约束的假设：不再让生成模型按提示词自报“支持/不支持”，而是把公开的 `deepset/roberta-base-squad2` 问答模型固定在 revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`，用模型原生的 CLS 空答案分数与最优合法 span 分数比较，直接形成支持门。模型不在 QuAC 上微调、校准或拟合阈值，QuAC validation 仅在所有开发门槛通过时才允许打开。

这项实验只检验“公开 SQuAD2 抽取式问答器能否跨域充当 QuAC 支持判断器，以及同一支持门下 FRC 选择器是否仍有增量”，不是 QuAC 官方答案生成评测，也不是 SetR 复现或洪水领域有效性证明。

## 2. 冻结设计与可复现性

- 开发数据：QuAC v0.2 `train_v0.2.json`，SHA-256 `ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a`。
- 样本：600 条，答案/无答案各 300 条，覆盖 587 段对话和 566 份文档；与 v57 的 600 条承诺样本重叠为 0。
- 模型：`deepset/roberta-base-squad2`，固定 revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`，不使用 QuAC 训练或微调。
- 权重登记：官方 PyTorch state dict SHA-256 为 `e0b64ccefc1bcb569b604baea27eb873e5482fdf6eb3ceff1fb5368397db5aed`；仅作等值安全序列化得到 `model.safetensors`，SHA-256 为 `5da10c5315517d9c8750f2e0ff4fd319476b4d93ae0e7f346f120d4c855bfa8c`，未改变权重值或网络结构。
- 支持规则：所有 overflow feature 中的最小 CLS 空答案分数与最大合法 span 分数直接比较；只有 `best_span_score > null_score` 才记为支持，平局、无效值和非有限值均 fail closed。
- 盲评边界：QA 输入只含 `id/question/context`；检索和 QA 缓存完整后才连接 gold。查询、神经评分和 QA 无回退或无效输出。
- 共享门公平性：所有 gated FRC 与非 FRC 方法使用同一条 RoBERTa 支持判断，不允许为候选方法单独调阈值。

协议、实现登记、模型登记、执行登记、结果和逐例压缩证据均有独立哈希；模型登记明确披露本地安全序列化的来源与过程。

## 3. 开发集结果

| 指标 | v62 结果 | 预注册门槛 | 结论 |
|---|---:|---:|---|
| 支持判断平衡准确率 | 0.590000 | ≥ 0.750000 | 失败 |
| 答案样本支持通过率 | 0.686667 | ≥ 0.750000 | 失败 |
| 无答案拒绝率 | 0.493333 | ≥ 0.650000 | 失败 |
| 候选 utility F1 | 0.308444 | 描述性 | — |
| 候选答案 F1 | 0.123556 | 描述性 | — |
| 候选答案召回 | 0.139722 | ≥ 0.700000 | 失败 |
| 候选无答案弃权准确率 | 0.493333 | ≥ 0.650000 | 失败 |
| 候选弃权率 | 0.403333 | [0.15, 0.70] | 通过 |

候选 `roberta_supported_adaptive_argmax_cardinality_frc_v43_v62` 相对共享门控精确锚点为 −0.013077，95% CI [−0.027091,+0.001271]；相对最强共享门非 FRC `roberta_supported_cross_encoder_topk_v62` 同为 −0.013077；相对最强同门 FRC `roberta_supported_guarded_adaptive_cardinality_frc_v44_v62` 为 −0.013272，95% CI [−0.026309,−0.000087]。支持门本身相对未门控精确锚点提高 0.219598，说明它会显著增加弃权，但并未形成足够准确且有用的跨域支持判断。

## 4. 结论与停止边界

结果固定为 `QUAC_V62_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION`。SQuAD2 训练的抽取式 QA 模型在 QuAC 上出现明显跨域失配：它同时漏掉大量可回答样本，也未能可靠拒绝无答案样本；FRC 候选在相同门控条件下还弱于最强非 FRC 和同门 FRC 对照。因此：

- QuAC validation 未解析、未打开，且不授权打开；
- v62 开发样本不得用于阈值拟合、调参、方法选择或支持性确认；
- 不采用该支持门或候选选择器，不进入 CANARY/DEFAULT；
- Gate 2 保持 `NO-GO/SHADOW`；
- 该负结果只否定本次固定的直接迁移方案，不否定使用目标域训练/校准、独立确认或其他可复现支持模型的后续研究。

## 5. 主要产物

- 协议：`quac_roberta_qa_support_transfer_protocol_v62.json`
- 模型登记：`quac_roberta_qa_support_transfer_model_registration_v62.json`
- 实现与执行登记：`quac_roberta_qa_support_transfer_implementation_v62.json`、`quac_roberta_qa_support_transfer_development_execution_v62.json`
- 锁定结果与关闭记录：`quac_roberta_qa_support_transfer_development_result_v62.json`、`quac_roberta_qa_support_transfer_development_closure_v62.json`
- 报告与逐例证据：`../../output/rag_evaluation/quac_roberta_qa_support_transfer/development/report.md`、`cases.jsonl.gz`

可使用以下命令重新聚合已冻结缓存；该命令不得自动打开 validation：

```powershell
D:\anaconda3\envs\rag_exp\python.exe scripts/run_quac_roberta_qa_support_transfer.py evaluate --stage development
```
