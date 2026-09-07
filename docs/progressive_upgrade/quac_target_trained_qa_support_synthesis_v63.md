# QuAC 目标域训练 QA 支持门实验综合结论（v63）

## 1. 研究问题与边界

v63 检验 v62 失败后的一个更直接假设：固定使用公开声明曾依次在 SQuAD2.0 与 QuAC 上微调的 `ixa-ehu/SciBERT-SQuAD-QuAC`，以模型原生 CLS 空答案分数与最优合法 span 分数比较，能否在 QuAC 上形成足以开放后续 FRC/非 FRC 公平检索比较的支持门。

协议、实现和模型 revision 均在首次解析 QuAC validation 前登记；本地不微调、不校准、不拟合阈值，也不复用 v57/v62 的逐例预测或标签。公开模型的精确 QuAC 训练 split、checkpoint 选择过程和模型许可证没有完整披露，因此本实验仅是来源受限的可行性评估，不声明严格独立确认、模型再分发权或生产可用性。

## 2. 分阶段盲执行

验证集按冻结 SHA-256 顺序选择 600 条机制样本，答案/无答案各 300 条，覆盖 477 段对话和 477 份文档；schema 排除率为 0。QA 输入只含盲化后的 `id`、当前问题和上下文，完整 600 行支持预测缓存写入后才连接 gold，验证器无效输出为 0，且未接触答案状态或答案文本。

v63 采用两级计算授权：只有支持门全部通过，才允许生成检索查询并运行 BGE dense/reranker 评分。这样可避免在支持判断本身不成立时继续计算并事后筛选 FRC 结果。

## 3. 支持门结果

| 指标 | v63 结果 | 预注册门槛 | 结论 |
|---|---:|---:|---|
| 样本数 | 600 | = 600 | 通过 |
| 答案/无答案 | 300 / 300 | 300 / 300 | 通过 |
| schema 排除率 | 0.000000 | ≤ 0.010000 | 通过 |
| QA 无效输出率 | 0.000000 | = 0 | 通过 |
| 支持判断平衡准确率 | 0.493333 | ≥ 0.750000 | 失败 |
| 答案例支持通过率 | 0.486667 | ≥ 0.750000 | 失败 |
| 无答案例拒绝率 | 0.500000 | ≥ 0.650000 | 失败 |

三个核心支持指标全部失败，结果接近随机平衡判断。公开模型曾在目标数据集上训练的声明没有转化为可用的无答案支持门；模型原生 null-vs-span 分数也不能直接视作经验证的支持概率。

## 4. 停止决定

状态锁定为 `QUAC_V63_VALIDATION_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_RETRIEVAL_SCORING`。按预注册停止规则：

- 不生成检索查询，不运行 BGE dense/reranker 评分；
- 不产生或比较 v63 的 FRC 与非 FRC 检索效用结果；
- 不对 validation 拟合阈值、调参或选择下一方法；
- 不采用该支持门或任何选择器；
- 不授权 CANARY/DEFAULT，Gate 2 保持 `NO-GO/SHADOW`。

v62 的事后标量阈值诊断最高平衡准确率仅 0.63，且答案通过率只有 0.50；该诊断未向 v63 传递阈值或逐例决定。v63 进一步说明，简单更换为公开目标域训练 QA checkpoint 仍不足以解决支持判断问题。后续研究若继续，应使用训练/验证来源和许可证明确的数据与模型，并将支持校准限定在独立开发集；当前 validation 不得再次用于方法选择。

## 5. 可复现证据

- 协议：`quac_target_trained_qa_support_protocol_v63.json`
- 实现登记：`quac_target_trained_qa_support_implementation_v63.json`
- 模型登记：`quac_target_trained_qa_support_model_registration_v63.json`
- 执行登记：`quac_target_trained_qa_support_validation_execution_v63.json`
- 支持结果与关闭记录：`quac_target_trained_qa_support_validation_support_result_v63.json`、`quac_target_trained_qa_support_validation_support_closure_v63.json`
- 人类可读报告：`../../output/rag_evaluation/quac_target_trained_qa_support/support_gate/report.md`
- 逐例盲证据：`../../output/rag_evaluation/quac_target_trained_qa_support/support_gate/cases.jsonl.gz`
