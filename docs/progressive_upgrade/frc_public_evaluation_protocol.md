# FRC-Select 公开数据公平评测协议

执行日期：2026-07-13。本协议用于验证 FRC-Select 的工程与理论流水线是否可行，不预设其优于基线，也不把覆盖贪心代理表述为真实 SetR。

## 1. 数据集与难例覆盖

| 难例 | 数据来源 | 当前状态 | 用途 |
|---|---|---|---|
| 正常检索 | ConditionalQA、MultiHop-RAG、HotpotQA | 已运行 | 检查常规证据召回与选择 |
| 条件、例外、不可回答 | [ConditionalQA](https://arxiv.org/abs/2110.06884) | 已运行 | 条件答案、长文档、多跳与 supporting evidence |
| 跨文档 | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/) | 已运行 | 2—4 篇文档联合证据 |
| 多跳证据 | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/)、[HotpotQA](https://hotpotqa.github.io/) | 已运行 | 多段 supporting facts 与干扰文档 |
| 缺失证据 | ConditionalQA 的确定性变换切片 | 已运行 | 对每个至少含两条金证据的用例删除首条金证据，再用保存的真实模型分数重新选择 |
| 冲突证据 | [Google CONFLICTS](https://github.com/google-research-datasets/rag_conflicts) | 已运行 | 458 例官方五类标签；六方法同预算选择后由同一本地 Qwen 分类 |
| 失效/过期文件 | CONFLICTS 的 `Conflict due to outdated information` | 已运行 | 62 例；报告类型 Recall、最新日期来源保留和时间端点覆盖 |

Google 对 CONFLICTS 的研究说明给出了检索增强生成中的知识冲突分类与专家标注评测背景，参见 [Dragged into a Conflict](https://research.google/pubs/dragged-into-a-conflict-detecting-and-addressing-conflicting-sources-in-search-augmented-llms/)。本仓库已记录 Apache-2.0 许可、数据 SHA-256、真实模型配置和逐方法结果；但没有复现论文依赖独立人类判断的 expected-behavior adherence，因此不是官方 leaderboard 结果。

## 2. 公平性约束

- 所有方法使用同一候选池、同一问题、同一金证据和同一数据划分。
- 统一 `Top-K=5`、证据预算 `1500 tokens`。
- 真实评分器固定为 `BAAI/bge-large-en-v1.5` 与 `BAAI/bge-reranker-large`。
- 分数校准固定为 `per_case_minmax`，`role_relevance_mix=0.15`。
- FRC 参数固定为 `alpha=2.0`、`beta=1.0`、`gamma=0.0`；任何调参必须形成新实验版本，不能覆盖本报告。
- 生成器固定为本地 `Qwen2.5-7B-Instruct-GPTQ-Int4`；证据选择指标与答案生成指标分开陈述。
- 基线包含 BM25、Dense、Hybrid、Cross-Encoder Top-K、MMR 和覆盖贪心代理。导入产物中的历史名称 `setr_style` 统一展示为 `coverage_greedy_proxy`；它没有使用 SetR 官方代码、权重或训练流程。
- 主比较采用逐样本配对差值和固定种子 2,000 次 bootstrap 95% 置信区间；区间跨 0 时不得声称显著更优。

## 3. 可复现输入与命令

输入来自只读参考目录 `D:\RAG_test\frc-select\outputs`，包含三套数据的汇总 CSV、逐样本候选分数和各方法选择结果。导入程序会在报告中记录关键文件 SHA-256，不修改参考目录。

```powershell
python scripts/import_frc_public_reference.py `
  --reference-root D:\RAG_test\frc-select `
  --output-dir output\rag_evaluation\public_frc_reference
```

输出：

- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.json`
- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.md`

CONFLICTS 全量评测在已缓存模型的 `rag_exp` 环境执行：

```powershell
conda run -n rag_exp python -m scripts.run_conflicts_frc_evaluation `
  --hf-home D:\RAG_test\.hf_cache `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
```

官方数据共 458 例，其中 237 例含非空正确答案、62 例为过时信息冲突、5 例为错误信息冲突。覆盖贪心代理 Accuracy 为 0.344978，FRC 为 0.334061；FRC 相对最强基线差值为 -0.010917，配对 95% CI 为 [-0.043668, +0.024017]。FRC 的过时信息类型 Recall 为 0.564516，覆盖贪心代理为 0.693548。完整结果见 `../../output/rag_evaluation/conflicts_frc/conflicts_frc_report.md`。

## 4. Gate 2 判据

本轮结论为 `THEORETICAL_PIPELINE_FEASIBLE_BUT_SUPERIORITY_NOT_PROVEN`：公开数据、真实神经评分器、FRC 选择和本地生成器已经形成可复现流水线，但三套主数据上 FRC 相对各自最强可复现基线的 Evidence F1 配对 95% 区间均跨 0。

因此 Gate 2 保持 `NO-GO/SHADOW`。至少补齐以下证据后才可重新评审 `CANARY`：

1. 基于 CONFLICTS 的冲突/过时类型分类已经完成，但还需复现 expected-behavior adherence 并由独立人员复核；
2. 主指标与安全指标预先冻结，不能只挑有利切片；
3. 主要数据集上相对最强可复现基线的改善具有一致方向和配对统计支持；
4. 冲突漏报、错误完整声明、不可回答误答等安全指标达到门槛；
5. 原始输入、配置、种子、哈希、逐样本结果与失败用例可审计。
