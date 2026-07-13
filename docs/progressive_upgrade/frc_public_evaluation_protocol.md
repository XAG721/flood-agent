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
$bgeSnapshot = (Get-ChildItem -Directory D:\RAG_test\.hf_cache\hub\models--BAAI--bge-large-en-v1.5\snapshots | Select-Object -First 1).FullName
$rerankerSnapshot = (Get-ChildItem -Directory D:\RAG_test\.hf_cache\hub\models--BAAI--bge-reranker-large\snapshots | Select-Object -First 1).FullName
python scripts/run_frc_wo_reranker_ablation.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --model-name $bgeSnapshot --device cuda
python scripts/run_frc_chunk_length_sensitivity.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --reranker-model-path $rerankerSnapshot --device cuda `
  --chunk-lengths 64,128,256 --overlap-ratio 0.2
python scripts/import_frc_public_reference.py `
  --reference-root D:\RAG_test\frc-select `
  --conflicts-path output\rag_evaluation\conflicts_frc\conflicts_frc_report.json `
  --supplemental-ablation output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --chunk-length-sensitivity output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --controlled-domain-sensitivity output\rag_evaluation\controlled_domain_sensitivity\controlled_domain_sensitivity.json `
  --output-dir output\rag_evaluation\public_frc_reference
```

先运行 `python scripts/run_frc_controlled_sensitivity.py` 可重建字段/角色权重与冲突阈值的受控诊断产物。该产物固定为 `SYNTHETIC`、无神经模型、单因素扫描；导入器会逐样本重算聚合值并验证来源，且将其标记为 `RUN_CONTROLLED_DOMAIN_PUBLIC_SCHEMA_BLOCKED`，不会把它计入公开真实模型完成度。

输出：

- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.json`
- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.md`

导入报告同时审计设计第 16.2 节实验覆盖：ConditionalQA 原始产物包含 5 组消融，仓库另以离线 BGE 双编码器完成真实 `w/o Reranker` 重评分，合计 6/9；三套数据均包含 K=2/3/5/8。Full Evidence F1 为 0.705754，`w/o Role` 为 0.706158，`w/o Reranker` 为 0.697327。Full 仍未优于 `w/o Role`，且 `w/o Field` 因主公开集没有字段标注而无法运行，因此 Gate 的消融硬条件失败。

Token 预算敏感性使用同一候选池、Top-K=5 和保存分数，严格限制 512/1024/2048 Token，任何方法超预算即失败；当前所有方法超预算率均为 0。缺失比例敏感性以固定 `SHA-256(case_id, evidence_id)` 分数与 0/25/50/75% 阈值比较，形成可重复、随比例嵌套的删除集合，不使用测试结果调参。字段/角色权重 0/0.5/1/2/4（字段另含 8）与冲突阈值 0/0.35/0.5/0.8/1 已在 3 例、18 文档的仓库构造受控领域诊断集上运行；16 项“字段→金证据”映射只用于评分，从不传入选择器。冻结策略（角色 1、字段 4、冲突阈值 0）的 Evidence F1 为 0.750000、金标准字段覆盖为 0.773810、选择器自报字段覆盖和角色覆盖均为 1、冲突候选用例率为 0。角色权重 0 使角色覆盖降至 0.888889，字段权重 0 只使选择器自报字段覆盖降至 0.944444而金标准字段覆盖不变；字段权重 8 或冲突阈值不低于 0.5 时 Evidence F1/金标准字段覆盖降至 0.607143/0.573810，且 1/3 用例选入冲突候选。该结果证明参数行为可辨识，也暴露了自报覆盖与金标准覆盖的差异、过度加权和放宽冲突阈值的风险，但仅为 `RUN_CONTROLLED_DOMAIN_PUBLIC_SCHEMA_BLOCKED`。机器可读 Schema 审计覆盖 3,841 例、172,212 个候选，确认主公开集只有 Cross-Encoder 和角色分，而无字段分、适用性或候选级冲突标注；`SCHEMA_BLOCKED` 不是通过，仍需另建领域真实模型标注基准。

分块长度采用 reranker tokenizer 的 64/128/256 内容 tokens、20% overlap；每档对全部 285 例重新运行问题相关性和五类角色 Cross-Encoder 评分，不复用原候选分数。评价时任一选中分块命中其唯一父证据 ID，Token 成本与重复父证据率仍按分块统计。三档 FRC Evidence F1 为 0.614475/0.632662/0.679077，相对最强基线的配对区间均跨 0。

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
3. 在包含字段、适用性和冲突标注的领域基准上补齐剩余 3 组真实模型消融，且 Full 严格优于 `w/o Role` 和 `w/o Field`；
4. 在带字段与候选级冲突标注的领域真实模型基准上复现字段权重和冲突阈值扫描；当前仓库构造受控诊断只证明参数行为可辨识；
5. 主要数据集上相对最强可复现基线的改善具有一致方向和配对统计支持；
6. 冲突漏报、错误完整声明、不可回答误答等安全指标达到门槛；
7. 原始输入、配置、种子、哈希、逐样本结果与失败用例可审计。
