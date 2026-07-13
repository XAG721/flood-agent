# FRC-Select 公开数据公平评测协议

执行日期：2026-07-14。本协议用于验证 FRC-Select 的工程与理论流水线是否可行，不预设其优于基线，也不把覆盖贪心代理表述为真实 SetR。

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
| 辖区适用性 | HousingQA | 已运行 | 40 个专家复合用例、160 字段、22 辖区、2021 快照 |
| 版本替换 | LawShift | 已运行 | 124 例、31 类专家审阅的假设修订前/后版本 |
| 权威生效/失效边界 | EU Publications Office CELLAR 与 EUR-Lex | 已运行 | 30 对法案、60 例旧法最后有效日/新法首个生效日；权威日期元数据 |

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
$housingDir = ".cache\benchmarks\housing_qa"
New-Item -ItemType Directory -Force $housingDir | Out-Null
curl.exe --fail --location --output "$housingDir\questions.json.zip" "https://huggingface.co/datasets/reglab/housing_qa/resolve/761550cc974fa1d9141ffd39014db89efa2a7230/data/questions.json.zip?download=true"
Expand-Archive -Force "$housingDir\questions.json.zip" $housingDir
D:\anaconda3\Scripts\hf.exe download triangularPeach/LawShift `
  --repo-type dataset --revision 0fce4f3821140bde29081ae0b20500e79aa065d5 `
  --include "*/articles_original.json" --include "*/articles_poisoned.json" `
  --include "*/original.json" --include "*/poisoned.json" `
  --local-dir .cache\benchmarks\lawshift
python scripts/run_frc_wo_reranker_ablation.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --model-name $bgeSnapshot --device cuda
python scripts/run_frc_chunk_length_sensitivity.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --reranker-model-path $rerankerSnapshot --device cuda `
  --chunk-lengths 64,128,256 --overlap-ratio 0.2
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_conflicts_frc_ablation `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_housing_frc_ablation `
  --hf-home D:\RAG_test\.hf_cache `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
python scripts/run_housing_weight_sensitivity.py
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_lawshift_temporal_ablation `
  --hf-home D:\RAG_test\.hf_cache
D:\anaconda3\envs\rag_exp\python.exe -m scripts.prepare_eurlex_temporal_source `
  --refresh-query --refresh-documents
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_eurlex_temporal_ablation `
  --hf-home D:\RAG_test\.hf_cache
python scripts/import_frc_public_reference.py `
  --reference-root D:\RAG_test\frc-select `
  --conflicts-path output\rag_evaluation\conflicts_frc\conflicts_frc_report.json `
  --supplemental-ablation output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --chunk-length-sensitivity output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --controlled-domain-sensitivity output\rag_evaluation\controlled_domain_sensitivity\controlled_domain_sensitivity.json `
  --conflicts-ablation output\rag_evaluation\conflicts_frc_ablation\conflicts_frc_ablation.json `
  --housing-ablation output\rag_evaluation\housing_frc_ablation\housing_frc_ablation.json `
  --housing-weight-sensitivity output\rag_evaluation\housing_weight_sensitivity\housing_weight_sensitivity.json `
  --lawshift-ablation output\rag_evaluation\lawshift_temporal_ablation\lawshift_temporal_ablation.json `
  --eurlex-ablation output\rag_evaluation\eurlex_temporal_ablation\eurlex_temporal_ablation.json `
  --output-dir output\rag_evaluation\public_frc_reference
```

先运行 `python scripts/run_frc_controlled_sensitivity.py` 可重建字段/角色权重与冲突阈值的受控诊断产物。该产物固定为 `SYNTHETIC`、无神经模型、单因素扫描；导入器会逐样本重算聚合值并验证来源，且将其标记为 `RUN_CONTROLLED_DOMAIN_PUBLIC_SCHEMA_BLOCKED`，不会把它计入公开真实模型完成度。随后运行 `python scripts/run_housing_weight_sensitivity.py`，可在 HousingQA 已冻结的 BGE/reranker 真实模型分数上扫描字段权重 0/0.5/1/2/4/8 和角色权重 0/0.5/1/2/4；它只复算选择，不重跑模型或生成，gold 字段映射仅在选择后评分。该结果属于跨领域事后诊断，不用于调参或 Gate 2 晋级；三类角色是确定性 FRC 标签，不是 HousingQA 专家角色标注。

输出：

- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.json`
- `output/rag_evaluation/public_frc_reference/public_frc_reference_report.md`
- `output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.json`
- `output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.md`
- `output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.json`
- `output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.md`
- `output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.json`
- `output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md`
- `output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity_cases.jsonl.gz`
- `output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation.json`
- `output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation.md`
- `output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.json`
- `output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.md`
- `output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation_cases.jsonl.gz`

导入报告同时审计设计第 16.2 节实验覆盖：ConditionalQA 原始产物包含 5 组消融，仓库另完成真实 `w/o Reranker`、CONFLICTS `w/o Conflict`、HousingQA `w/o Field`/辖区 `w/o Applicability`、LawShift 版本替换和 EUR-Lex 生效/失效边界 `w/o Applicability`，跨适用公开集的 9/9 命名消融均有执行工件。HousingQA 40 例、160 字段、22 辖区上，Full 相对 `w/o Field` 字段覆盖提高 0.050000（95% CI [+0.018750, +0.087500]），相对无辖区过滤提高 0.175000；但 Full 与最强字段分解基线字段覆盖同为 0.706250，Answer Accuracy 仅 0.306250。LawShift 124 例、31 类专家审阅假设修订上，Full 相对无版本过滤的精确版本证据提高 0.137097（95% CI [+0.080645, +0.201613]）、错误版本率降低 0.451613，但 Article Recall@1 降低 0.064516，且与公平过滤基线持平。

EUR-Lex/CELLAR 冻结切片从公开 SPARQL 法律元数据选择 30 对“旧法失效日 + 1 天 = 新法生效日”的唯一边界，其中 Decision/Directive/Regulation 为 12/6/12 对；每对分别评估旧法最后有效日和新法首个生效日，并加入两对确定性干扰法案。60 例上 Full/`w/o Applicability` 的精确证据准确率为 1.000000/0.483333，差值 +0.516667，95% CI [+0.383333, +0.650000]；无效适用率从 0.516667 降至 0。但获得相同有效期元数据的 `applicability_filtered_cross_encoder_top1` 也是 1.000000，因此日期过滤有效而 FRC 独有优势未获证明。源清单固定查询、CELEX、日期、官方 URL 及原始 HTML/规范文本 SHA-256，查询不含 gold CELEX，逐例 gzip 由导入器重新聚合。适用性三部分构造现已可识别；总体覆盖仍为 `PARTIAL`，因为这些公开语料属于住房法、假设法律修订和欧盟法律，不是同一真实防汛领域，且双专家评判未完成。

Token 预算敏感性使用同一候选池、Top-K=5 和保存分数，严格限制 512/1024/2048 Token，任何方法超预算即失败；当前所有方法超预算率均为 0。缺失比例敏感性以固定 `SHA-256(case_id, evidence_id)` 分数与 0/25/50/75% 阈值比较，形成可重复、随比例嵌套的删除集合，不使用测试结果调参。CONFLICTS 冲突披露角色阈值 0.25/0.40/0.55/0.70/0.85 的真实模型准确率为 0.336245/0.336245/0.334061/0.334061/0.325328；该扫描是在既有结果已可见后补充的事后诊断，不用于重新选择冻结参数。字段/角色权重与选择器冲突风险阈值先在 3 例、18 文档的仓库构造受控集上运行，用于验证选择器参数行为和金标准/自报覆盖的差异。随后在 HousingQA 40 例、160 字段、22 辖区和 400 个候选出现上，以冻结真实 BGE/reranker 分数完成字段权重 0/0.5/1/2/4/8 与角色权重 0/0.5/1/2/4 单因素扫描：字段权重从 0 提升到冻结值 2 时 Evidence F1 从 0.656250 增至 0.706250，权重 4/8 为 0.712500；角色权重从 0 到冻结值 1 时角色覆盖从 0.600000 增至 0.625000，权重 2/4 为 0.633333。字段权重 0 和角色权重 4 分别改变 29/40 与 13/40 个选择，两个维度均可辨识；但高于冻结参数的最大 Evidence F1 增益仅 0.006250，扫描不改冻结参数。公开技术敏感性矩阵因此已补齐，状态为 `RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC`；由于语料跨域、角色标签非专家标注且没有同一防汛领域双专家评判，不能据此通过 Gate 2。

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
3. 9/9 命名消融虽已运行，仍需在同一真实防汛领域专家基准上复现；当前 Full 未优于 `w/o Role`，HousingQA Full 与最强字段基线持平，LawShift 与 EUR-Lex Full 均与各自公平适用性过滤基线持平，CONFLICTS Full 也未优于 `w/o Conflict`；
4. HousingQA 冻结真实模型分数上的字段/角色权重扫描已完成，但仍需在带任务字段和专家角色标注的真实防汛基准上预注册复现；CONFLICTS 冲突阈值扫描也属于事后诊断，均不能用于追认调参；
5. 主要数据集上相对最强可复现基线的改善具有一致方向和配对统计支持；
6. 冲突漏报、错误完整声明、不可回答误答等安全指标达到门槛；
7. 将已完成的 EUR-Lex/CELLAR 权威日期协议迁移到具有权威版本登记的真实区县防汛文档，并在同一领域验证辖区、版本和有效期联合适用性；
8. 原始输入、配置、种子、哈希、逐样本结果与失败用例可审计。
