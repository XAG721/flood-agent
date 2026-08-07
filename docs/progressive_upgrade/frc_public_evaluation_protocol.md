# FRC-RAG 公开数据公平评测协议

首次执行日期：2026-07-14；最近更新：2026-08-01。本协议用于验证 FRC-RAG 证据选择组件的工程与理论流水线是否可行，不预设其优于基线，也不把覆盖贪心代理表述为真实 SetR。

## 1. 数据集与难例覆盖

| 难例 | 数据来源 | 当前状态 | 用途 |
|---|---|---|---|
| 正常检索 | ConditionalQA、MultiHop-RAG、HotpotQA、2WikiMultiHopQA | 已运行 | 检查常规证据召回与选择 |
| 条件、例外、不可回答 | [ConditionalQA](https://arxiv.org/abs/2110.06884) | 已运行 | 条件答案、长文档、多跳与 supporting evidence |
| 跨文档 | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/) | 已运行 | 2—4 篇文档联合证据 |
| 多跳证据 | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/)、[HotpotQA](https://hotpotqa.github.io/)、[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | 已运行 | 多段 supporting facts、推理类型与干扰文档 |
| 缺失证据 | ConditionalQA 的确定性变换切片 | 已运行 | 对每个至少含两条金证据的用例删除首条金证据，再用保存的真实模型分数重新选择 |
| 冲突证据 | [Google CONFLICTS](https://github.com/google-research-datasets/rag_conflicts) | 已运行 | 458 例官方五类标签；六方法同预算选择后由同一本地 Qwen 分类 |
| 同名实体多视角冲突 | [WhoQA](https://github.com/VinAIResearch/WhoQA) | 已运行 | 5,152 个未触碰问题；六选择器在 Top-K=4、1,500 Token 下比较不同观点覆盖 |
| 噪声、信息整合与反事实证据 | [RGB](https://github.com/chen700564/RGB) | 已运行 | 固定提交的 498 个有效英文案例；九方法在 Top-K=5 与三档 Token 预算下进行无 gold 盲评 |
| 失效/过期文件 | CONFLICTS 的 `Conflict due to outdated information` | 已运行 | 62 例；报告类型 Recall、最新日期来源保留和时间端点覆盖 |
| 辖区适用性 | HousingQA | 已运行 | 40 个专家复合用例、160 字段、22 辖区、2021 快照 |
| 版本替换 | LawShift | 已运行 | 124 例、31 类专家审阅的假设修订前/后版本 |
| 权威生效/失效边界 | EU Publications Office CELLAR 与 EUR-Lex | 已运行 | 30 对法案、60 例旧法最后有效日/新法首个生效日；权威日期元数据 |

Google 对 CONFLICTS 的研究说明给出了检索增强生成中的知识冲突分类与专家标注评测背景，参见 [Dragged into a Conflict](https://research.google/pubs/dragged-into-a-conflict-detecting-and-addressing-conflicting-sources-in-search-augmented-llms/)。本仓库已记录 Apache-2.0 许可、数据 SHA-256、真实模型配置和逐方法结果；但没有复现论文依赖独立人类判断的 expected-behavior adherence，因此不是官方 leaderboard 结果。

## 2. 公平性约束

- 所有方法使用同一候选池、同一问题、同一金证据和同一数据划分。
- 主系列统一 `Top-K=5`、证据预算 `1500 tokens`；WhoQA 独立协议按其最少 2 个候选的结构预先冻结 `Top-K=4`，预算仍为 `1500 tokens`，不得与主系列混写。
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

证据充分性安全头使用 ConditionalQA 冻结真实 BGE/reranker 候选与角色分数，筛选 261 个至少含两条金证据的 case，并以固定 0/25/50/75% 缺失比例形成 1044 个变体。同一 case 的所有变体通过 `SHA-256(case_id)` 固定进入同一 train/calibration/evaluation 分组（110/76/75 个 case），特征只读取候选分数、功能角色、Token 成本和选择结果，不读取 `gold`、`gold_roles` 或 `gold_evidence_ids`。确定性类别平衡 IRLS 仅在 train 拟合；split-conformal 阈值使用 calibration 中每个 case 的最大不完整变体分数，主分析 alpha=0.10 且只允许分数严格超过阈值时声明证据完整。evaluation AUC 为 0.931552；原“角色覆盖达到 0.55 即完整”的启发式在不完整变体上的误放行率为 0.566406、case 家族误放行率为 0.810811，校准后分别为 0.011719（Wilson 95% CI [0.003993, 0.033882]）和 0.040541，配对 McNemar 精确检验 `p=3.587e-43`。代价是放行率仅 0.053333、拒答率 0.946667、完整变体放行召回率 0.295455。该结果证明保守拒答机制可显著降低跨域合成缺失下的误放行风险，但当前实用性不足，且不替代真实防汛双专家完整性标注或独立行为评判；状态固定为 `RUN_PUBLIC_REAL_MODEL_SPLIT_CONFORMAL_ABSTENTION`，Gate 2 不变。

```powershell
python scripts/run_conformal_sufficiency.py `
  --input D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --reference-run-report D:\RAG_test\frc-select\outputs\run_report.md
```

提交的结果同时冻结分数文件和参考运行报告的 SHA-256；参考报告声明评分后端为 real、嵌入模型为 `BAAI/bge-large-en-v1.5`、重排模型为 `BAAI/bge-reranker-large`、分数校准为 `per_case_minmax`、角色相关性混合系数为 0.15、FRC alpha/beta/gamma 为 2.0/1.0/0.0。生成结果不参与本充分性判定。

为检验一次固定分组是否偶然，进一步预注册 10 个不同的 `SHA-256(split_version, case_id)` 分组版本；每组都重新形成约 40% train、30% calibration 和 30% evaluation，且同一 case 的四个缺失变体始终不跨分组。10 组结果不用于挑选最有利分组、alpha 或特征。角色覆盖启发式的 case 家族误放行率均值为 0.847827（范围 [0.810811, 0.909091]）；split-conformal alpha=0.10 的均值为 0.080422（范围 [0.037037, 0.147727]），在 10/10 组中均低于启发式，但只有 7/10 组的实测 case 家族误放行率不高于 0.10。主分析平均拒答率 0.930263、完整召回率 0.289074，说明安全改善可重复，但高拒答同样稳定存在。

直接把自动完成边界放宽到 alpha=0.20 会使 case 家族误放行率均值升到 0.209263，因此不采用。实验只把 alpha=0.10 与 0.20 阈值之间的样本标记为 `REVIEW_PRIORITY`，不绕过人工复核，也不改变 Gate 2。该复核带平均覆盖 0.099776 的变体，额外找回 0.319242 的完整变体；高置信候选与复核优先带合计覆盖 0.608317 的完整变体，但剩余 `INSUFFICIENT` 比例仍为 0.830487。

```powershell
python scripts/run_conformal_robustness.py `
  --input D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --reference-run-report D:\RAG_test\frc-select\outputs\run_report.md
```

重复分组共享同一批 ConditionalQA case，属于相关的描述性稳健性检查，不是独立重复试验或置信区间。完整工件位于 `output/rag_evaluation/conformal_robustness/`。

在重复分组确认高拒答稳定存在后，进一步执行严格嵌套的 train-only 选型实验。每个 outer train 内预注册 12 个候选配置：`linear/sqrt/log1p/quadratic` 四种特征变换分别配合 L2=1/4/16；每个候选在三次独立的内层 50% fit、25% calibration、25% validation case 分组上评估。内层 calibration 只定阈值，内层 validation 比较安全重复次数、case 风险、完整召回、精度和拒答；outer calibration 再定最终阈值，outer evaluation 只运行一次且不参与选型。

10 个 outer evaluation 上，固定线性 L2=4 与嵌套选中模型的完整召回率均值为 0.289074/0.354834，提升 0.065759；拒答率为 0.930263/0.907726，仅下降 0.022537；case 家族误放行率为 0.080422/0.121352，增加 0.040930，且只有 2/10 组风险不增。预注册采用判据要求完整召回和拒答至少各改善 0.05，同时 case 风险增加不超过 0.02；后两项失败，因此 `adopt_nested_selected_experimental_default=false`，固定线性 L2=4 不变。这一负结果表明，在现有 37 个可观察特征上添加简单非线性变换不能安全解决高拒答问题。

```powershell
python scripts/run_conformal_model_selection.py `
  --input D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --reference-run-report D:\RAG_test\frc-select\outputs\run_report.md
```

完整工件位于 `output/rag_evaluation/conformal_model_selection/`；候选网格、每次内层结果、outer 选中配置和采用判据均可重聚合审计。

由于 ConditionalQA 已多次用于诊断和选型，下一步不再在该 evaluation 上追加参数，而把固定线性 L2=4、alpha=0.10 和相同 10 个分组版本原样迁移到此前未参与选型的 HotpotQA 冻结真实模型分数。确认运行前固定三项要求：相对角色覆盖启发式的安全改善必须达到 10/10 组；实测 case 家族风险不高于 alpha 必须至少达到 7/10 组；case 家族误放行率均值必须不高于 0.10。确认阶段不增加特征、不比较正则化、不改变 alpha，也不挑选分组。

HotpotQA 结果中，角色覆盖启发式/case 分组 split-conformal 的 case 家族误放行率均值为 0.894262/0.096653；安全改善在 10/10 组复现，平均风险要求通过。但实测风险不高于 alpha 的分组只有 6/10，未达到预注册的 7/10，因此总体状态为 `PARTIAL_CONFIRMATION`，不能写为完整确认。相对 ConditionalQA，HotpotQA 的 case 风险均值增加 0.016231，拒答率下降 0.045339 至 0.884924，完整召回率下降 0.070271 至 0.218803。该结果支持“平均安全改善跨公开数据复现”，但不支持“逐分组一致性目标已满足”。

在揭示 MultiHop-RAG 结果前，`conformal_cross_dataset_protocol.json` 固定了第二确认集输入哈希、2,556 行规模、同一线性 L2=4 模型、alpha=0.10、缺失比例、10 个分组版本和相同三项通过标准。运行中未进行确认集特征、正则化、alpha 或 split 选择。MultiHop-RAG 的角色覆盖启发式/case 分组 split-conformal 风险均值为 0.822489/0.087890，安全改善达到 10/10，风险不高于 alpha 达到 8/10，三项条件全部通过，单集状态为 `FULL_CONFIRMATION`；拒答率均值为 0.884396，完整召回率均值为 0.415231。

第三确认在评分结果揭示前由 `conformal_third_confirmation_protocol.json` 冻结：2WikiMultiHopQA 官方 dev 原始文件 SHA-256、1000 例 ID 哈希抽样、候选句构造、BGE/reranker revision、角色查询、逐 case min-max、L2=4、alpha=0.10、10 个 split 和同一三项通过标准均不得按结果修改。抽样覆盖 32,100 个候选句、2,460 条 supporting facts 和四种问题类型，真实评分源 SHA-256 为 `7add5a5ec5206a33848de0eecc76a8697dc321e03946fcfefee20df0525a6104`。2Wiki 角色覆盖启发式/校准后 case 风险均值为 0.937874/0.090575，10/10 次降低误放行、7/10 次不高于 alpha，单集为 `FULL_CONFIRMATION`；拒答率 0.927512、完整召回率 0.162305，效用仍不足。

跨数据集系列按预注册规则聚合：只有每个确认集均为 `FULL_CONFIRMATION` 才能得到完整系列确认；任一数据集为 `NOT_CONFIRMED` 则系列不确认；其余为部分确认。因此 HotpotQA（部分）、MultiHop-RAG（完整）和 2WikiMultiHopQA（完整）的联合状态仍是 `PARTIAL_CONFIRMATION`，其中完整 2/3、部分 1/3。该规则防止用两个有利数据集覆盖另一数据集的逐组不稳定性。三个公开确认集共同支持平均安全信号的可迁移性，但仍不证明 FRC 检索优于公平基线、真实防汛领域有效或生产可用。

为降低大文件复现成本，重复分组实现把与 split 无关的证据变体和特征在一次 JSONL 解析中构建，再仅重算 case 分组、训练和校准。`source_preparation_benchmark.json` 在 MultiHop-RAG 全量输入上同时运行旧的 10 次解析和新的单次解析：耗时分别为 65.687275 秒和 7.297463 秒，提升 9.001385 倍；两者的 10 组 repeat 规范哈希均为 `4b754ef11b4400bd579b756db1371fa7c4144ff689812f3a88eae496af9e578c`。计时仅描述本机编排性能，正确性由逐字段相等和相同规范哈希证明。

### 有限样本上界与可观察子群

对每个 calibration split 的不完整 case 最大分数，阈值顺序统计量秩固定为 `min(n, ceil((n+1)(1-alpha)))`；声明要求分数严格高于该阈值。因此在新 case 与 calibration case 可交换时，全局边际错误概率的有限样本名义上界为 `(n+1-rank)/(n+1)`。四数据集 40 次运行的该上界均不高于 0.10：ConditionalQA 为 0.088235—0.100000，HotpotQA 为 0.096886—0.099644，MultiHop-RAG 为 0.098613—0.099853，2WikiMultiHopQA 为 0.097720—0.100000。交换性无法由这批固定公开数据证明，且全局边际保证不自动产生条件子群保证。

`conformal_subgroup_protocol.json` 在运行前固定三个不读取 gold 的维度：数据集提供的 `question_type`（缺失归为 `unspecified`）、`required_role_count`、原始 `candidate_count` 的 `<20/20–39/40–59/60+` 区间。只有某次 evaluation 至少含 30 个不完整 case 家族、且 10 次中至少 7 次满足该支持度的子群进入结论。子群结果不能重选模型、alpha、阈值、特征或 split。

共 22 个子群满足支持规则。ConditionalQA 的三个可评子群平均风险均不高于 0.10，并均达到至少 7/10 次一致性。HotpotQA `comparison` 问题的风险均值为 0.179924、最大值 0.269231，仅 1/10 次不高于 alpha；MultiHop-RAG 的两角色子群风险均值为 0.134452、最大值 0.185065，仅 2/10 次不高于 alpha；2Wiki 候选数 `<20` 子群风险均值为 0.189738、最大值 0.378378，仅 3/10 次不高于 alpha，`bridge_comparison` 风险均值为 0.138887，也仅 3/10 次不高于 alpha。因此审计状态为 `SUBGROUP_INSTABILITY_DETECTED`。这不否定全局 conformal 的交换性条件陈述，但否定“现有证据已经支持所有可观察子群稳定受控”的解释。

### 分层 Mondrian 事后方法开发

子群结果揭示后才提出分层校准，因此 `conformal_mondrian_protocol.json` 将其明确登记为事后方法开发，而非第三次独立确认。冻结规则仍使用每个 outer train 拟合的单一线性 L2=4 充分性头；alpha=0.10 阈值只读取 calibration 中每个 case 的最大不完整变体分数。阈值按 `question_type × required_role_count` 联合组校准，组内少于 30 个 calibration case 时依次回退到 `question_type`、`required_role_count` 和全局阈值；evaluation 不选择模型、阈值、分组、支持度或回退顺序。

三数据集的全局/Mondrian case 风险均值分别为：ConditionalQA 0.080422/0.075427，HotpotQA 0.096653/0.093480，MultiHop-RAG 0.087890/0.074545；最坏可评子群风险平均改善 0.047331，所有 evaluation case 均使用由 calibration 支持的分组阈值。但预注册采用检查并未全部通过：MultiHop-RAG 完整召回从 0.415231 降至 0.348327，下降 0.066904，超过 0.05 上限；ConditionalQA 的最坏子群只有 6/10 次不高于 alpha，HotpotQA `bridge` 只有 4/10、两角色子群只有 6/10，也未满足每个可评子群至少 7/10 的重复一致性要求。因此状态固定为 `DO_NOT_ADOPT`，现有全局方法不替换。

分组阈值的有限样本陈述只适用于“实际使用该阈值的精确校准组”且要求组内 case 可交换；回退到较粗分组不会为支持不足的更细子群创造条件保证。完整阈值表、逐变体决策和采用检查位于 `output/rag_evaluation/conformal_mondrian/`。由于方法是在现有三集结果已可见后设计，即使部分指标改善，也必须在全新未触碰数据或独立防汛专家数据上冻结确认后才可重新评审。

### 候选规模感知多轴 Mondrian 事后方法开发

2WikiMultiHopQA 的候选规模与问题类型子群结果揭示后，`conformal_multi_axis_mondrian_protocol.json` 在新一轮运行前冻结八级回退：`question_type × candidate_count_bucket × required_role_count`、`question_type × candidate_count_bucket`、`candidate_count_bucket × required_role_count`、`question_type × required_role_count`、单独候选区间、单独问题类型、单独角色数、全局。每一级仍要求至少 30 个 calibration case 家族；L2=4、alpha=0.10、10 个 split、风险/召回/覆盖/子群采用门槛均不按结果修改。该实验复用四个已经揭示结果的数据集，明确属于事后方法开发，不是第四次独立确认。

四数据集全局/多轴 case 风险均值分别为：2WikiMultiHopQA 0.090575/0.070206、ConditionalQA 0.080422/0.072862、HotpotQA 0.096653/0.099623、MultiHop-RAG 0.087890/0.074545。平均最坏子群风险改善 0.055714，所有 evaluation case 都使用 calibration 支持的非全局阈值，总体风险增量检查通过。采用仍失败：MultiHop-RAG 完整召回下降 0.066904，超过 0.05；HotpotQA `candidate_count_bucket=40_to_59` 风险均值 0.119606，`question_type=comparison` 为 0.118697；2Wiki `<20` 和 `bridge_comparison` 虽平均风险降至 0.088838/0.086204，但分别只有 4/10、6/10 次不高于 alpha。HotpotQA 的候选、bridge、comparison 与两角色子群也未达到 7/10 一致性。

因此八项冻结采用检查有三项失败，报告状态为 `DO_NOT_ADOPT`，下一步为 `STOP_WITHOUT_MUSIQUE_DOWNLOAD`。协议原先只允许“全部检查通过”后才接触未使用的 MuSiQue；本次失败后未下载、未查看也未使用 MuSiQue 调整层级。完整阈值、逐变体证据与可重算报告位于 `output/rag_evaluation/conformal_multi_axis_mondrian/`。

### Leave-one-dataset-out 充分性头迁移审计

HotpotQA、MultiHop-RAG 和 2Wiki 的既有“跨数据集确认”都在目标数据集 train split 重新拟合 37 维充分性头，只迁移了协议而不是模型参数。为单独检验参数可迁移性，`conformal_cross_dataset_head_transfer_protocol.json` 在结果前冻结四数据集 leave-one-dataset-out：每个目标只使用另外三集的 train 变体拟合线性 L2=4 头；三集总权重相同，各集完整/不完整类各占该集权重一半，标准化使用同一加权来源。目标 train 标签和特征均不参与，目标 calibration 标签仅计算不完整 case 最大分数的 alpha=0.10 阈值，evaluation 不选模型、权重或阈值。

代码测试除检查训练审计外，还会把目标数据集的全部完整性标签和 37 维特征改写为反事实值，并要求迁移模型的均值、尺度、权重和截距逐元素不变。真实运行中，目标拟合/迁移 AUC 均值分别为：2WikiMultiHopQA 0.748080/0.663910、ConditionalQA 0.892413/0.815340、HotpotQA 0.763211/0.706035、MultiHop-RAG 0.921522/0.679523，迁移相对目标拟合的跨数据集平均 AUC 下降 0.115104。迁移 case 风险均值为 0.104410/0.085829/0.102824/0.089497；ConditionalQA 与 MultiHop-RAG 完整召回分别从 0.289074/0.415231 降至 0.086731/0.064046。

全部 40 个目标 calibration 有限样本名义上界仍不高于 alpha，说明阈值计算本身保持冻结规则；但平均风险、相对风险增量、召回、AUC 和跨重复一致性五项性能门槛均失败，只有 MultiHop-RAG 达到至少 7/10 次实测风险不高于 alpha。状态因此为 `DO_NOT_ADOPT`，下一步固定为 `STOP_TRANSFER_METHOD_SELECTION_WITHOUT_NEW_DATA`。这不是零样本实验，因为目标 calibration 使用标签；它也证明当前 37 维头不能在缺少目标 train 拟合时维持效用。完整模型、训练来源审计、逐 case 分数和防篡改重算位于 `output/rag_evaluation/conformal_head_transfer/`。

### Calibration-only 跨域头准入守卫

为把迁移负结果转化为失败关闭机制，`conformal_transfer_admission_protocol.json` 在查看逐重复 calibration 指标前冻结一个只拒绝、不修改模型的准入证书。每个目标/分组必须同时具备至少 30 个不完整 calibration case、30 个完整 calibration 变体、迁移分数 calibration AUC≥0.75、冻结阈值下完整召回≥0.15。证书函数只接收 calibration 汇总，不接收 evaluation 风险、evaluation 召回或目标 train 字段；迁移分数、conformal 阈值和声明结果均不改变。

40 个冻结重复中仅 3 个获得证书：ConditionalQA repeat-01/repeat-08 的 calibration AUC/召回为 0.843733/0.166667、0.842287/0.162162，HotpotQA repeat-05 为 0.766068/0.235897。但三者 evaluation case 风险分别为 0.112500、0.153846、0.132616，全部高于 alpha；evaluation 完整召回虽均高于 0.10，安全且有效证书数仍为 0，证书精度为 0。证书覆盖 3/40 也低于预注册最低 8/40。

因此状态为 `DO_NOT_ADOPT`，下一步为 `REQUIRE_TARGET_FITTED_HEAD_STOP_GUARD_TUNING`。该结果说明在同一 calibration 上计算的 AUC/召回不能作为独立安全验证，继续调证书阈值只会使用已揭示 evaluation 选型。系统明确禁止跨域直接复用当前头，必须目标域拟合；完整 40 条证据与可重算报告位于 `output/rag_evaluation/conformal_transfer_admission/`。

### train-only 上下文充分性头事后方法开发

为检验“子群不稳定是否来自全局头未显式建模问题上下文”，`conformal_contextual_protocol.json` 在运行前固定：保留原 37 个连续特征，追加 `question_type`、`required_role_count` 及二者联合类别的 one-hot；类别词表只统计 outer train case，每类至少 10 个训练 case 才单列，其余进入 `other`。模型继续使用线性 L2=4 类别平衡 IRLS，calibration 只设一个全局 alpha=0.10 阈值，evaluation 不参与词表、特征、模型或阈值选择。

三集基线/上下文头 case 风险均值分别为 ConditionalQA 0.080422/0.091715、HotpotQA 0.096653/0.094070、MultiHop-RAG 0.087890/0.087812；完整召回分别为 0.289074/0.314925、0.218803/0.219413、0.415231/0.424655。三集平均召回增益仅 0.011962，未达到 0.02；ConditionalQA 风险增加 0.011292，超过 0.01 上限；最坏可评子群风险平均变化 -0.006680，说明并未改善。HotpotQA `comparison` 风险均值升至 0.185552，MultiHop-RAG `temporal_query` 为 0.137299，重复一致性仍失败。因此预注册状态为 `DO_NOT_ADOPT`。

该结果说明简单类别 one-hot 只能带来少量召回变化，不能安全消除子群异质性。全局 split-conformal 的边际陈述仍只依赖 train 固定打分函数和 calibration 阈值，不产生条件子群保证。完整模型系数、train-only 词表、逐变体决策与采用检查位于 `output/rag_evaluation/conformal_contextual/`；该实验同样是既有结果可见后的方法开发，不是独立确认。

### 多阶段检索分数稳定性充分性头

`conformal_score_stability_protocol.json` 在读取本轮输出前冻结四个真实模型候选分数文件与四个基线报告哈希。原 37 个特征外追加 BM25、dense、hybrid、cross-encoder 各自的 Top1–Top2 间隔、Top1–Top5 均值间隔、Top5 极差、选择集合与阶段 Top-K 重合率、选择集合平均倒数排名，再加入六组阶段间 Top-K Jaccard 和首位候选共识，共 27 个新特征。排序并列按候选 ID 打破；特征不读取 gold，四数据集分别用目标 train 拟合并标准化 64 维 L2=4 头，calibration-only 设置 alpha=0.10 全局阈值，evaluation 只报告一次。

四数据集均出现效用信号：平均完整召回提高 0.090658，2Wiki/ConditionalQA/HotpotQA/MultiHop-RAG 的 AUC 增益分别为 0.008122/0.015268/0.023048/0.012398。但安全门槛失败：case 风险分别由 0.090575/0.080422/0.096653/0.087890 升至 0.104750/0.110814/0.106772/0.093685，四集仅 6/3/6/6 次实测风险不高于 alpha；ConditionalQA 最坏可评子群风险增加 0.029127，超过 0.02 上限。40 个有限样本名义上界仍不高于 alpha，说明失败来自效用—安全折中而非阈值实现错误。

因此状态为 `DO_NOT_ADOPT`，下一步为 `STOP_SCORE_STABILITY_EXPANSION_ON_REVEALED_EVALUATIONS`。64 维头不替换冻结 37 维目标拟合头，也不进入自动完整声明；若以后研究人工复核排序，必须另行冻结不改变自动边界的协议。53,776 条 evaluation 证据、40 个模型和防篡改重聚合位于 `output/rag_evaluation/conformal_score_stability/`。

```powershell
python scripts/run_conformal_score_stability.py `
  --conditionalqa-input D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --hotpotqa-input D:\RAG_test\frc-select\outputs\role_scores\role_scores_hotpotqa.jsonl `
  --multihoprag-input D:\RAG_test\frc-select\outputs\role_scores\role_scores_multihoprag.jsonl `
  --twowiki-input .cache\benchmarks\frc_public_reference\role_scores_2wikimultihopqa.jsonl
```

### 固定工作量人工复核排序

在查看排序结果前，`conformal_review_ranking_protocol.json` 冻结了来源报告和 53,776 条逐变体证据的哈希、复核池、并列规则、5%/10%/20% 三档工作量及采用门槛。自动完整声明仍完全复制 37 维头在 alpha=0.10 下的结果；已自动声明的变体不得进入复核池。基线按 37 维分数降序，候选按冻结 64 维分数降序，benchmark 完整性标签只用于模拟审核收益，不参与排序。

候选排序在四集上的 AP 均提高，跨集平均增益 0.053574；三档完整变体捕获率平均提高 0.036405/0.049725/0.026372，且没有改变工作量或自动边界。然而 HotpotQA 最差合格子组在 10% 预算下下降 0.055752，超过预注册的最大 0.05 退化。所有采用检查中仅该公平性检查失败，因此按冻结规则得到 `KEEP_BASE_REVIEW_ORDER`，下一步为 `STOP_REVIEW_RANKER_DEVELOPMENT_ON_REVEALED_EVALUATIONS`。该结果既不采用 64 维排序，也不改变 `NO-GO/SHADOW`。

```powershell
python scripts/run_conformal_review_ranking.py
```

完整报告和可防篡改重聚合证据位于 `output/rag_evaluation/conformal_review_ranking/`。它是已揭示公开 QA 记录上的事后复核效用模拟，没有测量真实防汛人员时间、行为或专业判断。

### QASC 补充外部确认与问题类型诊断

`conformal_qasc_confirmation_protocol.json` 在数据下载和神经评分前固定 QASC 官方 validation 提交、parquet SHA-256、926 例全量样本，以及“2 条官方事实 + 从 validation 事实池确定性 BM25 选择 38 条非金事实”的 40 候选受控池。BGE-large-en-v1.5 与 bge-reranker-large revision、五类角色查询、逐 case 归一化、37 维 L2=4 目标拟合头、alpha=0.10 和 10 个分组版本全部复用。评分包装器先删除 answer、gold evidence ID、`gold` 与 `gold_roles`，再调用真实模型；这些字段只在评分完成后回接用于评测。

37,040 个候选的评分源 SHA-256 为 `6419428623c1c4ed1a5935acf1402331c166e9afa9a80f45441e9e7540c759c0`。QASC 角色覆盖启发式/校准后 case 风险均值为 0.873854/0.101786，安全改善为 10/10，但风险不高于 alpha 只有 4/10；拒答率 0.885408、完整召回 0.282873。冻结评估器因此给出 `NOT_CONFIRMED`。登记文字曾将“10/10 改善但任一绝对风险检查失败”写为部分确认，而既有冻结评估器还要求均值风险通过；`conformal_qasc_confirmation` 审计将其记录为 `PARTIAL_STATUS_DEFINITION_MISMATCH`，最终保留更严格结果，不事后改代码、阈值或协议。

总体结果揭示后，`conformal_qasc_subgroup_diagnostic_protocol.json` 仅冻结 `question_type` 描述性诊断，最低支持为每次 30 个不完整 case、至少 7/10 次可评。`qasc_what`/`qasc_other` 平均风险 0.097125/0.095696，但只有 6/10、5/10 次不高于 alpha，状态为 `DESCRIPTIVE_HETEROGENEITY_DETECTED`；其他类型支持不足。该结果不提供条件 conformal 保证，不选择阈值或方法，不运行失败的 64 维复核排序，不纳入前三个确认集的系列状态。

上述受控候选池不能表述为 QASC 原始 17M 句开放语料检索，也不是防汛领域专家验证。QASC 负确认不会触发在已揭示数据上的调参，MuSiQue 仍未下载或查看，Gate 2 保持 `NO-GO/SHADOW`。

### QASC 评分编排性能优化

QASC 结论揭示后只允许优化执行效率，不允许选择模型、特征、阈值或方法。`qasc_scoring_optimization_protocol.json` 在真实性能运行前冻结 32 例确定性样本、1,280 候选、实现文件哈希、单次模型加载、CUDA 同步计时、批大小 8/16/32、逐 case min-max 校准、`1e-6` 绝对容差和至少 1.10 倍采用门槛。全局评分包装器在一次批调用前同样移除 answer、gold evidence ID、`gold` 与 `gold_roles`，评测标签只在评分后回接。

RTX 4060 Ti 实测的逐 case 基线为 52.413108 秒。全局批大小 8 为 50.841583 秒，分数、四类分数排序、五类角色排序及 FRC 选择均等价，但加速仅 1.030910 倍；批大小 16/32 的排序与 FRC 选择仍一致，最大分数漂移分别约为 `2.40e-5`/`2.33e-5`，超过容差。没有候选同时通过等价与性能门槛，故冻结决策为 `KEEP_FROZEN_PER_CASE_SCORING_ENTRYPOINT`。该负结果只说明当前编排优化收益不足，不能支持或否定 FRC-RAG 方法本身，也不能改写 QASC `NOT_CONFIRMED`、Gate 2 或 MuSiQue 未访问边界。

### CONFLICTS expected-behavior 独立盲评包

[CONFLICTS 论文](https://arxiv.org/abs/2506.08500)把 expected-behavior adherence 与简单冲突类型分类区分开来；[Google 官方数据仓库](https://github.com/google-research-datasets/rag_conflicts)提供五类冲突标注。本仓库在既有检索与类型分类结果揭示后单独登记 `conflicts_expected_behavior_protocol.json`，冻结同一本地 `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` revision、确定性生成参数、逐方法同提示/预算、无 gold/方法/分数泄漏输入和逐例 A/B 盲化规则。458 例的两种方法共有 916 条待评回答；19 例选择相同，按协议只生成一次并复用，因此实际唯一生成提示为 897 个。

发布的 `output/rag_evaluation/conflicts_expected_behavior/package.jsonl.gz` 只包含问题、冲突类型、类型对应行为说明、可用官方答案、A/B 回答及各自可见来源，不包含方法名、检索分数、角色分数、选择解释或解盲表。解盲映射只留在 Git 忽略缓存，`manifest.json` 提交其规范 SHA-256 承诺。两名独立评审员需分别判定 expected behavior、事实依据、引用正确性和答案正确性，第三名独立裁决员完成分歧裁决后才允许验证承诺并解盲；空模板、占位身份或同一人员兼任都会被拒绝。

当前状态固定为 `GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION`。这只证明生成与盲评流程已可复现，不产生任何人工 adherence 数值；未完成的独立评审也不能由本地 Qwen 自评替代。全量只读诊断发现 2/916 条无引用、10/916 条越界引用、0 条超过 160 词，最长 157 词；19 例来源完全相同，另有 4 例在来源不同时仍生成相同 A/B 文本。所有样本原样进入评审，不按诊断结果后验重生成。即使后续完成，这仍是跨领域证据，不能替代真实防汛双专家评测、证明 FRC 检索优势、复现 SetR 或单独改变 Gate 2。

为了把“有空白全量模板”推进为可实际分派且可质控的人工流程，又在任何人员决定产生前登记 `conflicts_annotation_operations_protocol.json`。三名不同人员分别承担 `reviewer_1`、`reviewer_2`、`adjudicator`；每人 458 个主任务，并按五类冲突近似比例且每类至少一个地抽取 23 个隐藏复测。主任务按“冲突类型 × 官方答案可用性”稳定分层轮转，三个角色使用不同批次旋转；每人共 8 批、481 项，复测与原任务固定相隔 4 批。真实作业 ID 为 `CONFLICTS-ANNOTATION-OPS-5A6EE0F6420F2251`。

24 个公开批次与 24 个空提交模板位于 `output/rag_evaluation/conflicts_annotation_operations/`。它们不发布原条目 ID、主/复测标记、配对任务、方法身份、检索/角色分数或私有路由；本地路由的规范 SHA-256 承诺为 `b4818883beb4ed3e256895a2161f2641656707639ded4ee858633e5e4bb25ef3`。合并器要求同一角色 8 个批次齐全、使用一致且非占位的私有身份，并保留冻结任务顺序；随后恢复 expected-behavior 核心工作流要求的原 458 例顺序。四项评分与偏好分别计算隐藏复测精确一致率和描述性 Cohen's kappa；无边际方差时 kappa 明确记为 null，而不是伪造 1.0。任一维度精确一致率低于 0.80 即 `REVIEW_REQUIRED`，不得进入跨人员比较。隐藏复测只衡量同一人的稳定性，不产生 gold。

```powershell
python scripts/run_conflicts_annotation_operations.py prepare

python scripts/run_conflicts_annotation_operations.py merge `
  --protocol docs\progressive_upgrade\conflicts_annotation_operations_protocol.json `
  --operations-manifest output\rag_evaluation\conflicts_annotation_operations\manifest.json `
  --package output\rag_evaluation\conflicts_expected_behavior\package.jsonl.gz `
  --routing .cache\benchmarks\rag_conflicts\expected_behavior_workflow\annotation_operations_routing.json `
  --reviewer-slot reviewer_1 `
  --submissions <该角色的8份已完成表单> `
  --output <私有合并提交.json> `
  --qc-output <私有组内质控.json>
```

当前 operations 清单仍为 `PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW`、`human_evidence_complete=false`。不得用模型或同一人员填写三个角色，也不得把准备好的批次、空模板、路由承诺或通过代码测试表述成人工评测完成。

### CONFLICTS 本机独立盲评工作台

`conflicts_annotation_workstation_contract.json` 在不修改 expected-behavior 或分批作业协议的前提下，冻结了供真实人员逐批执行的本机 UI。工作台启动时验证 operations 清单状态、作业/模板哈希、批次身份和任务顺序，并再次拒绝公开载荷中的原条目 ID、复测路由、方法名和分数。服务只绑定 `127.0.0.1`/`localhost`，页面会话使用短时随机启动令牌、当前标签页 `sessionStorage`、`HttpOnly; SameSite=Strict` Cookie 与独立 API 请求头；Host/Origin、CSP、`no-store`、`no-referrer` 和禁止嵌入策略均失败关闭。草稿、导出件和回执只允许位于 Git 忽略 `.cache`，采用落盘同步和原子替换。

每个角色由不同人员在自己的隔离环境依次运行 1—8 批；同一角色八批填写同一个私有标识，不同角色必须使用不同标识：

```powershell
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot reviewer_1 --batch-index 1 --open-browser
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot reviewer_2 --batch-index 1 --open-browser
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot adjudicator --batch-index 1 --open-browser
```

完成当前批后关闭服务，将 `--batch-index` 依次改为 2—8。工作台允许随时恢复、鉴权导出和在完整性校验后锁定本批；需要修正时必须显式“重新打开本批”，任何修改都会使旧回执失效。不要把私有草稿提交到 Git，不要向其他角色共享草稿，不要查看 `.cache` 路由或解盲映射。

自动化专项测试覆盖草稿路径边界、续作、占位身份/未完成批次拒绝、完成/重开回执、公开文件篡改与路由泄漏、Host/Origin/令牌、刷新 Cookie 和无外部资源 UI。另在真实 `reviewer_1` 第 1 批 62 项上执行了明确标注为非人工的私有浏览器探针：1 项自动保存后，服务重启和无查询令牌刷新均恢复身份、7 个选项、2 条理由及 `1/62` 进度；390 px 视口无横向溢出且无新控制台错误。探针已删除，既不进入提交也不计作人工证据。当前合同状态是 `LOCAL_REVIEW_UI_READY_AWAITING_HUMANS`，Gate 2 仍为 `NO-GO/SHADOW`。

### CONFLICTS 三角色私有收集与合并前质控

v31 的 `conflicts_annotation_collection_contract.json` 补齐单批工作台与既有合并器之间的操作缺口。只读状态命令自动检查 24 个预期草稿/回执，验证回执 Schema、operation/package/role/batch、规范提交 SHA-256、带时区完成时间、批次完整度和不改 Gate 边界；同一角色八批身份必须一致，三角色身份必须互异。报告只保留身份 SHA-256 前 16 位，不输出原身份，也不读取 blind mapping：

```powershell
python scripts/run_conflicts_annotation_collection.py status
```

当前真实输出为 `AWAITING_INDEPENDENT_HUMAN_BATCHES`、0/24。该命令默认不写任何文件；如需保存私有状态，只能显式指定 `.cache` 内路径。

只有状态达到 `READY_FOR_PRIVATE_MERGE` 后才允许运行：

```powershell
python scripts/run_conflicts_annotation_collection.py merge
```

合并器核对 package 原始文件哈希和 `.cache` 私有 routing 的规范 SHA-256 承诺，先在内存中完成三个角色全部合并与验证，再使旧私有清单失效、写入每角色合并提交和组内质控，并最后原子发布新清单。任一角色的 expected behavior、事实依据、引用正确性、答案正确性或偏好隐藏复测精确一致率低于 0.80 时，状态为 `REVIEW_REQUIRED`，陈旧双评审比较被删除；全部通过时才生成两名评审员的私有比较，状态为 `READY_FOR_BLIND_MAPPING_VERIFICATION_AND_FINALIZATION`。即使如此，blind mapping 仍保持关闭，第三方最终裁决和公开报告尚未完成。

5 项合成私有提交测试验证了等待/就绪状态、角色身份碰撞、回执篡改、成功比较、0.80 失败关闭、路由篡改和旧状态失效。合成身份与决定只存在测试临时目录，既不是人工评测，也不会进入正式 `.cache` 或公开输出。真实三人提交开始后，仓库审计仍不读取或发布私有决定；只有完成既有最终裁决协议后才能讨论 adherence 数值和 Gate 2 复审。

### CONFLICTS 跨折选择器路由回顾性发现

在六种方法的 CONFLICTS 汇总结果已经可见后，v32 透明登记 `conflicts_selector_router_protocol.json`，因此明确属于回顾性发现而非预注册确认。输入固定为 458 个 scored case、2748 个选择结果和 2748 个共享 Qwen 分类结果；按 `sha256(frc-router-v1|case_id) mod 5` 做 case 级五折隔离。运行时白名单包含方法/预测标签 one-hot、标签投票、选择成本/域/词汇统计、四阶段检索分数统计、四角色分数及同一变体方法集内的选择 Jaccard；gold、正确答案、冲突类型和正确性指标禁止作为特征。严格 no-FRC 消融既不读取 FRC 预测，也不读取 FRC 选择重叠。

全方法跨折路由为 181/458、Accuracy 0.395197；冻结最强静态覆盖贪心代理为 158/458、0.344978，差值 +0.050218，95% 配对 bootstrap 区间 [+0.008734,+0.091703]。但 no-FRC 路由已达 180/458、0.393013，全方法相对它仅 +0.002183，区间 [-0.006550,+0.010917]；六方法 oracle 250/458、去 FRC oracle 248/458，FRC 只增加 2 个独有正确例。更重要的是，过时信息冲突 Recall 从静态基线 0.693548 降到 0.258065（-0.435483），只有 3/5 折不劣于静态基线，错误信息冲突 Recall 仍为 0。路由还要求每例取得全部六方法预测，平均 5.165939 个不同选择签名，推理成本显著高于单法。

因此状态固定为 `DISCOVERY_ROUTING_SIGNAL_NOT_ADOPTED_FRC_CONTRIBUTION_NOT_ESTABLISHED`。该结果只说明现有选择器存在可学习互补上界，同时直接否定“这 5.02 个百分点已经证明 FRC 贡献或安全可采用”的解释。路由不启用、FRC 不替换、不得继续在 CONFLICTS 上调参；任何后续采用都需要未触碰数据的独立确认和安全类别非退化检查，Gate 2 保持 `NO-GO/SHADOW`。

```powershell
python scripts/run_conflicts_selector_router.py
```

### WhoQA 未触碰数据独立选择审计

为响应“需要未触碰数据独立确认”的边界，v33 在下载 WhoQA 前登记 `whoqa_conflict_coverage_protocol.json`，冻结官方提交 `02c2b24004fc334c4bf6c9391285eb55e652a86a`、BSD-3-Clause 许可、5,152 个预期问题、六种选择器、Top-K=4、1,500 Token、真实 BGE-large/reranker revision、归一化不同观点覆盖主指标及 10,000 次逐问题 bootstrap。评分输入显式移除答案、官方观点数和主实体；gold 只在选择全部完成后用于评分。下载后仅对公开结构差异、缺失 context ID 和批处理运行方式形成连续 erratum，最终运行参数在评分前冻结为 CUDA/FP16、32 个问题一批、embedding 64、reranker 256，禁止结果可见后继续调优。

原始 5,152 个问题包含 26,957 个问题模板和 17,366 个候选上下文；全部模板评分后共执行 161,742 次选择。初始结果已经为负：FRC 0.995000、BM25 0.995282，差值 -0.000282，同时区间跨 0。结果揭示后发现官方答案别名组需要按传递闭包还原规范 viewpoint，遂单独登记 post-result correction；它不修改选择、模型分数、主指标、bootstrap 或门槛，并保留初始结果。修正后官方观点数量不一致为 0，BM25/FRC 主指标为 0.995203/0.994919，差值 -0.000283，同时区间 [-0.001100,+0.000039]；FRC 相对预注册覆盖贪心代理差值 -0.000110，区间 [-0.000524,+0.000298]。2/3/4/5—8/9+ 观点层的最差差值为 -0.002485，属性层最差为 -0.002156，均未触及 -0.05 安全退化线。

该数据的候选上下文本身高度相关，六方法主指标均接近 1.0，因而区分度有限；它也不覆盖时间失效、错误信息、缺失文件、政策例外或真实防汛任务字段。状态固定为 `WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED`：本实验没有确认 v32 所需的 FRC 优势，但也不能外推为对 FRC 整体理论的否定。5,152 条 gzip 证据只保存 ID、聚合评分和选择结果，不导出原问题、上下文或答案；报告 2/2 次确定性复跑字节一致。

```powershell
python scripts/run_whoqa_conflict_coverage.py --evaluate-only
```

### WhoQA 冻结容量与 Token 压力诊断

v33 指标接近饱和后，v34 在任何压力结果可见前登记 `whoqa_budget_stress_protocol.json`。实验完全复用 v33 原始文件和盲评分缓存，不重新调用模型，不修改六种选择器及 FRC 权重；冻结配置为 `K=2/3, Token=1500` 以及 `K=4, Token=256/512/1024/1500`。主指标改为 `选中不同规范观点数 / min(规范观点总数, K)`：分母不随 Token 预算实际可容纳条数降低，因此预算放不下候选时按真实短缺计分。每个问题先在模板内均值，再对六配置等权形成 family 分数；10,000 次逐问题 bootstrap 在每次重采样内部重新选择最强基线。

5,152 个问题、26,957 个模板共形成 970,452 次选择。六配置 family 主指标为 Dense 0.890719、FRC 0.887793，FRC 相对最强基线差值 -0.002926，同时区间 [-0.003858,-0.002025]；相对覆盖贪心代理差值 -0.000329，区间 [-0.000685,+0.000023]。逐配置差值为：`K2/1500` -0.000633、`K3/1500` -0.000160、`K4/256` -0.016657、`K4/512` -0.004657、`K4/1024` -0.000341、`K4/1500` -0.000283。最差配置仍未低于预注册 -0.02 安全线，但 family 点增益和两项区间支持均失败。

分层结果进一步表明，FRC 在 2/3 观点层相对 family 最强 Dense 为 +0.000217/+0.002113，而在 4、5—8、9+ 观点层为 -0.006325/-0.012057/-0.007660；观点数超过 v33 `K=4` 的 1,119 例差值 -0.011928，任一 K4 Token 配置受约束的 4,604 例差值 -0.002625。这说明现有 WhoQA 角色增益并未在更高观点数或紧 Token 预算下转化为优于 Dense 的成本—多样性权衡。状态固定为 `WHOQA_STRESS_SUPPORT_NOT_ESTABLISHED`；选择器保持不变，禁止在已揭示 WhoQA 上调权重，后续成本感知方法必须形成新版本并使用不重叠评估证据。

```powershell
python scripts/run_whoqa_budget_stress.py
```

### RGB 成本感知 FRC 独立盲评

v35 在读取 RGB 数据内容前登记 `rgb_cost_aware_frc_protocol.json`，固定官方提交 `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615`、CC BY-NC-SA 4.0 非商用边界、`en_refine/en_int/en_fact` 三文件、BGE-large/reranker revision、四个通用角色、九种选择器、Top-K=5、512/1024/1500 Token 和 Evidence F1。新方法的集合目标固定为 `2 × 四角色最大分数均值 + 选中 cross-encoder 分数和`，每步按边际目标增益/Token 选择，并与最佳可行 singleton 比较；公平基线额外包含 cross-encoder 密度贪心与精确背包。所有阈值、公式、提示和 10,000 次按数据集分层的配对 bootstrap 均在神经分数生成前锁定。

下载后的执行登记只做结构普查：500 行均可解析，`en_fact` 有 2 行因同一规范化文本同时属于正确证据和错误证据而排除，不替换；498 个有效案例形成 14,580 个分块。为防止原文件的正/负列表顺序通过候选 ID 影响并列，去重源文档按文本 SHA-256 排序后才分配中性序号。准备缓存和评分缓存都不含答案、标签或 gold；评分后才用候选 ID 回连标签。13,446 次选择的成本感知 FRC、旧 FRC 和最强覆盖贪心代理 Evidence F1 分别为 0.507421、0.551199、0.556141。新方法相对旧 FRC 差值 -0.043778，95% CI [-0.061536,-0.026205]；相对每次重采样内最强基线差值 -0.048719，同时区间 [-0.066307,-0.032655]。最差 `en_refine/1500` 差值 -0.127688，越过 -0.02 安全线。

诊断显示新方法在多数配置已占满五个槽位，却因密度目标偏向很短的候选而只使用少量 Token 预算；当 Top-K 与 Token 同时受限时，单纯 `marginal/token` 会让稀缺槽位过早分配给低绝对效用片段。`en_fact` 错误正例选择率相对旧 FRC 下降 0.022449，但主指标与多个数据集/预算安全检查仍失败。初始报告曾把只存在于 `en_fact` 的该次指标与另外两个结构零数据集等权；结果揭示后先保存初始哈希，再登记 `rgb_cost_aware_frc_post_result_correction.json`，仅把统计总体收窄到 `en_fact`，不重评分、不改主指标、公式、阈值或状态。最终结论为 `RGB_COST_AWARE_FRC_SAFETY_REGRESSION`：选择器不替换，禁止在 RGB 上调公式，Gate 2 保持 `NO-GO/SHADOW`。

```powershell
python scripts/run_rgb_cost_aware_frc.py --evaluate-only
```

### MuSiQue 双资源独立确认（v36）

v36 不在已揭示 RGB 上修补参数，而是在读取 MuSiQue 内容前登记 `musique_dual_resource_protocol.json`。新方法保留 v35 的集合目标，把贪心分母冻结为 `token_count / token_budget + 1 / top_k`，以无拟合权重的归一化资源份额同时约束 Token 与槽位；其余模型 revision、角色查询、Top-K=5、512/1024/1500 Token、十种方法和 10,000 次 source-row bootstrap 均预先固定。官方 MuSiQue 提交 `922ac98f19a201998dbdae6d7f2887a5258dbdeb` 的 Answerable dev 全量 2,417 例均结构有效，形成 48,656 个分块；2/3/4-hop 分布为 1,252/760/405。准备和评分缓存不包含答案、支持标签、分解或 gold，逐例发布证据不包含问题、答案或候选原文。

结果显示双资源公式消除了 v35 的一部分短片段偏置：support evidence F1 从 0.524143 提高到 0.528545，差值 +0.004402，95% CI [+0.002244,+0.006613]；但仍低于旧 FRC 0.530801 和最强非 FRC cross-encoder top-k 0.530399，相对后者 -0.001854，同时区间 [-0.003784,-0.000020]。三档预算最差 -0.004570，三个 hop 层最差 -0.004806，均未达到安全回归阈值；然而预注册要求的至少 0.01 优势和正区间均失败，故状态为 `MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_NOT_ESTABLISHED`。该结果发布但不采用；禁止在 MuSiQue 上回调公式，Gate 2 仍为 `NO-GO/SHADOW`。

```powershell
python scripts/run_musique_dual_resource.py --evaluate-only
```

### MuSiQue 结果后机制诊断（v37—v38）

v37 在精确选择结果未知时冻结 `musique_objective_gap_diagnostic_protocol.json`，随后穷举所有满足 Top-K=5 和三档 Token 预算的子集，精确最大化 v36 集合目标。平均归一化 regret 仅 0.000809，84.94% 配置已在 `1e-9` 内最优；精确解相对 v36 Evidence F1 为 +0.001399，95% CI [-0.000208,+0.003008]，相对 cross-encoder top-k 为 -0.000455。精确优化没有建立改善信号，状态为 `MIXED_OPTIMIZATION_AND_ALIGNMENT_DIAGNOSTIC`。

v38 再在任何交叉拟合结果未知时冻结 `musique_crossfit_support_protocol.json`：五折按 case ID 哈希隔离，模型固定为 L2=4 逻辑回归，每个训练 case 总权重为 1 且正负类各占 0.5；对应 held-out 标签不进入拟合。基础模型使用六项通用检索/成本特征，完整模型固定增加六项角色统计。完整模型相对 cross-encoder top-k +0.007383，区间为正但未达 +0.01；相对基础模型仅 +0.000548，区间跨零。故 `CROSSFIT_SUPPORT_SIGNAL_NOT_ESTABLISHED`，不能声明 FRC 角色增量。v37/v38 都是 MuSiQue 结果已揭示后的发现研究，不具采用、独立确认或 Gate 证据资格。

### HoVer 任务特定核验角色独立盲评（v39）

v39 不在已揭示 MuSiQue 上继续调模型族，而是在读取 HoVer dev、TF-IDF 候选和 Wikipedia 数据库内容前冻结 `hover_verification_roles_protocol.json`。协议固定官方提交、top-20 候选、384/64 分块、Top-K=5、512/1024/1500 Token、五类非 FRC/通用角色 FRC/任务特定角色 FRC 七种方法、角色分数不混入直接相关性、10,000 次 case bootstrap 和 +0.01 实质增益门槛。执行登记又在神经评分前冻结三份源哈希、SQLite 模式、42,260 个候选标题全解析、4,000 例结构普查和 58.37 MB 盲准备文件指纹。

全量实验包含 80,000 个候选文档、81,224 个分块和 84,000 次选择。官方 top-20 对 gold 支持文档的平均候选上限为 0.608500，完整覆盖只有 1,060 例；协议不补入缺失 gold，因而结果仅表示固定候选池内的选择能力。任务特定角色/通用角色/cross-encoder top-k 的 document evidence F1 为 0.433286/0.433321/0.433019；任务特定角色相对通用角色 -0.000035，95% CI [-0.000187,+0.000092]，相对最强非 FRC 仅 +0.000267。预算和全部预注册分层均未触发 -0.02 安全线，但角色增量和 +0.01 优势失败，故状态为 `HOVER_VERIFICATION_ROLE_SUPPORT_NOT_ESTABLISHED`。该结果发布但不采用，不在 HoVer 上回调任何参数，不复现 SetR、不证明洪水领域效果、不改变 Gate 2。

```powershell
python scripts/run_hover_verification_roles.py --evaluate-only
```

### HoVer 静态角色机制诊断与动态原子角色恢复确认（v39—v41）

v39 结果揭示后只使用已保存分数和选择结果执行无 gold 机制审计，不改方法也不作因果声称。通用/任务特定核验角色内部平均 Spearman 为 0.947723/0.946191，匹配跨族为 0.951934；四个角色共享同一首选候选的比例为 0.8580/0.8425。512/1024/1500 Token 下，通用与核验角色有序选择完全相同率为 0.96325/0.96150/0.96125，核验角色与直接 cross-encoder 选择集合 Jaccard 为 0.995303/0.998179/0.998250。因此状态固定为 `STATIC_ROLE_SIGNAL_COLLAPSE_OBSERVED`：它支持前瞻测试 claim-conditioned 原子查询，不证明静态提示是 v39 负结果的唯一原因。

初始 `hover_dynamic_atomic_roles_protocol.json` 在读取 HoVer train 内容前冻结 v40 动态原子角色算法；官方 train、TF-IDF 结果和 Wikipedia 数据库下载后，一次只读普查错误地打印了 v40 机制/确认分区的标签与 hop 聚合。虽然没有输出单条 claim、候选、supporting fact，也没有生成查询、神经分数、选择或指标，但这已经违反 gold-free 先导边界。`hover_dynamic_atomic_roles_v40_closure.json` 因而在任何评分前把 v40 固定为 `REGISTRATION_BOUNDARY_VIOLATED_BEFORE_SCORING`，禁止其支持任何机制或性能结论，并永久排除全部 2,512 个 v40 ID。

`hover_dynamic_atomic_roles_protocol_v41.json` 在进一步访问未暴露行内容前登记恢复实验：算法、Qwen revision、四个 JSON 查询键、回退模板、BGE/reranker revision、逐例 min-max、秩效用、0.5 角色边际权重、Top-K=5、三档预算和统计门槛均逐字节继承，不按内容平衡或替换样本。剩余 15,659 个 ID 经新盐排序，前 512 个只运行无 gold 机制门槛，后 2,000 个在先导全过且两阶段执行登记完成后才允许准备、生成和评分。先导只接触 claim 与盲候选：回退率 0.003906；静态/动态角色内部平均 Spearman 0.945471/0.521314，平均不同角色首选数 1.154297/2.230469，静态/动态有序选择相同率 0.432292，五项机制检查全部通过，状态为 `MECHANISM_ESTABLISHED_OPEN_CONFIRMATION`。

确认集包含 2,000 例、40,000 个候选文档和 40,616 个分块。claim-only 动态查询有 29 条使用冻结回退模板，回退率 0.0145；生成器与评分器均不读取 label、hop、supporting fact 或 candidate gold，完整 2,000 行分数缓存形成后才一次性连接 gold。动态秩覆盖、静态秩覆盖和最强非 FRC `cross_encoder_topk` 的 document evidence F1 为 0.428851/0.427258/0.427388。动态相对静态为 +0.001593，95% CI [+0.000453,+0.002762]；相对 bootstrap 内选择的最强非 FRC 为 +0.001463，95% CI [+0.000308,+0.002688]。两者区间均为正，且所有预算、SUPPORTED/NOT_SUPPORTED、2/3/4-hop 和候选上限完整性分层均高于 -0.02 安全线；但点增益未达到预注册 +0.005/+0.010 实质门槛。因此 `DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED`：只确认动态查询能打破静态信号坍缩并产生很小的同家族正增益，不采用选择器，不在已揭示 HoVer 分区上继续调参，不复现 SetR、不证明防汛领域有效性、不改变 Gate 2。

```powershell
python scripts/run_hover_role_mechanism_diagnostic.py
python scripts/run_hover_dynamic_atomic_roles.py evaluate
```

协议、关闭链、两阶段开放登记、执行登记、结果哈希和 2/2 字节一致复跑分别固化在 `hover_dynamic_atomic_roles_v40_closure.json`、`hover_dynamic_atomic_roles_protocol_v41.json`、`hover_dynamic_atomic_roles_confirmation_open_v41_v2.json`、`hover_dynamic_atomic_roles_execution_v41.json` 与 `hover_dynamic_atomic_roles_result_v41.json`；逐例 gzip 不包含 claim、标题、文章、候选原文或 supporting facts。

```powershell
python scripts/run_musique_objective_gap.py
python scripts/run_musique_crossfit_support.py
```

```powershell
python scripts/run_conformal_cross_dataset.py `
  --reference-robustness output\rag_evaluation\conformal_robustness\conformal_robustness.json `
  --confirmation-input D:\RAG_test\frc-select\outputs\role_scores\role_scores_hotpotqa.jsonl `
  --reference-run-report D:\RAG_test\frc-select\outputs\run_report.md
```

完整工件位于 `output/rag_evaluation/conformal_cross_dataset/`。该实验是协议在另一个公开数据集上的重新拟合与校准，不是模型权重零样本迁移，也不替代真实防汛领域评判。

分块长度采用 reranker tokenizer 的 64/128/256 内容 tokens、20% overlap；每档对全部 285 例重新运行问题相关性和五类角色 Cross-Encoder 评分，不复用原候选分数。评价时任一选中分块命中其唯一父证据 ID，Token 成本与重复父证据率仍按分块统计。三档 FRC Evidence F1 为 0.614475/0.632662/0.679077，相对最强基线的配对区间均跨 0。

CONFLICTS 全量评测在已缓存模型的 `rag_exp` 环境执行：

```powershell
conda run -n rag_exp python -m scripts.run_conflicts_frc_evaluation `
  --hf-home D:\RAG_test\.hf_cache `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
```

官方数据共 458 例，其中 237 例含非空正确答案、62 例为过时信息冲突、5 例为错误信息冲突。覆盖贪心代理 Accuracy 为 0.344978，FRC 为 0.334061；FRC 相对最强基线差值为 -0.010917，配对 95% CI 为 [-0.043668, +0.024017]。FRC 的过时信息类型 Recall 为 0.564516，覆盖贪心代理为 0.693548。完整结果见 `../../output/rag_evaluation/conflicts_frc/conflicts_frc_report.md`。

### Evidence Inference 2.0 低核心—次名分歧前瞻验证（v49）

v49 在下载、列举或打开数据归档前，仅用理论反例与合成夹具冻结 `low_core_divergence_guarded_frc_v49`：令 `A` 为四个动态角色各自 rank-1 候选的并集，`P` 为各角色 rank-1/rank-2 候选的并集；仅当 `|A| <= 2` 且 `|P - A| >= 3` 时，把 v43 的目标基数增加 1，最多选择 5 个单元，其余排序、秩覆盖权重、预算和并列规则沿用冻结 v41。该公式最多比 v43 增加一个证据单元，且不能读取文本位置、来源类型、标签、gold 或数据集统计量。协议明确禁止读取或使用 v48 逐例产物；已知的仅是 v48 聚合负状态。

最初计划的 ERASER v1.2 数据 URL 已返回 HTTP 404，因此在任何数据成员打开前，按公开来源优先级改用原作者 Evidence Inference 2.0 归档 `https://evidence-inference.ebm-nlp.com/v2.0.tar.gz`，固定归档 SHA-256 `6abe0d4ec0d331834981c0171c3c79d47515761867f82f1dc6066e43863a1586`、原作者仓库 revision `a661e8c14f973398380c8865cf2f27a535aaaf6d` 与 BigBio 加载器 revision `35dce6aba1b3eb9eb9af9bdc38ebeda73dad15b9`。train/test ID 文件和文章正文均未打开。validation 共 443 篇文章、1,258 条提示；排除 5 条官方 caveat 提示与 52 条缺少完整标注员参考的提示后有 1,201 条合格记录，按冻结 SHA-256 顺序选择 600 条。不同标注员作为替代参考，不合并成更宽松的联合 gold；同一标注员的多段证据才取并集。

本地 Qwen2.5-7B-Instruct-GPTQ-Int4、BGE-large-en-v1.5 与 bge-reranker-large 在 RTX 4060 Ti 上生成 600 条查询并评分完整文章句级候选，gold 只在完整评分缓存后联结。查询进程已写完 600 条并输出完成摘要，但外层命令编排在 4845.3 秒返回 124；缓存随后独立通过行数、顺序、schema、无 gold 字段、回退率与 SHA-256 校验，未重生成，也未更改方法或参数。候选/最强冻结 FRC v43/最强非 FRC 的 evidence macro F1 为 0.220684/0.223942/0.186549；候选相对非 FRC 为 +0.034135（95% CI [+0.021521,+0.046820]），三个预算与全部受支持分层安全检查通过，但相对最强 FRC 为 -0.003258（95% CI [-0.008086,+0.001747]），召回增益 0.008710 未达到 0.01。冻结结论为 `EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v49 案例调参，Gate 2 保持 `NO-GO/SHADOW`。这不是 ERASER 官方结果、Evidence Inference 端到端推断榜单、开放语料检索或洪水领域结论。

### ContractNLI 原生零共识拒答验证（v50）

v50 在官方归档下载前冻结参数无关的模型原生零边界：对同一官方跨度计算 anchor、support、contradiction 和 exception 四类确定性查询的原始 reranker logit；仅当 `anchor > 0` 且 `support > 0 or contradiction > 0` 时打开共识门，门关闭返回空集合，门打开原样使用 v43 自适应目标与 v41 动态秩覆盖选择器。共享门非 FRC 控制使用完全相同的共识门，anchor 控制只要求 `anchor > 0`。不从 ContractNLI 拟合分位数、温度、边距或先验；train/dev 永久排除，Span NLI BERT 只作为论文参考，不模拟未公开兼容权重。

官方 test 共 123 份合同、17 个固定假设和 2,091 条记录。冻结采样按 Entailment、Contradiction、NotMentioned 各 200 条，并限制每合同最多 6 条；文档 commitment 作为 bootstrap 聚类单元。首次结构准备在任何盲缓存、查询或神经评分前触发保护性错误：10,061 个官方跨度中有 138 个纯空白跨度，分布于 27 份合同且从未被任何官方 evidence 引用。schema 勘误预先固定“只跳过精确半开切片为空白的跨度，保留所有非空跨度的原始 `span_NNNN` 编号；若任何证据索引因此缺失则在查询前失败”。该修正不改变查询、模型、零阈值、选择器、预算、指标或支持门槛，但由于它来自 test 结构统计，v50 只能表述为透明 schema-repair confirmation，不能表述为完全未触碰数据确认。

更正后 600 条盲缓存的非空完整跨度池对 400 条 evidence case 的候选上限为 1.0，200 条 NotMentioned 的官方 evidence 均为空；600 条确定性查询无回退，约 4.9 万个候选跨度的完整 GPU 分数在 gold 联结前锁定。正式评测采用“有证据 case 的集合 F1；NotMentioned 仅空集合得 1”并对 123 个文档 commitment 做 10,000 次聚类 bootstrap。结果中 anchor 门与共识门均在 600/600 条上通过，候选拒答率和 NotMentioned 拒答准确率均为 0。候选 utility/evidence-bearing F1 为 0.165173/0.247760；相对最强共享门非 FRC 为 +0.004430（95% CI [-0.015105,+0.023482]），相对最强冻结 FRC v49 为 -0.001423（95% CI [-0.008765,+0.005032]），相对 anchor 门为 0；最差支持分层是 Contradiction，差值 -0.056798。冻结结论为 `CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED`：原生零边界不具备缺失证据判别力，不采用、不在 v50 上拟合替代阈值或选择新方法，Gate 2 保持 `NO-GO/SHADOW`。这不是 ContractNLI 官方 NLI 榜单、开放语料检索、真实 SetR 复现或洪水领域结论。

### ContractNLI 开发集稳健共识阈值验证（v51）

v51 永久排除已被 v50 使用的 test，并在打开 dev/train 内容前冻结两阶段协议：dev 只用于开发，train 只有在全部 OOF 开启门通过且最终阈值哈希锁定后才允许作为一次性确认集打开。稳健分数对 anchor、support、contradiction 三个原始 reranker logit 分别按当前 case 全跨度候选的中位数和 IQR 标准化；每跨度取 `min(z_anchor, max(z_support, z_contradiction))`，case 取最大值。门通过后原样使用 v49 选择器。阈值候选、evidence F1 损失上限 0.03、弃答率上限 0.8、五折文档隔离 OOF 和全部 train 开启门均在 dev 打开前冻结。

官方 dev 有 1,037 条合格记录；按 Entailment、Contradiction、NotMentioned 各 80 条、每合同最多 6 条形成 240 条/60 份合同，5 折均非空。完整跨度池对证据类候选上限为 1.0，NotMentioned 官方 evidence 全为空；240 条固定查询回退率为 0，完整 GPU 评分缓存 SHA-256 为 `e802cf128200918d5cf3a477d3bdcc0ad9e3c748ecd67a4783bc1925446edbc6`。盲样本准备后、查询/评分/指标前发现冻结 runner 将 gold 覆盖检查排在神经评分前，与协议的 late-gold 要求冲突；透明实现勘误把覆盖检查移到完整评分后，不改变任何结果相关公式或门槛。该勘误意味着执行顺序修正必须在结果中持续披露，但没有查看逐例分数或指标后修改方法。

五折 OOF 的 utility F1 为 0.191845，相对未门控 v49 的 0.154504 提升 +0.037341；弃答率为 0.141667。与此同时，evidence-bearing F1 从 0.231756 降至 0.194018，损失 0.037738，超过冻结的 0.03 上限；NotMentioned 弃答准确率 0.1875 也未达到 0.20。按“全部条件必须通过”的规则，状态锁定为 `CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN`。官方 train 未打开、确认实验未启动、最终 dev 阈值不授权 train 开启或选择器采用；v51 dev 与 v50 test 均不得用于事后重新选阈值或方法。Gate 2 保持 `NO-GO/SHADOW`，这仍不是 ContractNLI 官方 NLI 榜单、开放语料检索、真实 SetR 复现或洪水领域结论。

### ContractNLI 无参数排名一致性确认（v52）

v52 在打开官方 train 前冻结一个没有学习阈值、温度、边距、先验或候选数特征的门控：对 anchor、support、contradiction 各自按原始 reranker logit 降序、候选 ID 升序取唯一第一名；只有 anchor 第一名与后两者之一为同一跨度才通过，通过后原样运行 v49 低核心—次名分歧选择器，否则返回空集合。v50 test 与 v51 dev 永久排除，train 仅作为一次性角色反转确认集，不允许结果后调参或方法选型。

官方 train 共 7,191 条合格记录；固定抽样形成 600 条/319 份合同，三标签各 200 条且每合同最多 6 条。候选池最小 18、均值 78.035、最大 350；600 条固定查询无回退，完整 GPU 分数 SHA-256 为 `120055e204035c60800844993db2f22d399ac32011c7a3d0ee27fb44fc353175`，gold 只在完整评分后连接。候选 utility F1 为 0.250304、evidence F1 为 0.170455、弃答率为 0.373333、NotMentioned 弃答准确率为 0.410000。相对未门控 v49 与原生零控制均提升 +0.081783（95% CI [+0.047778,+0.117474]），但相对相同门控下最强非 FRC hybrid 仅 +0.009038（95% CI [-0.006814,+0.024925]），未满足点估计至少 0.01 与区间下界大于 0；evidence F1 相对未门控 v49 下降 0.082325，也超过 0.03 上限。按全部条件同时通过的冻结规则，状态为 `CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED`，不采用选择器，Gate 2 保持 `NO-GO/SHADOW`。这不是 ContractNLI 官方 NLI 榜单、开放语料检索、真实 SetR 复现或洪水领域专家结论。

```powershell
conda run -n rag_exp python scripts/run_evidence_inference_low_core_divergence_atomic_roles.py evaluate
```

## 4. Gate 2 判据

本轮结论为 `THEORETICAL_PIPELINE_FEASIBLE_BUT_SUPERIORITY_NOT_PROVEN`：公开数据、真实神经评分器、FRC 选择和本地生成器已经形成可复现流水线，但三套主数据上 FRC 相对各自最强可复现基线的 Evidence F1 配对 95% 区间均跨 0。

因此 Gate 2 保持 `NO-GO/SHADOW`。至少补齐以下证据后才可重新评审 `CANARY`：

1. 基于 CONFLICTS 的冲突/过时类型分类和 458 例/916 回答盲评包已经完成，但两名独立评审员与独立裁决员尚未完成，当前不能报告 expected-behavior adherence；
2. 主指标与安全指标预先冻结，不能只挑有利切片；
3. 9/9 命名消融虽已运行，仍需在同一真实防汛领域专家基准上复现；当前 Full 未优于 `w/o Role`，HousingQA Full 与最强字段基线持平，LawShift 与 EUR-Lex Full 均与各自公平适用性过滤基线持平，CONFLICTS Full 也未优于 `w/o Conflict`；
4. HousingQA 冻结真实模型分数上的字段/角色权重扫描已完成，但仍需在带任务字段和专家角色标注的真实防汛基准上预注册复现；CONFLICTS 冲突阈值扫描也属于事后诊断，均不能用于追认调参；
5. 主要数据集上相对最强可复现基线的改善具有一致方向和配对统计支持；
6. 冲突漏报、错误完整声明、不可回答误答等安全指标达到门槛；QASC 补充确认当前为 `NOT_CONFIRMED`，其问题类型重复一致性不足也必须在新数据或真实防汛专家基准上预注册解决；
7. 将已完成的 EUR-Lex/CELLAR 权威日期协议迁移到具有权威版本登记的真实区县防汛文档，并在同一领域验证辖区、版本和有效期联合适用性；
8. 原始输入、配置、种子、哈希、逐样本结果与失败用例可审计。

WhoQA 已补充一套未触碰公开数据上的选择级负结果，但最强基线差值和同时区间均未通过；后续冻结压力诊断又显示 family 差值显著为负，紧 Token 和高观点层没有出现 FRC 优势。候选池仍接近饱和且难例范围有限，因此两项结果都不满足第 5—6 项，也不能授权 `CANARY`。
