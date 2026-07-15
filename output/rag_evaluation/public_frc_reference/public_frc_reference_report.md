# FRC-RAG 公开数据真实模型参考审计

- Embedding：`BAAI/bge-large-en-v1.5`
- Reranker：`BAAI/bge-reranker-large`
- Generator：`Qwen2.5-7B-Instruct-GPTQ-Int4`
- 统一预算：Top-K=5，1500 tokens
- `coverage_greedy_proxy` 是覆盖贪心代理，不是 SetR 复现。

## 主结果

| 数据集 | 用例 | FRC Evidence F1 | 最强可复现基线 | 基线 F1 | 差值 | 95% CI |
|---|---:|---:|---|---:|---:|---:|
| conditionalqa | 285 | 0.705754 | cross_encoder_topk | 0.706158 | -0.000404 | [-0.003311, +0.002423] |
| multihoprag | 2556 | 0.394001 | cross_encoder_topk | 0.393660 | +0.000342 | [-0.000705, +0.001385] |
| hotpotqa | 1000 | 0.550754 | coverage_greedy_proxy | 0.550818 | -0.000063 | [-0.000857, +0.000667] |

## 设计第 16.2 节消融审计

- 覆盖状态：`PARTIAL`；跨适用公开集已运行 9/9 组。
- Gate 必需比较通过：`False`。缺失消融不得按通过处理。

| 消融 | 状态 | Evidence F1 | Role Coverage |
|---|---|---:|---:|
| full | RUN | 0.705754 | 1.000000 |
| w/o_role | RUN | 0.706158 | 0.975906 |
| w/o_field | RUN_PUBLIC_EXPERT_REAL_MODEL | 0.656250 | — |
| w/o_applicability | RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021 | 0.531250 | — |
| w/o_redundancy | RUN | 0.705754 | 1.000000 |
| w/o_conflict | RUN_REAL_MODEL_CONFLICTS | — | — |
| w/o_reranker | RUN | 0.697327 | 1.000000 |
| role_only | RUN | 0.694475 | 1.000000 |
| random_role | RUN | 0.705086 | 0.997953 |

`w/o Reranker` 相对 Full 的 Evidence F1 配对差值为 -0.008427，95% CI=[-0.018126, +0.001562]；区间跨 0。

### `w/o Conflict`（Google CONFLICTS，真实模型）

该事后诊断只移除 `alternative_claim` 与 `temporal_validity` 两个冲突披露角色；候选池、BGE、重排器、Qwen、K 和 Token 预算保持一致。

| 版本 | 冲突类型准确率 | Macro F1 | 选择变化用例 |
|---|---:|---:|---:|
| Full | 0.334061 | 0.228111 | 0 |
| w/o Conflict | 0.338428 | 0.232431 | 36 |

Full−`w/o Conflict` 准确率差值为 -0.004367，95% CI=[-0.013100, +0.004367]。该结果不证明 Full 更优，也不替代独立 expected-behavior adherence 评判。

### `w/o Field` 与 `w/o Applicability`（HousingQA，公开专家标注，真实模型）

40 个确定性复合用例包含 160 个任务字段、22 个司法辖区；每个字段的正确答案与支持法条仅用于事后评分。适用性消融只识别 2021 快照下的司法辖区过滤，不能识别法规版本替换或失效。

| 版本 | Field Coverage | Citation Support Precision | Wrong-jurisdiction Rate | Answer Accuracy |
|---|---:|---:|---:|---:|
| Full | 0.706250 | 0.706250 | 0.000000 | 0.306250 |
| w/o Field | 0.656250 | 0.656250 | 0.000000 | 0.312500 |
| w/o Applicability | 0.531250 | 0.531250 | 0.275000 | 0.337500 |

Full−w/o Field 的字段覆盖差值为 +0.050000，95% CI=[+0.018750, +0.087500]；Full−w/o Applicability 的字段覆盖差值为 +0.175000，错误司法辖区率差值为 -0.275000。
最强字段覆盖基线为 `field_decomposition_topk`；Full 相对其字段覆盖差值为 +0.000000，未达到 Gate 2 要求的 +0.05。

### 字段/角色权重敏感性（HousingQA，公开冻结真实模型分数）

该单因素扫描复用同一 40 用例、160 字段、22 辖区候选池；gold 字段证据只在选择完成后评分。结果用于辨识权重行为，不用于回看结果后修改冻结参数。

| 维度 | 值 | Evidence F1 | Field Coverage | Role Coverage | 相对冻结选择变化 |
|---|---:|---:|---:|---:|---:|
| field_weight | 0.00 | 0.656250 | 0.656250 | 0.633333 | 29 |
| field_weight | 0.50 | 0.681250 | 0.681250 | 0.625000 | 23 |
| field_weight | 1.00 | 0.687500 | 0.687500 | 0.625000 | 15 |
| field_weight | 2.00 | 0.706250 | 0.706250 | 0.625000 | 0 |
| field_weight | 4.00 | 0.712500 | 0.712500 | 0.625000 | 10 |
| field_weight | 8.00 | 0.712500 | 0.712500 | 0.625000 | 16 |
| role_weight | 0.00 | 0.706250 | 0.706250 | 0.600000 | 12 |
| role_weight | 0.50 | 0.712500 | 0.712500 | 0.616667 | 4 |
| role_weight | 1.00 | 0.706250 | 0.706250 | 0.625000 | 0 |
| role_weight | 2.00 | 0.712500 | 0.712500 | 0.633333 | 9 |
| role_weight | 4.00 | 0.712500 | 0.712500 | 0.633333 | 13 |

字段权重从 0 提高到冻结值 2 时 Evidence F1/字段覆盖由 0.656250 提高到 0.706250；角色权重从 0 提高到 2 时角色覆盖由 0.600000 提高到 0.633333。字段与角色权重均可辨识，但更高权重相对冻结配置的最大 Evidence F1 增益只有 0.006250，且角色标签不是 HousingQA 专家标注，因此 Gate 2 仍为 NO-GO。

### `w/o Applicability`（LawShift，31 类专家审阅修订，真实重排）

124 个用例平衡覆盖修订前/后快照；候选池同时包含目标法条两个版本和三对词法难负例。法条版本是专家审阅的假设修订，不含权威生效或失效日期。

| 版本 | Article Recall@1 | Version Accuracy | Exact Evidence Accuracy | Invalid Applicability |
|---|---:|---:|---:|---:|
| Full | 0.427419 | 1.000000 | 0.427419 | 0.000000 |
| w/o Applicability | 0.491935 | 0.548387 | 0.290323 | 0.451613 |
| Applicability-filtered Cross-Encoder | 0.427419 | 1.000000 | 0.427419 | 0.000000 |

Full−w/o Applicability 的精确版本证据差值为 +0.137097，95% CI=[+0.080645, +0.201613]；错误版本率差值为 -0.451613。
Full 的 Article Recall@1 相对无过滤版本差值为 -0.064516；相对公平过滤基线的精确版本证据差值为 +0.000000。因此版本过滤有效，但 FRC 独有优势未获证明。

### `w/o Applicability`（EUR-Lex/CELLAR，权威生效与失效边界，真实重排）

30 对废止/替代法案形成 60 个边界用例，分别查询旧法最后有效日和新法首个生效日；权威 CELLAR 日期只用于适用性过滤与事后评分，查询文本不包含 gold CELEX。

| 版本 | Exact Evidence Accuracy | Validity Accuracy | Invalid Applicability | Wrong Boundary Version |
|---|---:|---:|---:|---:|
| Full | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| w/o Applicability | 0.483333 | 0.483333 | 0.516667 | 0.516667 |
| Applicability-filtered Cross-Encoder | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

Full−w/o Applicability 的精确证据差值为 +0.516667，95% CI=[+0.383333, +0.650000]；无效适用率差值为 -0.516667。
Full 相对公平过滤基线的精确证据差值为 +0.000000。因此权威日期过滤有效，但 FRC 独有优势仍未获证明，且该语料属于欧盟法律而非区县防汛文档。

### 消融数据可识别性

| 消融 | 可识别状态 | 原因 |
|---|---|---|
| w/o_field | RUN_PUBLIC_EXPERT_REAL_MODEL | HousingQA supplies expert questions and supporting statutes; four independent questions are deterministically composed into fields and evaluated with frozen BGE, Cross-Encoder, and local Qwen models |
| w/o_applicability | RUN_PUBLIC_REAL_MODEL_JURISDICTION_REVISION_EFFECTIVE_EXPIRY | HousingQA identifies jurisdiction filtering under its 2021 snapshot; LawShift identifies expert-reviewed before/after statutory replacement; EUR-Lex/CELLAR supplies official effective and expiry dates at adjacent repeal boundaries. |
| w/o_conflict | RUN_REAL_MODEL_CONFLICTS | Google CONFLICTS supplies case-level conflict types; the ablation removes only alternative-claim and temporal-validity disclosure roles on the frozen real-model pool |
| w/o_reranker | RUN_REAL_BIENCODER_RESCORING | supplemental artifact recomputed both relevance and role scores without a Cross-Encoder |

## K 与参数敏感性审计

| 数据集 | K | FRC Recall | 最强基线 | 基线 Recall | 差值 |
|---|---:|---:|---|---:|---:|
| conditionalqa | 2 | 0.363240 | cross_encoder_topk | 0.358824 | +0.004415 |
| conditionalqa | 3 | 0.504181 | cross_encoder_topk | 0.502514 | +0.001667 |
| conditionalqa | 5 | 0.704547 | cross_encoder_topk | 0.704869 | -0.000322 |
| conditionalqa | 8 | 0.810879 | cross_encoder_topk | 0.811757 | -0.000877 |
| multihoprag | 2 | 0.422372 | cross_encoder_topk | 0.421557 | +0.000815 |
| multihoprag | 3 | 0.507075 | cross_encoder_topk | 0.506195 | +0.000880 |
| multihoprag | 5 | 0.602765 | cross_encoder_topk | 0.602113 | +0.000652 |
| multihoprag | 8 | 0.635824 | cross_encoder_topk | 0.635596 | +0.000228 |
| hotpotqa | 2 | 0.643902 | cross_encoder_topk | 0.644486 | -0.000583 |
| hotpotqa | 3 | 0.765162 | coverage_greedy_proxy | 0.765912 | -0.000750 |
| hotpotqa | 5 | 0.865712 | coverage_greedy_proxy | 0.865962 | -0.000250 |
| hotpotqa | 8 | 0.922648 | cross_encoder_topk | 0.922648 | +0.000000 |

ConditionalQA 保存分数上共审计 80 组 FRC 参数；最佳 Evidence F1=0.705754（alpha=1.0、gamma=0.0、role_threshold=0.55）。该扫参不是独立留出集结果，不用于事后改写冻结测试配置。

| 敏感性维度 | 状态 |
|---|---|
| k_2_3_5_8 | RUN |
| token_budget_512_1024_2048 | RUN |
| role_and_field_weights | RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC |
| conflict_threshold | RUN_REAL_MODEL_CONFLICTS |
| document_missing_ratio | RUN |
| chunk_length | RUN |

### Controlled-domain role/field/conflict sensitivity

This diagnostic uses a repository-constructed `SYNTHETIC` benchmark and no neural model. It verifies selector behavior but does not satisfy the public real-model Gate 2 requirement.

| Dimension | Value | Evidence F1 | Gold field coverage | Selector field coverage | Role coverage | Flagged cases | Case accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| role_weight | 0.00 | 0.750000 | 0.773810 | 1.000000 | 0.888889 | 0.000000 | 0.333333 |
| role_weight | 0.50 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| role_weight | 1.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| role_weight | 2.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| role_weight | 4.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.333333 | 0.000000 |
| field_weight | 0.00 | 0.750000 | 0.773810 | 0.944444 | 1.000000 | 0.000000 | 0.333333 |
| field_weight | 1.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| field_weight | 2.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| field_weight | 4.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| field_weight | 8.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.333333 | 0.000000 |
| conflict_threshold | 0.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| conflict_threshold | 0.35 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.333333 |
| conflict_threshold | 0.50 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.333333 | 0.000000 |
| conflict_threshold | 0.80 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.333333 | 0.000000 |
| conflict_threshold | 1.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.333333 | 0.000000 |

### CONFLICTS conflict-threshold sensitivity (real model)

Only the two conflict-disclosure role thresholds change; this is a post-hoc diagnostic.

| Threshold | Accuracy | Macro F1 | Exact answer support | Newest-date retention |
|---:|---:|---:|---:|---:|
| 0.25 | 0.336245 | 0.230210 | 0.738397 | 0.532258 |
| 0.40 | 0.336245 | 0.230210 | 0.738397 | 0.532258 |
| 0.55 | 0.334061 | 0.228111 | 0.738397 | 0.532258 |
| 0.70 | 0.334061 | 0.227926 | 0.738397 | 0.532258 |
| 0.85 | 0.325328 | 0.221382 | 0.738397 | 0.532258 |

### 分块长度敏感性（ConditionalQA，真实重评分）

| 分块 tokens | FRC Evidence F1 | 最强基线 | 基线 F1 | 差值 | 95% CI | 重复父证据率 |
|---:|---:|---|---:|---:|---:|---:|
| 64 | 0.614475 | coverage_greedy_proxy | 0.614854 | -0.000379 | [-0.006775, +0.006388] | 0.324912 |
| 128 | 0.632662 | coverage_greedy_proxy | 0.632700 | -0.000038 | [-0.004760, +0.005067] | 0.279298 |
| 256 | 0.679077 | coverage_greedy_proxy | 0.680232 | -0.001155 | [-0.006445, +0.003629] | 0.122807 |

### Token 预算敏感性

| 数据集 | Token 预算 | FRC Evidence F1 | 最强基线 | 基线 F1 | 差值 |
|---|---:|---:|---|---:|---:|
| conditionalqa | 512 | 0.484816 | coverage_greedy_proxy | 0.481744 | +0.003072 |
| conditionalqa | 1024 | 0.696558 | cross_encoder_topk | 0.696673 | -0.000115 |
| conditionalqa | 2048 | 0.705754 | cross_encoder_topk | 0.706158 | -0.000404 |
| multihoprag | 512 | 0.453156 | cross_encoder_topk | 0.452804 | +0.000352 |
| multihoprag | 1024 | 0.421133 | cross_encoder_topk | 0.420154 | +0.000979 |
| multihoprag | 2048 | 0.394001 | cross_encoder_topk | 0.393660 | +0.000341 |
| hotpotqa | 512 | 0.550754 | coverage_greedy_proxy | 0.550818 | -0.000064 |
| hotpotqa | 1024 | 0.550754 | coverage_greedy_proxy | 0.550818 | -0.000064 |
| hotpotqa | 2048 | 0.550754 | coverage_greedy_proxy | 0.550818 | -0.000064 |

### 多档缺失比例敏感性（ConditionalQA）

| 目标缺失 | 实际缺失 | FRC Evidence F1 | 最强基线 | 差值 | FRC 错误完整声明率 |
|---:|---:|---:|---|---:|---:|
| 0% | 0.00% | 0.756220 | cross_encoder_topk | -0.000441 | 0.000000 |
| 25% | 27.25% | 0.780936 | cross_encoder_topk | -0.001597 | 0.723005 |
| 50% | 51.63% | 0.731869 | coverage_greedy_proxy | -0.001749 | 0.514056 |
| 75% | 76.12% | 0.570430 | coverage_greedy_proxy | -0.000435 | 0.280156 |

## 缺失证据挑战

协议：remove the first gold passage from every ConditionalQA case with at least two gold passages, then rerun selectors from saved real-model scores。用例：261。

| 方法 | 可用证据 Recall | 完整可用证据集 | 角色覆盖 | 错误宣称完整率 |
|---|---:|---:|---:|---:|
| bm25_topk | 0.762184 | 0.467433 | 0.814496 | 0.777778 |
| dense_topk | 0.769794 | 0.478927 | 0.826628 | 0.777778 |
| hybrid_topk | 0.771657 | 0.471264 | 0.833653 | 0.796935 |
| cross_encoder_topk | 0.773883 | 0.482759 | 0.827267 | 0.793103 |
| coverage_greedy_proxy | 0.773130 | 0.478927 | 0.844828 | 0.816092 |
| frc_select | 0.772180 | 0.478927 | 0.844828 | 0.816092 |

## 冲突与失效证据

- 状态：`RUN`
- 说明：CONFLICTS retrieval and label classification are reproducible, but the paper's expected-behavior adherence evaluation and independent human judging are not reproduced; classification alone cannot authorize CANARY or DEFAULT.
- 用例：458；有正确答案标注：237
- 最强可复现基线：`coverage_greedy_proxy`，Accuracy=0.344978
- FRC Accuracy：0.334061
- FRC - 基线：-0.010917，95% CI=[-0.043668, +0.024017]
- FRC 过时信息类型 Recall：0.564516

## 判定

- 状态：`THEORETICAL_PIPELINE_FEASIBLE_BUT_SUPERIORITY_NOT_PROVEN`
- Gate 2：`NO-GO`
- 结论：real-model FRC runs are reproducible, but paired confidence intervals do not establish consistent superiority; Full does not outperform w/o Role; HousingQA Full does outperform w/o Field but ties the strongest field-decomposition baseline, and LawShift Full improves exact version evidence over w/o Applicability but ties the fair applicability-filtered Cross-Encoder baseline; EUR-Lex/CELLAR now identifies official effective/expiry boundaries, where Full improves over the unfiltered variant but again ties the fair filtered Cross-Encoder baseline; the CONFLICTS Full variant also does not outperform w/o Conflict; CONFLICTS is run but does not reproduce the paper's independent expected-behavior adherence judgment。

这组结果证明真实模型、公开数据和 FRC 选择器可以形成可复现流水线，但不能证明 FRC 已经稳定优于强重排基线。

## 限制

- The report imports existing real-model artifacts and recomputes paired evidence metrics; it does not retrain models.
- The SetR paper implementation is not available in this environment; coverage_greedy_proxy is not SetR.
- ConditionalQA generation scores are low, so evidence-selection feasibility must not be presented as answer-generation superiority.
- The deterministic missing-evidence challenge removes one gold passage and reuses saved scores; it is a robustness audit, not an official dataset split.
- The nine named ablation variants now have execution artifacts across compatible public datasets; applicability is separately identifiable through HousingQA jurisdiction, LawShift expert-reviewed hypothetical revisions, and EUR-Lex/CELLAR official effective/expiry dates; and HousingQA now supplies a frozen-score public real-model field/role-weight sweep. This is still not full design coverage because the weight and applicability sources are cross-domain rather than independently judged flood-response records.
- CONFLICTS conflict-type classification is complete, but the paper's expected-behavior adherence metric and independent human judging are not reproduced.
- HousingQA is a public housing-law benchmark, not a flood-response or district-government benchmark.
- The public corpus is explicitly accurate as of 2021; it has no paired historical/current statute versions, so version replacement and expiry remain untested.
- Composite four-field cases and hard-negative pools are deterministic transformations of expert single-question annotations, not separately expert-reviewed composite tasks.
- Public model pretraining contamination cannot be excluded.
- The frozen test configuration was not tuned after observing these results.
- LawShift evaluates Chinese criminal-law judgment adaptation, not flood-response or district-government evidence retrieval.
- The revised statutes are expert-reviewed hypothetical revisions rather than enacted historical versions with authoritative effective dates.
- The benchmark identifies before/after version replacement but cannot test exact effective or expiry dates.
- The frozen candidate pool is a retrieval stress slice built from the labeled article pair plus lexical hard negatives, not the paper's original legal-judgment-prediction protocol.
- Only evidence selection is evaluated; charge and sentence generation are not scored in this retrieval ablation.
- The corpus covers EU legal acts and does not establish performance on district flood-response documents.
- CELLAR may expose multiple partial-application dates; the frozen slice keeps only repeal pairs with one unique adjacent expiry/effective boundary.
- EUR-Lex legal texts and consolidated representations are reused for research and are not legal advice or an official legal edition claim.
- The experiment evaluates evidence selection at the document boundary, not answer generation or article-level expected-behavior adherence.
- FRC and the fair filtered Cross-Encoder baseline intentionally share validity metadata, preventing an unfair metadata advantage.
