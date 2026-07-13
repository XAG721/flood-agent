# FRC-Select 公开数据真实模型参考审计

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

- 覆盖状态：`PARTIAL`；已运行 6/9 组。
- Gate 必需比较通过：`False`。缺失消融不得按通过处理。

| 消融 | 状态 | Evidence F1 | Role Coverage |
|---|---|---:|---:|
| full | RUN | 0.705754 | 1.000000 |
| w/o_role | RUN | 0.706158 | 0.975906 |
| w/o_field | NOT_RUN | — | — |
| w/o_applicability | NOT_RUN | — | — |
| w/o_redundancy | RUN | 0.705754 | 1.000000 |
| w/o_conflict | NOT_RUN | — | — |
| w/o_reranker | RUN | 0.697327 | 1.000000 |
| role_only | RUN | 0.694475 | 1.000000 |
| random_role | RUN | 0.705086 | 0.997953 |

`w/o Reranker` 相对 Full 的 Evidence F1 配对差值为 -0.008427，95% CI=[-0.018126, +0.001562]；区间跨 0。

### 消融数据可识别性

| 消融 | 可识别状态 | 原因 |
|---|---|---|
| w/o_field | SCHEMA_BLOCKED | public artifacts contain neither required field schemas nor candidate field_scores |
| w/o_applicability | SCHEMA_BLOCKED | public artifacts contain no standardized applicability/version/jurisdiction/effective-period fields |
| w/o_conflict | SCHEMA_BLOCKED | the three primary public artifacts contain no candidate-level conflict annotations; CONFLICTS is a separate challenge dataset |
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
| role_and_field_weights | PARTIAL_ROLE_ONLY |
| conflict_threshold | NOT_RUN |
| document_missing_ratio | RUN |
| chunk_length | NOT_RUN |

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
- 结论：real-model FRC runs are reproducible, but paired confidence intervals do not establish consistent superiority; Full does not outperform w/o Role and w/o Field is schema-blocked on the primary public artifacts; CONFLICTS is run but does not reproduce the paper's independent expected-behavior adherence judgment。

这组结果证明真实模型、公开数据和 FRC 选择器可以形成可复现流水线，但不能证明 FRC 已经稳定优于强重排基线。

## 限制

- The report imports existing real-model artifacts and recomputes paired evidence metrics; it does not retrain models.
- The SetR paper implementation is not available in this environment; coverage_greedy_proxy is not SetR.
- ConditionalQA generation scores are low, so evidence-selection feasibility must not be presented as answer-generation superiority.
- The deterministic missing-evidence challenge removes one gold passage and reuses saved scores; it is a robustness audit, not an official dataset split.
- The real-model design audit remains incomplete; schema-blocked variants are not treated as run or passed, and several sensitivity dimensions remain NOT_RUN.
- CONFLICTS conflict-type classification is complete, but the paper's expected-behavior adherence metric and independent human judging are not reproduced.
