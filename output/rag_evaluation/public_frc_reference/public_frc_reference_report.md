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
- 结论：real-model FRC runs are reproducible, but paired confidence intervals do not establish consistent superiority; CONFLICTS is run but does not reproduce the paper's independent expected-behavior adherence judgment。

这组结果证明真实模型、公开数据和 FRC 选择器可以形成可复现流水线，但不能证明 FRC 已经稳定优于强重排基线。

## 限制

- The report imports existing real-model artifacts and recomputes paired evidence metrics; it does not retrain models.
- The SetR paper implementation is not available in this environment; coverage_greedy_proxy is not SetR.
- ConditionalQA generation scores are low, so evidence-selection feasibility must not be presented as answer-generation superiority.
- The deterministic missing-evidence challenge removes one gold passage and reuses saved scores; it is a robustness audit, not an official dataset split.
- CONFLICTS conflict-type classification is complete, but the paper's expected-behavior adherence metric and independent human judging are not reproduced.
