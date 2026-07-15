# Google CONFLICTS：FRC 检索与冲突分类审计

- 状态：`RUN`
- 用例：458；有非空正确答案：237
- 数据 SHA-256：`9ed5907314a4907c36219aaab0fd570ce54892bbd6dd8386aa729a1a236a2781`
- 统一预算：Top-K=5，1500 tokens
- 模型：`BAAI/bge-large-en-v1.5` / `BAAI/bge-reranker-large` / `Qwen2.5-7B-Instruct-GPTQ-Int4`
- `coverage_greedy_proxy` 是覆盖贪心代理，不是 SetR 复现。

## 冲突类型

| 类型 | 用例 |
|---|---:|
| No conflict | 161 |
| Complementary information | 115 |
| Conflicting opinions and research outcomes | 115 |
| Conflict due to outdated information | 62 |
| Conflict due to misinformation | 5 |

## 方法对比

| 方法 | Accuracy | Macro-F1 | 过时类型 Recall | 最新日期保留 | 时间端点覆盖 | 答案 Token Recall |
|---|---:|---:|---:|---:|---:|---:|
| bm25_topk | 0.336245 | 0.238886 | 0.516129 | 0.548387 | 0.580645 | 0.810717 |
| dense_topk | 0.340611 | 0.235845 | 0.548387 | 0.483871 | 0.540323 | 0.841596 |
| hybrid_topk | 0.318777 | 0.218964 | 0.548387 | 0.451613 | 0.524194 | 0.835361 |
| cross_encoder_topk | 0.342795 | 0.237312 | 0.645161 | 0.532258 | 0.532258 | 0.827957 |
| coverage_greedy_proxy | 0.344978 | 0.244282 | 0.693548 | 0.580645 | 0.572581 | 0.832880 |
| frc_select | 0.334061 | 0.228111 | 0.564516 | 0.532258 | 0.532258 | 0.827957 |

## 配对判定

- 最强可复现分类基线：`coverage_greedy_proxy`
- FRC - 基线 Accuracy：-0.010917
- 95% CI：[-0.043668, +0.024017]
- 胜/平/负：30/393/35
- Gate 2：`NO-GO`
- 说明：CONFLICTS retrieval and label classification are reproducible, but the paper's expected-behavior adherence evaluation and independent human judging are not reproduced; classification alone cannot authorize CANARY or DEFAULT.

## 限制

- This is a retrieval-selection and conflict-type classification audit, not the paper's official expected-behavior adherence reproduction.
- Only 237 of 458 public cases have nonblank correct_answer values; answer-support diagnostics exclude the rest.
- Date retention is a provenance diagnostic and does not imply that the newest source is factually correct.
- The local Qwen2.5-7B-Instruct-GPTQ-Int4 classifier is shared by all methods and is not an expert judge.
- coverage_greedy_proxy is not SetR and must not be reported as a SetR reproduction.
