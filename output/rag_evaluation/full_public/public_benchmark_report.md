# V3 公开 RAG Benchmark 全量可评测集报告

- 模型：`BAAI/bge-small-en-v1.5`；设备：cpu；Top-K：4
- 固定随机种子：20260712；单数据集样本上限：7405
- 范围：官方发布数据中的全部可映射证据查询；仍不是官方 leaderboard 提交。

| 数据集 | 用例 | 候选规模 | 方法 | Evidence Recall | 完整证据集命中率 | MRR | 延迟 ms |
|---|---:|---:|---|---:|---:|---:|---:|
| MultiHop-RAG | 2255 | 609 | neural_dense_top_k | 0.5294 | 0.2364 | 0.6839 | 11.4596 |
| MultiHop-RAG | 2255 | 609 | neural_hybrid_rrf | 0.6237 | 0.3082 | 0.7823 | 11.4596 |
| ConditionalQA | 271 | 15550 | neural_dense_top_k | 0.2627 | 0.1292 | 0.4397 | 13.8797 |
| ConditionalQA | 271 | 15550 | neural_hybrid_rrf | 0.2421 | 0.1181 | 0.4274 | 13.8797 |
| HotpotQA distractor validation | 7405 | 10 | neural_dense_top_k | 0.8866 | 0.7815 | 0.9390 | 390.2074 |
| HotpotQA distractor validation | 7405 | 10 | neural_hybrid_rrf | 0.8758 | 0.7589 | 0.9452 | 390.2074 |

## 数据溯源

```json
{
  "MultiHop-RAG": {
    "repository_revision": "3dd4d4e79fc9843008b8f832da99086a82f1a805",
    "corpus_sha256": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
    "queries_sha256": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    "full_query_count": 2556,
    "evaluable_query_count": 2255,
    "excluded_query_count": 301,
    "exclusion_rule": "fewer than two evidence URLs map to the released corpus"
  },
  "ConditionalQA": {
    "repository_revision": "77bd295952daf415548b3244db10880d3d55cfe0",
    "documents_sha256": "9a139019910865e21805110e72648340a59b638a8598c4beb20ef68eec746fb8",
    "dev_sha256": "6258c7218a9543cbbae9faaa095b42a0e50af0d26b91fe411ae1b1e97e5204c8",
    "full_dev_count": 285,
    "sampled_document_count": 159,
    "eligible_answerable_count": 271,
    "evaluable_query_count": 271,
    "excluded_query_count": 14,
    "exclusion_rule": "not answerable, missing evidence, or evidence paragraph cannot be mapped"
  },
  "HotpotQA": {
    "parquet_sha256": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    "full_validation_count": 7405
  }
}
```

## 限制

- 覆盖官方发布数据中的全部可评测查询，但使用自定义检索实现与指标，不能替代官方 leaderboard 提交。
- MultiHop-RAG 排除 301 条无法将至少两份证据 URL 映射回发布语料的查询。
- ConditionalQA 排除 14 条不可回答、缺少证据或证据段无法映射的开发样本。
- ConditionalQA 候选池由样本关联文档加 100 个固定干扰文档组成；HotpotQA 使用官方 per-query distractor context。
- 本报告只评估证据检索，不使用生成模型计算最终答案准确率。
- MultiHop-RAG 和 ConditionalQA 延迟只含常驻索引后的查询编码；HotpotQA 延迟还包含每题 10 段上下文编码，两者不可直接横向比较。
- 文档输入受模型 512 token 上限截断，长文档结果可能低估分块索引后的最佳性能。
