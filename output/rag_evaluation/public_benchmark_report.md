# V3 公开 RAG Benchmark 子集报告

- 模型：`BAAI/bge-small-en-v1.5`；设备：cpu；Top-K：4
- 固定随机种子：20260712；每个数据集最多 100 条
- 这是固定子集工程复现，不是官方全量 leaderboard 成绩。

| 数据集 | 用例 | 候选规模 | 方法 | Evidence Recall | 完整证据集命中率 | MRR | 延迟 ms |
|---|---:|---:|---|---:|---:|---:|---:|
| MultiHop-RAG | 100 | 609 | neural_dense_top_k | 0.5083 | 0.1800 | 0.6908 | 26.8311 |
| MultiHop-RAG | 100 | 609 | neural_hybrid_rrf | 0.5992 | 0.2600 | 0.7708 | 26.8311 |
| ConditionalQA | 100 | 14994 | neural_dense_top_k | 0.2700 | 0.1400 | 0.4650 | 27.4502 |
| ConditionalQA | 100 | 14994 | neural_hybrid_rrf | 0.2346 | 0.1200 | 0.4200 | 27.4502 |
| HotpotQA distractor validation | 100 | 10 | neural_dense_top_k | 0.8700 | 0.7500 | 0.9600 | 305.0966 |
| HotpotQA distractor validation | 100 | 10 | neural_hybrid_rrf | 0.8700 | 0.7600 | 0.9408 | 305.0966 |

## 数据溯源

```json
{
  "MultiHop-RAG": {
    "repository_revision": "3dd4d4e79fc9843008b8f832da99086a82f1a805",
    "corpus_sha256": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
    "queries_sha256": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    "full_query_count": 2556
  },
  "ConditionalQA": {
    "repository_revision": "77bd295952daf415548b3244db10880d3d55cfe0",
    "documents_sha256": "9a139019910865e21805110e72648340a59b638a8598c4beb20ef68eec746fb8",
    "dev_sha256": "6258c7218a9543cbbae9faaa095b42a0e50af0d26b91fe411ae1b1e97e5204c8",
    "full_dev_count": 285,
    "sampled_document_count": 151
  },
  "HotpotQA": {
    "parquet_sha256": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    "full_validation_count": 7405
  }
}
```

## 限制

- 每个数据集使用固定随机子集，不能替代官方全量评测或 leaderboard 提交。
- ConditionalQA 候选池由样本关联文档加 100 个固定干扰文档组成；HotpotQA 使用官方 per-query distractor context。
- 本报告只评估证据检索，不使用生成模型计算最终答案准确率。
- MultiHop-RAG 和 ConditionalQA 延迟只含常驻索引后的查询编码；HotpotQA 延迟还包含每题 10 段上下文编码，两者不可直接横向比较。
- 文档输入受模型 512 token 上限截断，长文档结果可能低估分块索引后的最佳性能。
