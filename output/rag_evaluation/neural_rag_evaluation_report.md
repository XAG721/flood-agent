# V3 神经向量 RAG 评测报告

- 模型：`BAAI/bge-small-zh-v1.5`
- 后端：transformers.AutoModel CLS pooling；设备：cpu；向量维度：512
- PyTorch：2.9.1+cpu；Transformers：4.57.1
- 用例数：3；Top-K：4；索引构建：69.69 ms

| 方法 | Evidence Recall | Role Coverage | Answer F1 | 任务要素完整率 | 引用正确率 | 无依据证据比例 | 查询延迟 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| neural_dense_top_k | 0.7222 | 0.8333 | 0.6428 | 0.8333 | 0.5833 | 0.4167 | 7.6364 |
| neural_hybrid_rrf | 0.7222 | 0.8333 | 0.6428 | 0.8333 | 0.5833 | 0.4167 | 7.6364 |

## 限制

- 本结果使用仓库内小规模人工构造工程验收集，不代表公开 benchmark 或真实业务标注集结论。
- 查询延迟在模型常驻后测量，不包含首次模型加载与下载时间。
