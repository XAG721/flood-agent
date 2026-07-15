# LawShift 跨版本适用性真实模型消融

- 固定来源：`triangularPeach/LawShift@0fce4f3821140bde29081ae0b20500e79aa065d5`
- 许可：`Apache-2.0`
- 用例：124（31 类修订，修订前/后各 62）
- Reranker：`BAAI/bge-reranker-large`
- 范围：专家审阅的假设修订版本替换；不含法定生效/失效日期。

| 方法 | Article Recall@1 | Version Accuracy | Exact Evidence Accuracy | Invalid Applicability |
|---|---:|---:|---:|---:|
| char_bm25_top1 | 0.282258 | 0.572581 | 0.185484 | 0.427419 |
| cross_encoder_top1 | 0.491935 | 0.548387 | 0.290323 | 0.451613 |
| applicability_filtered_cross_encoder_top1 | 0.427419 | 1.000000 | 0.427419 | 0.000000 |
| frc_full | 0.427419 | 1.000000 | 0.427419 | 0.000000 |
| w/o_applicability | 0.491935 | 0.548387 | 0.290323 | 0.451613 |

Full−w/o Applicability 的 Exact Evidence Accuracy 差值为 +0.137097，95% CI=[+0.080645, +0.201613]。
最强公平基线为 `applicability_filtered_cross_encoder_top1`；Full 相对其差值为 +0.000000，Gate 2 保持 `NO-GO`。

## 边界

- LawShift evaluates Chinese criminal-law judgment adaptation, not flood-response or district-government evidence retrieval.
- The revised statutes are expert-reviewed hypothetical revisions rather than enacted historical versions with authoritative effective dates.
- The benchmark identifies before/after version replacement but cannot test exact effective or expiry dates.
- The frozen candidate pool is a retrieval stress slice built from the labeled article pair plus lexical hard negatives, not the paper's original legal-judgment-prediction protocol.
- Only evidence selection is evaluated; charge and sentence generation are not scored in this retrieval ablation.
