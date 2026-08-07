# Evidence Inference 2.0 low-core-divergence evaluation (v49)

- Status: `EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED`
- Cases: 600
- Strongest frozen FRC control: `adaptive_argmax_cardinality_frc_v43`
- Strongest non-FRC: `cross_encoder_topk`
- Expansion-trigger rate: 0.150000
- Gate 2: `NO-GO/SHADOW`

## Aggregate methods

| Method | F1 | Precision | Recall | Complete | Units | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 0.149360 | 0.107491 | 0.279500 | 0.115000 | 4.963889 | 240.735000 |
| `dense_topk` | 0.136561 | 0.097491 | 0.261119 | 0.130000 | 4.927222 | 286.817222 |
| `hybrid_topk` | 0.149591 | 0.105611 | 0.287103 | 0.138889 | 4.943333 | 266.306111 |
| `cross_encoder_topk` | 0.186549 | 0.132306 | 0.356277 | 0.166111 | 4.943889 | 266.462778 |
| `cross_encoder_knapsack` | 0.182250 | 0.128898 | 0.349736 | 0.163889 | 4.994444 | 263.212778 |
| `static_threshold_frc_v39_replay` | 0.186590 | 0.132343 | 0.356416 | 0.166111 | 4.949444 | 266.497778 |
| `static_rank_coverage_frc_v41` | 0.185921 | 0.131769 | 0.356046 | 0.164444 | 4.951111 | 265.494444 |
| `dynamic_rank_coverage_frc_v41` | 0.190486 | 0.134676 | 0.363762 | 0.169444 | 4.946667 | 267.287222 |
| `adaptive_argmax_cardinality_frc_v43` | 0.223942 | 0.203843 | 0.287480 | 0.113889 | 2.845000 | 165.292778 |
| `guarded_adaptive_cardinality_frc_v44` | 0.207813 | 0.159546 | 0.333472 | 0.146111 | 3.927222 | 220.480000 |
| `anchor_guarded_adaptive_cardinality_frc_v45` | 0.208120 | 0.159796 | 0.333749 | 0.145556 | 3.927778 | 219.492778 |
| `consensus_guarded_adaptive_cardinality_frc_v46` | 0.215766 | 0.177935 | 0.305826 | 0.127778 | 3.260000 | 187.020000 |
| `low_core_divergence_guarded_frc_v49` | 0.220684 | 0.193009 | 0.296191 | 0.122222 | 2.994444 | 173.075556 |

## Boundary

This is a one-shot Evidence Inference 2.0 validation full-article sentence evidence-selection experiment, not an ERASER or treatment-effect classification leaderboard result. Annotator references are alternatives, and v49 cases may not be reused for tuning or method selection.
