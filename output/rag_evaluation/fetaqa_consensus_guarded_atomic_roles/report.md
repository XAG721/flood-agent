# FeTaQA consensus-guarded atomic-role evaluation (v47)

- Status: `FETAQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED`
- Cases: 600
- Strongest frozen FRC control: `guarded_adaptive_cardinality_frc_v44`
- Strongest non-FRC: `cross_encoder_knapsack`
- Consensus trigger rate: 0.496667
- Gate 2: `NO-GO/SHADOW`

## Aggregate methods

| Method | F1 | Precision | Recall | Complete | Units | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 0.330077 | 0.404139 | 0.315984 | 0.056667 | 4.997222 | 129.787778 |
| `dense_topk` | 0.364203 | 0.444722 | 0.345962 | 0.047222 | 4.999444 | 123.561667 |
| `hybrid_topk` | 0.364889 | 0.442944 | 0.349053 | 0.053333 | 4.999444 | 122.303889 |
| `cross_encoder_topk` | 0.365931 | 0.448713 | 0.348985 | 0.041667 | 4.996111 | 135.417778 |
| `cross_encoder_knapsack` | 0.366367 | 0.450861 | 0.349373 | 0.041667 | 4.990000 | 135.274444 |
| `static_threshold_frc_v39_replay` | 0.369958 | 0.453824 | 0.352684 | 0.043333 | 4.996111 | 135.155556 |
| `static_rank_coverage_frc_v41` | 0.372441 | 0.457491 | 0.354397 | 0.043333 | 4.995556 | 135.113333 |
| `dynamic_rank_coverage_frc_v41` | 0.388938 | 0.477111 | 0.370071 | 0.050000 | 4.996667 | 134.631667 |
| `adaptive_argmax_cardinality_frc_v43` | 0.331870 | 0.626389 | 0.250737 | 0.011667 | 2.656667 | 72.529444 |
| `guarded_adaptive_cardinality_frc_v44` | 0.375153 | 0.549806 | 0.316851 | 0.025000 | 3.748333 | 101.866111 |
| `anchor_guarded_adaptive_cardinality_frc_v45` | 0.375153 | 0.549806 | 0.316851 | 0.025000 | 3.748333 | 101.866111 |
| `consensus_guarded_adaptive_cardinality_frc_v46` | 0.358926 | 0.587444 | 0.284537 | 0.015000 | 3.153333 | 85.832778 |

## Boundary

This is a full-table supporting-cell selector experiment, not an official FeTaQA answer-generation leaderboard result. The v47 cases may not be reused for tuning or method selection.
