# QASPER top-2 proposal-guarded atomic-role evaluation (v48)

- Status: `QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_NOT_ESTABLISHED`
- Cases: 600
- Strongest frozen FRC control: `adaptive_argmax_cardinality_frc_v43`
- Strongest non-FRC: `cross_encoder_topk`
- Target-below-five rate: 0.613333
- Gate 2: `NO-GO/SHADOW`

## Aggregate methods

| Method | F1 | Precision | Recall | Complete | Units | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 0.171542 | 0.122194 | 0.368715 | 0.313333 | 4.174444 | 489.643889 |
| `dense_topk` | 0.213282 | 0.153333 | 0.457473 | 0.393889 | 4.306111 | 453.007778 |
| `hybrid_topk` | 0.213391 | 0.153065 | 0.454565 | 0.388889 | 4.197222 | 486.106667 |
| `cross_encoder_topk` | 0.247513 | 0.178194 | 0.516838 | 0.452778 | 4.180000 | 485.109444 |
| `cross_encoder_knapsack` | 0.241947 | 0.174120 | 0.508017 | 0.442778 | 4.252778 | 483.062222 |
| `static_threshold_frc_v39_replay` | 0.246629 | 0.177037 | 0.516811 | 0.452778 | 4.199444 | 484.140000 |
| `static_rank_coverage_frc_v41` | 0.242349 | 0.173593 | 0.509977 | 0.444444 | 4.205000 | 484.181667 |
| `dynamic_rank_coverage_frc_v41` | 0.250206 | 0.180917 | 0.520880 | 0.457222 | 4.150000 | 487.575556 |
| `adaptive_argmax_cardinality_frc_v43` | 0.316885 | 0.288148 | 0.424941 | 0.363889 | 2.251111 | 300.855000 |
| `guarded_adaptive_cardinality_frc_v44` | 0.275033 | 0.210213 | 0.483457 | 0.419444 | 3.253889 | 405.640000 |
| `anchor_guarded_adaptive_cardinality_frc_v45` | 0.274430 | 0.209417 | 0.483365 | 0.419444 | 3.259444 | 405.495556 |
| `consensus_guarded_adaptive_cardinality_frc_v46` | 0.297059 | 0.243954 | 0.455707 | 0.392778 | 2.706111 | 351.902778 |
| `top2_proposal_guarded_frc_v48` | 0.296239 | 0.245796 | 0.477256 | 0.412222 | 2.886667 | 408.017778 |

## Boundary

This is a full-paper evidence-selection experiment, not an official QASPER answer-generation leaderboard result. Alternative answer-level references are not unioned, and v48 cases may not be reused for tuning or method selection.
