# FinQA anchor-guarded atomic-role evaluation (v45)

- Status: `FINQA_ANCHOR_GUARDED_SUPPORT_NOT_ESTABLISHED`
- Cases: 360
- Strongest non-FRC: `cross_encoder_knapsack`
- Candidate ceiling: 1.000000
- Anchor retention: 1.000000
- Gate 2: `NO-GO/SHADOW`

## Aggregate methods

| Method | F1 | Precision | Recall | Complete | Units | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 0.392206 | 0.276667 | 0.795949 | 0.663889 | 4.999074 | 235.086111 |
| `dense_topk` | 0.395982 | 0.280926 | 0.799289 | 0.643519 | 4.998148 | 236.251852 |
| `hybrid_topk` | 0.403478 | 0.287037 | 0.811194 | 0.665741 | 5.000000 | 238.574074 |
| `cross_encoder_topk` | 0.408446 | 0.290556 | 0.823347 | 0.669444 | 4.998148 | 227.615741 |
| `cross_encoder_knapsack` | 0.409502 | 0.291481 | 0.824735 | 0.671296 | 4.994444 | 227.339815 |
| `static_threshold_frc_v39_replay` | 0.408138 | 0.290185 | 0.823578 | 0.670370 | 5.000000 | 227.587963 |
| `static_rank_coverage_frc_v41` | 0.407829 | 0.290185 | 0.821495 | 0.667593 | 5.000000 | 227.671296 |
| `dynamic_rank_coverage_frc_v41` | 0.412252 | 0.292963 | 0.830407 | 0.675926 | 4.998148 | 228.303704 |
| `adaptive_argmax_cardinality_frc_v43` | 0.623946 | 0.656019 | 0.663360 | 0.480556 | 1.858333 | 89.447222 |
| `guarded_adaptive_cardinality_frc_v44` | 0.510429 | 0.412176 | 0.770470 | 0.602778 | 3.236111 | 152.644444 |
| `anchor_guarded_adaptive_cardinality_frc_v45` | 0.510429 | 0.412176 | 0.770470 | 0.602778 | 3.236111 | 152.644444 |

## Boundary

This is a full-context supporting-fact selector experiment, not an official FinQA end-to-end leaderboard result or open-corpus financial-report retrieval evaluation. The v45 cases may not be reused for tuning or method selection.
