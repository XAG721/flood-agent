# ContractNLI native-zero consensus abstention evaluation (v50)

- Status: `CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED`
- Cases: 600
- Strongest shared-gate non-FRC: `native_zero_consensus_cross_encoder_topk_v50`
- Strongest ungated frozen FRC: `low_core_divergence_guarded_frc_v49`
- Gate 2: `NO-GO/SHADOW`

## Registered comparisons

- Candidate utility F1: 0.165173
- Candidate vs strongest shared-gate non-FRC: {'point': 0.00443, 'ci_low': -0.015105, 'ci_high': 0.023482, 'clusters': 123, 'resamples': 10000, 'seed': 20260810}
- Candidate vs strongest ungated frozen FRC: {'point': -0.001423, 'ci_low': -0.008765, 'ci_high': 0.005032, 'clusters': 123, 'resamples': 10000, 'seed': 20260810}
- Candidate vs anchor-gate FRC: {'point': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'clusters': 123, 'resamples': 10000, 'seed': 20260810}
- Candidate evidence-bearing F1: 0.247760
- Candidate NotMentioned abstention accuracy: 0.000000
- Candidate abstention rate: 0.000000

## Method aggregates

| Method | Utility F1 | Evidence F1 | Missing abstention | Abstention rate | Mean units | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 0.147765 | 0.221647 | 0.000000 | 0.000000 | 4.978889 | 235.799444 |
| `dense_topk` | 0.149786 | 0.224679 | 0.000000 | 0.000000 | 4.979444 | 215.689444 |
| `hybrid_topk` | 0.160292 | 0.240438 | 0.000000 | 0.000000 | 4.972778 | 235.336111 |
| `cross_encoder_topk` | 0.160743 | 0.241115 | 0.000000 | 0.000000 | 4.983889 | 204.146667 |
| `cross_encoder_knapsack` | 0.158950 | 0.238425 | 0.000000 | 0.000000 | 4.995556 | 203.203333 |
| `static_threshold_frc_v39_replay` | 0.160075 | 0.240112 | 0.000000 | 0.000000 | 4.986667 | 202.665000 |
| `static_rank_coverage_frc_v41` | 0.158561 | 0.237841 | 0.000000 | 0.000000 | 4.987222 | 201.969444 |
| `dynamic_rank_coverage_frc_v41` | 0.158638 | 0.237957 | 0.000000 | 0.000000 | 4.988889 | 201.105000 |
| `adaptive_argmax_cardinality_frc_v43` | 0.165173 | 0.247760 | 0.000000 | 0.000000 | 1.791667 | 68.800000 |
| `guarded_adaptive_cardinality_frc_v44` | 0.168229 | 0.252343 | 0.000000 | 0.000000 | 3.218333 | 129.192778 |
| `anchor_guarded_adaptive_cardinality_frc_v45` | 0.168599 | 0.252899 | 0.000000 | 0.000000 | 3.216111 | 129.429444 |
| `consensus_guarded_adaptive_cardinality_frc_v46` | 0.171157 | 0.256735 | 0.000000 | 0.000000 | 2.406667 | 93.890556 |
| `low_core_divergence_guarded_frc_v49` | 0.166597 | 0.249895 | 0.000000 | 0.000000 | 2.001667 | 78.399444 |
| `native_zero_consensus_bm25_topk_v50` | 0.147765 | 0.221647 | 0.000000 | 0.000000 | 4.978889 | 235.799444 |
| `native_zero_consensus_dense_topk_v50` | 0.149786 | 0.224679 | 0.000000 | 0.000000 | 4.979444 | 215.689444 |
| `native_zero_consensus_hybrid_topk_v50` | 0.160292 | 0.240438 | 0.000000 | 0.000000 | 4.972778 | 235.336111 |
| `native_zero_consensus_cross_encoder_topk_v50` | 0.160743 | 0.241115 | 0.000000 | 0.000000 | 4.983889 | 204.146667 |
| `native_zero_consensus_cross_encoder_knapsack_v50` | 0.158950 | 0.238425 | 0.000000 | 0.000000 | 4.995556 | 203.203333 |
| `native_zero_anchor_adaptive_frc_v50` | 0.165173 | 0.247760 | 0.000000 | 0.000000 | 1.791667 | 68.800000 |
| `native_zero_consensus_adaptive_frc_v50` | 0.165173 | 0.247760 | 0.000000 | 0.000000 | 1.791667 | 68.800000 |

## Boundary

This is a balanced full-contract span-selection and missing-evidence abstention mechanism test, not an official ContractNLI NLI leaderboard result. It does not reproduce Span NLI BERT, open-corpus retrieval, or flood-domain expert evaluation. v50 cases may not be reused for tuning or method selection.
