# WhoQA frozen budget-stress diagnostic

- Status: `WHOQA_STRESS_SUPPORT_NOT_ESTABLISHED`
- Cases: 5152
- Templates: 26957
- Selection runs: 970452
- v33 blind neural scores are reused unchanged; gold is joined only after selection.
- This post-result diagnostic is not independent confirmation or Gate evidence.

## Family-level primary comparison

- Observed strongest baseline: `dense_topk`
- FRC minus strongest: -0.002926
- Simultaneous 95% CI: [-0.003858, -0.002025]
- FRC minus coverage proxy: -0.000329 (95% CI [-0.000685, +0.000023])

## Frozen configurations

| Config | K | Token budget | Strongest baseline | FRC | Baseline | Delta | Simultaneous 95% CI |
|---|---:|---:|---|---:|---:|---:|---|
| `k2_b1500` | 2 | 1500 | `coverage_greedy_proxy` | 0.994847 | 0.995480 | -0.000633 | [-0.001276, -0.000024] |
| `k3_b1500` | 3 | 1500 | `coverage_greedy_proxy` | 0.994611 | 0.994771 | -0.000160 | [-0.000950, +0.000052] |
| `k4_b256` | 4 | 256 | `dense_topk` | 0.488595 | 0.505252 | -0.016657 | [-0.018765, -0.014633] |
| `k4_b512` | 4 | 512 | `dense_topk` | 0.859419 | 0.864076 | -0.004657 | [-0.006232, -0.003119] |
| `k4_b1024` | 4 | 1024 | `bm25_topk` | 0.994366 | 0.994708 | -0.000341 | [-0.001178, +0.000049] |
| `k4_b1500` | 4 | 1500 | `bm25_topk` | 0.994919 | 0.995203 | -0.000283 | [-0.001100, +0.000039] |

## Predeclared strata

| Stratum | Cases | FRC family primary | Baseline | Baseline primary | Delta |
|---|---:|---:|---|---:|---:|
| 2 viewpoints | 2450 | 0.920170 | `dense_topk` | 0.919953 | +0.000217 |
| 3 viewpoints | 919 | 0.872832 | `dense_topk` | 0.870719 | +0.002113 |
| 4 viewpoints | 664 | 0.838983 | `dense_topk` | 0.845308 | -0.006325 |
| 5-8 viewpoints | 1086 | 0.858149 | `dense_topk` | 0.870207 | -0.012057 |
| 9+ viewpoints | 33 | 0.858333 | `dense_topk` | 0.865993 | -0.007660 |
| slot_constrained | 1119 | 0.858155 | `dense_topk` | 0.870082 | -0.011928 |
| token_constrained | 4604 | 0.877512 | `dense_topk` | 0.880137 | -0.002625 |

## Decision boundary

- Checks: `{"every_config_delta_at_least_minus_0_02": true, "family_point_gain_at_least_0_01": false, "family_reference_ci_low_above_zero": false, "family_simultaneous_ci_low_above_zero": false}`
- Selector changed: `false`.
- Gate 2 remains `NO-GO/SHADOW`; CANARY/DEFAULT remains unauthorized.
- WhoQA does not cover temporal invalidation, misinformation, missing files, or flood-policy exceptions.
