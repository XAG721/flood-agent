# CONFLICTS real-model FRC conflict ablation

- Cases: 458
- Data origin: PUBLIC
- Real-model scores: True
- Generator: local_qwen
- Protocol: `post_hoc_one_factor_at_a_time_same_saved_real_model_candidate_pool`
- Gate 2: **NO-GO**

`w/o Conflict` removes only `alternative_claim` and `temporal_validity`; all other candidate, model, K and token-budget conditions remain fixed.

## Ablation

| Variant | Accuracy | Macro F1 | Conflict-role coverage proxy | Exact answer support | Newest-date retention | Changed selections |
|---|---:|---:|---:|---:|---:|---:|
| frc_full | 0.334061 | 0.228111 | 1.000000 | 0.738397 | 0.532258 | 0 |
| wo_conflict | 0.338428 | 0.232431 | 1.000000 | 0.738397 | 0.532258 | 36 |

Full minus `w/o Conflict` classification accuracy: -0.004367, paired 95% CI [-0.013100, +0.004367].

## Conflict-threshold sensitivity

Only the two conflict-disclosure role thresholds change; core-role thresholds remain 0.55.

| Threshold | Accuracy | Macro F1 | Conflict-role coverage proxy | Exact answer support | Newest-date retention |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.336245 | 0.230210 | 1.000000 | 0.738397 | 0.532258 |
| 0.40 | 0.336245 | 0.230210 | 1.000000 | 0.738397 | 0.532258 |
| 0.55 | 0.334061 | 0.228111 | 1.000000 | 0.738397 | 0.532258 |
| 0.70 | 0.334061 | 0.227926 | 1.000000 | 0.738397 | 0.532258 |
| 0.85 | 0.325328 | 0.221382 | 1.000000 | 0.738397 | 0.532258 |

## Limitations

- The experiment was added after earlier aggregate results were inspected and is post hoc, not preregistered confirmatory evidence.
- CONFLICTS supplies case-level conflict types but not candidate-level gold conflict spans; conflict-role coverage is therefore a reranker-score proxy.
- Exact Qwen predictions are reused only when the case and ordered selected evidence IDs are identical; changed selections are regenerated locally.
- Conflict-type classification is not the CONFLICTS paper's independent expected-behavior adherence judgment.
- The experiment does not supply task-field or applicability annotations and cannot satisfy w/o Field or w/o Applicability.

This public real-model ablation measures the conflict-disclosure component on CONFLICTS. Gate 2 remains NO-GO unless all predeclared comparisons and independent expected-behavior judging pass.
