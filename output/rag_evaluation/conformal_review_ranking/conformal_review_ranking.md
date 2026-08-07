# FRC-RAG review-only ranking audit

- Status: `KEEP_BASE_REVIEW_ORDER`
- Datasets: 2WikiMultiHopQA, ConditionalQA, HotpotQA, MultiHop-RAG
- Automatic decision or threshold changed: `False`
- Review budgets: [0.05, 0.1, 0.2]
- Gate 2: `NO-GO/SHADOW`

## Ranking comparison at fixed review workload

| Dataset | Baseline AP | Stability AP | AP gain | 5% capture gain | 10% capture gain | 20% capture gain |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.325574 | 0.353531 | 0.027956 | 0.017389 | 0.018282 | 0.012821 |
| ConditionalQA | 0.447403 | 0.519276 | 0.071874 | 0.070273 | 0.084211 | 0.018010 |
| HotpotQA | 0.496059 | 0.525628 | 0.029570 | 0.012694 | 0.016012 | 0.016350 |
| MultiHop-RAG | 0.514131 | 0.599029 | 0.084898 | 0.045263 | 0.080396 | 0.058305 |

## Pre-registered adoption checks

| Check | Measured | Target | Passed |
|---|---|---|---|
| automatic_decisions_exactly_unchanged | `true` | `True` | `True` |
| automatic_thresholds_and_scores_unchanged | `true` | `True` | `True` |
| ranking_and_budgets_not_selected_on_evaluation | `true` | `True` | `True` |
| identical_reviewed_count_at_every_budget | `true` | `True` | `True` |
| minimum_cross_dataset_mean_average_precision_gain | `0.053574` | `>=0.01` | `True` |
| minimum_datasets_with_positive_average_precision_gain | `{"count": 4, "per_dataset": {"2WikiMultiHopQA": 0.027956, "ConditionalQA": 0.071874, "HotpotQA": 0.02957, "MultiHop-RAG": 0.084898}}` | `>=3` | `True` |
| minimum_cross_dataset_mean_complete_capture_gain_at_0_05 | `0.036405` | `>=0.02` | `True` |
| minimum_cross_dataset_mean_complete_capture_gain_at_0_10 | `0.049725` | `>=0.02` | `True` |
| minimum_cross_dataset_mean_complete_capture_gain_at_0_20 | `0.026372` | `>=0.01` | `True` |
| minimum_datasets_with_positive_complete_capture_gain_each_budget | `{"0.05": 4, "0.1": 4, "0.2": 4}` | `>=3` | `True` |
| maximum_dataset_complete_capture_decrease_each_budget | `{"0.05": {"2WikiMultiHopQA": 0.0, "ConditionalQA": 0.0, "HotpotQA": 0.0, "MultiHop-RAG": 0.0}, "0.1": {"2WikiMultiHopQA": 0.0, "ConditionalQA": 0.0, "HotpotQA": 0.0, "MultiHop-RAG": 0.0}, "0.2": {"2WikiMultiHopQA": 0.0, "ConditionalQA": 0.0, "HotpotQA": 0.0, "MultiHop-RAG": 0.0}}` | `<=0.01` | `True` |
| maximum_dataset_review_precision_decrease_at_0_10 | `{"2WikiMultiHopQA": 0.0, "ConditionalQA": 0.0, "HotpotQA": 0.0, "MultiHop-RAG": 0.0}` | `<=0.01` | `True` |
| maximum_dataset_worst_eligible_subgroup_capture_decrease_at_0_10 | `{"2WikiMultiHopQA": 0.048916, "ConditionalQA": 0.0, "HotpotQA": 0.055752, "MultiHop-RAG": 0.0}` | `<=0.05` | `False` |

## Boundary

This audit only compares review order at fixed human-review workload. The automatic candidate set, frozen 37-feature score and conformal threshold remain unchanged. Benchmark labels simulate review utility and do not represent real flood-response reviewer behavior or production authorization.
