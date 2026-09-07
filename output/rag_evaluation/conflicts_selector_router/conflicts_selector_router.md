# CONFLICTS selector-routing discovery audit

- Status: `DISCOVERY_ROUTING_SIGNAL_NOT_ADOPTED_FRC_CONTRIBUTION_NOT_ESTABLISHED`
- Cases: 458
- Boundary: retrospective discovery; not an independent confirmation.
- Runtime features contain no gold label, correct answer, or correctness metric.
- Gate 2: `NO-GO/SHADOW`

## Cross-fitted comparison

| Method | Accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|
| coverage_greedy_proxy | 0.344978 | 0.244282 | 158 |
| majority_vote | 0.358079 | 0.248738 | 164 |
| without_frc | 0.393013 | 0.269048 | 180 |
| all_methods | 0.395197 | 0.268784 | 181 |

## Paired discovery signal

- All-method router - static baseline: +0.050218
- 95% paired bootstrap CI: [+0.008734, +0.091703]
- Wins/ties/losses: 61/359/38
- All-method router - strict no-FRC router: +0.002183
- FRC-only unique oracle cases: 2
- Outdated-conflict recall delta vs static: -0.435483
- Folds with nonnegative accuracy gain: 3/5

## Interpretation

The six frozen selectors contain learnable complementary behavior, but the strict no-FRC ablation reaches nearly the same result. The large outdated-conflict recall loss and two negative folds block adoption. Therefore this audit exposes selector-routing headroom, but supports neither a safe router nor FRC superiority. The result was developed after aggregate CONFLICTS results were visible and requires untouched confirmation.

The router remains disabled and Gate 2 remains `NO-GO/SHADOW`.
