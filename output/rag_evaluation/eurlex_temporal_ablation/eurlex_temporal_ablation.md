# EUR-Lex effective/expiry applicability ablation

- Status: `RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL`
- Gate 2: `NO-GO`
- Source snapshot: `2026-07-14`
- Pairs / cases: 30 / 60

## Aggregate results

| Method | Exact evidence | Validity accuracy | Invalid applicability | Wrong boundary version |
|---|---:|---:|---:|---:|
| `bm25_top1` | 0.466667 | 0.466667 | 0.533333 | 0.500000 |
| `cross_encoder_top1` | 0.483333 | 0.483333 | 0.516667 | 0.516667 |
| `applicability_filtered_cross_encoder_top1` | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| `frc_full` | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| `w/o_applicability` | 0.483333 | 0.483333 | 0.516667 | 0.516667 |

## Paired applicability effect

Full exact evidence accuracy is 1.000000; `w/o Applicability` is 0.483333. The paired difference is +0.516667 with 95% CI [+0.383333, +0.650000].

Strongest fair baseline: `applicability_filtered_cross_encoder_top1`.

## Decision

Official effective and expiry dates make temporal applicability identifiable, but the fair applicability-filtered Cross-Encoder baseline receives the same metadata and the corpus is EU law rather than flood-response evidence.

## Limitations

- The corpus covers EU legal acts and does not establish performance on district flood-response documents.
- CELLAR may expose multiple partial-application dates; the frozen slice keeps only repeal pairs with one unique adjacent expiry/effective boundary.
- EUR-Lex legal texts and consolidated representations are reused for research and are not legal advice or an official legal edition claim.
- The experiment evaluates evidence selection at the document boundary, not answer generation or article-level expected-behavior adherence.
- FRC and the fair filtered Cross-Encoder baseline intentionally share validity metadata, preventing an unfair metadata advantage.
