# FRC-RAG QASC Supplemental External Confirmation Audit

- Strict status: `NOT_CONFIRMED`
- Data: QASC validation, 926 controlled candidate pools
- Gate 2: `NO-GO/SHADOW`
- Method, thresholds, and existing confirmation series changed: `False`

## Frozen checks

| Metric | Observed | Frozen target | Pass |
|---|---:|---:|---|
| Repeats reducing case-family false completion | 10/10 | 10/10 | `True` |
| Repeats with risk at or below alpha | 4/10 | at least 7/10 | `False` |
| Mean case-family false-completion risk | 0.101786 | at most 0.10 | `False` |

## Effect and scope boundary

Baseline case-family risk was 0.873854; conformal risk was 0.101786; mean abstention was 0.885408; complete recall was 0.282873.

Each case contains two official facts and 38 frozen BM25 distractors. This is not an evaluation of QASC's original 17M-sentence open-corpus retrieval task and is not flood-domain expert validation.

## Protocol deviation

- Present: `True`
- Written-protocol status: `PARTIAL_CONFIRMATION`
- Frozen-evaluator status: `NOT_CONFIRMED`
- Resolution: Use the stricter frozen evaluator status without changing code, thresholds, metrics or the registered protocol.

The stricter `NOT_CONFIRMED` status is retained. No post-hoc reclassification or tuning is allowed, the failed 64-dimensional review ranking is not run, and MuSiQue remains neither downloaded nor inspected.
