# QASC Global-batch Scoring Optimization Benchmark

- Decision: `KEEP_FROZEN_PER_CASE_SCORING_ENTRYPOINT`
- Selected rerank batch size: `None`
- Frozen QASC confirmation changed: `False`
- Benchmark cases: 32

| Pipeline | Rerank batch | Seconds | Speedup | Exactness | Peak GPU MiB |
|---|---:|---:|---:|---|---:|
| per-case baseline | 8 | 52.413108 | 1.000000 | `True` | 3471.968 |
| global batch | 8 | 50.841583 | 1.030910 | `True` | 3522.096 |
| global batch | 16 | 50.560933 | 1.036633 | `False` | 3522.096 |
| global batch | 32 | 51.230346 | 1.023087 | `False` | 3612.18 |

The optimization is adopted only as a future scoring entrypoint when all score/rank/FRC-selection checks pass and measured speedup reaches the frozen threshold. It cannot alter the existing QASC scores, `NOT_CONFIRMED` result, Gate 2, or MuSiQue boundary.
