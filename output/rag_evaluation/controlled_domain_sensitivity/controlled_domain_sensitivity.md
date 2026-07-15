# FRC controlled-domain weight and conflict-threshold sensitivity

- Scope: `SYNTHETIC` controlled domain; neural model used: `False`
- Cases/documents: 3/18
- Gold field mappings: 16 (scoring only; never passed to selector)
- Budget: Top-K=4, 520 tokens
- Protocol: `one_factor_at_a_time`; benchmark SHA-256: `689a0e9ee694982b15417f0a37b62cab70f7b17890417aeb7ea6dbcf9cdad7c8`

| Dimension | Value | Evidence F1 | Gold field coverage | Selector field coverage | Role coverage | Flagged evidence | Flagged cases | Conflict-pair cases | Case accuracy | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| role_weight | 0.00 | 0.750000 | 0.773810 | 1.000000 | 0.888889 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 268.33 |
| role_weight | 0.50 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| role_weight | 1.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| role_weight | 2.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| role_weight | 4.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.111111 | 0.333333 | 0.000000 | 0.000000 | 225.00 |
| field_weight | 0.00 | 0.750000 | 0.773810 | 0.944444 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 271.67 |
| field_weight | 1.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| field_weight | 2.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| field_weight | 4.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| field_weight | 8.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.111111 | 0.333333 | 0.000000 | 0.000000 | 225.00 |
| conflict_threshold | 0.00 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| conflict_threshold | 0.35 | 0.750000 | 0.773810 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | 270.00 |
| conflict_threshold | 0.50 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.111111 | 0.333333 | 0.000000 | 0.000000 | 225.00 |
| conflict_threshold | 0.80 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.111111 | 0.333333 | 0.000000 | 0.000000 | 225.00 |
| conflict_threshold | 1.00 | 0.607143 | 0.573810 | 1.000000 | 1.000000 | 0.111111 | 0.333333 | 0.000000 | 0.000000 | 225.00 |

## Boundary

- This small benchmark was repository-constructed and validated for parameter identifiability; it is not held out and is not a public real-model result.
- It demonstrates selector behavior and parameter traceability; it cannot establish production effectiveness or Gate 2 superiority.
- Primary public artifacts still lack standardized field, applicability, and candidate-conflict annotations.
- The sweep is descriptive and must not be used to tune the frozen public test configuration after evaluation.
