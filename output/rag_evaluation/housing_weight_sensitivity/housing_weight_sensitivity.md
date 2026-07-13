# HousingQA public real-model field/role-weight sensitivity

- Status: `RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC`
- Source: `reglab/housing_qa@761550cc974fa1d9141ffd39014db89efa2a7230` (CC-BY-SA-4.0)
- Cases / fields / jurisdictions: 40 / 160 / 22
- Models: `BAAI/bge-large-en-v1.5` + `BAAI/bge-reranker-large`
- Gold usage: field_evidence_map and gold_evidence_ids are used only after selection for evaluation; the selector receives only frozen real-model relevance, field, role, applicability, and cost features
- This is a frozen-score, one-factor descriptive sweep; it is not parameter tuning.

| Dimension | Value | Frozen | Changed cases | Evidence F1 | Field coverage | Role coverage | Tokens |
|---|---:|---|---:|---:|---:|---:|---:|
| field_weight | 0.00 | no | 29 | 0.656250 | 0.656250 | 0.633333 | 398.875 |
| field_weight | 0.50 | no | 23 | 0.681250 | 0.681250 | 0.625000 | 407.950 |
| field_weight | 1.00 | no | 15 | 0.687500 | 0.687500 | 0.625000 | 405.550 |
| field_weight | 2.00 | yes | 0 | 0.706250 | 0.706250 | 0.625000 | 401.000 |
| field_weight | 4.00 | no | 10 | 0.712500 | 0.712500 | 0.625000 | 404.425 |
| field_weight | 8.00 | no | 16 | 0.712500 | 0.712500 | 0.625000 | 406.600 |
| role_weight | 0.00 | no | 12 | 0.706250 | 0.706250 | 0.600000 | 398.850 |
| role_weight | 0.50 | no | 4 | 0.712500 | 0.712500 | 0.616667 | 400.275 |
| role_weight | 1.00 | yes | 0 | 0.706250 | 0.706250 | 0.625000 | 401.000 |
| role_weight | 2.00 | no | 9 | 0.712500 | 0.712500 | 0.633333 | 398.550 |
| role_weight | 4.00 | no | 13 | 0.712500 | 0.712500 | 0.633333 | 398.550 |

## Decision boundary

- Gate 2: `NO-GO`.
- Both field and role weights are behaviorally identifiable on frozen public real-model scores, but this post-hoc cross-domain sensitivity does not prove FRC superiority or flood-domain validity.

## Limitations

- HousingQA is a public housing-law benchmark, not a flood-response benchmark.
- Field evidence comes from public expert question/statute mappings; the three FRC functional roles are deterministic evaluation labels rather than HousingQA expert role annotations.
- The sweep reuses frozen real-model scores and evaluates evidence selection only; it does not rerun answer generation for every weight setting.
- The sweep is descriptive and post-hoc; it cannot authorize parameter tuning or Gate 2 promotion.
