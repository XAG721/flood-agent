# HousingQA real-model field and applicability ablation

- Source: `reglab/housing_qa@761550cc974fa1d9141ffd39014db89efa2a7230` (CC-BY-SA-4.0)
- Cases / fields / jurisdictions: 40 / 160 / 22
- Snapshot: 2021; real-model scores: `True`; generator: `local_qwen`
- Gold usage: expected answers and field_evidence_map are used only after selection and generation for scoring; neither is included in scorer queries or generator prompts

| Method | Evidence F1 | Field coverage | Citation precision | Answer accuracy | Case accuracy | Wrong jurisdiction | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25_topk | 0.643750 | 0.643750 | 0.643750 | 0.306250 | 0.000000 | 0.000000 | 458.73 |
| dense_topk | 0.700000 | 0.700000 | 0.700000 | 0.325000 | 0.000000 | 0.000000 | 428.98 |
| hybrid_topk | 0.681250 | 0.681250 | 0.681250 | 0.318750 | 0.000000 | 0.000000 | 454.93 |
| cross_encoder_topk | 0.650000 | 0.650000 | 0.650000 | 0.312500 | 0.000000 | 0.000000 | 395.88 |
| field_decomposition_topk | 0.706250 | 0.706250 | 0.706250 | 0.318750 | 0.000000 | 0.000000 | 398.85 |
| frc_full | 0.706250 | 0.706250 | 0.706250 | 0.306250 | 0.000000 | 0.000000 | 401.00 |
| w/o_field | 0.656250 | 0.656250 | 0.656250 | 0.312500 | 0.000000 | 0.000000 | 398.88 |
| w/o_applicability | 0.531250 | 0.531250 | 0.531250 | 0.337500 | 0.000000 | 0.275000 | 410.35 |

## Paired comparisons

- `full_minus_w_o_field`: field coverage +0.050000 (95% CI [+0.018750, +0.087500]); answer accuracy -0.006250 (95% CI [-0.050000, +0.043750]).
- `full_minus_w_o_applicability`: field coverage +0.175000 (95% CI [+0.125000, +0.225000]); answer accuracy -0.031250 (95% CI [-0.081250, +0.012500]).
- `full_minus_strongest_baseline`: field coverage +0.000000 (95% CI [-0.018750, +0.018750]); answer accuracy -0.012500 (95% CI [-0.031250, +0.000000]).

## Decision boundary

- Gate 2: `NO-GO` — This public expert benchmark can test field coverage and jurisdiction filtering, but a single 2021 snapshot cannot identify version/expiry behavior and the cross-domain result cannot override the existing failed public comparisons.
- HousingQA is a public housing-law benchmark, not a flood-response or district-government benchmark.
- The public corpus is explicitly accurate as of 2021; it has no paired historical/current statute versions, so version replacement and expiry remain untested.
- Composite four-field cases and hard-negative pools are deterministic transformations of expert single-question annotations, not separately expert-reviewed composite tasks.
- Public model pretraining contamination cannot be excluded.
- The frozen test configuration was not tuned after observing these results.
