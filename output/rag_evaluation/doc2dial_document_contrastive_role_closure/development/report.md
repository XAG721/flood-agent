# Doc2Dial document-contrastive role closure (development, v54)

- Status: `DOC2DIAL_V54_SCHEMA_INCONCLUSIVE_STOP`
- Phase reached: pre-scoring data-eligibility check
- Eligible answer-bearing/no-answer turns: 20,431/0
- Schema exclusions: 0
- Query generation: `false`
- Neural scoring: `false`
- Validation opened: `false`
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

## Closure reason

The fixed official-repository v1.0.1 train split contains no empty-reference target agent turns under the preregistered user-to-agent adjacency rule, so the required 200/200 grounded-versus-irrelevant sample cannot be formed. The registered quota and label semantics were not changed after source access, and validation remains unopened.

## Next independent source

The official project page separately publishes the still-unopened v0.9 archive and documents a `wOOD` folder containing irrelevant turns. Any follow-up must be a new prospective protocol and may use only this aggregate v54 source-eligibility failure, not v54 case text or case-level artifacts.
