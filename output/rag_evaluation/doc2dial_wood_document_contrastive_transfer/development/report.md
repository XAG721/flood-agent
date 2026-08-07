# Doc2Dial wOOD document-contrastive transfer (development, v55)

- Status: `DOC2DIAL_WOOD_V55_SCHEMA_INCONCLUSIVE_STOP`
- Phase reached: pre-scoring schema adaptation
- Turn containers: 3,459 list / 12 dict
- Reference census: 19,946 non-empty / 2,772 empty+OOD / 431 empty+non-OOD
- Blind cache written: `false`
- Query generation: `false`
- Neural scoring: `false`
- Confirmation opened: `false`
- Selector adoption: `false`
- Gate 2: `NO-GO/SHADOW`

The frozen v55 adapter rejected the old-version keyed-turn container. It was not patched after source access. The official README also defines every empty target reference as not answerable or irrelevant, which is broader than v55's adjacent-OOD-only definition. Both issues require a separately registered schema correction before any outcome-bearing work.
