Place raw RAG source documents under these subdirectories:

- `policy/`
- `case/`
- `profile/`

Supported source formats:

- `.docx`
- `.pdf`

Optional sidecar metadata:

- For `example.docx`, place metadata at `example.meta.json`
- For `example.pdf`, place metadata at `example.meta.json`

Example metadata:

```json
{
  "doc_id": "policy_orange_manual_review",
  "title": "Orange warning manual review rule",
  "region": "西安市碑林区",
  "stage": "Warning",
  "risk_level": "Orange",
  "audience": "government"
}
```

If no raw documents are found for a corpus, the system falls back to built-in
sample documents for that corpus.
