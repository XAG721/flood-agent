from __future__ import annotations

import json
from pathlib import Path

from .models import CorpusType, RAGDocument
from .rag import SimpleRAGStore


class RuntimeRAGDocumentProvider:
    def __init__(self, runtime_path: str | Path, base_documents: list[RAGDocument]) -> None:
        self.runtime_path = Path(runtime_path)
        self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
        self._base_documents = list(base_documents)

    def load_runtime_rag_documents(self) -> list[RAGDocument]:
        if not self.runtime_path.exists():
            return []
        payload = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Runtime RAG document file must contain a JSON array.")
        return [RAGDocument.model_validate(item) for item in payload]

    def save_runtime_rag_documents(self, documents: list[RAGDocument]) -> None:
        serialized = [item.model_dump(mode="json") for item in documents]
        self.runtime_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_documents(self) -> list[RAGDocument]:
        runtime_documents = self.load_runtime_rag_documents()
        runtime_ids = {item.doc_id for item in runtime_documents}
        fallback_documents = [item for item in self._base_documents if item.doc_id not in runtime_ids]
        return runtime_documents + fallback_documents


class RAGService:
    def __init__(self, provider: RuntimeRAGDocumentProvider) -> None:
        self.provider = provider
        self._documents: list[RAGDocument] = []
        self._store = SimpleRAGStore([])
        self.reload_rag_store()

    def query(
        self,
        corpus: CorpusType,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 3,
    ) -> list[RAGDocument]:
        return self._store.query(corpus, query, filters=filters, top_k=top_k)

    def explain(self, document: RAGDocument) -> dict:
        return self._store.explain(document)

    def list_documents(self) -> list[RAGDocument]:
        return list(self._documents)

    def import_documents(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        duplicate_ids = _find_duplicates([item.doc_id for item in documents])
        if duplicate_ids:
            raise ValueError(f"Duplicate doc_id in import payload: {', '.join(sorted(duplicate_ids))}")
        runtime_documents = {item.doc_id: item for item in self.provider.load_runtime_rag_documents()}
        for document in documents:
            runtime_documents[document.doc_id] = document
        merged_runtime = list(runtime_documents.values())
        self.provider.save_runtime_rag_documents(merged_runtime)
        return self.reload_rag_store()

    def reload_rag_store(self) -> list[RAGDocument]:
        self._documents = self.provider.list_documents()
        self._store = SimpleRAGStore(self._documents)
        return list(self._documents)


def validate_rag_document_payload(documents: list[RAGDocument]) -> None:
    for document in documents:
        if not document.doc_id.strip():
            raise ValueError("RAG document doc_id cannot be empty.")
        if document.corpus not in {CorpusType.POLICY, CorpusType.CASE, CorpusType.PROFILE, CorpusType.MEMORY}:
            raise ValueError(f"Unsupported corpus: {document.corpus}")
        if not document.title.strip():
            raise ValueError(f"RAG document {document.doc_id} title cannot be empty.")
        if not document.content.strip():
            raise ValueError(f"RAG document {document.doc_id} content cannot be empty.")


def _find_duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
