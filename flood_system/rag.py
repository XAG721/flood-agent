from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .models import CorpusType, DocumentRef, RAGDocument


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+", re.IGNORECASE)
ASCII_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "mean",
    "now",
    "of",
    "on",
    "right",
    "the",
    "this",
    "to",
    "what",
}
CHINESE_STOP_TOKENS = {
    "这个",
    "现在",
    "什么",
    "为何",
    "怎么",
    "情况",
    "意味着",
    "请问",
    "一下",
}
CHINESE_FRAGMENT_STOPWORDS = CHINESE_STOP_TOKENS | {"对", "和", "与", "的", "及", "在", "于"}
RECENCY_KEYS = ("updated_at", "published_at", "effective_at", "timestamp", "created_at")
CORPUS_PRIOR_BONUS = {
    CorpusType.POLICY: 0.9,
    CorpusType.MEMORY: 0.45,
    CorpusType.PROFILE: 0.55,
    CorpusType.CASE: 0.35,
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _ngrams(value: str, n: int) -> set[str]:
    if len(value) < n:
        return set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens: set[str] = set()
    for fragment in TOKEN_PATTERN.findall(normalized):
        if re.fullmatch(r"[a-z0-9]+", fragment):
            if len(fragment) <= 1 or fragment in ASCII_STOPWORDS:
                continue
            tokens.add(fragment)
            if len(fragment) >= 5:
                tokens.update(_ngrams(fragment, 3))
            continue

        if fragment in CHINESE_STOP_TOKENS or len(fragment) <= 1:
            continue
        tokens.add(fragment)
        tokens.update(_ngrams(fragment, 2))
        tokens.update(_ngrams(fragment, 3))
    return tokens


def _query_fragments(text: str) -> list[str]:
    normalized = _normalize_text(text)
    fragments: list[str] = []
    for fragment in TOKEN_PATTERN.findall(normalized):
        if re.fullmatch(r"[a-z0-9]+", fragment):
            if len(fragment) <= 2 or fragment in ASCII_STOPWORDS:
                continue
            fragments.append(fragment)
            continue

        if fragment in CHINESE_STOP_TOKENS or len(fragment) <= 1:
            continue
        cleaned = fragment
        for stopword in CHINESE_FRAGMENT_STOPWORDS:
            cleaned = cleaned.replace(stopword, " ")
        for piece in cleaned.split():
            piece = piece.strip()
            if len(piece) <= 1:
                continue
            fragments.append(piece)
            max_ngram = min(6, len(piece))
            for n in range(2, max_ngram + 1):
                fragments.extend(sorted(_ngrams(piece, n)))
    return list(dict.fromkeys(fragments))


class SimpleRAGStore:
    def __init__(self, documents: list[RAGDocument]) -> None:
        self._documents = documents

    def query(
        self,
        corpus: CorpusType,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 3,
    ) -> list[RAGDocument]:
        filters = filters or {}
        query_tokens = _tokenize(query)
        query_fragments = _query_fragments(query)
        ranked: list[tuple[float, RAGDocument]] = []

        for document in self._documents:
            if document.corpus != corpus:
                continue
            if not self._matches_filters(document, filters):
                continue

            title_text = _normalize_text(document.title)
            content_text = _normalize_text(document.content)
            metadata_text = self._metadata_text(document.metadata)

            title_tokens = _tokenize(document.title)
            content_tokens = _tokenize(document.content)
            metadata_tokens = _tokenize(metadata_text)

            title_hits = sorted(query_tokens & title_tokens)
            content_hits = sorted(query_tokens & content_tokens)
            metadata_hits = sorted(query_tokens & metadata_tokens)

            title_score = len(title_hits) * 3.2
            content_score = len(content_hits) * 1.4
            metadata_score = len(metadata_hits) * 1.8
            phrase_score = self._phrase_bonus(query_fragments, title_text, content_text, metadata_text)
            filter_score = 0.75 * len(filters)
            corpus_bonus = CORPUS_PRIOR_BONUS.get(document.corpus, 0.0)
            recency_multiplier, matched_timestamp = self._recency_multiplier(document)

            raw_score = title_score + content_score + metadata_score + phrase_score + filter_score + corpus_bonus
            final_score = round(raw_score * recency_multiplier, 4)
            if final_score <= 0:
                continue

            explain = {
                "matched_terms": {
                    "title": title_hits[:8],
                    "content": content_hits[:8],
                    "metadata": metadata_hits[:8],
                },
                "matched_filters": {key: str(document.metadata.get(key)) for key in filters},
                "field_scores": {
                    "title": round(title_score, 4),
                    "content": round(content_score, 4),
                    "metadata": round(metadata_score, 4),
                    "phrase_bonus": round(phrase_score, 4),
                    "filter_bonus": round(filter_score, 4),
                    "corpus_prior": round(corpus_bonus, 4),
                },
                "matched_fragments": [fragment for fragment in query_fragments if fragment in title_text or fragment in content_text or fragment in metadata_text],
                "time_decay": round(recency_multiplier, 4),
                "matched_timestamp": matched_timestamp,
                "final_score": final_score,
            }
            ranked.append((final_score, self._with_explain(document, explain)))

        ranked.sort(key=lambda item: (-item[0], item[1].doc_id))
        return [document for _, document in ranked[:top_k]]

    def get_by_ids(self, doc_ids: list[str]) -> list[RAGDocument]:
        wanted = {str(item) for item in doc_ids}
        if not wanted:
            return []
        return [document for document in self._documents if document.doc_id in wanted]

    @staticmethod
    def cite(document: RAGDocument) -> DocumentRef:
        return DocumentRef(
            doc_id=document.doc_id,
            title=document.title,
            excerpt=document.content[:120],
            retrieval_explain=SimpleRAGStore.explain(document),
        )

    @staticmethod
    def explain(document: RAGDocument) -> dict[str, Any]:
        explain = document.metadata.get("_retrieval_explain", {})
        return dict(explain) if isinstance(explain, dict) else {}

    @staticmethod
    def _matches_filters(document: RAGDocument, filters: dict[str, str]) -> bool:
        for key, expected in filters.items():
            actual = document.metadata.get(key)
            if actual is None:
                return False
            if str(actual).lower() != str(expected).lower():
                return False
        return True

    @staticmethod
    def _with_explain(document: RAGDocument, explain: dict[str, Any]) -> RAGDocument:
        metadata = dict(document.metadata)
        metadata["_retrieval_explain"] = explain
        return document.model_copy(update={"metadata": metadata})

    @staticmethod
    def _metadata_text(metadata: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in metadata.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float)):
                parts.append(str(value))
        return " ".join(parts)

    @staticmethod
    def _phrase_bonus(query_fragments: list[str], title_text: str, content_text: str, metadata_text: str) -> float:
        score = 0.0
        for fragment in query_fragments:
            if fragment in title_text:
                score += 3.0
            elif fragment in metadata_text:
                score += 1.4
            elif fragment in content_text:
                score += 1.1
        return score

    @staticmethod
    def _recency_multiplier(document: RAGDocument) -> tuple[float, str | None]:
        timestamp = SimpleRAGStore._extract_timestamp(document.metadata)
        if timestamp is None:
            return 1.0, None

        age_days = max(0.0, (datetime.now(UTC) - timestamp).total_seconds() / 86400.0)
        if age_days <= 1:
            return 1.12, timestamp.isoformat()
        if age_days <= 7:
            return 1.08, timestamp.isoformat()
        if age_days <= 30:
            return 1.0, timestamp.isoformat()
        if age_days <= 180:
            return 0.92, timestamp.isoformat()
        if age_days <= 365:
            return 0.86, timestamp.isoformat()
        return 0.8, timestamp.isoformat()

    @staticmethod
    def _extract_timestamp(metadata: dict[str, Any]) -> datetime | None:
        for key in RECENCY_KEYS:
            raw = metadata.get(key)
            if raw in (None, ""):
                continue
            if isinstance(raw, datetime):
                return raw.astimezone(UTC) if raw.tzinfo else raw.replace(tzinfo=UTC)
            if isinstance(raw, str):
                normalized = raw.strip().replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(normalized)
                except ValueError:
                    continue
                return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return None
