from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from dataclasses import dataclass
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
CORPUS_TRUST_PRIOR = {
    CorpusType.POLICY: 0.92,
    CorpusType.PROFILE: 0.8,
    CorpusType.CASE: 0.72,
    CorpusType.MEMORY: 0.68,
}
TRUST_LABEL_PRIOR = {
    "official": 0.96,
    "verified": 0.92,
    "correct": 0.9,
    "primary": 0.88,
    "internal": 0.84,
    "community": 0.68,
    "unverified": 0.42,
    "noise": 0.25,
    "misinfo": 0.08,
    "misinformation": 0.08,
    "rumor": 0.12,
}
DOC_TYPE_TRUST_PRIOR = {
    "policy": 0.94,
    "plan": 0.9,
    "profile": 0.82,
    "case": 0.74,
    "memory": 0.7,
    "social": 0.35,
    "rumor": 0.12,
}
CONFLICT_MARKERS = {
    "no",
    "not",
    "never",
    "without",
    "closed",
    "blocked",
    "cancelled",
    "disabled",
    "fail",
    "failed",
    "false",
    "deny",
    "denied",
    "stop",
    "suspend",
    "停",
    "否",
    "不",
    "未",
    "无",
    "关闭",
    "封闭",
    "中断",
    "取消",
}
OBJECT_SCOPE_TERMS = {
    "下穿通道",
    "地下空间",
    "学校",
    "医院",
    "养老院",
    "地铁站",
    "居民区",
    "老年人",
    "老人",
    "underpass",
    "school",
    "hospital",
    "elder",
    "metro",
}


@dataclass(frozen=True)
class _EvidenceSlot:
    slot_id: str
    text: str
    tokens: set[str]


@dataclass(frozen=True)
class EvidenceSelectionPolicy:
    """Frozen, auditable weights for FRC evidence-set selection.

    Defaults carry forward the original field/trust/applicability coefficients
    and make the previously hard role constraint explicit in ranking, while
    exposing all dimensions for one-factor sensitivity experiments.
    ``conflict_threshold`` controls when a detected conflict
    score starts contributing to the penalty; it does not suppress conflict
    metadata or turn a selected conflict into a safe result.
    """

    field_weight: float = 4.0
    role_weight: float = 1.0
    trust_weight: float = 1.4
    applicability_weight: float = 1.5
    novelty_weight: float = 1.0
    support_weight: float = 1.0
    conflict_penalty_weight: float = 2.4
    cost_penalty_weight: float = 0.7
    conflict_threshold: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.field_weight,
            self.role_weight,
            self.trust_weight,
            self.applicability_weight,
            self.novelty_weight,
            self.support_weight,
            self.conflict_penalty_weight,
            self.cost_penalty_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("evidence selection weights must be finite and non-negative")
        if not 0.0 <= self.conflict_threshold <= 1.0:
            raise ValueError("conflict_threshold must be between 0 and 1")

    def as_dict(self) -> dict[str, float]:
        return {
            "field_weight": self.field_weight,
            "role_weight": self.role_weight,
            "trust_weight": self.trust_weight,
            "applicability_weight": self.applicability_weight,
            "novelty_weight": self.novelty_weight,
            "support_weight": self.support_weight,
            "conflict_penalty_weight": self.conflict_penalty_weight,
            "cost_penalty_weight": self.cost_penalty_weight,
            "conflict_threshold": self.conflict_threshold,
        }


@dataclass(frozen=True)
class _EvidenceCandidate:
    document: RAGDocument
    doc_tokens: set[str]
    token_cost: int
    trust_score: float
    applicability_score: float
    support_by_slot: dict[str, float]

    @property
    def covered_slot_ids(self) -> set[str]:
        return {slot_id for slot_id, score in self.support_by_slot.items() if score >= 0.18}

    @property
    def support_score(self) -> float:
        if not self.support_by_slot:
            return 0.0
        return max(self.support_by_slot.values())


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

    def query_evidence_set(
        self,
        corpus: CorpusType,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int = 3,
        candidate_k: int = 20,
        token_budget: int | None = 1200,
        slots: list[str] | None = None,
        required_roles: list[str] | None = None,
        selection_policy: EvidenceSelectionPolicy | None = None,
    ) -> list[RAGDocument]:
        if top_k <= 0:
            return []
        if token_budget is not None and token_budget <= 0:
            return []

        candidates = self.query(
            corpus,
            query,
            filters=filters,
            top_k=max(top_k, candidate_k),
        )
        if not candidates:
            return []

        evidence_slots = self._build_evidence_slots(query, slots=slots)
        policy = selection_policy or EvidenceSelectionPolicy()
        evidence_candidates = [
            self._build_evidence_candidate(document, evidence_slots, query=query)
            for document in candidates
        ]
        selected = self._select_evidence_candidates(
            evidence_candidates,
            evidence_slots,
            top_k=top_k,
            token_budget=token_budget,
            required_roles=set(required_roles or []),
            policy=policy,
        )
        return self._attach_evidence_selection(
            selected,
            evidence_slots,
            token_budget=token_budget,
            policy=policy,
        )

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
    def _build_evidence_slots(query: str, slots: list[str] | None = None) -> list[_EvidenceSlot]:
        if slots:
            raw_slots = slots
        else:
            raw_slots = [
                part.strip()
                for part in re.split(
                    r"[?？,，;；、/]|(?:\s+and\s+)|(?:\s+or\s+)|(?:\s+with\s+)|以及|并且|同时|和",
                    query,
                    flags=re.IGNORECASE,
                )
                if part.strip()
            ]
            if len(raw_slots) < 2:
                fragments = [fragment for fragment in _query_fragments(query) if len(fragment) >= 2]
                raw_slots = fragments[:12] or [query]

        normalized_slots: list[str] = []
        seen: set[str] = set()
        for slot in raw_slots:
            cleaned = _normalize_text(str(slot))
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized_slots.append(str(slot).strip())
            if len(normalized_slots) >= 12:
                break

        if not normalized_slots:
            normalized_slots = [query]

        evidence_slots: list[_EvidenceSlot] = []
        for index, slot in enumerate(normalized_slots, start=1):
            evidence_slots.append(
                _EvidenceSlot(
                    slot_id=f"slot_{index}",
                    text=slot,
                    tokens=_tokenize(slot) or _tokenize(query),
                )
            )
        return evidence_slots

    @classmethod
    def _build_evidence_candidate(
        cls, document: RAGDocument, slots: list[_EvidenceSlot], *, query: str
    ) -> _EvidenceCandidate:
        metadata_text = cls._metadata_text(document.metadata)
        title_tokens = _tokenize(document.title)
        content_tokens = _tokenize(document.content)
        metadata_tokens = _tokenize(metadata_text)
        doc_tokens = title_tokens | content_tokens | metadata_tokens
        support_by_slot = {
            slot.slot_id: cls._slot_support_score(slot, title_tokens, content_tokens, metadata_tokens, document)
            for slot in slots
        }
        return _EvidenceCandidate(
            document=document,
            doc_tokens=doc_tokens,
            token_cost=cls._token_cost(document),
            trust_score=cls._trust_score(document),
            applicability_score=cls._applicability_score(query, document),
            support_by_slot=support_by_slot,
        )

    @classmethod
    def _select_evidence_candidates(
        cls,
        candidates: list[_EvidenceCandidate],
        slots: list[_EvidenceSlot],
        *,
        top_k: int,
        token_budget: int | None,
        required_roles: set[str],
        policy: EvidenceSelectionPolicy,
    ) -> list[tuple[_EvidenceCandidate, dict[str, float]]]:
        selected: list[tuple[_EvidenceCandidate, dict[str, float]]] = []
        selected_candidates: list[_EvidenceCandidate] = []
        remaining = list(candidates)
        covered_slots: set[str] = set()
        covered_roles: set[str] = set()
        used_budget = 0

        while remaining and len(selected) < top_k:
            scored: list[tuple[float, _EvidenceCandidate, dict[str, float]]] = []
            for candidate in remaining:
                if token_budget is not None and used_budget + candidate.token_cost > token_budget:
                    continue

                coverage_gain = sum(
                    candidate.support_by_slot[slot_id]
                    for slot_id in candidate.covered_slot_ids
                    if slot_id not in covered_slots
                )
                coverage_score = coverage_gain / max(1, len(slots))
                role_values = candidate.document.metadata.get("evidence_roles", [])
                if isinstance(role_values, str):
                    role_values = [role_values]
                candidate_roles = {str(item) for item in role_values}
                role_gain = len((candidate_roles & required_roles) - covered_roles)
                role_score = role_gain / max(1, len(required_roles))
                novelty_score = cls._novelty_score(candidate, selected_candidates)
                raw_conflict_score = cls._conflict_score(candidate, selected_candidates)
                conflict_score = (
                    raw_conflict_score
                    if raw_conflict_score > 0 and raw_conflict_score >= policy.conflict_threshold
                    else 0.0
                )
                cost_score = candidate.token_cost / max(1, token_budget or candidate.token_cost)
                total_score = (
                    policy.field_weight * coverage_score
                    + policy.role_weight * role_score
                    + policy.trust_weight * candidate.trust_score
                    + policy.applicability_weight * candidate.applicability_score
                    + policy.novelty_weight * novelty_score
                    + policy.support_weight * candidate.support_score
                    - policy.conflict_penalty_weight * conflict_score
                    - policy.cost_penalty_weight * cost_score
                )
                score_terms = {
                    "total_score": round(total_score, 4),
                    "coverage_score": round(coverage_score, 4),
                    "coverage_gain": round(coverage_gain, 4),
                    "role_score": round(role_score, 4),
                    "role_gain": round(float(role_gain), 4),
                    "trust_score": round(candidate.trust_score, 4),
                    "applicability_score": round(candidate.applicability_score, 4),
                    "novelty_score": round(novelty_score, 4),
                    "support_score": round(candidate.support_score, 4),
                    "conflict_score": round(conflict_score, 4),
                    "raw_conflict_score": round(raw_conflict_score, 4),
                    "cost_score": round(cost_score, 4),
                }
                scored.append((total_score, candidate, score_terms))

            if not scored:
                break

            scored.sort(key=lambda item: (-item[0], -item[1].trust_score, item[1].token_cost, item[1].document.doc_id))
            _, best_candidate, best_terms = scored[0]
            selected.append((best_candidate, best_terms))
            selected_candidates.append(best_candidate)
            covered_slots.update(best_candidate.covered_slot_ids)
            role_values = best_candidate.document.metadata.get("evidence_roles", [])
            if isinstance(role_values, str):
                role_values = [role_values]
            covered_roles.update(str(item) for item in role_values)
            used_budget += best_candidate.token_cost
            remaining = [candidate for candidate in remaining if candidate.document.doc_id != best_candidate.document.doc_id]
            # FRC-RAG selects the smallest sufficient evidence set. Continuing
            # to fill top_k after every requested field is covered introduces
            # unsupported citations without adding task-field support.
            if required_roles and len(covered_slots) == len(slots) and required_roles.issubset(covered_roles):
                break

        return selected

    @classmethod
    def _attach_evidence_selection(
        cls,
        selected: list[tuple[_EvidenceCandidate, dict[str, float]]],
        slots: list[_EvidenceSlot],
        *,
        token_budget: int | None,
        policy: EvidenceSelectionPolicy,
    ) -> list[RAGDocument]:
        selected_token_cost = sum(candidate.token_cost for candidate, _ in selected)
        selected_slot_ids = set().union(*(candidate.covered_slot_ids for candidate, _ in selected)) if selected else set()
        documents: list[RAGDocument] = []

        for rank, (candidate, score_terms) in enumerate(selected, start=1):
            document = candidate.document
            covered_slots = [
                {
                    "slot_id": slot.slot_id,
                    "text": slot.text,
                    "support_score": round(candidate.support_by_slot.get(slot.slot_id, 0.0), 4),
                }
                for slot in slots
                if slot.slot_id in candidate.covered_slot_ids
            ]
            evidence_selection = {
                "mode": "budget_aware_trustworthy_set_selection",
                "selection_rank": rank,
                "selection_score": score_terms["total_score"],
                "covered_slots": covered_slots,
                "all_slots": [
                    {
                        "slot_id": slot.slot_id,
                        "text": slot.text,
                        "covered_by_selected_set": slot.slot_id in selected_slot_ids,
                    }
                    for slot in slots
                ],
                "trust_score": round(candidate.trust_score, 4),
                "applicability_score": round(candidate.applicability_score, 4),
                "support_score": round(candidate.support_score, 4),
                "token_cost": candidate.token_cost,
                "selected_token_cost": selected_token_cost,
                "token_budget": token_budget,
                "selection_policy": policy.as_dict(),
                "score_terms": score_terms,
                "source": cls._selection_source_summary(document),
            }
            explain = cls.explain(document)
            explain["evidence_selection"] = evidence_selection
            metadata = dict(document.metadata)
            metadata["_evidence_selection"] = evidence_selection
            metadata["_retrieval_explain"] = explain
            documents.append(document.model_copy(update={"metadata": metadata}))

        return documents

    @staticmethod
    def _slot_support_score(
        slot: _EvidenceSlot,
        title_tokens: set[str],
        content_tokens: set[str],
        metadata_tokens: set[str],
        document: RAGDocument,
    ) -> float:
        if not slot.tokens:
            return 0.0
        title_hits = slot.tokens & title_tokens
        content_hits = slot.tokens & content_tokens
        metadata_hits = slot.tokens & metadata_tokens
        weighted_hits = 2.2 * len(title_hits) + 1.25 * len(content_hits) + 1.6 * len(metadata_hits)
        lexical_score = min(1.0, weighted_hits / max(1.0, len(slot.tokens) * 2.2))
        retrieval_score = float(SimpleRAGStore.explain(document).get("final_score", 0.0) or 0.0)
        retrieval_norm = retrieval_score / (retrieval_score + 12.0) if retrieval_score > 0 else 0.0
        return round(0.8 * lexical_score + 0.2 * retrieval_norm, 4)

    @classmethod
    def _trust_score(cls, document: RAGDocument) -> float:
        score = CORPUS_TRUST_PRIOR.get(document.corpus, 0.65)
        metadata = document.metadata

        for key in ("trust_score", "source_trust", "reliability_score"):
            numeric_score = cls._coerce_score(metadata.get(key))
            if numeric_score is not None:
                score = (score + numeric_score) / 2.0

        for key in ("source_tier", "source_label", "reliability", "source_reliability"):
            mapped = cls._mapped_prior(metadata.get(key), TRUST_LABEL_PRIOR)
            if mapped is None:
                continue
            score = min(score, mapped) if mapped < 0.5 else max(score, mapped)

        doc_type_score = cls._mapped_prior(metadata.get("doc_type"), DOC_TYPE_TRUST_PRIOR)
        if doc_type_score is not None:
            score = (score + doc_type_score) / 2.0

        if metadata.get("verified") is True:
            score = max(score, 0.92)
        if metadata.get("verified") is False:
            score = min(score, 0.45)

        return round(cls._clamp(score), 4)

    @staticmethod
    def _applicability_score(query: str, document: RAGDocument) -> float:
        query_text = _normalize_text(query)
        document_text = _normalize_text(f"{document.title} {document.content}")
        query_scopes = {term for term in OBJECT_SCOPE_TERMS if term in query_text}
        document_scopes = {term for term in OBJECT_SCOPE_TERMS if term in document_text}
        if not query_scopes or not document_scopes:
            return 0.5
        return 1.0 if query_scopes & document_scopes else 0.0

    @staticmethod
    def _novelty_score(candidate: _EvidenceCandidate, selected: list[_EvidenceCandidate]) -> float:
        if not selected:
            return 1.0
        max_overlap = 0.0
        for selected_candidate in selected:
            union_size = len(candidate.doc_tokens | selected_candidate.doc_tokens)
            if union_size == 0:
                continue
            overlap = len(candidate.doc_tokens & selected_candidate.doc_tokens) / union_size
            max_overlap = max(max_overlap, overlap)
        return round(1.0 - max_overlap, 4)

    @classmethod
    def _conflict_score(cls, candidate: _EvidenceCandidate, selected: list[_EvidenceCandidate]) -> float:
        score = cls._coerce_score(candidate.document.metadata.get("conflict_risk_score")) or 0.0
        if cls._mapped_prior(candidate.document.metadata.get("source_label"), TRUST_LABEL_PRIOR) in {0.08, 0.12, 0.25}:
            score = max(score, 0.35)

        candidate_stance = cls._metadata_string(candidate.document.metadata.get("stance"))
        candidate_has_marker = cls._has_conflict_marker(candidate.document)
        for selected_candidate in selected:
            if cls._explicit_conflict(candidate.document, selected_candidate.document):
                score = max(score, 1.0)
                continue

            shared_slots = candidate.covered_slot_ids & selected_candidate.covered_slot_ids
            selected_stance = cls._metadata_string(selected_candidate.document.metadata.get("stance"))
            if shared_slots and candidate_stance and selected_stance and candidate_stance != selected_stance:
                score = max(score, 0.8)

            selected_has_marker = cls._has_conflict_marker(selected_candidate.document)
            shared_token_ratio = cls._token_overlap_ratio(candidate.doc_tokens, selected_candidate.doc_tokens)
            low_trust_pair = candidate.trust_score < 0.6 or selected_candidate.trust_score < 0.6
            if (
                shared_slots
                and shared_token_ratio >= 0.18
                and candidate_has_marker != selected_has_marker
                and low_trust_pair
            ):
                score = max(score, 0.45)

        return round(cls._clamp(score), 4)

    @classmethod
    def _explicit_conflict(cls, left: RAGDocument, right: RAGDocument) -> bool:
        return right.doc_id in cls._metadata_values(left.metadata.get("conflicts_with")) or left.doc_id in cls._metadata_values(
            right.metadata.get("conflicts_with")
        )

    @classmethod
    def _has_conflict_marker(cls, document: RAGDocument) -> bool:
        text = _normalize_text(f"{document.title} {document.content} {cls._metadata_text(document.metadata)}")
        return any(marker in text for marker in CONFLICT_MARKERS)

    @staticmethod
    def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
        union_size = len(left | right)
        if union_size == 0:
            return 0.0
        return len(left & right) / union_size

    @classmethod
    def _token_cost(cls, document: RAGDocument) -> int:
        explicit = cls._coerce_float(document.metadata.get("token_cost"))
        if explicit is not None and explicit > 0:
            return max(1, int(round(explicit)))

        text = f"{document.title} {document.content} {cls._metadata_text(document.metadata)}"
        ascii_words = re.findall(r"[a-z0-9]+", text, re.IGNORECASE)
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        other_chars = len(re.sub(r"[a-z0-9\s\u4e00-\u9fff]", "", text, flags=re.IGNORECASE))
        return max(1, int(round(len(ascii_words) * 1.25 + len(cjk_chars) / 1.6 + other_chars / 4.0)))

    @staticmethod
    def _selection_source_summary(document: RAGDocument) -> dict[str, Any]:
        metadata = document.metadata
        return {
            "corpus": document.corpus.value,
            "source": metadata.get("source"),
            "source_tier": metadata.get("source_tier"),
            "source_label": metadata.get("source_label"),
            "doc_type": metadata.get("doc_type"),
        }

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return SimpleRAGStore._clamp(float(value))
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return SimpleRAGStore._clamp(float(stripped))
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _mapped_prior(value: Any, mapping: dict[str, float]) -> float | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        return mapping.get(key)

    @staticmethod
    def _metadata_string(value: Any) -> str:
        return str(value).strip().lower() if value not in (None, "") else ""

    @staticmethod
    def _metadata_values(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        return {str(value)}

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return min(maximum, max(minimum, value))

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
