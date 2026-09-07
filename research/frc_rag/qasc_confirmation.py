"""Frozen QASC controlled evidence-pool preparation for external confirmation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from research.frc_rag.twowiki_confirmation import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    FrozenBgeScorer,
    _nested,
    _token_count,
    _tokenize,
    canonical_json_sha256,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


DATASET_NAME = "QASC"
SOURCE_REPOSITORY = "allenai/qasc"
SOURCE_REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
SOURCE_FILENAME = "validation-00000-of-00001.parquet"
SOURCE_URL = (
    "https://huggingface.co/datasets/allenai/qasc/resolve/"
    f"{SOURCE_REVISION}/data/{SOURCE_FILENAME}?download=true"
)
EXPECTED_SOURCE_SHA256 = (
    "d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae"
)
SOURCE_LICENSE = "CC-BY-4.0"
CANDIDATE_COUNT = 40
DISTRACTOR_COUNT = CANDIDATE_COUNT - 2
MINIMUM_ELIGIBLE_CASES = 800
ROLE_NAMES = ("condition", "attribution", "procedure", "answer", "exception")
REQUIRED_ROLES = ("procedure", "answer")


class CaseScorer(Protocol):
    def score_case(self, case: dict[str, Any]) -> dict[str, Any]: ...


def _normalize_fact(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _fact_key(text: str) -> str:
    return _normalize_fact(text).casefold()


def _fact_id(text: str) -> str:
    digest = hashlib.sha256(_fact_key(text).encode("utf-8")).hexdigest()
    return f"qasc::fact::{digest}"


def _question_type(question: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", question)
    token = match.group(1).casefold() if match else "other"
    if token not in {"what", "which", "how", "where", "when", "why", "who"}:
        token = "other"
    return f"qasc_{token}"


def _choices(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = _nested(row.get("choices")) or {}
    if not isinstance(choices, dict):
        raise ValueError("QASC choices must be an object")
    texts = [str(value) for value in (_nested(choices.get("text")) or [])]
    labels = [str(value) for value in (_nested(choices.get("label")) or [])]
    if len(texts) != len(labels) or len(texts) < 2:
        raise ValueError("QASC choices must contain aligned text and label arrays")
    return texts, labels


def _formatted_question(row: dict[str, Any]) -> str:
    provided = str(row.get("formatted_question") or "").strip()
    if provided:
        return provided
    question = str(row.get("question") or "").strip()
    texts, labels = _choices(row)
    options = " ".join(
        f"({label}) {text}" for label, text in zip(labels, texts, strict=True)
    )
    return f"{question} {options}".strip()


def _answer(row: dict[str, Any]) -> str:
    answer_key = str(row.get("answerKey") or "").strip()
    texts, labels = _choices(row)
    by_label = dict(zip(labels, texts, strict=True))
    if answer_key not in by_label:
        raise ValueError(f"QASC answer key is not in choices: {answer_key!r}")
    return by_label[answer_key]


class FrozenFactBm25Index:
    """Small deterministic BM25 index used only to choose non-gold distractors."""

    def __init__(self, facts: list[tuple[str, str]]) -> None:
        if not facts:
            raise ValueError("QASC fact corpus is empty")
        self.facts = facts
        self.documents = [_tokenize(text) for _, text in facts]
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            self.document_frequency.update(set(tokens))
        self.average_length = (
            float(np.mean([len(tokens) for tokens in self.documents])) or 1.0
        )

    def rank(self, query: str, *, excluded_ids: set[str]) -> list[str]:
        query_terms = _tokenize(query)
        document_count = len(self.documents)
        ranked: list[tuple[float, str]] = []
        for (fact_id, _), tokens in zip(self.facts, self.documents, strict=True):
            if fact_id in excluded_ids:
                continue
            frequencies = Counter(tokens)
            document_length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                term_document_count = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1
                    + (document_count - term_document_count + 0.5)
                    / (term_document_count + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * document_length / self.average_length
                )
                score += inverse_frequency * (frequency * 2.5) / denominator
            ranked.append((-score, fact_id))
        ranked.sort()
        return [fact_id for _, fact_id in ranked]


def _materialize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    case_ids = [str(row.get("id") or "").strip() for row in materialized]
    if any(not case_id for case_id in case_ids):
        raise ValueError("QASC row is missing id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("QASC rows contain duplicate ids")
    return materialized


def _fact_corpus(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    by_key: dict[str, str] = {}
    for row in rows:
        for field in ("fact1", "fact2"):
            text = _normalize_fact(row.get(field))
            if text:
                by_key.setdefault(_fact_key(text), text)
    facts = [(_fact_id(text), text) for text in by_key.values()]
    facts.sort(key=lambda item: item[0])
    if len(facts) <= CANDIDATE_COUNT:
        raise ValueError("QASC fact corpus is too small for the frozen pool")
    return facts


def _eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        fact1 = _normalize_fact(row.get("fact1"))
        fact2 = _normalize_fact(row.get("fact2"))
        if not fact1 or not fact2 or _fact_key(fact1) == _fact_key(fact2):
            continue
        if not str(row.get("question") or "").strip():
            continue
        _choices(row)
        _answer(row)
        eligible.append(row)
    if len(eligible) < MINIMUM_ELIGIBLE_CASES:
        raise ValueError(
            f"QASC has {len(eligible)} eligible cases, below {MINIMUM_ELIGIBLE_CASES}"
        )
    return eligible


def prepare_cases(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = _materialize_rows(rows)
    facts = _fact_corpus(materialized)
    fact_text = dict(facts)
    index = FrozenFactBm25Index(facts)
    cases = []
    for row in _eligible_rows(materialized):
        fact1 = _normalize_fact(row["fact1"])
        fact2 = _normalize_fact(row["fact2"])
        gold_by_id = {
            _fact_id(fact1): ["procedure"],
            _fact_id(fact2): ["answer"],
        }
        query = _formatted_question(row)
        distractor_ids = index.rank(query, excluded_ids=set(gold_by_id))[
            :DISTRACTOR_COUNT
        ]
        if len(distractor_ids) != DISTRACTOR_COUNT:
            raise ValueError("QASC hard-distractor pool is incomplete")
        candidate_ids = sorted([*gold_by_id, *distractor_ids])
        candidates = []
        for candidate_id in candidate_ids:
            text = fact_text.get(candidate_id)
            if text is None:
                if candidate_id == _fact_id(fact1):
                    text = fact1
                elif candidate_id == _fact_id(fact2):
                    text = fact2
                else:
                    raise AssertionError("QASC candidate text is missing")
            roles = gold_by_id.get(candidate_id, [])
            candidates.append(
                {
                    "id": candidate_id,
                    "text": text,
                    "source": "qasc_validation_fact_pool",
                    "metadata": {
                        "fact_sha256": candidate_id.rsplit("::", 1)[-1],
                    },
                    "gold": bool(roles),
                    "gold_roles": roles,
                    "token_count": _token_count(text),
                }
            )
        case_id = str(row["id"]).strip()
        cases.append(
            {
                "dataset": DATASET_NAME,
                "source": "official_validation_controlled_hard_pool",
                "id": case_id,
                "question": query,
                "answer": _answer(row),
                "question_type": _question_type(str(row["question"])),
                "not_answerable": False,
                "required_roles": list(REQUIRED_ROLES),
                "gold_evidence_ids": sorted(gold_by_id),
                "candidates": candidates,
            }
        )
    cases.sort(key=lambda case: str(case["id"]))
    return cases


def load_and_prepare_parquet(path: Path) -> list[dict[str, Any]]:
    actual_sha256 = sha256(path)
    if actual_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "QASC source hash mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_sha256}"
        )
    import pandas as pd

    frame = pd.read_parquet(path)
    return prepare_cases(frame.to_dict(orient="records"))


def _gold_blind_view(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key not in {"answer", "gold_evidence_ids"}
    } | {
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"gold", "gold_roles"}
            }
            for candidate in case["candidates"]
        ]
    }


class QascGoldBlindScorer:
    """Prevent the neural scorer from observing evaluation-only QASC labels."""

    def __init__(self, delegate: CaseScorer) -> None:
        self.delegate = delegate

    def score_case(self, case: dict[str, Any]) -> dict[str, Any]:
        labels = {
            str(candidate["id"]): {
                "gold": bool(candidate["gold"]),
                "gold_roles": list(candidate["gold_roles"]),
            }
            for candidate in case["candidates"]
        }
        scored = self.delegate.score_case(_gold_blind_view(case))
        scored_candidates = []
        for candidate in scored["candidates"]:
            candidate_id = str(candidate["id"])
            if candidate_id not in labels:
                raise ValueError("QASC scorer changed candidate ids")
            scored_candidates.append({**candidate, **labels[candidate_id]})
        if {str(item["id"]) for item in scored_candidates} != set(labels):
            raise ValueError("QASC scorer dropped candidates")
        return {
            **scored,
            "answer": case["answer"],
            "gold_evidence_ids": list(case["gold_evidence_ids"]),
            "candidates": scored_candidates,
        }


def score_prepared_cases(
    prepared_path: Path,
    output_path: Path,
    *,
    scorer: FrozenBgeScorer,
    progress: Any = None,
) -> int:
    return score_cases_resumable(
        prepared_path,
        output_path,
        scorer=QascGoldBlindScorer(scorer),
        progress=progress,
    )


def preparation_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("prepared QASC source is empty")
    candidate_counts = [len(case["candidates"]) for case in cases]
    gold_counts = [len(case["gold_evidence_ids"]) for case in cases]
    unique_fact_ids = {
        candidate["id"] for case in cases for candidate in case["candidates"]
    }
    return {
        "case_count": len(cases),
        "selected_case_ids_sha256": canonical_json_sha256(
            [case["id"] for case in cases]
        ),
        "candidate_pool_rule": (
            "two official gold facts plus top-38 non-gold validation facts by "
            "deterministic BM25 over formatted question; candidate-id tie break"
        ),
        "candidate_count": {
            "total": sum(candidate_counts),
            "minimum": min(candidate_counts),
            "maximum": max(candidate_counts),
            "mean": round(float(np.mean(candidate_counts)), 6),
        },
        "gold_evidence_count": {
            "total": sum(gold_counts),
            "minimum": min(gold_counts),
            "maximum": max(gold_counts),
            "mean": round(float(np.mean(gold_counts)), 6),
        },
        "unique_candidate_fact_count": len(unique_fact_ids),
        "question_types": dict(
            sorted(Counter(case["question_type"] for case in cases).items())
        ),
    }


__all__ = [
    "CANDIDATE_COUNT",
    "DATASET_NAME",
    "EMBEDDING_MODEL",
    "EMBEDDING_REVISION",
    "EXPECTED_SOURCE_SHA256",
    "FrozenBgeScorer",
    "QascGoldBlindScorer",
    "RERANKER_MODEL",
    "RERANKER_REVISION",
    "SOURCE_FILENAME",
    "SOURCE_LICENSE",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_URL",
    "load_and_prepare_parquet",
    "preparation_summary",
    "prepare_cases",
    "read_jsonl",
    "score_prepared_cases",
    "sha256",
    "write_jsonl",
]
