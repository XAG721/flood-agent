"""Frozen SciFact transfer test for the v41 dynamic atomic-role selector.

The adapter keeps corpus retrieval and neural scoring blind to SciFact evidence.
Gold document identifiers are reconstructed from the release only after the
complete score cache exists.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hover_dynamic_atomic_roles import (
    BASELINES,
    BUDGETS,
    DYNAMIC_RANK,
    METHODS,
    STATIC_RANK,
    FrozenDynamicHoVerScorer,
    merge_scored_candidates,
    select_candidates,
)
from research.frc_rag.hover_verification_roles import _chunk_document
from research.frc_rag.rgb_cost_aware_frc import sha256


SCHEMA_VERSION = "frc-scifact-dynamic-atomic-roles-v42"
EXPERIMENT_ID = "FRC-SCIFACT-DYNAMIC-ATOMIC-ROLES-V42"
DATASET_ID = "scifact_claims_dev_v42"
CAPABILITY = "scientific_claim_document_evidence_selection"
PROTOCOL_SHA256 = (
    "62987d6430cef9ef91c0de1f15f8f192749f5ab6b5341117da99933d3b7892ab"
)
INHERITED_V41_PROTOCOL_SHA256 = (
    "42486a98c01fd0b9663d3eb9bd0de63634d91605eed06060ac1236ad26997e86"
)
EXCLUDED_DOCUMENTATION_EXAMPLES = frozenset({3, 123, 263})
CANDIDATE_DOCUMENTS = 20
MINIMUM_PRIMARY_CASES = 100
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260802
MINIMUM_SAFETY_STRATUM_CASES = 30

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN_CACHE_KEYS = {
    "cited_doc_ids",
    "doc_id",
    "document_id",
    "evidence",
    "evidence_label",
    "gold",
    "gold_document_ids",
    "gold_documents",
    "label",
    "sentences",
    "title",
}


def _normalize(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _tokens(value: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(value)]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_forbidden_cache_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in {
                "gold_fields_visible_to_generator",
                "gold_fields_visible_to_scorer",
            }:
                if child is not False:
                    return True
            elif normalized in _FORBIDDEN_CACHE_KEYS or normalized.startswith(
                "gold_"
            ):
                return True
            if _contains_forbidden_cache_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_cache_key(item) for item in value)
    return False


def read_source_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"SciFact JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _integer_id(value: Any, *, kind: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"SciFact {kind} id is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ValueError(f"SciFact {kind} id is not an integer")
    if result < 0:
        raise ValueError(f"SciFact {kind} id is negative")
    return result


def parse_corpus(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        doc_id = _integer_id(row.get("doc_id"), kind="document")
        if doc_id in seen:
            raise ValueError(f"duplicate SciFact document id: {doc_id}")
        seen.add(doc_id)
        title = _normalize(row.get("title"))
        abstract = row.get("abstract")
        if not isinstance(abstract, list) or not all(
            isinstance(sentence, str) for sentence in abstract
        ):
            raise ValueError(f"SciFact document {doc_id} has invalid abstract")
        text = _normalize(" ".join([title, *abstract]))
        documents.append({"doc_id": doc_id, "text": text})
    if not documents:
        raise ValueError("SciFact corpus is empty")
    return sorted(documents, key=lambda row: int(row["doc_id"]))


def _claim_evidence(row: dict[str, Any]) -> tuple[set[int], set[str]]:
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("SciFact claim evidence is not an object")
    document_ids: set[int] = set()
    labels: set[str] = set()
    for raw_doc_id, evidence_annotations in evidence.items():
        doc_id = _integer_id(raw_doc_id, kind="evidence document")
        if not isinstance(evidence_annotations, list):
            raise ValueError("SciFact evidence annotations are not an array")
        valid_annotation = False
        for annotation in evidence_annotations:
            if not isinstance(annotation, dict):
                raise ValueError("SciFact evidence annotation is not an object")
            label = str(annotation.get("label", "")).strip().upper()
            sentences = annotation.get("sentences")
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"unsupported SciFact evidence label: {label!r}")
            if not isinstance(sentences, list) or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in sentences
            ):
                raise ValueError("SciFact evidence sentences are invalid")
            valid_annotation = True
            labels.add(label)
        if valid_annotation:
            document_ids.add(doc_id)
    return document_ids, labels


def eligible_claims(
    rows: Iterable[dict[str, Any]],
    *,
    corpus_ids: set[int],
    minimum_cases: int = MINIMUM_PRIMARY_CASES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    excluded_examples = 0
    empty_evidence = 0
    total = 0
    for row in rows:
        total += 1
        claim_id = _integer_id(row.get("id"), kind="claim")
        if claim_id in seen:
            raise ValueError(f"duplicate SciFact claim id: {claim_id}")
        seen.add(claim_id)
        claim = _normalize(row.get("claim"))
        if not claim:
            raise ValueError(f"SciFact claim {claim_id} is empty")
        evidence_ids, labels = _claim_evidence(row)
        if claim_id in EXCLUDED_DOCUMENTATION_EXAMPLES:
            excluded_examples += 1
            continue
        if not evidence_ids:
            empty_evidence += 1
            continue
        missing = sorted(evidence_ids - corpus_ids)
        if missing:
            raise ValueError(
                f"SciFact claim {claim_id} references missing documents: {missing[:3]}"
            )
        if not labels:
            raise ValueError(f"SciFact claim {claim_id} has no evidence label")
        selected.append({"id": claim_id, "claim": claim, "source": row})
    selected.sort(key=lambda row: int(row["id"]))
    if len(selected) < minimum_cases:
        raise ValueError(
            f"SciFact has {len(selected)} eligible claims, below {minimum_cases}"
        )
    return selected, {
        "claims_total": total,
        "documentation_examples_excluded": excluded_examples,
        "empty_evidence_structurally_non_evaluable": empty_evidence,
        "primary_eligible": len(selected),
    }


class CorpusBM25Index:
    """Deterministic corpus-wide BM25 with no gold-aware filtering."""

    def __init__(self, documents: Sequence[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("SciFact BM25 corpus is empty")
        self.doc_ids = [int(row["doc_id"]) for row in documents]
        if self.doc_ids != sorted(self.doc_ids) or len(self.doc_ids) != len(
            set(self.doc_ids)
        ):
            raise ValueError("SciFact BM25 documents must have sorted unique IDs")
        self.text_by_id = {
            int(row["doc_id"]): str(row["text"]) for row in documents
        }
        self.length_by_id: dict[int, int] = {}
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for row in documents:
            doc_id = int(row["doc_id"])
            frequencies = Counter(_tokens(str(row["text"])))
            self.length_by_id[doc_id] = sum(frequencies.values())
            document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                postings[term].append((doc_id, frequency))
        self.postings = dict(postings)
        self.document_frequency = document_frequency
        self.document_count = len(self.doc_ids)
        self.average_length = (
            float(np.mean(list(self.length_by_id.values()))) or 1.0
        )

    def rank(self, query: str, *, depth: int = CANDIDATE_DOCUMENTS) -> list[int]:
        if depth <= 0:
            raise ValueError("SciFact BM25 depth must be positive")
        scores: defaultdict[int, float] = defaultdict(float)
        for term in _tokens(query):
            document_count = self.document_frequency.get(term, 0)
            if not document_count:
                continue
            inverse_frequency = math.log(
                1
                + (self.document_count - document_count + 0.5)
                / (document_count + 0.5)
            )
            for doc_id, frequency in self.postings[term]:
                denominator = frequency + 1.5 * (
                    1
                    - 0.75
                    + 0.75
                    * self.length_by_id[doc_id]
                    / self.average_length
                )
                scores[doc_id] += (
                    inverse_frequency * (frequency * 2.5) / denominator
                )
        return sorted(self.doc_ids, key=lambda doc_id: (-scores[doc_id], doc_id))[
            :depth
        ]


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"total": 0, "minimum": 0, "mean": 0.0, "maximum": 0}
    return {
        "total": int(sum(values)),
        "minimum": int(min(values)),
        "mean": round(float(np.mean(values)), 6),
        "maximum": int(max(values)),
    }


def prepare_blind_cases(
    corpus_rows: Iterable[dict[str, Any]],
    claim_rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    minimum_cases: int = MINIMUM_PRIMARY_CASES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = parse_corpus(corpus_rows)
    corpus_ids = {int(row["doc_id"]) for row in documents}
    claims, claim_census = eligible_claims(
        claim_rows,
        corpus_ids=corpus_ids,
        minimum_cases=minimum_cases,
    )
    index = CorpusBM25Index(documents)
    prepared: list[dict[str, Any]] = []
    document_counts: list[int] = []
    chunk_counts: list[int] = []
    token_counts: list[int] = []
    for claim in claims:
        case_id = f"scifact-v42::{int(claim['id'])}"
        ranked_doc_ids = index.rank(str(claim["claim"]))
        candidates: list[dict[str, Any]] = []
        for rank, doc_id in enumerate(ranked_doc_ids):
            source_id = f"s{rank:04d}"
            candidates.extend(
                _chunk_document(
                    case_id=case_id,
                    source_id=source_id,
                    text=index.text_by_id[doc_id],
                    tokenizer=tokenizer,
                )
            )
        if not candidates:
            raise ValueError(f"SciFact claim {claim['id']} has no candidate chunks")
        case = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "id": case_id,
            "query": str(claim["claim"]),
            "candidates": candidates,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_cache_key(case):
            raise AssertionError("gold or source identity leaked into SciFact cache")
        prepared.append(case)
        document_counts.append(len({row["source_id"] for row in candidates}))
        chunk_counts.append(len(candidates))
        token_counts.extend(int(row["token_count"]) for row in candidates)
    summary = {
        **claim_census,
        "corpus_documents": len(documents),
        "valid_rows": len(prepared),
        "case_ids_sha256": canonical_json_sha256(
            [str(case["id"]) for case in prepared]
        ),
        "candidate_documents_with_nonempty_chunks": _distribution(document_counts),
        "candidate_chunks": _distribution(chunk_counts),
        "chunk_token_cost": _distribution(token_counts),
        "gold_document_ids_exported": False,
        "evidence_labels_exported": False,
        "gold_fields_visible_to_scorer": False,
    }
    return prepared, summary


def _label(labels: set[str]) -> str:
    if labels == {"SUPPORT"}:
        return "SUPPORT"
    if labels == {"CONTRADICT"}:
        return "CONTRADICT"
    if labels == {"SUPPORT", "CONTRADICT"}:
        return "MIXED"
    raise ValueError(f"unsupported SciFact label set: {sorted(labels)}")


def build_gold_rows(
    corpus_rows: Iterable[dict[str, Any]],
    claim_rows: Iterable[dict[str, Any]],
    prepared_rows: Sequence[dict[str, Any]],
    *,
    minimum_cases: int = MINIMUM_PRIMARY_CASES,
) -> list[dict[str, Any]]:
    documents = parse_corpus(corpus_rows)
    corpus_ids = {int(row["doc_id"]) for row in documents}
    claims, _ = eligible_claims(
        claim_rows,
        corpus_ids=corpus_ids,
        minimum_cases=minimum_cases,
    )
    expected_ids = [f"scifact-v42::{int(row['id'])}" for row in claims]
    if [str(row.get("id")) for row in prepared_rows] != expected_ids:
        raise ValueError("SciFact prepared cache does not align with eligible claims")
    index = CorpusBM25Index(documents)
    result: list[dict[str, Any]] = []
    for claim, prepared in zip(claims, prepared_rows, strict=True):
        gold_doc_ids, labels = _claim_evidence(dict(claim["source"]))
        ranked = index.rank(str(claim["claim"]))
        source_by_doc = {
            doc_id: f"s{rank:04d}" for rank, doc_id in enumerate(ranked)
        }
        available_sources = {
            str(candidate["source_id"])
            for candidate in prepared.get("candidates", [])
        }
        gold_sources = sorted(
            source_by_doc[doc_id]
            for doc_id in gold_doc_ids
            if doc_id in source_by_doc
            and source_by_doc[doc_id] in available_sources
        )
        ceiling = len(gold_sources) / len(gold_doc_ids)
        result.append(
            {
                "case_id": str(prepared["id"]),
                "label": _label(labels),
                "gold_document_count": len(gold_doc_ids),
                "gold_sources": gold_sources,
                "candidate_ceiling": ceiling,
                "candidate_ceiling_complete": ceiling == 1.0,
            }
        )
    return result


class FrozenSciFactScorer:
    """Metadata adapter around the byte-frozen v41 neural scorer."""

    def __init__(self, **kwargs: Any) -> None:
        self.delegate = FrozenDynamicHoVerScorer(**kwargs)

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = self.delegate.score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def _selection_metrics(
    selected: Sequence[dict[str, Any]], gold: dict[str, Any]
) -> dict[str, float | int]:
    selected_sources = sorted({str(item["source_id"]) for item in selected})
    gold_sources = set(gold["gold_sources"])
    correct = len(set(selected_sources) & gold_sources)
    precision = correct / len(selected_sources) if selected_sources else 0.0
    recall = correct / int(gold["gold_document_count"])
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "document_precision": precision,
        "document_recall": recall,
        "document_evidence_f1": f1,
        "complete_gold_document_coverage": float(recall == 1.0),
        "selected_chunk_count": len(selected),
        "selected_document_count": len(selected_sources),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "duplicate_document_selection_rate": (
            (len(selected) - len(selected_sources)) / len(selected)
            if selected
            else 0.0
        ),
    }


def _method_mean(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int]
) -> float:
    values = [
        row["configurations"][str(budget)]["methods"][method]["metrics"][
            "document_evidence_f1"
        ]
        for row in rows
        for budget in budgets
    ]
    return float(np.mean(values))


def _bootstrap(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    matrix = np.asarray(
        [
            [
                [
                    row["configurations"][str(budget)]["methods"][method][
                        "metrics"
                    ]["document_evidence_f1"]
                    for budget in BUDGETS
                ]
                for method in METHODS
            ]
            for row in evidence
        ],
        dtype=float,
    )
    observed = matrix.mean(axis=(0, 2))
    method_index = {method: index for index, method in enumerate(METHODS)}
    observed_strongest = max(
        BASELINES, key=lambda method: observed[method_index[method]]
    )
    dynamic_static = np.empty(resamples, dtype=float)
    dynamic_strongest = np.empty(resamples, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index in range(resamples):
        sample = rng.integers(0, len(evidence), size=len(evidence))
        means = matrix[sample].mean(axis=(0, 2))
        dynamic = float(means[method_index[DYNAMIC_RANK]])
        dynamic_static[index] = dynamic - float(means[method_index[STATIC_RANK]])
        dynamic_strongest[index] = dynamic - max(
            float(means[method_index[method]]) for method in BASELINES
        )

    def comparison(values: np.ndarray, point: float) -> dict[str, Any]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "point": round(point, 6),
            "ci_low": round(float(low), 6),
            "ci_high": round(float(high), 6),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        }

    return {
        "method_primary_means": {
            method: round(float(observed[index]), 6)
            for index, method in enumerate(METHODS)
        },
        "observed_strongest_non_frc": observed_strongest,
        "dynamic_minus_static_rank": comparison(
            dynamic_static,
            float(
                observed[method_index[DYNAMIC_RANK]]
                - observed[method_index[STATIC_RANK]]
            ),
        ),
        "dynamic_minus_bootstrap_strongest_non_frc": comparison(
            dynamic_strongest,
            float(
                observed[method_index[DYNAMIC_RANK]]
                - observed[method_index[observed_strongest]]
            ),
        ),
    }


def _stratum_rows(
    evidence: Sequence[dict[str, Any]], name: str
) -> list[dict[str, Any]]:
    if name.startswith("label="):
        label = name.split("=", 1)[1]
        return [row for row in evidence if row["label"] == label]
    if name == "gold_documents=1":
        return [row for row in evidence if row["gold_document_count"] == 1]
    if name == "gold_documents>=2":
        return [row for row in evidence if row["gold_document_count"] >= 2]
    if name == "candidate_ceiling_complete":
        return [row for row in evidence if row["candidate_ceiling_complete"]]
    if name == "candidate_ceiling_incomplete":
        return [row for row in evidence if not row["candidate_ceiling_complete"]]
    raise ValueError(f"unknown SciFact stratum: {name}")


def _build_report(
    evidence: list[dict[str, Any]],
    query_summary: dict[str, Any],
    *,
    resamples: int,
) -> dict[str, Any]:
    bootstrap = _bootstrap(evidence, resamples=resamples)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            BASELINES,
            key=lambda method: _method_mean(evidence, method, (budget,)),
        )
        budget_deltas[str(budget)] = round(
            _method_mean(evidence, DYNAMIC_RANK, (budget,))
            - _method_mean(evidence, strongest, (budget,)),
            6,
        )
    stratum_names = (
        "label=SUPPORT",
        "label=CONTRADICT",
        "label=MIXED",
        "gold_documents=1",
        "gold_documents>=2",
        "candidate_ceiling_complete",
        "candidate_ceiling_incomplete",
    )
    strata: dict[str, dict[str, Any]] = {}
    for name in stratum_names:
        rows = _stratum_rows(evidence, name)
        supported = len(rows) >= MINIMUM_SAFETY_STRATUM_CASES
        delta: float | None = None
        if supported:
            strongest = max(
                BASELINES,
                key=lambda method: _method_mean(rows, method, BUDGETS),
            )
            delta = round(
                _method_mean(rows, DYNAMIC_RANK, BUDGETS)
                - _method_mean(rows, strongest, BUDGETS),
                6,
            )
        strata[name] = {
            "cases": len(rows),
            "safety_supported": supported,
            "dynamic_minus_strongest_non_frc": delta,
        }
    dynamic_static = bootstrap["dynamic_minus_static_rank"]
    dynamic_strongest = bootstrap[
        "dynamic_minus_bootstrap_strongest_non_frc"
    ]
    safety_values = list(budget_deltas.values()) + [
        float(row["dynamic_minus_strongest_non_frc"])
        for row in strata.values()
        if row["safety_supported"]
    ]
    checks = {
        "dynamic_minus_static_point_at_least_0_005": (
            dynamic_static["point"] >= 0.005
        ),
        "dynamic_minus_static_ci_low_above_0": dynamic_static["ci_low"] > 0.0,
        "dynamic_minus_strongest_point_at_least_0_01": (
            dynamic_strongest["point"] >= 0.01
        ),
        "dynamic_minus_strongest_ci_low_above_0": (
            dynamic_strongest["ci_low"] > 0.0
        ),
        "every_budget_and_supported_stratum_at_least_minus_0_02": (
            min(safety_values) >= -0.02
        ),
        "generation_fallback_rate_at_most_0_05": (
            float(query_summary["fallback_rate"]) <= 0.05
        ),
        "no_forbidden_field_leak": True,
    }
    supported = all(checks.values())
    status = (
        "DYNAMIC_ATOMIC_ROLE_SUPPORT_ESTABLISHED_SINGLE_EXTERNAL_SCIFACT"
        if supported
        else "SCIFACT_DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED"
    )
    return {
        "schema_version": "frc-scifact-dynamic-atomic-report-v1",
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "methods": list(METHODS),
            "primary_metric": "document_evidence_f1",
            "raw_text_committed": False,
            "gold_joined_only_after_complete_blind_score_cache": True,
        },
        "analysis": {
            "family_comparison": bootstrap,
            "budget_deltas": budget_deltas,
            "strata": strata,
            "query_generation": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_established": supported,
                "single_external_dataset_only": True,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
                "scifact_reuse_for_tuning_authorized": False,
            },
        },
        "aggregates": {
            "method_primary_means": bootstrap["method_primary_means"]
        },
    }


def evaluate_scifact(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows) or len(scored_rows) != len(gold_rows):
        raise ValueError("SciFact score/gold cardinality mismatch")
    evidence: list[dict[str, Any]] = []
    for row in scored_rows:
        if _contains_forbidden_cache_key(row):
            raise ValueError("SciFact scored cache contains a forbidden field")
        case_id = str(row.get("id", ""))
        gold = gold_by_id.get(case_id)
        if gold is None:
            raise ValueError(f"SciFact scored unknown case: {case_id}")
        candidates = merge_scored_candidates(row)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_candidates(
                    candidates,
                    method,
                    token_budget=budget,
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, gold),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "label": str(gold["label"]),
                "gold_document_count": int(gold["gold_document_count"]),
                "candidate_ceiling": float(gold["candidate_ceiling"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "configurations": configurations,
                "raw_claim_title_abstract_or_evidence_text_exported": False,
                "raw_document_ids_exported": False,
            }
        )
    return (
        _build_report(evidence, query_summary, resamples=resamples),
        evidence,
    )


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scifact_dynamic_atomic_roles.json"
    markdown_path = output_dir / "scifact_dynamic_atomic_roles.md"
    evidence_path = output_dir / "scifact_dynamic_atomic_roles_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    comparison = report["analysis"]["family_comparison"]
    lines = [
        "# SciFact 动态原子角色外部确认（v42）",
        "",
        f"- 状态：`{report['analysis']['outcome']['status']}`",
        f"- 评测 claim：{report['metadata']['cases']}",
        "- 动态 - 静态秩覆盖："
        f"{comparison['dynamic_minus_static_rank']['point']:+.6f}，95% CI "
        f"[{comparison['dynamic_minus_static_rank']['ci_low']:+.6f}, "
        f"{comparison['dynamic_minus_static_rank']['ci_high']:+.6f}]",
        "- 动态 - bootstrap 最强非 FRC："
        f"{comparison['dynamic_minus_bootstrap_strongest_non_frc']['point']:+.6f}，"
        "95% CI "
        f"[{comparison['dynamic_minus_bootstrap_strongest_non_frc']['ci_low']:+.6f}, "
        f"{comparison['dynamic_minus_bootstrap_strongest_non_frc']['ci_high']:+.6f}]",
        "",
        "## 方法主指标",
        "",
        "| 方法 | Document Evidence F1 |",
        "|---|---:|",
    ]
    for method, value in report["aggregates"]["method_primary_means"].items():
        lines.append(f"| `{method}` | {value:.6f} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是冻结 v41 方法在一个外部公开数据集上的单次迁移确认。无论结果"
            "如何，都不授权选择器上线、不改变 Gate 2、不复现 SetR，也不证明"
            "洪水防汛领域效果；SciFact 此后不得用于方法调参。",
            "",
        ]
    )
    markdown_path.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    with evidence_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    return {
        "json": json_path,
        "markdown": markdown_path,
        "evidence": evidence_path,
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("SciFact v42 protocol changed after preregistration")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SciFact v42 protocol experiment mismatch")
    if value.get("frozen_methods") != list(METHODS):
        raise ValueError("SciFact v42 method registration mismatch")
    if value["frozen_selector"]["token_budgets"] != list(BUDGETS):
        raise ValueError("SciFact v42 budget registration mismatch")
    return value


def validate_implementation_registration(
    path: Path,
    *,
    protocol_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    inherited_module_path: Path,
) -> dict[str, Any]:
    validate_protocol(protocol_path)
    if not path.is_file():
        raise FileNotFoundError("SciFact v42 implementation registration is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": sha256(protocol_path),
        "module": sha256(module_path),
        "runner": sha256(runner_path),
        "test": sha256(test_path),
        "inherited_v41_module": sha256(inherited_module_path),
    }
    if value.get("artifact_sha256") != expected:
        raise ValueError("SciFact v42 implementation artifacts changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SciFact v42 implementation experiment mismatch")
    boundary = value.get("registration_boundary", {})
    if boundary.get("scifact_data_downloaded") is not False:
        raise ValueError("SciFact v42 implementation boundary is invalid")
    if boundary.get("scifact_content_accessed") is not False:
        raise ValueError("SciFact v42 content-access boundary is invalid")
    if value.get("blind_preparation_authorized") is not True:
        raise RuntimeError("SciFact v42 blind preparation is not authorized")
    return value


def validate_execution_registration(
    path: Path,
    *,
    implementation_path: Path,
    protocol_path: Path,
    archive_path: Path,
    corpus_path: Path,
    claims_path: Path,
    prepared_path: Path,
    census_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    inherited_module_path: Path,
    runtime_parameters: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("SciFact v42 execution registration is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SciFact v42 execution experiment mismatch")
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("SciFact v42 execution protocol mismatch")
    if value.get("implementation_registration_sha256") != sha256(
        implementation_path
    ):
        raise ValueError("SciFact v42 execution implementation mismatch")
    expected = {
        "protocol": sha256(protocol_path),
        "implementation_registration": sha256(implementation_path),
        "archive": sha256(archive_path),
        "corpus": sha256(corpus_path),
        "claims_dev": sha256(claims_path),
        "prepared_blind": sha256(prepared_path),
        "blind_census": sha256(census_path),
        "module": sha256(module_path),
        "runner": sha256(runner_path),
        "test": sha256(test_path),
        "inherited_v41_module": sha256(inherited_module_path),
    }
    if value.get("artifact_sha256") != expected:
        raise ValueError("SciFact v42 execution artifacts changed")
    if value.get("runtime_parameters") != runtime_parameters:
        raise ValueError("SciFact v42 runtime parameters changed")
    if value.get("blind_generation_and_scoring_authorized") is not True:
        raise RuntimeError("SciFact v42 blind generation/scoring is not authorized")
    if value.get("scores_or_metrics_computed_before_registration") is not False:
        raise ValueError("SciFact v42 execution boundary is invalid")
    if value.get("gold_aggregate_statistics_computed_before_registration") is not False:
        raise ValueError("SciFact v42 gold-aggregate boundary is invalid")
    return value
