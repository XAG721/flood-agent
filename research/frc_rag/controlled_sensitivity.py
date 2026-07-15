from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from flood_system.models import RAGDocument
from flood_system.rag import EvidenceSelectionPolicy, SimpleRAGStore
from flood_system.rag_evaluation import RAGEvaluationCase


CONTROLLED_SENSITIVITY_SCHEMA = "frc-controlled-domain-sensitivity-v1"
SENSITIVITY_VALUES: dict[str, tuple[float, ...]] = {
    "role_weight": (0.0, 0.5, 1.0, 2.0, 4.0),
    "field_weight": (0.0, 1.0, 2.0, 4.0, 8.0),
    "conflict_threshold": (0.0, 0.35, 0.5, 0.8, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row["metrics"][key]) for row in rows) / max(1, len(rows)), 6)


def _document_roles(document: RAGDocument) -> set[str]:
    values = document.metadata.get("evidence_roles", [])
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def _explicit_conflict_pairs(documents: list[RAGDocument]) -> list[list[str]]:
    selected_ids = {document.doc_id for document in documents}
    pairs: set[tuple[str, str]] = set()
    for document in documents:
        values = document.metadata.get("conflicts_with", [])
        if isinstance(values, str):
            values = [values]
        for other_id in values:
            other = str(other_id)
            if other in selected_ids and other != document.doc_id:
                pairs.add(tuple(sorted((document.doc_id, other))))
    return [list(pair) for pair in sorted(pairs)]


def _is_flagged(document: RAGDocument) -> bool:
    source_label = str(document.metadata.get("source_label", "")).strip().lower()
    return (
        bool(document.metadata.get("conflicts_with"))
        or document.metadata.get("is_conflicting_candidate") is True
        or source_label
        in {
        "unverified",
        "noise",
        "misinfo",
        "misinformation",
        "rumor",
        }
    )


def _selector_field_coverage(selected: list[RAGDocument]) -> float:
    if not selected:
        return 0.0
    selection = selected[0].metadata.get("_evidence_selection", {})
    slots = selection.get("all_slots", []) if isinstance(selection, dict) else []
    if not slots:
        return 0.0
    return sum(bool(slot.get("covered_by_selected_set")) for slot in slots) / len(slots)


def _score_case(case: RAGEvaluationCase, selected: list[RAGDocument]) -> dict[str, Any]:
    selected_ids = {document.doc_id for document in selected}
    expected_ids = set(case.relevant_doc_ids)
    true_positive = len(selected_ids & expected_ids)
    recall = true_positive / max(1, len(expected_ids))
    precision = true_positive / max(1, len(selected_ids))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    selected_roles = set().union(*(_document_roles(document) for document in selected)) if selected else set()
    required_roles = set(case.required_roles)
    role_coverage = len(selected_roles & required_roles) / max(1, len(required_roles))
    selector_field_coverage = _selector_field_coverage(selected)
    if case.field_evidence_map:
        field_coverage = sum(
            bool(selected_ids & set(evidence_ids))
            for evidence_ids in case.field_evidence_map.values()
        ) / len(case.field_evidence_map)
    else:
        field_coverage = selector_field_coverage
    conflict_pairs = _explicit_conflict_pairs(selected)
    flagged_count = sum(_is_flagged(document) for document in selected)
    selection = selected[0].metadata.get("_evidence_selection", {}) if selected else {}
    token_cost = int(selection.get("selected_token_cost", 0)) if isinstance(selection, dict) else 0
    case_accuracy = float(
        expected_ids <= selected_ids
        and field_coverage == 1.0
        and role_coverage == 1.0
        and not conflict_pairs
    )
    return {
        "selected_doc_ids": [document.doc_id for document in selected],
        "explicit_conflict_pairs": conflict_pairs,
        "metrics": {
            "evidence_recall": round(recall, 6),
            "evidence_precision": round(precision, 6),
            "evidence_f1": round(f1, 6),
            "field_coverage": round(field_coverage, 6),
            "selector_field_coverage": round(selector_field_coverage, 6),
            "role_coverage": round(role_coverage, 6),
            "unsupported_evidence_ratio": round(1.0 - precision, 6),
            "flagged_evidence_ratio": round(flagged_count / max(1, len(selected)), 6),
            "flagged_evidence_case": float(flagged_count > 0),
            "explicit_conflict_pair_selected": float(bool(conflict_pairs)),
            "case_accuracy": case_accuracy,
            "token_cost": token_cost,
        },
    }


def evaluate_controlled_sensitivity(
    documents: list[RAGDocument],
    cases: list[RAGEvaluationCase],
    *,
    benchmark_path: Path,
    top_k: int = 4,
    token_budget: int = 520,
    frozen_policy: EvidenceSelectionPolicy | None = None,
    sensitivity_values: dict[str, Iterable[float]] | None = None,
) -> dict[str, Any]:
    policy = frozen_policy or EvidenceSelectionPolicy()
    values_by_dimension = sensitivity_values or SENSITIVITY_VALUES
    store = SimpleRAGStore(documents)
    results: list[dict[str, Any]] = []
    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "field_coverage",
        "selector_field_coverage",
        "role_coverage",
        "unsupported_evidence_ratio",
        "flagged_evidence_ratio",
        "flagged_evidence_case",
        "explicit_conflict_pair_selected",
        "case_accuracy",
        "token_cost",
    )

    for dimension, values in values_by_dimension.items():
        if dimension not in SENSITIVITY_VALUES:
            raise ValueError(f"unsupported sensitivity dimension: {dimension}")
        for value in values:
            candidate_policy = replace(policy, **{dimension: float(value)})
            rows: list[dict[str, Any]] = []
            for case in cases:
                selected = store.query_evidence_set(
                    case.corpus,
                    case.query,
                    top_k=top_k,
                    candidate_k=max(20, len(documents)),
                    token_budget=token_budget,
                    slots=case.slots,
                    required_roles=case.required_roles,
                    selection_policy=candidate_policy,
                )
                rows.append({"case_id": case.case_id, **_score_case(case, selected)})
            results.append(
                {
                    "dimension": dimension,
                    "value": float(value),
                    "policy": candidate_policy.as_dict(),
                    "aggregate": {name: _mean(rows, name) for name in metric_names},
                    "case_results": rows,
                }
            )

    return {
        "schema_version": CONTROLLED_SENSITIVITY_SCHEMA,
        "metadata": {
            "benchmark": str(benchmark_path.as_posix()),
            "benchmark_sha256": _sha256(benchmark_path),
            "data_origin": "SYNTHETIC",
            "is_simulated": True,
            "selector_backend": "deterministic_lexical_fields_and_metadata",
            "neural_model_used": False,
            "case_count": len(cases),
            "document_count": len(documents),
            "cases_with_field_evidence_map": sum(bool(case.field_evidence_map) for case in cases),
            "field_evidence_mapping_count": sum(len(case.field_evidence_map) for case in cases),
            "top_k": top_k,
            "token_budget": token_budget,
            "frozen_policy": policy.as_dict(),
            "protocol": "one_factor_at_a_time",
            "gold_usage": "field_evidence_map is used for scoring only and is never passed to the selector",
        },
        "coverage": {
            "role_weight": "RUN_CONTROLLED_DOMAIN",
            "field_weight": "RUN_CONTROLLED_DOMAIN",
            "conflict_threshold": "RUN_CONTROLLED_DOMAIN",
        },
        "results": results,
        "limitations": [
            "This small benchmark was repository-constructed and validated for parameter identifiability; it is not held out and is not a public real-model result.",
            "It demonstrates selector behavior and parameter traceability; it cannot establish production effectiveness or Gate 2 superiority.",
            "Primary public artifacts still lack standardized field, applicability, and candidate-conflict annotations.",
            "The sweep is descriptive and must not be used to tune the frozen public test configuration after evaluation.",
        ],
    }


def render_controlled_sensitivity_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# FRC controlled-domain weight and conflict-threshold sensitivity",
        "",
        f"- Scope: `{metadata['data_origin']}` controlled domain; neural model used: `{metadata['neural_model_used']}`",
        f"- Cases/documents: {metadata['case_count']}/{metadata['document_count']}",
        f"- Gold field mappings: {metadata['field_evidence_mapping_count']} (scoring only; never passed to selector)",
        f"- Budget: Top-K={metadata['top_k']}, {metadata['token_budget']} tokens",
        f"- Protocol: `{metadata['protocol']}`; benchmark SHA-256: `{metadata['benchmark_sha256']}`",
        "",
        "| Dimension | Value | Evidence F1 | Gold field coverage | Selector field coverage | Role coverage | Flagged evidence | Flagged cases | Conflict-pair cases | Case accuracy | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        aggregate = row["aggregate"]
        lines.append(
            f"| {row['dimension']} | {row['value']:.2f} | {aggregate['evidence_f1']:.6f} | "
            f"{aggregate['field_coverage']:.6f} | {aggregate['selector_field_coverage']:.6f} | "
            f"{aggregate['role_coverage']:.6f} | "
            f"{aggregate['flagged_evidence_ratio']:.6f} | "
            f"{aggregate['flagged_evidence_case']:.6f} | "
            f"{aggregate['explicit_conflict_pair_selected']:.6f} | {aggregate['case_accuracy']:.6f} | "
            f"{aggregate['token_cost']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def write_controlled_sensitivity(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_controlled_sensitivity_markdown(report), encoding="utf-8")
    return json_path, markdown_path
