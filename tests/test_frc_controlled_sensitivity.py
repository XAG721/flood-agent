from __future__ import annotations

from pathlib import Path

import pytest

from research.frc_rag.controlled_sensitivity import (
    CONTROLLED_SENSITIVITY_SCHEMA,
    evaluate_controlled_sensitivity,
    render_controlled_sensitivity_markdown,
    write_controlled_sensitivity,
)
from research.frc_rag.public_evidence import load_controlled_domain_sensitivity
from flood_system.models import CorpusType, RAGDocument
from flood_system.rag import EvidenceSelectionPolicy, SimpleRAGStore
from flood_system.rag_evaluation import RAGBaselineEvaluator


BENCHMARK = Path("flood_system/rag_benchmarks/controlled_frc_sensitivity_benchmark.json")


def test_selection_policy_validates_weights_and_threshold() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceSelectionPolicy(field_weight=-0.1)
    with pytest.raises(ValueError, match="finite"):
        EvidenceSelectionPolicy(role_weight=float("nan"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        EvidenceSelectionPolicy(conflict_threshold=1.1)


def test_conflict_threshold_is_explicit_in_selection_trace() -> None:
    documents = [
        RAGDocument(
            doc_id="official",
            corpus=CorpusType.POLICY,
            title="Official closure condition",
            content="Close the underpass when water rises.",
            metadata={
                "source_label": "official",
                "evidence_roles": ["condition"],
                "token_cost": 20,
            },
        ),
        RAGDocument(
            doc_id="unverified",
            corpus=CorpusType.POLICY,
            title="Unverified closure procedure",
            content="The underpass closure procedure needs no water check.",
            metadata={
                "source_label": "unverified",
                "evidence_roles": ["procedure"],
                "token_cost": 20,
            },
        ),
    ]
    selected = SimpleRAGStore(documents).query_evidence_set(
        CorpusType.POLICY,
        "underpass closure condition and procedure",
        top_k=2,
        token_budget=100,
        slots=["underpass closure condition", "closure procedure"],
        required_roles=["condition", "procedure"],
        selection_policy=EvidenceSelectionPolicy(conflict_threshold=0.5),
    )

    unverified = next(document for document in selected if document.doc_id == "unverified")
    trace = unverified.metadata["_evidence_selection"]
    assert trace["selection_policy"]["conflict_threshold"] == 0.5
    assert 0.0 < trace["score_terms"]["raw_conflict_score"] < 0.5
    assert trace["score_terms"]["conflict_score"] == 0.0
    assert "role_score" in trace["score_terms"]


def test_official_negative_rule_is_not_penalized_only_for_containing_close_marker() -> None:
    documents = [
        RAGDocument(
            doc_id="official-condition",
            corpus=CorpusType.POLICY,
            title="Official underpass condition",
            content="Water depth triggers the response.",
            metadata={
                "source_label": "official",
                "evidence_roles": ["condition"],
                "token_cost": 20,
            },
        ),
        RAGDocument(
            doc_id="official-action",
            corpus=CorpusType.POLICY,
            title="Official underpass closure",
            content="Close the underpass after the trigger.",
            metadata={
                "source_label": "official",
                "evidence_roles": ["procedure"],
                "token_cost": 20,
            },
        ),
    ]
    selected = SimpleRAGStore(documents).query_evidence_set(
        CorpusType.POLICY,
        "official underpass trigger and closure",
        top_k=2,
        token_budget=100,
        slots=["underpass trigger", "underpass closure"],
        required_roles=["condition", "procedure"],
    )

    action = next(document for document in selected if document.doc_id == "official-action")
    assert action.metadata["_evidence_selection"]["score_terms"]["raw_conflict_score"] == 0.0


def test_controlled_sensitivity_is_complete_and_deterministic(tmp_path: Path) -> None:
    evaluator, cases, _ = RAGBaselineEvaluator.from_benchmark(BENCHMARK)
    first = evaluate_controlled_sensitivity(
        evaluator.documents,
        cases,
        benchmark_path=BENCHMARK,
    )
    second = evaluate_controlled_sensitivity(
        evaluator.documents,
        cases,
        benchmark_path=BENCHMARK,
    )

    assert first == second
    assert first["schema_version"] == CONTROLLED_SENSITIVITY_SCHEMA
    assert first["metadata"]["data_origin"] == "SYNTHETIC"
    assert first["metadata"]["neural_model_used"] is False
    assert first["coverage"] == {
        "role_weight": "RUN_CONTROLLED_DOMAIN",
        "field_weight": "RUN_CONTROLLED_DOMAIN",
        "conflict_threshold": "RUN_CONTROLLED_DOMAIN",
    }
    assert len(first["results"]) == 15
    assert {row["dimension"] for row in first["results"]} == {
        "role_weight",
        "field_weight",
        "conflict_threshold",
    }
    assert all(len(row["case_results"]) == len(cases) for row in first["results"])
    markdown = render_controlled_sensitivity_markdown(first)
    assert "RUN" not in markdown  # The table reports measurements, not a gate claim.
    assert "not a public real-model result" in markdown
    json_path, _ = write_controlled_sensitivity(
        first,
        json_path=tmp_path / "sensitivity.json",
        markdown_path=tmp_path / "sensitivity.md",
    )
    audited = load_controlled_domain_sensitivity(json_path)
    assert audited["status"] == "RUN_CONTROLLED_DOMAIN"
    assert audited["dimensions"]["field_weight"] == [0.0, 1.0, 2.0, 4.0, 8.0]
    assert len(audited["identifiability"]["field_weight"]["field_coverage"]) > 1
    assert len(audited["identifiability"]["conflict_threshold"]["flagged_evidence_case"]) > 1
