from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.housing_ablation import (
    HOUSING_ABLATION_SCHEMA,
    HOUSING_METHODS,
    build_housing_ablation_report,
    build_housing_composite_cases,
    housing_answer_prompt,
    parse_housing_answers,
    select_housing_evidence,
    select_housing_methods,
)
from research.frc_rag.public_evidence import load_housing_real_model_ablation


def _source_row(
    idx: int,
    state: str,
    group: int,
    answer: str,
    statute_idx: int,
    question: str,
) -> dict:
    return {
        "idx": idx,
        "state": state,
        "question": question,
        "answer": answer,
        "question_group": group,
        "statutes": [
            {
                "statute_idx": statute_idx,
                "citation": f"{state.upper()} CODE {statute_idx}",
                "excerpt": (
                    f"This statutory excerpt for {state} addresses {question.lower()} and "
                    "contains enough words for deterministic benchmark eligibility."
                ),
            }
        ],
    }


def test_public_expert_rows_become_fields_without_label_leakage() -> None:
    rows = [
        _source_row(1, "Alpha", 10, "Yes", 101, "Is notice required?"),
        _source_row(2, "Alpha", 20, "No", 102, "Is a hearing mandatory?"),
        _source_row(3, "Alpha", 30, "Yes", 103, "Is filing allowed?"),
        _source_row(4, "Beta", 10, "No", 201, "Is notice required?"),
        _source_row(5, "Beta", 20, "Yes", 202, "Is a hearing mandatory?"),
        _source_row(6, "Gamma", 10, "No", 301, "Is notice required?"),
        _source_row(7, "Gamma", 20, "Yes", 302, "Is a hearing mandatory?"),
    ]

    cases = build_housing_composite_cases(
        rows,
        case_count=1,
        fields_per_case=2,
        same_jurisdiction_distractors=1,
        seed=7,
    )

    case = cases[0]
    assert case["jurisdiction"] == "Alpha"
    assert len(case["fields"]) == 2
    assert len(case["gold_evidence_ids"]) == 2
    assert len(case["candidates"]) == 5
    assert set(case["expected_answers"].values()) == {"Yes", "No"}
    assert all("answer" not in candidate["metadata"] for candidate in case["candidates"])
    assert sum(
        "wrong_jurisdiction_same_field_opposite_answer"
        in candidate["metadata"]["candidate_origin"]
        for candidate in case["candidates"]
    ) == 2


def _scored_candidate(
    candidate_id: str,
    jurisdiction: str,
    *,
    cross: float,
    field_1: float,
    field_2: float,
    role: float,
) -> dict:
    return {
        "id": candidate_id,
        "text": f"Evidence {candidate_id}",
        "citation": candidate_id,
        "token_count": 10,
        "metadata": {"jurisdiction": jurisdiction, "snapshot_year": 2021},
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "field_scores": {"field-01": field_1, "field-02": field_2},
        "role_scores": {
            "governing_rule": role,
            "condition_exception": role,
            "source_attribution": role,
        },
    }


def _scored_case() -> dict:
    return {
        "id": "housing-fixture",
        "dataset": "housing_qa",
        "jurisdiction": "Alpha",
        "as_of_year": 2021,
        "question": "Two housing fields for Alpha in 2021",
        "fields": [
            {"field_id": "field-01", "question": "Field one?"},
            {"field_id": "field-02", "question": "Field two?"},
        ],
        "field_evidence_map": {"field-01": ["a1"], "field-02": ["a2"]},
        "expected_answers": {"field-01": "Yes", "field-02": "No"},
        "gold_evidence_ids": ["a1", "a2"],
        "required_roles": [
            "governing_rule",
            "condition_exception",
            "source_attribution",
        ],
        "candidates": [
            _scored_candidate(
                "a1", "Alpha", cross=0.70, field_1=0.95, field_2=0.10, role=0.60
            ),
            _scored_candidate(
                "a2", "Alpha", cross=0.60, field_1=0.10, field_2=0.95, role=0.60
            ),
            _scored_candidate(
                "d", "Alpha", cross=1.00, field_1=0.10, field_2=0.10, role=1.00
            ),
            _scored_candidate(
                "w1", "Beta", cross=0.95, field_1=1.00, field_2=0.10, role=0.90
            ),
            _scored_candidate(
                "w2", "Beta", cross=0.94, field_1=0.10, field_2=1.00, role=0.90
            ),
        ],
    }


def test_field_and_applicability_ablations_change_only_the_intended_component() -> None:
    case = _scored_case()
    full = select_housing_evidence(case, "frc_full", top_k=2, token_budget=100)
    without_field = select_housing_evidence(case, "w/o_field", top_k=2, token_budget=100)
    without_applicability = select_housing_evidence(
        case, "w/o_applicability", top_k=2, token_budget=100
    )

    assert [item["id"] for item in full] == ["a1", "a2"]
    assert [item["id"] for item in without_field][0] == "d"
    assert any(item["metadata"]["jurisdiction"] == "Beta" for item in without_applicability)
    assert all(item["metadata"]["jurisdiction"] == "Alpha" for item in full)
    assert all(item["metadata"]["jurisdiction"] == "Alpha" for item in without_field)


def test_answer_prompt_excludes_gold_and_parser_is_strict() -> None:
    selected = select_housing_methods([_scored_case()], top_k=2, token_budget=100)
    row = next(item for item in selected if item["method"] == "frc_full")
    prompt = housing_answer_prompt(row)

    assert "expected_answers" not in prompt
    assert "field_evidence_map" not in prompt
    parsed = parse_housing_answers(
        '```json\n{"field-01": "Yes", "field-02": "unknown - insufficient"}\n```',
        ["field-01", "field-02", "field-03"],
    )
    assert parsed == {
        "field-01": "Yes",
        "field-02": "Unknown",
        "field-03": "Invalid",
    }


def test_report_covers_all_methods_and_preserves_no_go_boundary(tmp_path: Path) -> None:
    case = _scored_case()
    selected = select_housing_methods([case], top_k=2, token_budget=100)
    predictions = []
    for row in selected:
        answers = (
            row["expected_answers"]
            if row["method"] == "frc_full"
            else {"field-01": "No", "field-02": "Yes"}
        )
        predictions.append(
            {
                "case_id": row["case_id"],
                "method": row["method"],
                "answers": answers,
            }
        )
    source = tmp_path / "questions.json"
    source.write_text(json.dumps([{"fixture": True}]), encoding="utf-8")
    source_zip = tmp_path / "questions.json.zip"
    source_zip.write_bytes(b"zip fixture")
    scored = tmp_path / "scored.jsonl"
    scored.write_text("{}\n", encoding="utf-8")

    report = build_housing_ablation_report(
        cases=[case],
        selected_rows=selected,
        predictions=predictions,
        config={
            "embedding_model": "fixture-embedder",
            "reranker_model": "fixture-reranker",
            "generator_model": "fixture-generator",
            "top_k": 2,
            "token_budget": 100,
            "field_threshold": 0.55,
            "role_threshold": 0.55,
            "field_weight": 2.0,
            "role_weight": 1.0,
        },
        source_path=source,
        source_zip_path=source_zip,
        scored_path=scored,
    )

    assert report["schema_version"] == HOUSING_ABLATION_SCHEMA
    assert set(report["aggregates"]) == set(HOUSING_METHODS)
    assert report["coverage"]["w/o_field"] == "RUN_PUBLIC_EXPERT_REAL_MODEL"
    assert report["coverage"]["version_or_expiry_variation"] == (
        "NOT_IDENTIFIABLE_SINGLE_SNAPSHOT"
    )
    assert report["decision"]["gate_2"] == "NO-GO"
    assert report["aggregates"]["frc_full"]["answer_accuracy"] == 1.0
    assert len(report["case_results"]) == len(HOUSING_METHODS)


def test_public_loader_reaggregates_committed_housing_ablation(tmp_path: Path) -> None:
    source = Path(
        "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.json"
    )
    loaded = load_housing_real_model_ablation(source)

    assert loaded["status"] == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
    assert loaded["metadata"]["case_count"] == 40
    assert loaded["metadata"]["field_count"] == 160
    assert loaded["strongest_baseline"] == "field_decomposition_topk"
    assert loaded["decision"]["gate_2"] == "NO-GO"

    report = json.loads(source.read_text(encoding="utf-8"))
    report["aggregates"]["frc_full"]["field_coverage"] = 0.123456
    case_file = report["case_results_artifact"]["file"]
    (tmp_path / case_file).write_bytes((source.parent / case_file).read_bytes())
    artifact = tmp_path / "tampered-housing-ablation.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="HousingQA aggregate mismatch"):
        load_housing_real_model_ablation(artifact)
