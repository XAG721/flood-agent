from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.conflicts_ablation import (
    FULL_METHOD,
    WITHOUT_CONFLICT_METHOD,
    build_conflict_ablation_report,
    seed_equivalent_predictions,
    select_conflict_ablation_evidence,
)
from research.frc_rag.public_evidence import load_conflicts_real_model_ablation, select_precomputed


def _candidate(
    candidate_id: str,
    *,
    cross: float,
    answer: float,
    source: float,
    alternative: float,
    temporal: float,
    date: str,
) -> dict:
    return {
        "id": candidate_id,
        "text": f"Evidence {candidate_id}",
        "title": candidate_id,
        "url": f"https://{candidate_id}.example.test/item",
        "date": date,
        "token_count": 20,
        "scores": {"cross_encoder": cross},
        "role_scores": {
            "answer_claim": answer,
            "source_attribution": source,
            "alternative_claim": alternative,
            "temporal_validity": temporal,
        },
    }


def _cases() -> list[dict]:
    required = [
        "answer_claim",
        "source_attribution",
        "alternative_claim",
        "temporal_validity",
    ]
    return [
        {
            "id": "case-outdated",
            "source": "fixture",
            "question": "Which statement is current?",
            "conflict_type": "Conflict due to outdated information",
            "correct_answer": "Evidence newest",
            "required_roles": required,
            "candidates": [
                _candidate(
                    "answer",
                    cross=1.0,
                    answer=1.0,
                    source=0.8,
                    alternative=0.1,
                    temporal=0.1,
                    date="2024",
                ),
                _candidate(
                    "alternative",
                    cross=0.5,
                    answer=0.2,
                    source=0.3,
                    alternative=0.9,
                    temporal=0.2,
                    date="2020",
                ),
                _candidate(
                    "newest",
                    cross=0.4,
                    answer=0.3,
                    source=0.4,
                    alternative=0.2,
                    temporal=1.0,
                    date="2025",
                ),
            ],
        },
        {
            "id": "case-no-conflict",
            "source": "fixture",
            "question": "What is supported?",
            "conflict_type": "No conflict",
            "correct_answer": "Evidence direct",
            "required_roles": required,
            "candidates": [
                _candidate(
                    "direct",
                    cross=1.0,
                    answer=1.0,
                    source=1.0,
                    alternative=0.8,
                    temporal=0.8,
                    date="2025",
                ),
                _candidate(
                    "secondary",
                    cross=0.5,
                    answer=0.5,
                    source=0.5,
                    alternative=0.5,
                    temporal=0.5,
                    date="2024",
                ),
            ],
        },
    ]


def test_role_specific_threshold_preserves_default_and_changes_only_target_roles() -> None:
    case = _cases()[0]
    default = select_precomputed(case, "frc_select", k=2, budget=100)
    explicit_default = select_precomputed(
        case,
        "frc_select",
        k=2,
        budget=100,
        role_thresholds={"alternative_claim": 0.55, "temporal_validity": 0.55},
    )
    high_conflict_threshold = select_precomputed(
        case,
        "frc_select",
        k=2,
        budget=100,
        role_thresholds={"alternative_claim": 0.95, "temporal_validity": 0.95},
    )

    assert [item["id"] for item in default] == [item["id"] for item in explicit_default]
    assert [item["id"] for item in high_conflict_threshold] != [item["id"] for item in default]


def test_ablation_covers_full_without_conflict_and_all_thresholds() -> None:
    rows = select_conflict_ablation_evidence(
        _cases(),
        top_k=2,
        token_budget=100,
        conflict_thresholds=(0.25, 0.55, 0.85),
    )
    methods = {row["method"] for row in rows}
    assert methods == {
        FULL_METHOD,
        WITHOUT_CONFLICT_METHOD,
        "conflict_threshold_0.25",
        "conflict_threshold_0.55",
        "conflict_threshold_0.85",
    }
    assert len(rows) == len(_cases()) * len(methods)
    assert all(len(row["selected_ids"]) <= 2 for row in rows)

    with pytest.raises(ValueError, match="frozen 0.55"):
        select_conflict_ablation_evidence(_cases(), conflict_thresholds=(0.25, 0.85))


def test_prediction_reuse_requires_identical_ordered_evidence() -> None:
    selected = [
        {
            "case_id": "case-1",
            "method": "new-method",
            "conflict_type": "No conflict",
            "selected_ids": ["a", "b"],
        },
        {
            "case_id": "case-1",
            "method": "reordered",
            "conflict_type": "No conflict",
            "selected_ids": ["b", "a"],
        },
    ]
    reference_selected = [
        {"case_id": "case-1", "method": "old", "selected_ids": ["a", "b"]}
    ]
    reference_predictions = [
        {
            "case_id": "case-1",
            "method": "old",
            "raw_prediction": "No conflict",
            "predicted_label": "No conflict",
        }
    ]

    seeds = seed_equivalent_predictions(
        selected,
        reference_selected_rows=reference_selected,
        reference_predictions=reference_predictions,
        cache_key="fixture-key",
    )

    assert [(row["case_id"], row["method"]) for row in seeds] == [
        ("case-1", "new-method")
    ]
    assert seeds[0]["reused_from_exact_reference_selection"] is True


def test_report_reaggregates_case_level_real_model_ablation(tmp_path: Path) -> None:
    cases = _cases()
    thresholds = (0.25, 0.55, 0.85)
    selected = select_conflict_ablation_evidence(
        cases,
        top_k=2,
        token_budget=100,
        conflict_thresholds=thresholds,
    )
    predictions = []
    for row in selected:
        correct = row["method"] not in {WITHOUT_CONFLICT_METHOD, "conflict_threshold_0.85"}
        predictions.append(
            {
                "case_id": row["case_id"],
                "method": row["method"],
                "gold_label": row["conflict_type"],
                "raw_prediction": row["conflict_type"] if correct else "No conflict",
                "predicted_label": row["conflict_type"] if correct else "No conflict",
            }
        )
    source_paths = {}
    for name in ("dataset", "scored_cache", "scored_metadata", "reference_selected", "reference_predictions"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        source_paths[name] = path

    report = build_conflict_ablation_report(
        cases=cases,
        selected_rows=selected,
        predictions=predictions,
        config={"top_k": 2, "token_budget": 100, "generator_model": "fixture"},
        conflict_thresholds=thresholds,
        source_paths=source_paths,
    )

    assert report["schema_version"] == "frc-conflicts-real-model-ablation-v1"
    assert report["ablation"]["status"] == "RUN_REAL_MODEL_CONFLICTS"
    assert report["conflict_threshold_sensitivity"]["status"] == "RUN_REAL_MODEL_CONFLICTS"
    assert report["ablation"]["full"]["classification"]["accuracy"] == 1.0
    assert report["ablation"]["without_conflict"]["classification"]["accuracy"] == 0.5
    assert report["decision"]["gate_2"] == "NO-GO"
    assert len(report["case_results"]) == len(selected)

    assert tmp_path.is_dir()


def test_public_loader_reaggregates_committed_conflicts_ablation(tmp_path: Path) -> None:
    source = Path("output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.json")
    loaded = load_conflicts_real_model_ablation(source)
    assert loaded["status"] == "RUN_REAL_MODEL_CONFLICTS"
    assert loaded["metadata"]["case_count"] == 458

    report = json.loads(source.read_text(encoding="utf-8"))
    report["aggregates"][0]["classification"]["accuracy"] = 0.123456
    case_file = report["case_results_artifact"]["file"]
    (tmp_path / case_file).write_bytes((source.parent / case_file).read_bytes())
    artifact = tmp_path / "tampered-conflicts-ablation.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="classification aggregate mismatch"):
        load_conflicts_real_model_ablation(artifact)
