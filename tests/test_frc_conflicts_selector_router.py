from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.frc_rag.conflicts_evaluation import (
    CONFLICT_LABELS,
    CONFLICT_METHODS,
)
from research.frc_rag.conflicts_selector_router import (
    SCHEMA_VERSION,
    evaluate_selector_router,
    load_selector_router,
    read_gzip_jsonl,
    write_selector_router,
)


def _fixture_rows(
    case_count: int = 40,
) -> tuple[list[dict], list[dict], list[dict]]:
    scored: list[dict] = []
    selected: list[dict] = []
    predictions: list[dict] = []
    for case_index in range(case_count):
        case_id = f"case-{case_index:03d}"
        gold = CONFLICT_LABELS[case_index % len(CONFLICT_LABELS)]
        scored.append(
            {
                "id": case_id,
                "conflict_type": "must-not-be-a-runtime-feature",
                "correct_answer": "secret-answer",
            }
        )
        for method_index, method in enumerate(CONFLICT_METHODS):
            correct = (case_index + method_index) % 3 == 0
            predicted = (
                gold
                if correct
                else CONFLICT_LABELS[
                    (case_index + method_index + 1) % len(CONFLICT_LABELS)
                ]
            )
            identifiers = [
                f"{case_id}-candidate-{(method_index + offset) % 8}"
                for offset in range(3)
            ]
            evidence = []
            for offset, identifier in enumerate(identifiers):
                base = (case_index + 1) * (method_index + 2) * (offset + 1)
                evidence.append(
                    {
                        "id": identifier,
                        "text": f"observable evidence {base} method {method_index}",
                        "url": f"https://source{(method_index + offset) % 4}.example/item",
                        "token_count": 20 + (base % 17),
                        "scores": {
                            "bm25": ((base + 1) % 19) / 18,
                            "dense": ((base + 3) % 23) / 22,
                            "hybrid": ((base + 5) % 29) / 28,
                            "cross_encoder": ((base + 7) % 31) / 30,
                        },
                        "role_scores": {
                            "answer_claim": ((base + 2) % 17) / 16,
                            "source_attribution": ((base + 4) % 19) / 18,
                            "alternative_claim": ((base + 6) % 23) / 22,
                            "temporal_validity": ((base + 8) % 29) / 28,
                        },
                    }
                )
            selected.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "correct_answer": "secret-answer",
                    "selected_ids": identifiers,
                    "selected_evidence": evidence,
                }
            )
            predictions.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "gold_label": gold,
                    "predicted_label": predicted,
                }
            )
    return scored, selected, predictions


def _evaluate() -> tuple[dict, list[dict]]:
    scored, selected, predictions = _fixture_rows()
    return evaluate_selector_router(
        scored,
        selected,
        predictions,
        source_artifacts={"fixture": True},
        prior_exposure={"retrospective": True},
    )


def test_selector_router_is_grouped_deterministic_and_recomputable(
    tmp_path: Path,
) -> None:
    first_report, first_evidence = _evaluate()
    second_report, second_evidence = _evaluate()
    assert first_report["metadata"]["schema_version"] == SCHEMA_VERSION
    assert first_report["metadata"]["cases"] == 40
    assert sorted({row["fold"] for row in first_evidence}) == [0, 1, 2, 3, 4]
    assert len(first_report["feature_contract"]["all_methods"]) == 47
    assert len(first_report["feature_contract"]["without_frc"]) == 45
    assert all(
        "frc_select" not in name
        for name in first_report["feature_contract"]["without_frc"]
    )
    first_paths = write_selector_router(
        first_report,
        first_evidence,
        json_path=tmp_path / "first" / "report.json",
        markdown_path=tmp_path / "first" / "report.md",
        evidence_path=tmp_path / "first" / "cases.jsonl.gz",
    )
    second_paths = write_selector_router(
        second_report,
        second_evidence,
        json_path=tmp_path / "second" / "report.json",
        markdown_path=tmp_path / "second" / "report.md",
        evidence_path=tmp_path / "second" / "cases.jsonl.gz",
    )
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    loaded = load_selector_router(first_paths[0], first_paths[2])
    assert loaded["analysis"] == first_report["analysis"]
    evidence_text = first_paths[2].read_bytes()
    assert b"secret-answer" not in evidence_text
    assert len(read_gzip_jsonl(first_paths[2])) == 40


def test_selector_router_rejects_missing_method_and_inconsistent_gold() -> None:
    scored, selected, predictions = _fixture_rows()
    with pytest.raises(ValueError, match="selection rows do not cover"):
        evaluate_selector_router(
            scored,
            selected[:-1],
            predictions,
            source_artifacts={},
            prior_exposure={},
        )
    corrupted = copy.deepcopy(predictions)
    corrupted[1]["gold_label"] = CONFLICT_LABELS[
        (CONFLICT_LABELS.index(corrupted[1]["gold_label"]) + 1)
        % len(CONFLICT_LABELS)
    ]
    with pytest.raises(ValueError, match="inconsistent or unsupported gold"):
        evaluate_selector_router(
            scored,
            selected,
            corrupted,
            source_artifacts={},
            prior_exposure={},
        )


def test_strict_no_frc_features_do_not_read_frc_inputs() -> None:
    scored, selected, predictions = _fixture_rows()
    _, baseline = evaluate_selector_router(
        scored,
        selected,
        predictions,
        source_artifacts={},
        prior_exposure={},
    )
    changed_selection = copy.deepcopy(selected)
    changed_predictions = copy.deepcopy(predictions)
    for row in changed_selection:
        if row["method"] == "frc_select":
            row["selected_ids"] = [f"changed-{value}" for value in row["selected_ids"]]
            for identifier, item in zip(
                row["selected_ids"], row["selected_evidence"], strict=True
            ):
                item["id"] = identifier
                item["scores"]["cross_encoder"] = 0.999
    for row in changed_predictions:
        if row["method"] == "frc_select":
            row["predicted_label"] = CONFLICT_LABELS[
                (CONFLICT_LABELS.index(row["gold_label"]) + 2)
                % len(CONFLICT_LABELS)
            ]
    _, changed = evaluate_selector_router(
        scored,
        changed_selection,
        changed_predictions,
        source_artifacts={},
        prior_exposure={},
    )
    assert [row["features"]["without_frc"] for row in baseline] == [
        row["features"]["without_frc"] for row in changed
    ]
    assert [row["features"]["all_methods"] for row in baseline] != [
        row["features"]["all_methods"] for row in changed
    ]


def test_selector_router_loader_fails_closed_on_report_or_evidence_tamper(
    tmp_path: Path,
) -> None:
    report, evidence = _evaluate()
    json_path, _, evidence_path = write_selector_router(
        report,
        evidence,
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        evidence_path=tmp_path / "cases.jsonl.gz",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["analysis"]["classification"]["all_methods"]["correct"] += 1
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregates do not match"):
        load_selector_router(json_path, evidence_path)
    report, evidence = _evaluate()
    write_selector_router(
        report,
        evidence,
        json_path=json_path,
        markdown_path=tmp_path / "report.md",
        evidence_path=evidence_path,
    )
    evidence_path.write_bytes(evidence_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        load_selector_router(json_path, evidence_path)
