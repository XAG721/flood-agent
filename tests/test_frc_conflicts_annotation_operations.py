from __future__ import annotations

import json

import pytest

from research.frc_rag.conflicts_annotation_operations import (
    REVIEWER_SLOTS,
    build_annotation_operations,
    merge_batch_submissions,
)
from research.frc_rag.conflicts_expected_behavior import (
    SUBMISSION_SCHEMA_VERSION,
    canonical_json_sha256,
    validate_submission,
)


CONFLICT_TYPES = (
    "No conflict",
    "Complementary information",
    "Conflicting opinions and research outcomes",
    "Conflict due to misinformation",
    "Conflict due to outdated information",
)


def _items() -> list[dict]:
    items = []
    for index in range(20):
        conflict_type = CONFLICT_TYPES[index % len(CONFLICT_TYPES)]
        correct_answer = None if index % 3 == 0 else f"answer-{index}"
        items.append(
            {
                "schema_version": "frc-conflicts-expected-behavior-package-v1",
                "item_id": f"case-{index:02d}",
                "question": f"Question {index}?",
                "conflict_type": conflict_type,
                "expected_behavior": f"Expected behavior for {conflict_type}",
                "correct_answer": correct_answer,
                "responses": [
                    {
                        "response_id": alias,
                        "response": f"Response {alias} for case {index} [1]",
                        "sources": [
                            {
                                "citation": "[1]",
                                "title": "Source",
                                "url": "https://example.test/source",
                                "date": "2026-01-01",
                                "text": "Evidence text.",
                            }
                        ],
                    }
                    for alias in ("A", "B")
                ],
            }
        )
    return items


def _operations() -> dict:
    return build_annotation_operations(
        _items(),
        package_id="CONFLICTS-BEHAVIOR-TEST",
        seed="annotation-operations-test-v1",
        batch_count=4,
        repeat_count=5,
        minimum_repeat_batch_distance=2,
    )


def _manifest(operations: dict) -> dict:
    return {
        "operations_id": operations["operations_id"],
        "package_id": operations["package_id"],
        "reviewer_slots": list(REVIEWER_SLOTS),
        "routing_sha256": canonical_json_sha256(operations["routing"]),
        "seed": operations["configuration"]["seed"],
    }


def _completed_submissions(operations: dict, reviewer_slot: str) -> list[dict]:
    submissions = []
    for source in operations["templates"]:
        if source["reviewer_slot"] != reviewer_slot:
            continue
        submission = json.loads(json.dumps(source))
        submission["annotator_id"] = f"private-{reviewer_slot}-person"
        for decision in submission["decisions"]:
            for rating in decision["ratings"].values():
                rating["expected_behavior_adherence"] = "PASS"
                rating["factual_grounding"] = "PASS"
                rating["citation_correctness"] = "PASS"
                if rating["answer_correctness"] == "REQUIRED":
                    rating["answer_correctness"] = "PASS"
                rating["rationale"] = "The response follows the supplied evidence."
            decision["preference"] = "TIE"
        submissions.append(submission)
    return submissions


def test_operations_are_deterministic_balanced_and_method_blind() -> None:
    first = _operations()
    second = _operations()

    assert first == second
    assert len(first["batches"]) == 12
    assert len(first["templates"]) == 12
    assert len(first["repeat_item_ids"]) == 5
    assert set(first["repeat_counts_by_conflict_type"]) == set(CONFLICT_TYPES)
    assert all(value == 1 for value in first["repeat_counts_by_conflict_type"].values())

    encoded_public = json.dumps(
        {"batches": first["batches"], "templates": first["templates"]},
        ensure_ascii=False,
    )
    assert '"item_id"' not in encoded_public
    assert '"occurrence"' not in encoded_public
    assert "coverage_greedy_proxy" not in encoded_public
    assert "frc_select" not in encoded_public

    rows = first["routing"]["rows"]
    for slot in REVIEWER_SLOTS:
        slot_rows = [row for row in rows if row["reviewer_slot"] == slot]
        assert sum(row["occurrence"] == "primary" for row in slot_rows) == 20
        assert sum(row["occurrence"] == "repeat" for row in slot_rows) == 5
        primary_batch = {
            row["item_id"]: row["batch_index"]
            for row in slot_rows
            if row["occurrence"] == "primary"
        }
        assert all(
            abs(row["batch_index"] - primary_batch[row["item_id"]]) == 2
            for row in slot_rows
            if row["occurrence"] == "repeat"
        )

    no_answer_item_ids = {
        item["item_id"] for item in _items() if item["correct_answer"] is None
    }
    routing_by_task = {
        row["review_task_id"]: row for row in first["routing"]["rows"]
    }
    for template in first["templates"]:
        for decision in template["decisions"]:
            item_id = routing_by_task[decision["review_task_id"]]["item_id"]
            answer_values = {
                rating["answer_correctness"]
                for rating in decision["ratings"].values()
            }
            assert answer_values == (
                {"NOT_APPLICABLE"} if item_id in no_answer_item_ids else {"REQUIRED"}
            )


def test_completed_batches_merge_to_frozen_submission_with_intrarater_qc() -> None:
    operations = _operations()
    merged, qc = merge_batch_submissions(
        _items(),
        operations_manifest=_manifest(operations),
        routing=operations["routing"],
        reviewer_slot="reviewer_1",
        submissions=_completed_submissions(operations, "reviewer_1"),
        minimum_exact_agreement=0.8,
    )

    assert merged["schema_version"] == SUBMISSION_SCHEMA_VERSION
    assert [item["item_id"] for item in merged["decisions"]] == [
        item["item_id"] for item in _items()
    ]
    validate_submission(_items(), merged)
    assert qc["status"] == "PASS"
    assert qc["primary_decision_count"] == 20
    assert qc["repeat_decision_count"] == 5
    assert qc["overall_exact_agreement"] == 1.0
    assert qc["eligible_for_inter_annotator_comparison"] is True
    assert all(
        item["exact_agreement"] == 1.0 for item in qc["agreement"].values()
    )
    for field in (
        "expected_behavior_adherence",
        "factual_grounding",
        "citation_correctness",
        "preference",
    ):
        assert qc["agreement"][field]["cohen_kappa"] is None
        assert (
            qc["agreement"][field]["cohen_kappa_status"]
            == "UNDEFINED_NO_VARIANCE"
        )
    assert qc["agreement"]["answer_correctness"]["cohen_kappa"] == 1.0
    assert qc["agreement"]["answer_correctness"]["cohen_kappa_status"] == "DEFINED"


def test_repeat_disagreement_fails_closed_without_discarding_primary_submission() -> None:
    operations = _operations()
    submissions = _completed_submissions(operations, "reviewer_2")
    routing_by_task = {
        row["review_task_id"]: row
        for row in operations["routing"]["rows"]
        if row["reviewer_slot"] == "reviewer_2"
    }
    repeat_task_ids = [
        task_id
        for task_id, row in routing_by_task.items()
        if row["occurrence"] == "repeat"
    ][:2]
    for submission in submissions:
        for decision in submission["decisions"]:
            if decision["review_task_id"] in repeat_task_ids:
                for rating in decision["ratings"].values():
                    rating["expected_behavior_adherence"] = "FAIL"

    merged, qc = merge_batch_submissions(
        _items(),
        operations_manifest=_manifest(operations),
        routing=operations["routing"],
        reviewer_slot="reviewer_2",
        submissions=submissions,
        minimum_exact_agreement=0.8,
    )

    validate_submission(_items(), merged)
    assert qc["status"] == "REVIEW_REQUIRED"
    assert qc["eligible_for_inter_annotator_comparison"] is False
    assert qc["agreement"]["expected_behavior_adherence"]["exact_agreement"] == 0.6
    assert qc["repeat_disagreement_count"] == 2


def test_merge_rejects_missing_batch_placeholder_identity_and_tampered_routing() -> None:
    operations = _operations()
    submissions = _completed_submissions(operations, "adjudicator")
    manifest = _manifest(operations)

    with pytest.raises(ValueError, match="every expected batch"):
        merge_batch_submissions(
            _items(),
            operations_manifest=manifest,
            routing=operations["routing"],
            reviewer_slot="adjudicator",
            submissions=submissions[:-1],
            minimum_exact_agreement=0.8,
        )

    placeholder = json.loads(json.dumps(submissions))
    placeholder[0]["annotator_id"] = "REPLACE_WITH_PERSON"
    with pytest.raises(ValueError, match="non-placeholder"):
        merge_batch_submissions(
            _items(),
            operations_manifest=manifest,
            routing=operations["routing"],
            reviewer_slot="adjudicator",
            submissions=placeholder,
            minimum_exact_agreement=0.8,
        )

    tampered = json.loads(json.dumps(operations["routing"]))
    tampered["rows"][0]["item_id"] = "tampered-case"
    with pytest.raises(ValueError, match="commitment"):
        merge_batch_submissions(
            _items(),
            operations_manifest=manifest,
            routing=tampered,
            reviewer_slot="adjudicator",
            submissions=submissions,
            minimum_exact_agreement=0.8,
        )
