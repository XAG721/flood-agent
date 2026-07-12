from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flood_system.rag_annotation import (
    AdjudicationResolution,
    AdjudicationSubmission,
    AnnotationDecision,
    AnnotationSubmission,
    blank_submission,
    compare_submissions,
    finalize_adjudication,
    prepare_annotation_package,
    validate_submission,
)


BENCHMARK = Path("flood_system/rag_benchmarks/district_policy_benchmark.json")


def _submission(package, annotator_id: str, source: dict, *, disagree: bool = False) -> AnnotationSubmission:
    documents = {item["doc_id"]: item for item in source["documents"]}
    decisions = []
    for index, case in enumerate(source["cases"]):
        relevant = list(case["relevant_doc_ids"])
        if disagree and index == 0:
            relevant = relevant[:-1]
        decisions.append(
            AnnotationDecision(
                item_id=case["case_id"],
                relevant_doc_ids=relevant,
                role_labels={
                    doc_id: documents[doc_id]["metadata"]["evidence_roles"] for doc_id in relevant
                },
                reference_answer=f"reference answer {index}",
            )
        )
    return AnnotationSubmission(
        package_id=package.package_id,
        annotator_id=annotator_id,
        decisions=decisions,
        submitted_at=datetime.now(UTC),
    )


def test_prepare_annotation_package_is_reproducibly_blinded() -> None:
    first = prepare_annotation_package(BENCHMARK, seed=17)
    second = prepare_annotation_package(BENCHMARK, seed=17)

    assert first.package_id == second.package_id
    assert [candidate.doc_id for candidate in first.items[0].candidates] == [
        candidate.doc_id for candidate in second.items[0].candidates
    ]
    payload = first.model_dump_json()
    for forbidden in (
        "relevant_doc_ids",
        "required_roles",
        "evidence_roles",
        "source_label",
        "trust_score",
        "conflicts_with",
        "_retrieval_explain",
    ):
        assert forbidden not in payload


def test_compare_requires_independent_complete_valid_annotations() -> None:
    package = prepare_annotation_package(BENCHMARK)
    source = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    first = _submission(package, "annotator-alpha", source)
    second = _submission(package, "annotator-beta", source, disagree=True)

    comparison = compare_submissions(package, first, second)

    assert comparison.evidence_cohen_kappa < 1.0
    assert comparison.mean_evidence_jaccard < 1.0
    assert [item["item_id"] for item in comparison.disagreements] == ["underpass_orange_response"]
    assert "annotator-alpha" not in comparison.model_dump_json()
    with pytest.raises(ValueError, match="distinct annotators"):
        compare_submissions(package, first, first)

    blank = blank_submission(package, "REPLACE_WITH_ANNOTATOR_A_ID")
    with pytest.raises(ValueError, match="placeholder annotator_id"):
        validate_submission(package, blank)
    blank.annotator_id = "annotator-empty"
    with pytest.raises(ValueError, match="explain why no evidence"):
        validate_submission(package, blank)

    invalid = first.model_copy(deep=True)
    invalid.decisions[0].role_labels.pop(invalid.decisions[0].relevant_doc_ids[0])
    with pytest.raises(ValueError, match="assign roles to every selected"):
        validate_submission(package, invalid)


def test_finalize_adjudication_requires_third_person_and_emits_hashed_provenance() -> None:
    package = prepare_annotation_package(BENCHMARK)
    source = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    first = _submission(package, "annotator-alpha", source)
    second = _submission(package, "annotator-beta", source, disagree=True)
    documents = {item["doc_id"]: item for item in source["documents"]}
    resolutions = [
        AdjudicationResolution(
            item_id=case["case_id"],
            relevant_doc_ids=case["relevant_doc_ids"],
            role_labels={
                doc_id: documents[doc_id]["metadata"]["evidence_roles"]
                for doc_id in case["relevant_doc_ids"]
            },
            reference_answer=f"adjudicated answer {index}",
            rationale="The selected documents directly cover the requested evidence slots.",
        )
        for index, case in enumerate(source["cases"])
    ]
    adjudication = AdjudicationSubmission(
        package_id=package.package_id,
        adjudicator_id="adjudicator-gamma",
        resolutions=resolutions,
        submitted_at=datetime.now(UTC),
    )

    dataset = finalize_adjudication(package, first, second, adjudication)

    assert len(dataset["metadata"]["dataset_sha256"]) == 64
    assert dataset["cases"][0]["reference_answer"] == "adjudicated answer 0"
    assert dataset["metadata"]["agreement"]["mean_evidence_jaccard"] < 1.0
    serialized = json.dumps(dataset, ensure_ascii=False)
    assert "annotator-alpha" not in serialized
    assert "annotator-beta" not in serialized
    assert "adjudicator-gamma" not in serialized

    same_person = adjudication.model_copy(update={"adjudicator_id": first.annotator_id})
    with pytest.raises(ValueError, match="independent"):
        finalize_adjudication(package, first, second, same_person)

    placeholder = adjudication.model_copy(deep=True)
    placeholder.adjudicator_id = "REPLACE_WITH_INDEPENDENT_ADJUDICATOR_ID"
    with pytest.raises(ValueError, match="placeholder adjudicator_id"):
        finalize_adjudication(package, first, second, placeholder)
