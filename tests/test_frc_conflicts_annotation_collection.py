from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.conflicts_annotation_collection import CollectionManager
from research.frc_rag.conflicts_annotation_operations import (
    REVIEWER_SLOTS,
    build_annotation_operations,
)
from research.frc_rag.conflicts_annotation_workstation import WorkstationSession
from research.frc_rag.conflicts_expected_behavior import canonical_json_sha256


CONFLICT_TYPES = (
    "No conflict",
    "Complementary information",
    "Conflicting opinions and research outcomes",
    "Conflict due to misinformation",
    "Conflict due to outdated information",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _items() -> list[dict]:
    return [
        {
            "schema_version": "frc-conflicts-expected-behavior-package-v1",
            "item_id": f"case-{index:02d}",
            "question": f"Question {index}?",
            "conflict_type": CONFLICT_TYPES[index % len(CONFLICT_TYPES)],
            "expected_behavior": "Use the shown evidence and preserve uncertainty.",
            "correct_answer": None if index % 3 == 0 else f"answer-{index}",
            "responses": [
                {
                    "response_id": alias,
                    "response": f"Blinded response {alias} for case {index} [1].",
                    "sources": [
                        {
                            "citation": "[1]",
                            "title": "Visible source",
                            "url": "https://example.test/source",
                            "date": "2026-01-01",
                            "text": "Visible evidence text.",
                        }
                    ],
                }
                for alias in ("A", "B")
            ],
        }
        for index in range(10)
    ]


def _fixture(tmp_path: Path) -> tuple[CollectionManager, Path, Path, dict]:
    items = _items()
    operations = build_annotation_operations(
        items,
        package_id="CONFLICTS-BEHAVIOR-COLLECTION-TEST",
        seed="collection-test-v1",
        batch_count=2,
        repeat_count=5,
        minimum_repeat_batch_distance=1,
    )
    public_root = tmp_path / "public"
    public_files = []
    for batch, template in zip(
        operations["batches"], operations["templates"], strict=True
    ):
        slot_directory = batch["reviewer_slot"].replace("_", "-")
        stem = f"batch-{int(batch['batch_index']):02d}"
        batch_path = public_root / slot_directory / f"{stem}.json"
        template_path = public_root / slot_directory / f"{stem}-submission-template.json"
        _write_json(batch_path, batch)
        _write_json(template_path, template)
        for kind, path in (
            ("batch", batch_path),
            ("submission_template", template_path),
        ):
            public_files.append(
                {
                    "kind": kind,
                    "reviewer_slot": batch["reviewer_slot"],
                    "batch_id": batch["batch_id"],
                    "path": path.relative_to(public_root).as_posix(),
                    "sha256": _file_sha256(path),
                }
            )

    package_path = tmp_path / "package.jsonl.gz"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with package_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
            for item in items:
                archive.write(
                    (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
                )
    routing_path = tmp_path / ".cache" / "routing.json"
    _write_json(routing_path, operations["routing"])
    manifest = {
        "schema_version": "frc-conflicts-annotation-operations-manifest-v1",
        "status": "PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW",
        "operations_id": operations["operations_id"],
        "package_id": operations["package_id"],
        "package_file_sha256": _file_sha256(package_path),
        "routing_sha256": canonical_json_sha256(operations["routing"]),
        "seed": operations["configuration"]["seed"],
        "reviewer_slots": list(REVIEWER_SLOTS),
        "batch_count": 2,
        "public_files": public_files,
    }
    manifest_path = public_root / "manifest.json"
    _write_json(manifest_path, manifest)
    manager = CollectionManager(
        repository_root=tmp_path,
        public_root=public_root,
        draft_root=tmp_path / ".cache" / "drafts",
        expected_manifest_sha256=_file_sha256(manifest_path),
    )
    return manager, package_path, routing_path, operations["routing"]


def _completed_submission(
    draft: dict,
    *,
    identity: str,
    disagreeing_repeat_ids: set[str] | None = None,
) -> dict:
    completed = copy.deepcopy(draft)
    completed["annotator_id"] = identity
    disagreeing_repeat_ids = disagreeing_repeat_ids or set()
    for decision in completed["decisions"]:
        for rating in decision["ratings"].values():
            rating["expected_behavior_adherence"] = "PASS"
            rating["factual_grounding"] = "PASS"
            rating["citation_correctness"] = "PASS"
            if rating["answer_correctness"] == "REQUIRED":
                rating["answer_correctness"] = "PASS"
            rating["rationale"] = "The visible response is supported by its source."
        if decision["review_task_id"] in disagreeing_repeat_ids:
            decision["ratings"]["A"]["expected_behavior_adherence"] = "FAIL"
        decision["preference"] = "TIE"
    return completed


def _finalize_all(
    manager: CollectionManager,
    *,
    identities: dict[str, str],
    disagreeing_repeat_ids: set[str] | None = None,
) -> None:
    for slot in REVIEWER_SLOTS:
        for batch_index in range(1, manager.batch_count + 1):
            session = WorkstationSession(
                repository_root=manager.repository_root,
                public_root=manager.public_root,
                draft_root=manager.draft_root,
                reviewer_slot=slot,
                batch_index=batch_index,
                expected_manifest_sha256=manager.expected_manifest_sha256,
            )
            completed = _completed_submission(
                session.draft,
                identity=identities[slot],
                disagreeing_repeat_ids=(
                    disagreeing_repeat_ids if slot == "reviewer_1" else None
                ),
            )
            session.finalize(completed)


def test_collection_tracks_missing_finalized_and_distinct_private_identities(
    tmp_path: Path,
) -> None:
    manager, _, _, _ = _fixture(tmp_path)
    initial = manager.audit()
    assert initial["status"] == "AWAITING_INDEPENDENT_HUMAN_BATCHES"
    assert initial["present_batch_count"] == 0
    assert initial["finalized_batch_count"] == 0
    assert initial["expected_batch_count"] == 6

    identities = {
        "reviewer_1": "private-reviewer-one",
        "reviewer_2": "private-reviewer-two",
        "adjudicator": "private-adjudicator",
    }
    _finalize_all(manager, identities=identities)
    ready = manager.audit()

    assert ready["status"] == "READY_FOR_PRIVATE_MERGE"
    assert ready["present_batch_count"] == 6
    assert ready["finalized_batch_count"] == 6
    assert ready["independent_role_identity_count"] == 3
    assert ready["independent_batch_collection_complete"] is True
    assert ready["human_evidence_complete"] is False
    encoded = json.dumps(ready, ensure_ascii=False)
    assert all(identity not in encoded for identity in identities.values())
    assert all(ready["slots"][slot]["annotator_hash"] for slot in REVIEWER_SLOTS)


def test_collection_rejects_role_identity_collision_and_tampered_receipt(
    tmp_path: Path,
) -> None:
    collision, _, _, _ = _fixture(tmp_path / "collision")
    _finalize_all(
        collision,
        identities={
            "reviewer_1": "same-private-person",
            "reviewer_2": "same-private-person",
            "adjudicator": "independent-adjudicator",
        },
    )
    collision_report = collision.audit()
    assert collision_report["status"] == "INVALID_REVIEW_COLLECTION"
    assert {item["code"] for item in collision_report["issues"]} == {
        "ROLE_IDENTITY_COLLISION"
    }

    tampered, _, _, _ = _fixture(tmp_path / "tampered")
    _finalize_all(
        tampered,
        identities={
            "reviewer_1": "private-reviewer-one",
            "reviewer_2": "private-reviewer-two",
            "adjudicator": "private-adjudicator",
        },
    )
    _, receipt_path = tampered._private_paths("reviewer_1", 1)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["submission_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    tampered_report = tampered.audit()
    assert tampered_report["status"] == "INVALID_REVIEW_COLLECTION"
    assert tampered_report["slots"]["reviewer_1"]["batches"][0]["status"] == (
        "INVALID"
    )


def test_private_merge_writes_qc_and_comparison_without_opening_mapping(
    tmp_path: Path,
) -> None:
    manager, package_path, routing_path, _ = _fixture(tmp_path)
    identities = {
        "reviewer_1": "private-reviewer-one",
        "reviewer_2": "private-reviewer-two",
        "adjudicator": "private-adjudicator",
    }
    _finalize_all(manager, identities=identities)
    output_root = tmp_path / ".cache" / "private-merge"
    result = manager.merge_private_collection(
        package_path=package_path,
        routing_path=routing_path,
        output_root=output_root,
        minimum_exact_agreement=0.8,
    )

    assert result["status"] == (
        "READY_FOR_BLIND_MAPPING_VERIFICATION_AND_FINALIZATION"
    )
    assert result["inter_annotator_comparison_complete"] is True
    assert result["disagreement_count"] == 0
    assert len(result["private_files"]) == 7
    assert result["blind_mapping_opened"] is False
    assert result["final_adjudication_complete"] is False
    assert result["human_evidence_complete"] is False
    assert all(
        item["status"] == "PASS" for item in result["intrarater_qc"].values()
    )
    encoded = (output_root / "manifest.json").read_text(encoding="utf-8")
    assert all(identity not in encoded for identity in identities.values())

    with pytest.raises(ValueError, match="outputs must remain inside .cache"):
        manager.merge_private_collection(
            package_path=package_path,
            routing_path=routing_path,
            output_root=tmp_path / "tracked-output",
            minimum_exact_agreement=0.8,
        )


def test_private_merge_propagates_qc_failure_and_rejects_routing_tampering(
    tmp_path: Path,
) -> None:
    manager, package_path, routing_path, routing = _fixture(tmp_path)
    repeat_ids = {
        str(row["review_task_id"])
        for row in routing["rows"]
        if row["reviewer_slot"] == "reviewer_1" and row["occurrence"] == "repeat"
    }
    _finalize_all(
        manager,
        identities={
            "reviewer_1": "private-reviewer-one",
            "reviewer_2": "private-reviewer-two",
            "adjudicator": "private-adjudicator",
        },
        disagreeing_repeat_ids=repeat_ids,
    )
    qc_failure_root = tmp_path / ".cache" / "qc-failure"
    _write_json(qc_failure_root / "manifest.json", {"status": "STALE_READY"})
    _write_json(qc_failure_root / "reviewer-comparison.json", {"stale": True})
    result = manager.merge_private_collection(
        package_path=package_path,
        routing_path=routing_path,
        output_root=qc_failure_root,
        minimum_exact_agreement=0.8,
    )
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["inter_annotator_comparison_complete"] is False
    assert result["intrarater_qc"]["reviewer_1"]["status"] == "REVIEW_REQUIRED"
    assert len(result["private_files"]) == 6
    assert not (qc_failure_root / "reviewer-comparison.json").exists()
    assert json.loads(
        (qc_failure_root / "manifest.json").read_text(encoding="utf-8")
    )["status"] == "REVIEW_REQUIRED"

    tampered_routing = copy.deepcopy(routing)
    tampered_routing["rows"][0]["item_id"] = "tampered-item"
    _write_json(routing_path, tampered_routing)
    with pytest.raises(ValueError, match="routing differs"):
        manager.merge_private_collection(
            package_path=package_path,
            routing_path=routing_path,
            output_root=tmp_path / ".cache" / "tampered-routing",
            minimum_exact_agreement=0.8,
        )


def test_collection_status_output_and_manifest_commitment_are_private(
    tmp_path: Path,
) -> None:
    manager, _, _, _ = _fixture(tmp_path)
    output = tmp_path / ".cache" / "collection-status.json"
    report = manager.write_status(output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report

    with pytest.raises(ValueError, match="status must remain inside .cache"):
        manager.write_status(tmp_path / "tracked-status.json")
    with pytest.raises(ValueError, match="manifest hash differs"):
        CollectionManager(
            repository_root=tmp_path,
            public_root=manager.public_root,
            draft_root=manager.draft_root,
            expected_manifest_sha256="0" * 64,
        )
