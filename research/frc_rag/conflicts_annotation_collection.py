"""Private collection audit and merge orchestration for CONFLICTS review batches."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from research.frc_rag.conflicts_annotation_operations import (
    REVIEWER_SLOTS,
    merge_batch_submissions,
)
from research.frc_rag.conflicts_annotation_workstation import (
    EXPECTED_OPERATIONS_MANIFEST_SHA256,
    RECEIPT_SCHEMA_VERSION,
    WorkstationSession,
)
from research.frc_rag.conflicts_expected_behavior import (
    canonical_json_sha256,
    compare_submissions,
    read_jsonl,
)


COLLECTION_STATUS_SCHEMA_VERSION = "frc-conflicts-annotation-collection-status-v1"
PRIVATE_MERGE_MANIFEST_SCHEMA_VERSION = (
    "frc-conflicts-annotation-private-merge-manifest-v1"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _identity_hash(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _BatchInspection:
    public: dict[str, Any]
    identity: str | None
    finalized: bool
    valid: bool


class CollectionManager:
    """Audit all private batches and merge them only after receipt verification."""

    def __init__(
        self,
        *,
        repository_root: Path,
        public_root: Path,
        draft_root: Path,
        expected_manifest_sha256: str = EXPECTED_OPERATIONS_MANIFEST_SHA256,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.public_root = public_root.resolve()
        self.draft_root = draft_root.resolve()
        self.expected_manifest_sha256 = expected_manifest_sha256
        cache_root = (self.repository_root / ".cache").resolve()
        if not _is_relative_to(self.draft_root, cache_root):
            raise ValueError("annotation collection drafts must remain inside .cache")

        self.manifest_path = self.public_root / "manifest.json"
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
            or not self.manifest_path.is_file()
            or not hmac.compare_digest(
                _file_sha256(self.manifest_path), expected_manifest_sha256
            )
        ):
            raise ValueError(
                "annotation operations manifest hash differs from frozen contract"
            )
        self.manifest = _load_json(self.manifest_path)
        if self.manifest.get("status") != (
            "PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW"
        ):
            raise ValueError("annotation operations are not ready for collection")
        if tuple(self.manifest.get("reviewer_slots", [])) != REVIEWER_SLOTS:
            raise ValueError("annotation collection requires all three frozen roles")
        self.batch_count = int(self.manifest.get("batch_count", 0))
        if self.batch_count <= 0:
            raise ValueError("annotation collection batch count is invalid")
        self._validate_public_files()

    def _validate_public_files(self) -> None:
        expected_pairs = {
            (slot, index, kind)
            for slot in REVIEWER_SLOTS
            for index in range(1, self.batch_count + 1)
            for kind in ("batch", "submission_template")
        }
        observed_pairs: set[tuple[str, int, str]] = set()
        observed_paths: set[str] = set()
        for entry in self.manifest.get("public_files", []):
            relative_label = str(entry.get("path", ""))
            relative_path = Path(relative_label)
            if (
                not relative_label
                or relative_path.is_absolute()
                or relative_path.drive
                or ".." in relative_path.parts
                or relative_path.as_posix() != relative_label
                or relative_label in observed_paths
            ):
                raise ValueError("annotation manifest contains an unsafe public path")
            path = (self.public_root / relative_path).resolve()
            if (
                not _is_relative_to(path, self.public_root)
                or not path.is_file()
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                is None
                or not hmac.compare_digest(_file_sha256(path), str(entry["sha256"]))
            ):
                raise ValueError("annotation public file differs from its manifest")
            slot = str(entry.get("reviewer_slot", ""))
            batch_id = str(entry.get("batch_id", ""))
            match = re.fullmatch(rf"{re.escape(slot.upper())}-B(\d{{2}})", batch_id)
            if match is None:
                raise ValueError("annotation public file batch id is invalid")
            pair = (slot, int(match.group(1)), str(entry.get("kind", "")))
            if pair in observed_pairs:
                raise ValueError("annotation manifest contains a duplicate batch file")
            observed_pairs.add(pair)
            observed_paths.add(relative_label)
        if observed_pairs != expected_pairs:
            raise ValueError("annotation manifest does not cover every frozen batch pair")

    def _private_paths(self, reviewer_slot: str, batch_index: int) -> tuple[Path, Path]:
        slot_directory = self.draft_root / reviewer_slot.replace("_", "-")
        stem = f"batch-{batch_index:02d}"
        return (
            slot_directory / f"{stem}-submission.json",
            slot_directory / f"{stem}-receipt.json",
        )

    def _inspect_batch(self, reviewer_slot: str, batch_index: int) -> _BatchInspection:
        draft_path, receipt_path = self._private_paths(reviewer_slot, batch_index)
        batch_id = f"{reviewer_slot.upper()}-B{batch_index:02d}"
        base = {"batch_id": batch_id, "batch_index": batch_index}
        if not draft_path.exists() and not receipt_path.exists():
            return _BatchInspection(
                public={**base, "status": "MISSING", "completed_task_count": 0},
                identity=None,
                finalized=False,
                valid=True,
            )
        if not draft_path.is_file():
            return _BatchInspection(
                public={**base, "status": "INVALID", "reason": "draft_missing"},
                identity=None,
                finalized=False,
                valid=False,
            )
        try:
            session = WorkstationSession(
                repository_root=self.repository_root,
                public_root=self.public_root,
                draft_root=self.draft_root,
                reviewer_slot=reviewer_slot,
                batch_index=batch_index,
                expected_manifest_sha256=self.expected_manifest_sha256,
            )
            progress = session.progress()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return _BatchInspection(
                public={
                    **base,
                    "status": "INVALID",
                    "reason": "draft_validation_failed",
                    "error": str(exc),
                },
                identity=None,
                finalized=False,
                valid=False,
            )
        identity = (
            str(session.draft["annotator_id"])
            if progress["identity_ready"]
            else None
        )
        if not receipt_path.exists():
            return _BatchInspection(
                public={
                    **base,
                    "status": (
                        "COMPLETE_NOT_FINALIZED"
                        if progress["batch_complete"]
                        else "IN_PROGRESS"
                    ),
                    "completed_task_count": progress["completed_task_count"],
                    "total_task_count": progress["total_task_count"],
                },
                identity=identity,
                finalized=False,
                valid=True,
            )
        try:
            receipt = _load_json(receipt_path)
            expected_fields = {
                "schema_version",
                "status",
                "operations_id",
                "package_id",
                "reviewer_slot",
                "batch_id",
                "submission_sha256",
                "finalized_at",
                "human_evidence_complete",
                "gate_2",
            }
            finalized_at = datetime.fromisoformat(str(receipt.get("finalized_at", "")))
            receipt_valid = (
                set(receipt) == expected_fields
                and receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
                and receipt.get("status") == "FINALIZED_PRIVATE_BATCH"
                and receipt.get("operations_id") == self.manifest["operations_id"]
                and receipt.get("package_id") == self.manifest["package_id"]
                and receipt.get("reviewer_slot") == reviewer_slot
                and receipt.get("batch_id") == batch_id
                and receipt.get("submission_sha256")
                == canonical_json_sha256(session.draft)
                and finalized_at.tzinfo is not None
                and receipt.get("human_evidence_complete") is False
                and receipt.get("gate_2") == "NO-GO/SHADOW"
                and progress["batch_complete"] is True
            )
        except (OSError, ValueError, KeyError, TypeError):
            receipt_valid = False
        if not receipt_valid:
            return _BatchInspection(
                public={
                    **base,
                    "status": "INVALID",
                    "reason": "receipt_validation_failed",
                    "completed_task_count": progress["completed_task_count"],
                    "total_task_count": progress["total_task_count"],
                },
                identity=identity,
                finalized=False,
                valid=False,
            )
        return _BatchInspection(
            public={
                **base,
                "status": "FINALIZED",
                "completed_task_count": progress["completed_task_count"],
                "total_task_count": progress["total_task_count"],
            },
            identity=identity,
            finalized=True,
            valid=True,
        )

    def audit(self) -> dict[str, Any]:
        slot_reports: dict[str, dict[str, Any]] = {}
        slot_identities: dict[str, str] = {}
        issues: list[dict[str, str]] = []
        finalized_total = 0
        draft_total = 0
        for slot in REVIEWER_SLOTS:
            inspections = [
                self._inspect_batch(slot, index)
                for index in range(1, self.batch_count + 1)
            ]
            identities = {
                inspection.identity
                for inspection in inspections
                if inspection.identity is not None
            }
            finalized_count = sum(item.finalized for item in inspections)
            present_count = sum(item.public["status"] != "MISSING" for item in inspections)
            finalized_total += finalized_count
            draft_total += present_count
            if any(not item.valid for item in inspections):
                issues.append({"scope": slot, "code": "INVALID_BATCH_OR_RECEIPT"})
            if len(identities) > 1:
                issues.append({"scope": slot, "code": "INCONSISTENT_ANNOTATOR_ID"})
            if len(identities) == 1:
                slot_identities[slot] = next(iter(identities))
            slot_reports[slot] = {
                "batch_count": self.batch_count,
                "present_batch_count": present_count,
                "finalized_batch_count": finalized_count,
                "annotator_hash": (
                    _identity_hash(slot_identities[slot])
                    if slot in slot_identities and len(identities) == 1
                    else None
                ),
                "batches": [item.public for item in inspections],
            }

        if len(slot_identities) > 1 and len(set(slot_identities.values())) != len(
            slot_identities
        ):
            issues.append({"scope": "all_roles", "code": "ROLE_IDENTITY_COLLISION"})
        expected_total = self.batch_count * len(REVIEWER_SLOTS)
        ready = (
            not issues
            and finalized_total == expected_total
            and len(slot_identities) == len(REVIEWER_SLOTS)
        )
        status = (
            "INVALID_REVIEW_COLLECTION"
            if issues
            else (
                "READY_FOR_PRIVATE_MERGE"
                if ready
                else "AWAITING_INDEPENDENT_HUMAN_BATCHES"
            )
        )
        return {
            "schema_version": COLLECTION_STATUS_SCHEMA_VERSION,
            "status": status,
            "operations_id": self.manifest["operations_id"],
            "package_id": self.manifest["package_id"],
            "expected_batch_count": expected_total,
            "present_batch_count": draft_total,
            "finalized_batch_count": finalized_total,
            "independent_role_identity_count": len(set(slot_identities.values())),
            "slots": slot_reports,
            "issues": issues,
            "private_decisions_present": draft_total > 0,
            "independent_batch_collection_complete": ready,
            "human_evidence_complete": False,
            "blind_mapping_opened": False,
            "gate_2": "NO-GO/SHADOW",
        }

    def write_status(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        cache_root = (self.repository_root / ".cache").resolve()
        if not _is_relative_to(resolved, cache_root):
            raise ValueError("annotation collection status must remain inside .cache")
        report = self.audit()
        _atomic_write_json(resolved, report)
        return report

    def merge_private_collection(
        self,
        *,
        package_path: Path,
        routing_path: Path,
        output_root: Path,
        minimum_exact_agreement: float,
    ) -> dict[str, Any]:
        cache_root = (self.repository_root / ".cache").resolve()
        resolved_routing = routing_path.resolve()
        resolved_output = output_root.resolve()
        if not _is_relative_to(resolved_routing, cache_root):
            raise ValueError("annotation routing must remain inside .cache")
        if not _is_relative_to(resolved_output, cache_root):
            raise ValueError("annotation merge outputs must remain inside .cache")
        status = self.audit()
        if status["status"] != "READY_FOR_PRIVATE_MERGE":
            raise ValueError("annotation collection is not ready for private merge")
        resolved_package = package_path.resolve()
        if (
            not resolved_package.is_file()
            or _file_sha256(resolved_package)
            != self.manifest.get("package_file_sha256")
        ):
            raise ValueError("annotation package differs from the frozen manifest")
        if not resolved_routing.is_file():
            raise ValueError("annotation private routing file is missing")
        routing = _load_json(resolved_routing)
        if canonical_json_sha256(routing) != self.manifest["routing_sha256"]:
            raise ValueError("annotation routing differs from its public commitment")
        items = read_jsonl(resolved_package)

        merged_by_slot: dict[str, dict[str, Any]] = {}
        qc_by_slot: dict[str, dict[str, Any]] = {}
        for slot in REVIEWER_SLOTS:
            submissions = [
                _load_json(self._private_paths(slot, index)[0])
                for index in range(1, self.batch_count + 1)
            ]
            merged, qc = merge_batch_submissions(
                items,
                operations_manifest=self.manifest,
                routing=routing,
                reviewer_slot=slot,
                submissions=submissions,
                minimum_exact_agreement=minimum_exact_agreement,
            )
            merged_by_slot[slot] = merged
            qc_by_slot[slot] = qc

        manifest_path = resolved_output / "manifest.json"
        comparison_path = resolved_output / "reviewer-comparison.json"
        if manifest_path.exists():
            manifest_path.unlink()
        private_files: list[dict[str, str]] = []
        for slot in REVIEWER_SLOTS:
            merged = merged_by_slot[slot]
            qc = qc_by_slot[slot]
            for kind, payload in (("merged_submission", merged), ("intrarater_qc", qc)):
                suffix = "submission" if kind == "merged_submission" else "qc"
                path = resolved_output / slot.replace("_", "-") / f"{suffix}.json"
                _atomic_write_json(path, payload)
                private_files.append(
                    {
                        "kind": kind,
                        "reviewer_slot": slot,
                        "path": path.relative_to(resolved_output).as_posix(),
                        "sha256": _file_sha256(path),
                    }
                )

        all_qc_passed = all(
            qc["status"] == "PASS"
            and qc["eligible_for_inter_annotator_comparison"] is True
            for qc in qc_by_slot.values()
        )
        comparison: dict[str, Any] | None = None
        if all_qc_passed:
            comparison = compare_submissions(
                items,
                merged_by_slot["reviewer_1"],
                merged_by_slot["reviewer_2"],
            )
            _atomic_write_json(comparison_path, comparison)
            private_files.append(
                {
                    "kind": "inter_annotator_comparison",
                    "path": comparison_path.relative_to(resolved_output).as_posix(),
                    "sha256": _file_sha256(comparison_path),
                }
            )
        elif comparison_path.exists():
            comparison_path.unlink()
        merge_status = (
            "READY_FOR_BLIND_MAPPING_VERIFICATION_AND_FINALIZATION"
            if all_qc_passed
            else "REVIEW_REQUIRED"
        )
        manifest = {
            "schema_version": PRIVATE_MERGE_MANIFEST_SCHEMA_VERSION,
            "status": merge_status,
            "operations_id": self.manifest["operations_id"],
            "package_id": self.manifest["package_id"],
            "source_collection_status": status["status"],
            "reviewer_slots": list(REVIEWER_SLOTS),
            "batch_count_per_slot": self.batch_count,
            "minimum_exact_agreement": minimum_exact_agreement,
            "intrarater_qc": {
                slot: {
                    "status": qc["status"],
                    "eligible_for_inter_annotator_comparison": qc[
                        "eligible_for_inter_annotator_comparison"
                    ],
                    "annotator_hash": qc["annotator_hash"],
                    "overall_exact_agreement": qc["overall_exact_agreement"],
                }
                for slot, qc in qc_by_slot.items()
            },
            "inter_annotator_comparison_complete": comparison is not None,
            "disagreement_count": (
                comparison["disagreement_count"] if comparison is not None else None
            ),
            "private_files": private_files,
            "blind_mapping_opened": False,
            "final_adjudication_complete": False,
            "human_evidence_complete": False,
            "gate_2": "NO-GO/SHADOW",
            "limitations": [
                "Private merge readiness is not final adjudication or a public result.",
                "A REVIEW_REQUIRED status blocks inter-annotator comparison and finalization.",
                "The blind mapping remains closed until all registered human checks pass.",
            ],
        }
        _atomic_write_json(manifest_path, manifest)
        return manifest
