"""Method-blind batching and quality control for CONFLICTS human review."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

from research.frc_rag.conflicts_expected_behavior import (
    ALIASES,
    ANSWER_RATINGS,
    PREFERENCES,
    RATING_FIELDS,
    SUBMISSION_SCHEMA_VERSION,
    TERNARY_RATINGS,
    canonical_json_sha256,
    validate_submission,
)


OPERATIONS_SCHEMA_VERSION = "frc-conflicts-annotation-operations-v1"
BATCH_SCHEMA_VERSION = "frc-conflicts-annotation-batch-v1"
BATCH_SUBMISSION_SCHEMA_VERSION = "frc-conflicts-annotation-batch-submission-v1"
ROUTING_SCHEMA_VERSION = "frc-conflicts-annotation-routing-v1"
QC_SCHEMA_VERSION = "frc-conflicts-annotation-intrarater-qc-v1"
REVIEWER_SLOTS = ("reviewer_1", "reviewer_2", "adjudicator")


def _stable_hash(*parts: Any) -> str:
    return canonical_json_sha256(list(parts))


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return len(value.strip()) < 3 or normalized.startswith("REPLACE_WITH")


def _item_stratum(item: dict[str, Any]) -> str:
    answer_state = "answer_available" if item.get("correct_answer") is not None else "no_answer"
    return f"{item['conflict_type']}|{answer_state}"


def _repeat_allocations(
    items: list[dict[str, Any]], repeat_count: int
) -> dict[str, int]:
    by_label = Counter(str(item["conflict_type"]) for item in items)
    labels = sorted(by_label)
    if repeat_count < len(labels):
        raise ValueError("repeat_count must cover every conflict type")
    if repeat_count > len(items):
        raise ValueError("repeat_count cannot exceed the package case count")
    allocations = {label: 1 for label in labels}
    targets = {
        label: repeat_count * by_label[label] / len(items) for label in labels
    }
    while sum(allocations.values()) < repeat_count:
        candidates = [
            label for label in labels if allocations[label] < by_label[label]
        ]
        if not candidates:
            raise ValueError("repeat allocation exhausted the available cases")
        selected = max(
            candidates,
            key=lambda label: (
                targets[label] - allocations[label],
                by_label[label],
                label,
            ),
        )
        allocations[selected] += 1
    return allocations


def _blank_ratings(item: dict[str, Any]) -> dict[str, dict[str, str]]:
    answer_value = (
        "NOT_APPLICABLE" if item.get("correct_answer") is None else "REQUIRED"
    )
    return {
        alias: {
            "expected_behavior_adherence": "REQUIRED",
            "factual_grounding": "REQUIRED",
            "citation_correctness": "REQUIRED",
            "answer_correctness": answer_value,
            "rationale": "REQUIRED: explain all four ratings",
        }
        for alias in ALIASES
    }


def _public_task(item: dict[str, Any], task_id: str) -> dict[str, Any]:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "review_task_id": task_id,
        "question": item["question"],
        "conflict_type": item["conflict_type"],
        "expected_behavior": item["expected_behavior"],
        "correct_answer": item.get("correct_answer"),
        "responses": item["responses"],
    }


def _blank_task_decision(item: dict[str, Any], task_id: str) -> dict[str, Any]:
    return {
        "review_task_id": task_id,
        "ratings": _blank_ratings(item),
        "preference": "REQUIRED",
        "notes": "",
    }


def build_annotation_operations(
    items: list[dict[str, Any]],
    *,
    package_id: str,
    seed: str,
    batch_count: int,
    repeat_count: int,
    minimum_repeat_batch_distance: int,
    reviewer_slots: tuple[str, ...] = REVIEWER_SLOTS,
) -> dict[str, Any]:
    """Create deterministic public batches and a private task-to-case routing map."""

    if not items:
        raise ValueError("CONFLICTS behavior package is empty")
    if batch_count < 2 or batch_count % 2:
        raise ValueError("batch_count must be an even integer of at least two")
    if not reviewer_slots or len(set(reviewer_slots)) != len(reviewer_slots):
        raise ValueError("reviewer slots must be non-empty and unique")
    half_turn = batch_count // 2
    if minimum_repeat_batch_distance > half_turn:
        raise ValueError("minimum repeat distance exceeds the balanced half-turn")
    item_ids = [str(item["item_id"]) for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("CONFLICTS behavior package contains duplicate item ids")

    operation_signature = {
        "package_id": package_id,
        "seed": seed,
        "batch_count": batch_count,
        "repeat_count": repeat_count,
        "minimum_repeat_batch_distance": minimum_repeat_batch_distance,
        "reviewer_slots": list(reviewer_slots),
        "item_ids_sha256": canonical_json_sha256(item_ids),
    }
    operations_id = (
        "CONFLICTS-ANNOTATION-OPS-"
        + canonical_json_sha256(operation_signature)[:16].upper()
    )
    item_by_id = {str(item["item_id"]): item for item in items}

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_stratum[_item_stratum(item)].append(str(item["item_id"]))
    primary_base_batch: dict[str, int] = {}
    for stratum, stratum_ids in sorted(by_stratum.items()):
        ordered = sorted(
            stratum_ids,
            key=lambda item_id: _stable_hash(seed, "primary", stratum, item_id),
        )
        offset = int(_stable_hash(seed, "offset", stratum)[:8], 16) % batch_count
        for index, item_id in enumerate(ordered):
            primary_base_batch[item_id] = (offset + index) % batch_count

    repeat_allocations = _repeat_allocations(items, repeat_count)
    repeat_ids: list[str] = []
    for label, count in sorted(repeat_allocations.items()):
        candidates = sorted(
            (
                str(item["item_id"])
                for item in items
                if str(item["conflict_type"]) == label
            ),
            key=lambda item_id: _stable_hash(seed, "repeat", label, item_id),
        )
        repeat_ids.extend(candidates[:count])
    repeat_ids = sorted(repeat_ids, key=lambda value: _stable_hash(seed, "repeat-order", value))

    routing_rows: list[dict[str, Any]] = []
    batch_tasks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    slot_rotation_step = max(1, batch_count // len(reviewer_slots))
    for slot_index, slot in enumerate(reviewer_slots):
        rotation = (slot_index * slot_rotation_step) % batch_count
        primary_task_by_item: dict[str, str] = {}
        for item_id in item_ids:
            batch_index = (primary_base_batch[item_id] + rotation) % batch_count
            task_id = (
                "TASK-"
                + _stable_hash(operations_id, slot, item_id, "primary")[:16].upper()
            )
            primary_task_by_item[item_id] = task_id
            routing_rows.append(
                {
                    "reviewer_slot": slot,
                    "review_task_id": task_id,
                    "item_id": item_id,
                    "occurrence": "primary",
                    "batch_index": batch_index + 1,
                    "paired_primary_task_id": task_id,
                }
            )
            batch_tasks[(slot, batch_index)].append(
                _public_task(item_by_id[item_id], task_id)
            )
        for item_id in repeat_ids:
            primary_batch = (primary_base_batch[item_id] + rotation) % batch_count
            repeat_batch = (primary_batch + half_turn) % batch_count
            if abs(repeat_batch - primary_batch) < minimum_repeat_batch_distance:
                raise AssertionError("repeat task violates the frozen batch distance")
            task_id = (
                "TASK-"
                + _stable_hash(operations_id, slot, item_id, "repeat")[:16].upper()
            )
            routing_rows.append(
                {
                    "reviewer_slot": slot,
                    "review_task_id": task_id,
                    "item_id": item_id,
                    "occurrence": "repeat",
                    "batch_index": repeat_batch + 1,
                    "paired_primary_task_id": primary_task_by_item[item_id],
                }
            )
            batch_tasks[(slot, repeat_batch)].append(
                _public_task(item_by_id[item_id], task_id)
            )

    batches: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    item_id_by_task = {
        str(row["review_task_id"]): str(row["item_id"]) for row in routing_rows
    }
    for slot in reviewer_slots:
        for batch_index in range(batch_count):
            tasks = sorted(
                batch_tasks[(slot, batch_index)],
                key=lambda task: _stable_hash(
                    seed,
                    "task-order",
                    slot,
                    batch_index,
                    task["review_task_id"],
                ),
            )
            batch_id = f"{slot.upper()}-B{batch_index + 1:02d}"
            batches.append(
                {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "operations_id": operations_id,
                    "package_id": package_id,
                    "reviewer_slot": slot,
                    "batch_id": batch_id,
                    "batch_index": batch_index + 1,
                    "batch_count": batch_count,
                    "task_count": len(tasks),
                    "instructions": [
                        "Complete this batch independently without viewing another person's decisions.",
                        "Do not change review_task_id, batch_id, package_id, or operations_id.",
                        "Use the same private annotator_id in every batch assigned to this reviewer slot.",
                        "Some tasks may repeat for within-reviewer quality control; rate every task independently.",
                        "Do not infer or record retrieval method identity.",
                    ],
                    "tasks": tasks,
                }
            )
            templates.append(
                {
                    "schema_version": BATCH_SUBMISSION_SCHEMA_VERSION,
                    "operations_id": operations_id,
                    "package_id": package_id,
                    "reviewer_slot": slot,
                    "batch_id": batch_id,
                    "annotator_id": f"REPLACE_WITH_PRIVATE_{slot.upper()}_ID",
                    "decisions": [
                        _blank_task_decision(
                            item_by_id[item_id_by_task[task["review_task_id"]]],
                            task["review_task_id"],
                        )
                        for task in tasks
                    ],
                }
            )

    routing_rows.sort(
        key=lambda row: (
            reviewer_slots.index(str(row["reviewer_slot"])),
            int(row["batch_index"]),
            str(row["review_task_id"]),
        )
    )
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "operations_id": operations_id,
        "package_id": package_id,
        "seed": seed,
        "rows": routing_rows,
    }
    return {
        "schema_version": OPERATIONS_SCHEMA_VERSION,
        "operations_id": operations_id,
        "package_id": package_id,
        "configuration": operation_signature,
        "repeat_item_ids": repeat_ids,
        "repeat_counts_by_conflict_type": repeat_allocations,
        "primary_counts_by_stratum": {
            stratum: len(values) for stratum, values in sorted(by_stratum.items())
        },
        "batches": batches,
        "templates": templates,
        "routing": routing,
    }


def _categorical_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _validate_task_decision(
    item: dict[str, Any], decision: dict[str, Any]
) -> None:
    if set(decision.get("ratings", {})) != set(ALIASES):
        raise ValueError("batch decision must rate both blinded responses")
    for rating in decision["ratings"].values():
        if rating.get("expected_behavior_adherence") not in TERNARY_RATINGS:
            raise ValueError("invalid expected-behavior rating")
        if rating.get("factual_grounding") not in TERNARY_RATINGS:
            raise ValueError("invalid factual-grounding rating")
        if rating.get("citation_correctness") not in TERNARY_RATINGS:
            raise ValueError("invalid citation-correctness rating")
        if rating.get("answer_correctness") not in ANSWER_RATINGS:
            raise ValueError("invalid answer-correctness rating")
        if item.get("correct_answer") is None:
            if rating["answer_correctness"] != "NOT_APPLICABLE":
                raise ValueError(
                    "answer correctness must be NOT_APPLICABLE without a gold answer"
                )
        elif rating["answer_correctness"] == "NOT_APPLICABLE":
            raise ValueError(
                "answer correctness cannot be NOT_APPLICABLE when gold exists"
            )
        if len(str(rating.get("rationale", "")).strip()) < 3:
            raise ValueError("every batch response rating requires a rationale")
    if decision.get("preference") not in PREFERENCES:
        raise ValueError("invalid batch pair preference")


def merge_batch_submissions(
    items: list[dict[str, Any]],
    *,
    operations_manifest: dict[str, Any],
    routing: dict[str, Any],
    reviewer_slot: str,
    submissions: list[dict[str, Any]],
    minimum_exact_agreement: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge one role's completed batches into the frozen core submission schema."""

    if reviewer_slot not in operations_manifest["reviewer_slots"]:
        raise ValueError("unknown reviewer slot")
    if not 0.0 <= minimum_exact_agreement <= 1.0:
        raise ValueError("minimum exact agreement must be between zero and one")
    if canonical_json_sha256(routing) != operations_manifest["routing_sha256"]:
        raise ValueError("annotation routing map does not match its public commitment")
    if routing.get("operations_id") != operations_manifest.get("operations_id"):
        raise ValueError("annotation routing operation id differs")
    item_by_id = {str(item["item_id"]): item for item in items}
    slot_rows = [
        row for row in routing["rows"] if row["reviewer_slot"] == reviewer_slot
    ]
    rows_by_task = {str(row["review_task_id"]): row for row in slot_rows}
    expected_batches: dict[str, list[str]] = defaultdict(list)
    for row in slot_rows:
        batch_id = f"{reviewer_slot.upper()}-B{int(row['batch_index']):02d}"
        expected_batches[batch_id].append(str(row["review_task_id"]))
    for batch_id in expected_batches:
        expected_batches[batch_id].sort(
            key=lambda task_id: _stable_hash(
                operations_manifest["seed"],
                "task-order",
                reviewer_slot,
                int(batch_id.rsplit("B", 1)[1]) - 1,
                task_id,
            )
        )

    submission_by_batch: dict[str, dict[str, Any]] = {}
    annotator_ids: set[str] = set()
    for submission in submissions:
        if submission.get("schema_version") != BATCH_SUBMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported annotation batch submission schema")
        if submission.get("operations_id") != operations_manifest["operations_id"]:
            raise ValueError("batch submission operation id differs")
        if submission.get("package_id") != operations_manifest["package_id"]:
            raise ValueError("batch submission package id differs")
        if submission.get("reviewer_slot") != reviewer_slot:
            raise ValueError("batch submission reviewer slot differs")
        batch_id = str(submission.get("batch_id", ""))
        if batch_id in submission_by_batch:
            raise ValueError("duplicate annotation batch submission")
        submission_by_batch[batch_id] = submission
        annotator_id = str(submission.get("annotator_id", ""))
        if _is_placeholder(annotator_id):
            raise ValueError("batch submission requires a non-placeholder annotator id")
        annotator_ids.add(annotator_id)
    if set(submission_by_batch) != set(expected_batches):
        raise ValueError("batch submissions do not cover every expected batch exactly once")
    if len(annotator_ids) != 1:
        raise ValueError("all batches for one reviewer slot must use the same annotator id")

    decision_by_task: dict[str, dict[str, Any]] = {}
    for batch_id, expected_task_ids in expected_batches.items():
        decisions = submission_by_batch[batch_id].get("decisions", [])
        actual_task_ids = [str(item.get("review_task_id", "")) for item in decisions]
        if actual_task_ids != expected_task_ids:
            raise ValueError("batch decisions must preserve the frozen task order")
        for decision in decisions:
            task_id = str(decision["review_task_id"])
            row = rows_by_task[task_id]
            _validate_task_decision(item_by_id[str(row["item_id"])], decision)
            decision_by_task[task_id] = decision

    primary_by_item = {
        str(row["item_id"]): decision_by_task[str(row["review_task_id"])]
        for row in slot_rows
        if row["occurrence"] == "primary"
    }
    merged = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "package_id": operations_manifest["package_id"],
        "annotator_id": next(iter(annotator_ids)),
        "decisions": [
            {
                "item_id": str(item["item_id"]),
                "ratings": primary_by_item[str(item["item_id"])]["ratings"],
                "preference": primary_by_item[str(item["item_id"])]["preference"],
                "notes": primary_by_item[str(item["item_id"])].get("notes", ""),
            }
            for item in items
        ],
    }
    validate_submission(items, merged)

    left_values: dict[str, list[str]] = defaultdict(list)
    right_values: dict[str, list[str]] = defaultdict(list)
    repeat_disagreements: list[dict[str, Any]] = []
    comparison_unit_count = 0
    exact_comparison_count = 0
    for row in slot_rows:
        if row["occurrence"] != "repeat":
            continue
        repeat = decision_by_task[str(row["review_task_id"])]
        primary = decision_by_task[str(row["paired_primary_task_id"])]
        row_disagrees = False
        for field in RATING_FIELDS:
            for alias in ALIASES:
                left = str(primary["ratings"][alias][field])
                right = str(repeat["ratings"][alias][field])
                left_values[field].append(left)
                right_values[field].append(right)
                comparison_unit_count += 1
                exact_comparison_count += left == right
                row_disagrees = row_disagrees or left != right
        left_preference = str(primary["preference"])
        right_preference = str(repeat["preference"])
        left_values["preference"].append(left_preference)
        right_values["preference"].append(right_preference)
        comparison_unit_count += 1
        exact_comparison_count += left_preference == right_preference
        row_disagrees = row_disagrees or left_preference != right_preference
        if row_disagrees:
            repeat_disagreements.append(
                {
                    "primary_review_task_id": row["paired_primary_task_id"],
                    "repeat_review_task_id": row["review_task_id"],
                }
            )

    agreement = {}
    for field in (*RATING_FIELDS, "preference"):
        kappa = _categorical_kappa(left_values[field], right_values[field])
        agreement[field] = {
            "exact_agreement": round(
                sum(
                    left == right
                    for left, right in zip(
                        left_values[field], right_values[field], strict=True
                    )
                )
                / len(left_values[field]),
                6,
            ),
            "cohen_kappa": None if kappa is None else round(kappa, 6),
            "cohen_kappa_status": (
                "UNDEFINED_NO_VARIANCE" if kappa is None else "DEFINED"
            ),
            "comparison_count": len(left_values[field]),
        }
    eligible = all(
        metrics["exact_agreement"] >= minimum_exact_agreement
        for metrics in agreement.values()
    )
    qc = {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "PASS" if eligible else "REVIEW_REQUIRED",
        "operations_id": operations_manifest["operations_id"],
        "package_id": operations_manifest["package_id"],
        "reviewer_slot": reviewer_slot,
        "annotator_hash": hashlib.sha256(
            merged["annotator_id"].encode("utf-8")
        ).hexdigest()[:16],
        "batch_count": len(expected_batches),
        "primary_decision_count": len(merged["decisions"]),
        "repeat_decision_count": sum(
            row["occurrence"] == "repeat" for row in slot_rows
        ),
        "minimum_exact_agreement": minimum_exact_agreement,
        "eligible_for_inter_annotator_comparison": eligible,
        "overall_exact_agreement": round(
            exact_comparison_count / comparison_unit_count, 6
        ),
        "agreement": agreement,
        "repeat_disagreement_count": len(repeat_disagreements),
        "repeat_disagreements": repeat_disagreements,
        "merged_submission_sha256": canonical_json_sha256(merged),
    }
    return merged, qc
