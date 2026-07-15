from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


EVIDENCE_ROLES = ("condition", "object", "responsibility", "procedure", "exception", "attribution")
BLINDED_METADATA_KEYS = {"evidence_roles", "source_label", "trust_score", "conflicts_with", "stance", "_retrieval_explain"}


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return not normalized or normalized.startswith("REPLACE_WITH_") or normalized.startswith("REQUIRED:")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AnnotationCandidate(BaseModel):
    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationItem(BaseModel):
    item_id: str
    query: str
    slots: list[str]
    candidates: list[AnnotationCandidate] = Field(min_length=1)


class AnnotationPackage(BaseModel):
    package_id: str
    source_name: str
    source_sha256: str
    seed: int
    role_schema: list[str]
    instructions: list[str]
    items: list[AnnotationItem] = Field(min_length=1)
    created_at: datetime


class AnnotationDecision(BaseModel):
    item_id: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    role_labels: dict[str, list[str]] = Field(default_factory=dict)
    reference_answer: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def normalize_unique_values(self):
        self.relevant_doc_ids = list(dict.fromkeys(self.relevant_doc_ids))
        self.role_labels = {
            doc_id: list(dict.fromkeys(roles)) for doc_id, roles in self.role_labels.items()
        }
        return self


class AnnotationSubmission(BaseModel):
    package_id: str
    annotator_id: str = Field(min_length=3)
    decisions: list[AnnotationDecision]
    submitted_at: datetime


class AnnotationComparison(BaseModel):
    package_id: str
    annotator_hashes: list[str]
    evidence_cohen_kappa: float
    mean_evidence_jaccard: float
    role_exact_agreement: float
    answer_exact_agreement: float
    disagreements: list[dict[str, Any]]
    submission_sha256: dict[str, str]
    compared_at: datetime


class AdjudicationResolution(BaseModel):
    item_id: str
    relevant_doc_ids: list[str]
    role_labels: dict[str, list[str]]
    reference_answer: str = ""
    rationale: str = Field(min_length=3)


class AdjudicationSubmission(BaseModel):
    package_id: str
    adjudicator_id: str = Field(min_length=3)
    resolutions: list[AdjudicationResolution]
    submitted_at: datetime


def prepare_annotation_package(source_path: str | Path, *, seed: int = 20260712) -> AnnotationPackage:
    path = Path(source_path)
    source = json.loads(path.read_text(encoding="utf-8"))
    documents = {item["doc_id"]: item for item in source["documents"]}
    rng = random.Random(seed)
    items = []
    for case in source["cases"]:
        candidates = []
        for document in documents.values():
            metadata = {
                key: value
                for key, value in document.get("metadata", {}).items()
                if key not in BLINDED_METADATA_KEYS
            }
            candidates.append(
                AnnotationCandidate(
                    doc_id=document["doc_id"],
                    title=document["title"],
                    content=document["content"],
                    metadata=metadata,
                )
            )
        rng.shuffle(candidates)
        items.append(
            AnnotationItem(
                item_id=case["case_id"],
                query=case["query"],
                slots=list(case.get("slots", [])),
                candidates=candidates,
            )
        )
    unsigned = {
        "source_name": source.get("metadata", {}).get("name", path.name),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": seed,
        "role_schema": list(EVIDENCE_ROLES),
        "items": [item.model_dump(mode="json") for item in items],
    }
    return AnnotationPackage(
        package_id=f"RAG-ANNOTATION-{canonical_sha256(unsigned)[:16]}",
        source_name=unsigned["source_name"],
        source_sha256=unsigned["source_sha256"],
        seed=seed,
        role_schema=list(EVIDENCE_ROLES),
        instructions=[
            "独立完成标注，不查看其他标注员结果、检索分数或原始 gold 标签。",
            "选择回答问题必需且有直接依据的证据文档；不要因主题相关就选择。",
            "对已选证据标注 condition/object/responsibility/procedure/exception/attribution 角色。",
            "reference_answer 只写证据直接支持的简短答案；无充分依据时留空并说明。",
        ],
        items=items,
        created_at=datetime.now(UTC),
    )


def blank_submission(package: AnnotationPackage, annotator_id: str) -> AnnotationSubmission:
    return AnnotationSubmission(
        package_id=package.package_id,
        annotator_id=annotator_id,
        decisions=[AnnotationDecision(item_id=item.item_id) for item in package.items],
        submitted_at=datetime.now(UTC),
    )


def validate_submission(package: AnnotationPackage, submission: AnnotationSubmission) -> None:
    if submission.package_id != package.package_id:
        raise ValueError("annotation submission package_id does not match")
    if _is_placeholder(submission.annotator_id):
        raise ValueError("annotation submission must replace the placeholder annotator_id")
    item_by_id = {item.item_id: item for item in package.items}
    if len(submission.decisions) != len(package.items):
        raise ValueError("annotation submission must contain exactly one decision per item")
    if len({item.item_id for item in submission.decisions}) != len(submission.decisions):
        raise ValueError("annotation submission contains duplicate item_id values")
    for decision in submission.decisions:
        item = item_by_id.get(decision.item_id)
        if item is None:
            raise ValueError(f"unknown annotation item: {decision.item_id}")
        candidate_ids = {candidate.doc_id for candidate in item.candidates}
        if not set(decision.relevant_doc_ids) <= candidate_ids:
            raise ValueError(f"decision {decision.item_id} references an unknown candidate")
        if not decision.relevant_doc_ids and _is_placeholder(decision.notes):
            raise ValueError(f"decision {decision.item_id} must explain why no evidence was selected")
        if set(decision.role_labels) != set(decision.relevant_doc_ids):
            raise ValueError(
                f"decision {decision.item_id} must assign roles to every selected evidence document and no others"
            )
        for roles in decision.role_labels.values():
            if not roles:
                raise ValueError(f"decision {decision.item_id} contains an empty evidence role list")
            if not set(roles) <= set(package.role_schema):
                raise ValueError(f"decision {decision.item_id} contains an unknown evidence role")


def _cohen_kappa(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_positive = sum(left) / len(left)
    right_positive = sum(right) / len(right)
    expected = left_positive * right_positive + (1 - left_positive) * (1 - right_positive)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def compare_submissions(
    package: AnnotationPackage,
    first: AnnotationSubmission,
    second: AnnotationSubmission,
) -> AnnotationComparison:
    validate_submission(package, first)
    validate_submission(package, second)
    if first.annotator_id == second.annotator_id:
        raise ValueError("two distinct annotators are required")
    first_by_id = {item.item_id: item for item in first.decisions}
    second_by_id = {item.item_id: item for item in second.decisions}
    binary_first: list[int] = []
    binary_second: list[int] = []
    jaccards: list[float] = []
    role_pairs = 0
    role_matches = 0
    answer_matches = 0
    disagreements = []
    for item in package.items:
        left = first_by_id[item.item_id]
        right = second_by_id[item.item_id]
        left_set = set(left.relevant_doc_ids)
        right_set = set(right.relevant_doc_ids)
        candidate_ids = [candidate.doc_id for candidate in item.candidates]
        binary_first.extend(int(doc_id in left_set) for doc_id in candidate_ids)
        binary_second.extend(int(doc_id in right_set) for doc_id in candidate_ids)
        union = left_set | right_set
        jaccards.append(len(left_set & right_set) / len(union) if union else 1.0)
        for doc_id in left_set & right_set:
            role_pairs += 1
            role_matches += set(left.role_labels.get(doc_id, [])) == set(right.role_labels.get(doc_id, []))
        normalized_left_answer = " ".join(left.reference_answer.lower().split())
        normalized_right_answer = " ".join(right.reference_answer.lower().split())
        answer_matches += normalized_left_answer == normalized_right_answer
        if left_set != right_set or left.role_labels != right.role_labels or normalized_left_answer != normalized_right_answer:
            disagreements.append(
                {
                    "item_id": item.item_id,
                    "first": left.model_dump(mode="json"),
                    "second": right.model_dump(mode="json"),
                }
            )
    hashes = {
        hashlib.sha256(first.annotator_id.encode("utf-8")).hexdigest()[:16]: canonical_sha256(first.model_dump(mode="json")),
        hashlib.sha256(second.annotator_id.encode("utf-8")).hexdigest()[:16]: canonical_sha256(second.model_dump(mode="json")),
    }
    return AnnotationComparison(
        package_id=package.package_id,
        annotator_hashes=sorted(hashes),
        evidence_cohen_kappa=round(_cohen_kappa(binary_first, binary_second), 4),
        mean_evidence_jaccard=round(sum(jaccards) / len(jaccards), 4),
        role_exact_agreement=round(role_matches / role_pairs, 4) if role_pairs else 1.0,
        answer_exact_agreement=round(answer_matches / len(package.items), 4),
        disagreements=disagreements,
        submission_sha256=hashes,
        compared_at=datetime.now(UTC),
    )


def finalize_adjudication(
    package: AnnotationPackage,
    first: AnnotationSubmission,
    second: AnnotationSubmission,
    adjudication: AdjudicationSubmission,
) -> dict[str, Any]:
    comparison = compare_submissions(package, first, second)
    if adjudication.package_id != package.package_id:
        raise ValueError("adjudication package_id does not match")
    if _is_placeholder(adjudication.adjudicator_id):
        raise ValueError("adjudication must replace the placeholder adjudicator_id")
    if adjudication.adjudicator_id in {first.annotator_id, second.annotator_id}:
        raise ValueError("adjudicator must be independent from both annotators")
    if len(adjudication.resolutions) != len(package.items):
        raise ValueError("adjudication must resolve every annotation item")
    if any(_is_placeholder(item.rationale) for item in adjudication.resolutions):
        raise ValueError("adjudication must replace every placeholder rationale")
    decisions = AnnotationSubmission(
        package_id=package.package_id,
        annotator_id=adjudication.adjudicator_id,
        decisions=[
            AnnotationDecision(
                item_id=item.item_id,
                relevant_doc_ids=item.relevant_doc_ids,
                role_labels=item.role_labels,
                reference_answer=item.reference_answer,
                notes=item.rationale,
            )
            for item in adjudication.resolutions
        ],
        submitted_at=adjudication.submitted_at,
    )
    validate_submission(package, decisions)
    candidates: dict[str, AnnotationCandidate] = {}
    for item in package.items:
        for candidate in item.candidates:
            candidates[candidate.doc_id] = candidate
    resolution_by_id = {item.item_id: item for item in adjudication.resolutions}
    cases = []
    for item in package.items:
        resolution = resolution_by_id[item.item_id]
        roles = sorted({role for values in resolution.role_labels.values() for role in values})
        cases.append(
            {
                "case_id": item.item_id,
                "query": item.query,
                "slots": item.slots,
                "relevant_doc_ids": resolution.relevant_doc_ids,
                "required_roles": roles,
                "reference_answer": resolution.reference_answer,
            }
        )
    dataset = {
        "metadata": {
            "name": f"{package.source_name} - independent adjudicated annotations",
            "source_package_id": package.package_id,
            "source_sha256": package.source_sha256,
            "annotator_hashes": comparison.annotator_hashes,
            "adjudicator_hash": hashlib.sha256(adjudication.adjudicator_id.encode("utf-8")).hexdigest()[:16],
            "agreement": comparison.model_dump(mode="json", exclude={"disagreements"}),
            "finalized_at": datetime.now(UTC).isoformat(),
        },
        "documents": [candidate.model_dump(mode="json") | {"corpus": "policy"} for candidate in candidates.values()],
        "cases": cases,
    }
    dataset["metadata"]["dataset_sha256"] = canonical_sha256({"documents": dataset["documents"], "cases": cases})
    return dataset
