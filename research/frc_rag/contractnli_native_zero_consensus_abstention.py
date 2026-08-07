"""Prospective ContractNLI test of a model-native missing-evidence guard."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    FrozenEvidenceInferenceScorer,
    _round_for_display,
    select_v49,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    merge_scored_candidates,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    BUDGETS,
    NON_FRC_BASELINES,
)


SCHEMA_VERSION = "frc-contractnli-native-zero-consensus-abstention-v50"
EXPERIMENT_ID = "FRC-CONTRACTNLI-NATIVE-ZERO-CONSENSUS-ABSTENTION-V50"
DATASET_ID = "contractnli_official_test_bounded_span_selection_v50"
CAPABILITY = "full_contract_span_evidence_selection_and_missing_evidence_abstention"
PROTOCOL_SHA256 = "519b6105cade0869dde3a5b0805bb06022319ab0b8234ade6e8e77f28efebc86"
OFFICIAL_REPOSITORY_REVISION = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"

LABELS = ("Entailment", "Contradiction", "NotMentioned")
EVIDENCE_LABELS = ("Entailment", "Contradiction")
SAMPLE_SALT = "FRC-CONTRACTNLI-V50|"
TARGET_CASES = 600
TARGET_PER_LABEL = 200
MAX_CASES_PER_DOCUMENT = 6
MINIMUM_CASES = 500
MINIMUM_STRATUM_CASES = 60
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260810

NATIVE_ZERO_CONSENSUS_PREFIX = "native_zero_consensus_"
GATED_NON_FRC_METHODS = tuple(
    f"{NATIVE_ZERO_CONSENSUS_PREFIX}{method}_v50" for method in NON_FRC_BASELINES
)
ANCHOR_GATE_FRC_V50 = "native_zero_anchor_adaptive_frc_v50"
NATIVE_ZERO_CONSENSUS_FRC_V50 = "native_zero_consensus_adaptive_frc_v50"
METHODS = (*V49_METHODS, *GATED_NON_FRC_METHODS, ANCHOR_GATE_FRC_V50, NATIVE_ZERO_CONSENSUS_FRC_V50)
UNGATED_FRC_CONTROLS = (DYNAMIC_RANK, ADAPTIVE_ARGMAX, LOW_CORE_DIVERGENCE_V49)

QUERY_TEMPLATES = {
    "anchor": "{hypothesis}",
    "first_fact": (
        "Find contract language that directly entails or supports the following "
        "statement: {hypothesis}"
    ),
    "second_fact_or_bridge": (
        "Find contract language that contradicts, negates, or rules out the following "
        "statement: {hypothesis}"
    ),
    "counterevidence": (
        "Find exceptions, exclusions, limitations, or scope conditions relevant to the "
        "following statement: {hypothesis}"
    ),
}

_FORBIDDEN_BLIND_KEYS = {
    "annotation_sets",
    "annotations",
    "choice",
    "document_id",
    "document_type",
    "evidence",
    "evidence_indices",
    "file_name",
    "gold",
    "gold_candidate_ids",
    "label",
    "label_group",
    "offset",
    "offsets",
    "sampling_label",
    "spans",
    "url",
}


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _nested_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            result.add(str(key).lower())
            result.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_nested_keys(item))
    return result


def _contains_forbidden_blind_key(value: Any) -> bool:
    return bool(_nested_keys(value) & _FORBIDDEN_BLIND_KEYS)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _opaque(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:20]
    return f"{prefix}{digest}"


def _token_count(tokenizer: Any, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(max(1, len(text.split()))))


def read_test_source(source_archive: Path) -> dict[str, Any]:
    """Read exactly one test JSON while permanently ignoring train and dev members."""

    with zipfile.ZipFile(source_archive) as archive:
        test_members = [
            name for name in archive.namelist() if Path(name).name.lower() == "test.json"
        ]
        if len(test_members) != 1:
            raise ValueError("ContractNLI archive must contain exactly one test.json")
        raw = archive.read(test_members[0])
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("ContractNLI test JSON must be an object")
    if not isinstance(value.get("documents"), list) or not isinstance(
        value.get("labels"), dict
    ):
        raise ValueError("ContractNLI test JSON lacks documents or labels")
    return value


def _document_id(document: dict[str, Any]) -> str:
    value = _normalise(document.get("id"))
    if not value:
        raise ValueError("ContractNLI document id is missing")
    return value


def build_span_units(document: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    text = str(document.get("text", ""))
    spans = document.get("spans")
    if not text or not isinstance(spans, list) or not spans:
        raise ValueError("ContractNLI document text or spans are missing")
    units: list[dict[str, Any]] = []
    last_end = -1
    for index, raw in enumerate(spans):
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or not all(isinstance(value, int) for value in raw)
        ):
            raise ValueError("ContractNLI span offsets must be integer pairs")
        start, end = raw
        if start < 0 or end <= start or end > len(text) or start < last_end:
            raise ValueError("ContractNLI span offsets are invalid or out of order")
        content = text[start:end]
        last_end = end
        if not content.strip():
            # Official test contracts contain whitespace-only structural spans.
            # The registered erratum excludes only these units and preserves the
            # original index for every semantic span, so evidence ids never shift.
            continue
        units.append(
            {
                "canonical_id": f"span_{index:04d}",
                "source_id": f"span_{index:04d}",
                "source_kind": "contract_span",
                "text": content,
                "start": start,
                "end": end,
                "token_count": _token_count(tokenizer, content),
            }
        )
    if not units:
        raise ValueError("ContractNLI document has no non-whitespace official spans")
    return units


def _iter_cases(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    labels = source["labels"]
    for raw_document in source["documents"]:
        if not isinstance(raw_document, dict):
            raise ValueError("ContractNLI document must be an object")
        document = dict(raw_document)
        document_id = _document_id(document)
        annotation_sets = document.get("annotation_sets")
        if not isinstance(annotation_sets, list) or len(annotation_sets) != 1:
            raise ValueError("ContractNLI requires exactly one annotation set")
        annotations = annotation_sets[0].get("annotations")
        if not isinstance(annotations, dict):
            raise ValueError("ContractNLI annotations are missing")
        for hypothesis_key, metadata in labels.items():
            if not isinstance(metadata, dict):
                raise ValueError("ContractNLI label metadata must be an object")
            hypothesis = _normalise(metadata.get("hypothesis"))
            annotation = annotations.get(hypothesis_key)
            if not hypothesis or not isinstance(annotation, dict):
                raise ValueError("ContractNLI hypothesis or annotation is missing")
            choice = str(annotation.get("choice"))
            evidence = annotation.get("spans")
            if choice not in LABELS or not isinstance(evidence, list):
                raise ValueError("ContractNLI label or evidence list is invalid")
            if not all(isinstance(index, int) for index in evidence):
                raise ValueError("ContractNLI evidence indices must be integers")
            if choice == "NotMentioned" and evidence:
                raise ValueError("ContractNLI NotMentioned case has evidence")
            if choice in EVIDENCE_LABELS and not evidence:
                raise ValueError("ContractNLI evidence-bearing case has no evidence")
            yield {
                "document": document,
                "document_id": document_id,
                "hypothesis_key": str(hypothesis_key),
                "hypothesis": hypothesis,
                "label": choice,
                "evidence_indices": tuple(sorted(set(evidence))),
            }


def _sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    digest = hashlib.sha256(
        (
            SAMPLE_SALT
            + str(row["label"])
            + "|"
            + str(row["document_id"])
            + "|"
            + str(row["hypothesis_key"])
        ).encode()
    ).hexdigest()
    return digest, str(row["document_id"]), str(row["hypothesis_key"])


def select_sample(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(_iter_cases(source))
    by_label = {
        label: sorted(
            [row for row in rows if row["label"] == label], key=_sample_key
        )
        for label in LABELS
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    document_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    # Advance the three frozen per-label orders in lockstep. This preserves each
    # label's registered ordering while preventing the shared document cap from
    # being consumed by whichever label happens to appear first in LABELS.
    cursors = {label: 0 for label in LABELS}
    while any(label_counts[label] < TARGET_PER_LABEL for label in LABELS):
        progress = False
        for label in LABELS:
            if label_counts[label] >= TARGET_PER_LABEL:
                continue
            values = by_label[label]
            while cursors[label] < len(values):
                row = values[cursors[label]]
                cursors[label] += 1
                document_id = str(row["document_id"])
                if document_counts[document_id] >= MAX_CASES_PER_DOCUMENT:
                    continue
                key = (document_id, str(row["hypothesis_key"]))
                selected.append(row)
                selected_keys.add(key)
                document_counts[document_id] += 1
                label_counts[label] += 1
                progress = True
                break
        if not progress:
            break

    remaining = sorted(
        (
            row
            for row in rows
            if (str(row["document_id"]), str(row["hypothesis_key"]))
            not in selected_keys
        ),
        key=_sample_key,
    )
    for row in remaining:
        if len(selected) >= TARGET_CASES:
            break
        document_id = str(row["document_id"])
        if document_counts[document_id] >= MAX_CASES_PER_DOCUMENT:
            continue
        selected.append(row)
        document_counts[document_id] += 1
        label_counts[str(row["label"])] += 1
    selected = sorted(selected, key=_sample_key)
    if len(selected) != TARGET_CASES:
        raise ValueError("ContractNLI frozen sampling could not form exactly 600 cases")
    if max(document_counts.values(), default=0) > MAX_CASES_PER_DOCUMENT:
        raise AssertionError("ContractNLI document cap was violated")
    return selected, {
        "eligible_cases": len(rows),
        "selected_cases": len(selected),
        "eligible_label_counts": {label: len(by_label[label]) for label in LABELS},
        "selected_label_counts": {label: label_counts[label] for label in LABELS},
        "selected_documents": len(document_counts),
        "maximum_cases_per_document": max(document_counts.values(), default=0),
        "target_per_label": TARGET_PER_LABEL,
        "document_cap": MAX_CASES_PER_DOCUMENT,
        "sample_salt": SAMPLE_SALT,
    }


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "maximum": 0, "mean": 0.0}
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(float(np.mean(values)), 6),
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    document_lengths: list[int] = []
    for row in selected:
        document_id = str(row["document_id"])
        hypothesis_key = str(row["hypothesis_key"])
        case_id = _opaque("cnli", document_id, hypothesis_key)
        units = build_span_units(dict(row["document"]), tokenizer)
        candidates = [
            {
                "id": str(unit["canonical_id"]),
                "source_id": str(unit["source_id"]),
                "source_kind": str(unit["source_kind"]),
                "text": str(unit["text"]),
                "token_count": int(unit["token_count"]),
            }
            for unit in units
        ]
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "id": case_id,
            "query": str(row["hypothesis"]),
            "candidates": candidates,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("ContractNLI gold field leaked into blind cache")
        prepared.append(blind)
        maps.append(
            {
                "id": case_id,
                "document_id_sha256": hashlib.sha256(document_id.encode()).hexdigest(),
                "hypothesis_key_sha256": hashlib.sha256(
                    hypothesis_key.encode()
                ).hexdigest(),
                "units": [
                    {
                        "canonical_id": str(unit["canonical_id"]),
                        "token_count": int(unit["token_count"]),
                    }
                    for unit in units
                ],
            }
        )
        pool_sizes.append(len(units))
        token_costs.extend(int(unit["token_count"]) for unit in units)
        document_lengths.append(len(str(row["document"].get("text", ""))))
    boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    length_boundaries = [
        round(float(value), 6)
        for value in np.quantile(document_lengths, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, {
        "prepared_cases": len(prepared),
        "candidate_pool_size": _distribution(pool_sizes),
        "candidate_pool_quartile_boundaries": boundaries,
        "candidate_token_cost": _distribution(token_costs),
        "document_length_codepoints": _distribution(document_lengths),
        "document_length_quartile_boundaries": length_boundaries,
        "gold_fields_exported_to_blind_cache": False,
        "ready_for_query_generation": len(prepared) == len(selected),
    }


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if _contains_forbidden_blind_key(row):
            raise ValueError("ContractNLI query builder received a forbidden field")
        hypothesis = _normalise(row["query"])
        atomic = {
            role: QUERY_TEMPLATES[role].format(hypothesis=hypothesis)
            for role in DYNAMIC_ROLES
        }
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "id": str(row["id"]),
                "atomic_queries": atomic,
                "fallback_used": False,
                "fallback_reason": None,
                "gold_fields_visible_to_generator": False,
            }
        )
    return result


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = [str(row["id"]) for row in prepared_rows]
    actual = [str(row["id"]) for row in query_rows]
    if actual != expected:
        raise ValueError("ContractNLI query cache is incomplete or out of order")
    for prepared, query in zip(prepared_rows, query_rows, strict=True):
        if _contains_forbidden_blind_key(query):
            raise ValueError("ContractNLI query cache contains a forbidden field")
        hypothesis = _normalise(prepared["query"])
        expected_atomic = {
            role: QUERY_TEMPLATES[role].format(hypothesis=hypothesis)
            for role in DYNAMIC_ROLES
        }
        if query.get("atomic_queries") != expected_atomic:
            raise ValueError("ContractNLI deterministic query changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenContractNliScorer(FrozenEvidenceInferenceScorer):
    """Frozen v49 scorer plus raw dynamic logits for the registered native-zero gate."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        pairs: list[tuple[str, str]] = []
        work: list[tuple[int, int]] = []
        for case in cases:
            case_id = str(case["id"])
            atomic = self.query_by_id.get(case_id)
            if atomic is None:
                raise ValueError("ContractNLI atomic queries are missing")
            texts = [str(candidate["text"]) for candidate in case["candidates"]]
            start = len(pairs)
            pairs.extend(
                (str(atomic[role]), text)
                for role in DYNAMIC_ROLES
                for text in texts
            )
            work.append((start, len(texts)))
        raw_scores = self._predict(pairs).reshape(-1)
        for row, (start, count) in zip(rows, work, strict=True):
            matrix = raw_scores[
                start : start + len(DYNAMIC_ROLES) * count
            ].reshape(len(DYNAMIC_ROLES), count)
            for index, candidate_score in enumerate(row["candidate_scores"]):
                candidate_score["raw_dynamic_role_scores"] = {
                    role: float(matrix[role_index, index])
                    for role_index, role in enumerate(DYNAMIC_ROLES)
                }
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            row["raw_native_zero_boundary"] = 0.0
        return rows


def merge_v50_scored_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = merge_scored_candidates(row)
    raw_by_id = {
        str(item["id"]): dict(item.get("raw_dynamic_role_scores", {}))
        for item in row.get("candidate_scores", [])
    }
    for candidate in candidates:
        raw = raw_by_id.get(str(candidate["id"]))
        if raw is None or set(raw) != set(DYNAMIC_ROLES):
            raise ValueError("ContractNLI raw dynamic role scores are incomplete")
        candidate["raw_dynamic_role_scores"] = raw
    return candidates


def native_zero_gate_details(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchor_pass_ids: list[str] = []
    consensus_pass_ids: list[str] = []
    maximum_logits = {role: -math.inf for role in DYNAMIC_ROLES}
    for candidate in candidates:
        raw = candidate.get("raw_dynamic_role_scores")
        if not isinstance(raw, dict) or set(raw) != set(DYNAMIC_ROLES):
            raise ValueError("ContractNLI candidate lacks registered raw role logits")
        identifier = str(candidate["id"])
        for role in DYNAMIC_ROLES:
            maximum_logits[role] = max(maximum_logits[role], float(raw[role]))
        anchor = float(raw["anchor"]) > 0.0
        polarity = max(
            float(raw["first_fact"]), float(raw["second_fact_or_bridge"])
        ) > 0.0
        if anchor:
            anchor_pass_ids.append(identifier)
        if anchor and polarity:
            consensus_pass_ids.append(identifier)
    return {
        "native_zero_boundary": 0.0,
        "anchor_gate_passed": bool(anchor_pass_ids),
        "consensus_gate_passed": bool(consensus_pass_ids),
        "anchor_pass_ids": sorted(anchor_pass_ids),
        "consensus_pass_ids": sorted(consensus_pass_ids),
        "maximum_raw_logits": {
            role: (None if value == -math.inf else float(value))
            for role, value in maximum_logits.items()
        },
    }


def select_v50(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    if method in V49_METHODS:
        return select_v49(values, method, token_budget=token_budget)
    gate = native_zero_gate_details(values)
    if method == ANCHOR_GATE_FRC_V50:
        if not gate["anchor_gate_passed"]:
            return []
        return select_v49(values, ADAPTIVE_ARGMAX, token_budget=token_budget)
    if method == NATIVE_ZERO_CONSENSUS_FRC_V50:
        if not gate["consensus_gate_passed"]:
            return []
        return select_v49(values, ADAPTIVE_ARGMAX, token_budget=token_budget)
    if method in GATED_NON_FRC_METHODS:
        if not gate["consensus_gate_passed"]:
            return []
        base_method = method[len(NATIVE_ZERO_CONSENSUS_PREFIX) : -len("_v50")]
        if base_method not in NON_FRC_BASELINES:
            raise ValueError("ContractNLI gated baseline mapping is invalid")
        return select_v49(values, base_method, token_budget=token_budget)
    raise ValueError(f"Unsupported ContractNLI method: {method}")


def _quartile(value: int, boundaries: Sequence[float]) -> str:
    if value <= boundaries[0]:
        return "q1"
    if value <= boundaries[1]:
        return "q2"
    if value <= boundaries[2]:
        return "q3"
    return "q4"


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    *,
    pool_quartile_boundaries: Sequence[float],
    document_length_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    rows_by_commitment: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_cases(source):
        key = (
            hashlib.sha256(str(row["document_id"]).encode()).hexdigest(),
            hashlib.sha256(str(row["hypothesis_key"]).encode()).hexdigest(),
        )
        if key in rows_by_commitment:
            raise ValueError("ContractNLI source commitment is not unique")
        rows_by_commitment[key] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        key = (
            str(candidate_map["document_id_sha256"]),
            str(candidate_map["hypothesis_key_sha256"]),
        )
        row = rows_by_commitment.get(key)
        if row is None:
            raise ValueError("ContractNLI candidate map commitment is absent")
        available = {
            str(unit["canonical_id"]) for unit in candidate_map.get("units", [])
        }
        gold_ids = tuple(
            f"span_{index:04d}" for index in row["evidence_indices"]
        )
        if any(identifier not in available for identifier in gold_ids):
            raise ValueError("ContractNLI candidate ceiling is incomplete")
        document = dict(row["document"])
        pool_size = len(available)
        document_length = len(str(document.get("text", "")))
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "document_cluster": key[0],
                "label_group": str(row["label"]),
                "gold_candidate_ids": list(gold_ids),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _quartile(
                    pool_size, pool_quartile_boundaries
                ),
                "document_length_codepoints": document_length,
                "document_length_quartile": _quartile(
                    document_length, document_length_quartile_boundaries
                ),
                "document_type": _normalise(document.get("document_type"))
                or "unknown",
                "candidate_ceiling_complete": (
                    not gold_ids or set(gold_ids) <= available
                ),
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    evidence_rows = [
        row for row in gold_rows if row["label_group"] in EVIDENCE_LABELS
    ]
    missing_rows = [
        row for row in gold_rows if row["label_group"] == "NotMentioned"
    ]
    ceiling = (
        float(np.mean([row["candidate_ceiling_complete"] for row in evidence_rows]))
        if evidence_rows
        else 0.0
    )
    empty_rate = (
        float(np.mean([not row["gold_candidate_ids"] for row in missing_rows]))
        if missing_rows
        else 0.0
    )
    checks = {
        "minimum_cases_met": len(gold_rows) >= MINIMUM_CASES,
        "exact_target_cases_met": len(gold_rows) == TARGET_CASES,
        "candidate_ceiling_complete_rate_on_evidence_cases_equals_1": ceiling
        == 1.0,
        "not_mentioned_official_empty_evidence_rate_equals_1": empty_rate == 1.0,
    }
    return {
        "schema_version": "frc-contractnli-v50-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "evidence_cases": len(evidence_rows),
        "not_mentioned_cases": len(missing_rows),
        "candidate_ceiling_complete_rate_on_evidence_cases": round(ceiling, 6),
        "not_mentioned_official_empty_evidence_rate": round(empty_rate, 6),
        "label_groups": dict(Counter(row["label_group"] for row in gold_rows)),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "document_length_quartiles": dict(
            Counter(row["document_length_quartile"] for row in gold_rows)
        ),
        "document_types": dict(Counter(row["document_type"] for row in gold_rows)),
        "sampling": sampling,
        "checks": checks,
        "minimum_cases_and_ceiling_checks_passed": all(checks.values()),
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def _selection_metrics(
    selected: Sequence[dict[str, Any]], label: str, gold_ids: Sequence[str]
) -> dict[str, Any]:
    selected_ids = {str(item["id"]) for item in selected}
    gold = set(gold_ids)
    if label == "NotMentioned":
        utility = float(not selected_ids)
        return {
            "utility_f1": utility,
            "evidence_f1": None,
            "precision": None,
            "recall": None,
            "complete_recall": None,
            "gold_unit_hits": 0,
            "selected_unit_count": len(selected_ids),
            "selected_token_cost": sum(int(item["token_count"]) for item in selected),
            "abstained": not selected_ids,
        }
    hits = len(selected_ids & gold)
    precision = hits / len(selected_ids) if selected_ids else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "utility_f1": f1,
        "evidence_f1": f1,
        "precision": precision,
        "recall": recall,
        "complete_recall": bool(gold and gold <= selected_ids),
        "gold_unit_hits": hits,
        "selected_unit_count": len(selected_ids),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "abstained": not selected_ids,
    }


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    values = [
        configuration["methods"][method]["metrics"]
        for row in rows
        for configuration in row["configurations"].values()
    ]
    evidence = [value for value in values if value["evidence_f1"] is not None]
    missing = [
        configuration["methods"][method]["metrics"]
        for row in rows
        if row["label_group"] == "NotMentioned"
        for configuration in row["configurations"].values()
    ]
    def mean_or_zero(items: Sequence[float | int | bool]) -> float:
        return float(np.mean(items)) if items else 0.0

    return {
        "evidence_or_abstention_macro_f1": round(
            mean_or_zero([value["utility_f1"] for value in values]), 6
        ),
        "evidence_bearing_macro_f1": round(
            mean_or_zero([value["evidence_f1"] for value in evidence]), 6
        ),
        "evidence_macro_precision": round(
            mean_or_zero([value["precision"] for value in evidence]), 6
        ),
        "evidence_macro_recall": round(
            mean_or_zero([value["recall"] for value in evidence]), 6
        ),
        "complete_evidence_recall": round(
            mean_or_zero([value["complete_recall"] for value in evidence]), 6
        ),
        "not_mentioned_abstention_accuracy": round(
            mean_or_zero([value["utility_f1"] for value in missing]), 6
        ),
        "abstention_rate": round(
            mean_or_zero([value["abstained"] for value in values]), 6
        ),
        "mean_selected_unit_count": round(
            mean_or_zero([value["selected_unit_count"] for value in values]), 6
        ),
        "mean_selected_token_cost": round(
            mean_or_zero([value["selected_token_cost"] for value in values]), 6
        ),
    }


def _case_method_utility(row: dict[str, Any], method: str) -> float:
    return float(
        np.mean(
            [
                configuration["methods"][method]["metrics"]["utility_f1"]
                for configuration in row["configurations"].values()
            ]
        )
    )


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, float | int]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["document_cluster"])].append(
            _case_method_utility(row, candidate)
            - _case_method_utility(row, baseline)
        )
    clusters = sorted(by_cluster)
    arrays = [np.asarray(by_cluster[cluster], dtype=float) for cluster in clusters]
    point = float(np.mean(np.concatenate(arrays)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = float(np.mean(np.concatenate([arrays[item] for item in picked])))
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "clusters": len(clusters),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _strongest(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> str:
    return sorted(
        methods,
        key=lambda method: (
            -float(_aggregate(rows, method)["evidence_or_abstention_macro_f1"]),
            method,
        ),
    )[0]


def _stratum_delta(
    rows: Sequence[dict[str, Any]], candidate: str
) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    candidate_value = float(
        np.mean([_case_method_utility(row, candidate) for row in rows])
    )
    baseline_value = float(
        np.mean([_case_method_utility(row, strongest) for row in rows])
    )
    return strongest, round(candidate_value - baseline_value, 6)


def evaluate_contractnli(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("ContractNLI gold and score caches differ")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("ContractNLI gold and score case ids differ")
        candidates = merge_v50_scored_candidates(dict(scored))
        gate = native_zero_gate_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v50(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
                        selected,
                        str(gold["label_group"]),
                        list(gold["gold_candidate_ids"]),
                    ),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "document_cluster": str(gold["document_cluster"]),
                "label_group": str(gold["label_group"]),
                "candidate_unit_count": int(gold["candidate_unit_count"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "document_length_quartile": str(
                    gold["document_length_quartile"]
                ),
                "document_type": str(gold["document_type"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "gate": gate,
                "configurations": configurations,
            }
        )

    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_frozen_frc = _strongest(evidence, UNGATED_FRC_CONTROLS)
    candidate_vs_non_frc = _cluster_bootstrap(
        evidence, NATIVE_ZERO_CONSENSUS_FRC_V50, strongest_non_frc
    )
    candidate_vs_frozen_frc = _cluster_bootstrap(
        evidence, NATIVE_ZERO_CONSENSUS_FRC_V50, strongest_frozen_frc
    )
    candidate_vs_anchor = _cluster_bootstrap(
        evidence, NATIVE_ZERO_CONSENSUS_FRC_V50, ANCHOR_GATE_FRC_V50
    )
    candidate_aggregate = aggregates[NATIVE_ZERO_CONSENSUS_FRC_V50]
    v43_aggregate = aggregates[ADAPTIVE_ARGMAX]
    anchor_aggregate = aggregates[ANCHOR_GATE_FRC_V50]
    evidence_f1_drop = round(
        v43_aggregate["evidence_bearing_macro_f1"]
        - candidate_aggregate["evidence_bearing_macro_f1"],
        6,
    )
    missing_improvement = round(
        candidate_aggregate["not_mentioned_abstention_accuracy"]
        - anchor_aggregate["not_mentioned_abstention_accuracy"],
        6,
    )
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        budget_rows = []
        for row in evidence:
            clone = dict(row)
            clone["configurations"] = {
                str(budget): row["configurations"][str(budget)]
            }
            budget_rows.append(clone)
        _, delta = _stratum_delta(budget_rows, NATIVE_ZERO_CONSENSUS_FRC_V50)
        budget_deltas[str(budget)] = delta

    strata: dict[str, dict[str, Any]] = {}
    dimensions = {
        "label_group": lambda row: row["label_group"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "document_length_quartile": lambda row: row["document_length_quartile"],
        "document_type": lambda row: row["document_type"],
        "gate_outcome": lambda row: (
            "consensus_passed" if row["gate"]["consensus_gate_passed"] else "abstained"
        ),
    }
    for dimension, getter in dimensions.items():
        values = sorted({str(getter(row)) for row in evidence})
        for value in values:
            subset = [row for row in evidence if str(getter(row)) == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(
                subset, NATIVE_ZERO_CONSENSUS_FRC_V50
            )
            strata[f"{dimension}:{value}"] = {
                "cases": len(subset),
                "strongest_shared_gate_non_frc": strongest,
                "delta": delta,
            }
    minimum_safety_delta = min(
        [*budget_deltas.values(), *(item["delta"] for item in strata.values())],
        default=-math.inf,
    )
    checks = {
        "minimum_cases_met": len(evidence) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_on_evidence_cases_equals_1": all(
            row["candidate_ceiling_complete"]
            for row in evidence
            if row["label_group"] in EVIDENCE_LABELS
        ),
        "not_mentioned_official_empty_evidence_rate_equals_1": all(
            row["label_group"] != "NotMentioned" or not row["gold_candidate_ids"]
            for row in gold_rows
        ),
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01": candidate_vs_non_frc[
            "point"
        ]
        >= 0.01,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": candidate_vs_non_frc[
            "ci_low"
        ]
        > 0.0,
        "candidate_minus_strongest_ungated_frozen_frc_point_at_least_0_01": candidate_vs_frozen_frc[
            "point"
        ]
        >= 0.01,
        "candidate_minus_strongest_ungated_frozen_frc_ci_low_above_0": candidate_vs_frozen_frc[
            "ci_low"
        ]
        > 0.0,
        "candidate_minus_anchor_gate_frc_point_at_least_0_005": candidate_vs_anchor[
            "point"
        ]
        >= 0.005,
        "candidate_minus_anchor_gate_frc_ci_low_above_0": candidate_vs_anchor[
            "ci_low"
        ]
        > 0.0,
        "evidence_bearing_macro_f1_drop_vs_ungated_v43_at_most_0_02": evidence_f1_drop
        <= 0.02,
        "not_mentioned_abstention_accuracy_improvement_vs_anchor_gate_at_least_0_05": missing_improvement
        >= 0.05,
        "candidate_abstention_rate_at_least_0_05": candidate_aggregate[
            "abstention_rate"
        ]
        >= 0.05,
        "candidate_abstention_rate_at_most_0_95": candidate_aggregate[
            "abstention_rate"
        ]
        <= 0.95,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": minimum_safety_delta
        >= -0.02,
        "deterministic_query_fallback_rate_equals_0": query_summary["fallback_rate"]
        == 0.0,
    }
    supported = all(checks.values())
    status = (
        "CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_ESTABLISHED"
        if supported
        else "CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-contractnli-native-zero-consensus-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "full_contract_bounded_pool": True,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_ungated_frozen_frc": strongest_frozen_frc,
            "family_comparison": {
                "candidate_minus_strongest_shared_gate_non_frc": candidate_vs_non_frc,
                "candidate_minus_strongest_ungated_frozen_frc": candidate_vs_frozen_frc,
                "candidate_minus_anchor_gate_frc": candidate_vs_anchor,
            },
            "evidence_bearing_macro_f1_drop_vs_ungated_v43": evidence_f1_drop,
            "not_mentioned_abstention_accuracy_improvement_vs_anchor_gate": missing_improvement,
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "minimum_budget_or_supported_stratum_delta": round(
                minimum_safety_delta, 6
            ),
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_v50_cases_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    outcome = analysis["outcome"]
    candidate = analysis["aggregates"][NATIVE_ZERO_CONSENSUS_FRC_V50]
    non_frc = analysis["strongest_shared_gate_non_frc"]
    frozen_frc = analysis["strongest_ungated_frozen_frc"]
    comparison = analysis["family_comparison"]
    lines = [
        "# ContractNLI native-zero consensus abstention evaluation (v50)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases: {rounded['metadata']['cases']}",
        f"- Strongest shared-gate non-FRC: `{non_frc}`",
        f"- Strongest ungated frozen FRC: `{frozen_frc}`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Registered comparisons",
        "",
        f"- Candidate utility F1: {candidate['evidence_or_abstention_macro_f1']:.6f}",
        f"- Candidate vs strongest shared-gate non-FRC: {comparison['candidate_minus_strongest_shared_gate_non_frc']}",
        f"- Candidate vs strongest ungated frozen FRC: {comparison['candidate_minus_strongest_ungated_frozen_frc']}",
        f"- Candidate vs anchor-gate FRC: {comparison['candidate_minus_anchor_gate_frc']}",
        f"- Candidate evidence-bearing F1: {candidate['evidence_bearing_macro_f1']:.6f}",
        f"- Candidate NotMentioned abstention accuracy: {candidate['not_mentioned_abstention_accuracy']:.6f}",
        f"- Candidate abstention rate: {candidate['abstention_rate']:.6f}",
        "",
        "## Method aggregates",
        "",
        "| Method | Utility F1 | Evidence F1 | Missing abstention | Abstention rate | Mean units | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = analysis["aggregates"][method]
        lines.append(
            f"| `{method}` | {row['evidence_or_abstention_macro_f1']:.6f} | "
            f"{row['evidence_bearing_macro_f1']:.6f} | "
            f"{row['not_mentioned_abstention_accuracy']:.6f} | "
            f"{row['abstention_rate']:.6f} | {row['mean_selected_unit_count']:.6f} | "
            f"{row['mean_selected_token_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a balanced full-contract span-selection and missing-evidence abstention mechanism test, not an official ContractNLI NLI leaderboard result. It does not reproduce Span NLI BERT, open-corpus retrieval, or flood-domain expert evaluation. v50 cases may not be reused for tuning or method selection.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            _round_for_display(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("ContractNLI v50 protocol hash changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = protocol.get("development_boundary", {})
    if boundary.get("contractnli_train_split_permanently_excluded") is not True:
        raise ValueError("ContractNLI train exclusion is not frozen")
    if boundary.get("contractnli_dev_split_permanently_excluded") is not True:
        raise ValueError("ContractNLI dev exclusion is not frozen")
    if protocol.get("methods", {}).get("candidate_method") != NATIVE_ZERO_CONSENSUS_FRC_V50:
        raise ValueError("ContractNLI v50 candidate method changed")
    if protocol.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("ContractNLI v50 budgets changed")
    if protocol.get("query_templates") != {
        **QUERY_TEMPLATES,
        "query_text_or_template_may_not_change_after_registration": True,
    }:
        raise ValueError("ContractNLI v50 deterministic queries changed")
    return protocol


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("ContractNLI implementation registration hash mismatch")
    if value.get("archive_downloaded_before_registration") is not False:
        raise ValueError("ContractNLI implementation was not prospectively frozen")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("ContractNLI synthetic invariants are incomplete")
    return value


def validate_execution_registration(
    execution_path: Path,
    *,
    implementation_path: Path,
    source_archive: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    value = json.loads(execution_path.read_text(encoding="utf-8"))
    hashes = value.get("hashes", {})
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if hashes != expected:
        raise ValueError("ContractNLI execution registration hash mismatch")
    if value.get("query_generation_started") is not False:
        raise ValueError("ContractNLI execution was registered too late")
    return value


__all__ = [
    "ANCHOR_GATE_FRC_V50",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CAPABILITY",
    "DATASET_ID",
    "EVIDENCE_LABELS",
    "EXPERIMENT_ID",
    "FrozenContractNliScorer",
    "GATED_NON_FRC_METHODS",
    "LABELS",
    "METHODS",
    "NATIVE_ZERO_CONSENSUS_FRC_V50",
    "OFFICIAL_REPOSITORY_REVISION",
    "PROTOCOL_SHA256",
    "QUERY_TEMPLATES",
    "SCHEMA_VERSION",
    "TARGET_CASES",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_span_units",
    "canonical_json_sha256",
    "evaluate_contractnli",
    "merge_v50_scored_candidates",
    "native_zero_gate_details",
    "prepare_blind_cases",
    "read_test_source",
    "select_sample",
    "select_v50",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "write_report",
]
