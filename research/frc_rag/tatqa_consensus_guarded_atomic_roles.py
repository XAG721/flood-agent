"""Prospective TAT-QA consensus-guarded atomic-role experiment (v46).

Dataset preparation, blind neural scoring, and gold joining are deliberately
separate.  The candidate method uses only within-role ranks and is frozen by
synthetic invariants before the registered TAT-QA archive is acquired.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from research.frc_rag.finqa_anchor_guarded_atomic_roles import (
    ADAPTIVE_V43,
    ANCHOR_GUARDED_V45,
    GUARDED_V44,
    select_v45,
)
from research.frc_rag.hover_dynamic_atomic_roles import (
    BASELINES,
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    METHODS as V41_METHODS,
    RANK_COVERAGE_WEIGHT,
    FrozenDynamicHoVerScorer,
    merge_scored_candidates,
    select_candidates,
)


SCHEMA_VERSION = "frc-tatqa-consensus-guarded-atomic-roles-v46"
EXPERIMENT_ID = "FRC-TATQA-CONSENSUS-GUARDED-ATOMIC-ROLES-V46"
DATASET_ID = "tatqa_dev_full_context_evidence_v46"
CAPABILITY = "full_context_table_cell_and_paragraph_evidence_selection"
PROTOCOL_SHA256 = "107ceca66c00bb70e424923c7ea879f16a24004debfe90f738ee7d89d8903ce8"
OFFICIAL_REPOSITORY_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"

SAMPLE_SALT = "FRC-TATQA-V46|"
SOURCE_MODES = ("table_only", "text_only", "hybrid")
TARGET_CASES_PER_MODE = 120
TARGET_CASES = 360
MINIMUM_TOTAL_CASES = 300
MINIMUM_CASES_PER_MODE = 60

TOP_K = 5
BUDGETS = (256, 512, 1024)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260806

CONSENSUS_GUARDED_V46 = "consensus_guarded_adaptive_cardinality_frc_v46"
NON_FRC_BASELINES = tuple(BASELINES)
FRC_CONTROLS = (ADAPTIVE_V43, GUARDED_V44, ANCHOR_GUARDED_V45)
METHODS = (*V41_METHODS, ADAPTIVE_V43, GUARDED_V44, ANCHOR_GUARDED_V45, CONSENSUS_GUARDED_V46)

_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "answer_from",
    "answer_type",
    "candidate_ceiling",
    "derivation",
    "facts",
    "gold",
    "gold_candidate_ids",
    "gold_count",
    "gold_source_mode",
    "mapping",
    "scale",
    "source_mode",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _opaque(prefix: str, *values: str) -> str:
    payload = "|".join(values).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:20]


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


def _contains_forbidden_blind_key(value: Any) -> bool:
    return bool(_nested_keys(value) & _FORBIDDEN_BLIND_KEYS)


def _table_matrix(context: dict[str, Any]) -> list[list[Any]]:
    raw = context.get("table")
    if isinstance(raw, dict):
        raw = raw.get("table")
    if not isinstance(raw, list) or not raw:
        return []
    if not all(isinstance(row, list) for row in raw):
        return []
    return [list(row) for row in raw]


def _context_id(context: dict[str, Any]) -> str:
    raw = context.get("table")
    candidates = [context.get("uid"), context.get("id")]
    if isinstance(raw, dict):
        candidates.extend([raw.get("uid"), raw.get("id")])
    for candidate in candidates:
        value = _normalise(candidate)
        if value:
            return value
    return canonical_json_sha256(
        {
            "table": _table_matrix(context),
            "paragraph_orders": [
                item.get("order")
                for item in context.get("paragraphs", [])
                if isinstance(item, dict)
            ],
        }
    )


def _paragraphs(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("paragraphs")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return []
        order = _normalise(item.get("order", index + 1))
        text = _normalise(item.get("text"))
        if not order or not text or order in seen:
            return []
        seen.add(order)
        result.append({"order": order, "text": text})
    return sorted(result, key=lambda item: (item["order"], item["text"]))


def _questions(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("questions")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _table_coordinate(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        row = int(value[0])
        column = int(value[1])
    except (TypeError, ValueError):
        return None
    if row < 0 or column < 0:
        return None
    return row, column


def _paragraph_mapping_orders(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [_normalise(key) for key in value if _normalise(key)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                order = _normalise(item.get("order"))
            else:
                order = _normalise(item)
            if order:
                result.append(order)
        return result
    return []


def mapped_candidate_ids(question: dict[str, Any]) -> tuple[str, ...]:
    mapping = question.get("mapping")
    if not isinstance(mapping, dict):
        return ()
    result: list[str] = []
    table_mapping = mapping.get("table")
    if isinstance(table_mapping, list):
        for raw in table_mapping:
            coordinate = _table_coordinate(raw)
            if coordinate is None:
                return ()
            result.append(f"table_{coordinate[0]}_{coordinate[1]}")
    for order in _paragraph_mapping_orders(mapping.get("paragraph")):
        result.append(f"paragraph_{order}")
    return tuple(dict.fromkeys(result))


def gold_source_mode(question: dict[str, Any]) -> str:
    identifiers = mapped_candidate_ids(question)
    if identifiers and all(value.startswith("table_") for value in identifiers):
        return "table_only"
    if identifiers and all(value.startswith("paragraph_") for value in identifiers):
        return "text_only"
    return "hybrid"


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return max(1, len(encoded))


def build_candidate_units(
    context: dict[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    matrix = _table_matrix(context)
    units: list[dict[str, Any]] = []
    for row_index, row in enumerate(matrix):
        row_header = _normalise(row[0]) if row else ""
        for column_index, raw_cell in enumerate(row):
            cell = _normalise(raw_cell)
            if not cell:
                continue
            column_header = (
                _normalise(matrix[0][column_index])
                if matrix and column_index < len(matrix[0])
                else ""
            )
            if row_index == 0:
                text = f"column header {column_index}: {cell}"
            elif column_index == 0:
                text = f"row header {row_index}: {cell}"
            else:
                text = (
                    f"row {row_header or row_index}; "
                    f"column {column_header or column_index}: {cell}"
                )
            units.append(
                {
                    "canonical_id": f"table_{row_index}_{column_index}",
                    "source_kind": "table",
                    "text": text,
                    "token_count": _token_count(tokenizer, text),
                }
            )
    for paragraph in _paragraphs(context):
        text = str(paragraph["text"])
        units.append(
            {
                "canonical_id": f"paragraph_{paragraph['order']}",
                "source_kind": "text",
                "text": text,
                "token_count": _token_count(tokenizer, text),
            }
        )
    return units


def _iter_question_rows(
    contexts: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for raw_context in contexts:
        context = dict(raw_context)
        context_id = _context_id(context)
        for question in _questions(context):
            uid = _normalise(question.get("uid", question.get("id")))
            yield {
                "context": context,
                "context_id": context_id,
                "question": question,
                "uid": uid,
            }


def _eligible(row: dict[str, Any]) -> bool:
    question = row["question"]
    uid = str(row["uid"])
    query = _normalise(question.get("question"))
    units = build_candidate_units(row["context"], _CharacterTokenizer())
    available = {str(item["canonical_id"]) for item in units}
    mapped = mapped_candidate_ids(question)
    return (
        bool(uid)
        and bool(query)
        and len(units) >= 3
        and bool(mapped)
        and set(mapped) <= available
    )


class _CharacterTokenizer:
    """Schema-only tokenizer used before the frozen tokenizer is loaded."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return list(text)


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    uid = str(row["uid"])
    digest = hashlib.sha256((SAMPLE_SALT + uid).encode("utf-8")).hexdigest()
    return digest, uid


def select_sample(
    contexts: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = {mode: [] for mode in SOURCE_MODES}
    seen_uids: set[str] = set()
    total = 0
    excluded: Counter[str] = Counter()
    for row in _iter_question_rows(contexts):
        total += 1
        uid = str(row["uid"])
        if not uid:
            excluded["missing_uid"] += 1
            continue
        if uid in seen_uids:
            raise ValueError("TAT-QA question uid is not unique")
        seen_uids.add(uid)
        if not _eligible(row):
            excluded["ineligible_schema_or_mapping"] += 1
            continue
        pools[gold_source_mode(row["question"])].append(row)
    for values in pools.values():
        values.sort(key=_sample_key)

    selected_by_mode = {
        mode: list(pools[mode][:TARGET_CASES_PER_MODE]) for mode in SOURCE_MODES
    }
    selected_uids = {
        str(row["uid"])
        for values in selected_by_mode.values()
        for row in values
    }
    if sum(len(values) for values in selected_by_mode.values()) < TARGET_CASES:
        remainder = sorted(
            (
                row
                for mode in SOURCE_MODES
                for row in pools[mode]
                if str(row["uid"]) not in selected_uids
            ),
            key=_sample_key,
        )
        for row in remainder:
            if sum(len(values) for values in selected_by_mode.values()) >= TARGET_CASES:
                break
            selected_by_mode[gold_source_mode(row["question"])].append(row)
            selected_uids.add(str(row["uid"]))

    selected = sorted(
        (row for values in selected_by_mode.values() for row in values),
        key=_sample_key,
    )
    eligible_counts = {mode: len(pools[mode]) for mode in SOURCE_MODES}
    selected_counts = {mode: len(selected_by_mode[mode]) for mode in SOURCE_MODES}
    checks = {
        "minimum_total_cases_met": len(selected) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            selected_counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
    }
    return selected, {
        "total_questions": total,
        "eligible_by_source_mode": eligible_counts,
        "selected_by_source_mode": selected_counts,
        "selected_questions": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "checks": checks,
        "ready_for_blind_preparation": all(checks.values()),
        "sample_order_commitment": canonical_json_sha256(
            [hashlib.sha256(str(row["uid"]).encode()).hexdigest() for row in selected]
        ),
    }


def prepare_blind_cases(
    contexts: Iterable[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(contexts)
    if not sampling["ready_for_blind_preparation"]:
        return [], [], {
            "sampling": sampling,
            "prepared_cases": 0,
            "ready_for_query_generation": False,
            "gold_fields_exported_to_blind_cache": False,
        }
    prepared: list[dict[str, Any]] = []
    candidate_maps: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    for row in selected:
        uid = str(row["uid"])
        units = build_candidate_units(row["context"], tokenizer)
        unit_ids = {str(unit["canonical_id"]) for unit in units}
        if len(unit_ids) != len(units):
            raise AssertionError("TAT-QA candidate ids are not unique")
        if not set(mapped_candidate_ids(row["question"])) <= unit_ids:
            raise AssertionError("TAT-QA mapping did not map to the full context")
        case_id = _opaque("t", SAMPLE_SALT, uid)
        candidates = [
            {
                "id": str(unit["canonical_id"]),
                "source_id": str(unit["canonical_id"]),
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
            "query": _normalise(row["question"].get("question")),
            "candidates": candidates,
            "gold_fields_visible_to_generator": False,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("TAT-QA gold field leaked into blind cache")
        prepared.append(blind)
        candidate_maps.append(
            {
                "schema_version": "frc-tatqa-v46-candidate-map-v1",
                "id": case_id,
                "question_uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                "context_id_sha256": hashlib.sha256(
                    str(row["context_id"]).encode()
                ).hexdigest(),
                "units": [
                    {
                        "candidate_id": str(unit["canonical_id"]),
                        "canonical_id": str(unit["canonical_id"]),
                        "source_kind": str(unit["source_kind"]),
                        "text": str(unit["text"]),
                        "text_sha256": hashlib.sha256(
                            str(unit["text"]).encode()
                        ).hexdigest(),
                        "token_count": int(unit["token_count"]),
                    }
                    for unit in units
                ],
            }
        )
        pool_sizes.append(len(candidates))
        token_costs.extend(int(item["token_count"]) for item in candidates)

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "minimum": min(values) if values else 0,
            "mean": round(float(np.mean(values)), 6) if values else 0.0,
            "maximum": max(values) if values else 0,
        }

    return prepared, candidate_maps, {
        "sampling": sampling,
        "prepared_cases": len(prepared),
        "candidate_pool_size": distribution(pool_sizes),
        "candidate_token_cost": distribution(token_costs),
        "ready_for_query_generation": len(prepared) == len(selected),
        "gold_fields_exported_to_blind_cache": False,
    }


class FrozenTatqaScorer(FrozenDynamicHoVerScorer):
    """Frozen v41 neural scorer with v46 dataset metadata."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def role_consensus_details(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rankings: dict[str, list[str]] = {}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["dynamic_role_scores"][role]),
                str(candidate["id"]),
            ),
        )
        rankings[role] = [str(candidate["id"]) for candidate in ordered]
    argmax_ids = {values[0] for values in rankings.values() if values}
    runner_up_votes: Counter[str] = Counter()
    for values in rankings.values():
        if len(values) >= 2 and values[1] not in argmax_ids:
            runner_up_votes[values[1]] += 1
    maximum_votes = max(runner_up_votes.values(), default=0)
    expansion = int(maximum_votes >= 2)
    target = min(TOP_K, max(1, len(argmax_ids) + expansion)) if candidates else 0
    return {
        "argmax_ids": sorted(argmax_ids),
        "distinct_role_argmax": len(argmax_ids),
        "runner_up_votes": dict(sorted(runner_up_votes.items())),
        "maximum_runner_up_votes": maximum_votes,
        "consensus_expansion": bool(expansion),
        "target_cardinality": target,
    }


def consensus_guarded_target_cardinality(
    candidates: Sequence[dict[str, Any]],
) -> int:
    return int(role_consensus_details(candidates)["target_cardinality"])


def _rank_utilities(candidates: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {str(candidate["id"]): {} for candidate in candidates}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["dynamic_role_scores"][role]),
                str(candidate["id"]),
            ),
        )
        for rank, candidate in enumerate(ordered):
            result[str(candidate["id"])][role] = 1.0 / math.log2(2 + rank)
    return result


def _adaptive_select(
    candidates: Sequence[dict[str, Any]], *, token_budget: int, target: int
) -> list[dict[str, Any]]:
    utilities = _rank_utilities(candidates)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sources: set[str] = set()
    best_by_role = {role: 0.0 for role in DYNAMIC_ROLES}
    total = 0
    while len(selected) < target:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            source_id = str(candidate["source_id"])
            cost = int(candidate["token_count"])
            if (
                candidate_id in selected_ids
                or source_id in selected_sources
                or total + cost > token_budget
            ):
                continue
            role_gain = sum(
                max(0.0, utilities[candidate_id][role] - best_by_role[role])
                for role in DYNAMIC_ROLES
            )
            gain = (
                float(candidate["scores"]["cross_encoder"])
                + RANK_COVERAGE_WEIGHT * role_gain
            )
            eligible.append((gain, candidate_id, candidate))
        if not eligible:
            break
        eligible.sort(key=lambda value: (-value[0], value[1]))
        best = eligible[0][2]
        selected.append(best)
        selected_ids.add(str(best["id"]))
        selected_sources.add(str(best["source_id"]))
        total += int(best["token_count"])
        for role in DYNAMIC_ROLES:
            best_by_role[role] = max(
                best_by_role[role], utilities[str(best["id"])][role]
            )
    return selected


def select_v46(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method in FRC_CONTROLS:
        return select_v45(candidates, method, token_budget=token_budget)
    if method == CONSENSUS_GUARDED_V46:
        return _adaptive_select(
            candidates,
            token_budget=token_budget,
            target=consensus_guarded_target_cardinality(candidates),
        )
    return select_candidates(list(candidates), method, token_budget=token_budget)


def build_gold_rows(
    source_contexts: Iterable[dict[str, Any]],
    candidate_maps: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_commitment: dict[str, dict[str, Any]] = {}
    for row in _iter_question_rows(source_contexts):
        uid = str(row["uid"])
        commitment = hashlib.sha256(uid.encode()).hexdigest()
        if commitment in source_by_commitment:
            raise ValueError("TAT-QA question uid commitment is not unique")
        source_by_commitment[commitment] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        commitment = str(candidate_map["question_uid_sha256"])
        source = source_by_commitment.get(commitment)
        if source is None:
            raise ValueError("TAT-QA candidate map question is missing from source")
        candidate_by_canonical = {
            str(unit["canonical_id"]): str(unit["candidate_id"])
            for unit in candidate_map["units"]
        }
        gold_keys = mapped_candidate_ids(source["question"])
        if not gold_keys or not set(gold_keys) <= set(candidate_by_canonical):
            raise ValueError("TAT-QA gold mapping is incomplete")
        answer_type = _normalise(source["question"].get("answer_type")) or "unknown"
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "question_uid": str(source["uid"]),
                "source_mode": gold_source_mode(source["question"]),
                "answer_type": answer_type,
                "gold_candidate_ids": [
                    candidate_by_canonical[key] for key in gold_keys
                ],
                "gold_count": len(gold_keys),
                "candidate_ceiling_complete": True,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    source_counts = Counter(str(row["source_mode"]) for row in gold_rows)
    ceiling_rate = (
        float(np.mean([bool(row["candidate_ceiling_complete"]) for row in gold_rows]))
        if gold_rows
        else 0.0
    )
    checks = {
        "minimum_total_cases_met": len(gold_rows) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            source_counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
        "candidate_ceiling_complete_rate_at_least_0_95": ceiling_rate >= 0.95,
    }
    return {
        "schema_version": "frc-tatqa-v46-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "source_modes": {mode: source_counts[mode] for mode in SOURCE_MODES},
        "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
        "sampling": sampling,
        "checks": checks,
        "minimum_cases_and_ceiling_checks_passed": all(checks.values()),
        "candidate_method_or_threshold_changed": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def _selection_metrics(
    selected: Sequence[dict[str, Any]], gold_ids: Sequence[str]
) -> dict[str, float | int | bool]:
    selected_ids = {str(candidate["id"]) for candidate in selected}
    gold = {str(value) for value in gold_ids}
    hits = len(selected_ids & gold)
    precision = hits / len(selected_ids) if selected_ids else 1.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "complete_recall": bool(gold) and hits == len(gold),
        "selected_unit_count": len(selected),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "gold_unit_hits": hits,
    }


def _aggregate(
    rows: Sequence[dict[str, Any]],
    method: str,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, float]:
    metrics = [
        row["configurations"][str(budget)]["methods"][method]["metrics"]
        for row in rows
        for budget in budgets
    ]
    return {
        "evidence_macro_f1": float(np.mean([item["f1"] for item in metrics])),
        "macro_precision": float(np.mean([item["precision"] for item in metrics])),
        "macro_recall": float(np.mean([item["recall"] for item in metrics])),
        "complete_recall": float(np.mean([item["complete_recall"] for item in metrics])),
        "mean_selected_unit_count": float(
            np.mean([item["selected_unit_count"] for item in metrics])
        ),
        "mean_selected_token_cost": float(
            np.mean([item["selected_token_cost"] for item in metrics])
        ),
    }


def _comparison(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": round(float(point), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _paired_bootstrap(
    rows: Sequence[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    versus_frc = np.empty(resamples, dtype=float)
    versus_non_frc = np.empty(resamples, dtype=float)
    for index in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(value)] for value in indices]
        candidate = _aggregate(sample, CONSENSUS_GUARDED_V46)["evidence_macro_f1"]
        strongest_frc = max(
            _aggregate(sample, method)["evidence_macro_f1"] for method in FRC_CONTROLS
        )
        strongest_non_frc = max(
            _aggregate(sample, method)["evidence_macro_f1"]
            for method in NON_FRC_BASELINES
        )
        versus_frc[index] = candidate - strongest_frc
        versus_non_frc[index] = candidate - strongest_non_frc
    candidate = _aggregate(rows, CONSENSUS_GUARDED_V46)["evidence_macro_f1"]
    strongest_frc = max(
        _aggregate(rows, method)["evidence_macro_f1"] for method in FRC_CONTROLS
    )
    strongest_non_frc = max(
        _aggregate(rows, method)["evidence_macro_f1"] for method in NON_FRC_BASELINES
    )
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "v46_minus_strongest_frozen_frc_control": _comparison(
            candidate - strongest_frc, versus_frc
        ),
        "v46_minus_strongest_non_frc": _comparison(
            candidate - strongest_non_frc, versus_non_frc
        ),
    }


def _set_difference_rate(
    rows: Sequence[dict[str, Any]], reference: str
) -> float:
    values = []
    for row in rows:
        for budget in BUDGETS:
            methods = row["configurations"][str(budget)]["methods"]
            candidate = set(methods[CONSENSUS_GUARDED_V46]["selected_ids"])
            baseline = set(methods[reference]["selected_ids"])
            values.append(candidate != baseline)
    return float(np.mean(values)) if values else 0.0


def evaluate_tatqa(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not gold_rows or len(gold_rows) != len(scored_rows):
        raise ValueError("TAT-QA gold and scored rows are empty or misaligned")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("TAT-QA gold and scored case ids differ")
        candidates = merge_scored_candidates(dict(scored))
        consensus = role_consensus_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v46(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
                        selected, list(gold["gold_candidate_ids"])
                    ),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "source_mode": str(gold["source_mode"]),
                "answer_type": str(gold["answer_type"]),
                "gold_count": int(gold["gold_count"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "candidate_unit_count": len(candidates),
                "consensus": consensus,
                "configurations": configurations,
            }
        )

    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = max(
        NON_FRC_BASELINES,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    strongest_frc = max(
        FRC_CONTROLS,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    comparisons = _paired_bootstrap(evidence, resamples=resamples)
    trigger_rate = float(
        np.mean([bool(row["consensus"]["consensus_expansion"]) for row in evidence])
    )
    difference_v43 = _set_difference_rate(evidence, ADAPTIVE_V43)
    difference_v44 = _set_difference_rate(evidence, GUARDED_V44)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: (
                _aggregate(evidence, method, (budget,))["evidence_macro_f1"],
                method,
            ),
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, CONSENSUS_GUARDED_V46, (budget,))[
                "evidence_macro_f1"
            ]
            - _aggregate(evidence, strongest, (budget,))["evidence_macro_f1"],
            6,
        )
    source_deltas: dict[str, Any] = {}
    for mode in SOURCE_MODES:
        subset = [row for row in evidence if row["source_mode"] == mode]
        if len(subset) < MINIMUM_CASES_PER_MODE:
            continue
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: (_aggregate(subset, method)["evidence_macro_f1"], method),
        )
        source_deltas[mode] = {
            "cases": len(subset),
            "strongest_non_frc": strongest,
            "delta": round(
                _aggregate(subset, CONSENSUS_GUARDED_V46)["evidence_macro_f1"]
                - _aggregate(subset, strongest)["evidence_macro_f1"],
                6,
            ),
        }
    ceiling_rate = float(
        np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence])
    )
    unit_reduction = (
        aggregates[DYNAMIC_RANK]["mean_selected_unit_count"]
        - aggregates[CONSENSUS_GUARDED_V46]["mean_selected_unit_count"]
    )
    recall_drop = (
        aggregates[DYNAMIC_RANK]["macro_recall"]
        - aggregates[CONSENSUS_GUARDED_V46]["macro_recall"]
    )
    safety_deltas = [
        *budget_deltas.values(),
        *(float(item["delta"]) for item in source_deltas.values()),
    ]
    source_counts = Counter(str(row["source_mode"]) for row in evidence)
    query_fallback_rate = float(query_summary.get("fallback_rate", 1.0))
    frc_comparison = comparisons["v46_minus_strongest_frozen_frc_control"]
    non_frc_comparison = comparisons["v46_minus_strongest_non_frc"]
    checks = {
        "minimum_total_cases_met": len(evidence) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            source_counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
        "candidate_ceiling_complete_rate_at_least_0_95": ceiling_rate >= 0.95,
        "query_parser_fallback_rate_at_most_0_05": query_fallback_rate <= 0.05,
        "consensus_trigger_rate_at_least_0_05": trigger_rate >= 0.05,
        "consensus_trigger_rate_at_most_0_95": trigger_rate <= 0.95,
        "selection_set_difference_rate_vs_v43_at_least_0_05": difference_v43
        >= 0.05,
        "selection_set_difference_rate_vs_v44_at_least_0_05": difference_v44
        >= 0.05,
        "v46_minus_strongest_frc_point_at_least_0_005": frc_comparison["point"]
        >= 0.005,
        "v46_minus_strongest_frc_ci_low_above_0": frc_comparison["ci_low"] > 0,
        "v46_minus_strongest_non_frc_point_at_least_0_01": non_frc_comparison[
            "point"
        ]
        >= 0.01,
        "v46_minus_strongest_non_frc_ci_low_above_0": non_frc_comparison["ci_low"]
        > 0,
        "every_budget_and_supported_source_mode_delta_at_least_minus_0_02": min(
            safety_deltas,
            default=-1.0,
        )
        >= -0.02,
        "mean_selected_unit_reduction_at_least_0_50": unit_reduction >= 0.50,
        "evidence_recall_drop_at_most_0_02": recall_drop <= 0.02,
    }
    if not (
        checks["minimum_total_cases_met"]
        and checks["minimum_cases_per_source_mode_met"]
        and checks["candidate_ceiling_complete_rate_at_least_0_95"]
    ):
        status = "TATQA_FULL_CONTEXT_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = "TATQA_CONSENSUS_GUARDED_SUPPORT_ESTABLISHED_ON_FULL_CONTEXT_POOL"
    else:
        status = "TATQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED"
    target_distribution = Counter(
        int(row["consensus"]["target_cardinality"]) for row in evidence
    )
    report = {
        "schema_version": "frc-tatqa-consensus-guarded-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "full_context_bounded_pool": True,
            "gold_joined_after_complete_score_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_non_frc": strongest_non_frc,
            "strongest_frozen_frc_control": strongest_frc,
            "family_comparison": comparisons,
            "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
            "consensus_trigger_rate": round(trigger_rate, 6),
            "selection_set_difference_rate_vs_v43": round(difference_v43, 6),
            "selection_set_difference_rate_vs_v44": round(difference_v44, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(unit_reduction, 6),
            "evidence_recall_drop_vs_dynamic": round(recall_drop, 6),
            "target_cardinality_distribution": {
                str(key): target_distribution[key] for key in sorted(target_distribution)
            },
            "budget_deltas": budget_deltas,
            "supported_source_mode_deltas": source_deltas,
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_v46_cases_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "The candidate pool is the complete context attached to each TAT-QA dev record, not open-corpus retrieval.",
            "Mapped text spans are evaluated at containing-paragraph granularity.",
            "The endpoint is evidence selection, not answer generation or numerical reasoning.",
            "This experiment cannot establish real SetR reproduction, flood-domain validity, or production readiness.",
        ],
    }
    return report, evidence


def _round_for_display(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_for_display(item) for item in value]
    return value


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    lines = [
        "# TAT-QA consensus-guarded atomic-role evaluation (v46)",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases: {rounded['metadata']['cases']}",
        f"- Strongest frozen FRC control: `{analysis['strongest_frozen_frc_control']}`",
        f"- Strongest non-FRC: `{analysis['strongest_non_frc']}`",
        f"- Consensus trigger rate: {analysis['consensus_trigger_rate']:.6f}",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Aggregate methods",
        "",
        "| Method | F1 | Precision | Recall | Complete | Units | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in analysis["aggregates"].items():
        lines.append(
            f"| `{method}` | {values['evidence_macro_f1']:.6f} | "
            f"{values['macro_precision']:.6f} | {values['macro_recall']:.6f} | "
            f"{values['complete_recall']:.6f} | "
            f"{values['mean_selected_unit_count']:.6f} | "
            f"{values['mean_selected_token_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a full-context evidence-selector experiment, not an official TAT-QA end-to-end leaderboard result or open-corpus financial-report retrieval evaluation. The v46 cases may not be reused for tuning or method selection.",
            "",
        ]
    )
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


def read_source_contexts(source_archive: Path) -> list[dict[str, Any]]:
    suffix = "/dataset_raw/tatqa_dataset_dev.json"
    with tarfile.open(source_archive, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("TAT-QA archive must contain exactly one registered dev JSON")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise ValueError("TAT-QA dev member is not a regular file")
        value = json.loads(handle.read().decode("utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("TAT-QA dev source schema mismatch")
    return [dict(row) for row in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("TAT-QA v46 protocol hash mismatch")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("TAT-QA v46 protocol experiment mismatch")
    if value.get("methods", {}).get("candidate_method") != CONSENSUS_GUARDED_V46:
        raise ValueError("TAT-QA v46 candidate method mismatch")
    if value.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("TAT-QA v46 budget registration mismatch")
    return value


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    validate_protocol(protocol_path)
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "frc-tatqa-consensus-guarded-atomic-roles-implementation-v46"
    ):
        raise ValueError("TAT-QA v46 implementation registration schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("TAT-QA v46 implementation experiment mismatch")
    if value.get("data_json_downloaded_or_opened_before_registration") is not False:
        raise ValueError("TAT-QA v46 implementation crossed the data boundary")
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("TAT-QA v46 implementation hash registration mismatch")
    invariants = value.get("synthetic_invariants_verified")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("TAT-QA v46 synthetic invariants are incomplete")
    if value.get("method_or_threshold_change_after_registration_forbidden") is not True:
        raise ValueError("TAT-QA v46 implementation is not frozen")
    return value


def validate_execution_registration(
    execution_path: Path,
    *,
    implementation_path: Path,
    source_archive_path: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    value = json.loads(execution_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "frc-tatqa-consensus-guarded-atomic-roles-execution-v46"
    ):
        raise ValueError("TAT-QA v46 execution registration schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("TAT-QA v46 execution experiment mismatch")
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive_path),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("TAT-QA v46 execution artifact hash mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("TAT-QA v46 execution coverage gate is not open")
    for field in ("query_generation_started", "neural_scoring_started", "metrics_computed"):
        if value.get(field) is not False:
            raise ValueError(f"TAT-QA v46 execution field changed: {field}")
    if value.get("negative_null_or_inconclusive_result_must_be_published") is not True:
        raise ValueError("TAT-QA v46 publication boundary is missing")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "CONSENSUS_GUARDED_V46",
    "EXPERIMENT_ID",
    "FrozenTatqaScorer",
    "OFFICIAL_REPOSITORY_REVISION",
    "build_candidate_coverage",
    "build_candidate_units",
    "build_gold_rows",
    "consensus_guarded_target_cardinality",
    "evaluate_tatqa",
    "gold_source_mode",
    "mapped_candidate_ids",
    "prepare_blind_cases",
    "read_source_contexts",
    "role_consensus_details",
    "select_sample",
    "select_v46",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
