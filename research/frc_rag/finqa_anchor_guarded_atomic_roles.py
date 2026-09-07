"""Prospective FinQA anchor-guarded atomic-role experiment (v45).

Dataset preparation, blind neural scoring, and gold joining stay separate.
Only sealed sampling may inspect ``qa.gold_inds`` before scoring; neither blind
queries nor neural score caches contain answer, program, or gold fields.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hover_dynamic_atomic_roles import (
    BASELINES,
    BUDGETS,
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    METHODS as V41_METHODS,
    RANK_COVERAGE_WEIGHT,
    FrozenDynamicHoVerScorer,
    merge_scored_candidates,
    select_candidates,
)
from research.frc_rag.ottqa_guarded_adaptive_atomic_roles import (
    ADAPTIVE_V43,
    GUARDED_V44,
    adaptive_v43_target_cardinality,
    guarded_target_cardinality,
)


SCHEMA_VERSION = "frc-finqa-anchor-guarded-atomic-roles-v45"
EXPERIMENT_ID = "FRC-FINQA-ANCHOR-GUARDED-ATOMIC-ROLES-V45"
DATASET_ID = "finqa_dev_full_context_supporting_facts_v45"
CAPABILITY = "full_context_supporting_fact_selection"
PROTOCOL_SHA256 = "17aeb0865234a644640f2672c9f1dc420fe9f1fbd477548ff8e54618755a1c25"
PROTOCOL_ERRATUM_SHA256 = (
    "00f0f76b13e2c973f70c5b2dc7f26b63911164d87b0c809b53306adc1464d29a"
)
OFFICIAL_REPOSITORY_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"

SAMPLE_SALT = "FRC-FINQA-V45|"
TARGET_CASES_PER_MODE = 120
TARGET_CASES = 360
MINIMUM_TOTAL_CASES = 240
MINIMUM_CASES_PER_MODE = 60
SOURCE_MODES = ("table_only", "text_only", "hybrid")
PERMANENT_EXCLUSIONS = {
    "ETR/2016/page_23.pdf-2",
    "INTC/2015/page_41.pdf-4",
}

TOP_K = 5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805

ANCHOR_GUARDED_V45 = "anchor_guarded_adaptive_cardinality_frc_v45"
NON_FRC_BASELINES = tuple(BASELINES)
METHODS = (*V41_METHODS, ADAPTIVE_V43, GUARDED_V44, ANCHOR_GUARDED_V45)

_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "candidate_ceiling",
    "exe_ans",
    "gold",
    "gold_candidate_ids",
    "gold_count",
    "gold_inds",
    "gold_source_mode",
    "program",
    "program_re",
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


def remove_space(text_in: str) -> str:
    """Byte-equivalent logic to the pinned official helper."""

    return " ".join(part for part in text_in.split(" ") if part != "")


def table_row_to_text(header: Sequence[Any], row: Sequence[Any]) -> str:
    """Corrected official FinQA table-row template at the pinned revision."""

    header_values = [str(value) for value in header]
    row_values = [str(value) for value in row]
    result = ""
    if header_values and header_values[0]:
        result += header_values[0] + " "
    row_name = row_values[0] if row_values else ""
    for head, cell in zip(header_values[1:], row_values[1:], strict=False):
        result += "the " + row_name + " of " + head + " is " + cell + " ; "
    return remove_space(result).strip()


def _candidate_keys(row: dict[str, Any]) -> set[str]:
    pre_text = row.get("pre_text")
    post_text = row.get("post_text")
    table = row.get("table")
    if (
        not isinstance(pre_text, list)
        or not pre_text
        or not isinstance(post_text, list)
        or not post_text
    ):
        return set()
    if (
        not isinstance(table, list)
        or len(table) < 2
        or not isinstance(table[0], list)
        or not all(isinstance(table_row, list) for table_row in table[1:])
    ):
        return set()
    texts = [*pre_text, *post_text]
    result = {
        f"text_{index}" for index, text in enumerate(texts) if _normalise(text)
    }
    header = table[0]
    result.update(
        f"table_{index}"
        for index, table_row in enumerate(table[1:], start=1)
        if _normalise(table_row_to_text(header, table_row))
    )
    return result


def _gold_keys(row: dict[str, Any]) -> tuple[str, ...]:
    qa = row.get("qa")
    if not isinstance(qa, dict) or not isinstance(qa.get("gold_inds"), dict):
        return ()
    return tuple(str(value) for value in qa["gold_inds"])


def gold_source_mode(row: dict[str, Any]) -> str:
    keys = _gold_keys(row)
    if keys and all(key.startswith("table_") for key in keys):
        return "table_only"
    if keys and all(key.startswith("text_") for key in keys):
        return "text_only"
    return "hybrid"


def _eligible(row: dict[str, Any]) -> bool:
    record_id = row.get("id")
    qa = row.get("qa")
    if not isinstance(record_id, str) or not record_id or record_id in PERMANENT_EXCLUSIONS:
        return False
    if not isinstance(qa, dict) or not _normalise(qa.get("question")):
        return False
    keys = _gold_keys(row)
    candidate_keys = _candidate_keys(row)
    return bool(keys) and len(candidate_keys) >= 3 and set(keys) <= candidate_keys


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    record_id = str(row["id"])
    digest = hashlib.sha256((SAMPLE_SALT + record_id).encode("utf-8")).hexdigest()
    return digest, record_id


def select_sample(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = {mode: [] for mode in SOURCE_MODES}
    total_rows = 0
    excluded = Counter()
    for raw in rows:
        total_rows += 1
        row = dict(raw)
        if str(row.get("id", "")) in PERMANENT_EXCLUSIONS:
            excluded["readme_example_id"] += 1
            continue
        if not _eligible(row):
            excluded["ineligible_schema_or_gold_mapping"] += 1
            continue
        pools[gold_source_mode(row)].append(row)
    for values in pools.values():
        values.sort(key=_sample_key)

    selected_by_mode = {
        mode: list(pools[mode][:TARGET_CASES_PER_MODE]) for mode in SOURCE_MODES
    }
    selected_ids = {
        str(row["id"])
        for values in selected_by_mode.values()
        for row in values
    }
    if sum(len(values) for values in selected_by_mode.values()) < TARGET_CASES:
        remainder = sorted(
            (
                row
                for mode in SOURCE_MODES
                for row in pools[mode]
                if str(row["id"]) not in selected_ids
            ),
            key=_sample_key,
        )
        for row in remainder:
            if sum(len(values) for values in selected_by_mode.values()) >= TARGET_CASES:
                break
            selected_by_mode[gold_source_mode(row)].append(row)
            selected_ids.add(str(row["id"]))

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
        "total_source_rows": total_rows,
        "eligible_by_source_mode": eligible_counts,
        "selected_by_source_mode": selected_counts,
        "selected_rows": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "checks": checks,
        "ready_for_blind_preparation": all(checks.values()),
        "sample_order_commitment": canonical_json_sha256(
            [hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest() for row in selected]
        ),
    }


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return max(1, len(encoded))


def build_candidate_units(row: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    texts = [*row["pre_text"], *row["post_text"]]
    for index, raw in enumerate(texts):
        text = _normalise(raw)
        if not text:
            continue
        units.append(
            {
                "canonical_id": f"text_{index}",
                "source_kind": "text",
                "text": text,
                "token_count": _token_count(tokenizer, text),
            }
        )
    table = row["table"]
    header = table[0]
    for index, table_row in enumerate(table[1:], start=1):
        text = _normalise(table_row_to_text(header, table_row))
        if not text:
            continue
        units.append(
            {
                "canonical_id": f"table_{index}",
                "source_kind": "table",
                "text": text,
                "token_count": _token_count(tokenizer, text),
            }
        )
    return units


def prepare_blind_cases(
    source_rows: Iterable[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(source_rows)
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
        record_id = str(row["id"])
        units = build_candidate_units(row, tokenizer)
        unit_by_id = {str(unit["canonical_id"]): unit for unit in units}
        if len(unit_by_id) != len(units):
            raise AssertionError("FinQA candidate canonical ids are not unique")
        if not set(_gold_keys(row)) <= set(unit_by_id):
            raise AssertionError("FinQA gold key did not map to the full-context pool")
        case_id = _opaque("f", SAMPLE_SALT, record_id)
        candidates: list[dict[str, Any]] = []
        map_units: list[dict[str, Any]] = []
        for unit in units:
            canonical = str(unit["canonical_id"])
            candidate_id = canonical
            candidates.append(
                {
                    "id": candidate_id,
                    "source_id": candidate_id,
                    "text": str(unit["text"]),
                    "token_count": int(unit["token_count"]),
                }
            )
            map_units.append(
                {
                    "candidate_id": candidate_id,
                    "canonical_id": canonical,
                    "source_kind": str(unit["source_kind"]),
                    "text": str(unit["text"]),
                    "text_sha256": hashlib.sha256(
                        str(unit["text"]).encode("utf-8")
                    ).hexdigest(),
                    "token_count": int(unit["token_count"]),
                }
            )
            token_costs.append(int(unit["token_count"]))
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "id": case_id,
            "query": str(row["qa"]["question"]),
            "candidates": candidates,
            "gold_fields_visible_to_generator": False,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("FinQA gold field leaked into blind cache")
        prepared.append(blind)
        candidate_maps.append(
            {
                "schema_version": "frc-finqa-v45-candidate-map-v1",
                "id": case_id,
                "record_id_sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
                "units": map_units,
            }
        )
        pool_sizes.append(len(candidates))

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


class FrozenFinqaScorer(FrozenDynamicHoVerScorer):
    """The registered v41 generator-independent neural scorer."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


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


def select_v45_anchor_guarded(
    candidates: Sequence[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    target = guarded_target_cardinality(candidates)
    feasible = [
        candidate
        for candidate in candidates
        if int(candidate["token_count"]) <= token_budget
    ]
    if not feasible:
        return []
    anchor = min(
        feasible,
        key=lambda candidate: (
            -float(candidate["scores"]["cross_encoder"]),
            str(candidate["id"]),
        ),
    )
    selected = [anchor]
    selected_ids = {str(anchor["id"])}
    selected_sources = {str(anchor["source_id"])}
    total = int(anchor["token_count"])
    utilities = _rank_utilities(candidates)
    best_by_role = {
        role: float(utilities[str(anchor["id"])][role]) for role in DYNAMIC_ROLES
    }
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
                max(
                    0.0,
                    float(utilities[candidate_id][role]) - best_by_role[role],
                )
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
                best_by_role[role],
                float(utilities[str(best["id"])][role]),
            )
    return selected


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


def select_v45(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method == ADAPTIVE_V43:
        return _adaptive_select(
            candidates,
            token_budget=token_budget,
            target=adaptive_v43_target_cardinality(candidates),
        )
    if method == GUARDED_V44:
        return _adaptive_select(
            candidates,
            token_budget=token_budget,
            target=guarded_target_cardinality(candidates),
        )
    if method == ANCHOR_GUARDED_V45:
        return select_v45_anchor_guarded(candidates, token_budget=token_budget)
    return select_candidates(list(candidates), method, token_budget=token_budget)


def build_gold_rows(
    source_rows: Iterable[dict[str, Any]], candidate_maps: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    source_by_commitment: dict[str, dict[str, Any]] = {}
    for raw in source_rows:
        row = dict(raw)
        record_id = str(row["id"])
        commitment = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        if commitment in source_by_commitment:
            raise ValueError("FinQA source record commitment is not unique")
        source_by_commitment[commitment] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        commitment = str(candidate_map["record_id_sha256"])
        source = source_by_commitment.get(commitment)
        if source is None:
            raise ValueError("FinQA candidate map record is missing from source")
        record_id = str(source["id"])
        candidate_by_canonical = {
            str(unit["canonical_id"]): str(unit["candidate_id"])
            for unit in candidate_map["units"]
        }
        gold_keys = _gold_keys(source)
        if not gold_keys or not set(gold_keys) <= set(candidate_by_canonical):
            raise ValueError("FinQA gold mapping is incomplete")
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "record_id": record_id,
                "source_mode": gold_source_mode(source),
                "gold_candidate_ids": [candidate_by_canonical[key] for key in gold_keys],
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
        "schema_version": "frc-finqa-v45-candidate-coverage-v1",
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
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int] = BUDGETS
) -> dict[str, float]:
    metrics = [
        row["configurations"][str(budget)]["methods"][method]["metrics"]
        for row in rows
        for budget in budgets
    ]
    return {
        "supporting_fact_macro_f1": float(np.mean([item["f1"] for item in metrics])),
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


def _bootstrap(rows: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    versus_v44 = np.empty(resamples, dtype=float)
    versus_dynamic = np.empty(resamples, dtype=float)
    versus_strongest = np.empty(resamples, dtype=float)
    for index in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(value)] for value in indices]
        candidate = _aggregate(sample, ANCHOR_GUARDED_V45)["supporting_fact_macro_f1"]
        versus_v44[index] = candidate - _aggregate(sample, GUARDED_V44)[
            "supporting_fact_macro_f1"
        ]
        versus_dynamic[index] = candidate - _aggregate(sample, DYNAMIC_RANK)[
            "supporting_fact_macro_f1"
        ]
        strongest = max(
            _aggregate(sample, method)["supporting_fact_macro_f1"]
            for method in NON_FRC_BASELINES
        )
        versus_strongest[index] = candidate - strongest
    candidate = _aggregate(rows, ANCHOR_GUARDED_V45)["supporting_fact_macro_f1"]
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "v45_minus_v44": _comparison(
            candidate - _aggregate(rows, GUARDED_V44)["supporting_fact_macro_f1"],
            versus_v44,
        ),
        "v45_minus_dynamic": _comparison(
            candidate - _aggregate(rows, DYNAMIC_RANK)["supporting_fact_macro_f1"],
            versus_dynamic,
        ),
        "v45_minus_bootstrap_strongest_non_frc": _comparison(
            candidate
            - max(
                _aggregate(rows, method)["supporting_fact_macro_f1"]
                for method in NON_FRC_BASELINES
            ),
            versus_strongest,
        ),
    }


def evaluate_finqa(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if [str(row["case_id"]) for row in gold_rows] != [
        str(row["id"]) for row in scored_rows
    ]:
        raise ValueError("FinQA gold and score rows are incomplete or misordered")
    evidence: list[dict[str, Any]] = []
    anchor_checks = 0
    anchor_retained = 0
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        candidates = merge_scored_candidates(dict(scored))
        configurations: dict[str, Any] = {}
        v43_target = adaptive_v43_target_cardinality(candidates)
        v44_target = guarded_target_cardinality(candidates)
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            feasible = [
                item for item in candidates if int(item["token_count"]) <= budget
            ]
            anchor_id = (
                str(
                    min(
                        feasible,
                        key=lambda item: (
                            -float(item["scores"]["cross_encoder"]),
                            str(item["id"]),
                        ),
                    )["id"]
                )
                if feasible
                else None
            )
            for method in METHODS:
                selected = select_v45(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
                        selected, list(gold["gold_candidate_ids"])
                    ),
                }
                if method == ADAPTIVE_V43:
                    methods[method]["target_cardinality"] = v43_target
                if method in {GUARDED_V44, ANCHOR_GUARDED_V45}:
                    methods[method]["target_cardinality"] = v44_target
                if method == ANCHOR_GUARDED_V45:
                    retained = anchor_id is None or anchor_id in methods[method]["selected_ids"]
                    methods[method]["anchor_id"] = anchor_id
                    methods[method]["anchor_retained"] = retained
                    anchor_checks += 1
                    anchor_retained += int(retained)
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "record_id_sha256": hashlib.sha256(
                    str(gold["record_id"]).encode("utf-8")
                ).hexdigest(),
                "source_mode": str(gold["source_mode"]),
                "gold_count": int(gold["gold_count"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "candidate_unit_count": len(candidates),
                "v43_target_cardinality": v43_target,
                "v44_v45_target_cardinality": v44_target,
                "configurations": configurations,
            }
        )

    aggregates = {
        method: {key: round(value, 6) for key, value in _aggregate(evidence, method).items()}
        for method in METHODS
    }
    strongest_non_frc = max(
        NON_FRC_BASELINES,
        key=lambda method: float(aggregates[method]["supporting_fact_macro_f1"]),
    )
    comparisons = _bootstrap(evidence, resamples=resamples)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(evidence, method, (budget,))[
                "supporting_fact_macro_f1"
            ],
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, ANCHOR_GUARDED_V45, (budget,))[
                "supporting_fact_macro_f1"
            ]
            - _aggregate(evidence, strongest, (budget,))["supporting_fact_macro_f1"],
            6,
        )
    source_deltas: dict[str, Any] = {}
    for mode in SOURCE_MODES:
        subset = [row for row in evidence if row["source_mode"] == mode]
        if len(subset) < MINIMUM_CASES_PER_MODE:
            continue
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(subset, method)["supporting_fact_macro_f1"],
        )
        source_deltas[mode] = {
            "cases": len(subset),
            "strongest_non_frc": strongest,
            "delta": round(
                _aggregate(subset, ANCHOR_GUARDED_V45)["supporting_fact_macro_f1"]
                - _aggregate(subset, strongest)["supporting_fact_macro_f1"],
                6,
            ),
        }
    source_counts = Counter(str(row["source_mode"]) for row in evidence)
    ceiling_rate = float(
        np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence])
    )
    anchor_rate = anchor_retained / anchor_checks if anchor_checks else 0.0
    unit_reduction = float(
        aggregates[DYNAMIC_RANK]["mean_selected_unit_count"]
    ) - float(aggregates[ANCHOR_GUARDED_V45]["mean_selected_unit_count"])
    recall_drop = float(aggregates[DYNAMIC_RANK]["macro_recall"]) - float(
        aggregates[ANCHOR_GUARDED_V45]["macro_recall"]
    )
    safety = [
        *budget_deltas.values(),
        *(float(value["delta"]) for value in source_deltas.values()),
    ]
    versus_v44 = comparisons["v45_minus_v44"]
    versus_dynamic = comparisons["v45_minus_dynamic"]
    versus_strongest = comparisons["v45_minus_bootstrap_strongest_non_frc"]
    checks = {
        "minimum_total_cases_met": len(evidence) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            source_counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
        "candidate_ceiling_complete_rate_at_least_0_95": ceiling_rate >= 0.95,
        "query_parser_fallback_rate_at_most_0_05": float(query_summary["fallback_rate"])
        <= 0.05,
        "anchor_retention_rate_equals_1": anchor_rate == 1.0,
        "v45_minus_v44_point_at_least_0_005": float(versus_v44["point"]) >= 0.005,
        "v45_minus_v44_ci_low_above_0": float(versus_v44["ci_low"]) > 0.0,
        "v45_minus_dynamic_point_at_least_0_005": float(versus_dynamic["point"])
        >= 0.005,
        "v45_minus_dynamic_ci_low_above_0": float(versus_dynamic["ci_low"]) > 0.0,
        "v45_minus_strongest_point_at_least_0_01": float(versus_strongest["point"])
        >= 0.01,
        "v45_minus_strongest_ci_low_above_0": float(versus_strongest["ci_low"])
        > 0.0,
        "every_budget_and_supported_source_mode_delta_at_least_minus_0_02": min(safety)
        >= -0.02
        if safety
        else False,
        "mean_selected_unit_reduction_at_least_0_50": unit_reduction >= 0.50,
        "supporting_fact_recall_drop_at_most_0_02": recall_drop <= 0.02,
    }
    informative = (
        checks["minimum_total_cases_met"]
        and checks["minimum_cases_per_source_mode_met"]
        and checks["candidate_ceiling_complete_rate_at_least_0_95"]
    )
    if not informative:
        status = "FINQA_FULL_CONTEXT_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = "FINQA_ANCHOR_GUARDED_SUPPORT_ESTABLISHED_ON_FULL_CONTEXT_POOL"
    else:
        status = "FINQA_ANCHOR_GUARDED_SUPPORT_NOT_ESTABLISHED"
    targets = Counter(int(row["v44_v45_target_cardinality"]) for row in evidence)
    report = {
        "schema_version": "frc-finqa-anchor-guarded-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_erratum_sha256": PROTOCOL_ERRATUM_SHA256,
            "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "methods": list(METHODS),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "gold_joined_after_complete_score_cache": True,
            "official_leaderboard_result": False,
            "full_context_bounded_pool": True,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_non_frc": strongest_non_frc,
            "family_comparison": comparisons,
            "budget_deltas": budget_deltas,
            "supported_source_mode_deltas": source_deltas,
            "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
            "anchor_retention_rate": round(anchor_rate, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(unit_reduction, 6),
            "supporting_fact_recall_drop_vs_dynamic": round(recall_drop, 6),
            "target_cardinality_distribution": {
                str(key): targets[key] for key in sorted(targets)
            },
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "The candidate pool is the complete context attached to each FinQA dev record, not open-corpus report retrieval.",
            "The endpoint is supporting-fact selection, not answer execution or program accuracy.",
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
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    lines = [
        "# FinQA anchor-guarded atomic-role evaluation (v45)",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases: {rounded['metadata']['cases']}",
        f"- Strongest non-FRC: `{analysis['strongest_non_frc']}`",
        f"- Candidate ceiling: {analysis['candidate_ceiling_complete_rate']:.6f}",
        f"- Anchor retention: {analysis['anchor_retention_rate']:.6f}",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Aggregate methods",
        "",
        "| Method | F1 | Precision | Recall | Complete | Units | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in analysis["aggregates"].items():
        lines.append(
            f"| `{method}` | {values['supporting_fact_macro_f1']:.6f} | "
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
            "This is a full-context supporting-fact selector experiment, not an official FinQA end-to-end leaderboard result or open-corpus financial-report retrieval evaluation. The v45 cases may not be reused for tuning or method selection.",
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


def read_source_rows(source_archive: Path) -> list[dict[str, Any]]:
    suffix = "/dataset/dev.json"
    with tarfile.open(source_archive, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("FinQA archive must contain exactly one dataset/dev.json")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise ValueError("FinQA dev member is not a regular file")
        value = json.loads(handle.read().decode("utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("FinQA dev source schema mismatch")
    return [dict(row) for row in value]


def validate_protocol(protocol_path: Path, erratum_path: Path) -> dict[str, Any]:
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("FinQA v45 protocol hash mismatch")
    if hashlib.sha256(erratum_path.read_bytes()).hexdigest() != PROTOCOL_ERRATUM_SHA256:
        raise ValueError("FinQA v45 protocol erratum hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FinQA v45 protocol experiment mismatch")
    if protocol.get("methods", {}).get("candidate_method") != ANCHOR_GUARDED_V45:
        raise ValueError("FinQA v45 candidate method mismatch")
    if erratum.get("base_protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("FinQA v45 erratum base hash mismatch")
    if erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("FinQA v45 erratum changed the method")
    return {"protocol": protocol, "erratum": erratum}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    protocol_erratum_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    official_finqa_utils_path: Path,
    official_general_utils_path: Path,
) -> dict[str, Any]:
    validate_protocol(protocol_path, protocol_erratum_path)
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "frc-finqa-anchor-guarded-atomic-roles-implementation-v45"
    ):
        raise ValueError("FinQA v45 implementation registration schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FinQA v45 implementation experiment mismatch")
    if value.get("data_json_downloaded_or_opened_before_registration") is not False:
        raise ValueError("FinQA v45 implementation registration crossed the data boundary")
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "protocol_erratum_sha256": _sha256(protocol_erratum_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
        "official_finqa_utils_sha256": _sha256(official_finqa_utils_path),
        "official_general_utils_sha256": _sha256(official_general_utils_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("FinQA v45 implementation hash registration mismatch")
    invariants = value.get("synthetic_invariants_verified")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("FinQA v45 synthetic invariants are incomplete")
    if value.get("method_or_threshold_change_after_registration_forbidden") is not True:
        raise ValueError("FinQA v45 implementation is not frozen")
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
        "frc-finqa-anchor-guarded-atomic-roles-execution-v45"
    ):
        raise ValueError("FinQA v45 execution registration schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FinQA v45 execution experiment mismatch")
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive_path),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("FinQA v45 execution artifact hash mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("FinQA v45 execution coverage gate is not open")
    for field in ("query_generation_started", "neural_scoring_started", "metrics_computed"):
        if value.get(field) is not False:
            raise ValueError(f"FinQA v45 execution registration field changed: {field}")
    if value.get("negative_null_or_inconclusive_result_must_be_published") is not True:
        raise ValueError("FinQA v45 publication boundary is missing")
    return value


__all__ = [
    "ADAPTIVE_V43",
    "ANCHOR_GUARDED_V45",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPERIMENT_ID",
    "FrozenFinqaScorer",
    "GUARDED_V44",
    "build_candidate_coverage",
    "build_candidate_units",
    "build_gold_rows",
    "evaluate_finqa",
    "gold_source_mode",
    "prepare_blind_cases",
    "read_source_rows",
    "remove_space",
    "select_sample",
    "select_v45",
    "select_v45_anchor_guarded",
    "table_row_to_text",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
