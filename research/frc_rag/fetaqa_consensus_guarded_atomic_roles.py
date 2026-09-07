"""Prospective FeTaQA confirmation of the frozen v46 selector (v47)."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.finqa_anchor_guarded_atomic_roles import (
    ADAPTIVE_V43,
    GUARDED_V44,
)
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    merge_scored_candidates,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    BUDGETS,
    CONSENSUS_GUARDED_V46,
    FRC_CONTROLS,
    METHODS,
    NON_FRC_BASELINES,
    FrozenTatqaScorer,
    role_consensus_details,
    select_v46,
)


SCHEMA_VERSION = "frc-fetaqa-consensus-guarded-atomic-roles-v47"
EXPERIMENT_ID = "FRC-FETAQA-CONSENSUS-GUARDED-ATOMIC-ROLES-V47"
DATASET_ID = "fetaqa_dev_full_table_supporting_cells_v47"
CAPABILITY = "full_table_supporting_cell_selection"
PROTOCOL_SHA256 = "29e10028399535034b5384186565660ac4d051d2cd37d4990b7bdcc175540a7d"
OFFICIAL_REPOSITORY_REVISION = "bbc441be807212d736c8c35a18146b530dde11d0"

SAMPLE_SALT = "FRC-FETAQA-V47|"
TARGET_CASES = 600
MINIMUM_CASES = 400
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260807
MINIMUM_STRATUM_CASES = 60

_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "gold",
    "gold_candidate_ids",
    "gold_count",
    "highlighted_cell_ids",
    "supporting_cell_ids",
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


def _official_id(row: dict[str, Any]) -> str:
    for key in ("feta_id", "id", "uid"):
        value = _normalise(row.get(key))
        if value:
            return value
    return ""


def _table(row: dict[str, Any]) -> list[list[Any]]:
    raw = row.get("table_array", row.get("table"))
    if isinstance(raw, dict):
        raw = raw.get("table")
    if not isinstance(raw, list) or not raw:
        return []
    if not all(isinstance(table_row, list) for table_row in raw):
        return []
    return [list(table_row) for table_row in raw]


def _coordinate(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        row, column = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    if row < 0 or column < 0:
        return None
    return row, column


def highlighted_candidate_ids(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("highlighted_cell_ids", row.get("highlighted_cells"))
    if not isinstance(raw, list):
        return ()
    result: list[str] = []
    for value in raw:
        coordinate = _coordinate(value)
        if coordinate is None:
            return ()
        result.append(f"cell_{coordinate[0]}_{coordinate[1]}")
    return tuple(dict.fromkeys(result))


def _token_count(tokenizer: Any, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def build_candidate_units(row: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    matrix = _table(row)
    page = _normalise(row.get("table_page_title", row.get("page_title")))
    section = _normalise(
        row.get("table_section_title", row.get("section_title"))
    )
    prefix = "; ".join(
        value for value in (f"page {page}" if page else "", f"section {section}" if section else "") if value
    )
    units: list[dict[str, Any]] = []
    for row_index, table_row in enumerate(matrix):
        row_header = _normalise(table_row[0]) if table_row else ""
        for column_index, raw_cell in enumerate(table_row):
            cell = _normalise(raw_cell)
            if not cell:
                continue
            column_header = (
                _normalise(matrix[0][column_index])
                if matrix and column_index < len(matrix[0])
                else ""
            )
            if row_index == 0:
                body = f"column header {column_index}: {cell}"
            elif column_index == 0:
                body = f"row header {row_index}: {cell}"
            else:
                body = (
                    f"row {row_header or row_index}; "
                    f"column {column_header or column_index}: {cell}"
                )
            text = f"{prefix}; {body}" if prefix else body
            units.append(
                {
                    "canonical_id": f"cell_{row_index}_{column_index}",
                    "text": text,
                    "token_count": _token_count(tokenizer, text),
                }
            )
    return units


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return list(text)


def _eligible(row: dict[str, Any]) -> bool:
    official_id = _official_id(row)
    question = _normalise(row.get("question"))
    units = build_candidate_units(row, _CharacterTokenizer())
    unit_ids = {str(unit["canonical_id"]) for unit in units}
    highlighted = highlighted_candidate_ids(row)
    return (
        bool(official_id)
        and bool(question)
        and len(units) >= 3
        and bool(highlighted)
        and set(highlighted) <= unit_ids
    )


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    official_id = _official_id(row)
    digest = hashlib.sha256((SAMPLE_SALT + official_id).encode()).hexdigest()
    return digest, official_id


def select_sample(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = 0
    excluded: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        total += 1
        row = dict(raw)
        official_id = _official_id(row)
        if not official_id:
            excluded["missing_id"] += 1
            continue
        if official_id in seen:
            raise ValueError("FeTaQA official id is not unique")
        seen.add(official_id)
        if not _eligible(row):
            excluded["ineligible_schema_or_highlight_mapping"] += 1
            continue
        eligible.append(row)
    eligible.sort(key=_sample_key)
    selected = eligible[:TARGET_CASES]
    checks = {"minimum_cases_met": len(selected) >= MINIMUM_CASES}
    gold_counts = Counter(
        min(3, len(highlighted_candidate_ids(row))) for row in selected
    )
    return selected, {
        "total_rows": total,
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "selected_gold_count_groups": {
            "one": gold_counts[1],
            "two": gold_counts[2],
            "three_or_more": gold_counts[3],
        },
        "checks": checks,
        "ready_for_blind_preparation": all(checks.values()),
        "sample_order_commitment": canonical_json_sha256(
            [hashlib.sha256(_official_id(row).encode()).hexdigest() for row in selected]
        ),
    }


def prepare_blind_cases(
    rows: Iterable[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(rows)
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
        official_id = _official_id(row)
        units = build_candidate_units(row, tokenizer)
        unit_ids = {str(unit["canonical_id"]) for unit in units}
        if len(unit_ids) != len(units):
            raise AssertionError("FeTaQA candidate ids are not unique")
        if not set(highlighted_candidate_ids(row)) <= unit_ids:
            raise AssertionError("FeTaQA highlighted cells do not map to candidates")
        case_id = "q" + hashlib.sha256(
            (SAMPLE_SALT + official_id).encode()
        ).hexdigest()[:20]
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
            "query": _normalise(row.get("question")),
            "candidates": candidates,
            "gold_fields_visible_to_generator": False,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("FeTaQA gold field leaked into blind cache")
        prepared.append(blind)
        candidate_maps.append(
            {
                "schema_version": "frc-fetaqa-v47-candidate-map-v1",
                "id": case_id,
                "official_id_sha256": hashlib.sha256(official_id.encode()).hexdigest(),
                "units": [
                    {
                        "candidate_id": str(unit["canonical_id"]),
                        "canonical_id": str(unit["canonical_id"]),
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

    quartiles = (
        [round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])]
        if pool_sizes
        else [0.0, 0.0, 0.0]
    )
    return prepared, candidate_maps, {
        "sampling": sampling,
        "prepared_cases": len(prepared),
        "candidate_pool_size": distribution(pool_sizes),
        "candidate_pool_quartile_boundaries": quartiles,
        "candidate_token_cost": distribution(token_costs),
        "ready_for_query_generation": len(prepared) == len(selected),
        "gold_fields_exported_to_blind_cache": False,
    }


class FrozenFetaqaScorer(FrozenTatqaScorer):
    """The unchanged neural scorer with FeTaQA metadata."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def _pool_quartile(size: int, boundaries: Sequence[float]) -> str:
    if size <= boundaries[0]:
        return "q1"
    if size <= boundaries[1]:
        return "q2"
    if size <= boundaries[2]:
        return "q3"
    return "q4"


def build_gold_rows(
    source_rows: Iterable[dict[str, Any]],
    candidate_maps: Sequence[dict[str, Any]],
    *,
    pool_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    source_by_commitment: dict[str, dict[str, Any]] = {}
    for raw in source_rows:
        row = dict(raw)
        official_id = _official_id(row)
        commitment = hashlib.sha256(official_id.encode()).hexdigest()
        if commitment in source_by_commitment:
            raise ValueError("FeTaQA official id commitment is not unique")
        source_by_commitment[commitment] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        commitment = str(candidate_map["official_id_sha256"])
        source = source_by_commitment.get(commitment)
        if source is None:
            raise ValueError("FeTaQA candidate-map row is missing from source")
        candidate_by_canonical = {
            str(unit["canonical_id"]): str(unit["candidate_id"])
            for unit in candidate_map["units"]
        }
        gold_keys = highlighted_candidate_ids(source)
        if not gold_keys or not set(gold_keys) <= set(candidate_by_canonical):
            raise ValueError("FeTaQA highlighted-cell mapping is incomplete")
        gold_rows = {int(value.split("_")[1]) for value in gold_keys}
        gold_count = len(gold_keys)
        pool_size = len(candidate_by_canonical)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "official_id": _official_id(source),
                "gold_candidate_ids": [candidate_by_canonical[key] for key in gold_keys],
                "gold_count": gold_count,
                "gold_count_group": (
                    "one" if gold_count == 1 else "two" if gold_count == 2 else "three_or_more"
                ),
                "row_dispersion": "single_row" if len(gold_rows) == 1 else "multiple_rows",
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _pool_quartile(
                    pool_size, pool_quartile_boundaries
                ),
                "candidate_ceiling_complete": True,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]],
    sampling: dict[str, Any],
    pool_quartile_boundaries: Sequence[float],
) -> dict[str, Any]:
    ceiling_rate = (
        float(np.mean([bool(row["candidate_ceiling_complete"]) for row in gold_rows]))
        if gold_rows
        else 0.0
    )
    checks = {
        "minimum_cases_met": len(gold_rows) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling_rate == 1.0,
    }
    return {
        "schema_version": "frc-fetaqa-v47-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
        "gold_count_groups": dict(Counter(row["gold_count_group"] for row in gold_rows)),
        "row_dispersion": dict(Counter(row["row_dispersion"] for row in gold_rows)),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "candidate_pool_quartile_boundaries": list(pool_quartile_boundaries),
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
        "supporting_cell_macro_f1": float(np.mean([item["f1"] for item in metrics])),
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
        sample_indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(value)] for value in sample_indices]
        candidate = _aggregate(sample, CONSENSUS_GUARDED_V46)[
            "supporting_cell_macro_f1"
        ]
        strongest_frc = max(
            _aggregate(sample, method)["supporting_cell_macro_f1"]
            for method in FRC_CONTROLS
        )
        strongest_non_frc = max(
            _aggregate(sample, method)["supporting_cell_macro_f1"]
            for method in NON_FRC_BASELINES
        )
        versus_frc[index] = candidate - strongest_frc
        versus_non_frc[index] = candidate - strongest_non_frc
    candidate = _aggregate(rows, CONSENSUS_GUARDED_V46)["supporting_cell_macro_f1"]
    strongest_frc = max(
        _aggregate(rows, method)["supporting_cell_macro_f1"] for method in FRC_CONTROLS
    )
    strongest_non_frc = max(
        _aggregate(rows, method)["supporting_cell_macro_f1"]
        for method in NON_FRC_BASELINES
    )
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "v47_minus_strongest_frozen_frc_control": _comparison(
            candidate - strongest_frc, versus_frc
        ),
        "v47_minus_strongest_non_frc": _comparison(
            candidate - strongest_non_frc, versus_non_frc
        ),
    }


def _set_difference_rate(rows: Sequence[dict[str, Any]], reference: str) -> float:
    values = []
    for row in rows:
        for budget in BUDGETS:
            methods = row["configurations"][str(budget)]["methods"]
            values.append(
                set(methods[CONSENSUS_GUARDED_V46]["selected_ids"])
                != set(methods[reference]["selected_ids"])
            )
    return float(np.mean(values)) if values else 0.0


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = max(
        NON_FRC_BASELINES,
        key=lambda method: (_aggregate(rows, method)["supporting_cell_macro_f1"], method),
    )
    delta = (
        _aggregate(rows, CONSENSUS_GUARDED_V46)["supporting_cell_macro_f1"]
        - _aggregate(rows, strongest)["supporting_cell_macro_f1"]
    )
    return strongest, round(delta, 6)


def evaluate_fetaqa(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not gold_rows or len(gold_rows) != len(scored_rows):
        raise ValueError("FeTaQA gold and scored rows are empty or misaligned")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("FeTaQA gold and scored case ids differ")
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
                "gold_count": int(gold["gold_count"]),
                "gold_count_group": str(gold["gold_count_group"]),
                "row_dispersion": str(gold["row_dispersion"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
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
        key=lambda method: (aggregates[method]["supporting_cell_macro_f1"], method),
    )
    strongest_frc = max(
        FRC_CONTROLS,
        key=lambda method: (aggregates[method]["supporting_cell_macro_f1"], method),
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
                _aggregate(evidence, method, (budget,))["supporting_cell_macro_f1"],
                method,
            ),
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, CONSENSUS_GUARDED_V46, (budget,))[
                "supporting_cell_macro_f1"
            ]
            - _aggregate(evidence, strongest, (budget,))["supporting_cell_macro_f1"],
            6,
        )
    strata: dict[str, Any] = {}
    for field, values in (
        ("gold_count_group", ("one", "two", "three_or_more")),
        ("row_dispersion", ("single_row", "multiple_rows")),
        ("candidate_pool_quartile", ("q1", "q2", "q3", "q4")),
    ):
        for value in values:
            subset = [row for row in evidence if row[field] == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{field}:{value}"] = {
                "cases": len(subset),
                "strongest_non_frc": strongest,
                "delta": delta,
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
        *(float(item["delta"]) for item in strata.values()),
    ]
    frc_comparison = comparisons["v47_minus_strongest_frozen_frc_control"]
    non_frc_comparison = comparisons["v47_minus_strongest_non_frc"]
    query_fallback_rate = float(query_summary.get("fallback_rate", 1.0))
    checks = {
        "minimum_cases_met": len(evidence) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling_rate == 1.0,
        "query_parser_fallback_rate_at_most_0_05": query_fallback_rate <= 0.05,
        "consensus_trigger_rate_at_least_0_05": trigger_rate >= 0.05,
        "consensus_trigger_rate_at_most_0_95": trigger_rate <= 0.95,
        "selection_set_difference_rate_vs_v43_at_least_0_05": difference_v43
        >= 0.05,
        "selection_set_difference_rate_vs_v44_at_least_0_05": difference_v44
        >= 0.05,
        "v47_minus_strongest_frc_point_at_least_0_005": frc_comparison["point"]
        >= 0.005,
        "v47_minus_strongest_frc_ci_low_above_0": frc_comparison["ci_low"] > 0,
        "v47_minus_strongest_non_frc_point_at_least_0_01": non_frc_comparison[
            "point"
        ]
        >= 0.01,
        "v47_minus_strongest_non_frc_ci_low_above_0": non_frc_comparison["ci_low"]
        > 0,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": min(
            safety_deltas,
            default=-1.0,
        )
        >= -0.02,
        "mean_selected_unit_reduction_at_least_0_50": unit_reduction >= 0.50,
        "supporting_cell_recall_drop_at_most_0_02": recall_drop <= 0.02,
    }
    if not (
        checks["minimum_cases_met"]
        and checks["candidate_ceiling_complete_rate_equals_1"]
    ):
        status = "FETAQA_FULL_TABLE_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = "FETAQA_CONSENSUS_GUARDED_SUPPORT_ESTABLISHED_ON_FULL_TABLE_POOL"
    else:
        status = "FETAQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED"
    targets = Counter(int(row["consensus"]["target_cardinality"]) for row in evidence)
    report = {
        "schema_version": "frc-fetaqa-consensus-guarded-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "full_table_bounded_pool": True,
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
            "supporting_cell_recall_drop_vs_dynamic": round(recall_drop, 6),
            "target_cardinality_distribution": {
                str(key): targets[key] for key in sorted(targets)
            },
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_v47_cases_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "FeTaQA supplies attached tables rather than open-corpus retrieval.",
            "The endpoint is supporting-cell selection, not free-form answer generation.",
            "The dataset does not test text-table routing.",
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
        "# FeTaQA consensus-guarded atomic-role evaluation (v47)",
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
            f"| `{method}` | {values['supporting_cell_macro_f1']:.6f} | "
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
            "This is a full-table supporting-cell selector experiment, not an official FeTaQA answer-generation leaderboard result. The v47 cases may not be reused for tuning or method selection.",
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
    suffix = "/data/fetaQA-v1_dev.jsonl"
    with tarfile.open(source_archive, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("FeTaQA archive must contain one registered dev JSONL")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise ValueError("FeTaQA dev member is not a regular file")
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("FeTaQA dev source schema mismatch")
    return [dict(row) for row in rows]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("FeTaQA v47 protocol hash mismatch")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FeTaQA v47 experiment mismatch")
    if value.get("methods", {}).get("candidate_method") != CONSENSUS_GUARDED_V46:
        raise ValueError("FeTaQA v47 changed the frozen v46 candidate")
    if value.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("FeTaQA v47 budget registration mismatch")
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
        "frc-fetaqa-consensus-guarded-atomic-roles-implementation-v47"
    ):
        raise ValueError("FeTaQA v47 implementation schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FeTaQA v47 implementation experiment mismatch")
    if value.get("data_jsonl_downloaded_or_opened_before_registration") is not False:
        raise ValueError("FeTaQA v47 crossed the data-access boundary")
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("FeTaQA v47 implementation hash mismatch")
    if value.get("v46_selector_replayed_without_change") is not True:
        raise ValueError("FeTaQA v47 did not freeze the v46 selector")
    invariants = value.get("synthetic_invariants_verified")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("FeTaQA v47 synthetic invariants are incomplete")
    if value.get("method_or_threshold_change_after_registration_forbidden") is not True:
        raise ValueError("FeTaQA v47 implementation is not frozen")
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
        "frc-fetaqa-consensus-guarded-atomic-roles-execution-v47"
    ):
        raise ValueError("FeTaQA v47 execution schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FeTaQA v47 execution experiment mismatch")
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive_path),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("FeTaQA v47 execution artifact hash mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("FeTaQA v47 coverage gate is not open")
    for field in ("query_generation_started", "neural_scoring_started", "metrics_computed"):
        if value.get(field) is not False:
            raise ValueError(f"FeTaQA v47 execution field changed: {field}")
    if value.get("negative_null_or_inconclusive_result_must_be_published") is not True:
        raise ValueError("FeTaQA v47 publication boundary is missing")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPERIMENT_ID",
    "FrozenFetaqaScorer",
    "OFFICIAL_REPOSITORY_REVISION",
    "build_candidate_coverage",
    "build_candidate_units",
    "build_gold_rows",
    "evaluate_fetaqa",
    "highlighted_candidate_ids",
    "prepare_blind_cases",
    "read_source_rows",
    "select_sample",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
