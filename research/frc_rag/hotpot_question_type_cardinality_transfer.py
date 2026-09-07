"""Prospective HotpotQA evaluation for the frozen v84 cardinality router."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import (
    COMMON_RESOURCES,
    METHODS as V76_METHODS,
    _paired_bootstrap,
    load_history_ids,
    load_router_artifact as load_v76_router_artifact,
    prepare_case,
    select_method as select_v76_method,
)
from research.frc_rag.hotpot_question_type_cardinality_router import (
    BRIDGE_ACTION,
    CANDIDATE_METHOD,
    COMPARISON_ACTION,
    EXPERIMENT_ID as MODEL_DEVELOPMENT_EXPERIMENT_ID,
    load_router_artifact as load_v84_router_artifact,
    select_method as select_v84_method,
)
from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-HOTPOT-QUESTION-TYPE-CARDINALITY-PROSPECTIVE-V84"
SCHEMA_VERSION = "frc-hotpot-question-type-cardinality-prospective-v84"
STAGES = ("development", "confirmation")
STAGE_SALTS = {
    "development": "FRC-HOTPOT-V84-DEVELOPMENT|",
    "confirmation": "FRC-HOTPOT-V84-CONFIRMATION|",
}
QUESTION_TYPES = ("bridge", "comparison")
QUESTION_TYPE_QUOTAS = {"bridge": 400, "comparison": 200}
CASES = sum(QUESTION_TYPE_QUOTAS.values())
HISTORY_CASES = 1000
V76_DEVELOPMENT_CASES = 800
V76_CONFIRMATION_CASES = 800
PRIOR_EXCLUDED_CASES = (
    HISTORY_CASES + V76_DEVELOPMENT_CASES + V76_CONFIRMATION_CASES
)
PREFIX3_METHOD = "frozen_v76_prefix3_cardinality_control"
PREFIX4_METHOD = "frozen_v76_prefix4_cardinality_control"
CONTROL_METHODS = (*V76_METHODS, PREFIX3_METHOD, PREFIX4_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
METRIC_SEEDS = {"development": 20261207, "confirmation": 20261208}
SAFETY_ELIGIBILITY = {"complete_evidence_recall_at_least": 0.62}
STRICT_GATES = {
    "exact_cases": CASES,
    "exact_question_type_quota": QUESTION_TYPE_QUOTAS,
    "prior_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.62,
    "candidate_complete_evidence_recall_at_least": 0.62,
    "candidate_minus_strongest_safety_eligible_control_f1_at_least": 0.005,
    "candidate_minus_strongest_safety_eligible_control_ci_low_above": 0.0,
    "candidate_minus_fixed_prefix4_f1_at_least": 0.005,
    "candidate_minus_fixed_prefix4_ci_low_above": 0.0,
    "every_question_type_delta_vs_best_safety_eligible_control_at_least": -0.005,
    "comparison_classifier_balanced_accuracy_at_least": 0.88,
    "comparison_action_fraction_at_least": 0.2,
    "comparison_action_fraction_at_most": 0.5,
    "both_actions_each_cover_at_least_fraction": 0.1,
    "mean_selected_tokens_not_above_strongest_safety_eligible_control": 0.0,
    "complete_evidence_recall_delta_vs_strongest_safety_eligible_at_least": -0.08,
    "evidence_macro_recall_delta_vs_strongest_safety_eligible_at_least": -0.08,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_safety_eligible_control_point_at_least": 0.0,
    "candidate_minus_strongest_safety_eligible_control_ci_low_at_least": -0.005,
    "every_question_type_delta_vs_best_safety_eligible_control_at_least": -0.01,
    "candidate_complete_evidence_recall_at_least": 0.62,
    "invalid_selector_output_rate": 0.0,
}


def _order_key(stage: str, case_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{STAGE_SALTS[stage]}{case_id}".encode("utf-8")
    ).hexdigest()
    return digest, case_id


def _read_exact_ids(path: Path, expected: int, label: str) -> set[str]:
    rows = read_jsonl(path)
    values = {str(row.get("id") or row.get("case_id") or "") for row in rows}
    if len(rows) != expected or len(values) != expected or "" in values:
        raise ValueError(f"v84 {label} exclusion source changed")
    return values


def load_prior_excluded_ids(
    history_path: Path,
    v76_development_gold_path: Path,
    v76_confirmation_gold_path: Path,
) -> tuple[set[str], dict[str, set[str]]]:
    groups = {
        "history": load_history_ids(history_path),
        "v76_development": _read_exact_ids(
            v76_development_gold_path,
            V76_DEVELOPMENT_CASES,
            "v76 development",
        ),
        "v76_confirmation": _read_exact_ids(
            v76_confirmation_gold_path,
            V76_CONFIRMATION_CASES,
            "v76 confirmation",
        ),
    }
    names = tuple(groups)
    overlaps = {
        f"{names[left]}__{names[right]}": len(
            groups[names[left]] & groups[names[right]]
        )
        for left in range(len(names))
        for right in range(left + 1, len(names))
    }
    if any(overlaps.values()):
        raise ValueError(f"v84 prior exclusion groups overlap: {overlaps}")
    union = set().union(*groups.values())
    if len(union) != PRIOR_EXCLUDED_CASES:
        raise ValueError("v84 prior exclusion union changed")
    return union, groups


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_ids: set[str],
    stage: str,
    question_type_quotas: dict[str, int] | None = None,
) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v84 stage: {stage}")
    quotas = QUESTION_TYPE_QUOTAS if question_type_quotas is None else question_type_quotas
    if set(quotas) != set(QUESTION_TYPES) or any(value <= 0 for value in quotas.values()):
        raise ValueError("v84 question-type quotas are invalid")
    rows = [
        {"id": str(row.get("id") or ""), "type": str(row.get("type") or "")}
        for row in metadata_rows
    ]
    if any(not row["id"] or row["type"] not in QUESTION_TYPES for row in rows):
        raise ValueError("v84 HotpotQA source metadata is invalid")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("v84 HotpotQA source metadata ids are duplicated")

    def choose(local_stage: str, local_excluded: set[str]) -> list[str]:
        selected: list[str] = []
        for question_type in QUESTION_TYPES:
            eligible = [
                row["id"]
                for row in rows
                if row["type"] == question_type and row["id"] not in local_excluded
            ]
            eligible.sort(key=lambda case_id: _order_key(local_stage, case_id))
            quota = int(quotas[question_type])
            if len(eligible) < quota:
                raise ValueError(
                    f"v84 lacks {quota} untouched {question_type} cases"
                )
            selected.extend(eligible[:quota])
        return selected

    development = choose("development", excluded_ids)
    if stage == "development":
        return development
    return choose("confirmation", excluded_ids | set(development))


def prepare_stage(
    source_path: Path,
    history_path: Path,
    v76_development_gold_path: Path,
    v76_confirmation_gold_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    excluded, groups = load_prior_excluded_ids(
        history_path, v76_development_gold_path, v76_confirmation_gold_path
    )
    metadata = pq.read_table(source_path, columns=["id", "type"]).to_pylist()
    source_counts = Counter(str(row["type"]) for row in metadata)
    remaining_counts = Counter(
        str(row["type"])
        for row in metadata
        if str(row["id"]) not in excluded
    )
    selected_ids = select_stage_ids(metadata, excluded_ids=excluded, stage=stage)
    selected_set = set(selected_ids)
    frame = pd.read_parquet(source_path)
    rows_by_id = {
        str(row["id"]): row
        for row in frame.to_dict(orient="records")
        if str(row["id"]) in selected_set
    }
    if set(rows_by_id) != selected_set:
        raise ValueError("v84 selected ids are missing from the source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[case_id])
        blind["source"] = f"hotpotqa_distractor_validation_v84_{stage}"
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    counts = Counter(str(row["question_type"]) for row in gold_rows)
    remaining_after_both_stages = {
        question_type: remaining_counts[question_type]
        - 2 * QUESTION_TYPE_QUOTAS[question_type]
        for question_type in QUESTION_TYPES
    }
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "question_type_counts": dict(sorted(counts.items())),
        "source_question_type_counts": dict(sorted(source_counts.items())),
        "untouched_before_v84_question_type_counts": dict(
            sorted(remaining_counts.items())
        ),
        "remaining_after_both_registered_stages": remaining_after_both_stages,
        "prior_excluded_ids": len(excluded),
        "prior_exclusion_group_counts": {
            name: len(values) for name, values in groups.items()
        },
        "prior_overlap": len(selected_set & excluded),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
        "gold_fields_visible_to_model_or_router": False,
    }


def _method_output(
    selected: Sequence[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "selected_ids": [str(candidate["id"]) for candidate in selected],
        "selected_tokens": sum(
            int(candidate.get("token_count", 1)) for candidate in selected
        ),
        "route": metadata.get("route"),
        "action": metadata.get("action"),
        "comparison_probability": metadata.get("comparison_probability"),
        "predicted_soft_gain": metadata.get("predicted_soft_gain"),
        "base_retained_count": metadata.get("base_retained_count"),
        "retained_count": len(selected),
    }


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    *,
    v84_router: dict[str, Any],
    v84_classifier: dict[str, Any],
    v76_router: dict[str, Any],
    v75_model: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    forbidden = {"answer", "gold_evidence_ids", "gold", "gold_roles"}
    if any(
        forbidden & set(row)
        or any(forbidden & set(candidate) for candidate in row.get("candidates", []))
        for row in scored_rows
    ):
        raise ValueError("v84 model rows contain forbidden gold fields")
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        candidate_ids = [str(item["id"]) for item in row["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("v84 scored row has duplicate candidate ids")
        methods: dict[str, Any] = {}
        for method in V76_METHODS:
            selected, metadata = select_v76_method(
                row, method, v76_router, v75_model
            )
            methods[method] = _method_output(selected, metadata)
        base_ids = methods[V76_METHODS[-1]]["selected_ids"]
        candidates_by_id = {str(item["id"]): item for item in row["candidates"]}
        for method, count in ((PREFIX3_METHOD, 3), (PREFIX4_METHOD, 4)):
            selected = [candidates_by_id[case_id] for case_id in base_ids[:count]]
            methods[method] = _method_output(
                selected,
                {"route": V76_METHODS[-1], "action": f"retain_first_{count}"},
            )
        selected, metadata = select_v84_method(
            row, v84_router, v84_classifier, v76_router, v75_model
        )
        methods[CANDIDATE_METHOD] = _method_output(selected, metadata)
        if tuple(methods) != METHODS:
            raise ValueError("v84 registered method order drifted")
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_ids": candidate_ids,
                "methods": methods,
            }
        )
    write_jsonl(path, outputs)
    return outputs


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "evidence_macro_f1": round(
            sum(float(row["evidence_f1"]) for row in rows) / len(rows), 6
        ),
        "evidence_macro_recall": round(
            sum(float(row["evidence_recall"]) for row in rows) / len(rows), 6
        ),
        "complete_evidence_recall": round(
            sum(float(row["complete_evidence"]) for row in rows) / len(rows), 6
        ),
        "mean_selected_tokens": round(
            sum(float(row["selected_tokens"]) for row in rows) / len(rows), 6
        ),
    }


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    v84_router_path: Path,
    v76_router_path: Path,
    v75_model_path: Path,
    *,
    stage: str,
    output_dir: Path,
    prior_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v84 stage: {stage}")
    v84_router, v84_classifier = load_v84_router_artifact(v84_router_path)
    v76_router = load_v76_router_artifact(v76_router_path)
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if (
        v84_router["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
        or v84_router["v76_router_file_sha256"] != sha256(v76_router_path)
    ):
        raise ValueError("v84 router dependency chain differs from frozen models")
    selections = write_selection_outputs(
        read_jsonl(scored_path),
        v84_router=v84_router,
        v84_classifier=v84_classifier,
        v76_router=v76_router,
        v75_model=v75_model,
        path=selection_path,
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v84 gold and selections do not align")
    cases: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold):
        gold_row = gold[case_id]
        selection = selected[case_id]
        candidate_ids = set(selection["candidate_ids"])
        method_rows: dict[str, Any] = {}
        for method in METHODS:
            values = selection["methods"][method]
            ids = list(values["selected_ids"])
            row_invalid = (
                len(ids) != len(set(ids))
                or len(ids) > int(COMMON_RESOURCES["top_k"])
                or not set(ids) <= candidate_ids
                or int(values["selected_tokens"])
                > int(COMMON_RESOURCES["token_budget"])
            )
            invalid += int(row_invalid)
            metrics = evidence_metrics(gold_row["gold_evidence_ids"], ids)
            method_rows[method] = {
                **metrics,
                "complete_evidence": float(
                    set(gold_row["gold_evidence_ids"]) <= set(ids)
                ),
                "selected_tokens": int(values["selected_tokens"]),
                "route": values.get("route"),
                "action": values.get("action"),
                "comparison_probability": values.get("comparison_probability"),
                "retained_count": len(ids),
                "invalid": row_invalid,
            }
        cases.append(
            {
                "case_id": case_id,
                "question_type": str(gold_row["question_type"]),
                "gold_evidence_count": len(gold_row["gold_evidence_ids"]),
                "methods": method_rows,
            }
        )
    aggregates = {
        method: _aggregate([case["methods"][method] for case in cases])
        for method in METHODS
    }
    raw_strongest = max(
        CONTROL_METHODS,
        key=lambda method: (
            aggregates[method]["evidence_macro_f1"],
            -CONTROL_METHODS.index(method),
        ),
    )
    eligible_controls = [
        method
        for method in CONTROL_METHODS
        if aggregates[method]["complete_evidence_recall"]
        >= SAFETY_ELIGIBILITY["complete_evidence_recall_at_least"]
    ]
    if not eligible_controls:
        raise ValueError("v84 has no safety-eligible registered control")
    strongest = max(
        eligible_controls,
        key=lambda method: (
            aggregates[method]["evidence_macro_f1"],
            -CONTROL_METHODS.index(method),
        ),
    )
    candidate_values = np.asarray(
        [case["methods"][CANDIDATE_METHOD]["evidence_f1"] for case in cases]
    )
    comparisons = {
        method: _paired_bootstrap(
            candidate_values,
            np.asarray([case["methods"][method]["evidence_f1"] for case in cases]),
            seed=METRIC_SEEDS[stage] + index,
        )
        for index, method in enumerate(CONTROL_METHODS)
    }
    per_type: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        subset = [case for case in cases if case["question_type"] == question_type]
        best_control = max(
            eligible_controls,
            key=lambda method: (
                sum(float(case["methods"][method]["evidence_f1"]) for case in subset),
                -CONTROL_METHODS.index(method),
            ),
        )
        candidate_f1 = sum(
            float(case["methods"][CANDIDATE_METHOD]["evidence_f1"])
            for case in subset
        ) / len(subset)
        control_f1 = sum(
            float(case["methods"][best_control]["evidence_f1"]) for case in subset
        ) / len(subset)
        per_type[question_type] = {
            "cases": len(subset),
            "best_safety_eligible_control": best_control,
            "candidate_evidence_macro_f1": round(candidate_f1, 6),
            "best_control_evidence_macro_f1": round(control_f1, 6),
            "delta": round(candidate_f1 - control_f1, 6),
        }
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_aggregate = aggregates[strongest]
    strongest_delta = comparisons[strongest]
    prefix4_delta = comparisons[PREFIX4_METHOD]
    action_counts = Counter(
        str(case["methods"][CANDIDATE_METHOD]["action"]) for case in cases
    )
    retained_counts = Counter(
        int(case["methods"][CANDIDATE_METHOD]["retained_count"]) for case in cases
    )
    labels = np.asarray(
        [case["question_type"] == "comparison" for case in cases], dtype=bool
    )
    predictions = np.asarray(
        [
            case["methods"][CANDIDATE_METHOD]["action"] == COMPARISON_ACTION
            for case in cases
        ],
        dtype=bool,
    )
    comparison_balanced_accuracy = float(
        (predictions[labels].mean() + (~predictions[~labels]).mean()) / 2
    )
    comparison_fraction = action_counts[COMPARISON_ACTION] / len(cases)
    action_fractions = {
        action: action_counts[action] / len(cases)
        for action in (COMPARISON_ACTION, BRIDGE_ACTION)
    }
    complete_delta = (
        candidate["complete_evidence_recall"]
        - strongest_aggregate["complete_evidence_recall"]
    )
    recall_delta = (
        candidate["evidence_macro_recall"]
        - strongest_aggregate["evidence_macro_recall"]
    )
    strict_checks = {
        "exact_cases_equals_600": len(cases) == STRICT_GATES["exact_cases"],
        "exact_question_type_quota": Counter(case["question_type"] for case in cases)
        == Counter(STRICT_GATES["exact_question_type_quota"]),
        "prior_or_stage_overlap_equals_0": prior_overlap == 0 and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_62": candidate["evidence_macro_f1"]
        >= STRICT_GATES["candidate_evidence_macro_f1_at_least"],
        "candidate_complete_evidence_recall_at_least_0_62": candidate[
            "complete_evidence_recall"
        ]
        >= STRICT_GATES["candidate_complete_evidence_recall_at_least"],
        "candidate_minus_strongest_safety_eligible_f1_at_least_0_005": strongest_delta[
            "point"
        ]
        >= STRICT_GATES[
            "candidate_minus_strongest_safety_eligible_control_f1_at_least"
        ],
        "candidate_minus_strongest_safety_eligible_ci_low_above_0": strongest_delta[
            "ci_low"
        ]
        > STRICT_GATES[
            "candidate_minus_strongest_safety_eligible_control_ci_low_above"
        ],
        "candidate_minus_fixed_prefix4_f1_at_least_0_005": prefix4_delta["point"]
        >= STRICT_GATES["candidate_minus_fixed_prefix4_f1_at_least"],
        "candidate_minus_fixed_prefix4_ci_low_above_0": prefix4_delta["ci_low"]
        > STRICT_GATES["candidate_minus_fixed_prefix4_ci_low_above"],
        "every_question_type_delta_vs_best_safety_eligible_at_least_minus_0_005": min(
            row["delta"] for row in per_type.values()
        )
        >= STRICT_GATES[
            "every_question_type_delta_vs_best_safety_eligible_control_at_least"
        ],
        "comparison_classifier_balanced_accuracy_at_least_0_88": comparison_balanced_accuracy
        >= STRICT_GATES["comparison_classifier_balanced_accuracy_at_least"],
        "comparison_action_fraction_at_least_0_2": comparison_fraction
        >= STRICT_GATES["comparison_action_fraction_at_least"],
        "comparison_action_fraction_at_most_0_5": comparison_fraction
        <= STRICT_GATES["comparison_action_fraction_at_most"],
        "both_actions_each_cover_at_least_0_1": min(action_fractions.values())
        >= STRICT_GATES["both_actions_each_cover_at_least_fraction"],
        "mean_selected_tokens_not_above_strongest_safety_eligible": candidate[
            "mean_selected_tokens"
        ]
        <= strongest_aggregate["mean_selected_tokens"]
        + STRICT_GATES[
            "mean_selected_tokens_not_above_strongest_safety_eligible_control"
        ],
        "complete_evidence_recall_delta_vs_strongest_safety_eligible_at_least_minus_0_08": complete_delta
        >= STRICT_GATES[
            "complete_evidence_recall_delta_vs_strongest_safety_eligible_at_least"
        ],
        "evidence_macro_recall_delta_vs_strongest_safety_eligible_at_least_minus_0_08": recall_delta
        >= STRICT_GATES[
            "evidence_macro_recall_delta_vs_strongest_safety_eligible_at_least"
        ],
    }
    noninferiority_checks = {
        "candidate_minus_strongest_safety_eligible_point_at_least_0": strongest_delta[
            "point"
        ]
        >= NONINFERIORITY_ENVELOPE[
            "candidate_minus_strongest_safety_eligible_control_point_at_least"
        ],
        "candidate_minus_strongest_safety_eligible_ci_low_at_least_minus_0_005": strongest_delta[
            "ci_low"
        ]
        >= NONINFERIORITY_ENVELOPE[
            "candidate_minus_strongest_safety_eligible_control_ci_low_at_least"
        ],
        "every_question_type_delta_vs_best_safety_eligible_at_least_minus_0_01": min(
            row["delta"] for row in per_type.values()
        )
        >= NONINFERIORITY_ENVELOPE[
            "every_question_type_delta_vs_best_safety_eligible_control_at_least"
        ],
        "candidate_complete_evidence_recall_at_least_0_62": candidate[
            "complete_evidence_recall"
        ]
        >= NONINFERIORITY_ENVELOPE[
            "candidate_complete_evidence_recall_at_least"
        ],
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    strict_passed = all(strict_checks.values())
    if stage == "development":
        status = (
            "HOTPOT_V84_CARDINALITY_DEVELOPMENT_CONSTRAINED_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION"
            if strict_passed
            else "HOTPOT_V84_CARDINALITY_DEVELOPMENT_CONSTRAINED_ADVANTAGE_NOT_ESTABLISHED_STOP"
        )
    else:
        status = (
            "HOTPOT_V84_CARDINALITY_CASE_DISJOINT_CONSTRAINED_ADVANTAGE_ESTABLISHED"
            if strict_passed
            else "HOTPOT_V84_CARDINALITY_CONFIRMATION_CONSTRAINED_ADVANTAGE_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "model_development_experiment_id": MODEL_DEVELOPMENT_EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "question_type_distribution": dict(
                sorted(Counter(case["question_type"] for case in cases).items())
            ),
            "prior_overlap": prior_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "selection_written_before_gold_join": True,
            "official_question_type_answer_or_gold_used_by_runtime_router": False,
            "same_dataset_case_disjoint_target": True,
            "official_hotpotqa_leaderboard_result": False,
        },
        "analysis": {
            "methods": aggregates,
            "raw_strongest_registered_control": {
                "name": raw_strongest,
                **aggregates[raw_strongest],
                "safety_eligible": raw_strongest in eligible_controls,
            },
            "safety_eligibility": {
                **SAFETY_ELIGIBILITY,
                "eligible_controls": eligible_controls,
                "ineligible_controls": [
                    method for method in CONTROL_METHODS if method not in eligible_controls
                ],
            },
            "strongest_safety_eligible_control": {
                "name": strongest,
                **strongest_aggregate,
            },
            "paired_f1_delta": comparisons,
            "per_question_type_delta_vs_best_safety_eligible_control": per_type,
            "cardinality_control": {
                "action_counts": dict(sorted(action_counts.items())),
                "action_fractions": {
                    key: round(value, 6) for key, value in action_fractions.items()
                },
                "comparison_action_fraction": round(comparison_fraction, 6),
                "retained_count_distribution": {
                    str(key): retained_counts[key] for key in sorted(retained_counts)
                },
                "comparison_classifier_balanced_accuracy": round(
                    comparison_balanced_accuracy, 6
                ),
                "complete_evidence_recall_delta_vs_strongest_safety_eligible": round(
                    complete_delta, 6
                ),
                "evidence_macro_recall_delta_vs_strongest_safety_eligible": round(
                    recall_delta, 6
                ),
            },
            "strict_gate_checks": strict_checks,
            "noninferiority_checks": noninferiority_checks,
            "outcome": {
                "status": status,
                "strict_gate_passed": strict_passed,
                "noninferiority_envelope_supported": all(
                    noninferiority_checks.values()
                ),
                "confirmation_open_authorized": stage == "development" and strict_passed,
                "further_same_source_confirmation_authorized": stage == "development"
                and strict_passed,
                "raw_f1_superiority_over_every_registered_control_established": comparisons[
                    raw_strongest
                ]["ci_low"]
                > 0,
                "reuse_stage_for_model_feature_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_gzip(output_dir / "cases.jsonl.gz", cases)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    candidate = analysis["methods"][CANDIDATE_METHOD]
    strongest = analysis["strongest_safety_eligible_control"]
    raw = analysis["raw_strongest_registered_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    cardinality = analysis["cardinality_control"]
    lines = [
        f"# HotpotQA v84 问题类型感知证据基数实验（{report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 候选完整证据召回：`{candidate['complete_evidence_recall']:.6f}`",
        f"- 最强安全合格对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 约束内 F1 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- 原始 F1 最强对照：`{raw['name']}` / `{raw['evidence_macro_f1']:.6f}`（安全合格：`{raw['safety_eligible']}`）",
        f"- comparison 分类平衡准确率：`{cardinality['comparison_classifier_balanced_accuracy']:.6f}`",
        f"- 动作分布：`{cardinality['action_counts']}`",
        f"- 保留条数：`{cardinality['retained_count_distribution']}`",
        f"- Gate 2：`{analysis['outcome']['gate_2']}`",
        "",
        "## 严格门槛",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["strict_gate_checks"].items()
    )
    lines.extend(
        [
            "",
            "该实验检验的是完整证据召回不低于 0.62 条件下的 F1 优势；原始 F1 最强但不满足安全下限的对照仍完整披露。它不是 HotpotQA 官方排行榜、答案生成、真实 SetR、洪水专家评测或生产放行证据，Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v84 protocol experiment id")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("name") != CANDIDATE_METHOD
        or candidate.get("base_method") != V76_METHODS[-1]
        or candidate.get("comparison_retained_count") != 3
        or candidate.get("bridge_retained_count") != 4
        or candidate.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v84 protocol candidate configuration drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v84 protocol controls drifted")
    if protocol.get("common_resources") != COMMON_RESOURCES:
        raise ValueError("v84 protocol resources drifted")
    if protocol.get("safety_eligibility") != SAFETY_ELIGIBILITY:
        raise ValueError("v84 protocol safety eligibility drifted")
    if protocol.get("strict_gates") != STRICT_GATES:
        raise ValueError("v84 protocol strict gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v84 protocol noninferiority envelope drifted")
    for stage in STAGES:
        contract = protocol.get("stages", {}).get(stage, {})
        if (
            contract.get("cases") != CASES
            or contract.get("question_type_quota") != QUESTION_TYPE_QUOTAS
            or contract.get("selection_salt") != STAGE_SALTS[stage]
            or contract.get("paired_bootstrap_seed") != METRIC_SEEDS[stage]
        ):
            raise ValueError(f"v84 {stage} protocol drifted")


__all__ = [
    "CASES",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "METHODS",
    "NONINFERIORITY_ENVELOPE",
    "PREFIX3_METHOD",
    "PREFIX4_METHOD",
    "PRIOR_EXCLUDED_CASES",
    "QUESTION_TYPE_QUOTAS",
    "SAFETY_ELIGIBILITY",
    "STRICT_GATES",
    "evaluate_stage",
    "load_prior_excluded_ids",
    "prepare_stage",
    "select_stage_ids",
    "validate_registered_protocol",
    "write_selection_outputs",
]
