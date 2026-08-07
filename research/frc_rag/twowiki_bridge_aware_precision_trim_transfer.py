"""Prospective evaluation for the frozen v83 bridge-aware precision trimmer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import (
    COMMON_RESOURCES,
    load_router_artifact as load_v76_router_artifact,
)
from research.frc_rag.hotpot_three_route_router import (
    load_router_artifact as load_v78_router_artifact,
)
from research.frc_rag.musique_anchor_default_router import (
    load_router_artifact as load_v80_router_artifact,
)
from research.frc_rag.musique_target_three_route_router import (
    load_router_artifact as load_v79_router_artifact,
)
from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_bridge_aware_precision_trim_router import (
    BRIDGE_TRIM_ACTION,
    CANDIDATE_METHOD,
    EXPERIMENT_ID as MODEL_DEVELOPMENT_EXPERIMENT_ID,
    load_router_artifact as load_v83_router_artifact,
    select_method as select_v83_method,
)
from research.frc_rag.twowiki_cascaded_style_residual_router import (
    CANDIDATE_METHOD as V82_CANDIDATE_METHOD,
    load_router_artifact as load_v82_router_artifact,
    select_method as select_v82_method,
)
from research.frc_rag.twowiki_cascaded_style_residual_transfer import (
    CONTROL_METHODS as V82_BASE_CONTROL_METHODS,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    load_router_artifact as load_v81_router_artifact,
)
from research.frc_rag.twowiki_residual_three_route_transfer import (
    _aggregate,
    _paired_bootstrap,
    _select_all_methods,
)
from research.frc_rag.twowiki_support_path_closure import (
    QUESTION_TYPES,
    canonical_json_sha256,
    load_history_ids,
    prepare_case,
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-2WIKI-BRIDGE-AWARE-PRECISION-TRIM-PROSPECTIVE-V83"
SCHEMA_VERSION = "frc-twowiki-bridge-aware-precision-trim-prospective-v83"
STAGES = ("development", "confirmation")
STAGE_SALTS = {
    "development": "FRC-2WIKI-V83-DEVELOPMENT|",
    "confirmation": "FRC-2WIKI-V83-CONFIRMATION|",
}
QUESTION_TYPE_QUOTAS = {question_type: 200 for question_type in QUESTION_TYPES}
CASES = sum(QUESTION_TYPE_QUOTAS.values())
HISTORY_CASES = 1000
V74_CASES = 800
V75_DEVELOPMENT_CASES = 800
V75_CONFIRMATION_CASES = 800
V81_DEVELOPMENT_CASES = 800
V82_DEVELOPMENT_CASES = 800
PRIOR_EXCLUDED_CASES = (
    HISTORY_CASES
    + V74_CASES
    + V75_DEVELOPMENT_CASES
    + V75_CONFIRMATION_CASES
    + V81_DEVELOPMENT_CASES
    + V82_DEVELOPMENT_CASES
)
CONTROL_METHODS = (*V82_BASE_CONTROL_METHODS, V82_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
METRIC_SEEDS = {"development": 20261204, "confirmation": 20261205}
STRICT_GATES = {
    "exact_cases": CASES,
    "exact_question_type_quota": QUESTION_TYPE_QUOTAS,
    "prior_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.55,
    "candidate_complete_evidence_recall_at_least": 0.58,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_frozen_v82_f1_at_least": 0.005,
    "candidate_minus_frozen_v82_ci_low_above": 0.0,
    "every_question_type_delta_vs_best_control_at_least": -0.005,
    "bridge_classifier_balanced_accuracy_at_least": 0.95,
    "trim_action_fraction_at_least": 0.15,
    "trim_action_fraction_at_most": 0.35,
    "both_retained_counts_each_cover_at_least_fraction": 0.05,
    "mean_selected_tokens_not_above_strongest_control": 0.0,
    "complete_evidence_recall_delta_vs_strongest_at_least": -0.03,
    "evidence_macro_recall_delta_vs_strongest_at_least": -0.03,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_control_point_at_least": 0.0,
    "candidate_minus_strongest_control_ci_low_at_least": -0.005,
    "every_question_type_delta_vs_best_control_at_least": -0.01,
    "complete_evidence_recall_delta_vs_strongest_at_least": -0.03,
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
        raise ValueError(f"v83 {label} exclusion source changed")
    return values


def load_prior_excluded_ids(
    history_pool_path: Path,
    v74_gold_path: Path,
    v75_development_gold_path: Path,
    v75_confirmation_gold_path: Path,
    v81_development_gold_path: Path,
    v82_development_gold_path: Path,
) -> tuple[set[str], dict[str, set[str]]]:
    groups = {
        "history": load_history_ids(history_pool_path, expected_count=HISTORY_CASES),
        "v74": _read_exact_ids(v74_gold_path, V74_CASES, "v74"),
        "v75_development": _read_exact_ids(
            v75_development_gold_path,
            V75_DEVELOPMENT_CASES,
            "v75 development",
        ),
        "v75_confirmation": _read_exact_ids(
            v75_confirmation_gold_path,
            V75_CONFIRMATION_CASES,
            "v75 confirmation",
        ),
        "v81_development": _read_exact_ids(
            v81_development_gold_path,
            V81_DEVELOPMENT_CASES,
            "v81 development",
        ),
        "v82_development": _read_exact_ids(
            v82_development_gold_path,
            V82_DEVELOPMENT_CASES,
            "v82 development",
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
        raise ValueError(f"v83 prior exclusion groups overlap: {overlaps}")
    union = set().union(*groups.values())
    if len(union) != PRIOR_EXCLUDED_CASES:
        raise ValueError("v83 prior exclusion union changed")
    return union, groups


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_ids: set[str],
    stage: str,
    question_type_quotas: dict[str, int] | None = None,
) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v83 stage: {stage}")
    quotas = (
        QUESTION_TYPE_QUOTAS if question_type_quotas is None else question_type_quotas
    )
    if set(quotas) != set(QUESTION_TYPES) or any(
        value <= 0 for value in quotas.values()
    ):
        raise ValueError("v83 question-type quotas are invalid")
    rows = [
        {"id": str(row.get("_id") or row.get("id") or ""), "type": str(row["type"])}
        for row in metadata_rows
    ]
    if any(not row["id"] for row in rows) or len({row["id"] for row in rows}) != len(
        rows
    ):
        raise ValueError("v83 source metadata ids are missing or duplicated")

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
                raise ValueError(f"v83 lacks {quota} untouched {question_type} cases")
            selected.extend(eligible[:quota])
        return selected

    development = choose("development", excluded_ids)
    if stage == "development":
        return development
    return choose("confirmation", excluded_ids | set(development))


def prepare_stage(
    source_path: Path,
    history_pool_path: Path,
    v74_gold_path: Path,
    v75_development_gold_path: Path,
    v75_confirmation_gold_path: Path,
    v81_development_gold_path: Path,
    v82_development_gold_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    excluded, groups = load_prior_excluded_ids(
        history_pool_path,
        v74_gold_path,
        v75_development_gold_path,
        v75_confirmation_gold_path,
        v81_development_gold_path,
        v82_development_gold_path,
    )
    metadata = pq.read_table(source_path, columns=["_id", "type"]).to_pylist()
    selected_ids = select_stage_ids(metadata, excluded_ids=excluded, stage=stage)
    selected_set = set(selected_ids)
    frame = pd.read_parquet(source_path)
    rows_by_id = {
        str(row["_id"]): row
        for row in frame.to_dict(orient="records")
        if str(row["_id"]) in selected_set
    }
    if set(rows_by_id) != selected_set:
        raise ValueError("v83 selected ids are missing from the source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[case_id])
        blind["source"] = f"2wikimultihopqa_dev_v83_{stage}"
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    counts = Counter(str(row["question_type"]) for row in gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "question_type_counts": dict(sorted(counts.items())),
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
        "v82_action": metadata.get("v82_action"),
        "v82_route": metadata.get("v82_route"),
        "bridge_probability": metadata.get("bridge_probability"),
        "base_retained_count": metadata.get("base_retained_count"),
        "retained_count": metadata.get("retained_count"),
        "v75_default_route": metadata.get("v75_default_route"),
        "v75_comparison_probability": metadata.get(
            "v75_comparison_probability", metadata.get("comparison_probability")
        ),
        "direct_comparison_probability": metadata.get("direct_comparison_probability"),
        "predicted_cross_override_gain": metadata.get("predicted_cross_override_gain"),
        "predicted_flip_override_gain": metadata.get("predicted_flip_override_gain"),
    }


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    *,
    v83_router: dict[str, Any],
    v83_classifier: dict[str, Any],
    v82_router: dict[str, Any],
    v82_classifier: dict[str, Any],
    v81_router: dict[str, Any],
    v80_router: dict[str, Any],
    v79_router: dict[str, Any],
    v78_router: dict[str, Any],
    v76_router: dict[str, Any],
    v75_model: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    forbidden = {"answer", "gold_evidence_ids", "gold", "gold_roles"}
    if any(forbidden & set(row) for row in scored_rows):
        raise ValueError("v83 model rows contain forbidden gold fields")
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        candidate_ids = [str(item["id"]) for item in row["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("v83 scored row has duplicate candidate ids")
        methods = _select_all_methods(
            row,
            v81_router=v81_router,
            v80_router=v80_router,
            v79_router=v79_router,
            v78_router=v78_router,
            v76_router=v76_router,
            v75_model=v75_model,
        )
        selected, metadata = select_v82_method(
            row, v82_router, v82_classifier, v81_router, v75_model
        )
        methods[V82_CANDIDATE_METHOD] = _method_output(selected, metadata)
        selected, metadata = select_v83_method(
            row,
            v83_router,
            v83_classifier,
            v82_router,
            v82_classifier,
            v81_router,
            v75_model,
        )
        methods[CANDIDATE_METHOD] = _method_output(selected, metadata)
        if tuple(methods) != METHODS:
            raise ValueError("v83 registered method order drifted")
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_ids": candidate_ids,
                "methods": methods,
            }
        )
    write_jsonl(path, outputs)
    return outputs


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    v83_router_path: Path,
    v82_router_path: Path,
    v81_router_path: Path,
    v80_router_path: Path,
    v79_router_path: Path,
    v78_router_path: Path,
    v76_router_path: Path,
    v75_model_path: Path,
    *,
    stage: str,
    output_dir: Path,
    prior_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v83 stage: {stage}")
    v83_router, v83_classifier = load_v83_router_artifact(v83_router_path)
    v82_router, v82_classifier = load_v82_router_artifact(v82_router_path)
    v81_router = load_v81_router_artifact(v81_router_path)
    v80_router = load_v80_router_artifact(v80_router_path)
    v79_router = load_v79_router_artifact(v79_router_path)
    v78_router = load_v78_router_artifact(v78_router_path)
    v76_router = load_v76_router_artifact(v76_router_path)
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if (
        v83_router["v75_router_model_payload_sha256"] != v75_artifact["model_sha256"]
        or v83_router["v81_router_file_sha256"] != sha256(v81_router_path)
        or v83_router["v82_router_file_sha256"] != sha256(v82_router_path)
    ):
        raise ValueError("v83 router dependency chain differs from frozen models")
    selections = write_selection_outputs(
        read_jsonl(scored_path),
        v83_router=v83_router,
        v83_classifier=v83_classifier,
        v82_router=v82_router,
        v82_classifier=v82_classifier,
        v81_router=v81_router,
        v80_router=v80_router,
        v79_router=v79_router,
        v78_router=v78_router,
        v76_router=v76_router,
        v75_model=v75_model,
        path=selection_path,
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v83 gold and selections do not align")
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
                "route": values["route"],
                "action": values["action"],
                "v82_action": values.get("v82_action"),
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
    strongest = max(
        CONTROL_METHODS,
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
            CONTROL_METHODS,
            key=lambda method: (
                sum(float(case["methods"][method]["evidence_f1"]) for case in subset),
                -CONTROL_METHODS.index(method),
            ),
        )
        candidate_f1 = sum(
            float(case["methods"][CANDIDATE_METHOD]["evidence_f1"]) for case in subset
        ) / len(subset)
        control_f1 = sum(
            float(case["methods"][best_control]["evidence_f1"]) for case in subset
        ) / len(subset)
        per_type[question_type] = {
            "cases": len(subset),
            "best_control": best_control,
            "candidate_evidence_macro_f1": round(candidate_f1, 6),
            "best_control_evidence_macro_f1": round(control_f1, 6),
            "delta": round(candidate_f1 - control_f1, 6),
        }
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_delta = comparisons[strongest]
    v82_delta = comparisons[V82_CANDIDATE_METHOD]
    action_counts = Counter(
        str(case["methods"][CANDIDATE_METHOD]["action"]) for case in cases
    )
    route_counts = Counter(
        str(case["methods"][CANDIDATE_METHOD]["route"]) for case in cases
    )
    retained_counts = Counter(
        int(case["methods"][CANDIDATE_METHOD]["retained_count"]) for case in cases
    )
    labels = np.asarray(
        [case["question_type"] == "bridge_comparison" for case in cases], dtype=bool
    )
    predictions = np.asarray(
        [
            case["methods"][CANDIDATE_METHOD]["action"] == BRIDGE_TRIM_ACTION
            for case in cases
        ],
        dtype=bool,
    )
    bridge_balanced_accuracy = float(
        (predictions[labels].mean() + (~predictions[~labels]).mean()) / 2
    )
    trim_fraction = action_counts[BRIDGE_TRIM_ACTION] / len(cases)
    retained_fractions = {
        str(count): retained_counts[count] / len(cases) for count in (4, 5)
    }
    strongest_aggregate = aggregates[strongest]
    complete_delta = (
        candidate["complete_evidence_recall"]
        - strongest_aggregate["complete_evidence_recall"]
    )
    recall_delta = (
        candidate["evidence_macro_recall"]
        - strongest_aggregate["evidence_macro_recall"]
    )
    strict_checks = {
        "exact_cases_equals_800": len(cases) == STRICT_GATES["exact_cases"],
        "exact_question_type_quota": Counter(case["question_type"] for case in cases)
        == Counter(STRICT_GATES["exact_question_type_quota"]),
        "prior_or_stage_overlap_equals_0": prior_overlap == 0 and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_55": candidate["evidence_macro_f1"]
        >= STRICT_GATES["candidate_evidence_macro_f1_at_least"],
        "candidate_complete_evidence_recall_at_least_0_58": candidate[
            "complete_evidence_recall"
        ]
        >= STRICT_GATES["candidate_complete_evidence_recall_at_least"],
        "candidate_minus_strongest_control_f1_at_least_0_005": strongest_delta["point"]
        >= STRICT_GATES["candidate_minus_strongest_registered_control_f1_at_least"],
        "candidate_minus_strongest_control_ci_low_above_0": strongest_delta["ci_low"]
        > STRICT_GATES["candidate_minus_strongest_registered_control_ci_low_above"],
        "candidate_minus_frozen_v82_f1_at_least_0_005": v82_delta["point"]
        >= STRICT_GATES["candidate_minus_frozen_v82_f1_at_least"],
        "candidate_minus_frozen_v82_ci_low_above_0": v82_delta["ci_low"]
        > STRICT_GATES["candidate_minus_frozen_v82_ci_low_above"],
        "every_question_type_delta_vs_best_control_at_least_minus_0_005": min(
            row["delta"] for row in per_type.values()
        )
        >= STRICT_GATES["every_question_type_delta_vs_best_control_at_least"],
        "bridge_classifier_balanced_accuracy_at_least_0_95": bridge_balanced_accuracy
        >= STRICT_GATES["bridge_classifier_balanced_accuracy_at_least"],
        "trim_action_fraction_at_least_0_15": trim_fraction
        >= STRICT_GATES["trim_action_fraction_at_least"],
        "trim_action_fraction_at_most_0_35": trim_fraction
        <= STRICT_GATES["trim_action_fraction_at_most"],
        "both_retained_counts_each_cover_at_least_0_05": min(
            retained_fractions.values()
        )
        >= STRICT_GATES["both_retained_counts_each_cover_at_least_fraction"],
        "mean_selected_tokens_not_above_strongest_control": candidate[
            "mean_selected_tokens"
        ]
        <= strongest_aggregate["mean_selected_tokens"]
        + STRICT_GATES["mean_selected_tokens_not_above_strongest_control"],
        "complete_evidence_recall_delta_vs_strongest_at_least_minus_0_03": complete_delta
        >= STRICT_GATES["complete_evidence_recall_delta_vs_strongest_at_least"],
        "evidence_macro_recall_delta_vs_strongest_at_least_minus_0_03": recall_delta
        >= STRICT_GATES["evidence_macro_recall_delta_vs_strongest_at_least"],
    }
    noninferiority_checks = {
        "candidate_minus_strongest_control_point_at_least_0": strongest_delta["point"]
        >= NONINFERIORITY_ENVELOPE["candidate_minus_strongest_control_point_at_least"],
        "candidate_minus_strongest_control_ci_low_at_least_minus_0_005": strongest_delta[
            "ci_low"
        ]
        >= NONINFERIORITY_ENVELOPE["candidate_minus_strongest_control_ci_low_at_least"],
        "every_question_type_delta_vs_best_control_at_least_minus_0_01": min(
            row["delta"] for row in per_type.values()
        )
        >= NONINFERIORITY_ENVELOPE[
            "every_question_type_delta_vs_best_control_at_least"
        ],
        "complete_evidence_recall_delta_vs_strongest_at_least_minus_0_03": complete_delta
        >= NONINFERIORITY_ENVELOPE[
            "complete_evidence_recall_delta_vs_strongest_at_least"
        ],
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    strict_passed = all(strict_checks.values())
    if stage == "development":
        status = (
            "2WIKI_V83_PRECISION_TRIM_DEVELOPMENT_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION"
            if strict_passed
            else "2WIKI_V83_PRECISION_TRIM_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP"
        )
    else:
        status = (
            "2WIKI_V83_PRECISION_TRIM_CASE_DISJOINT_ADVANTAGE_ESTABLISHED"
            if strict_passed
            else "2WIKI_V83_PRECISION_TRIM_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED"
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
            "official_2wiki_leaderboard_result": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_registered_control": {
                "name": strongest,
                **strongest_aggregate,
            },
            "paired_f1_delta": comparisons,
            "per_question_type_delta_vs_best_control": per_type,
            "precision_trim": {
                "action_counts": dict(sorted(action_counts.items())),
                "trim_action_fraction": round(trim_fraction, 6),
                "retained_count_distribution": {
                    str(key): retained_counts[key] for key in sorted(retained_counts)
                },
                "retained_count_fractions": {
                    key: round(value, 6) for key, value in retained_fractions.items()
                },
                "bridge_classifier_balanced_accuracy": round(
                    bridge_balanced_accuracy, 6
                ),
                "route_counts": dict(sorted(route_counts.items())),
                "complete_evidence_recall_delta_vs_strongest": round(complete_delta, 6),
                "evidence_macro_recall_delta_vs_strongest": round(recall_delta, 6),
            },
            "strict_gate_checks": strict_checks,
            "noninferiority_checks": noninferiority_checks,
            "outcome": {
                "status": status,
                "strict_gate_passed": strict_passed,
                "noninferiority_envelope_supported": all(
                    noninferiority_checks.values()
                ),
                "confirmation_open_authorized": (
                    stage == "development" and strict_passed
                ),
                "further_same_source_confirmation_authorized": (
                    stage == "development" and strict_passed
                ),
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
    strongest = analysis["strongest_registered_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    trim = analysis["precision_trim"]
    lines = [
        f"# 2Wiki v83 精度裁剪器{report['metadata']['stage']}实验",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强登记对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- bridge 分类平衡准确率：`{trim['bridge_classifier_balanced_accuracy']:.6f}`",
        f"- 动作分布：`{trim['action_counts']}`",
        f"- 保留条数：`{trim['retained_count_distribution']}`",
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
            "v83 使用同一公开数据集内、与 5000 个既往 2Wiki 样本互斥的新样本。",
            "它不是独立数据集、官方榜单、答案生成、真实 SetR、洪水专家验证或生产证据；",
            "无论本阶段结果如何，Gate 2 都保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v83 protocol experiment id")
    if tuple(protocol.get("stages", {})) != STAGES:
        raise ValueError("v83 protocol stages drifted")
    for stage in STAGES:
        contract = protocol["stages"].get(stage, {})
        if (
            contract.get("cases") != CASES
            or contract.get("question_type_quota") != QUESTION_TYPE_QUOTAS
        ):
            raise ValueError(f"v83 {stage} sample contract drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v83 registered controls drifted")
    if protocol.get("strict_gates") != STRICT_GATES:
        raise ValueError("v83 strict gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v83 noninferiority envelope drifted")
    if protocol.get("metrics", {}).get("seeds") != METRIC_SEEDS:
        raise ValueError("v83 metric seeds drifted")
    router = protocol.get("selector", {})
    if (
        router.get("name") != CANDIDATE_METHOD
        or router.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v83 registered selector drifted")
    if protocol.get("outcomes", {}).get("gate_2") != "NO-GO/SHADOW":
        raise ValueError("v83 Gate 2 contract drifted")


__all__ = [
    "CASES",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "METHODS",
    "NONINFERIORITY_ENVELOPE",
    "PRIOR_EXCLUDED_CASES",
    "QUESTION_TYPE_QUOTAS",
    "SCHEMA_VERSION",
    "STAGES",
    "STRICT_GATES",
    "evaluate_stage",
    "load_prior_excluded_ids",
    "prepare_stage",
    "render_markdown",
    "select_stage_ids",
    "validate_registered_protocol",
    "write_selection_outputs",
]
