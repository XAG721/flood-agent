"""Prospective evaluation for the frozen v82 cascaded 2Wiki router."""

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
from research.frc_rag.twowiki_cascaded_style_residual_router import (
    CANDIDATE_METHOD,
    DEFAULT_ACTION,
    DIRECT_CROSS_ACTION,
    EXPERIMENT_ID as MODEL_DEVELOPMENT_EXPERIMENT_ID,
    LEARNED_ACTIONS,
    ROUTES,
    load_router_artifact as load_v82_router_artifact,
    select_method as select_v82_method,
)
from research.frc_rag.twowiki_question_router import (
    CANDIDATE_METHOD as V75_CANDIDATE_METHOD,
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    CANDIDATE_METHOD as V81_CANDIDATE_METHOD,
    load_router_artifact as load_v81_router_artifact,
)
from research.frc_rag.twowiki_residual_three_route_transfer import (
    CONTROL_METHODS as V81_CONTROL_METHODS,
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


EXPERIMENT_ID = "FRC-2WIKI-CASCADED-STYLE-RESIDUAL-PROSPECTIVE-V82"
SCHEMA_VERSION = "frc-twowiki-cascaded-style-residual-prospective-v82"
STAGES = ("development", "confirmation")
STAGE_SALTS = {
    "development": "FRC-2WIKI-V82-DEVELOPMENT|",
    "confirmation": "FRC-2WIKI-V82-CONFIRMATION|",
}
QUESTION_TYPE_QUOTAS = {question_type: 200 for question_type in QUESTION_TYPES}
CASES = sum(QUESTION_TYPE_QUOTAS.values())
HISTORY_CASES = 1000
V74_CASES = 800
V75_DEVELOPMENT_CASES = 800
V75_CONFIRMATION_CASES = 800
V81_DEVELOPMENT_CASES = 800
PRIOR_EXCLUDED_CASES = (
    HISTORY_CASES
    + V74_CASES
    + V75_DEVELOPMENT_CASES
    + V75_CONFIRMATION_CASES
    + V81_DEVELOPMENT_CASES
)
CONTROL_METHODS = (*V81_CONTROL_METHODS, V81_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
METRIC_SEEDS = {"development": 20261192, "confirmation": 20261193}
STRICT_GATES = {
    "exact_cases": CASES,
    "exact_question_type_quota": QUESTION_TYPE_QUOTAS,
    "prior_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.535,
    "candidate_complete_evidence_recall_at_least": 0.58,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_frozen_v75_f1_at_least": 0.005,
    "candidate_minus_frozen_v75_ci_low_above": 0.0,
    "every_question_type_delta_vs_best_control_at_least": -0.005,
    "direct_comparison_balanced_accuracy_at_least": 0.90,
    "override_action_fraction_at_least": 0.05,
    "at_least_two_routes_each_cover_at_least_fraction": 0.05,
    "largest_route_fraction_at_most": 0.9,
    "mean_selected_tokens_not_above_strongest_control_by_more_than_fraction": 0.05,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_control_point_at_least": -0.005,
    "candidate_minus_strongest_control_ci_low_at_least": -0.01,
    "candidate_minus_frozen_v75_point_at_least": 0.0,
    "every_question_type_delta_vs_best_control_at_least": -0.02,
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
        raise ValueError(f"v82 {label} exclusion source changed")
    return values


def load_prior_excluded_ids(
    history_pool_path: Path,
    v74_gold_path: Path,
    v75_development_gold_path: Path,
    v75_confirmation_gold_path: Path,
    v81_development_gold_path: Path,
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
        raise ValueError(f"v82 prior exclusion groups overlap: {overlaps}")
    union = set().union(*groups.values())
    if len(union) != PRIOR_EXCLUDED_CASES:
        raise ValueError("v82 prior exclusion union changed")
    return union, groups


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_ids: set[str],
    stage: str,
    question_type_quotas: dict[str, int] | None = None,
) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v82 stage: {stage}")
    quotas = QUESTION_TYPE_QUOTAS if question_type_quotas is None else question_type_quotas
    if set(quotas) != set(QUESTION_TYPES) or any(value <= 0 for value in quotas.values()):
        raise ValueError("v82 question-type quotas are invalid")
    rows = [
        {"id": str(row.get("_id") or row.get("id") or ""), "type": str(row["type"])}
        for row in metadata_rows
    ]
    if any(not row["id"] for row in rows) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("v82 source metadata ids are missing or duplicated")

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
                raise ValueError(f"v82 lacks {quota} untouched {question_type} cases")
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
        raise ValueError("v82 selected ids are missing from the source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[case_id])
        blind["source"] = f"2wikimultihopqa_dev_v82_{stage}"
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
        "v75_default_route": metadata.get("v75_default_route"),
        "v75_comparison_probability": metadata.get(
            "v75_comparison_probability", metadata.get("comparison_probability")
        ),
        "direct_comparison_probability": metadata.get(
            "direct_comparison_probability"
        ),
        "predicted_cross_override_gain": metadata.get(
            "predicted_cross_override_gain"
        ),
        "predicted_flip_override_gain": metadata.get(
            "predicted_flip_override_gain"
        ),
    }


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    *,
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
        raise ValueError("v82 model rows contain forbidden gold fields")
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        candidate_ids = [str(item["id"]) for item in row["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("v82 scored row has duplicate candidate ids")
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
        methods[CANDIDATE_METHOD] = _method_output(selected, metadata)
        if tuple(methods) != METHODS:
            raise ValueError("v82 registered method order drifted")
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
        raise ValueError(f"unsupported v82 stage: {stage}")
    v82_router, v82_classifier = load_v82_router_artifact(v82_router_path)
    v81_router = load_v81_router_artifact(v81_router_path)
    v80_router = load_v80_router_artifact(v80_router_path)
    v79_router = load_v79_router_artifact(v79_router_path)
    v78_router = load_v78_router_artifact(v78_router_path)
    v76_router = load_v76_router_artifact(v76_router_path)
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if (
        v82_router["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
        or v82_router["v81_router_file_sha256"] != sha256(v81_router_path)
    ):
        raise ValueError("v82 router dependency chain differs from frozen models")
    selections = write_selection_outputs(
        read_jsonl(scored_path),
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
        raise ValueError("v82 gold and selections do not align")
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
            float(case["methods"][CANDIDATE_METHOD]["evidence_f1"])
            for case in subset
        ) / len(subset)
        control_f1 = sum(
            float(case["methods"][best_control]["evidence_f1"])
            for case in subset
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
    v75_delta = comparisons[V75_CANDIDATE_METHOD]
    route_counts = Counter(
        str(case["methods"][CANDIDATE_METHOD]["route"]) for case in cases
    )
    action_counts = Counter(
        str(case["methods"][CANDIDATE_METHOD]["action"]) for case in cases
    )
    route_fractions = {route: route_counts[route] / len(cases) for route in ROUTES}
    override_fraction = sum(action_counts[action] for action in LEARNED_ACTIONS) / len(cases)
    material_routes = sum(value >= 0.05 for value in route_fractions.values())
    direct_labels = np.asarray(
        [case["question_type"] == "comparison" for case in cases], dtype=bool
    )
    direct_predictions = np.asarray(
        [
            case["methods"][CANDIDATE_METHOD]["action"] == DIRECT_CROSS_ACTION
            for case in cases
        ],
        dtype=bool,
    )
    direct_balanced_accuracy = float(
        (
            direct_predictions[direct_labels].mean()
            + (~direct_predictions[~direct_labels]).mean()
        )
        / 2.0
    )
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(aggregates[strongest]["mean_selected_tokens"])
    )
    counts = Counter(case["question_type"] for case in cases)
    checks = {
        "exact_cases_equals_800": len(cases) == CASES,
        "exact_question_type_quota": counts == Counter(QUESTION_TYPE_QUOTAS),
        "prior_or_stage_overlap_equals_0": prior_overlap == 0 and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_535": float(candidate["evidence_macro_f1"]) >= 0.535,
        "candidate_complete_evidence_recall_at_least_0_58": float(candidate["complete_evidence_recall"]) >= 0.58,
        "candidate_minus_strongest_control_f1_at_least_0_005": float(strongest_delta["point"]) >= 0.005,
        "candidate_minus_strongest_control_ci_low_above_0": float(strongest_delta["ci_low"]) > 0.0,
        "candidate_minus_frozen_v75_f1_at_least_0_005": float(v75_delta["point"]) >= 0.005,
        "candidate_minus_frozen_v75_ci_low_above_0": float(v75_delta["ci_low"]) > 0.0,
        "every_question_type_delta_vs_best_control_at_least_minus_0_005": all(
            float(row["delta"]) >= -0.005 for row in per_type.values()
        ),
        "direct_comparison_balanced_accuracy_at_least_0_90": direct_balanced_accuracy >= 0.90,
        "override_action_fraction_at_least_0_05": override_fraction >= 0.05,
        "at_least_two_routes_each_cover_at_least_0_05": material_routes >= 2,
        "largest_route_fraction_at_most_0_9": max(route_fractions.values()) <= 0.9,
        "mean_selected_tokens_within_1_05_of_strongest_control": token_ratio <= 1.05,
    }
    passed = all(checks.values())
    noninferiority_checks = {
        "candidate_minus_strongest_control_point_at_least_minus_0_005": float(strongest_delta["point"]) >= -0.005,
        "candidate_minus_strongest_control_ci_low_at_least_minus_0_01": float(strongest_delta["ci_low"]) >= -0.01,
        "candidate_minus_frozen_v75_point_at_least_0": float(v75_delta["point"]) >= 0.0,
        "every_question_type_delta_vs_best_control_at_least_minus_0_02": all(
            float(row["delta"]) >= -0.02 for row in per_type.values()
        ),
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    noninferiority_supported = all(noninferiority_checks.values())
    if stage == "development":
        status = (
            "2WIKI_V82_CASCADED_ROUTER_DEVELOPMENT_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "2WIKI_V82_CASCADED_ROUTER_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP"
        )
    else:
        status = (
            "2WIKI_V82_CASCADED_ROUTER_CASE_DISJOINT_ADVANTAGE_ESTABLISHED"
            if passed
            else "2WIKI_V82_CASCADED_ROUTER_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "model_development_experiment_id": MODEL_DEVELOPMENT_EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "question_type_distribution": dict(sorted(counts.items())),
            "prior_overlap": prior_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "v82_router_sha256": sha256(v82_router_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "gold_answer_or_official_type_used_by_runtime_router": False,
            "same_dataset_case_disjoint_target": True,
            "official_2wiki_leaderboard_result": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_registered_control": {
                "name": strongest,
                **aggregates[strongest],
            },
            "paired_f1_delta": comparisons,
            "router": {
                "action_counts": dict(sorted(action_counts.items())),
                "route_counts": dict(sorted(route_counts.items())),
                "route_fractions": {
                    route: round(value, 6) for route, value in route_fractions.items()
                },
                "override_action_fraction": round(override_fraction, 6),
                "default_action_fraction": round(
                    action_counts[DEFAULT_ACTION] / len(cases), 6
                ),
                "material_route_count": material_routes,
                "direct_comparison_balanced_accuracy": round(
                    direct_balanced_accuracy, 6
                ),
            },
            "per_question_type_delta_vs_best_control": per_type,
            "strict_gate_checks": checks,
            "noninferiority_checks": noninferiority_checks,
            "outcome": {
                "status": status,
                "strict_gate_passed": passed,
                "noninferiority_envelope_supported": noninferiority_supported,
                "confirmation_open_authorized": stage == "development" and passed,
                "further_same_source_confirmation_authorized": False,
                "reuse_stage_for_model_feature_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_gzip(output_dir / "cases.jsonl.gz", cases)
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    candidate = analysis["methods"][CANDIDATE_METHOD]
    strongest = analysis["strongest_registered_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    lines = [
        f"# 2Wiki v82 级联路由器{report['metadata']['stage']}实验",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强登记对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- 直接比较识别平衡准确率：`{analysis['router']['direct_comparison_balanced_accuracy']:.6f}`",
        f"- 动作分布：`{analysis['router']['action_counts']}`",
        f"- 路由分布：`{analysis['router']['route_counts']}`",
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
            "v82 使用同一公开数据集内、与 4200 个既往 2Wiki 样本互斥的新样本。",
            "它不是独立数据集、官方榜单、答案生成、真实 SetR、洪水专家验证或生产证据；",
            "无论本阶段结果如何，Gate 2 都保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v82 protocol experiment id")
    if tuple(protocol.get("stages", {})) != STAGES:
        raise ValueError("v82 protocol stages drifted")
    for stage in STAGES:
        contract = protocol["stages"].get(stage, {})
        if (
            contract.get("cases") != CASES
            or contract.get("question_type_quota") != QUESTION_TYPE_QUOTAS
        ):
            raise ValueError(f"v82 {stage} sample contract drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v82 registered controls drifted")
    if protocol.get("strict_gates") != STRICT_GATES:
        raise ValueError("v82 strict gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v82 noninferiority envelope drifted")
    if protocol.get("metrics", {}).get("seeds") != METRIC_SEEDS:
        raise ValueError("v82 metric seeds drifted")
    router = protocol.get("router", {})
    if (
        router.get("name") != CANDIDATE_METHOD
        or router.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v82 registered router drifted")
    if protocol.get("outcomes", {}).get("gate_2") != "NO-GO/SHADOW":
        raise ValueError("v82 Gate 2 contract drifted")


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
    "select_stage_ids",
    "validate_registered_protocol",
    "write_selection_outputs",
]
