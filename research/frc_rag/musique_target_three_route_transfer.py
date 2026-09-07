"""Prospective MuSiQue evaluation of the frozen v79 target-trained router."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import (
    COMMON_RESOURCES,
    select_method as select_v76_method,
)
from research.frc_rag.hotpot_three_route_router import ROUTES
from research.frc_rag.musique_graph_router_transfer import (
    hop_count_from_id,
    load_source_commitments,
    prepare_case as prepare_v77_case,
    source_id_commitment,
)
from research.frc_rag.musique_mean_calibrated_three_route import (
    CANDIDATE_METHOD as V78_CANDIDATE_METHOD,
    CONTROL_METHODS as V78_BASE_CONTROL_METHODS,
    load_calibration,
    select_calibrated_method,
)
from research.frc_rag.musique_target_three_route_router import (
    CANDIDATE_METHOD,
    EXPERIMENT_ID as MODEL_DEVELOPMENT_EXPERIMENT_ID,
    MODEL_CONFIGURATIONS,
    ROUTE_POLICY_CONFIGURATIONS,
    load_router_artifact as load_target_router_artifact,
    select_method as select_target_method,
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


EXPERIMENT_ID = "FRC-MUSIQUE-TARGET-TRAINED-THREE-ROUTE-TRANSFER-V79"
SCHEMA_VERSION = "frc-musique-target-trained-three-route-transfer-v79"
CONTROL_METHODS = (*V78_BASE_CONTROL_METHODS, V78_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
STAGE_SALTS = {
    "development": "FRC-MUSIQUE-V79-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V79-CONFIRMATION|",
}
HOP_QUOTAS = {2: 250, 3: 150, 4: 70}
CASES_PER_STAGE = sum(HOP_QUOTAS.values())
PRIOR_COMMITMENT_UNION = 16900
V77_CALIBRATION_CASES = 800
V78_TRAINING_CASES = 600
TOTAL_EXCLUDED_TARGET_IDS = (
    PRIOR_COMMITMENT_UNION + V77_CALIBRATION_CASES + V78_TRAINING_CASES
)
METRIC_SEEDS = {"development_seed": 20261141, "confirmation_seed": 20261142}
DEVELOPMENT_GATES = {
    "exact_cases": CASES_PER_STAGE,
    "exact_hop_quota": {str(key): value for key, value in HOP_QUOTAS.items()},
    "prior_training_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.55,
    "candidate_complete_evidence_recall_at_least": 0.55,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_frozen_v78_f1_at_least": 0.005,
    "candidate_minus_frozen_v78_ci_low_above": 0.0,
    "every_hop_delta_vs_best_control_at_least": -0.005,
    "at_least_two_routes_each_cover_at_least_fraction": 0.05,
    "largest_route_fraction_at_most": 0.9,
    "mean_selected_tokens_not_above_strongest_control_by_more_than_fraction": 0.05,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_control_point_at_least": -0.005,
    "candidate_minus_strongest_control_ci_low_at_least": -0.01,
    "candidate_minus_frozen_v78_point_at_least": 0.0,
    "every_hop_delta_vs_best_control_at_least": -0.02,
    "invalid_selector_output_rate": 0.0,
}


def _order_key(stage: str, source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{STAGE_SALTS[stage]}{source_id}".encode("utf-8")
    ).hexdigest()
    return digest, source_id


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_source_commitments: set[str],
    v77_calibration_source_ids: set[str],
    v78_training_source_ids: set[str],
    stage: str,
    hop_quotas: dict[int, int] | None = None,
) -> list[str]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"unsupported v79 stage: {stage}")
    quotas = HOP_QUOTAS if hop_quotas is None else hop_quotas
    if not quotas or any(
        hop not in {2, 3, 4} or quota <= 0 for hop, quota in quotas.items()
    ):
        raise ValueError("v79 hop quotas are invalid")
    ids = [
        str(row.get("id", "")).strip()
        for row in metadata_rows
        if row.get("answerable") is True
    ]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("v79 answerable metadata ids are empty or duplicated")
    if len(v77_calibration_source_ids) != V77_CALIBRATION_CASES:
        raise ValueError("v79 requires exactly 800 v77 calibration ids")
    if len(v78_training_source_ids) != V78_TRAINING_CASES:
        raise ValueError("v79 requires exactly 600 v78 training ids")
    if v77_calibration_source_ids & v78_training_source_ids:
        raise ValueError("v79 registered v77 and v78 exclusions overlap")

    def choose(current_stage: str, extra_excluded: set[str]) -> list[str]:
        selected: list[str] = []
        for hop, quota in sorted(quotas.items()):
            eligible = [
                source_id
                for source_id in ids
                if hop_count_from_id(source_id) == hop
                and source_id_commitment(source_id)
                not in excluded_source_commitments
                and source_id not in v77_calibration_source_ids
                and source_id not in v78_training_source_ids
                and source_id not in extra_excluded
            ]
            eligible.sort(key=lambda value: _order_key(current_stage, value))
            if len(eligible) < quota:
                raise ValueError(f"v79 lacks {quota} untouched {hop}-hop cases")
            selected.extend(eligible[:quota])
        return selected

    development = choose("development", set())
    if stage == "development":
        return development
    return choose("confirmation", set(development))


def prepare_stage(
    source_path: Path,
    prior_exclusion_paths: Sequence[Path],
    v77_gold_path: Path,
    v78_gold_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    prior = load_source_commitments(prior_exclusion_paths)
    if len(prior) != PRIOR_COMMITMENT_UNION:
        raise ValueError("v79 prior commitment union changed")
    v77_rows = read_jsonl(v77_gold_path)
    v78_rows = read_jsonl(v78_gold_path)
    v77_ids = {str(row["id"]) for row in v77_rows}
    v78_ids = {str(row["id"]) for row in v78_rows}
    if len(v77_rows) != V77_CALIBRATION_CASES or len(v77_ids) != len(v77_rows):
        raise ValueError("v79 v77 calibration id source changed")
    if len(v78_rows) != V78_TRAINING_CASES or len(v78_ids) != len(v78_rows):
        raise ValueError("v79 v78 training id source changed")
    metadata: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                metadata.append(
                    {"id": row.get("id"), "answerable": row.get("answerable")}
                )
    selected_ids = select_stage_ids(
        metadata,
        excluded_source_commitments=prior,
        v77_calibration_source_ids=v77_ids,
        v78_training_source_ids=v78_ids,
        stage=stage,
    )
    selected_set = set(selected_ids)
    rows_by_id: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("id", ""))
            if row.get("answerable") is True and source_id in selected_set:
                rows_by_id[source_id] = row
    if set(rows_by_id) != selected_set:
        raise ValueError("v79 selected ids are missing from the source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for source_id in selected_ids:
        blind, gold = prepare_v77_case(rows_by_id[source_id])
        blind["source"] = "official_full_v1.0_train_answerable_untouched_v79"
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    selected_commitments = {source_id_commitment(value) for value in selected_ids}
    counts = Counter(int(row["hop_count"]) for row in gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "hop_count_distribution": {
            str(key): counts[key] for key in sorted(counts)
        },
        "prior_excluded_source_commitments": len(prior),
        "v77_calibration_excluded_ids": len(v77_ids),
        "v78_training_excluded_ids": len(v78_ids),
        "prior_source_overlap": len(selected_commitments & prior),
        "v77_calibration_overlap": len(selected_set & v77_ids),
        "v78_training_overlap": len(selected_set & v78_ids),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "selected_source_commitments_sha256": canonical_json_sha256(
            sorted(selected_commitments)
        ),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
        "gold_fields_visible_to_model_or_router": False,
    }


def write_selection_outputs(
    scored_rows: Iterable[dict[str, Any]],
    target_router: dict[str, Any],
    history_router: dict[str, Any],
    v76_router: dict[str, Any],
    calibration: dict[str, Any],
    v75_model: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        candidate_ids = [str(candidate["id"]) for candidate in row["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("v79 scored row has duplicate candidate ids")
        methods: dict[str, Any] = {}
        for method in METHODS:
            if method == CANDIDATE_METHOD:
                selected, metadata = select_target_method(
                    row, target_router, v75_model
                )
            elif method == V78_CANDIDATE_METHOD:
                selected, metadata = select_calibrated_method(
                    row, history_router, calibration, v75_model
                )
            else:
                selected, metadata = select_v76_method(
                    row, method, v76_router, v75_model
                )
            selected_ids = [str(candidate["id"]) for candidate in selected]
            methods[method] = {
                "selected_ids": selected_ids,
                "selected_tokens": sum(
                    int(candidate.get("token_count", 1)) for candidate in selected
                ),
                "route": metadata.get("route"),
                "predicted_soft_gain": metadata.get("predicted_soft_gain"),
                "predicted_anchor_gain": metadata.get("predicted_anchor_gain"),
                "predicted_winning_gain": metadata.get("predicted_winning_gain"),
                "adjusted_soft_gain": metadata.get("adjusted_soft_gain"),
                "adjusted_anchor_gain": metadata.get("adjusted_anchor_gain"),
                "adjusted_winning_gain": metadata.get("adjusted_winning_gain"),
            }
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_ids": candidate_ids,
                "methods": methods,
            }
        )
    write_jsonl(path, outputs)
    return outputs


def _paired_bootstrap(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, Any]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        control, dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, resamples, 250):
        batch = min(250, resamples - start)
        indices = generator.integers(
            0, len(difference), size=(batch, len(difference))
        )
        estimates.append(difference[indices].mean(axis=1))
    values = np.concatenate(estimates)
    return {
        "point": round(float(difference.mean()), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
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
    target_router_path: Path,
    history_router_path: Path,
    v76_router_path: Path,
    calibration_path: Path,
    v75_model_path: Path,
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    prior_overlap: int,
    v77_overlap: int,
    v78_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"unsupported v79 stage: {stage}")
    target_router = load_target_router_artifact(target_router_path)
    from research.frc_rag.hotpot_three_route_router import (
        load_router_artifact as load_history_router_artifact,
    )

    history_router = load_history_router_artifact(history_router_path)
    calibration = load_calibration(calibration_path, history_router)
    v76_router = json.loads(v76_router_path.read_text(encoding="utf-8"))
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if target_router["v75_router_model_payload_sha256"] != v75_artifact["model_sha256"]:
        raise ValueError("v79 target router and frozen v75 model differ")
    scored = read_jsonl(scored_path)
    selections = write_selection_outputs(
        scored,
        target_router,
        history_router,
        v76_router,
        calibration,
        v75_model,
        selection_path,
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v79 gold and selections do not align")
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
                or len(ids) > COMMON_RESOURCES["top_k"]
                or not set(ids) <= candidate_ids
                or int(values["selected_tokens"]) > COMMON_RESOURCES["token_budget"]
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
                "predicted_soft_gain": values["predicted_soft_gain"],
                "predicted_anchor_gain": values["predicted_anchor_gain"],
                "predicted_winning_gain": values["predicted_winning_gain"],
                "adjusted_soft_gain": values["adjusted_soft_gain"],
                "adjusted_anchor_gain": values["adjusted_anchor_gain"],
                "adjusted_winning_gain": values["adjusted_winning_gain"],
                "invalid": row_invalid,
            }
        cases.append(
            {
                "case_id": case_id,
                "hop_count": int(gold_row["hop_count"]),
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
            np.asarray(
                [case["methods"][method]["evidence_f1"] for case in cases]
            ),
            seed=seed + index,
        )
        for index, method in enumerate(CONTROL_METHODS)
    }
    per_hop: dict[str, Any] = {}
    for hop in sorted(HOP_QUOTAS):
        subset = [case for case in cases if case["hop_count"] == hop]
        best_control = max(
            CONTROL_METHODS,
            key=lambda method: (
                sum(
                    float(case["methods"][method]["evidence_f1"])
                    for case in subset
                ),
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
        per_hop[str(hop)] = {
            "cases": len(subset),
            "best_control": best_control,
            "candidate_evidence_macro_f1": round(candidate_f1, 6),
            "best_control_evidence_macro_f1": round(control_f1, 6),
            "delta": round(candidate_f1 - control_f1, 6),
        }
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_delta = comparisons[strongest]
    v78_delta = comparisons[V78_CANDIDATE_METHOD]
    route_counts = Counter(
        case["methods"][CANDIDATE_METHOD]["route"] for case in cases
    )
    route_fractions = {
        route: route_counts[route] / len(cases) for route in ROUTES
    }
    material_routes = sum(value >= 0.05 for value in route_fractions.values())
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(aggregates[strongest]["mean_selected_tokens"])
    )
    counts = Counter(case["hop_count"] for case in cases)
    no_overlap = (
        prior_overlap == 0
        and v77_overlap == 0
        and v78_overlap == 0
        and stage_overlap == 0
    )
    checks = {
        "exact_cases_equals_470": len(cases) == CASES_PER_STAGE,
        "exact_hop_quota": counts == Counter(HOP_QUOTAS),
        "prior_training_or_stage_overlap_equals_0": no_overlap,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_55": (
            float(candidate["evidence_macro_f1"]) >= 0.55
        ),
        "candidate_complete_evidence_recall_at_least_0_55": (
            float(candidate["complete_evidence_recall"]) >= 0.55
        ),
        "candidate_minus_strongest_control_f1_at_least_0_005": (
            float(strongest_delta["point"]) >= 0.005
        ),
        "candidate_minus_strongest_control_ci_low_above_0": (
            float(strongest_delta["ci_low"]) > 0.0
        ),
        "candidate_minus_frozen_v78_f1_at_least_0_005": (
            float(v78_delta["point"]) >= 0.005
        ),
        "candidate_minus_frozen_v78_ci_low_above_0": (
            float(v78_delta["ci_low"]) > 0.0
        ),
        "every_hop_delta_vs_best_control_at_least_minus_0_005": all(
            float(row["delta"]) >= -0.005 for row in per_hop.values()
        ),
        "at_least_two_routes_each_cover_at_least_0_05": material_routes >= 2,
        "largest_route_fraction_at_most_0_9": max(route_fractions.values()) <= 0.9,
        "mean_selected_tokens_within_1_05_of_strongest_control": (
            token_ratio <= 1.05
        ),
    }
    passed = all(checks.values())
    noninferiority_checks = {
        "candidate_minus_strongest_control_point_at_least_minus_0_005": (
            float(strongest_delta["point"]) >= -0.005
        ),
        "candidate_minus_strongest_control_ci_low_at_least_minus_0_01": (
            float(strongest_delta["ci_low"]) >= -0.01
        ),
        "candidate_minus_frozen_v78_point_at_least_0": (
            float(v78_delta["point"]) >= 0.0
        ),
        "every_hop_delta_vs_best_control_at_least_minus_0_02": all(
            float(row["delta"]) >= -0.02 for row in per_hop.values()
        ),
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    noninferiority_supported = all(noninferiority_checks.values())
    if stage == "development":
        status = (
            "MUSIQUE_V79_TARGET_THREE_ROUTE_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V79_TARGET_THREE_ROUTE_ADVANTAGE_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V79_TARGET_THREE_ROUTE_CASE_DISJOINT_ADVANTAGE_ESTABLISHED"
            if passed
            else "MUSIQUE_V79_TARGET_THREE_ROUTE_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "model_development_experiment_id": MODEL_DEVELOPMENT_EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "hop_count_distribution": {
                str(key): counts[key] for key in sorted(counts)
            },
            "prior_source_overlap": prior_overlap,
            "v77_calibration_overlap": v77_overlap,
            "v78_training_overlap": v78_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "target_router_sha256": sha256(target_router_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "gold_or_official_hop_used_by_runtime_router": False,
            "v78_used_for_target_domain_model_selection": True,
            "v79_target_cases_disjoint_from_v65_through_v78": True,
            "official_musique_leaderboard_result": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_registered_control": {
                "name": strongest,
                **aggregates[strongest],
            },
            "paired_f1_delta": comparisons,
            "router": {
                "route_counts": dict(sorted(route_counts.items())),
                "route_fractions": {
                    route: round(value, 6)
                    for route, value in route_fractions.items()
                },
                "material_route_count": material_routes,
            },
            "per_hop_delta_vs_best_control": per_hop,
            "support_checks": checks,
            "noninferiority_checks": noninferiority_checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "noninferiority_envelope_supported": noninferiority_supported,
                "confirmation_open_authorized": stage == "development" and passed,
                "reuse_v79_target_stage_for_model_threshold_gate_or_selection": False,
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
    (output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    candidate = analysis["methods"][CANDIDATE_METHOD]
    strongest = analysis["strongest_registered_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    lines = [
        f"# MuSiQue 目标域三路路由实验（v79 {report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强登记对照：`{strongest['name']}` / "
        f"`{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI "
        f"[`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- 路由分布：`{analysis['router']['route_counts']}`",
        f"- 非劣包络：`{analysis['outcome']['noninferiority_envelope_supported']}`",
        f"- Gate 2：`{analysis['outcome']['gate_2']}`",
        "",
        "## 严格门槛",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "v78 是显式目标域训练与模型选择集；v79 只检验同一数据集的新样本泛化。"
            "这不是零目标调参、独立数据集确认、官方榜单、真实 SetR、洪水专家或"
            "生产证据。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v79 protocol experiment id")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("name") != CANDIDATE_METHOD
        or candidate.get("model_development_experiment_id")
        != MODEL_DEVELOPMENT_EXPERIMENT_ID
        or candidate.get("v78_target_domain_training_cases")
        != V78_TRAINING_CASES
        or candidate.get("v79_target_training_or_tuning_cases") != 0
        or candidate.get("model_configurations") != MODEL_CONFIGURATIONS
        or candidate.get("route_policy_configurations")
        != ROUTE_POLICY_CONFIGURATIONS
    ):
        raise ValueError("v79 candidate contract drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v79 protocol controls drifted")
    if protocol.get("common_resources") != COMMON_RESOURCES:
        raise ValueError("v79 protocol resources drifted")
    if protocol.get("development_gates") != DEVELOPMENT_GATES:
        raise ValueError("v79 protocol gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v79 protocol noninferiority envelope drifted")
    metrics = protocol.get("metrics", {})
    if any(metrics.get(name) != value for name, value in METRIC_SEEDS.items()):
        raise ValueError("v79 protocol metric seeds drifted")
    for stage in STAGE_SALTS:
        contract = protocol.get("stages", {}).get(stage, {})
        if contract.get("cases") != CASES_PER_STAGE or contract.get(
            "hop_quota"
        ) != {str(key): value for key, value in HOP_QUOTAS.items()}:
            raise ValueError("v79 protocol stage sampling drifted")
    if (
        protocol["stages"]["confirmation"].get(
            "open_only_if_every_development_gate_passes"
        )
        is not True
    ):
        raise ValueError("v79 confirmation policy drifted")


__all__ = [
    "CASES_PER_STAGE",
    "CANDIDATE_METHOD",
    "CONTROL_METHODS",
    "DEVELOPMENT_GATES",
    "EXPERIMENT_ID",
    "HOP_QUOTAS",
    "METRIC_SEEDS",
    "METHODS",
    "NONINFERIORITY_ENVELOPE",
    "PRIOR_COMMITMENT_UNION",
    "TOTAL_EXCLUDED_TARGET_IDS",
    "V77_CALIBRATION_CASES",
    "V78_TRAINING_CASES",
    "evaluate_stage",
    "prepare_stage",
    "select_stage_ids",
    "validate_registered_protocol",
    "write_selection_outputs",
]
