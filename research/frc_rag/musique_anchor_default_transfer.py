"""Terminal prospective MuSiQue holdout for the frozen v80 anchor-default router."""

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
from research.frc_rag.musique_anchor_default_router import (
    CANDIDATE_METHOD,
    EXPERIMENT_ID as MODEL_DEVELOPMENT_EXPERIMENT_ID,
    MODEL_CONFIGURATIONS,
    ROUTES,
    ROUTE_POLICY_CONFIGURATIONS,
    load_router_artifact as load_anchor_router_artifact,
    select_method as select_anchor_method,
)
from research.frc_rag.musique_graph_router_transfer import (
    hop_count_from_id,
    load_source_commitments,
    prepare_case as prepare_v77_case,
    source_id_commitment,
)
from research.frc_rag.musique_mean_calibrated_three_route import (
    CANDIDATE_METHOD as V78_CANDIDATE_METHOD,
    load_calibration,
    select_calibrated_method,
)
from research.frc_rag.musique_target_three_route_router import (
    CANDIDATE_METHOD as V79_CANDIDATE_METHOD,
    load_router_artifact as load_v79_router_artifact,
    select_method as select_v79_method,
)
from research.frc_rag.musique_target_three_route_transfer import (
    CONTROL_METHODS as V79_BASE_CONTROL_METHODS,
    _aggregate,
    _paired_bootstrap,
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


EXPERIMENT_ID = "FRC-MUSIQUE-ANCHOR-DEFAULT-TERMINAL-HOLDOUT-V80"
SCHEMA_VERSION = "frc-musique-anchor-default-terminal-holdout-v80"
CONTROL_METHODS = (*V79_BASE_CONTROL_METHODS, V79_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
STAGE = "terminal_holdout"
STAGE_SALT = "FRC-MUSIQUE-V80-TERMINAL-HOLDOUT|"
HOP_QUOTAS = {2: 400, 3: 150, 4: 50}
CASES = sum(HOP_QUOTAS.values())
PRIOR_COMMITMENT_UNION = 16900
V77_CALIBRATION_CASES = 800
V78_TRAINING_CASES = 600
V79_TRAINING_CASES = 470
TOTAL_EXCLUDED_TARGET_IDS = (
    PRIOR_COMMITMENT_UNION
    + V77_CALIBRATION_CASES
    + V78_TRAINING_CASES
    + V79_TRAINING_CASES
)
METRIC_SEED = 20261161
HOLDOUT_GATES = {
    "exact_cases": CASES,
    "exact_hop_quota": {str(key): value for key, value in HOP_QUOTAS.items()},
    "prior_training_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.55,
    "candidate_complete_evidence_recall_at_least": 0.55,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_frozen_v79_f1_at_least": 0.005,
    "candidate_minus_frozen_v79_ci_low_above": 0.0,
    "every_hop_delta_vs_best_control_at_least": -0.005,
    "at_least_two_routes_each_cover_at_least_fraction": 0.05,
    "largest_route_fraction_at_most": 0.9,
    "mean_selected_tokens_not_above_strongest_control_by_more_than_fraction": 0.05,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_control_point_at_least": -0.005,
    "candidate_minus_strongest_control_ci_low_at_least": -0.01,
    "candidate_minus_frozen_v79_point_at_least": 0.0,
    "every_hop_delta_vs_best_control_at_least": -0.02,
    "invalid_selector_output_rate": 0.0,
}


def _order_key(source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{STAGE_SALT}{source_id}".encode("utf-8")).hexdigest()
    return digest, source_id


def select_holdout_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_source_commitments: set[str],
    v77_source_ids: set[str],
    v78_source_ids: set[str],
    v79_source_ids: set[str],
    hop_quotas: dict[int, int] | None = None,
) -> list[str]:
    quotas = HOP_QUOTAS if hop_quotas is None else hop_quotas
    if not quotas or any(
        hop not in {2, 3, 4} or quota <= 0 for hop, quota in quotas.items()
    ):
        raise ValueError("v80 hop quotas are invalid")
    ids = [
        str(row.get("id", "")).strip()
        for row in metadata_rows
        if row.get("answerable") is True
    ]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("v80 answerable metadata ids are empty or duplicated")
    expected = (
        (v77_source_ids, V77_CALIBRATION_CASES, "v77"),
        (v78_source_ids, V78_TRAINING_CASES, "v78"),
        (v79_source_ids, V79_TRAINING_CASES, "v79"),
    )
    for values, count, label in expected:
        if len(values) != count:
            raise ValueError(f"v80 requires exactly {count} {label} exclusion ids")
    if (
        v77_source_ids & v78_source_ids
        or v77_source_ids & v79_source_ids
        or v78_source_ids & v79_source_ids
    ):
        raise ValueError("v80 registered target-domain exclusion sets overlap")
    excluded_ids = v77_source_ids | v78_source_ids | v79_source_ids
    selected: list[str] = []
    for hop, quota in sorted(quotas.items()):
        eligible = [
            source_id
            for source_id in ids
            if hop_count_from_id(source_id) == hop
            and source_id_commitment(source_id) not in excluded_source_commitments
            and source_id not in excluded_ids
        ]
        eligible.sort(key=_order_key)
        if len(eligible) < quota:
            raise ValueError(f"v80 lacks {quota} untouched {hop}-hop cases")
        selected.extend(eligible[:quota])
    return selected


def prepare_holdout(
    source_path: Path,
    prior_exclusion_paths: Sequence[Path],
    v77_gold_path: Path,
    v78_gold_path: Path,
    v79_gold_path: Path,
    *,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    prior = load_source_commitments(prior_exclusion_paths)
    if len(prior) != PRIOR_COMMITMENT_UNION:
        raise ValueError("v80 prior commitment union changed")

    def ids(path: Path, expected: int, label: str) -> set[str]:
        rows = read_jsonl(path)
        values = {str(row["id"]) for row in rows}
        if len(rows) != expected or len(values) != expected:
            raise ValueError(f"v80 {label} exclusion source changed")
        return values

    v77_ids = ids(v77_gold_path, V77_CALIBRATION_CASES, "v77")
    v78_ids = ids(v78_gold_path, V78_TRAINING_CASES, "v78")
    v79_ids = ids(v79_gold_path, V79_TRAINING_CASES, "v79")
    metadata: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                metadata.append(
                    {"id": row.get("id"), "answerable": row.get("answerable")}
                )
    selected_ids = select_holdout_ids(
        metadata,
        excluded_source_commitments=prior,
        v77_source_ids=v77_ids,
        v78_source_ids=v78_ids,
        v79_source_ids=v79_ids,
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
        raise ValueError("v80 selected ids are missing from the source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for source_id in selected_ids:
        blind, gold = prepare_v77_case(rows_by_id[source_id])
        blind["source"] = "official_full_v1.0_train_answerable_terminal_v80"
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    selected_commitments = {source_id_commitment(value) for value in selected_ids}
    counts = Counter(int(row["hop_count"]) for row in gold_rows)
    return {
        "stage": STAGE,
        "selected_cases": len(selected_ids),
        "hop_count_distribution": {
            str(key): counts[key] for key in sorted(counts)
        },
        "prior_excluded_source_commitments": len(prior),
        "v77_excluded_ids": len(v77_ids),
        "v78_excluded_ids": len(v78_ids),
        "v79_excluded_ids": len(v79_ids),
        "prior_source_overlap": len(selected_commitments & prior),
        "v77_overlap": len(selected_set & v77_ids),
        "v78_overlap": len(selected_set & v78_ids),
        "v79_overlap": len(selected_set & v79_ids),
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
    anchor_router: dict[str, Any],
    v79_router: dict[str, Any],
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
            raise ValueError("v80 scored row has duplicate candidate ids")
        methods: dict[str, Any] = {}
        for method in METHODS:
            if method == CANDIDATE_METHOD:
                selected, metadata = select_anchor_method(
                    row, anchor_router, v75_model
                )
            elif method == V79_CANDIDATE_METHOD:
                selected, metadata = select_v79_method(row, v79_router, v75_model)
            elif method == V78_CANDIDATE_METHOD:
                selected, metadata = select_calibrated_method(
                    row, history_router, calibration, v75_model
                )
            else:
                selected, metadata = select_v76_method(
                    row, method, v76_router, v75_model
                )
            methods[method] = {
                "selected_ids": [str(candidate["id"]) for candidate in selected],
                "selected_tokens": sum(
                    int(candidate.get("token_count", 1)) for candidate in selected
                ),
                "route": metadata.get("route"),
                "predicted_cross_gain_vs_anchor": metadata.get(
                    "predicted_cross_gain_vs_anchor"
                ),
                "predicted_soft_gain_vs_anchor": metadata.get(
                    "predicted_soft_gain_vs_anchor"
                ),
                "predicted_winning_gain_vs_anchor": metadata.get(
                    "predicted_winning_gain_vs_anchor"
                ),
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


def evaluate_holdout(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    anchor_router_path: Path,
    v79_router_path: Path,
    history_router_path: Path,
    v76_router_path: Path,
    calibration_path: Path,
    v75_model_path: Path,
    *,
    output_dir: Path,
    overlap_counts: dict[str, int],
) -> dict[str, Any]:
    anchor_router = load_anchor_router_artifact(anchor_router_path)
    v79_router = load_v79_router_artifact(v79_router_path)
    from research.frc_rag.hotpot_three_route_router import (
        load_router_artifact as load_history_router_artifact,
    )

    history_router = load_history_router_artifact(history_router_path)
    calibration = load_calibration(calibration_path, history_router)
    v76_router = json.loads(v76_router_path.read_text(encoding="utf-8"))
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if anchor_router["v75_router_model_payload_sha256"] != v75_artifact[
        "model_sha256"
    ]:
        raise ValueError("v80 router and frozen v75 model differ")
    selections = write_selection_outputs(
        read_jsonl(scored_path),
        anchor_router,
        v79_router,
        history_router,
        v76_router,
        calibration,
        v75_model,
        selection_path,
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v80 gold and selections do not align")
    cases: list[dict[str, Any]] = []
    invalid = 0
    metadata_keys = (
        "predicted_cross_gain_vs_anchor",
        "predicted_soft_gain_vs_anchor",
        "predicted_winning_gain_vs_anchor",
        "predicted_soft_gain",
        "predicted_anchor_gain",
        "predicted_winning_gain",
        "adjusted_soft_gain",
        "adjusted_anchor_gain",
        "adjusted_winning_gain",
    )
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
                **{key: values[key] for key in metadata_keys},
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
            seed=METRIC_SEED + index,
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
    v79_delta = comparisons[V79_CANDIDATE_METHOD]
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
    no_overlap = all(int(value) == 0 for value in overlap_counts.values())
    checks = {
        "exact_cases_equals_600": len(cases) == CASES,
        "exact_hop_quota": counts == Counter(HOP_QUOTAS),
        "prior_training_overlap_equals_0": no_overlap,
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
        "candidate_minus_frozen_v79_f1_at_least_0_005": (
            float(v79_delta["point"]) >= 0.005
        ),
        "candidate_minus_frozen_v79_ci_low_above_0": (
            float(v79_delta["ci_low"]) > 0.0
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
        "candidate_minus_frozen_v79_point_at_least_0": (
            float(v79_delta["point"]) >= 0.0
        ),
        "every_hop_delta_vs_best_control_at_least_minus_0_02": all(
            float(row["delta"]) >= -0.02 for row in per_hop.values()
        ),
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    noninferiority_supported = all(noninferiority_checks.values())
    status = (
        "MUSIQUE_V80_ANCHOR_DEFAULT_TERMINAL_HOLDOUT_ADVANTAGE_ESTABLISHED"
        if passed
        else "MUSIQUE_V80_ANCHOR_DEFAULT_TERMINAL_HOLDOUT_ADVANTAGE_NOT_ESTABLISHED"
    )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "model_development_experiment_id": MODEL_DEVELOPMENT_EXPERIMENT_ID,
            "stage": STAGE,
            "cases": len(cases),
            "hop_count_distribution": {
                str(key): counts[key] for key in sorted(counts)
            },
            "overlap_counts": overlap_counts,
            "invalid_selector_output_count": invalid,
            "anchor_router_sha256": sha256(anchor_router_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "gold_or_official_hop_used_by_runtime_router": False,
            "v78_v79_used_for_target_domain_model_selection": True,
            "terminal_holdout_due_remaining_four_hop_capacity": True,
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
                "holdout_gate_passed": passed,
                "noninferiority_envelope_supported": noninferiority_supported,
                "further_same_source_confirmation_authorized": False,
                "reuse_v80_holdout_for_model_threshold_gate_or_selection": False,
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
        "# MuSiQue 锚点默认三路路由终末留出实验（v80）",
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
            "v78+v79 是显式目标域训练与模型选择集；v80 是同一数据集剩余容量"
            "上的单次终末留出。它不是零目标调参、独立训练数据、官方榜单、真实 "
            "SetR、洪水专家或生产证据。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v80 protocol experiment id")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("name") != CANDIDATE_METHOD
        or candidate.get("model_development_experiment_id")
        != MODEL_DEVELOPMENT_EXPERIMENT_ID
        or candidate.get("target_domain_training_cases") != 1070
        or candidate.get("v80_target_training_or_tuning_cases") != 0
        or candidate.get("model_configurations") != MODEL_CONFIGURATIONS
        or candidate.get("route_policy_configurations")
        != ROUTE_POLICY_CONFIGURATIONS
    ):
        raise ValueError("v80 candidate contract drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v80 protocol controls drifted")
    if protocol.get("common_resources") != COMMON_RESOURCES:
        raise ValueError("v80 protocol resources drifted")
    stage = protocol.get("stage", {})
    if stage.get("name") != STAGE or stage.get("cases") != CASES or stage.get(
        "hop_quota"
    ) != {str(key): value for key, value in HOP_QUOTAS.items()}:
        raise ValueError("v80 protocol stage sampling drifted")
    if protocol.get("metric_seed") != METRIC_SEED:
        raise ValueError("v80 protocol metric seed drifted")
    if protocol.get("holdout_gates") != HOLDOUT_GATES:
        raise ValueError("v80 protocol gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v80 protocol noninferiority envelope drifted")
    if stage.get("further_same_source_confirmation_authorized") is not False:
        raise ValueError("v80 terminal holdout policy drifted")


__all__ = [
    "CASES",
    "CANDIDATE_METHOD",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "HOLDOUT_GATES",
    "HOP_QUOTAS",
    "METHODS",
    "NONINFERIORITY_ENVELOPE",
    "PRIOR_COMMITMENT_UNION",
    "STAGE",
    "TOTAL_EXCLUDED_TARGET_IDS",
    "V77_CALIBRATION_CASES",
    "V78_TRAINING_CASES",
    "V79_TRAINING_CASES",
    "evaluate_holdout",
    "prepare_holdout",
    "select_holdout_ids",
    "validate_registered_protocol",
    "write_selection_outputs",
]
