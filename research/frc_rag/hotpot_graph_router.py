"""History-trained graph router for held-out HotpotQA evidence selection (v76)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    CANDIDATE_METHOD as V75_METHOD,
    LEXICAL_METHOD,
    SOFT_METHOD,
    load_model_artifact as load_v75_model_artifact,
    route_probability,
    select_method as select_v75_method,
)
from research.frc_rag.twowiki_support_path_closure import (
    build_title_graph,
    canonical_json_sha256,
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-HOTPOT-GRAPH-SOFT-OVERRIDE-ROUTER-V76"
SCHEMA_VERSION = "frc-hotpot-graph-soft-override-router-v76"
DEVELOPMENT_SALT = "FRC-HOTPOT-V76-DEVELOPMENT|"
CONFIRMATION_SALT = "FRC-HOTPOT-V76-CONFIRMATION|"
HISTORY_FOLD_SALT = "FRC-HOTPOT-V76-TREE-FOLD|"
QUESTION_TYPES = ("bridge", "comparison")
CASES_PER_STAGE = 800
QUOTA_PER_TYPE = 400
FOLD_COUNT = 5
GBR_ESTIMATORS = 25
GBR_MAX_DEPTH = 2
GBR_MIN_SAMPLES_LEAF = 40
GBR_LEARNING_RATE = 0.1
GBR_LOSS = "huber"
GBR_RANDOM_STATE = 7
OVERRIDE_THRESHOLD = 0.0
CANDIDATE_METHOD = "graph_gbr_soft_override_router_v76"
ZERO_SHOT_METHOD = "zero_shot_v75_question_router_control"
CONTROL_METHODS = (
    "cross_encoder_topk",
    "frc_select",
    SOFT_METHOD,
    ANCHOR_METHOD,
    LEXICAL_METHOD,
    ZERO_SHOT_METHOD,
)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
COMMON_RESOURCES = {
    "top_k": 5,
    "token_budget": 1500,
    "embedding_model": (
        "BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    ),
    "reranker_model": (
        "BAAI/bge-reranker-large@55611d7bca2a7133960a6d3b71e083071bbfc312"
    ),
    "score_calibration": "per-case min-max",
    "role_relevance_mix": 0.15,
    "rrf_k": 60,
}
METRIC_SEEDS = {"development_seed": 20261031, "confirmation_seed": 20261032}
DEVELOPMENT_GATES = {
    "exact_cases": 800,
    "exact_question_type_balance": True,
    "history_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.55,
    "candidate_complete_evidence_recall_at_least": 0.55,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_cross_encoder_f1_at_least": 0.005,
    "candidate_minus_cross_encoder_ci_low_above": 0.0,
    "every_question_type_delta_vs_best_control_at_least": -0.005,
    "soft_override_rate_between_inclusive": [0.15, 0.55],
    "mean_selected_tokens_not_above_strongest_control_by_more_than_fraction": 0.05,
}


def _nested(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple)):
        converted = value.tolist()
        if converted is not value:
            return converted
    return value


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)))


def blind_history_row(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"answer", "gold_evidence_ids"}
    blind = {key: value for key, value in row.items() if key not in forbidden}
    blind["candidates"] = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"gold", "gold_roles"}
        }
        for candidate in row["candidates"]
    ]
    return blind


def _selection_ids(
    row: dict[str, Any], method: str, v75_model: dict[str, Any]
) -> list[str]:
    selected, _ = select_v75_method(row, method, v75_model)
    return [str(candidate["id"]) for candidate in selected]


def graph_feature_names() -> tuple[str, ...]:
    names = [
        "candidate_count_scaled",
        "unique_source_fraction",
        "linked_node_fraction",
        "mean_degree_fraction",
        "max_degree_fraction",
        "component_fraction",
        "largest_component_fraction",
        "v75_comparison_probability",
        "question_title_mention_fraction",
    ]
    for score_name in ("cross_encoder", "dense", "bm25", "hybrid"):
        names.extend(
            (
                f"{score_name}_top1",
                f"{score_name}_top1_top2_gap",
                f"{score_name}_top5_mean",
                f"{score_name}_top5_std",
                f"{score_name}_top5_floor",
            )
        )
    for selection_name in ("cross", "soft", "anchor"):
        names.extend(
            (
                f"{selection_name}_cross_overlap_fraction",
                f"{selection_name}_cross_added_fraction",
                f"{selection_name}_cross_score_mean",
                f"{selection_name}_cross_score_min",
                f"{selection_name}_cross_score_max",
                f"{selection_name}_internal_edge_fraction",
                f"{selection_name}_linked_node_fraction",
                f"{selection_name}_unique_source_fraction",
                f"{selection_name}_token_budget_fraction",
                f"{selection_name}_added_link_to_cross_fraction",
                f"{selection_name}_added_mean_cross_rank_fraction",
                f"{selection_name}_added_max_cross_rank_fraction",
            )
        )
    names.extend(
        (
            "pair_soft_cross_overlap_fraction",
            "pair_soft_minus_cross_fraction",
            "pair_anchor_cross_overlap_fraction",
            "pair_anchor_minus_cross_fraction",
            "pair_soft_anchor_overlap_fraction",
            "pair_soft_minus_anchor_fraction",
        )
    )
    return tuple(names)


FEATURE_NAMES = graph_feature_names()


def graph_features(row: dict[str, Any], v75_model: dict[str, Any]) -> np.ndarray:
    candidates = list(row.get("candidates", []))
    if not candidates:
        raise ValueError("v76 graph router requires candidates")
    candidate_count = len(candidates)
    by_id = {str(candidate["id"]): candidate for candidate in candidates}
    if len(by_id) != candidate_count:
        raise ValueError("v76 graph router candidate ids are duplicated")
    graph = build_title_graph(candidates)
    degrees = {candidate_id: len(graph[candidate_id]) for candidate_id in by_id}
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["scores"]["cross_encoder"]),
            str(candidate["id"]),
        ),
    )
    rank_by_id = {str(candidate["id"]): rank for rank, candidate in enumerate(ranked)}
    cross_ids = [str(candidate["id"]) for candidate in ranked[:5]]
    soft_ids = _selection_ids(row, SOFT_METHOD, v75_model)
    anchor_ids = _selection_ids(row, ANCHOR_METHOD, v75_model)
    selections = (cross_ids, soft_ids, anchor_ids)

    seen: set[str] = set()
    component_sizes: list[int] = []
    for candidate_id in by_id:
        if candidate_id in seen:
            continue
        stack = [candidate_id]
        seen.add(candidate_id)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    divisor = max(candidate_count - 1, 1)
    degree_values = np.asarray(list(degrees.values()), dtype=np.float64)
    question = str(row["question"])
    lowered_question = question.lower()
    values: list[float] = [
        candidate_count / 60.0,
        len({str(candidate.get("source", "")) for candidate in candidates})
        / candidate_count,
        float(np.mean(degree_values > 0)),
        float(degree_values.mean()) / divisor,
        float(degree_values.max()) / divisor,
        len(component_sizes) / candidate_count,
        max(component_sizes) / candidate_count,
        route_probability(question, v75_model),
        sum(
            str(candidate.get("source", "")).lower() in lowered_question
            for candidate in candidates
        )
        / candidate_count,
    ]
    for score_name in ("cross_encoder", "dense", "bm25", "hybrid"):
        scores = sorted(
            (float(candidate["scores"][score_name]) for candidate in candidates),
            reverse=True,
        )
        top = scores[:5]
        if len(top) < 5:
            top.extend([0.0] * (5 - len(top)))
        values.extend(
            (
                top[0],
                top[0] - top[1],
                float(np.mean(top)),
                float(np.std(top)),
                top[-1],
            )
        )
    cross_set = set(cross_ids)
    for selected_ids in selections:
        selected_set = set(selected_ids)
        selected_scores = np.asarray(
            [
                float(by_id[candidate_id]["scores"]["cross_encoder"])
                for candidate_id in selected_ids
            ],
            dtype=np.float64,
        )
        added = selected_set - cross_set
        internal_edges = (
            sum(
                neighbor in selected_set
                for candidate_id in selected_set
                for neighbor in graph[candidate_id]
            )
            / 2.0
        )
        added_ranks = [rank_by_id[candidate_id] for candidate_id in added]
        values.extend(
            (
                len(selected_set & cross_set) / 5.0,
                len(added) / 5.0,
                float(selected_scores.mean()),
                float(selected_scores.min()),
                float(selected_scores.max()),
                internal_edges / 10.0,
                sum(degrees[candidate_id] > 0 for candidate_id in selected_set) / 5.0,
                len(
                    {
                        str(by_id[candidate_id].get("source", ""))
                        for candidate_id in selected_set
                    }
                )
                / 5.0,
                sum(
                    int(by_id[candidate_id].get("token_count", 1))
                    for candidate_id in selected_set
                )
                / 1500.0,
                sum(bool(graph[candidate_id] & cross_set) for candidate_id in added)
                / 5.0,
                (float(np.mean(added_ranks)) / candidate_count if added_ranks else 0.0),
                max(added_ranks) / candidate_count if added_ranks else 0.0,
            )
        )
    for left, right in ((1, 0), (2, 0), (1, 2)):
        left_set = set(selections[left])
        right_set = set(selections[right])
        values.extend(
            (
                len(left_set & right_set) / 5.0,
                len(left_set - right_set) / 5.0,
            )
        )
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("v76 graph features are invalid")
    return result


def _export_tree(tree: Any) -> dict[str, Any]:
    structure = tree.tree_
    return {
        "children_left": [int(value) for value in structure.children_left],
        "children_right": [int(value) for value in structure.children_right],
        "features": [int(value) for value in structure.feature],
        "thresholds": [round(float(value), 15) for value in structure.threshold],
        "values": [round(float(value[0][0]), 15) for value in structure.value],
    }


def _tree_predict(tree: dict[str, Any], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    predictions = np.empty(len(matrix), dtype=np.float64)
    for row_index, row in enumerate(matrix):
        node = 0
        while int(tree["children_left"][node]) != -1:
            feature = int(tree["features"][node])
            if row[feature] <= float(tree["thresholds"][node]):
                node = int(tree["children_left"][node])
            else:
                node = int(tree["children_right"][node])
        predictions[row_index] = float(tree["values"][node])
    return predictions


def export_gbr(model: Any, feature_count: int) -> dict[str, Any]:
    init_prediction = float(model.init_.predict(np.zeros((1, feature_count)))[0])
    payload = {
        "model_type": "gradient_boosting_regressor_export",
        "sklearn_version": "1.7.2",
        "feature_count": feature_count,
        "estimators": GBR_ESTIMATORS,
        "max_depth": GBR_MAX_DEPTH,
        "min_samples_leaf": GBR_MIN_SAMPLES_LEAF,
        "learning_rate": GBR_LEARNING_RATE,
        "loss": GBR_LOSS,
        "random_state": GBR_RANDOM_STATE,
        "init_prediction": round(init_prediction, 15),
        "trees": [_export_tree(estimator[0]) for estimator in model.estimators_],
    }
    if len(payload["trees"]) != GBR_ESTIMATORS:
        raise ValueError("v76 exported model has the wrong tree count")
    return payload


def predict_gbr(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != int(payload["feature_count"]):
        raise ValueError("v76 model feature dimension mismatch")
    predictions = np.full(
        len(matrix), float(payload["init_prediction"]), dtype=np.float64
    )
    for tree in payload["trees"]:
        predictions += float(payload["learning_rate"]) * _tree_predict(tree, matrix)
    return predictions


def _fit_gbr(features: np.ndarray, targets: np.ndarray) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=GBR_ESTIMATORS,
        learning_rate=GBR_LEARNING_RATE,
        max_depth=GBR_MAX_DEPTH,
        min_samples_leaf=GBR_MIN_SAMPLES_LEAF,
        loss=GBR_LOSS,
        random_state=GBR_RANDOM_STATE,
    )
    model.fit(features, targets)
    return model


def _history_fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{HISTORY_FOLD_SALT}{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def _paired_bootstrap(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, Any]:
    difference = np.asarray(candidate) - np.asarray(control)
    generator = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, resamples, 250):
        batch = min(250, resamples - start)
        indices = generator.integers(0, len(difference), size=(batch, len(difference)))
        estimates.append(difference[indices].mean(axis=1))
    values = np.concatenate(estimates)
    return {
        "point": round(float(difference.mean()), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


def _selected_metrics(
    row: dict[str, Any],
    gold_ids: Sequence[str],
    method: str,
    v75_model: dict[str, Any],
) -> dict[str, float]:
    selected, _ = select_v75_method(row, method, v75_model)
    selected_ids = [str(candidate["id"]) for candidate in selected]
    metrics = evidence_metrics(gold_ids, selected_ids)
    return {
        "evidence_f1": float(metrics["evidence_f1"]),
        "evidence_recall": float(metrics["evidence_recall"]),
        "complete_evidence": float(set(gold_ids) <= set(selected_ids)),
        "selected_tokens": float(
            sum(int(candidate.get("token_count", 1)) for candidate in selected)
        ),
    }


def develop_model(history_path: Path, v75_model_path: Path) -> dict[str, Any]:
    history = read_jsonl(history_path)
    if len(history) != 1000 or len({str(row["id"]) for row in history}) != 1000:
        raise ValueError("v76 history requires exactly 1000 unique HotpotQA cases")
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    case_ids: list[str] = []
    question_types: list[str] = []
    features: list[np.ndarray] = []
    metrics_by_method: dict[str, list[dict[str, float]]] = {
        method: []
        for method in (
            "cross_encoder_topk",
            SOFT_METHOD,
            ANCHOR_METHOD,
            LEXICAL_METHOD,
            V75_METHOD,
        )
    }
    for original in history:
        if str(original.get("dataset", "")).lower() != "hotpotqa":
            raise ValueError("v76 history contains a non-HotpotQA row")
        row = blind_history_row(original)
        case_ids.append(str(original["id"]))
        question_types.append(str(original["question_type"]))
        features.append(graph_features(row, v75_model))
        for method in metrics_by_method:
            metrics_by_method[method].append(
                _selected_metrics(
                    row,
                    original["gold_evidence_ids"],
                    method,
                    v75_model,
                )
            )
    feature_matrix = np.vstack(features)
    cross_f1 = np.asarray(
        [row["evidence_f1"] for row in metrics_by_method["cross_encoder_topk"]]
    )
    soft_f1 = np.asarray([row["evidence_f1"] for row in metrics_by_method[SOFT_METHOD]])
    targets = soft_f1 - cross_f1
    folds = np.asarray([_history_fold(case_id) for case_id in case_ids])
    predictions = np.zeros(len(history), dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        training = folds != fold
        held_out = ~training
        model = _fit_gbr(feature_matrix[training], targets[training])
        export = export_gbr(model, feature_matrix.shape[1])
        exported_predictions = predict_gbr(export, feature_matrix[held_out])
        sklearn_predictions = model.predict(feature_matrix[held_out])
        if not np.allclose(exported_predictions, sklearn_predictions, atol=1e-12):
            raise ValueError("v76 exported fold model differs from sklearn")
        predictions[held_out] = exported_predictions
        fold_models.append(
            {
                "fold": fold,
                "training_cases": int(training.sum()),
                "held_out_cases": int(held_out.sum()),
                "model_sha256": canonical_json_sha256(export),
            }
        )
    override = predictions > OVERRIDE_THRESHOLD
    routed_f1 = np.where(override, soft_f1, cross_f1)
    full_model = _fit_gbr(feature_matrix, targets)
    model_payload = export_gbr(full_model, feature_matrix.shape[1])
    if not np.allclose(
        predict_gbr(model_payload, feature_matrix),
        full_model.predict(feature_matrix),
        atol=1e-12,
    ):
        raise ValueError("v76 exported full model differs from sklearn")

    def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
        return {
            "evidence_macro_f1": round(
                sum(row["evidence_f1"] for row in rows) / len(rows), 6
            ),
            "evidence_macro_recall": round(
                sum(row["evidence_recall"] for row in rows) / len(rows), 6
            ),
            "complete_evidence_recall": round(
                sum(row["complete_evidence"] for row in rows) / len(rows), 6
            ),
            "mean_selected_tokens": round(
                sum(row["selected_tokens"] for row in rows) / len(rows), 6
            ),
        }

    routed_rows = [
        metrics_by_method[SOFT_METHOD if use_soft else "cross_encoder_topk"][index]
        for index, use_soft in enumerate(override)
    ]
    per_type: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        mask = np.asarray([value == question_type for value in question_types])
        per_type[question_type] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "cross_encoder_evidence_macro_f1": round(float(cross_f1[mask].mean()), 6),
            "delta": round(float((routed_f1[mask] - cross_f1[mask]).mean()), 6),
            "override_rate": round(float(override[mask].mean()), 6),
        }
    return {
        "schema_version": "frc-hotpot-graph-router-model-development-v76",
        "experiment_id": EXPERIMENT_ID,
        "history": {
            "cases": len(history),
            "unique_ids": len(set(case_ids)),
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "question_type_counts": dict(sorted(Counter(question_types).items())),
            "source_sha256": sha256(history_path),
            "eligible_for_v76_target_stages": False,
        },
        "exploration_disclosure": {
            "zero_shot_v75_transfer_diagnostic_run": True,
            "single_soft_ridge_configurations": 525,
            "three_way_ridge_configurations": 200,
            "tree_configurations": 112,
            "gradient_boosting_configurations": 252,
            "total_historical_configurations": 1089,
            "best_single_soft_ridge_oof_f1": 0.557323,
            "best_three_way_ridge_oof_f1": 0.558123,
            "selected_gradient_boosting_oof_f1": round(float(routed_f1.mean()), 6),
            "all_exploration_used_only_permanently_excluded_history": True,
        },
        "selected_configuration": {
            "runtime_target": "soft_title_link_f1_minus_cross_encoder_f1",
            "features": "71 gold-free graph, score, selection-overlap and frozen v75 question-probability features",
            "feature_names": list(FEATURE_NAMES),
            "estimators": GBR_ESTIMATORS,
            "max_depth": GBR_MAX_DEPTH,
            "min_samples_leaf": GBR_MIN_SAMPLES_LEAF,
            "learning_rate": GBR_LEARNING_RATE,
            "loss": GBR_LOSS,
            "random_state": GBR_RANDOM_STATE,
            "override_threshold": OVERRIDE_THRESHOLD,
            "default_route": "cross_encoder_topk",
            "positive_route": SOFT_METHOD,
            "official_type_answer_or_gold_used_at_runtime": False,
        },
        "crossfit": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "override_rate": round(float(override.mean()), 6),
            "candidate": aggregate(routed_rows),
            "controls": {
                method: aggregate(rows) for method, rows in metrics_by_method.items()
            },
            "candidate_minus_cross_encoder": _paired_bootstrap(
                routed_f1, cross_f1, seed=20261021
            ),
            "per_question_type": per_type,
        },
        "model": model_payload,
        "model_sha256": canonical_json_sha256(model_payload),
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "prospective_boundary": {
            "all_1000_history_ids_excluded_from_v76_target_stages": True,
            "v76_target_rows_ids_scores_or_metrics_seen": False,
        },
    }


def load_router_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v76 router artifact experiment id")
    payload = artifact["model"]
    if canonical_json_sha256(payload) != artifact.get("model_sha256"):
        raise ValueError("v76 router model payload hash mismatch")
    if (
        payload.get("feature_count") != len(FEATURE_NAMES)
        or payload.get("estimators") != GBR_ESTIMATORS
        or payload.get("max_depth") != GBR_MAX_DEPTH
        or payload.get("min_samples_leaf") != GBR_MIN_SAMPLES_LEAF
        or payload.get("learning_rate") != GBR_LEARNING_RATE
        or payload.get("loss") != GBR_LOSS
        or payload.get("random_state") != GBR_RANDOM_STATE
        or len(payload.get("trees", [])) != GBR_ESTIMATORS
    ):
        raise ValueError("v76 router model differs from frozen configuration")
    selected = artifact.get("selected_configuration", {})
    if (
        selected.get("feature_names") != list(FEATURE_NAMES)
        or len(set(selected.get("feature_names", []))) != len(FEATURE_NAMES)
        or selected.get("official_type_answer_or_gold_used_at_runtime") is not False
    ):
        raise ValueError("v76 router feature contract drifted")
    if not validate_finite(payload):
        raise ValueError("v76 router model is non-finite")
    return artifact


def _order_key(salt: str, case_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{salt}{case_id}".encode()).hexdigest(), case_id


def load_history_ids(path: Path) -> set[str]:
    rows = read_jsonl(path)
    ids = {str(row["id"]) for row in rows}
    if len(rows) != 1000 or len(ids) != 1000:
        raise ValueError("v76 history exclusion requires 1000 unique ids")
    return ids


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    history_ids: set[str],
    stage: str,
    quota_per_type: int = QUOTA_PER_TYPE,
) -> list[str]:
    if stage not in {"development", "confirmation"}:
        raise ValueError(f"unsupported v76 stage: {stage}")
    rows = [
        {"id": str(row.get("id") or ""), "type": str(row.get("type") or "")}
        for row in metadata_rows
    ]
    if any(not row["id"] or row["type"] not in QUESTION_TYPES for row in rows):
        raise ValueError("v76 Hotpot metadata is invalid")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("v76 Hotpot metadata contains duplicate ids")

    def choose(salt: str, excluded: set[str]) -> list[str]:
        selected: list[str] = []
        for question_type in QUESTION_TYPES:
            eligible = [
                row["id"]
                for row in rows
                if row["type"] == question_type and row["id"] not in excluded
            ]
            eligible.sort(key=lambda case_id: _order_key(salt, case_id))
            if len(eligible) < quota_per_type:
                raise ValueError(
                    f"v76 lacks {quota_per_type} eligible {question_type} cases"
                )
            selected.extend(eligible[:quota_per_type])
        return selected

    development = choose(DEVELOPMENT_SALT, history_ids)
    if stage == "development":
        return development
    return choose(CONFIRMATION_SALT, history_ids | set(development))


def prepare_case(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(row.get("id") or "").strip()
    question_type = str(row.get("type") or "").strip()
    if not case_id or question_type not in QUESTION_TYPES:
        raise ValueError("invalid v76 Hotpot case metadata")
    context = _nested(row.get("context")) or {}
    supporting = _nested(row.get("supporting_facts")) or {}
    titles = list(_nested(context.get("title")) or [])
    paragraphs = list(_nested(context.get("sentences")) or [])
    gold_pairs = set(
        zip(
            [str(value) for value in (_nested(supporting.get("title")) or [])],
            [int(value) for value in (_nested(supporting.get("sent_id")) or [])],
            strict=True,
        )
    )
    if len(gold_pairs) < 2 or len(titles) != len(paragraphs):
        raise ValueError(f"v76 Hotpot case {case_id} has invalid evidence structure")
    candidates: list[dict[str, Any]] = []
    gold_ids: list[str] = []
    matched: set[tuple[str, int]] = set()
    candidate_index = 0
    for paragraph_index, (title, sentences) in enumerate(
        zip(titles, paragraphs, strict=True)
    ):
        title = str(title)
        for sentence_index, sentence in enumerate(_nested(sentences) or []):
            candidate_id = f"hotpot-{candidate_index}::{title}::{sentence_index}"
            candidate_index += 1
            pair = (title, sentence_index)
            if pair in gold_pairs:
                matched.add(pair)
                gold_ids.append(candidate_id)
            text = f"{title}. {str(sentence)}"
            candidates.append(
                {
                    "id": candidate_id,
                    "text": text,
                    "source": title,
                    "metadata": {
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "sentence_index": sentence_index,
                    },
                    "token_count": _token_count(text),
                }
            )
    if matched != gold_pairs or not candidates:
        raise ValueError(f"v76 Hotpot case {case_id} is missing supporting evidence")
    blind = {
        "dataset": "HotpotQA",
        "source": "official_distractor_validation_case_disjoint_v76",
        "id": case_id,
        "question": str(row.get("question") or ""),
        "question_type": question_type,
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": candidates,
    }
    gold = {
        "id": case_id,
        "question_type": question_type,
        "gold_evidence_ids": sorted(gold_ids),
    }
    return blind, gold


def prepare_stage(
    source_path: Path,
    history_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    history_ids = load_history_ids(history_path)
    metadata = pq.read_table(source_path, columns=["id", "type"]).to_pylist()
    selected_ids = select_stage_ids(metadata, history_ids=history_ids, stage=stage)
    selected_set = set(selected_ids)
    frame = pd.read_parquet(source_path)
    rows_by_id = {
        str(row["id"]): row
        for row in frame.to_dict(orient="records")
        if str(row["id"]) in selected_set
    }
    if set(rows_by_id) != selected_set:
        raise ValueError("v76 selected ids are missing from source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[case_id])
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    counts = Counter(row["question_type"] for row in gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "question_type_counts": dict(sorted(counts.items())),
        "history_excluded_ids": len(history_ids),
        "history_overlap": len(selected_set & history_ids),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
    }


def select_method(
    row: dict[str, Any],
    method: str,
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
    *,
    k: int = 5,
    token_budget: int = 1500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if method in {"cross_encoder_topk", "frc_select", SOFT_METHOD, ANCHOR_METHOD}:
        selected, _ = select_v75_method(
            row, method, v75_model, k=k, token_budget=token_budget
        )
        return selected, {"route": method, "predicted_soft_gain": None}
    if method == LEXICAL_METHOD:
        selected, routing = select_v75_method(
            row, LEXICAL_METHOD, v75_model, k=k, token_budget=token_budget
        )
        return selected, {
            "route": routing["route"],
            "predicted_soft_gain": None,
        }
    if method == ZERO_SHOT_METHOD:
        selected, routing = select_v75_method(
            row, V75_METHOD, v75_model, k=k, token_budget=token_budget
        )
        return selected, {
            "route": routing["route"],
            "predicted_soft_gain": None,
        }
    if method == CANDIDATE_METHOD:
        predicted_gain = float(
            predict_gbr(router_artifact["model"], graph_features(row, v75_model))[0]
        )
        route = (
            SOFT_METHOD if predicted_gain > OVERRIDE_THRESHOLD else "cross_encoder_topk"
        )
        selected, _ = select_v75_method(
            row, route, v75_model, k=k, token_budget=token_budget
        )
        return selected, {
            "route": route,
            "predicted_soft_gain": predicted_gain,
        }
    raise ValueError(f"unsupported v76 method: {method}")


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    forbidden = {"answer", "gold_evidence_ids", "gold", "gold_roles"}
    if any(
        forbidden & set(row)
        or any(forbidden & set(candidate) for candidate in row.get("candidates", []))
        for row in scored_rows
    ):
        raise ValueError("v76 model rows contain forbidden gold fields")
    case_ids = [str(row.get("id") or "") for row in scored_rows]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("v76 model rows require unique non-empty case ids")
    if any(not str(row.get("question") or "").strip() for row in scored_rows):
        raise ValueError("v76 model rows require question text")
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        methods: dict[str, Any] = {}
        for method in METHODS:
            selected, routing = select_method(row, method, router_artifact, v75_model)
            methods[method] = {
                "selected_ids": [str(candidate["id"]) for candidate in selected],
                "selected_tokens": sum(
                    int(candidate.get("token_count", 1)) for candidate in selected
                ),
                **routing,
            }
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_ids": sorted(
                    str(candidate["id"]) for candidate in row["candidates"]
                ),
                "methods": methods,
            }
        )
    write_jsonl(path, outputs)
    return outputs


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    router_artifact_path: Path,
    v75_model_path: Path,
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    history_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    router_artifact = load_router_artifact(router_artifact_path)
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if (
        router_artifact["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
    ):
        raise ValueError("v76 frozen v75 feature model changed")
    scored = read_jsonl(scored_path)
    selections = write_selection_outputs(
        scored, router_artifact, v75_model, selection_path
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v76 gold and selections do not align")
    cases: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold):
        gold_row = gold[case_id]
        selection = selected[case_id]
        candidate_ids = set(selection["candidate_ids"])
        methods: dict[str, Any] = {}
        for method in METHODS:
            values = selection["methods"][method]
            ids = list(values["selected_ids"])
            row_invalid = (
                len(ids) != len(set(ids))
                or len(ids) > 5
                or not set(ids) <= candidate_ids
                or int(values["selected_tokens"]) > 1500
            )
            invalid += int(row_invalid)
            metrics = evidence_metrics(gold_row["gold_evidence_ids"], ids)
            methods[method] = {
                **metrics,
                "complete_evidence": float(
                    set(gold_row["gold_evidence_ids"]) <= set(ids)
                ),
                "selected_tokens": int(values["selected_tokens"]),
                "route": values["route"],
                "predicted_soft_gain": values["predicted_soft_gain"],
                "invalid": row_invalid,
            }
        cases.append(
            {
                "case_id": case_id,
                "question_type": str(gold_row["question_type"]),
                "gold_evidence_count": len(gold_row["gold_evidence_ids"]),
                "methods": methods,
            }
        )
    aggregates: dict[str, Any] = {}
    for method in METHODS:
        rows = [case["methods"][method] for case in cases]
        aggregates[method] = {
            "cases": len(rows),
            "evidence_macro_f1": round(
                sum(float(row["evidence_f1"]) for row in rows) / len(rows), 6
            ),
            "evidence_macro_recall": round(
                sum(float(row["evidence_recall"]) for row in rows) / len(rows),
                6,
            ),
            "complete_evidence_recall": round(
                sum(float(row["complete_evidence"]) for row in rows) / len(rows),
                6,
            ),
            "mean_selected_tokens": round(
                sum(float(row["selected_tokens"]) for row in rows) / len(rows),
                6,
            ),
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
            seed=seed + index,
        )
        for index, method in enumerate(CONTROL_METHODS)
    }
    per_type: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        subset = [case for case in cases if case["question_type"] == question_type]
        best_control = max(
            CONTROL_METHODS,
            key=lambda method: sum(
                float(case["methods"][method]["evidence_f1"]) for case in subset
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
    cross_delta = comparisons["cross_encoder_topk"]
    override_rate = sum(
        case["methods"][CANDIDATE_METHOD]["route"] == SOFT_METHOD for case in cases
    ) / len(cases)
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(aggregates[strongest]["mean_selected_tokens"])
    )
    counts = Counter(case["question_type"] for case in cases)
    checks = {
        "exact_cases_equals_800": len(cases) == CASES_PER_STAGE,
        "exact_question_type_balance": counts
        == Counter({name: QUOTA_PER_TYPE for name in QUESTION_TYPES}),
        "history_or_stage_overlap_equals_0": history_overlap == 0
        and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_55": float(
            candidate["evidence_macro_f1"]
        )
        >= 0.55,
        "candidate_complete_evidence_recall_at_least_0_55": float(
            candidate["complete_evidence_recall"]
        )
        >= 0.55,
        "candidate_minus_strongest_control_f1_at_least_0_005": float(
            strongest_delta["point"]
        )
        >= 0.005,
        "candidate_minus_strongest_control_ci_low_above_0": float(
            strongest_delta["ci_low"]
        )
        > 0.0,
        "candidate_minus_cross_encoder_f1_at_least_0_005": float(cross_delta["point"])
        >= 0.005,
        "candidate_minus_cross_encoder_ci_low_above_0": float(cross_delta["ci_low"])
        > 0.0,
        "every_question_type_delta_vs_best_control_at_least_minus_0_005": all(
            float(row["delta"]) >= -0.005 for row in per_type.values()
        ),
        "soft_override_rate_between_0_15_and_0_55": 0.15 <= override_rate <= 0.55,
        "mean_selected_tokens_within_1_05_of_strongest_control": token_ratio <= 1.05,
    }
    passed = all(checks.values())
    if stage == "development":
        status = (
            "HOTPOT_V76_GRAPH_ROUTER_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "HOTPOT_V76_GRAPH_ROUTER_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "HOTPOT_V76_GRAPH_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED"
            if passed
            else "HOTPOT_V76_GRAPH_ROUTER_CONFIRMATION_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "question_type_counts": dict(sorted(counts.items())),
            "history_overlap": history_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "router_artifact_sha256": sha256(router_artifact_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "official_type_answer_or_gold_used_by_runtime_router": False,
            "official_hotpotqa_leaderboard_result": False,
            "independent_public_dataset_from_v75_2wiki_history": True,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_registered_control": {
                "name": strongest,
                **aggregates[strongest],
            },
            "paired_f1_delta": comparisons,
            "router": {
                "soft_override_rate": round(override_rate, 6),
                "history_crossfit_override_rate": router_artifact["crossfit"][
                    "override_rate"
                ],
            },
            "per_question_type_delta_vs_best_control": per_type,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "reuse_target_stage_for_feature_model_threshold_gate_or_selection": False,
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
        f"# HotpotQA 图路由实验（v76 {report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强登记对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- 软闭包覆盖率：`{analysis['router']['soft_override_rate']:.6f}`",
        f"- Gate 2：`{analysis['outcome']['gate_2']}`",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "该实验不是 HotpotQA 官方排行榜、答案生成、真实 SetR、洪水专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v76 protocol experiment id")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("name") != CANDIDATE_METHOD
        or candidate.get("model_type") != "gradient_boosting_regressor_export"
        or candidate.get("feature_count") != len(FEATURE_NAMES)
        or candidate.get("feature_names_sha256")
        != canonical_json_sha256(list(FEATURE_NAMES))
        or candidate.get("estimators") != GBR_ESTIMATORS
        or candidate.get("max_depth") != GBR_MAX_DEPTH
        or candidate.get("min_samples_leaf") != GBR_MIN_SAMPLES_LEAF
        or candidate.get("learning_rate") != GBR_LEARNING_RATE
        or candidate.get("loss") != GBR_LOSS
        or candidate.get("random_state") != GBR_RANDOM_STATE
        or candidate.get("override_threshold") != OVERRIDE_THRESHOLD
        or candidate.get("default_route") != "cross_encoder_topk"
        or candidate.get("positive_route") != SOFT_METHOD
        or candidate.get("official_type_answer_or_gold_used_at_runtime") is not False
    ):
        raise ValueError("v76 protocol candidate configuration drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v76 protocol controls drifted")
    if protocol.get("common_resources") != COMMON_RESOURCES:
        raise ValueError("v76 protocol resources drifted")
    metrics = protocol.get("metrics", {})
    if (
        metrics.get("primary") != "evidence_macro_f1"
        or metrics.get("paired_uncertainty") != "10000 case bootstrap resamples"
        or any(metrics.get(name) != value for name, value in METRIC_SEEDS.items())
    ):
        raise ValueError("v76 protocol metrics drifted")
    if protocol.get("development_gates") != DEVELOPMENT_GATES:
        raise ValueError("v76 protocol gates drifted")
    for stage in ("development", "confirmation"):
        stage_contract = protocol.get("stages", {}).get(stage, {})
        if stage_contract.get("cases") != CASES_PER_STAGE or stage_contract.get(
            "question_type_quota"
        ) != {name: QUOTA_PER_TYPE for name in QUESTION_TYPES}:
            raise ValueError("v76 protocol stage sampling drifted")
    if (
        protocol.get("stages", {})
        .get("confirmation", {})
        .get("open_only_if_every_development_gate_passes")
        is not True
    ):
        raise ValueError("v76 protocol confirmation policy drifted")
    policy = protocol.get("confirmation_policy", {})
    required_policy = {
        "all_development_gates_required": True,
        "no_target_stage_tuning": True,
        "confirmation_uses_identical_router_selectors_controls_metrics_and_thresholds": True,
        "confirmation_success_does_not_authorize_production_or_gate_2_promotion": True,
    }
    if policy != required_policy:
        raise ValueError("v76 protocol confirmation policy drifted")


def validate_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(validate_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(validate_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


__all__ = [
    "CANDIDATE_METHOD",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "FEATURE_NAMES",
    "METHODS",
    "QUESTION_TYPES",
    "ZERO_SHOT_METHOD",
    "blind_history_row",
    "develop_model",
    "evaluate_stage",
    "export_gbr",
    "graph_features",
    "load_history_ids",
    "load_router_artifact",
    "predict_gbr",
    "prepare_case",
    "prepare_stage",
    "select_method",
    "select_stage_ids",
    "validate_finite",
    "validate_registered_protocol",
    "write_selection_outputs",
]
