"""Safety-constrained score-gap cardinality router for IIRC v87.

Model development uses only the 400 fully exposed v86 development cases.  The
runtime feature extractor accepts blind scored rows and cannot consume answers,
answer types, gold contexts, or gold cardinalities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.iirc_cardinality_transfer_router import rank_evidence_sources
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-IIRC-SCORE-GAP-ROUTER-MODEL-V87"
SCHEMA_VERSION = "frc-iirc-score-gap-router-model-v87"
FOLD_COUNT = 5
FOLD_SALT = "FRC-IIRC-V87-CROSSFIT|"
CARDINALITIES = (1, 2, 3, 4)
ESTIMATORS = 150
MAX_FEATURES = 0.8
MODEL_KINDS = ("extra_trees", "random_forest")
MAX_DEPTHS: tuple[int | None, ...] = (3, 5, None)
MIN_SAMPLES_LEAVES = (5, 10, 20)
ACTION_1_BIASES = tuple(round(-0.10 + index * 0.01, 2) for index in range(21))
ACTION_3_BIASES = tuple(round(-0.06 + index * 0.02, 2) for index in range(7))
ACTION_4_BIASES = ACTION_3_BIASES
RECALL_DELTA_FLOOR = -0.01
COMPLETE_DELTA_FLOOR = -0.01
MEAN_SELECTED_SOURCES_CEILING = 2.5
SECOND_ACTION_MINIMUM_COUNT = 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260808
EXPECTED_TRAINING_CASES = 400
EXPECTED_SCORED_SHA256 = (
    "f7b54f2873a85f623e4d42d6a990323ee1429f1f70ece9c16362a339e5a4477f"
)
EXPECTED_GOLD_SHA256 = (
    "29a350a3ada1cb4362abed8ae4a2d8bfd9796ad3fd143ca87c047ac7e4427e61"
)
QUESTION_CUES = (
    "how many",
    "which",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "both",
    "two",
    "first",
    "after",
    "before",
    "between",
    "same",
    "different",
    "respectively",
    "according",
    "name",
    "list",
)
SCORE_NAMES = ("cross_encoder", "bm25", "dense", "hybrid")
FORBIDDEN_RUNTIME_FIELDS = {
    "answer",
    "answer_type",
    "context",
    "gold",
    "gold_cardinality",
    "gold_evidence_sources",
    "question_links",
}


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % FOLD_COUNT


def _finite(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"v87 requires finite {label}")
    return float(value)


def blind_features(row: dict[str, Any]) -> np.ndarray:
    """Extract the frozen 154-dimensional feature vector from one blind row."""

    if FORBIDDEN_RUNTIME_FIELDS & set(row):
        raise ValueError("v87 blind row contains a forbidden gold field")
    candidates = row.get("candidates")
    question = str(row.get("question") or "").strip().lower()
    if not isinstance(candidates, list) or len(candidates) < 4 or not question:
        raise ValueError("v87 blind row requires a question and at least four candidates")
    ordered = rank_evidence_sources(row, limit=4)
    if len(ordered) != 4:
        raise ValueError("v87 FRC order requires four sources")
    values: list[float] = [
        float(len(candidates)),
        math.log1p(len(candidates)),
        float(len(question)),
        float(len(question.split())),
        float(question.count("?")),
        float(question.count(",")),
        float(question.count(" and ")),
        float(question.count(" or ")),
    ]
    values.extend(float(cue in question) for cue in QUESTION_CUES)
    for score_name in SCORE_NAMES:
        scores = sorted(
            (
                _finite(candidate.get("scores", {}).get(score_name), label=score_name)
                for candidate in candidates
            ),
            reverse=True,
        )
        padded = (scores + [0.0] * 8)[:8]
        values.extend(padded)
        values.extend(padded[index] - padded[index + 1] for index in range(7))
        values.extend(
            (
                float(np.mean(scores)),
                float(np.std(scores)),
                float(np.median(scores)),
                float(np.quantile(scores, 0.25)),
                float(np.quantile(scores, 0.75)),
            )
        )
    for item in ordered:
        roles = [
            _finite(value, label="role score")
            for value in item.get("role_scores", {}).values()
        ]
        if len(roles) != 5:
            raise ValueError("v87 expects five frozen FRC roles")
        values.extend(
            (
                _finite(item.get("cross_encoder"), label="FRC cross-encoder score"),
                float(np.mean(roles)),
                float(np.std(roles)),
                max(roles),
                min(roles),
                *sorted(roles, reverse=True),
            )
        )
    for index in range(3):
        current_roles = list(ordered[index]["role_scores"].values())
        next_roles = list(ordered[index + 1]["role_scores"].values())
        values.extend(
            (
                float(ordered[index]["cross_encoder"])
                - float(ordered[index + 1]["cross_encoder"]),
                float(np.mean(current_roles) - np.mean(next_roles)),
            )
        )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (154,) or not np.all(np.isfinite(result)):
        raise ValueError("v87 blind feature schema changed")
    return result


def evidence_metrics(
    gold_sources: Sequence[str], selected_sources: Sequence[str]
) -> tuple[float, float, float, float]:
    gold = set(gold_sources)
    selected = set(selected_sources)
    overlap = len(gold & selected)
    precision = overlap / len(selected) if selected else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, float(gold <= selected)


def load_training_cases(
    scored_path: Path, gold_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sha256(scored_path) != EXPECTED_SCORED_SHA256:
        raise ValueError("v87 exposed v86 scored source changed")
    if sha256(gold_path) != EXPECTED_GOLD_SHA256:
        raise ValueError("v87 exposed v86 gold source changed")
    scored = read_jsonl(scored_path)
    gold = read_jsonl(gold_path)
    scored_by_id = {str(row.get("id") or ""): row for row in scored}
    gold_by_id = {str(row.get("id") or ""): row for row in gold}
    if (
        len(scored) != EXPECTED_TRAINING_CASES
        or len(gold) != EXPECTED_TRAINING_CASES
        or len(scored_by_id) != EXPECTED_TRAINING_CASES
        or len(gold_by_id) != EXPECTED_TRAINING_CASES
        or set(scored_by_id) != set(gold_by_id)
    ):
        raise ValueError("v87 exposed v86 training coverage changed")
    ordered_scored: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for case_id in sorted(scored_by_id):
        scored_row = scored_by_id[case_id]
        gold_sources = [str(value) for value in gold_by_id[case_id]["gold_evidence_sources"]]
        source_order = [
            str(item["source"]) for item in rank_evidence_sources(scored_row, limit=4)
        ]
        utilities = []
        recalls = []
        completes = []
        for cardinality in CARDINALITIES:
            _, recall, f1, complete = evidence_metrics(
                gold_sources, source_order[:cardinality]
            )
            utilities.append(f1)
            recalls.append(recall)
            completes.append(complete)
        ordered_scored.append(scored_row)
        training_rows.append(
            {
                "case_id": case_id,
                "utilities": utilities,
                "recalls": recalls,
                "completes": completes,
            }
        )
    def portable(path: Path) -> str:
        parts = path.resolve().parts
        try:
            start = parts.index(".cache")
        except ValueError:
            return path.as_posix()
        return Path(*parts[start:]).as_posix()

    contract = {
        "scored_path": portable(scored_path),
        "scored_sha256": EXPECTED_SCORED_SHA256,
        "gold_path": portable(gold_path),
        "gold_sha256": EXPECTED_GOLD_SHA256,
        "cases": EXPECTED_TRAINING_CASES,
    }
    return ordered_scored, training_rows, contract


def _make_model(
    kind: str,
    *,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
) -> Any:
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model_type = ExtraTreesRegressor
    elif kind == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model_type = RandomForestRegressor
    else:
        raise ValueError(f"unsupported v87 model kind: {kind}")
    return model_type(
        n_estimators=ESTIMATORS,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=MAX_FEATURES,
        random_state=seed,
        n_jobs=1,
    )


def serialize_forest(model: Any) -> dict[str, Any]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = np.asarray(tree.value, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != 1:
            raise ValueError("v87 forest output shape changed")
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": values[:, :, 0].astype(float).tolist(),
            }
        )
    if len(trees) != ESTIMATORS:
        raise ValueError("v87 forest estimator count changed")
    return {
        "n_features": int(model.n_features_in_),
        "n_outputs": int(model.n_outputs_),
        "estimators": len(trees),
        "trees": trees,
    }


def predict_serialized_forest(
    forest: dict[str, Any], features: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    single = matrix.ndim == 1
    if single:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != int(forest.get("n_features", 0)):
        raise ValueError("v87 forest feature dimension changed")
    output_count = int(forest.get("n_outputs", 0))
    predictions = np.zeros((matrix.shape[0], output_count), dtype=np.float64)
    trees = forest.get("trees")
    if not isinstance(trees, list) or len(trees) != int(forest.get("estimators", 0)):
        raise ValueError("v87 serialized forest is incomplete")
    for tree in trees:
        left = tree["children_left"]
        right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        values = tree["value"]
        for row_index, row in enumerate(matrix):
            node = 0
            while int(left[node]) != int(right[node]):
                node = (
                    int(left[node])
                    if row[int(feature[node])] <= float(threshold[node])
                    else int(right[node])
                )
            predictions[row_index] += np.asarray(values[node], dtype=np.float64)
    predictions /= len(trees)
    return predictions[0] if single else predictions


def _candidate_metrics(
    actions: np.ndarray,
    utilities: np.ndarray,
    recalls: np.ndarray,
    completes: np.ndarray,
) -> dict[str, Any]:
    indexes = np.asarray(actions, dtype=int) - 1
    row_indexes = np.arange(len(indexes))
    counts = Counter(int(value) for value in actions.tolist())
    return {
        "evidence_macro_f1": float(np.mean(utilities[row_indexes, indexes])),
        "evidence_macro_recall": float(np.mean(recalls[row_indexes, indexes])),
        "complete_evidence_recall": float(np.mean(completes[row_indexes, indexes])),
        "mean_selected_sources": float(np.mean(actions)),
        "action_counts": dict(sorted((str(key), value) for key, value in counts.items())),
    }


def _policy_actions(predictions: np.ndarray, biases: Sequence[float]) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64) + np.asarray(biases, dtype=np.float64)
    return np.argmax(values, axis=1).astype(int) + 1


def _policy_is_safe(candidate: dict[str, Any], fixed2: dict[str, Any]) -> bool:
    counts = sorted(candidate["action_counts"].values(), reverse=True)
    return (
        candidate["evidence_macro_recall"]
        >= fixed2["evidence_macro_recall"] + RECALL_DELTA_FLOOR
        and candidate["complete_evidence_recall"]
        >= fixed2["complete_evidence_recall"] + COMPLETE_DELTA_FLOOR
        and candidate["mean_selected_sources"] <= MEAN_SELECTED_SOURCES_CEILING
        and len(counts) >= 2
        and counts[1] >= SECOND_ACTION_MINIMUM_COUNT
    )


def _paired_bootstrap(candidate: np.ndarray, control: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(candidate), len(candidate))
        values[index] = float(np.mean(candidate[sample] - control[sample]))
    return {
        "point": round(float(np.mean(candidate - control)), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _model_complexity_key(kind: str, depth: int | None, leaf: int) -> tuple[int, int, int]:
    return (
        0 if kind == "random_forest" else 1,
        10_000 if depth is None else depth,
        -leaf,
    )


def train_router(scored_path: Path, gold_path: Path) -> dict[str, Any]:
    scored, rows, source_contract = load_training_cases(scored_path, gold_path)
    features = np.vstack([blind_features(row) for row in scored])
    utilities = np.asarray([row["utilities"] for row in rows], dtype=np.float64)
    recalls = np.asarray([row["recalls"] for row in rows], dtype=np.float64)
    completes = np.asarray([row["completes"] for row in rows], dtype=np.float64)
    folds = np.asarray([_fold(row["case_id"]) for row in rows], dtype=int)
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v87 crossfit folds are incomplete")
    fixed_metrics = {
        str(cardinality): _candidate_metrics(
            np.full(len(rows), cardinality, dtype=int), utilities, recalls, completes
        )
        for cardinality in CARDINALITIES
    }
    fixed2 = fixed_metrics["2"]
    configurations: list[dict[str, Any]] = []
    for kind, max_depth, min_leaf in product(
        MODEL_KINDS, MAX_DEPTHS, MIN_SAMPLES_LEAVES
    ):
        predictions = np.zeros_like(utilities)
        for fold in range(FOLD_COUNT):
            train = folds != fold
            test = ~train
            model = _make_model(
                kind,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                seed=8700 + fold,
            )
            model.fit(features[train], utilities[train])
            predictions[test] = model.predict(features[test])
        best_policy: dict[str, Any] | None = None
        for action1, action3, action4 in product(
            ACTION_1_BIASES, ACTION_3_BIASES, ACTION_4_BIASES
        ):
            biases = (action1, 0.0, action3, action4)
            actions = _policy_actions(predictions, biases)
            metrics = _candidate_metrics(actions, utilities, recalls, completes)
            if not _policy_is_safe(metrics, fixed2):
                continue
            objective = (
                metrics["evidence_macro_f1"],
                metrics["evidence_macro_recall"],
                metrics["complete_evidence_recall"],
                -metrics["mean_selected_sources"],
                -sum(abs(value) for value in biases),
                tuple(-value for value in biases),
            )
            if best_policy is None or objective > best_policy["objective"]:
                best_policy = {
                    "objective": objective,
                    "biases": biases,
                    "actions": actions,
                    "metrics": metrics,
                }
        if best_policy is None:
            continue
        configurations.append(
            {
                "kind": kind,
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "predictions": predictions,
                "policy": best_policy,
            }
        )
    if not configurations:
        raise ValueError("v87 finite model grid has no safety-eligible policy")
    selected = max(
        configurations,
        key=lambda value: (
            *value["policy"]["objective"],
            tuple(
                -part
                for part in _model_complexity_key(
                    value["kind"], value["max_depth"], value["min_samples_leaf"]
                )
            ),
        ),
    )
    final_model = _make_model(
        selected["kind"],
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        seed=8787,
    )
    final_model.fit(features, utilities)
    forest = serialize_forest(final_model)
    actions = selected["policy"]["actions"]
    indexes = actions - 1
    candidate_f1 = utilities[np.arange(len(rows)), indexes]
    fixed2_f1 = utilities[:, 1]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training_source": source_contract,
        "training_cases": len(rows),
        "training_case_ids_sha256": canonical_json_sha256(
            [row["case_id"] for row in rows]
        ),
        "feature_schema": {
            "dimension": features.shape[1],
            "question_cues": list(QUESTION_CUES),
            "score_names": list(SCORE_NAMES),
            "frc_prefix_sources": 4,
            "runtime_inputs_are_blind_only": True,
        },
        "model_selection": {
            "folds": FOLD_COUNT,
            "fold_salt": FOLD_SALT,
            "model_kinds": list(MODEL_KINDS),
            "max_depth_grid": [value for value in MAX_DEPTHS],
            "min_samples_leaf_grid": list(MIN_SAMPLES_LEAVES),
            "estimators": ESTIMATORS,
            "max_features": MAX_FEATURES,
            "model_configurations": len(MODEL_KINDS)
            * len(MAX_DEPTHS)
            * len(MIN_SAMPLES_LEAVES),
            "policy_configurations_per_model": len(ACTION_1_BIASES)
            * len(ACTION_3_BIASES)
            * len(ACTION_4_BIASES),
            "safety_constraints": {
                "recall_delta_vs_fixed2_at_least": RECALL_DELTA_FLOOR,
                "complete_delta_vs_fixed2_at_least": COMPLETE_DELTA_FLOOR,
                "mean_selected_sources_at_most": MEAN_SELECTED_SOURCES_CEILING,
                "second_largest_action_count_at_least": SECOND_ACTION_MINIMUM_COUNT,
            },
            "objective_order": [
                "maximize crossfit evidence macro F1 among safety-eligible policies",
                "maximize evidence recall",
                "maximize complete-evidence recall",
                "minimize mean selected sources",
                "minimize absolute policy bias",
                "prefer deterministic lower-complexity tie breaks",
            ],
            "selected_kind": selected["kind"],
            "selected_max_depth": selected["max_depth"],
            "selected_min_samples_leaf": selected["min_samples_leaf"],
            "selected_biases": [float(value) for value in selected["policy"]["biases"]],
        },
        "crossfit_diagnostic": {
            "candidate": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in selected["policy"]["metrics"].items()
            },
            "fixed_controls": {
                key: {
                    name: round(value, 6) if isinstance(value, float) else value
                    for name, value in metrics.items()
                }
                for key, metrics in fixed_metrics.items()
            },
            "candidate_minus_fixed2_f1": _paired_bootstrap(candidate_f1, fixed2_f1),
            "diagnostic_is_exposed_v86_development_not_v87_target_evidence": True,
        },
        "forest": forest,
        "forbidden_runtime_inputs": sorted(FORBIDDEN_RUNTIME_FIELDS),
    }
    payload["model_payload_sha256"] = canonical_json_sha256(
        {
            "feature_schema": payload["feature_schema"],
            "selected_biases": payload["model_selection"]["selected_biases"],
            "forest": forest,
        }
    )
    return payload


def load_model_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v87 score-gap router schema changed")
    if int(artifact.get("training_cases", 0)) != EXPECTED_TRAINING_CASES:
        raise ValueError("v87 score-gap router training count changed")
    expected = canonical_json_sha256(
        {
            "feature_schema": artifact["feature_schema"],
            "selected_biases": artifact["model_selection"]["selected_biases"],
            "forest": artifact["forest"],
        }
    )
    if artifact.get("model_payload_sha256") != expected:
        raise ValueError("v87 score-gap router payload hash mismatch")
    return artifact


def predict_cardinality(row: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    features = blind_features(row)
    predicted_utilities = predict_serialized_forest(artifact["forest"], features)
    biases = np.asarray(
        artifact["model_selection"]["selected_biases"], dtype=np.float64
    )
    if predicted_utilities.shape != (4,) or biases.shape != (4,):
        raise ValueError("v87 policy shape changed")
    adjusted = predicted_utilities + biases
    cardinality = int(np.argmax(adjusted)) + 1
    return {
        "cardinality": cardinality,
        "predicted_utilities": {
            str(index + 1): float(value)
            for index, value in enumerate(predicted_utilities)
        },
        "adjusted_utilities": {
            str(index + 1): float(value) for index, value in enumerate(adjusted)
        },
    }


def select_method(row: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    routing = predict_cardinality(row, artifact)
    ordered = rank_evidence_sources(row, limit=4)
    selected = ordered[: routing["cardinality"]]
    return {
        **routing,
        "selected_sources": [str(item["source"]) for item in selected],
        "selected_ids": [str(item["candidate_id"]) for item in selected],
    }
