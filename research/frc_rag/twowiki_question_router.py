"""Question-text router between frozen 2Wiki graph selectors (v75)."""

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
from research.frc_rag.twowiki_support_path_closure import (
    QUESTION_TYPES,
    _alternating_anchor_select,
    canonical_json_sha256,
    load_history_ids,
    prepare_case,
    read_jsonl,
    select_method as select_v74_method,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-2WIKI-QUESTION-ROUTED-SUPPORT-PATH-V75"
SCHEMA_VERSION = "frc-twowiki-question-routed-support-path-v75"
FEATURE_DIMENSION = 64
LOGISTIC_L2 = 8.0
LOGISTIC_ITERATIONS = 80
ROUTER_THRESHOLD = 0.5
FOLD_COUNT = 5
DEVELOPMENT_SALT = "FRC-2WIKI-V75-DEVELOPMENT|"
CONFIRMATION_SALT = "FRC-2WIKI-V75-CONFIRMATION|"
CANDIDATE_METHOD = "learned_question_routed_support_path_v75"
SOFT_METHOD = "soft_title_link_support_path_closure_v74"
ANCHOR_METHOD = "alternating_anchor_link_control"
LEXICAL_METHOD = "lexical_question_router_control"
CONTROL_METHODS = (
    "cross_encoder_topk",
    "frc_select",
    SOFT_METHOD,
    ANCHOR_METHOD,
    LEXICAL_METHOD,
)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
HAND_TERMS = (
    " or ",
    "older",
    "younger",
    "earlier",
    "later",
    "more",
    "less",
    "longer",
    "shorter",
    "higher",
    "lower",
    "first",
    "before",
    "after",
    "same",
    "both",
    "difference",
    "different",
    "born",
    "founded",
    "released",
    "published",
    "died",
)
START_TERMS = ("which", "who", "what", "where", "when", "how")


def _tokens(question: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", question.lower())


def question_features(
    question: str, *, dimension: int = FEATURE_DIMENSION
) -> np.ndarray:
    if dimension <= 0:
        raise ValueError("router hash dimension must be positive")
    lowered = question.lower()
    tokens = _tokens(question)
    offset = 2 + len(START_TERMS) + len(HAND_TERMS)
    values = np.zeros(offset + dimension, dtype=np.float64)
    values[0] = min(len(tokens), 40) / 40.0
    values[1] = float(lowered.count("?"))
    for index, term in enumerate(START_TERMS, start=2):
        values[index] = float(bool(tokens) and tokens[0] == term)
    hand_offset = 2 + len(START_TERMS)
    for index, term in enumerate(HAND_TERMS, start=hand_offset):
        values[index] = float(term in lowered)
    grams = set(tokens)
    grams.update(
        "__".join(tokens[index : index + 2]) for index in range(max(0, len(tokens) - 1))
    )
    for gram in grams:
        bucket = (
            int.from_bytes(hashlib.sha256(gram.encode("utf-8")).digest()[:8], "big")
            % dimension
        )
        values[offset + bucket] = 1.0
    return values


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = LOGISTIC_L2,
    iterations: int = LOGISTIC_ITERATIONS,
) -> dict[str, Any]:
    if features.ndim != 2 or len(features) != len(labels) or not len(features):
        raise ValueError("v75 router training matrix is invalid")
    if set(np.asarray(labels, dtype=int)) != {0, 1}:
        raise ValueError("v75 router training labels require both classes")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    design = np.column_stack([np.ones(len(features)), (features - means) / scales])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    converged = False
    completed = 0
    for iteration in range(iterations):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        hessian += np.eye(design.shape[1], dtype=np.float64) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        completed = iteration + 1
        if float(np.max(np.abs(step))) < 1e-9:
            converged = True
            break
    return {
        "dimension": features.shape[1],
        "means": means,
        "scales": scales,
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:],
        "l2": l2,
        "iterations": completed,
        "converged": converged,
    }


def predict_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    means = np.asarray(model["means"], dtype=np.float64)
    scales = np.asarray(model["scales"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(weights):
        raise ValueError("v75 router feature dimension does not match model")
    logits = np.clip(
        float(model["intercept"]) + ((matrix - means) / scales) @ weights,
        -35.0,
        35.0,
    )
    return 1.0 / (1.0 + np.exp(-logits))


def serialize_model(model: dict[str, Any]) -> dict[str, Any]:
    total_feature_count = len(model["weights"])
    fixed_feature_count = 2 + len(START_TERMS) + len(HAND_TERMS)
    hash_dimension = total_feature_count - fixed_feature_count
    if hash_dimension <= 0:
        raise ValueError("v75 router model does not contain hashed question features")
    return {
        "feature_dimension": hash_dimension,
        "total_feature_count": total_feature_count,
        "l2": float(model["l2"]),
        "iterations": int(model["iterations"]),
        "converged": bool(model["converged"]),
        "threshold": ROUTER_THRESHOLD,
        "intercept": round(float(model["intercept"]), 15),
        "means": [round(float(value), 15) for value in model["means"]],
        "scales": [round(float(value), 15) for value in model["scales"]],
        "weights": [round(float(value), 15) for value in model["weights"]],
    }


def deserialize_model(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "means": np.asarray(payload["means"], dtype=np.float64),
        "scales": np.asarray(payload["scales"], dtype=np.float64),
        "intercept": float(payload["intercept"]),
        "weights": np.asarray(payload["weights"], dtype=np.float64),
        "l2": float(payload["l2"]),
        "iterations": int(payload["iterations"]),
        "converged": bool(payload["converged"]),
    }


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(
        f"FRC-2WIKI-V75-ROUTER-FOLD|{case_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    sensitivity = float((predictions[positive] == 1).mean())
    specificity = float((predictions[negative] == 0).mean())
    return (sensitivity + specificity) / 2.0


def develop_router(
    scored_rows: Sequence[dict[str, Any]],
    audited_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    ids = [str(case["case_id"]) for case in audited_cases]
    if set(ids) != set(scored_by_id) or len(set(ids)) != len(ids):
        raise ValueError("v75 router history rows do not align")
    labels = np.asarray(
        [
            float(case["question_type"] in {"bridge_comparison", "comparison"})
            for case in audited_cases
        ],
        dtype=np.float64,
    )
    folds = np.asarray([_fold(case_id) for case_id in ids], dtype=int)
    soft = np.asarray(
        [float(case["methods"][SOFT_METHOD]["evidence_f1"]) for case in audited_cases]
    )
    anchor = np.asarray(
        [float(case["methods"][ANCHOR_METHOD]["evidence_f1"]) for case in audited_cases]
    )
    search_rows: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    selected: dict[str, Any] | None = None
    for dimension in (64, 128, 256):
        features = np.vstack(
            [
                question_features(
                    scored_by_id[case_id]["question"], dimension=dimension
                )
                for case_id in ids
            ]
        )
        for l2 in (0.5, 2.0, 8.0):
            probabilities = np.zeros(len(ids), dtype=np.float64)
            fold_models: list[dict[str, Any]] = []
            for fold in range(FOLD_COUNT):
                train = folds != fold
                held_out = ~train
                model = fit_logistic(features[train], labels[train], l2=l2)
                if not model["converged"]:
                    raise ValueError("v75 historical router fold did not converge")
                probabilities[held_out] = predict_logistic(model, features[held_out])
                fold_models.append(
                    {
                        "fold": fold,
                        "training_cases": int(train.sum()),
                        "held_out_cases": int(held_out.sum()),
                        "model_sha256": canonical_json_sha256(serialize_model(model)),
                    }
                )
            for threshold in (0.4, 0.5, 0.6):
                predictions = probabilities >= threshold
                routed = np.where(predictions, anchor, soft)
                balanced = _balanced_accuracy(labels, predictions.astype(int))
                row = {
                    "hash_dimension": dimension,
                    "l2": l2,
                    "threshold": threshold,
                    "crossfit_router_balanced_accuracy": round(balanced, 6),
                    "crossfit_routed_evidence_macro_f1": round(float(routed.mean()), 6),
                    "comparison_route_rate": round(float(predictions.mean()), 6),
                    "fold_models": fold_models,
                }
                search_rows.append(row)
                key = (
                    float(routed.mean()),
                    balanced,
                    -dimension,
                    l2,
                    -abs(threshold - 0.5),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    selected = row
    assert selected is not None
    if (
        selected["hash_dimension"] != FEATURE_DIMENSION
        or selected["l2"] != LOGISTIC_L2
        or selected["threshold"] != ROUTER_THRESHOLD
    ):
        raise AssertionError(
            "v75 finite search did not select the frozen configuration"
        )
    full_features = np.vstack(
        [question_features(scored_by_id[case_id]["question"]) for case_id in ids]
    )
    full_model = fit_logistic(full_features, labels)
    model_payload = serialize_model(full_model)
    oracle = np.where(labels == 1, anchor, soft)
    return {
        "schema_version": "frc-twowiki-question-router-model-development-v75",
        "experiment_id": EXPERIMENT_ID,
        "history_cases": len(ids),
        "history_case_ids_sha256": canonical_json_sha256(ids),
        "label_definition": "comparison family iff official history type is bridge_comparison or comparison; type is training label only",
        "runtime_inputs": "question text only",
        "search_space": {
            "hash_dimensions": [64, 128, 256],
            "l2": [0.5, 2.0, 8.0],
            "thresholds": [0.4, 0.5, 0.6],
            "configurations": 27,
            "folds": FOLD_COUNT,
            "selection_order": "routed F1, balanced accuracy, smaller dimension, larger L2, threshold closest to 0.5",
        },
        "search_results": search_rows,
        "selected_crossfit": selected,
        "history_reference": {
            "soft_f1": round(float(soft.mean()), 6),
            "anchor_f1": round(float(anchor.mean()), 6),
            "oracle_type_router_f1": round(float(oracle.mean()), 6),
            "selected_crossfit_router_f1": selected[
                "crossfit_routed_evidence_macro_f1"
            ],
        },
        "model": model_payload,
        "model_sha256": canonical_json_sha256(model_payload),
        "prospective_boundary": {
            "all_history_ids_excluded_from_v75_target_stages": True,
            "official_type_available_to_runtime_router": False,
            "v75_target_rows_or_metrics_seen": False,
        },
    }


def load_model_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v75 model artifact experiment id")
    payload = artifact["model"]
    if canonical_json_sha256(payload) != artifact["model_sha256"]:
        raise ValueError("v75 model payload hash mismatch")
    if (
        payload["feature_dimension"] != FEATURE_DIMENSION
        or payload["l2"] != LOGISTIC_L2
        or payload["threshold"] != ROUTER_THRESHOLD
    ):
        raise ValueError("v75 model artifact differs from frozen configuration")
    expected_features = 2 + len(START_TERMS) + len(HAND_TERMS) + FEATURE_DIMENSION
    if (
        payload.get("total_feature_count") != expected_features
        or len(payload.get("means", [])) != expected_features
        or len(payload.get("scales", [])) != expected_features
        or len(payload.get("weights", [])) != expected_features
    ):
        raise ValueError("v75 model artifact feature arrays are inconsistent")
    if not payload.get("converged") or not validate_finite(payload):
        raise ValueError("v75 model artifact is non-converged or non-finite")
    return artifact, deserialize_model(payload)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v75 protocol experiment id")
    router = protocol.get("router", {})
    if (
        router.get("name") != CANDIDATE_METHOD
        or router.get("hash_dimension") != FEATURE_DIMENSION
        or router.get("l2") != LOGISTIC_L2
        or router.get("threshold") != ROUTER_THRESHOLD
        or router.get("runtime_input") != "question text only"
        or router.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v75 protocol router configuration drifted")
    resources = protocol.get("common_resources", {})
    if resources.get("top_k") != 5 or resources.get("token_budget") != 1500:
        raise ValueError("v75 protocol retrieval resource limits drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v75 protocol controls drifted")
    for stage in ("development", "confirmation"):
        stage_contract = protocol.get("stages", {}).get(stage, {})
        if stage_contract.get("cases") != 800 or stage_contract.get(
            "question_type_quota"
        ) != {name: 200 for name in QUESTION_TYPES}:
            raise ValueError("v75 protocol stage sampling drifted")
    expected_gates = {
        "exact_cases": 800,
        "exact_question_type_balance": True,
        "prior_or_stage_overlap": 0,
        "invalid_selector_output_rate": 0.0,
        "router_balanced_accuracy_at_least": 0.90,
        "comparison_route_rate_between_inclusive": [0.35, 0.65],
        "candidate_evidence_macro_f1_at_least": 0.53,
        "candidate_complete_evidence_recall_at_least": 0.55,
        "candidate_minus_strongest_static_control_f1_at_least": 0.01,
        "candidate_minus_strongest_static_control_ci_low_above": 0.0,
        "candidate_minus_frc_select_f1_at_least": 0.05,
        "candidate_minus_frc_select_ci_low_above": 0.0,
        "every_question_type_delta_vs_better_of_two_routed_static_selectors_at_least": -0.01,
        "mean_selected_tokens_not_above_strongest_static_control_by_more_than_fraction": 0.05,
    }
    if protocol.get("development_gates") != expected_gates:
        raise ValueError("v75 protocol development gates drifted")


def _order_key(salt: str, case_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{salt}{case_id}".encode("utf-8")).hexdigest(), case_id


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_ids: set[str],
    stage: str,
    quota_per_type: int = 200,
) -> list[str]:
    rows = [
        {"id": str(row.get("_id") or row.get("id") or ""), "type": str(row["type"])}
        for row in metadata_rows
    ]
    if len({row["id"] for row in rows}) != len(rows) or any(
        not row["id"] for row in rows
    ):
        raise ValueError("v75 source metadata ids are missing or duplicated")

    def choose(salt: str, excluded: set[str]) -> list[str]:
        result: list[str] = []
        for question_type in QUESTION_TYPES:
            eligible = [
                row["id"]
                for row in rows
                if row["type"] == question_type and row["id"] not in excluded
            ]
            eligible.sort(key=lambda case_id: _order_key(salt, case_id))
            if len(eligible) < quota_per_type:
                raise ValueError(f"v75 lacks {quota_per_type} {question_type} rows")
            result.extend(eligible[:quota_per_type])
        return result

    development = choose(DEVELOPMENT_SALT, excluded_ids)
    if stage == "development":
        return development
    if stage != "confirmation":
        raise ValueError(f"unsupported v75 stage: {stage}")
    return choose(CONFIRMATION_SALT, excluded_ids | set(development))


def prepare_stage(
    source_path: Path,
    history_pool_path: Path,
    v74_blind_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    history_ids = load_history_ids(history_pool_path)
    v74_rows = read_jsonl(v74_blind_path)
    v74_ids = {str(row["id"]) for row in v74_rows}
    if len(v74_rows) != 800 or len(v74_ids) != 800 or history_ids & v74_ids:
        raise ValueError("v75 prior history ids are invalid or overlapping")
    excluded = history_ids | v74_ids
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
        raise ValueError("v75 selected ids are missing from source")
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
        "prior_excluded_ids": len(excluded),
        "prior_overlap": len(selected_set & excluded),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
    }


def _lexical_comparison(question: str) -> bool:
    lowered = question.lower()
    tokens = _tokens(question)
    return (
        bool(tokens and tokens[0] == "which")
        or " or " in lowered
        or any(term in lowered for term in HAND_TERMS[1:15])
    )


def route_probability(question: str, model: dict[str, Any]) -> float:
    return float(predict_logistic(model, question_features(question))[0])


def select_method(
    row: dict[str, Any],
    method: str,
    model: dict[str, Any],
    *,
    k: int = 5,
    token_budget: int = 1500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if method in {"cross_encoder_topk", "frc_select", SOFT_METHOD}:
        selected = select_v74_method(row, method, k=k, token_budget=token_budget)
        return selected, {"route": method, "comparison_probability": None}
    if method == ANCHOR_METHOD:
        selected = _alternating_anchor_select(row, k=k, token_budget=token_budget)
        return selected, {"route": ANCHOR_METHOD, "comparison_probability": None}
    if method == LEXICAL_METHOD:
        comparison = _lexical_comparison(str(row["question"]))
        route = ANCHOR_METHOD if comparison else SOFT_METHOD
        return select_method(row, route, model, k=k, token_budget=token_budget)[0], {
            "route": route,
            "comparison_probability": float(comparison),
        }
    if method == CANDIDATE_METHOD:
        probability = route_probability(str(row["question"]), model)
        route = ANCHOR_METHOD if probability >= ROUTER_THRESHOLD else SOFT_METHOD
        return select_method(row, route, model, k=k, token_budget=token_budget)[0], {
            "route": route,
            "comparison_probability": probability,
        }
    raise ValueError(f"unsupported v75 method: {method}")


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    forbidden = {"answer", "gold_evidence_ids", "gold", "gold_roles"}
    if any(forbidden & set(row) for row in scored_rows):
        raise ValueError("v75 model rows contain forbidden gold fields")
    for row in scored_rows:
        methods: dict[str, Any] = {}
        for method in METHODS:
            selected, routing = select_method(row, method, model)
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
                "candidate_ids": sorted(str(item["id"]) for item in row["candidates"]),
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
    difference = np.asarray(candidate, dtype=float) - np.asarray(control, dtype=float)
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


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    model_artifact_path: Path,
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    prior_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    artifact, model = load_model_artifact(model_artifact_path)
    scored = read_jsonl(scored_path)
    selections = write_selection_outputs(scored, model, selection_path)
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v75 gold and selections do not align")
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
                "comparison_probability": values["comparison_probability"],
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
                sum(float(row["evidence_recall"]) for row in rows) / len(rows), 6
            ),
            "complete_evidence_recall": round(
                sum(float(row["complete_evidence"]) for row in rows) / len(rows), 6
            ),
            "mean_selected_tokens": round(
                sum(float(row["selected_tokens"]) for row in rows) / len(rows), 6
            ),
        }
    static_controls = ("cross_encoder_topk", "frc_select", SOFT_METHOD, ANCHOR_METHOD)
    strongest = max(
        static_controls,
        key=lambda name: (
            aggregates[name]["evidence_macro_f1"],
            -static_controls.index(name),
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
    type_deltas: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        subset = [case for case in cases if case["question_type"] == question_type]
        candidate_f1 = sum(
            float(case["methods"][CANDIDATE_METHOD]["evidence_f1"]) for case in subset
        ) / len(subset)
        best_static = max(
            (SOFT_METHOD, ANCHOR_METHOD),
            key=lambda name: sum(
                float(case["methods"][name]["evidence_f1"]) for case in subset
            ),
        )
        control_f1 = sum(
            float(case["methods"][best_static]["evidence_f1"]) for case in subset
        ) / len(subset)
        type_deltas[question_type] = {
            "cases": len(subset),
            "best_static_method": best_static,
            "candidate_evidence_macro_f1": round(candidate_f1, 6),
            "best_static_evidence_macro_f1": round(control_f1, 6),
            "delta": round(candidate_f1 - control_f1, 6),
        }
    candidate_routes = [
        case["methods"][CANDIDATE_METHOD]["route"] == ANCHOR_METHOD for case in cases
    ]
    labels = np.asarray(
        [case["question_type"] in {"bridge_comparison", "comparison"} for case in cases]
    )
    predictions = np.asarray(candidate_routes)
    router_ba = _balanced_accuracy(labels.astype(int), predictions.astype(int))
    oracle_f1 = np.mean(
        [
            case["methods"][ANCHOR_METHOD if label else SOFT_METHOD]["evidence_f1"]
            for case, label in zip(cases, labels, strict=True)
        ]
    )
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_delta = comparisons[strongest]
    frc_delta = comparisons["frc_select"]
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(aggregates[strongest]["mean_selected_tokens"])
    )
    counts = Counter(case["question_type"] for case in cases)
    route_rate = float(predictions.mean())
    checks = {
        "exact_cases_equals_800": len(cases) == 800,
        "exact_question_type_balance": counts
        == Counter({name: 200 for name in QUESTION_TYPES}),
        "prior_or_stage_overlap_equals_0": prior_overlap == 0 and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "router_balanced_accuracy_at_least_0_90": router_ba >= 0.90,
        "comparison_route_rate_between_0_35_and_0_65": 0.35 <= route_rate <= 0.65,
        "candidate_evidence_macro_f1_at_least_0_53": float(
            candidate["evidence_macro_f1"]
        )
        >= 0.53,
        "candidate_complete_evidence_recall_at_least_0_55": float(
            candidate["complete_evidence_recall"]
        )
        >= 0.55,
        "candidate_minus_strongest_static_f1_at_least_0_01": float(
            strongest_delta["point"]
        )
        >= 0.01,
        "candidate_minus_strongest_static_ci_low_above_0": float(
            strongest_delta["ci_low"]
        )
        > 0.0,
        "candidate_minus_frc_select_f1_at_least_0_05": float(frc_delta["point"])
        >= 0.05,
        "candidate_minus_frc_select_ci_low_above_0": float(frc_delta["ci_low"]) > 0.0,
        "every_question_type_delta_vs_best_static_at_least_minus_0_01": all(
            float(row["delta"]) >= -0.01 for row in type_deltas.values()
        ),
        "mean_selected_tokens_within_1_05_of_strongest_static": token_ratio <= 1.05,
    }
    passed = all(checks.values())
    if stage == "development":
        status = (
            "2WIKI_V75_QUESTION_ROUTER_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "2WIKI_V75_QUESTION_ROUTER_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "2WIKI_V75_QUESTION_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED"
            if passed
            else "2WIKI_V75_QUESTION_ROUTER_CONFIRMATION_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "question_type_counts": dict(sorted(counts.items())),
            "prior_overlap": prior_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "model_artifact_sha256": sha256(model_artifact_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "official_type_used_by_runtime_router": False,
            "official_2wiki_leaderboard_result": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_static_control": {"name": strongest, **aggregates[strongest]},
            "paired_f1_delta": comparisons,
            "router": {
                "balanced_accuracy": round(router_ba, 6),
                "comparison_route_rate": round(route_rate, 6),
                "oracle_type_router_evidence_macro_f1": round(float(oracle_f1), 6),
                "history_crossfit_balanced_accuracy": artifact["selected_crossfit"][
                    "crossfit_router_balanced_accuracy"
                ],
            },
            "per_question_type_delta_vs_best_static": type_deltas,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "reuse_target_stage_for_router_feature_model_threshold_gate_or_selection": False,
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
    strongest = analysis["strongest_static_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    lines = [
        f"# 2Wiki 问题路由支持路径实验（v75 {report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 路由平衡准确率：`{analysis['router']['balanced_accuracy']:.6f}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强静态对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
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
            "该实验不是官方排行榜、答案生成、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(validate_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(validate_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


__all__ = [
    "ANCHOR_METHOD",
    "CANDIDATE_METHOD",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "FEATURE_DIMENSION",
    "HAND_TERMS",
    "LEXICAL_METHOD",
    "LOGISTIC_L2",
    "METHODS",
    "ROUTER_THRESHOLD",
    "SOFT_METHOD",
    "deserialize_model",
    "develop_router",
    "evaluate_stage",
    "fit_logistic",
    "load_model_artifact",
    "predict_logistic",
    "prepare_stage",
    "question_features",
    "route_probability",
    "select_method",
    "select_stage_ids",
    "serialize_model",
    "validate_finite",
    "write_selection_outputs",
]
