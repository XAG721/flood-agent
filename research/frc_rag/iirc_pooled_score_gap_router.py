"""Pooled v86+v87 safety-constrained IIRC score-gap router for v88."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from research.frc_rag.iirc_cardinality_transfer_router import rank_evidence_sources
from research.frc_rag.iirc_score_gap_router import (
    ACTION_1_BIASES,
    ACTION_3_BIASES,
    ACTION_4_BIASES,
    CARDINALITIES,
    COMPLETE_DELTA_FLOOR,
    ESTIMATORS,
    MAX_DEPTHS,
    MAX_FEATURES,
    MEAN_SELECTED_SOURCES_CEILING,
    MIN_SAMPLES_LEAVES,
    MODEL_KINDS,
    RECALL_DELTA_FLOOR,
    SECOND_ACTION_MINIMUM_COUNT,
    _candidate_metrics,
    _make_model,
    _model_complexity_key,
    _policy_actions,
    _policy_is_safe,
    blind_features,
    evidence_metrics,
    predict_serialized_forest,
    serialize_forest,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-IIRC-POOLED-SCORE-GAP-ROUTER-MODEL-V88"
SCHEMA_VERSION = "frc-iirc-pooled-score-gap-router-model-v88"
FOLD_COUNT = 5
FOLD_SALT = "FRC-IIRC-V88-CROSSFIT|"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260811
EXPECTED_TRAINING_CASES = 800
TRAINING_SOURCES: tuple[tuple[str, str, str, int, str, str], ...] = (
    (
        "v86_development",
        ".cache/benchmarks/iirc/v86/development/scored_blind.jsonl",
        ".cache/benchmarks/iirc/v86/development/sealed_gold.jsonl",
        400,
        "f7b54f2873a85f623e4d42d6a990323ee1429f1f70ece9c16362a339e5a4477f",
        "29a350a3ada1cb4362abed8ae4a2d8bfd9796ad3fd143ca87c047ac7e4427e61",
    ),
    (
        "v87_development",
        ".cache/benchmarks/iirc/v87/development/scored_blind.jsonl",
        ".cache/benchmarks/iirc/v87/development/sealed_gold.jsonl",
        400,
        "c1ab79f190fdf6c78d89cb97c07b92bc4fe4587717e533f64df50259bacbe930",
        "cd614a5872205741cdfa8ecf4cd5b344a480f627b3719199420c421c7c818f37",
    ),
)


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % FOLD_COUNT


def _load_training_cases(
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cohort, scored_name, gold_name, expected, scored_hash, gold_hash in TRAINING_SOURCES:
        scored_path = repo_root / scored_name
        gold_path = repo_root / gold_name
        if sha256(scored_path) != scored_hash or sha256(gold_path) != gold_hash:
            raise ValueError(f"v88 exposed training source changed: {cohort}")
        scored = read_jsonl(scored_path)
        gold = read_jsonl(gold_path)
        scored_by_id = {str(row.get("id") or ""): row for row in scored}
        gold_by_id = {str(row.get("id") or ""): row for row in gold}
        if (
            len(scored) != expected
            or len(gold) != expected
            or len(scored_by_id) != expected
            or len(gold_by_id) != expected
            or set(scored_by_id) != set(gold_by_id)
            or seen & set(scored_by_id)
        ):
            raise ValueError(f"v88 exposed training coverage changed: {cohort}")
        for case_id in sorted(scored_by_id):
            scored_row = scored_by_id[case_id]
            gold_sources = [
                str(value) for value in gold_by_id[case_id]["gold_evidence_sources"]
            ]
            source_order = [
                str(item["source"])
                for item in rank_evidence_sources(scored_row, limit=4)
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
            scored_rows.append(scored_row)
            training_rows.append(
                {
                    "case_id": case_id,
                    "cohort": cohort,
                    "utilities": utilities,
                    "recalls": recalls,
                    "completes": completes,
                }
            )
        seen.update(scored_by_id)
        contracts.append(
            {
                "cohort": cohort,
                "cases": expected,
                "scored_path": scored_name,
                "scored_sha256": scored_hash,
                "gold_path": gold_name,
                "gold_sha256": gold_hash,
            }
        )
    if len(training_rows) != EXPECTED_TRAINING_CASES:
        raise ValueError("v88 pooled training count changed")
    return scored_rows, training_rows, contracts


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


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def train_router(repo_root: Path) -> dict[str, Any]:
    scored, rows, contracts = _load_training_cases(repo_root)
    features = np.vstack([blind_features(row) for row in scored])
    utilities = np.asarray([row["utilities"] for row in rows], dtype=np.float64)
    recalls = np.asarray([row["recalls"] for row in rows], dtype=np.float64)
    completes = np.asarray([row["completes"] for row in rows], dtype=np.float64)
    folds = np.asarray([_fold(row["case_id"]) for row in rows], dtype=int)
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
                seed=8800 + fold,
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
        if best_policy is not None:
            configurations.append(
                {
                    "kind": kind,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "policy": best_policy,
                }
            )
    if not configurations:
        raise ValueError("v88 finite grid has no safety-eligible policy")
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
        seed=8888,
    )
    final_model.fit(features, utilities)
    forest = serialize_forest(final_model)
    actions = selected["policy"]["actions"]
    indexes = actions - 1
    candidate_f1 = utilities[np.arange(len(rows)), indexes]
    cohort_transfer: dict[str, Any] = {}
    biases = selected["policy"]["biases"]
    cohort_values = np.asarray([row["cohort"] for row in rows])
    for held_out in sorted(set(cohort_values.tolist())):
        train = cohort_values != held_out
        test = ~train
        model = _make_model(
            selected["kind"],
            max_depth=selected["max_depth"],
            min_samples_leaf=selected["min_samples_leaf"],
            seed=8890,
        )
        model.fit(features[train], utilities[train])
        local_actions = _policy_actions(model.predict(features[test]), biases)
        cohort_transfer[held_out] = _round_metrics(
            _candidate_metrics(
                local_actions, utilities[test], recalls[test], completes[test]
            )
        )
        cohort_transfer[held_out]["fixed2_f1"] = round(
            float(np.mean(utilities[test, 1])), 6
        )
        cohort_transfer[held_out]["delta_vs_fixed2_f1"] = round(
            cohort_transfer[held_out]["evidence_macro_f1"]
            - cohort_transfer[held_out]["fixed2_f1"],
            6,
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training_sources": contracts,
        "training_cases": len(rows),
        "training_case_ids_sha256": canonical_json_sha256(
            [row["case_id"] for row in rows]
        ),
        "training_cohort_counts": dict(
            sorted(Counter(row["cohort"] for row in rows).items())
        ),
        "feature_schema": {
            "dimension": features.shape[1],
            "source": "frozen v87 blind feature schema",
            "runtime_inputs_are_blind_only": True,
        },
        "model_selection": {
            "folds": FOLD_COUNT,
            "fold_salt": FOLD_SALT,
            "model_kinds": list(MODEL_KINDS),
            "max_depth_grid": list(MAX_DEPTHS),
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
            "selected_kind": selected["kind"],
            "selected_max_depth": selected["max_depth"],
            "selected_min_samples_leaf": selected["min_samples_leaf"],
            "selected_biases": [float(value) for value in biases],
        },
        "crossfit_diagnostic": {
            "candidate": _round_metrics(selected["policy"]["metrics"]),
            "fixed_controls": {
                key: _round_metrics(metrics) for key, metrics in fixed_metrics.items()
            },
            "candidate_minus_fixed2_f1": _paired_bootstrap(
                candidate_f1, utilities[:, 1]
            ),
            "cohort_transfer": cohort_transfer,
            "diagnostic_is_exposed_training_not_v88_target_evidence": True,
        },
        "forest": forest,
        "forbidden_runtime_inputs": [
            "answer",
            "answer type",
            "gold context",
            "gold cardinality",
            "target metric",
        ],
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
        raise ValueError("v88 pooled score-gap router schema changed")
    if int(artifact.get("training_cases", 0)) != EXPECTED_TRAINING_CASES:
        raise ValueError("v88 pooled score-gap training count changed")
    expected = canonical_json_sha256(
        {
            "feature_schema": artifact["feature_schema"],
            "selected_biases": artifact["model_selection"]["selected_biases"],
            "forest": artifact["forest"],
        }
    )
    if artifact.get("model_payload_sha256") != expected:
        raise ValueError("v88 pooled score-gap model payload hash mismatch")
    return artifact


def predict_cardinality(row: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    utilities = predict_serialized_forest(artifact["forest"], blind_features(row))
    biases = np.asarray(
        artifact["model_selection"]["selected_biases"], dtype=np.float64
    )
    adjusted = utilities + biases
    return {
        "cardinality": int(np.argmax(adjusted)) + 1,
        "predicted_utilities": {
            str(index + 1): float(value) for index, value in enumerate(utilities)
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
