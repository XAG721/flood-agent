"""Cross-dataset question-text evidence-cardinality router for IIRC v86.

Model development is restricted to already exposed HotpotQA, 2WikiMultiHopQA,
and MuSiQue artifacts.  IIRC questions, labels, evidence, and scores are not
accepted by the training API.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.twowiki_question_router import (
    deserialize_model,
    fit_logistic,
    predict_logistic,
    question_features,
    serialize_model,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-IIRC-CARDINALITY-TRANSFER-ROUTER-V86"
SCHEMA_VERSION = "frc-iirc-cardinality-transfer-router-v86"
FOLD_COUNT = 5
FOLD_SALT = "FRC-IIRC-V86-MODEL-FOLD|"
HASH_DIMENSION_GRID = (64, 128, 256, 512)
LOGISTIC_L2_GRID = (8.0, 32.0)
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
CARDINALITIES = (1, 2, 3, 4)
ORDINAL_TARGETS = (2, 3, 4)
ROLE_COVERAGE_WEIGHT = 0.15
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805

TRAINING_SOURCES: tuple[tuple[str, str, str, int], ...] = (
    (
        "hotpot_v84_development",
        ".cache/benchmarks/hotpot_cardinality_v84/development/scored_blind.jsonl",
        ".cache/benchmarks/hotpot_cardinality_v84/development/sealed_gold.jsonl",
        600,
    ),
    (
        "hotpot_v84_confirmation",
        ".cache/benchmarks/hotpot_cardinality_v84/confirmation/scored_blind.jsonl",
        ".cache/benchmarks/hotpot_cardinality_v84/confirmation/sealed_gold.jsonl",
        600,
    ),
    (
        "twowiki_v83_development",
        ".cache/benchmarks/twowiki_precision_trim_v83/development/scored_blind.jsonl",
        ".cache/benchmarks/twowiki_precision_trim_v83/development/sealed_gold.jsonl",
        800,
    ),
    (
        "twowiki_v83_confirmation",
        ".cache/benchmarks/twowiki_precision_trim_v83/confirmation/scored_blind.jsonl",
        ".cache/benchmarks/twowiki_precision_trim_v83/confirmation/sealed_gold.jsonl",
        800,
    ),
    (
        "musique_v77_development",
        ".cache/benchmarks/musique_graph_router_transfer_v77/development/scored_blind.jsonl",
        ".cache/benchmarks/musique_graph_router_transfer_v77/development/sealed_gold.jsonl",
        800,
    ),
    (
        "musique_v78_development",
        ".cache/benchmarks/musique_mean_calibrated_three_route_v78/development/scored_blind.jsonl",
        ".cache/benchmarks/musique_mean_calibrated_three_route_v78/development/sealed_gold.jsonl",
        600,
    ),
    (
        "musique_v79_development",
        ".cache/benchmarks/musique_target_three_route_v79/development/scored_blind.jsonl",
        ".cache/benchmarks/musique_target_three_route_v79/development/sealed_gold.jsonl",
        470,
    ),
    (
        "musique_v80_terminal_holdout",
        ".cache/benchmarks/musique_anchor_default_v80/terminal_holdout/scored_blind.jsonl",
        ".cache/benchmarks/musique_anchor_default_v80/terminal_holdout/sealed_gold.jsonl",
        600,
    ),
)
TRAINING_CASES = sum(source[3] for source in TRAINING_SOURCES)


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def _score(candidate: dict[str, Any], name: str) -> float:
    value = candidate.get("scores", {}).get(name)
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"v86 candidate lacks finite {name} score")
    return float(value)


def rank_evidence_sources(
    row: dict[str, Any], *, limit: int = max(CARDINALITIES)
) -> list[dict[str, Any]]:
    """Return a deterministic FRC role-coverage order with one chunk per source."""

    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("v86 scored row requires candidates")
    by_source: dict[str, dict[str, Any]] = {}
    role_names: set[str] | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("v86 candidate must be an object")
        source = str(candidate.get("source") or "")
        candidate_id = str(candidate.get("id") or "")
        roles = candidate.get("role_scores")
        if not source or not candidate_id or not isinstance(roles, dict) or not roles:
            raise ValueError("v86 candidate source, id, or role scores are invalid")
        local_names = set(roles)
        if role_names is None:
            role_names = local_names
        elif local_names != role_names:
            raise ValueError("v86 role score columns vary within a case")
        cross = _score(candidate, "cross_encoder")
        normalized_roles = {name: float(value) for name, value in roles.items()}
        if any(not np.isfinite(value) for value in normalized_roles.values()):
            raise ValueError("v86 role score is not finite")
        existing = by_source.get(source)
        representative_key = (-cross, candidate_id)
        if existing is None:
            by_source[source] = {
                "source": source,
                "candidate_id": candidate_id,
                "cross_encoder": cross,
                "role_scores": normalized_roles,
                "representative_key": representative_key,
            }
            continue
        existing["role_scores"] = {
            name: max(float(existing["role_scores"][name]), value)
            for name, value in normalized_roles.items()
        }
        if representative_key < existing["representative_key"]:
            existing["candidate_id"] = candidate_id
            existing["cross_encoder"] = cross
            existing["representative_key"] = representative_key
    if role_names is None:
        raise ValueError("v86 source ranking has no roles")
    covered = {name: 0.0 for name in role_names}
    selected: list[dict[str, Any]] = []
    while len(selected) < min(limit, len(by_source)):
        selected_sources = {str(item["source"]) for item in selected}
        eligible = [
            item for source, item in by_source.items() if source not in selected_sources
        ]

        def key(item: dict[str, Any]) -> tuple[float, float, str, str]:
            gain = sum(
                max(0.0, float(item["role_scores"][name]) - covered[name])
                for name in sorted(role_names)
            ) / len(role_names)
            utility = float(item["cross_encoder"]) + ROLE_COVERAGE_WEIGHT * gain
            return (
                -utility,
                -float(item["cross_encoder"]),
                str(item["source"]),
                str(item["candidate_id"]),
            )

        chosen = min(eligible, key=key)
        selected.append(chosen)
        for name in role_names:
            covered[name] = max(covered[name], float(chosen["role_scores"][name]))
    return [
        {key: value for key, value in item.items() if key != "representative_key"}
        for item in selected
    ]


def evidence_f1(gold_sources: Sequence[str], selected_sources: Sequence[str]) -> float:
    gold = set(gold_sources)
    selected = set(selected_sources)
    if not gold and not selected:
        return 1.0
    if not gold or not selected:
        return 0.0
    overlap = len(gold & selected)
    return 2.0 * overlap / (len(gold) + len(selected))


def derive_training_cases(
    scored_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
    *,
    cohort: str,
) -> list[dict[str, Any]]:
    scored_by_id = {str(row.get("id") or ""): row for row in scored_rows}
    gold_by_id = {str(row.get("id") or ""): row for row in gold_rows}
    if (
        "" in scored_by_id
        or "" in gold_by_id
        or len(scored_by_id) != len(scored_rows)
        or len(gold_by_id) != len(gold_rows)
        or set(scored_by_id) != set(gold_by_id)
    ):
        raise ValueError(f"v86 {cohort} scored/gold rows are misaligned")
    cases: list[dict[str, Any]] = []
    for case_id in sorted(scored_by_id):
        scored = scored_by_id[case_id]
        if "iirc" in str(scored.get("dataset", "")).lower():
            raise ValueError("v86 model development cannot accept IIRC rows")
        candidate_source = {
            str(candidate.get("id") or ""): str(candidate.get("source") or "")
            for candidate in scored.get("candidates", [])
        }
        gold_ids = [str(value) for value in gold_by_id[case_id].get("gold_evidence_ids", [])]
        if not gold_ids or any(value not in candidate_source for value in gold_ids):
            raise ValueError(f"v86 {cohort} gold evidence is outside candidates")
        gold_sources = sorted({candidate_source[value] for value in gold_ids})
        if not gold_sources or "" in gold_sources:
            raise ValueError(f"v86 {cohort} gold source mapping is invalid")
        cases.append(
            {
                "id": case_id,
                "cohort": cohort,
                "question": str(scored.get("question") or ""),
                "gold_sources": gold_sources,
                "source_order": [
                    str(item["source"]) for item in rank_evidence_sources(scored)
                ],
            }
        )
    if any(not case["question"] for case in cases):
        raise ValueError(f"v86 {cohort} contains an empty question")
    return cases


def load_training_cases(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cohort, scored_relative, gold_relative, expected in TRAINING_SOURCES:
        scored_path = repo_root / scored_relative
        gold_path = repo_root / gold_relative
        scored_rows = read_jsonl(scored_path)
        gold_rows = read_jsonl(gold_path)
        if len(scored_rows) != expected or len(gold_rows) != expected:
            raise ValueError(f"v86 {cohort} row count changed")
        local = derive_training_cases(scored_rows, gold_rows, cohort=cohort)
        local_ids = {case["id"] for case in local}
        if seen & local_ids:
            raise ValueError(f"v86 {cohort} overlaps another exposed cohort")
        seen.update(local_ids)
        cases.extend(local)
        contracts.append(
            {
                "cohort": cohort,
                "cases": expected,
                "scored_path": scored_relative,
                "scored_sha256": sha256(scored_path),
                "gold_path": gold_relative,
                "gold_sha256": sha256(gold_path),
            }
        )
    if len(cases) != TRAINING_CASES:
        raise ValueError("v86 exposed training case count changed")
    return cases, contracts


def cardinality_from_probabilities(
    probabilities: dict[int, float], thresholds: dict[int, float]
) -> int:
    if set(probabilities) != set(ORDINAL_TARGETS) or set(thresholds) != set(
        ORDINAL_TARGETS
    ):
        raise ValueError("v86 ordinal probability columns are incomplete")
    return 1 + sum(
        float(probabilities[target]) >= float(thresholds[target])
        for target in ORDINAL_TARGETS
    )


def _crossfit_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    l2: float,
) -> dict[int, np.ndarray]:
    probabilities: dict[int, np.ndarray] = {}
    for target in ORDINAL_TARGETS:
        binary = (labels >= target).astype(int)
        values = np.zeros(len(features), dtype=np.float64)
        for fold in range(FOLD_COUNT):
            training = folds != fold
            held_out = ~training
            model = fit_logistic(features[training], binary[training], l2=l2)
            if not model["converged"]:
                raise ValueError("v86 ordinal logistic model did not converge")
            values[held_out] = predict_logistic(model, features[held_out])
        probabilities[target] = values
    return probabilities


def _metrics(cases: Sequence[dict[str, Any]], cardinalities: np.ndarray) -> dict[str, float]:
    f1_values: list[float] = []
    recalls: list[float] = []
    complete: list[float] = []
    for case, cardinality in zip(cases, cardinalities, strict=True):
        selected = case["source_order"][: int(cardinality)]
        gold = set(case["gold_sources"])
        selected_set = set(selected)
        f1_values.append(evidence_f1(case["gold_sources"], selected))
        recalls.append(len(gold & selected_set) / len(gold))
        complete.append(float(gold <= selected_set))
    return {
        "evidence_macro_f1": round(float(np.mean(f1_values)), 6),
        "evidence_macro_recall": round(float(np.mean(recalls)), 6),
        "complete_evidence_recall": round(float(np.mean(complete)), 6),
        "mean_selected_sources": round(float(np.mean(cardinalities)), 6),
    }


def _paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("v86 paired bootstrap arrays are invalid")
    differences = candidate - baseline
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(differences), len(differences))
        values[index] = float(np.mean(differences[sample]))
    return {
        "point": round(float(np.mean(differences)), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def develop_model(repo_root: Path) -> dict[str, Any]:
    cases, contracts = load_training_cases(repo_root)
    case_ids = [case["id"] for case in cases]
    labels = np.asarray(
        [min(max(CARDINALITIES), len(case["gold_sources"])) for case in cases],
        dtype=int,
    )
    folds = np.asarray([_fold(case_id) for case_id in case_ids], dtype=int)
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v86 deterministic folds are incomplete")
    fixed_metrics = {
        f"fixed_{cardinality}": _metrics(
            cases, np.full(len(cases), cardinality, dtype=int)
        )
        for cardinality in CARDINALITIES
    }
    search: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    selected: dict[str, Any] | None = None
    for dimension, l2 in product(HASH_DIMENSION_GRID, LOGISTIC_L2_GRID):
        features = np.vstack(
            [question_features(case["question"], dimension=dimension) for case in cases]
        )
        probabilities = _crossfit_probabilities(features, labels, folds, l2=l2)
        for threshold_values in product(THRESHOLD_GRID, repeat=len(ORDINAL_TARGETS)):
            thresholds = dict(zip(ORDINAL_TARGETS, threshold_values, strict=True))
            cardinalities = np.asarray(
                [
                    cardinality_from_probabilities(
                        {target: probabilities[target][index] for target in ORDINAL_TARGETS},
                        thresholds,
                    )
                    for index in range(len(cases))
                ],
                dtype=int,
            )
            metrics = _metrics(cases, cardinalities)
            key = (
                float(metrics["evidence_macro_f1"]),
                float(metrics["complete_evidence_recall"]),
                -float(metrics["mean_selected_sources"]),
                -dimension,
                -l2,
                *(-float(thresholds[target]) for target in ORDINAL_TARGETS),
            )
            if best_key is None or key > best_key:
                best_key = key
                selected = {
                    "hash_dimension": dimension,
                    "l2": l2,
                    "thresholds": thresholds,
                    "cardinalities": cardinalities,
                    "metrics": metrics,
                    "probabilities": probabilities,
                }
        local_best = max(
            (
                item
                for item in search
                if item["hash_dimension"] == dimension and item["l2"] == l2
            ),
            key=lambda item: item["evidence_macro_f1"],
            default=None,
        )
        search.append(
            {
                "hash_dimension": dimension,
                "l2": l2,
                "evaluated_threshold_triplets": len(THRESHOLD_GRID)
                ** len(ORDINAL_TARGETS),
                "note": "Only the globally selected configuration is materialized; all configurations used identical folds and finite grids.",
                "prior_local_entry": local_best,
            }
        )
    if selected is None:
        raise AssertionError("v86 model search produced no candidate")
    selected_dimension = int(selected["hash_dimension"])
    selected_l2 = float(selected["l2"])
    full_features = np.vstack(
        [
            question_features(case["question"], dimension=selected_dimension)
            for case in cases
        ]
    )
    ordinal_models: dict[str, Any] = {}
    for target in ORDINAL_TARGETS:
        model = fit_logistic(
            full_features,
            (labels >= target).astype(int),
            l2=selected_l2,
        )
        if not model["converged"]:
            raise ValueError("v86 final ordinal logistic model did not converge")
        ordinal_models[str(target)] = serialize_model(model)
    cardinalities = np.asarray(selected["cardinalities"], dtype=int)
    candidate_f1 = np.asarray(
        [
            evidence_f1(case["gold_sources"], case["source_order"][: int(value)])
            for case, value in zip(cases, cardinalities, strict=True)
        ]
    )
    fixed_f1 = {
        cardinality: np.asarray(
            [
                evidence_f1(case["gold_sources"], case["source_order"][:cardinality])
                for case in cases
            ]
        )
        for cardinality in CARDINALITIES
    }
    strongest = max(
        CARDINALITIES,
        key=lambda value: (
            float(np.mean(fixed_f1[value])),
            -value,
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training_cases": len(cases),
        "training_case_ids_sha256": canonical_json_sha256(sorted(case_ids)),
        "training_sources": contracts,
        "training_cohort_counts": dict(sorted(Counter(case["cohort"] for case in cases).items())),
        "gold_source_cardinality_clipped_at_4": dict(
            sorted((str(key), value) for key, value in Counter(labels.tolist()).items())
        ),
        "model_selection": {
            "folds": FOLD_COUNT,
            "fold_salt": FOLD_SALT,
            "hash_dimension_grid": list(HASH_DIMENSION_GRID),
            "l2_grid": list(LOGISTIC_L2_GRID),
            "threshold_grid": list(THRESHOLD_GRID),
            "classifier_configurations": len(HASH_DIMENSION_GRID)
            * len(LOGISTIC_L2_GRID),
            "policy_configurations": len(HASH_DIMENSION_GRID)
            * len(LOGISTIC_L2_GRID)
            * len(THRESHOLD_GRID) ** len(ORDINAL_TARGETS),
            "objective_order": [
                "maximize crossfit evidence_macro_f1",
                "maximize complete_evidence_recall",
                "minimize mean_selected_sources",
                "minimize feature dimension, l2, and thresholds",
            ],
            "selected_hash_dimension": selected_dimension,
            "selected_l2": selected_l2,
            "selected_thresholds": {
                str(key): float(value) for key, value in selected["thresholds"].items()
            },
        },
        "source_ranking": {
            "one_representative_chunk_per_source": True,
            "cross_encoder_aggregation": "maximum per source",
            "role_score_aggregation": "independent maximum per role and source",
            "greedy_utility": "source_max_cross_encoder + 0.15 * mean_positive_role_coverage_gain",
            "role_coverage_weight": ROLE_COVERAGE_WEIGHT,
            "maximum_sources": max(CARDINALITIES),
        },
        "crossfit_diagnostic": {
            "candidate": selected["metrics"],
            "candidate_action_counts": dict(
                sorted(
                    (str(key), value)
                    for key, value in Counter(cardinalities.tolist()).items()
                )
            ),
            "fixed_controls": fixed_metrics,
            "strongest_fixed_control": f"fixed_{strongest}",
            "candidate_minus_strongest_fixed_f1": _paired_bootstrap(
                candidate_f1, fixed_f1[strongest]
            ),
            "diagnostic_is_exposed_history_not_iirc_target_evidence": True,
        },
        "ordinal_models": ordinal_models,
        "runtime_inputs": [
            "question text",
            "blind candidate source ids",
            "blind cross-encoder scores",
            "blind FRC role scores",
        ],
        "forbidden_training_or_runtime_inputs": [
            "IIRC answer",
            "IIRC answer type",
            "IIRC question_links",
            "IIRC gold context",
            "IIRC gold passage ids",
            "IIRC target metrics",
        ],
    }
    payload["model_payload_sha256"] = canonical_json_sha256(
        {
            "thresholds": payload["model_selection"]["selected_thresholds"],
            "ordinal_models": ordinal_models,
            "source_ranking": payload["source_ranking"],
        }
    )
    return payload


def load_model_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v86 cardinality router schema changed")
    expected = canonical_json_sha256(
        {
            "thresholds": artifact["model_selection"]["selected_thresholds"],
            "ordinal_models": artifact["ordinal_models"],
            "source_ranking": artifact["source_ranking"],
        }
    )
    if artifact.get("model_payload_sha256") != expected:
        raise ValueError("v86 cardinality router payload hash mismatch")
    if int(artifact.get("training_cases", 0)) != TRAINING_CASES:
        raise ValueError("v86 cardinality router training count changed")
    return artifact


def predict_cardinality(question: str, artifact: dict[str, Any]) -> dict[str, Any]:
    dimension = int(artifact["model_selection"]["selected_hash_dimension"])
    features = question_features(question, dimension=dimension)
    probabilities = {
        target: float(
            predict_logistic(
                deserialize_model(artifact["ordinal_models"][str(target)]), features
            )[0]
        )
        for target in ORDINAL_TARGETS
    }
    thresholds = {
        target: float(artifact["model_selection"]["selected_thresholds"][str(target)])
        for target in ORDINAL_TARGETS
    }
    return {
        "cardinality": cardinality_from_probabilities(probabilities, thresholds),
        "probabilities": {str(key): value for key, value in probabilities.items()},
    }


def select_method(row: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    routing = predict_cardinality(str(row.get("question") or ""), artifact)
    ordered = rank_evidence_sources(row)
    selected = ordered[: int(routing["cardinality"])]
    return {
        **routing,
        "selected_sources": [str(item["source"]) for item in selected],
        "selected_ids": [str(item["candidate_id"]) for item in selected],
    }
