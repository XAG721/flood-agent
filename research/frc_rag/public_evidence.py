"""Public benchmark evidence aggregation for offline FRC-RAG research."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_METHODS = {
    "conditionalqa": (
        "bm25_topk",
        "dense_topk",
        "hybrid_topk",
        "cross_encoder_topk",
        "mmr",
        "setr_style",
        "frc_select",
    ),
    "multihoprag": ("cross_encoder_topk", "mmr", "setr_style", "frc_select"),
    "hotpotqa": ("cross_encoder_topk", "mmr", "setr_style", "frc_select"),
}

DISPLAY_METHOD = {"setr_style": "coverage_greedy_proxy"}
PRIMARY_METRICS = (
    "evidence_recall",
    "evidence_precision",
    "evidence_f1",
    "role_coverage",
)

PLANNED_ABLATIONS = (
    "full",
    "w/o_role",
    "w/o_field",
    "w/o_applicability",
    "w/o_redundancy",
    "w/o_conflict",
    "w/o_reranker",
    "role_only",
    "random_role",
)

ABLATION_METHOD = {
    "frc_full": "full",
    "w/o Role": "w/o_role",
    "w/o Redundancy": "w/o_redundancy",
    "Role-only": "role_only",
    "random_role": "random_role",
    "setr_style": "coverage_greedy_proxy",
}


def display_method(method: str) -> str:
    return DISPLAY_METHOD.get(method, method)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _safe_float(value: str) -> float | None:
    parsed = float(value)
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def load_aggregate_csv(path: Path) -> dict[str, dict[str, float | int | None]]:
    rows: dict[str, dict[str, float | int | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = display_method(row["Method"])
            rows[method] = {
                "evidence_recall": _safe_float(row["Evidence Recall@5"]),
                "evidence_precision": _safe_float(row["Evidence Precision@5"]),
                "evidence_f1": _safe_float(row["Evidence F1@5"]),
                "role_coverage": _safe_float(row["Role Coverage@5"]),
                "condition_coverage": _safe_float(row["Condition Coverage"]),
                "redundancy": _safe_float(row["Redundancy"]),
                "token_cost": _safe_float(row["Token Cost"]),
                "cases": int(row["Cases"]),
            }
    return rows


def load_answer_csv(path: Path) -> dict[str, dict[str, float | int | None]]:
    rows: dict[str, dict[str, float | int | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = display_method(row["Method"])
            rows[method] = {
                "answer_em": _safe_float(row["Answer EM"]),
                "answer_f1": _safe_float(row["Answer F1"]),
                "conditional_f1": _safe_float(row["Conditional F1"]),
                "not_answerable_accuracy": _safe_float(row["Not-answerable Accuracy"]),
                "answer_accuracy": _safe_float(row["Answer Accuracy"]),
                "supporting_fact_f1": _safe_float(row["Supporting Fact F1"]),
                "joint_f1": _safe_float(row["Joint F1"]),
                "cases": int(row["Cases"]),
            }
    return rows


def load_supplemental_ablation(
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-real-model-ablation-v1":
        raise ValueError(f"unsupported supplemental ablation schema: {path}")
    variant = str(payload.get("variant", ""))
    if variant not in PLANNED_ABLATIONS:
        raise ValueError(f"unsupported supplemental ablation variant: {variant}")
    case_rows = payload.get("case_results", [])
    if not case_rows:
        raise ValueError(f"supplemental ablation has no case results: {path}")
    metadata = payload.get("metadata", {})
    if variant == "w/o_reranker" and (
        metadata.get("cross_encoder_used") is not False
        or metadata.get("scoring_backend") != "bge_biencoder_no_cross_encoder"
    ):
        raise ValueError(
            "w/o_reranker must recompute relevance and role scores without a Cross-Encoder"
        )
    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "role_coverage",
        "token_cost",
    )
    recomputed = {
        name: round(
            sum(float(row["metrics"][name]) for row in case_rows) / len(case_rows),
            6,
        )
        for name in metric_names
    }
    claimed = payload.get("aggregate", {})
    for name, value in recomputed.items():
        if not math.isclose(
            value, float(claimed.get(name, float("nan"))), abs_tol=1e-6
        ):
            raise ValueError(
                f"supplemental ablation aggregate mismatch for {name}: {path}"
            )
    metrics = {
        **recomputed,
        "condition_coverage": claimed.get("condition_coverage"),
        "redundancy": claimed.get("redundancy"),
        "cases": len(case_rows),
    }
    provenance = {
        "artifact": str(path.resolve()),
        "artifact_sha256": sha256(path),
        "metadata": metadata,
        "paired_against_full_evidence_f1": payload.get(
            "paired_w_o_reranker_minus_full_evidence_f1"
        ),
    }
    return variant, metrics, provenance


def load_controlled_domain_sensitivity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-controlled-domain-sensitivity-v1":
        raise ValueError(f"unsupported controlled sensitivity schema: {path}")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("data_origin") != "SYNTHETIC"
        or metadata.get("is_simulated") is not True
    ):
        raise ValueError("controlled sensitivity must preserve synthetic provenance")
    if metadata.get("neural_model_used") is not False:
        raise ValueError(
            "controlled sensitivity must not be presented as a real-model run"
        )
    if int(metadata.get("cases_with_field_evidence_map", 0)) != int(
        metadata.get("case_count", -1)
    ):
        raise ValueError(
            "controlled sensitivity requires an independent field-to-evidence map for every case"
        )
    results = payload.get("results", [])
    if not results:
        raise ValueError("controlled sensitivity has no result rows")
    required_dimensions = {"role_weight", "field_weight", "conflict_threshold"}
    dimensions: dict[str, set[float]] = defaultdict(set)
    aggregates_by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "field_coverage",
        "selector_field_coverage",
        "role_coverage",
        "unsupported_evidence_ratio",
        "flagged_evidence_ratio",
        "flagged_evidence_case",
        "explicit_conflict_pair_selected",
        "case_accuracy",
        "token_cost",
    )
    for row in results:
        dimension = str(row.get("dimension", ""))
        if dimension not in required_dimensions:
            raise ValueError(
                f"unsupported controlled sensitivity dimension: {dimension}"
            )
        dimensions[dimension].add(float(row["value"]))
        case_rows = row.get("case_results", [])
        if not case_rows:
            raise ValueError(f"controlled sensitivity row has no cases: {dimension}")
        aggregate = row.get("aggregate", {})
        aggregates_by_dimension[dimension].append(aggregate)
        for metric in metric_names:
            recomputed = round(
                sum(float(case["metrics"][metric]) for case in case_rows)
                / len(case_rows),
                6,
            )
            if not math.isclose(
                recomputed, float(aggregate.get(metric, float("nan"))), abs_tol=1e-6
            ):
                raise ValueError(
                    f"controlled sensitivity aggregate mismatch for {dimension}/{metric}: {path}"
                )
    if set(dimensions) != required_dimensions or any(
        len(values) < 2 for values in dimensions.values()
    ):
        raise ValueError(
            "controlled sensitivity must scan every required dimension at multiple values"
        )
    frozen_policy = metadata.get("frozen_policy", {})
    if any(
        float(frozen_policy.get(dimension, float("nan"))) not in values
        for dimension, values in dimensions.items()
    ):
        raise ValueError(
            "controlled sensitivity must include every frozen parameter value"
        )
    identifiability_metrics = {
        "role_weight": ("role_coverage", "evidence_f1"),
        "field_weight": ("field_coverage", "evidence_f1"),
        "conflict_threshold": ("flagged_evidence_case", "evidence_f1"),
    }
    identifiability: dict[str, dict[str, list[float]]] = {}
    for dimension, names in identifiability_metrics.items():
        ranges = {
            name: sorted(
                {
                    float(aggregate[name])
                    for aggregate in aggregates_by_dimension[dimension]
                }
            )
            for name in names
        }
        if all(len(values) == 1 for values in ranges.values()):
            raise ValueError(
                f"controlled sensitivity dimension is not identifiable: {dimension}"
            )
        identifiability[dimension] = ranges
    return {
        "status": "RUN_CONTROLLED_DOMAIN",
        "scope": "SYNTHETIC_CONTROLLED_NO_NEURAL_MODEL",
        "metadata": metadata,
        "dimensions": {
            key: sorted(values) for key, values in sorted(dimensions.items())
        },
        "identifiability": identifiability,
        "results": [
            {
                "dimension": row["dimension"],
                "value": row["value"],
                "aggregate": row["aggregate"],
            }
            for row in results
        ],
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def _conflicts_classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = (
        "No conflict",
        "Complementary information",
        "Conflicting opinions and research outcomes",
        "Conflict due to outdated information",
        "Conflict due to misinformation",
    )
    confusion = {label: defaultdict(int) for label in labels}
    support = defaultdict(int)
    parsed = 0
    correct = 0
    for row in rows:
        gold = str(row["conflict_type"])
        predicted = row.get("predicted_label")
        if gold not in labels:
            raise ValueError(f"unsupported CONFLICTS gold label: {gold}")
        support[gold] += 1
        if predicted in labels:
            parsed += 1
            confusion[gold][predicted] += 1
            correct += int(predicted == gold)
    per_type = {}
    f1_values = []
    for label in labels:
        true_positive = confusion[label][label]
        false_negative = support[label] - true_positive
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        per_type[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support[label],
        }
    return {
        "cases": len(rows),
        "parsed_predictions": parsed,
        "parse_rate": round(parsed / max(1, len(rows)), 6),
        "accuracy": round(correct / max(1, len(rows)), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_type": per_type,
    }


def _conflicts_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_rows = [row for row in rows if row.get("answer_annotated") is True]
    outdated_rows = [
        row
        for row in rows
        if row["conflict_type"] == "Conflict due to outdated information"
    ]
    return {
        "cases": len(rows),
        "domain_coverage": round(
            sum(float(row["metrics"]["domain_coverage"]) for row in rows)
            / max(1, len(rows)),
            6,
        ),
        "lexical_diversity": round(
            sum(float(row["metrics"]["lexical_diversity"]) for row in rows)
            / max(1, len(rows)),
            6,
        ),
        "mean_token_cost": round(
            sum(float(row["metrics"]["token_cost"]) for row in rows)
            / max(1, len(rows)),
            3,
        ),
        "answer_cases": len(answer_rows),
        "exact_answer_support": round(
            sum(float(row["metrics"]["exact_answer_support"]) for row in answer_rows)
            / max(1, len(answer_rows)),
            6,
        ),
        "answer_token_recall": round(
            sum(float(row["metrics"]["answer_token_recall"]) for row in answer_rows)
            / max(1, len(answer_rows)),
            6,
        ),
        "outdated_cases": len(outdated_rows),
        "newest_date_retention": round(
            sum(float(row["metrics"]["newest_date_retention"]) for row in outdated_rows)
            / max(1, len(outdated_rows)),
            6,
        ),
        "temporal_endpoint_coverage": round(
            sum(
                float(row["metrics"]["temporal_endpoint_coverage"])
                for row in outdated_rows
            )
            / max(1, len(outdated_rows)),
            6,
        ),
    }


def load_conflicts_real_model_ablation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-conflicts-real-model-ablation-v1":
        raise ValueError(f"unsupported CONFLICTS ablation schema: {path}")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("data_origin") != "PUBLIC"
        or metadata.get("is_simulated") is not False
    ):
        raise ValueError(
            "CONFLICTS ablation must preserve public non-simulated provenance"
        )
    if (
        metadata.get("real_model_scores") is not True
        or metadata.get("generator") != "local_qwen"
    ):
        raise ValueError(
            "CONFLICTS ablation must use real-model scores and local Qwen generation"
        )
    case_count = int(metadata.get("case_count", 0))
    if case_count != 458:
        raise ValueError("CONFLICTS ablation must cover all 458 public cases")
    thresholds = [float(value) for value in metadata.get("conflict_thresholds", [])]
    if len(thresholds) < 2 or 0.55 not in thresholds:
        raise ValueError(
            "CONFLICTS ablation must scan multiple thresholds including 0.55"
        )
    expected_methods = {
        "frc_full",
        "wo_conflict",
        *(f"conflict_threshold_{value:.2f}" for value in thresholds),
    }
    case_artifact = payload.get("case_results_artifact", {})
    if case_artifact.get("format") != "gzip-jsonl":
        raise ValueError(
            "CONFLICTS ablation case results must use deterministic gzip JSONL"
        )
    case_path = path.parent / str(case_artifact.get("file", ""))
    if not case_path.is_file() or sha256(case_path) != case_artifact.get("sha256"):
        raise ValueError(
            "CONFLICTS ablation case-result artifact is missing or hash-mismatched"
        )
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        case_results = [json.loads(line) for line in handle if line.strip()]
    if len(case_results) != int(case_artifact.get("rows", -1)):
        raise ValueError("CONFLICTS ablation case-result row count mismatch")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in case_results:
        key = (str(row["case_id"]), str(row["method"]))
        if key in seen:
            raise ValueError(f"duplicate CONFLICTS ablation case row: {key}")
        seen.add(key)
        grouped[key[1]].append(row)
    if set(grouped) != expected_methods or any(
        len(rows) != case_count for rows in grouped.values()
    ):
        raise ValueError("CONFLICTS ablation must cover every case for every variant")
    claimed = {row["method"]: row for row in payload.get("aggregates", [])}
    if set(claimed) != expected_methods:
        raise ValueError("CONFLICTS ablation aggregate method coverage mismatch")
    proxy_names = (
        "prediction_parsed",
        "classification_correct",
        "alternative_claim_coverage_proxy",
        "temporal_validity_coverage_proxy",
        "conflict_role_coverage_proxy",
    )
    for method, rows in grouped.items():
        classification = _conflicts_classification_summary(rows)
        selection = _conflicts_selection_summary(rows)
        proxies = {
            name: round(
                sum(float(row["metrics"][name]) for row in rows) / len(rows),
                6,
            )
            for name in proxy_names
        }
        if classification != claimed[method].get("classification"):
            raise ValueError(f"CONFLICTS classification aggregate mismatch: {method}")
        if selection != claimed[method].get("selection"):
            raise ValueError(f"CONFLICTS selection aggregate mismatch: {method}")
        if proxies != claimed[method].get("coverage_proxies"):
            raise ValueError(f"CONFLICTS coverage aggregate mismatch: {method}")
    ablation = payload.get("ablation", {})
    if ablation.get("status") != "RUN_REAL_MODEL_CONFLICTS":
        raise ValueError("CONFLICTS w/o Conflict ablation status is not complete")
    if ablation.get("full") != claimed["frc_full"]:
        raise ValueError("CONFLICTS Full summary does not match reaggregated cases")
    if ablation.get("without_conflict") != claimed["wo_conflict"]:
        raise ValueError(
            "CONFLICTS w/o Conflict summary does not match reaggregated cases"
        )
    sensitivity = payload.get("conflict_threshold_sensitivity", {})
    if sensitivity.get("status") != "RUN_REAL_MODEL_CONFLICTS":
        raise ValueError(
            "CONFLICTS conflict-threshold sensitivity status is not complete"
        )
    sensitivity_rows = sensitivity.get("results", [])
    if [float(row["conflict_threshold"]) for row in sensitivity_rows] != thresholds:
        raise ValueError("CONFLICTS conflict-threshold result ordering mismatch")
    for row in sensitivity_rows:
        method = f"conflict_threshold_{float(row['conflict_threshold']):.2f}"
        expected = {
            "conflict_threshold": float(row["conflict_threshold"]),
            **claimed[method],
        }
        if row != expected:
            raise ValueError(f"CONFLICTS threshold summary mismatch: {method}")
    full = {row["case_id"]: row for row in grouped["frc_full"]}
    without = {row["case_id"]: row for row in grouped["wo_conflict"]}
    changed = sum(
        full[case_id]["selected_ids"] != without[case_id]["selected_ids"]
        for case_id in full
    )
    comparison = payload["ablation"]["comparison"]
    if int(comparison.get("selection_changed_cases", -1)) != changed:
        raise ValueError("CONFLICTS ablation changed-selection count mismatch")
    if int(comparison.get("selection_unchanged_cases", -1)) != case_count - changed:
        raise ValueError("CONFLICTS ablation unchanged-selection count mismatch")
    correctness_differences = [
        float(full[case_id]["metrics"]["classification_correct"])
        - float(without[case_id]["metrics"]["classification_correct"])
        for case_id in sorted(full)
    ]
    recomputed_ci = paired_bootstrap(correctness_differences, resamples=2000)
    if recomputed_ci != comparison["full_minus_wo_conflict"]["classification_accuracy"]:
        raise ValueError("CONFLICTS ablation paired classification interval mismatch")
    source_hashes = metadata.get("source_sha256", {})
    if not source_hashes or any(
        len(str(value)) != 64 for value in source_hashes.values()
    ):
        raise ValueError("CONFLICTS ablation source hashes are incomplete")
    decision = payload.get("decision", {})
    if decision.get("gate_2") != "NO-GO":
        raise ValueError("CONFLICTS ablation must not independently promote Gate 2")
    if decision.get("full_superiority_over_wo_conflict_proven") is not (
        float(recomputed_ci["ci_low"]) > 0.0
    ):
        raise ValueError("CONFLICTS ablation superiority decision mismatch")
    return {
        "status": "RUN_REAL_MODEL_CONFLICTS",
        "metadata": metadata,
        "ablation": ablation,
        "conflict_threshold_sensitivity": sensitivity,
        "decision": decision,
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def load_housing_real_model_ablation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-housing-field-applicability-ablation-v1":
        raise ValueError(f"unsupported HousingQA ablation schema: {path}")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("public_dataset") is not True
        or metadata.get("expert_annotated_supporting_statutes") is not True
        or metadata.get("real_model_scores") is not True
        or metadata.get("generator") != "local_qwen"
    ):
        raise ValueError(
            "HousingQA ablation provenance or real-model status is invalid"
        )
    if metadata.get("source_repository") != "reglab/housing_qa":
        raise ValueError("HousingQA ablation source repository mismatch")
    if metadata.get("source_revision") != "761550cc974fa1d9141ffd39014db89efa2a7230":
        raise ValueError("HousingQA ablation source revision mismatch")
    if metadata.get("source_license") != "CC-BY-SA-4.0":
        raise ValueError("HousingQA ablation license provenance mismatch")
    expected_source_hashes = {
        "source_json_sha256": "4f6eb35e1b865a6c5cfd0cf73a0b514a86bd1d8306bf6719f2b4fb94793c7e4b",
        "source_zip_sha256": "7e7722c267d44ecc1f9f52d69109116d8a0a352b89f002438585872f67de9985",
    }
    for name, expected in expected_source_hashes.items():
        if metadata.get(name) != expected:
            raise ValueError(f"HousingQA ablation pinned source hash mismatch: {name}")
    case_count = int(metadata.get("case_count", 0))
    field_count = int(metadata.get("field_count", 0))
    if case_count != 40 or field_count != 160:
        raise ValueError(
            "HousingQA ablation must cover 40 composite cases and 160 fields"
        )
    if int(metadata.get("jurisdiction_count", 0)) < 20:
        raise ValueError(
            "HousingQA ablation must preserve broad multi-jurisdiction coverage"
        )
    if int(metadata.get("snapshot_year", 0)) != 2021:
        raise ValueError(
            "HousingQA ablation must preserve the official 2021 snapshot boundary"
        )

    expected_methods = {
        "bm25_topk",
        "dense_topk",
        "hybrid_topk",
        "cross_encoder_topk",
        "field_decomposition_topk",
        "frc_full",
        "w/o_field",
        "w/o_applicability",
    }
    if set(metadata.get("methods", [])) != expected_methods:
        raise ValueError("HousingQA ablation method coverage mismatch")
    case_artifact = payload.get("case_results_artifact", {})
    if case_artifact.get("format") != "gzip-jsonl":
        raise ValueError(
            "HousingQA ablation case results must use deterministic gzip JSONL"
        )
    case_path = path.parent / str(case_artifact.get("file", ""))
    if not case_path.is_file() or sha256(case_path) != case_artifact.get("sha256"):
        raise ValueError(
            "HousingQA ablation case-result artifact is missing or hash-mismatched"
        )
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        case_results = [json.loads(line) for line in handle if line.strip()]
    if len(case_results) != int(case_artifact.get("rows", -1)):
        raise ValueError("HousingQA ablation case-result row count mismatch")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "field_coverage",
        "core_field_coverage",
        "citation_support_precision",
        "unsupported_field_rate",
        "wrong_jurisdiction_rate",
        "wrong_jurisdiction_case",
        "answer_accuracy",
        "valid_answer_rate",
        "case_accuracy",
        "token_cost",
    )
    for row in case_results:
        key = (str(row["case_id"]), str(row["method"]))
        if key in seen:
            raise ValueError(f"duplicate HousingQA ablation case row: {key}")
        seen.add(key)
        if set(row.get("metrics", {})) != set(metric_names):
            raise ValueError(f"HousingQA case metric coverage mismatch: {key}")
        if len(row.get("expected_answers", {})) != 4:
            raise ValueError(
                f"HousingQA composite case does not contain four fields: {key}"
            )
        grouped[key[1]].append(row)
    if set(grouped) != expected_methods or any(
        len(rows) != case_count for rows in grouped.values()
    ):
        raise ValueError("HousingQA ablation must cover every case for every method")
    prediction_provenance = metadata.get("prediction_provenance", {})
    unique_selection_prompts = len(
        {
            (str(row["case_id"]), tuple(str(value) for value in row["selected_ids"]))
            for row in case_results
        }
    )
    if (
        int(prediction_provenance.get("total_method_rows", -1)) != len(case_results)
        or int(prediction_provenance.get("unique_case_selection_prompts", -1))
        != unique_selection_prompts
        or int(prediction_provenance.get("equivalent_selection_reuse_rows", -1))
        != len(case_results) - unique_selection_prompts
    ):
        raise ValueError("HousingQA generation provenance mismatch")
    if any(
        len(str(metadata.get(name, ""))) != 64
        for name in ("manifest_sha256", "scored_cases_sha256")
    ):
        raise ValueError("HousingQA scored-case or manifest provenance is incomplete")

    claimed = payload.get("aggregates", {})
    if set(claimed) != expected_methods:
        raise ValueError("HousingQA aggregate method coverage mismatch")
    for method, rows in grouped.items():
        recomputed = {
            "cases": len(rows),
            **{
                name: round(
                    sum(float(row["metrics"][name]) for row in rows) / len(rows),
                    6,
                )
                for name in metric_names
            },
        }
        if recomputed != claimed[method]:
            raise ValueError(f"HousingQA aggregate mismatch: {method}")

    comparisons = payload.get("paired_comparisons", {})
    expected_comparisons = {
        "full_minus_w_o_field": ("frc_full", "w/o_field"),
        "full_minus_w_o_applicability": ("frc_full", "w/o_applicability"),
        "full_minus_strongest_baseline": (
            "frc_full",
            str(payload.get("strongest_baseline_by_field_coverage")),
        ),
    }
    paired_metrics = (
        "evidence_f1",
        "field_coverage",
        "citation_support_precision",
        "answer_accuracy",
        "case_accuracy",
        "wrong_jurisdiction_rate",
    )
    for comparison_name, (left_method, right_method) in expected_comparisons.items():
        comparison = comparisons.get(comparison_name, {})
        if (
            comparison.get("left") != left_method
            or comparison.get("right") != right_method
            or comparison.get("direction") != "left_minus_right"
        ):
            raise ValueError(
                f"HousingQA paired comparison definition mismatch: {comparison_name}"
            )
        left = {row["case_id"]: row for row in grouped[left_method]}
        right = {row["case_id"]: row for row in grouped[right_method]}
        if set(left) != set(right):
            raise ValueError(
                f"HousingQA paired case coverage mismatch: {comparison_name}"
            )
        for metric in paired_metrics:
            recomputed = paired_bootstrap(
                [
                    float(left[case_id]["metrics"][metric])
                    - float(right[case_id]["metrics"][metric])
                    for case_id in sorted(left)
                ]
            )
            if recomputed != comparison.get("metrics", {}).get(metric):
                raise ValueError(
                    f"HousingQA paired metric mismatch: {comparison_name}/{metric}"
                )

    baseline_methods = {
        "bm25_topk",
        "dense_topk",
        "hybrid_topk",
        "cross_encoder_topk",
        "field_decomposition_topk",
    }
    strongest = max(
        baseline_methods,
        key=lambda method: (
            float(claimed[method]["field_coverage"]),
            float(claimed[method]["answer_accuracy"]),
            method,
        ),
    )
    if strongest != payload.get("strongest_baseline_by_field_coverage"):
        raise ValueError("HousingQA strongest-baseline selection mismatch")
    full_rows = {row["case_id"]: row for row in grouped["frc_full"]}
    for variant in ("w/o_field", "w/o_applicability"):
        variant_rows = {row["case_id"]: row for row in grouped[variant]}
        changed = sum(
            full_rows[case_id]["selected_ids"] != variant_rows[case_id]["selected_ids"]
            for case_id in full_rows
        )
        if int(payload.get("selection_changed_cases", {}).get(variant, -1)) != changed:
            raise ValueError(f"HousingQA changed-selection count mismatch: {variant}")

    coverage = payload.get("coverage", {})
    if coverage.get("w/o_field") != "RUN_PUBLIC_EXPERT_REAL_MODEL":
        raise ValueError("HousingQA w/o Field coverage is not complete")
    if (
        coverage.get("w/o_applicability")
        != "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
    ):
        raise ValueError(
            "HousingQA jurisdiction applicability coverage is not complete"
        )
    if (
        coverage.get("version_or_expiry_variation")
        != "NOT_IDENTIFIABLE_SINGLE_SNAPSHOT"
    ):
        raise ValueError(
            "HousingQA must retain its single-snapshot temporal limitation"
        )
    decision = payload.get("decision", {})
    field_difference = comparisons["full_minus_w_o_field"]["metrics"]["field_coverage"]
    baseline_difference = comparisons["full_minus_strongest_baseline"]["metrics"][
        "field_coverage"
    ]
    if decision.get("full_strictly_better_than_w_o_field") is not (
        float(field_difference["mean_difference"]) > 0.0
    ):
        raise ValueError("HousingQA Full versus w/o Field decision mismatch")
    if decision.get("full_w_o_field_ci_excludes_zero") is not (
        float(field_difference["ci_low"]) > 0.0
    ):
        raise ValueError("HousingQA w/o Field confidence decision mismatch")
    if decision.get(
        "full_field_coverage_gain_over_strongest_baseline_at_least_0_05"
    ) is not (float(baseline_difference["mean_difference"]) >= 0.05):
        raise ValueError("HousingQA strongest-baseline gain decision mismatch")
    if decision.get("gate_2") != "NO-GO":
        raise ValueError("HousingQA ablation must not independently promote Gate 2")
    return {
        "status": "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021",
        "metadata": metadata,
        "coverage": coverage,
        "aggregates": claimed,
        "strongest_baseline": strongest,
        "paired_comparisons": comparisons,
        "selection_changed_cases": payload.get("selection_changed_cases", {}),
        "decision": decision,
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def load_housing_public_weight_sensitivity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_status = "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
    if payload.get("schema_version") != "frc-housing-public-weight-sensitivity-v1":
        raise ValueError(f"unsupported HousingQA weight-sensitivity schema: {path}")
    if payload.get("status") != expected_status:
        raise ValueError("HousingQA public weight sensitivity status is incomplete")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("data_origin") != "PUBLIC"
        or metadata.get("is_simulated") is not False
        or metadata.get("public_dataset") is not True
        or metadata.get("expert_annotated_supporting_statutes") is not True
        or metadata.get("real_model_scores") is not True
        or metadata.get("generator_used") is not False
    ):
        raise ValueError("HousingQA public weight-sensitivity provenance is invalid")
    if metadata.get("source_repository") != "reglab/housing_qa":
        raise ValueError("HousingQA weight-sensitivity source repository mismatch")
    if metadata.get("source_revision") != "761550cc974fa1d9141ffd39014db89efa2a7230":
        raise ValueError("HousingQA weight-sensitivity source revision mismatch")
    if metadata.get("source_license") != "CC-BY-SA-4.0":
        raise ValueError("HousingQA weight-sensitivity license mismatch")
    if metadata.get("source_json_sha256") != (
        "4f6eb35e1b865a6c5cfd0cf73a0b514a86bd1d8306bf6719f2b4fb94793c7e4b"
    ):
        raise ValueError("HousingQA weight-sensitivity source hash mismatch")
    if (
        int(metadata.get("case_count", 0)) != 40
        or int(metadata.get("field_count", 0)) != 160
        or int(metadata.get("jurisdiction_count", 0)) < 20
        or metadata.get("role_count_per_case") != [3]
        or int(metadata.get("candidate_occurrences", 0)) != 400
    ):
        raise ValueError(
            "HousingQA weight sensitivity has incomplete benchmark coverage"
        )
    for name in ("scored_cases_sha256", "scored_metadata_sha256", "case_signature"):
        value = metadata.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(
                f"HousingQA weight-sensitivity provenance hash is invalid: {name}"
            )
    if metadata.get("models") != {
        "embedding": "BAAI/bge-large-en-v1.5",
        "reranker": "BAAI/bge-reranker-large",
    }:
        raise ValueError(
            "HousingQA weight sensitivity must use the frozen real-model stack"
        )
    expected_dimensions = {
        "field_weight": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        "role_weight": [0.0, 0.5, 1.0, 2.0, 4.0],
    }
    dimensions = {
        str(name): [float(value) for value in values]
        for name, values in payload.get("dimensions", {}).items()
    }
    if dimensions != expected_dimensions:
        raise ValueError(
            "HousingQA weight sensitivity does not cover the frozen scan grid"
        )
    frozen = metadata.get("frozen_parameters", {})
    frozen_values = {
        "field_weight": float(frozen.get("field_weight", float("nan"))),
        "role_weight": float(frozen.get("role_weight", float("nan"))),
    }
    if frozen_values != {"field_weight": 2.0, "role_weight": 1.0}:
        raise ValueError(
            "HousingQA weight sensitivity changed the frozen selection policy"
        )

    case_artifact = payload.get("case_results_artifact", {})
    if case_artifact.get("format") != "gzip-jsonl":
        raise ValueError(
            "HousingQA weight sensitivity requires deterministic gzip case rows"
        )
    case_path = path.parent / str(case_artifact.get("file", ""))
    if not case_path.is_file() or sha256(case_path) != case_artifact.get("sha256"):
        raise ValueError(
            "HousingQA weight-sensitivity case artifact is missing or hash-mismatched"
        )
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        case_results = [json.loads(line) for line in handle if line.strip()]
    expected_rows = 40 * sum(len(values) for values in expected_dimensions.values())
    if (
        len(case_results) != expected_rows
        or int(case_artifact.get("rows", -1)) != expected_rows
    ):
        raise ValueError(
            "HousingQA weight sensitivity has incomplete case-result coverage"
        )

    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "field_coverage",
        "selector_field_coverage",
        "role_coverage",
        "unsupported_field_rate",
        "wrong_jurisdiction_rate",
        "token_cost",
        "selected_count",
    )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, float, str]] = set()
    for row in case_results:
        dimension = str(row.get("dimension", ""))
        value = float(row.get("value", float("nan")))
        case_id = str(row.get("case_id", ""))
        key = (dimension, value, case_id)
        if (
            dimension not in expected_dimensions
            or value not in expected_dimensions[dimension]
            or not case_id
            or key in seen
        ):
            raise ValueError(f"invalid or duplicate HousingQA weight case row: {key}")
        seen.add(key)
        if set(row.get("metrics", {})) != set(metric_names):
            raise ValueError(f"HousingQA weight case metric coverage mismatch: {key}")
        if float(row["metrics"]["wrong_jurisdiction_rate"]) != 0.0:
            raise ValueError(
                "HousingQA applicable selector retained a wrong-jurisdiction item"
            )
        grouped[(dimension, value)].append(row)
    if any(
        len(grouped[(dimension, value)]) != 40
        for dimension, values in expected_dimensions.items()
        for value in values
    ):
        raise ValueError(
            "HousingQA weight sensitivity must cover all 40 cases per setting"
        )

    summaries = {
        (str(row["dimension"]), float(row["value"])): row
        for row in payload.get("results", [])
    }
    expected_keys = {
        (dimension, value)
        for dimension, values in expected_dimensions.items()
        for value in values
    }
    if set(summaries) != expected_keys:
        raise ValueError("HousingQA weight-sensitivity summary grid is incomplete")
    aggregates: dict[tuple[str, float], dict[str, float | int]] = {}
    for key in sorted(expected_keys):
        rows = grouped[key]
        aggregate = {
            "cases": len(rows),
            **{
                metric: round(
                    sum(float(row["metrics"][metric]) for row in rows) / len(rows),
                    6,
                )
                for metric in metric_names
            },
        }
        if aggregate != summaries[key].get("aggregate"):
            raise ValueError(f"HousingQA weight aggregate mismatch: {key}")
        frozen_key = (key[0], frozen_values[key[0]])
        frozen_rows = {row["case_id"]: row for row in grouped[frozen_key]}
        changed = sum(
            row["selected_ids"] != frozen_rows[row["case_id"]]["selected_ids"]
            for row in rows
        )
        if (
            int(summaries[key].get("selection_changed_cases_from_frozen", -1))
            != changed
        ):
            raise ValueError(
                f"HousingQA weight changed-selection count mismatch: {key}"
            )
        if summaries[key].get("is_frozen") is not (key[1] == frozen_values[key[0]]):
            raise ValueError(f"HousingQA weight frozen-setting marker mismatch: {key}")
        aggregates[key] = aggregate

    recomputed_identifiability = {}
    for dimension, values in expected_dimensions.items():
        metric_names_for_dimension = (
            ("field_coverage", "evidence_f1")
            if dimension == "field_weight"
            else ("role_coverage", "evidence_f1")
        )
        ranges = {
            metric: sorted(
                {float(aggregates[(dimension, value)][metric]) for value in values}
            )
            for metric in metric_names_for_dimension
        }
        if all(len(metric_values) == 1 for metric_values in ranges.values()):
            raise ValueError(
                f"HousingQA public weight dimension is not identifiable: {dimension}"
            )
        recomputed_identifiability[dimension] = {
            "status": "IDENTIFIABLE",
            "metric_values": ranges,
        }
    if payload.get("identifiability") != recomputed_identifiability:
        raise ValueError(
            "HousingQA weight-sensitivity identifiability summary mismatch"
        )
    decision = payload.get("decision", {})
    if (
        decision.get("gate_2") != "NO-GO"
        or decision.get("public_real_model_weight_sensitivity_complete") is not True
        or decision.get("frozen_parameters_changed") is not False
    ):
        raise ValueError("HousingQA weight-sensitivity decision boundary is invalid")
    return {
        "status": expected_status,
        "metadata": metadata,
        "dimensions": expected_dimensions,
        "identifiability": recomputed_identifiability,
        "results": [
            {
                "dimension": dimension,
                "value": value,
                "aggregate": aggregates[(dimension, value)],
                "selection_changed_cases_from_frozen": summaries[(dimension, value)][
                    "selection_changed_cases_from_frozen"
                ],
            }
            for dimension, values in expected_dimensions.items()
            for value in values
        ],
        "decision": decision,
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def load_lawshift_temporal_ablation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "frc-lawshift-temporal-applicability-ablation-v1"
    ):
        raise ValueError(f"unsupported LawShift temporal ablation schema: {path}")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("public_dataset") is not True
        or metadata.get("expert_annotated_revisions") is not True
        or metadata.get("real_model_scores") is not True
    ):
        raise ValueError("LawShift temporal ablation provenance is invalid")
    if metadata.get("source_repository") != "triangularPeach/LawShift":
        raise ValueError("LawShift source repository mismatch")
    if metadata.get("source_revision") != "0fce4f3821140bde29081ae0b20500e79aa065d5":
        raise ValueError("LawShift source revision mismatch")
    if metadata.get("source_license") != "Apache-2.0":
        raise ValueError("LawShift source license mismatch")
    if (
        int(metadata.get("source_file_count", 0)) != 124
        or len(str(metadata.get("source_manifest_sha256", ""))) != 64
        or len(str(metadata.get("scored_cases_sha256", ""))) != 64
        or len(str(metadata.get("case_manifest_sha256", ""))) != 64
    ):
        raise ValueError("LawShift source or scored-case provenance is incomplete")
    case_count = int(metadata.get("case_count", 0))
    if (
        case_count != 124
        or int(metadata.get("paired_case_count", 0)) != 62
        or int(metadata.get("revision_type_count", 0)) != 31
        or int(metadata.get("original_case_count", 0)) != 62
        or int(metadata.get("revised_case_count", 0)) != 62
        or int(metadata.get("candidate_occurrences", 0)) != 992
    ):
        raise ValueError("LawShift frozen case coverage mismatch")
    expected_methods = {
        "char_bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
        "frc_full",
        "w/o_applicability",
    }
    if set(metadata.get("methods", [])) != expected_methods:
        raise ValueError("LawShift method coverage mismatch")
    parameters = metadata.get("selection_parameters", {})
    if (
        int(parameters.get("top_k", 0)) != 1
        or int(parameters.get("token_budget", 0)) != 512
        or int(parameters.get("pairs_per_revision", 0)) != 2
        or int(parameters.get("distractor_articles", 0)) != 3
        or int(parameters.get("seed", 0)) != 20260713
    ):
        raise ValueError("LawShift frozen selection parameters mismatch")

    case_artifact = payload.get("case_results_artifact", {})
    if case_artifact.get("format") != "gzip-jsonl":
        raise ValueError("LawShift case results must use deterministic gzip JSONL")
    case_path = path.parent / str(case_artifact.get("file", ""))
    if not case_path.is_file() or sha256(case_path) != case_artifact.get("sha256"):
        raise ValueError("LawShift case-result artifact is missing or hash-mismatched")
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        case_results = [json.loads(line) for line in handle if line.strip()]
    if len(case_results) != int(case_artifact.get("rows", -1)):
        raise ValueError("LawShift case-result row count mismatch")

    metric_names = {
        "article_recall_at_1",
        "version_accuracy",
        "exact_evidence_accuracy",
        "invalid_applicability_rate",
        "token_cost",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    case_versions: dict[str, str] = {}
    case_revision_types: dict[str, str] = {}
    for row in case_results:
        key = (str(row["case_id"]), str(row["method"]))
        if key in seen:
            raise ValueError(f"duplicate LawShift case row: {key}")
        seen.add(key)
        if set(row.get("metrics", {})) != metric_names:
            raise ValueError(f"LawShift case metric coverage mismatch: {key}")
        version = str(row.get("as_of_version"))
        if version not in {"original", "revised"}:
            raise ValueError(f"LawShift case version mismatch: {key}")
        case_versions.setdefault(key[0], version)
        if case_versions[key[0]] != version:
            raise ValueError(f"LawShift case version changed across methods: {key[0]}")
        revision_type = str(row.get("revision_type"))
        case_revision_types.setdefault(key[0], revision_type)
        if case_revision_types[key[0]] != revision_type:
            raise ValueError(f"LawShift revision type changed across methods: {key[0]}")
        grouped[key[1]].append(row)
    if set(grouped) != expected_methods or any(
        len(rows) != case_count for rows in grouped.values()
    ):
        raise ValueError("LawShift must cover every case for every method")
    if len(set(case_revision_types.values())) != 31:
        raise ValueError("LawShift revision-type coverage is incomplete")
    if Counter(case_versions.values()) != Counter({"original": 62, "revised": 62}):
        raise ValueError("LawShift before/after case balance mismatch")
    reference_rows = grouped["frc_full"]
    paired_snapshots: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in reference_rows:
        paired_snapshots[
            (str(row["revision_type"]), int(row["source_case_index"]))
        ].add(str(row["as_of_version"]))
    if (
        len(paired_snapshots) != 62
        or any(
            versions != {"original", "revised"}
            for versions in paired_snapshots.values()
        )
        or Counter(key[0] for key in paired_snapshots)
        != Counter(
            {revision_type: 2 for revision_type in set(case_revision_types.values())}
        )
    ):
        raise ValueError("LawShift paired before/after source-case structure mismatch")

    claimed = payload.get("aggregates", {})
    if set(claimed) != expected_methods:
        raise ValueError("LawShift aggregate method coverage mismatch")
    for method, rows in grouped.items():
        recomputed = {
            "cases": len(rows),
            **{
                name: round(
                    sum(float(row["metrics"][name]) for row in rows) / len(rows),
                    6,
                )
                for name in metric_names
            },
        }
        if recomputed != claimed[method]:
            raise ValueError(f"LawShift aggregate mismatch: {method}")
    claimed_by_revision = {
        str(row["revision_type"]): row for row in payload.get("by_revision_type", [])
    }
    if set(claimed_by_revision) != set(case_revision_types.values()):
        raise ValueError("LawShift by-revision coverage mismatch")
    for revision_type, claimed_revision in claimed_by_revision.items():
        full_revision = [
            row for row in grouped["frc_full"] if row["revision_type"] == revision_type
        ]
        without_revision = [
            row
            for row in grouped["w/o_applicability"]
            if row["revision_type"] == revision_type
        ]
        recomputed_revision = {
            "revision_type": revision_type,
            "cases": len(full_revision),
            "frc_full_exact_evidence_accuracy": round(
                sum(
                    float(row["metrics"]["exact_evidence_accuracy"])
                    for row in full_revision
                )
                / len(full_revision),
                6,
            ),
            "w_o_applicability_exact_evidence_accuracy": round(
                sum(
                    float(row["metrics"]["exact_evidence_accuracy"])
                    for row in without_revision
                )
                / len(without_revision),
                6,
            ),
        }
        if recomputed_revision != claimed_revision:
            raise ValueError(
                f"LawShift by-revision aggregate mismatch: {revision_type}"
            )

    baseline_methods = {
        "char_bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
    }
    strongest = max(
        baseline_methods,
        key=lambda method: (
            float(claimed[method]["exact_evidence_accuracy"]),
            float(claimed[method]["article_recall_at_1"]),
            method,
        ),
    )
    if strongest != payload.get("strongest_baseline_by_exact_evidence_accuracy"):
        raise ValueError("LawShift strongest-baseline selection mismatch")
    comparisons = payload.get("paired_comparisons", {})
    expected_comparisons = {
        "full_minus_w_o_applicability": ("frc_full", "w/o_applicability"),
        "full_minus_strongest_baseline": ("frc_full", strongest),
    }
    paired_metrics = (
        "article_recall_at_1",
        "version_accuracy",
        "exact_evidence_accuracy",
        "invalid_applicability_rate",
    )
    for comparison_name, (left_method, right_method) in expected_comparisons.items():
        comparison = comparisons.get(comparison_name, {})
        if (
            comparison.get("left") != left_method
            or comparison.get("right") != right_method
            or comparison.get("direction") != "left_minus_right"
        ):
            raise ValueError(
                f"LawShift comparison definition mismatch: {comparison_name}"
            )
        left = {row["case_id"]: row for row in grouped[left_method]}
        right = {row["case_id"]: row for row in grouped[right_method]}
        for metric in paired_metrics:
            recomputed = paired_bootstrap(
                [
                    float(left[case_id]["metrics"][metric])
                    - float(right[case_id]["metrics"][metric])
                    for case_id in sorted(left)
                ],
                seed=20260713,
            )
            if recomputed != comparison.get("metrics", {}).get(metric):
                raise ValueError(
                    f"LawShift paired metric mismatch: {comparison_name}/{metric}"
                )
    full_rows = {row["case_id"]: row for row in grouped["frc_full"]}
    without_rows = {row["case_id"]: row for row in grouped["w/o_applicability"]}
    changed = sum(
        full_rows[case_id]["selected_ids"] != without_rows[case_id]["selected_ids"]
        for case_id in full_rows
    )
    if int(payload.get("selection_changed_cases", -1)) != changed:
        raise ValueError("LawShift changed-selection count mismatch")
    coverage = payload.get("coverage", {})
    if (
        coverage.get("version_replacement")
        != "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
        or coverage.get("original_and_revised_snapshots") != "IDENTIFIABLE"
        or coverage.get("effective_or_expiry_dates")
        != "NOT_IDENTIFIABLE_NO_EFFECTIVE_DATES"
    ):
        raise ValueError("LawShift temporal coverage boundary mismatch")
    decision = payload.get("decision", {})
    applicability_difference = comparisons["full_minus_w_o_applicability"]["metrics"][
        "exact_evidence_accuracy"
    ]
    baseline_difference = comparisons["full_minus_strongest_baseline"]["metrics"][
        "exact_evidence_accuracy"
    ]
    if decision.get("full_strictly_better_than_w_o_applicability") is not (
        float(applicability_difference["mean_difference"]) > 0.0
    ):
        raise ValueError("LawShift applicability superiority decision mismatch")
    if decision.get("full_exact_gain_over_strongest_baseline_at_least_0_05") is not (
        float(baseline_difference["mean_difference"]) >= 0.05
    ):
        raise ValueError("LawShift strongest-baseline decision mismatch")
    if decision.get("gate_2") != "NO-GO":
        raise ValueError("LawShift ablation must not independently promote Gate 2")
    return {
        "status": "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL",
        "metadata": metadata,
        "coverage": coverage,
        "aggregates": claimed,
        "by_revision_type": payload.get("by_revision_type", []),
        "strongest_baseline": strongest,
        "paired_comparisons": comparisons,
        "selection_changed_cases": changed,
        "decision": decision,
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def load_eurlex_temporal_ablation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-eurlex-effective-expiry-ablation-v1":
        raise ValueError(f"unsupported EUR-Lex temporal ablation schema: {path}")
    if (
        payload.get("dataset") != "EU Publications Office CELLAR and EUR-Lex"
        or payload.get("status") != "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
    ):
        raise ValueError("EUR-Lex dataset or run status mismatch")

    metadata = payload.get("metadata", {})
    if any(
        metadata.get(flag) is not True
        for flag in (
            "public_dataset",
            "official_temporal_metadata",
            "real_model_scores",
        )
    ):
        raise ValueError("EUR-Lex temporal provenance is invalid")
    if metadata.get("source_snapshot_date") != "2026-07-14":
        raise ValueError("EUR-Lex source snapshot mismatch")
    hash_fields = (
        "source_manifest_sha256",
        "source_query_sha256",
        "scored_cases_sha256",
        "case_manifest_sha256",
    )
    if any(len(str(metadata.get(field, ""))) != 64 for field in hash_fields):
        raise ValueError("EUR-Lex source or scored-case provenance is incomplete")
    if (
        int(metadata.get("pair_count", 0)) != 30
        or int(metadata.get("case_count", 0)) != 60
        or int(metadata.get("candidate_occurrences", 0)) != 360
        or metadata.get("family_counts")
        != {"decision": 12, "directive": 6, "regulation": 12}
    ):
        raise ValueError("EUR-Lex frozen case coverage mismatch")
    expected_methods = {
        "bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
        "frc_full",
        "w/o_applicability",
    }
    if set(metadata.get("methods", [])) != expected_methods:
        raise ValueError("EUR-Lex method coverage mismatch")
    parameters = metadata.get("selection_parameters", {})
    if (
        int(parameters.get("top_k", 0)) != 1
        or int(parameters.get("token_budget", 0)) != 512
        or int(parameters.get("distractor_pairs", 0)) != 2
        or int(parameters.get("seed", 0)) != 20260714
    ):
        raise ValueError("EUR-Lex frozen selection parameters mismatch")

    case_artifact = payload.get("case_results_artifact", {})
    if case_artifact.get("format") != "gzip-jsonl":
        raise ValueError("EUR-Lex case results must use deterministic gzip JSONL")
    case_path = path.parent / str(case_artifact.get("file", ""))
    if not case_path.is_file() or sha256(case_path) != case_artifact.get("sha256"):
        raise ValueError("EUR-Lex case-result artifact is missing or hash-mismatched")
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        case_results = [json.loads(line) for line in handle if line.strip()]
    if (
        len(case_results) != int(case_artifact.get("rows", -1))
        or len(case_results) != 300
    ):
        raise ValueError("EUR-Lex case-result row count mismatch")

    metric_names = {
        "exact_evidence_accuracy",
        "validity_accuracy",
        "invalid_applicability_rate",
        "wrong_boundary_version_rate",
        "token_cost",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    case_boundaries: dict[str, str] = {}
    case_families: dict[str, str] = {}
    for row in case_results:
        case_id = str(row.get("case_id"))
        method = str(row.get("method"))
        key = (case_id, method)
        if key in seen:
            raise ValueError(f"duplicate EUR-Lex case row: {key}")
        seen.add(key)
        if set(row.get("metrics", {})) != metric_names:
            raise ValueError(f"EUR-Lex case metric coverage mismatch: {key}")
        boundary = str(row.get("boundary"))
        family = str(row.get("family"))
        if boundary not in {"first_effective_day", "last_valid_day"}:
            raise ValueError(f"EUR-Lex boundary mismatch: {key}")
        if family not in {"decision", "directive", "regulation"}:
            raise ValueError(f"EUR-Lex family mismatch: {key}")
        case_boundaries.setdefault(case_id, boundary)
        case_families.setdefault(case_id, family)
        if case_boundaries[case_id] != boundary or case_families[case_id] != family:
            raise ValueError(f"EUR-Lex case metadata changed across methods: {case_id}")
        grouped[method].append(row)
    if set(grouped) != expected_methods or any(
        len(rows) != 60 for rows in grouped.values()
    ):
        raise ValueError("EUR-Lex must cover every case for every method")
    if Counter(case_boundaries.values()) != Counter(
        {"first_effective_day": 30, "last_valid_day": 30}
    ):
        raise ValueError("EUR-Lex boundary balance mismatch")
    if Counter(case_families.values()) != Counter(
        {"decision": 24, "directive": 12, "regulation": 24}
    ):
        raise ValueError("EUR-Lex family balance mismatch")

    claimed = payload.get("aggregates", {})
    if set(claimed) != expected_methods:
        raise ValueError("EUR-Lex aggregate method coverage mismatch")
    for method, rows in grouped.items():
        recomputed = {
            "cases": len(rows),
            **{
                name: round(
                    sum(float(row["metrics"][name]) for row in rows) / len(rows),
                    6,
                )
                for name in metric_names
            },
        }
        if recomputed != claimed[method]:
            raise ValueError(f"EUR-Lex aggregate mismatch: {method}")

    def recompute_slice(dimension: str, value: str) -> dict[str, Any]:
        full_rows = [row for row in grouped["frc_full"] if row[dimension] == value]
        without_rows = [
            row for row in grouped["w/o_applicability"] if row[dimension] == value
        ]
        return {
            dimension: value,
            "cases": len(full_rows),
            "frc_full_exact_evidence_accuracy": round(
                sum(
                    float(row["metrics"]["exact_evidence_accuracy"])
                    for row in full_rows
                )
                / len(full_rows),
                6,
            ),
            "w_o_applicability_exact_evidence_accuracy": round(
                sum(
                    float(row["metrics"]["exact_evidence_accuracy"])
                    for row in without_rows
                )
                / len(without_rows),
                6,
            ),
        }

    claimed_by_boundary = {
        str(row["boundary"]): row for row in payload.get("by_boundary", [])
    }
    if set(claimed_by_boundary) != {"first_effective_day", "last_valid_day"}:
        raise ValueError("EUR-Lex by-boundary coverage mismatch")
    for boundary, row in claimed_by_boundary.items():
        if row != recompute_slice("boundary", boundary):
            raise ValueError(f"EUR-Lex by-boundary aggregate mismatch: {boundary}")
    claimed_by_family = {
        str(row["family"]): row for row in payload.get("by_family", [])
    }
    if set(claimed_by_family) != {"decision", "directive", "regulation"}:
        raise ValueError("EUR-Lex by-family coverage mismatch")
    for family, row in claimed_by_family.items():
        if row != recompute_slice("family", family):
            raise ValueError(f"EUR-Lex by-family aggregate mismatch: {family}")

    baseline_methods = {
        "bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
    }
    strongest = max(
        baseline_methods,
        key=lambda method: (float(claimed[method]["exact_evidence_accuracy"]), method),
    )
    if strongest != payload.get("strongest_baseline_by_exact_evidence_accuracy"):
        raise ValueError("EUR-Lex strongest-baseline selection mismatch")
    comparisons = payload.get("paired_comparisons", {})
    expected_comparisons = {
        "full_minus_w_o_applicability": ("frc_full", "w/o_applicability"),
        "full_minus_strongest_baseline": ("frc_full", strongest),
    }
    paired_metrics = (
        "exact_evidence_accuracy",
        "validity_accuracy",
        "invalid_applicability_rate",
        "wrong_boundary_version_rate",
    )
    for comparison_name, (left_method, right_method) in expected_comparisons.items():
        comparison = comparisons.get(comparison_name, {})
        if (
            comparison.get("left") != left_method
            or comparison.get("right") != right_method
            or comparison.get("direction") != "left_minus_right"
        ):
            raise ValueError(
                f"EUR-Lex comparison definition mismatch: {comparison_name}"
            )
        left = {row["case_id"]: row for row in grouped[left_method]}
        right = {row["case_id"]: row for row in grouped[right_method]}
        for metric in paired_metrics:
            recomputed = paired_bootstrap(
                [
                    float(left[case_id]["metrics"][metric])
                    - float(right[case_id]["metrics"][metric])
                    for case_id in sorted(left)
                ],
                seed=20260714,
            )
            if recomputed != comparison.get("metrics", {}).get(metric):
                raise ValueError(
                    f"EUR-Lex paired metric mismatch: {comparison_name}/{metric}"
                )
    full_rows = {row["case_id"]: row for row in grouped["frc_full"]}
    without_rows = {row["case_id"]: row for row in grouped["w/o_applicability"]}
    changed = sum(
        full_rows[case_id]["selected_ids"] != without_rows[case_id]["selected_ids"]
        for case_id in full_rows
    )
    if int(payload.get("selection_changed_cases", -1)) != changed:
        raise ValueError("EUR-Lex changed-selection count mismatch")

    coverage = payload.get("coverage", {})
    if coverage != {
        "effective_dates": "IDENTIFIABLE_OFFICIAL_CELLAR_METADATA",
        "expiry_dates": "IDENTIFIABLE_OFFICIAL_CELLAR_METADATA",
        "adjacent_repeal_boundary": "IDENTIFIABLE",
        "old_and_new_boundary_snapshots": "RUN",
    }:
        raise ValueError("EUR-Lex temporal coverage boundary mismatch")
    decision = payload.get("decision", {})
    applicability_difference = comparisons["full_minus_w_o_applicability"]["metrics"][
        "exact_evidence_accuracy"
    ]
    baseline_difference = comparisons["full_minus_strongest_baseline"]["metrics"][
        "exact_evidence_accuracy"
    ]
    if decision.get("full_strictly_better_than_w_o_applicability") is not (
        float(applicability_difference["mean_difference"]) > 0.0
    ):
        raise ValueError("EUR-Lex applicability superiority decision mismatch")
    if decision.get("full_exact_gain_over_strongest_baseline_at_least_0_05") is not (
        float(baseline_difference["mean_difference"]) >= 0.05
    ):
        raise ValueError("EUR-Lex strongest-baseline decision mismatch")
    if decision.get("gate_2") != "NO-GO":
        raise ValueError("EUR-Lex ablation must not independently promote Gate 2")
    return {
        "status": "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL",
        "metadata": metadata,
        "coverage": coverage,
        "aggregates": claimed,
        "by_boundary": payload.get("by_boundary", []),
        "by_family": payload.get("by_family", []),
        "strongest_baseline": strongest,
        "paired_comparisons": comparisons,
        "selection_changed_cases": changed,
        "decision": decision,
        "limitations": payload.get("limitations", []),
        "source_sha256": sha256(path),
    }


def load_ablation_audit(
    path: Path,
    supplemental_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    variants: dict[str, dict[str, float | int | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = ABLATION_METHOD.get(row["Method"], row["Method"])
            variants[method] = {
                "evidence_recall": _safe_float(row["Evidence Recall@5"]),
                "evidence_precision": _safe_float(row["Evidence Precision@5"]),
                "evidence_f1": _safe_float(row["Evidence F1@5"]),
                "role_coverage": _safe_float(row["Role Coverage@5"]),
                "condition_coverage": _safe_float(row["Condition Coverage"]),
                "redundancy": _safe_float(row["Redundancy"]),
                "token_cost": _safe_float(row["Token Cost"]),
                "cases": int(row["Cases"]),
            }
    supplemental_sources: dict[str, Any] = {}
    for supplemental_path in supplemental_paths:
        variant, metrics, provenance = load_supplemental_ablation(supplemental_path)
        if variant in variants:
            raise ValueError(
                f"duplicate ablation result for {variant}: {supplemental_path}"
            )
        variants[variant] = metrics
        supplemental_sources[variant] = provenance
    run = [variant for variant in PLANNED_ABLATIONS if variant in variants]
    full_f1 = float(variants.get("full", {}).get("evidence_f1") or 0.0)
    wo_role_f1 = variants.get("w/o_role", {}).get("evidence_f1")
    wo_field_f1 = variants.get("w/o_field", {}).get("evidence_f1")
    full_beats_wo_role = wo_role_f1 is not None and full_f1 > float(wo_role_f1)
    full_beats_wo_field = wo_field_f1 is not None and full_f1 > float(wo_field_f1)
    return {
        "status": "RUN" if len(run) == len(PLANNED_ABLATIONS) else "PARTIAL",
        "dataset": "conditionalqa",
        "planned_variants": list(PLANNED_ABLATIONS),
        "run_variants": run,
        "missing_variants": [
            variant for variant in PLANNED_ABLATIONS if variant not in variants
        ],
        "variants": variants,
        "gate_required_comparison": {
            "full_beats_w_o_role": full_beats_wo_role,
            "full_beats_w_o_field": full_beats_wo_field,
            "passed": full_beats_wo_role and full_beats_wo_field,
            "interpretation": (
                "Gate 2 requires Full to outperform both w/o Role and w/o Field. "
                "A missing ablation is not treated as a pass."
            ),
        },
        "source_sha256": sha256(path),
        "supplemental_sources": supplemental_sources,
    }


def audit_public_ablation_schema(role_scores_dir: Path) -> dict[str, Any]:
    count_fields = (
        "cases",
        "cases_with_required_field_schema",
        "candidates",
        "candidates_with_field_scores",
        "candidates_with_applicability",
        "candidates_with_standardized_validity_fields",
        "candidates_with_conflict_annotations",
        "candidates_with_cross_encoder_score",
        "candidates_with_role_scores",
    )
    datasets: dict[str, Any] = {}
    totals: dict[str, int] = defaultdict(int)
    for dataset in DATASET_METHODS:
        path = role_scores_dir / f"role_scores_{dataset}.jsonl"
        counts: dict[str, int] = defaultdict(int)
        for row in read_jsonl(path):
            counts["cases"] += 1
            if any(key in row for key in ("required_fields", "slots", "field_schema")):
                counts["cases_with_required_field_schema"] += 1
            for candidate in row.get("candidates", []):
                counts["candidates"] += 1
                if candidate.get("field_scores"):
                    counts["candidates_with_field_scores"] += 1
                if candidate.get("applicability") is not None:
                    counts["candidates_with_applicability"] += 1
                if any(
                    candidate.get(key) is not None
                    for key in (
                        "version",
                        "jurisdiction",
                        "valid_from",
                        "valid_to",
                        "effective_at",
                    )
                ):
                    counts["candidates_with_standardized_validity_fields"] += 1
                if (
                    candidate.get("conflict_key") is not None
                    or candidate.get("conflict_value") is not None
                ):
                    counts["candidates_with_conflict_annotations"] += 1
                scores = candidate.get("scores", {})
                if scores.get("cross_encoder") is not None:
                    counts["candidates_with_cross_encoder_score"] += 1
                if candidate.get("role_scores"):
                    counts["candidates_with_role_scores"] += 1
        datasets[dataset] = {name: counts[name] for name in count_fields}
        for name in count_fields:
            value = counts[name]
            totals[name] += value
    variants = {
        "w/o_field": {
            "status": "SCHEMA_BLOCKED",
            "reason": "public artifacts contain neither required field schemas nor candidate field_scores",
        },
        "w/o_applicability": {
            "status": "SCHEMA_BLOCKED",
            "reason": "public artifacts contain no standardized applicability/version/jurisdiction/effective-period fields",
        },
        "w/o_conflict": {
            "status": "SCHEMA_BLOCKED",
            "reason": "the three primary public artifacts contain no candidate-level conflict annotations; CONFLICTS is a separate challenge dataset",
        },
        "w/o_reranker": {
            "status": "REQUIRES_REAL_RESCORING",
            "reason": "saved relevance and role scores depend on the Cross-Encoder, so swapping only the relevance column is not a valid ablation",
        },
    }
    return {
        "datasets": datasets,
        "totals": {name: totals[name] for name in count_fields},
        "variants": variants,
        "interpretation": (
            "SCHEMA_BLOCKED means the variant cannot be identified on these public artifacts. "
            "It is not counted as RUN or PASS and must be evaluated on a field-, applicability-, "
            "or conflict-annotated benchmark."
        ),
    }


def load_k_sensitivity_audit(metrics_dir: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASET_METHODS:
        path = metrics_dir / f"k_sensitivity_{dataset}.csv"
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "k": int(row["K"]),
                        "method": display_method(row["Method"]),
                        "evidence_recall": _safe_float(row["Evidence Recall@K"]),
                        "role_coverage": _safe_float(row["Role Coverage@K"]),
                        "condition_coverage": _safe_float(row["Condition Coverage@K"]),
                        "answer_f1": _safe_float(row["Answer F1@K"]),
                        "token_cost": _safe_float(row["Token Cost@K"]),
                        "cases": int(row["Cases"]),
                    }
                )
        summaries = []
        for k in sorted({row["k"] for row in rows}):
            at_k = [row for row in rows if row["k"] == k]
            frc = next(row for row in at_k if row["method"] == "frc_select")
            baselines = [row for row in at_k if row["method"] != "frc_select"]
            strongest = max(
                baselines, key=lambda row: float(row["evidence_recall"] or 0.0)
            )
            summaries.append(
                {
                    "k": k,
                    "frc_evidence_recall": frc["evidence_recall"],
                    "frc_role_coverage": frc["role_coverage"],
                    "strongest_baseline": strongest["method"],
                    "baseline_evidence_recall": strongest["evidence_recall"],
                    "frc_minus_baseline_recall": round(
                        float(frc["evidence_recall"] or 0.0)
                        - float(strongest["evidence_recall"] or 0.0),
                        6,
                    ),
                }
            )
        datasets[dataset] = {
            "status": "RUN",
            "k_values": sorted({row["k"] for row in rows}),
            "rows": rows,
            "summary": summaries,
            "source_sha256": sha256(path),
        }
    expected_k = [2, 3, 5, 8]
    complete = all(dataset["k_values"] == expected_k for dataset in datasets.values())
    return {
        "status": "RUN" if complete else "PARTIAL",
        "expected_k": expected_k,
        "datasets": datasets,
    }


def load_parameter_sensitivity_audit(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Method"] != "frc_select":
                continue
            rows.append(
                {
                    "alpha": _safe_float(row["alpha"]),
                    "gamma": _safe_float(row["gamma"]),
                    "role_threshold": _safe_float(row["role_threshold"]),
                    "role_mix": _safe_float(row["role_mix"]),
                    "evidence_recall": _safe_float(row["Evidence Recall@5"]),
                    "evidence_f1": _safe_float(row["Evidence F1@5"]),
                    "role_coverage": _safe_float(row["Role Coverage@5"]),
                    "condition_coverage": _safe_float(row["Condition Coverage"]),
                    "redundancy": _safe_float(row["Redundancy"]),
                    "token_cost": _safe_float(row["Token Cost"]),
                    "cases": int(row["Cases"]),
                }
            )
    best = max(
        rows,
        key=lambda row: (
            float(row["evidence_f1"] or 0.0),
            float(row["role_coverage"] or 0.0),
            -float(row["token_cost"] or 0.0),
        ),
    )
    frozen = next(
        (
            row
            for row in rows
            if row["alpha"] == 2.0
            and row["gamma"] == 0.0
            and row["role_threshold"] == 0.55
            and row["role_mix"] == 0.15
        ),
        None,
    )
    return {
        "status": "RUN",
        "dataset": "conditionalqa",
        "configuration_count": len(rows),
        "grid": {
            "alpha": sorted({row["alpha"] for row in rows}),
            "gamma": sorted({row["gamma"] for row in rows}),
            "role_threshold": sorted({row["role_threshold"] for row in rows}),
            "role_mix": sorted({row["role_mix"] for row in rows}),
        },
        "best_by_evidence_f1": best,
        "frozen_configuration": frozen,
        "frozen_minus_best_evidence_f1": (
            round(
                float(frozen["evidence_f1"] or 0.0) - float(best["evidence_f1"] or 0.0),
                6,
            )
            if frozen
            else None
        ),
        "interpretation": (
            "This is a descriptive sweep over saved ConditionalQA real-model scores. "
            "It is not a held-out tuning result and does not authorize changing the frozen test configuration."
        ),
        "source_sha256": sha256(path),
    }


def build_design_experiment_audit(
    metrics_dir: Path,
    supplemental_ablation_paths: Iterable[Path] = (),
    chunk_length_sensitivity_path: Path | None = None,
    controlled_domain_sensitivity_path: Path | None = None,
    conflicts_ablation_path: Path | None = None,
    housing_ablation_path: Path | None = None,
    housing_weight_sensitivity_path: Path | None = None,
    lawshift_ablation_path: Path | None = None,
    eurlex_ablation_path: Path | None = None,
) -> dict[str, Any]:
    supplemental_paths = tuple(supplemental_ablation_paths)
    ablation = load_ablation_audit(
        metrics_dir / "ablation_conditionalqa.csv",
        supplemental_paths,
    )
    k_sensitivity = load_k_sensitivity_audit(metrics_dir)
    parameters = load_parameter_sensitivity_audit(
        metrics_dir / "frc_param_sweep_conditionalqa.csv"
    )
    role_scores_dir = metrics_dir.parent / "role_scores"
    schema_audit = audit_public_ablation_schema(role_scores_dir)
    if "w/o_reranker" in ablation["run_variants"]:
        schema_audit["variants"]["w/o_reranker"] = {
            "status": "RUN_REAL_BIENCODER_RESCORING",
            "reason": "supplemental artifact recomputed both relevance and role scores without a Cross-Encoder",
        }
    token_budgets = token_budget_sensitivity(role_scores_dir)
    missing_ratios = missing_ratio_sensitivity(
        role_scores_dir / "role_scores_conditionalqa.jsonl"
    )
    chunk_lengths = (
        load_chunk_length_sensitivity(chunk_length_sensitivity_path)
        if chunk_length_sensitivity_path
        else {
            "status": "NOT_RUN",
            "interpretation": "no real-model chunk-length sensitivity artifact was supplied",
        }
    )
    controlled_sensitivity = (
        load_controlled_domain_sensitivity(controlled_domain_sensitivity_path)
        if controlled_domain_sensitivity_path
        else {
            "status": "NOT_RUN",
            "interpretation": "no controlled-domain field/role/conflict sensitivity artifact was supplied",
        }
    )
    controlled_status = controlled_sensitivity["status"]
    conflicts_ablation = (
        load_conflicts_real_model_ablation(conflicts_ablation_path)
        if conflicts_ablation_path
        else {
            "status": "NOT_RUN",
            "interpretation": "no public real-model CONFLICTS ablation artifact was supplied",
        }
    )
    conflicts_ablation_status = conflicts_ablation["status"]
    housing_ablation = (
        load_housing_real_model_ablation(housing_ablation_path)
        if housing_ablation_path
        else {
            "status": "NOT_RUN",
            "interpretation": (
                "no public expert HousingQA field/jurisdiction ablation artifact was supplied"
            ),
        }
    )
    housing_ablation_status = housing_ablation["status"]
    housing_run = (
        housing_ablation_status == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
    )
    housing_weight_sensitivity = (
        load_housing_public_weight_sensitivity(housing_weight_sensitivity_path)
        if housing_weight_sensitivity_path
        else {
            "status": "NOT_RUN",
            "interpretation": (
                "no public HousingQA real-model field/role-weight sensitivity artifact was supplied"
            ),
        }
    )
    housing_weight_status = housing_weight_sensitivity["status"]
    housing_weight_run = (
        housing_weight_status == "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
    )
    lawshift_ablation = (
        load_lawshift_temporal_ablation(lawshift_ablation_path)
        if lawshift_ablation_path
        else {
            "status": "NOT_RUN",
            "interpretation": (
                "no public expert-reviewed LawShift version-replacement artifact was supplied"
            ),
        }
    )
    lawshift_ablation_status = lawshift_ablation["status"]
    lawshift_run = (
        lawshift_ablation_status == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
    )
    eurlex_ablation = (
        load_eurlex_temporal_ablation(eurlex_ablation_path)
        if eurlex_ablation_path
        else {
            "status": "NOT_RUN",
            "interpretation": (
                "no public official EUR-Lex effective/expiry-date artifact was supplied"
            ),
        }
    )
    eurlex_ablation_status = eurlex_ablation["status"]
    eurlex_run = (
        eurlex_ablation_status == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
    )
    version_replacement_status = (
        lawshift_ablation.get("coverage", {}).get("version_replacement")
        if lawshift_run
        else "NOT_RUN"
    )
    expiry_status = (
        eurlex_ablation_status
        if eurlex_run
        else lawshift_ablation.get("coverage", {}).get("effective_or_expiry_dates")
        if lawshift_run
        else "NOT_RUN"
    )
    applicability_complete = housing_run and lawshift_run and eurlex_run
    flood_domain_expert_validation = False
    combined_run_variants = sorted(
        {
            *ablation["run_variants"],
            *(
                ["w/o_conflict"]
                if conflicts_ablation_status == "RUN_REAL_MODEL_CONFLICTS"
                else []
            ),
            *(["w/o_field"] if housing_run else []),
            *(
                ["w/o_applicability"]
                if housing_run or lawshift_run or eurlex_run
                else []
            ),
        }
    )
    schema_audit["variants"]["w/o_conflict"] = (
        {
            "status": "RUN_REAL_MODEL_CONFLICTS",
            "reason": (
                "Google CONFLICTS supplies case-level conflict types; the ablation removes only "
                "alternative-claim and temporal-validity disclosure roles on the frozen real-model pool"
            ),
        }
        if conflicts_ablation_status == "RUN_REAL_MODEL_CONFLICTS"
        else schema_audit["variants"]["w/o_conflict"]
    )
    if housing_run:
        schema_audit["variants"]["w/o_field"] = {
            "status": "RUN_PUBLIC_EXPERT_REAL_MODEL",
            "reason": (
                "HousingQA supplies expert questions and supporting statutes; four independent "
                "questions are deterministically composed into fields and evaluated with frozen "
                "BGE, Cross-Encoder, and local Qwen models"
            ),
        }
    if housing_run or lawshift_run or eurlex_run:
        schema_audit["variants"]["w/o_applicability"] = {
            "status": (
                "RUN_PUBLIC_REAL_MODEL_JURISDICTION_REVISION_EFFECTIVE_EXPIRY"
                if housing_run and lawshift_run and eurlex_run
                else "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_AND_REVISION"
                if housing_run and lawshift_run
                else "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
                if housing_run
                else "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
                if lawshift_run
                else "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
            ),
            "reason": (
                "HousingQA identifies jurisdiction filtering under its 2021 snapshot; LawShift "
                "identifies expert-reviewed before/after statutory replacement; EUR-Lex/CELLAR "
                "supplies official effective and expiry dates at adjacent repeal boundaries."
                if housing_run and lawshift_run and eurlex_run
                else "HousingQA identifies jurisdiction filtering under its 2021 snapshot; LawShift "
                "identifies expert-reviewed before/after statutory replacement. Neither supplies "
                "authoritative effective or expiry dates."
                if housing_run and lawshift_run
                else "HousingQA identifies jurisdiction filtering under its official 2021 snapshot; "
                "it does not identify version replacement or expiry behavior"
                if housing_run
                else "LawShift identifies expert-reviewed before/after statutory replacement but "
                "does not supply authoritative effective or expiry dates"
                if lawshift_run
                else "EUR-Lex/CELLAR identifies official effective and expiry dates at adjacent "
                "repeal boundaries but does not identify HousingQA jurisdiction or LawShift revisions"
            ),
        }
    controlled_only_status = (
        "RUN_CONTROLLED_DOMAIN_PUBLIC_SCHEMA_BLOCKED"
        if controlled_status == "RUN_CONTROLLED_DOMAIN"
        else "NOT_RUN"
    )
    sensitivity_coverage = {
        "k_2_3_5_8": "RUN",
        "token_budget_512_1024_2048": token_budgets["status"],
        "role_and_field_weights": (
            housing_weight_status
            if housing_weight_run
            else controlled_only_status
            if controlled_status == "RUN_CONTROLLED_DOMAIN"
            else "PARTIAL_ROLE_ONLY"
        ),
        "conflict_threshold": (
            "RUN_REAL_MODEL_CONFLICTS"
            if conflicts_ablation_status == "RUN_REAL_MODEL_CONFLICTS"
            else controlled_only_status
        ),
        "document_missing_ratio": missing_ratios["status"],
        "chunk_length": chunk_lengths["status"],
    }
    return {
        "status": "PARTIAL",
        "ablation": ablation,
        "k_sensitivity": k_sensitivity,
        "parameter_sensitivity": parameters,
        "token_budget_sensitivity": token_budgets,
        "missing_ratio_sensitivity": missing_ratios,
        "ablation_schema_applicability": schema_audit,
        "chunk_length_sensitivity": chunk_lengths,
        "controlled_domain_sensitivity": controlled_sensitivity,
        "conflicts_real_model_ablation": conflicts_ablation,
        "housing_real_model_ablation": housing_ablation,
        "housing_public_weight_sensitivity": housing_weight_sensitivity,
        "lawshift_temporal_ablation": lawshift_ablation,
        "eurlex_temporal_ablation": eurlex_ablation,
        "combined_ablation_coverage": {
            "planned_variants": list(PLANNED_ABLATIONS),
            "run_variants": combined_run_variants,
            "run_count": len(combined_run_variants),
            "planned_count": len(PLANNED_ABLATIONS),
            "missing_variants": sorted(
                set(PLANNED_ABLATIONS) - set(combined_run_variants)
            ),
            "scope_qualifications": {
                "w/o_field": (
                    "public expert HousingQA composite fields"
                    if housing_run
                    else "not run on an identifiable public expert field schema"
                ),
                "w/o_applicability": (
                    "HousingQA jurisdiction, LawShift expert-reviewed before/after revisions, and "
                    "EUR-Lex/CELLAR official effective/expiry dates"
                    if housing_run and lawshift_run and eurlex_run
                    else "HousingQA jurisdiction plus LawShift expert-reviewed before/after revisions; "
                    "authoritative effective/expiry dates untested"
                    if housing_run and lawshift_run
                    else "jurisdiction only under the HousingQA 2021 snapshot; version/expiry untested"
                    if housing_run
                    else "before/after LawShift revisions only; jurisdiction and expiry untested"
                    if lawshift_run
                    else "official EUR-Lex effective/expiry dates only; jurisdiction and expert-reviewed revisions untested"
                    if eurlex_run
                    else "not run on an identifiable public applicability schema"
                ),
            },
        },
        "applicability_identifiability": {
            "jurisdiction": "RUN_PUBLIC_EXPERT_REAL_MODEL"
            if housing_run
            else "NOT_RUN",
            "version_replacement": version_replacement_status,
            "effective_or_expiry_dates": expiry_status,
            "complete": applicability_complete,
        },
        "sensitivity_coverage": sensitivity_coverage,
        "technical_sensitivity_matrix_complete": (
            k_sensitivity["status"] == "RUN"
            and token_budgets["status"] == "RUN"
            and housing_weight_run
            and conflicts_ablation_status == "RUN_REAL_MODEL_CONFLICTS"
            and missing_ratios["status"] == "RUN"
            and chunk_lengths["status"] == "RUN"
        ),
        "flood_domain_expert_validation_complete": flood_domain_expert_validation,
        "coverage_complete": (
            len(combined_run_variants) == len(PLANNED_ABLATIONS)
            and housing_weight_run
            and conflicts_ablation_status == "RUN_REAL_MODEL_CONFLICTS"
            and all(
                sensitivity_coverage[dimension] == "RUN"
                for dimension in (
                    "k_2_3_5_8",
                    "token_budget_512_1024_2048",
                    "document_missing_ratio",
                    "chunk_length",
                )
            )
            and applicability_complete
            and flood_domain_expert_validation
        ),
        "interpretation": (
            f"Real-model artifacts cover {len(combined_run_variants)} of nine planned FRC ablations, "
            "all K and token-budget values, a ConditionalQA role/redundancy parameter grid, and "
            "multi-ratio missing-evidence diagnostics. Chunk-length status is "
            f"{chunk_lengths['status']}; controlled field/role/conflict sensitivity status is "
            f"{controlled_status}; public CONFLICTS ablation status is {conflicts_ablation_status}. "
            f"public HousingQA field/jurisdiction ablation status is {housing_ablation_status}. "
            f"public HousingQA field/role-weight sensitivity status is {housing_weight_status}. "
            f"public LawShift version-replacement status is {lawshift_ablation_status}. "
            f"public EUR-Lex effective/expiry-date status is {eurlex_ablation_status}. "
            "Even when all nine named variants have an execution artifact across compatible "
            "datasets, the three-part applicability construct is identifiable, and the public "
            "real-model weight matrix is complete, same-domain flood-response expert validity "
            "and independent behavior judging remain incomplete."
        ),
    }


def evidence_metrics(
    expected_ids: Iterable[str], selected_ids: Iterable[str]
) -> dict[str, float]:
    expected = set(expected_ids)
    selected = set(selected_ids)
    true_positive = len(expected & selected)
    recall = true_positive / max(1, len(expected))
    precision = true_positive / max(1, len(selected))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "evidence_recall": recall,
        "evidence_precision": precision,
        "evidence_f1": f1,
        "complete_evidence_set": float(expected <= selected),
    }


def selected_role_coverage(row: dict[str, Any], *, threshold: float = 0.55) -> float:
    required = list(row.get("required_roles", []))
    if not required:
        return 1.0
    selected = row.get("selected_evidence", [])
    covered = sum(
        max(
            (float(item.get("role_scores", {}).get(role, 0.0)) for item in selected),
            default=0.0,
        )
        >= threshold
        for role in required
    )
    return covered / len(required)


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        **evidence_metrics(
            row.get("gold_evidence_ids", []), row.get("selected_ids", [])
        ),
        "role_coverage": selected_role_coverage(row),
    }


def paired_bootstrap(
    differences: list[float],
    *,
    seed: int = 20260713,
    resamples: int = 2000,
) -> dict[str, float | int]:
    if not differences:
        return {"cases": 0, "mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    size = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    return {
        "cases": size,
        "mean_difference": round(sum(differences) / size, 6),
        "ci_low": round(samples[int(0.025 * (resamples - 1))], 6),
        "ci_high": round(samples[int(0.975 * (resamples - 1))], 6),
    }


def load_chunk_length_sensitivity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frc-chunk-length-sensitivity-v1":
        raise ValueError(f"unsupported chunk-length sensitivity schema: {path}")
    if payload.get("dataset") != "conditionalqa":
        raise ValueError("chunk-length sensitivity currently requires ConditionalQA")
    metadata = payload.get("metadata", {})
    chunk_lengths = [int(value) for value in metadata.get("chunk_lengths", [])]
    methods = [str(value) for value in metadata.get("methods", [])]
    if len(chunk_lengths) < 2 or "frc_select" not in methods:
        raise ValueError(
            "chunk-length sensitivity needs multiple lengths and frc_select"
        )
    case_rows = payload.get("case_results", [])
    claimed_rows = {
        (int(row["chunk_length"]), str(row["method"])): row["metrics"]
        for row in payload.get("aggregates", [])
    }
    metric_names = (
        "evidence_recall",
        "evidence_precision",
        "evidence_f1",
        "role_coverage",
        "token_cost",
        "selected_chunk_count",
        "unique_parent_count",
        "duplicate_parent_ratio",
    )
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(int(row["chunk_length"]), str(row["method"]))].append(row)
    aggregates: dict[tuple[int, str], dict[str, float | int]] = {}
    for chunk_length in chunk_lengths:
        for method in methods:
            key = (chunk_length, method)
            rows = grouped.get(key, [])
            if not rows:
                raise ValueError(f"missing chunk-length case results: {key}")
            if len({row["case_id"] for row in rows}) != len(rows):
                raise ValueError(f"duplicate chunk-length case results: {key}")
            recomputed = {
                name: round(
                    sum(float(row["metrics"][name]) for row in rows) / len(rows),
                    6,
                )
                for name in metric_names
            }
            claimed = claimed_rows.get(key)
            if claimed is None:
                raise ValueError(f"missing chunk-length aggregate: {key}")
            for name, value in recomputed.items():
                if not math.isclose(
                    value, float(claimed.get(name, float("nan"))), abs_tol=1e-6
                ):
                    raise ValueError(
                        f"chunk-length aggregate mismatch for {key} {name}"
                    )
            aggregates[key] = {**recomputed, "cases": len(rows)}
    results = []
    for chunk_length in chunk_lengths:
        metrics = {method: aggregates[(chunk_length, method)] for method in methods}
        baselines = [method for method in methods if method != "frc_select"]
        strongest = max(
            baselines,
            key=lambda method: float(metrics[method]["evidence_f1"]),
        )
        frc_by_case = {
            row["case_id"]: float(row["metrics"]["evidence_f1"])
            for row in grouped[(chunk_length, "frc_select")]
        }
        baseline_by_case = {
            row["case_id"]: float(row["metrics"]["evidence_f1"])
            for row in grouped[(chunk_length, strongest)]
        }
        case_ids = sorted(set(frc_by_case) & set(baseline_by_case))
        paired = paired_bootstrap(
            [frc_by_case[case_id] - baseline_by_case[case_id] for case_id in case_ids]
        )
        results.append(
            {
                "chunk_length": chunk_length,
                "metrics": metrics,
                "strongest_baseline": strongest,
                "frc_minus_baseline_evidence_f1": round(
                    float(metrics["frc_select"]["evidence_f1"])
                    - float(metrics[strongest]["evidence_f1"]),
                    6,
                ),
                "paired_frc_minus_baseline_evidence_f1": paired,
            }
        )
    return {
        "status": "RUN",
        "dataset": "conditionalqa",
        "chunk_lengths": chunk_lengths,
        "methods": methods,
        "metadata": metadata,
        "results": results,
        "source_sha256": sha256(path),
        "interpretation": (
            "Every chunk length is rescored by the real reranker for question relevance and all "
            "five role queries. Evidence metrics use unique parent evidence IDs, while token cost "
            "and duplicate-parent ratio remain chunk-level diagnostics."
        ),
    }


def compare_selected_files(frc_path: Path, baseline_path: Path) -> dict[str, Any]:
    frc = {row["case_id"]: row_metrics(row) for row in read_jsonl(frc_path)}
    baseline = {row["case_id"]: row_metrics(row) for row in read_jsonl(baseline_path)}
    case_ids = sorted(set(frc) & set(baseline))
    output: dict[str, Any] = {"cases": len(case_ids), "metrics": {}}
    for metric in PRIMARY_METRICS:
        differences = [
            frc[case_id][metric] - baseline[case_id][metric] for case_id in case_ids
        ]
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        output["metrics"][metric] = {
            **paired_bootstrap(differences),
            "wins": wins,
            "ties": len(differences) - wins - losses,
            "losses": losses,
        }
    return output


def _take_with_budget(
    candidates: list[dict[str, Any]], *, k: int, budget: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for candidate in candidates:
        cost = int(candidate.get("token_count", 1))
        if total + cost > budget:
            continue
        selected.append(candidate)
        total += cost
        if len(selected) >= k:
            break
    return selected


def select_precomputed(
    row: dict[str, Any],
    method: str,
    *,
    k: int = 5,
    budget: int = 1500,
    threshold: float = 0.55,
    role_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    score_name = {
        "bm25_topk": "bm25",
        "dense_topk": "dense",
        "hybrid_topk": "hybrid",
        "cross_encoder_topk": "cross_encoder",
    }.get(method)
    if score_name:
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("scores", {}).get(score_name, 0.0)),
                item["id"],
            ),
        )
        return _take_with_budget(ordered, k=k, budget=budget)

    if method == "coverage_greedy_proxy":
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        total = 0
        for role in row.get("required_roles", []):
            eligible = [item for item in candidates if item["id"] not in selected_ids]
            if not eligible:
                break
            best = max(
                eligible,
                key=lambda item: (
                    float(item.get("role_scores", {}).get(role, 0.0)),
                    float(item.get("scores", {}).get("cross_encoder", 0.0)),
                    item["id"],
                ),
            )
            cost = int(best.get("token_count", 1))
            if total + cost > budget:
                continue
            selected.append(best)
            selected_ids.add(best["id"])
            total += cost
            if len(selected) >= k:
                return selected
        fill = sorted(
            (item for item in candidates if item["id"] not in selected_ids),
            key=lambda item: (
                -float(item.get("scores", {}).get("cross_encoder", 0.0)),
                item["id"],
            ),
        )
        return selected + _take_with_budget(
            fill, k=k - len(selected), budget=max(0, budget - total)
        )

    if method != "frc_select":
        raise ValueError(f"unsupported precomputed selector: {method}")
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    total = 0
    role_best = {role: 0.0 for role in row.get("required_roles", [])}
    while remaining and len(selected) < k:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate.get("token_count", 1))
            if total + cost > budget:
                continue
            improvements = []
            for role, old in role_best.items():
                score = float(candidate.get("role_scores", {}).get(role, 0.0))
                role_threshold = float((role_thresholds or {}).get(role, threshold))
                if old < role_threshold <= max(old, score):
                    improvements.append(1.0)
                else:
                    improvements.append(max(0.0, score - old) * 0.25)
            role_gain = sum(improvements) / max(1, len(role_best))
            relevance = float(candidate.get("scores", {}).get("cross_encoder", 0.0))
            scored.append((2.0 * role_gain + relevance, candidate["id"], candidate))
        if not scored:
            break
        _, _, best = max(scored, key=lambda item: (item[0], item[1]))
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best.get("token_count", 1))
        for role in role_best:
            role_best[role] = max(
                role_best[role], float(best.get("role_scores", {}).get(role, 0.0))
            )
    return selected


def missing_evidence_challenge(path: Path) -> dict[str, Any]:
    methods = (
        "bm25_topk",
        "dense_topk",
        "hybrid_topk",
        "cross_encoder_topk",
        "coverage_greedy_proxy",
        "frc_select",
    )
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cases = 0
    for source in read_jsonl(path):
        gold = list(source.get("gold_evidence_ids", []))
        if len(gold) < 2:
            continue
        cases += 1
        removed = gold[0]
        available_gold = set(gold[1:])
        row = {
            **source,
            "candidates": [
                item for item in source.get("candidates", []) if item["id"] != removed
            ],
        }
        for method in methods:
            selected = select_precomputed(row, method)
            metrics = evidence_metrics(
                available_gold, [item["id"] for item in selected]
            )
            selected_row = {**row, "selected_evidence": selected}
            role_coverage = selected_role_coverage(selected_row)
            for key, value in metrics.items():
                totals[method][key] += value
            totals[method]["role_coverage"] += role_coverage
            totals[method]["false_complete"] += float(role_coverage == 1.0)
    return {
        "protocol": "remove the first gold passage from every ConditionalQA case with at least two gold passages, then rerun selectors from saved real-model scores",
        "cases": cases,
        "removed_gold_per_case": 1,
        "metrics": {
            method: {
                key: round(value / max(1, cases), 6) for key, value in values.items()
            }
            for method, values in totals.items()
        },
        "interpretation": "false_complete is the rate at which role scores still claim full coverage after a required gold passage was removed; lower is safer.",
    }


def aggregate_precomputed_selectors(
    rows: Iterable[dict[str, Any]],
    methods: Iterable[str],
    *,
    k: int,
    budget: int,
) -> dict[str, dict[str, float | int]]:
    method_list = list(methods)
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cases = 0
    for row in rows:
        cases += 1
        for method in method_list:
            selected = select_precomputed(row, method, k=k, budget=budget)
            metrics = evidence_metrics(
                row.get("gold_evidence_ids", []),
                [item["id"] for item in selected],
            )
            for key, value in metrics.items():
                totals[method][key] += value
            totals[method]["role_coverage"] += selected_role_coverage(
                {**row, "selected_evidence": selected}
            )
            token_cost = sum(int(item.get("token_count", 1)) for item in selected)
            totals[method]["token_cost"] += token_cost
            totals[method]["budget_violation"] += float(token_cost > budget)
    return {
        method: {
            **{
                key: round(value / max(1, cases), 6)
                for key, value in totals[method].items()
            },
            "cases": cases,
        }
        for method in method_list
    }


def token_budget_sensitivity(
    role_scores_dir: Path,
    *,
    budgets: tuple[int, ...] = (512, 1024, 2048),
    k: int = 5,
) -> dict[str, Any]:
    methods = ("cross_encoder_topk", "coverage_greedy_proxy", "frc_select")
    datasets: dict[str, Any] = {}
    for dataset in DATASET_METHODS:
        path = role_scores_dir / f"role_scores_{dataset}.jsonl"
        rows = list(read_jsonl(path))
        budget_results = []
        for budget in budgets:
            metrics = aggregate_precomputed_selectors(
                rows,
                methods,
                k=k,
                budget=budget,
            )
            strongest = max(
                (method for method in methods if method != "frc_select"),
                key=lambda method: float(metrics[method]["evidence_f1"]),
            )
            budget_results.append(
                {
                    "token_budget": budget,
                    "metrics": metrics,
                    "strongest_baseline": strongest,
                    "frc_minus_baseline_evidence_f1": round(
                        float(metrics["frc_select"]["evidence_f1"])
                        - float(metrics[strongest]["evidence_f1"]),
                        6,
                    ),
                }
            )
        datasets[dataset] = {
            "status": "RUN",
            "cases": len(rows),
            "budgets": budget_results,
            "source_sha256": sha256(path),
        }
    all_within_budget = all(
        method_metrics["budget_violation"] == 0.0
        for dataset in datasets.values()
        for item in dataset["budgets"]
        for method_metrics in item["metrics"].values()
    )
    return {
        "status": "RUN" if all_within_budget else "FAIL",
        "k": k,
        "token_budgets": list(budgets),
        "datasets": datasets,
        "all_methods_within_budget": all_within_budget,
        "interpretation": (
            "Selectors are rerun from the same saved real-model candidate and role scores. "
            "No generation metric is reused as a retrieval-selection claim."
        ),
    }


def _stable_missing_evidence_ids(
    case_id: str,
    gold_ids: Iterable[str],
    ratio: float,
) -> set[str]:
    removed: set[str] = set()
    for evidence_id in gold_ids:
        digest = hashlib.sha256(
            f"missing-ratio-v1\0{case_id}\0{evidence_id}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        if unit < ratio:
            removed.add(evidence_id)
    return removed


def missing_ratio_sensitivity(
    path: Path,
    *,
    ratios: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
) -> dict[str, Any]:
    methods = ("cross_encoder_topk", "coverage_greedy_proxy", "frc_select")
    rows = [
        row for row in read_jsonl(path) if len(row.get("gold_evidence_ids", [])) >= 2
    ]
    ratio_results = []
    for ratio in ratios:
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_gold = 0
        removed_gold = 0
        cases_with_removal = 0
        empty_available = 0
        for source in rows:
            gold = list(source["gold_evidence_ids"])
            removed = _stable_missing_evidence_ids(source["id"], gold, ratio)
            available = set(gold) - removed
            total_gold += len(gold)
            removed_gold += len(removed)
            cases_with_removal += int(bool(removed))
            empty_available += int(not available)
            row = {
                **source,
                "candidates": [
                    item
                    for item in source.get("candidates", [])
                    if item["id"] not in removed
                ],
            }
            for method in methods:
                selected = select_precomputed(row, method)
                metrics = evidence_metrics(available, [item["id"] for item in selected])
                for key, value in metrics.items():
                    totals[method][key] += value
                role_coverage = selected_role_coverage(
                    {**row, "selected_evidence": selected}
                )
                totals[method]["role_coverage"] += role_coverage
                if removed:
                    totals[method]["false_complete"] += float(role_coverage == 1.0)
        metrics = {
            method: {
                "evidence_recall": round(
                    totals[method]["evidence_recall"] / max(1, len(rows)), 6
                ),
                "evidence_precision": round(
                    totals[method]["evidence_precision"] / max(1, len(rows)), 6
                ),
                "evidence_f1": round(
                    totals[method]["evidence_f1"] / max(1, len(rows)), 6
                ),
                "complete_available_evidence_set": round(
                    totals[method]["complete_evidence_set"] / max(1, len(rows)), 6
                ),
                "role_coverage": round(
                    totals[method]["role_coverage"] / max(1, len(rows)), 6
                ),
                "false_complete_on_removed_cases": round(
                    totals[method]["false_complete"] / max(1, cases_with_removal), 6
                ),
            }
            for method in methods
        }
        strongest = max(
            (method for method in methods if method != "frc_select"),
            key=lambda method: metrics[method]["evidence_f1"],
        )
        ratio_results.append(
            {
                "target_missing_ratio": ratio,
                "achieved_missing_ratio": round(removed_gold / max(1, total_gold), 6),
                "cases": len(rows),
                "cases_with_removal": cases_with_removal,
                "empty_available_evidence_rate": round(
                    empty_available / max(1, len(rows)), 6
                ),
                "metrics": metrics,
                "strongest_baseline": strongest,
                "frc_minus_baseline_evidence_f1": round(
                    metrics["frc_select"]["evidence_f1"]
                    - metrics[strongest]["evidence_f1"],
                    6,
                ),
            }
        )
    return {
        "status": "RUN",
        "protocol": (
            "For each case/evidence pair, a fixed SHA-256 uniform score is compared with the target "
            "ratio, producing nested deterministic removals from the same saved real-model pool."
        ),
        "target_ratios": list(ratios),
        "results": ratio_results,
        "source_sha256": sha256(path),
    }


def conflict_inventory(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "status": "NOT_RUN",
            "reason": "official Google CONFLICTS file is not locally available; no result is fabricated",
        }
    if path.suffix.lower() == ".json":
        report = json.loads(path.read_text(encoding="utf-8"))
        metadata = report.get("metadata", {})
        if (
            metadata.get("name")
            == "Google CONFLICTS FRC retrieval and conflict-classification audit"
        ):
            strongest = report["strongest_reproducible_baseline_by_accuracy"]
            return {
                "status": metadata.get("status", "RUN"),
                "cases": metadata["cases"],
                "answer_annotated_cases": metadata["answer_annotated_cases"],
                "conflict_types": metadata["conflict_type_counts"],
                "dataset_sha256": metadata["dataset_sha256"],
                "report_sha256": sha256(path),
                "strongest_reproducible_baseline": strongest,
                "baseline_accuracy": report["classification_metrics"][strongest][
                    "accuracy"
                ],
                "frc_accuracy": report["classification_metrics"]["frc_select"][
                    "accuracy"
                ],
                "paired_frc_minus_baseline_accuracy": report[
                    "paired_frc_minus_baseline_accuracy"
                ],
                "frc_outdated_recall": report["classification_metrics"]["frc_select"][
                    "per_type"
                ]["Conflict due to outdated information"]["recall"],
                "note": report["decision"]["reason"],
            }
        raise ValueError(f"unsupported CONFLICTS report format: {path}")
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for row in read_jsonl(path):
        total += 1
        counts[str(row.get("conflict_type", "unknown"))] += 1
    return {
        "status": "DATA_READY",
        "cases": total,
        "conflict_types": dict(sorted(counts.items())),
        "sha256": sha256(path),
        "note": "inventory only; model comparison requires the same BGE/reranker scoring pass before metrics can be reported",
    }


def build_public_reference_report(
    reference_root: Path,
    *,
    conflicts_path: Path | None = None,
    supplemental_ablation_paths: Iterable[Path] = (),
    chunk_length_sensitivity_path: Path | None = None,
    controlled_domain_sensitivity_path: Path | None = None,
    conflicts_ablation_path: Path | None = None,
    housing_ablation_path: Path | None = None,
    housing_weight_sensitivity_path: Path | None = None,
    lawshift_ablation_path: Path | None = None,
    eurlex_ablation_path: Path | None = None,
) -> dict[str, Any]:
    output = reference_root / "outputs"
    metrics_dir = output / "metrics"
    selected_dir = output / "selected_evidence"
    datasets: dict[str, Any] = {}
    for dataset, methods in DATASET_METHODS.items():
        evidence_csv = metrics_dir / f"evidence_metrics_{dataset}.csv"
        answer_csv = metrics_dir / f"answer_metrics_{dataset}.csv"
        evidence = load_aggregate_csv(evidence_csv)
        answers = load_answer_csv(answer_csv)
        baseline_methods = [
            display_method(method) for method in methods if method != "frc_select"
        ]
        strongest = max(
            baseline_methods,
            key=lambda method: float(evidence[method]["evidence_f1"] or 0.0),
        )
        source_method = (
            "setr_style" if strongest == "coverage_greedy_proxy" else strongest
        )
        comparison = compare_selected_files(
            selected_dir / f"{dataset}_frc_select.jsonl",
            selected_dir / f"{dataset}_{source_method}.jsonl",
        )
        datasets[dataset] = {
            "evidence_metrics": evidence,
            "answer_metrics": answers,
            "strongest_reproducible_baseline_by_evidence_f1": strongest,
            "paired_frc_minus_baseline": comparison,
            "source_hashes": {
                "evidence_csv": sha256(evidence_csv),
                "answer_csv": sha256(answer_csv),
                "frc_selected": sha256(selected_dir / f"{dataset}_frc_select.jsonl"),
                "baseline_selected": sha256(
                    selected_dir / f"{dataset}_{source_method}.jsonl"
                ),
            },
        }
    superiority = all(
        dataset["paired_frc_minus_baseline"]["metrics"]["evidence_f1"]["ci_low"] > 0
        for dataset in datasets.values()
    )
    conflict = conflict_inventory(conflicts_path)
    conflict_run = conflict.get("status") == "RUN"
    experiment_audit = build_design_experiment_audit(
        metrics_dir,
        supplemental_ablation_paths,
        chunk_length_sensitivity_path,
        controlled_domain_sensitivity_path,
        conflicts_ablation_path,
        housing_ablation_path,
        housing_weight_sensitivity_path,
        lawshift_ablation_path,
        eurlex_ablation_path,
    )
    ablation_gate = experiment_audit["ablation"]["gate_required_comparison"]
    housing_ablation = experiment_audit["housing_real_model_ablation"]
    housing_run = (
        housing_ablation["status"] == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
    )
    lawshift_ablation = experiment_audit["lawshift_temporal_ablation"]
    lawshift_run = (
        lawshift_ablation["status"] == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
    )
    eurlex_ablation = experiment_audit["eurlex_temporal_ablation"]
    eurlex_run = (
        eurlex_ablation["status"] == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
    )
    limitations = [
        "The report imports existing real-model artifacts and recomputes paired evidence metrics; it does not retrain models.",
        "The SetR paper implementation is not available in this environment; coverage_greedy_proxy is not SetR.",
        "ConditionalQA generation scores are low, so evidence-selection feasibility must not be presented as answer-generation superiority.",
        "The deterministic missing-evidence challenge removes one gold passage and reuses saved scores; it is a robustness audit, not an official dataset split.",
        "The nine named ablation variants now have execution artifacts across compatible public datasets; applicability is separately identifiable through HousingQA jurisdiction, LawShift expert-reviewed hypothetical revisions, and EUR-Lex/CELLAR official effective/expiry dates; and HousingQA now supplies a frozen-score public real-model field/role-weight sweep. This is still not full design coverage because the weight and applicability sources are cross-domain rather than independently judged flood-response records.",
    ]
    if conflict_run:
        limitations.append(
            "CONFLICTS conflict-type classification is complete, but the paper's expected-behavior adherence metric and independent human judging are not reproduced."
        )
    else:
        limitations.append(
            "CONFLICTS metrics remain NOT_RUN until the official file and an equal-scoring pass are available."
        )
    if housing_run:
        limitations.extend(housing_ablation.get("limitations", []))
    if lawshift_run:
        limitations.extend(lawshift_ablation.get("limitations", []))
    if eurlex_run:
        limitations.extend(eurlex_ablation.get("limitations", []))
    return {
        "metadata": {
            "name": "FRC-RAG public-dataset real-model reference audit",
            "reference_root": str(reference_root.resolve()),
            "scoring_backend": "real",
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "reranker_model": "BAAI/bge-reranker-large",
            "generator": "Qwen2.5-7B-Instruct-GPTQ-Int4",
            "top_k": 5,
            "token_budget": 1500,
            "score_calibration": "per_case_minmax",
            "role_relevance_mix": 0.15,
            "selection_parameters": {"alpha": 2.0, "beta": 1.0, "gamma": 0.0},
            "setr_status": "not reproduced; setr_style source rows are reported as coverage_greedy_proxy",
        },
        "datasets": datasets,
        "design_16_2_experiment_audit": experiment_audit,
        "challenge_slices": {
            "normal": {"status": "RUN", "datasets": list(datasets)},
            "exception_and_condition": {"status": "RUN", "dataset": "conditionalqa"},
            "cross_document": {"status": "RUN", "dataset": "multihoprag"},
            "multi_hop": {"status": "RUN", "datasets": ["multihoprag", "hotpotqa"]},
            "missing_evidence": missing_evidence_challenge(
                output / "role_scores" / "role_scores_conditionalqa.jsonl"
            ),
            "conflict_and_stale": conflict,
            "field_and_jurisdiction_applicability": {
                "status": housing_ablation["status"] if housing_run else "NOT_RUN",
                "dataset": "reglab/housing_qa" if housing_run else None,
                "cases": housing_ablation.get("metadata", {}).get("case_count"),
                "fields": housing_ablation.get("metadata", {}).get("field_count"),
                "jurisdictions": housing_ablation.get("metadata", {}).get(
                    "jurisdiction_count"
                ),
                "snapshot_year": housing_ablation.get("metadata", {}).get(
                    "snapshot_year"
                ),
                "version_or_expiry": housing_ablation.get("coverage", {}).get(
                    "version_or_expiry_variation"
                ),
            },
            "version_replacement_applicability": {
                "status": lawshift_ablation["status"] if lawshift_run else "NOT_RUN",
                "dataset": "triangularPeach/LawShift" if lawshift_run else None,
                "cases": lawshift_ablation.get("metadata", {}).get("case_count"),
                "revision_types": lawshift_ablation.get("metadata", {}).get(
                    "revision_type_count"
                ),
                "effective_or_expiry_dates": lawshift_ablation.get("coverage", {}).get(
                    "effective_or_expiry_dates"
                ),
            },
            "effective_expiry_applicability": {
                "status": eurlex_ablation["status"] if eurlex_run else "NOT_RUN",
                "dataset": (
                    "EU Publications Office CELLAR and EUR-Lex" if eurlex_run else None
                ),
                "cases": eurlex_ablation.get("metadata", {}).get("case_count"),
                "pairs": eurlex_ablation.get("metadata", {}).get("pair_count"),
                "effective_dates": eurlex_ablation.get("coverage", {}).get(
                    "effective_dates"
                ),
                "expiry_dates": eurlex_ablation.get("coverage", {}).get("expiry_dates"),
            },
        },
        "decision": {
            "status": "THEORETICAL_PIPELINE_FEASIBLE_BUT_SUPERIORITY_NOT_PROVEN",
            "pipeline_feasible": True,
            "evidence_f1_superiority_on_all_primary_datasets": superiority,
            "full_outperforms_w_o_role_and_w_o_field": (
                ablation_gate["full_beats_w_o_role"]
                and housing_run
                and housing_ablation["decision"]["full_strictly_better_than_w_o_field"]
            ),
            "design_16_2_experiment_coverage_complete": experiment_audit[
                "coverage_complete"
            ],
            "gate_2": "NO-GO",
            "reason": (
                "real-model FRC runs are reproducible, but paired confidence intervals do not establish "
                "consistent superiority; Full does not outperform w/o Role; HousingQA Full does "
                "outperform w/o Field but ties the strongest field-decomposition baseline, and "
                "LawShift Full improves exact version evidence over w/o Applicability but ties "
                "the fair applicability-filtered Cross-Encoder baseline; EUR-Lex/CELLAR now "
                "identifies official effective/expiry boundaries, where Full improves over the "
                "unfiltered variant but again ties the fair filtered Cross-Encoder baseline; "
                "the CONFLICTS Full variant also does not outperform w/o Conflict; "
                "CONFLICTS is run but does not reproduce the paper's independent expected-behavior "
                "adherence judgment"
                if conflict_run
                else "real-model FRC runs are reproducible and competitive, but paired confidence intervals "
                "do not establish consistent superiority; conflict/stale comparison is not yet run"
            ),
        },
        "limitations": limitations,
    }


def render_public_reference_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# FRC-RAG 公开数据真实模型参考审计",
        "",
        f"- Embedding：`{metadata['embedding_model']}`",
        f"- Reranker：`{metadata['reranker_model']}`",
        f"- Generator：`{metadata['generator']}`",
        f"- 统一预算：Top-K={metadata['top_k']}，{metadata['token_budget']} tokens",
        "- `coverage_greedy_proxy` 是覆盖贪心代理，不是 SetR 复现。",
        "",
        "## 主结果",
        "",
        "| 数据集 | 用例 | FRC Evidence F1 | 最强可复现基线 | 基线 F1 | 差值 | 95% CI |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, dataset in report["datasets"].items():
        frc = dataset["evidence_metrics"]["frc_select"]
        baseline_name = dataset["strongest_reproducible_baseline_by_evidence_f1"]
        baseline = dataset["evidence_metrics"][baseline_name]
        paired = dataset["paired_frc_minus_baseline"]["metrics"]["evidence_f1"]
        lines.append(
            f"| {name} | {frc['cases']} | {frc['evidence_f1']:.6f} | {baseline_name} | "
            f"{baseline['evidence_f1']:.6f} | {paired['mean_difference']:+.6f} | "
            f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}] |"
        )
    experiment = report["design_16_2_experiment_audit"]
    ablation = experiment["ablation"]
    combined_ablation = experiment["combined_ablation_coverage"]
    conflicts_ablation = experiment["conflicts_real_model_ablation"]
    housing_ablation = experiment["housing_real_model_ablation"]
    housing_weights = experiment["housing_public_weight_sensitivity"]
    lawshift_ablation = experiment["lawshift_temporal_ablation"]
    eurlex_ablation = experiment["eurlex_temporal_ablation"]
    lines.extend(
        [
            "",
            "## 设计第 16.2 节消融审计",
            "",
            f"- 覆盖状态：`{ablation['status']}`；跨适用公开集已运行 {combined_ablation['run_count']}/{combined_ablation['planned_count']} 组。",
            f"- Gate 必需比较通过：`{ablation['gate_required_comparison']['passed']}`。缺失消融不得按通过处理。",
            "",
            "| 消融 | 状态 | Evidence F1 | Role Coverage |",
            "|---|---|---:|---:|",
        ]
    )
    for variant in ablation["planned_variants"]:
        values = ablation["variants"].get(variant)
        if values:
            lines.append(
                f"| {variant} | RUN | {values['evidence_f1']:.6f} | {values['role_coverage']:.6f} |"
            )
        elif (
            variant == "w/o_conflict"
            and conflicts_ablation["status"] == "RUN_REAL_MODEL_CONFLICTS"
        ):
            lines.append("| w/o_conflict | RUN_REAL_MODEL_CONFLICTS | — | — |")
        elif (
            variant in {"w/o_field", "w/o_applicability"}
            and housing_ablation["status"]
            == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
        ):
            housing_variant = housing_ablation["aggregates"][variant]
            lines.append(
                f"| {variant} | {housing_ablation['coverage'][variant]} | "
                f"{housing_variant['evidence_f1']:.6f} | — |"
            )
        else:
            lines.append(f"| {variant} | NOT_RUN | — | — |")
    wo_reranker_source = ablation["supplemental_sources"].get("w/o_reranker")
    if wo_reranker_source and wo_reranker_source["paired_against_full_evidence_f1"]:
        paired = wo_reranker_source["paired_against_full_evidence_f1"]
        lines.extend(
            [
                "",
                "`w/o Reranker` 相对 Full 的 Evidence F1 配对差值为 "
                f"{paired['mean_difference']:+.6f}，95% CI="
                f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]；区间跨 0。",
            ]
        )
    if conflicts_ablation["status"] == "RUN_REAL_MODEL_CONFLICTS":
        full = conflicts_ablation["ablation"]["full"]
        without = conflicts_ablation["ablation"]["without_conflict"]
        paired = conflicts_ablation["ablation"]["comparison"]["full_minus_wo_conflict"][
            "classification_accuracy"
        ]
        lines.extend(
            [
                "",
                "### `w/o Conflict`（Google CONFLICTS，真实模型）",
                "",
                "该事后诊断只移除 `alternative_claim` 与 `temporal_validity` 两个冲突披露角色；"
                "候选池、BGE、重排器、Qwen、K 和 Token 预算保持一致。",
                "",
                "| 版本 | 冲突类型准确率 | Macro F1 | 选择变化用例 |",
                "|---|---:|---:|---:|",
                f"| Full | {full['classification']['accuracy']:.6f} | {full['classification']['macro_f1']:.6f} | 0 |",
                f"| w/o Conflict | {without['classification']['accuracy']:.6f} | {without['classification']['macro_f1']:.6f} | {conflicts_ablation['ablation']['comparison']['selection_changed_cases']} |",
                "",
                f"Full−`w/o Conflict` 准确率差值为 {paired['mean_difference']:+.6f}，95% CI="
                f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]。该结果不证明 Full 更优，"
                "也不替代独立 expected-behavior adherence 评判。",
            ]
        )
    if housing_ablation["status"] == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021":
        housing_full = housing_ablation["aggregates"]["frc_full"]
        housing_without_field = housing_ablation["aggregates"]["w/o_field"]
        housing_without_applicability = housing_ablation["aggregates"][
            "w/o_applicability"
        ]
        housing_field_pair = housing_ablation["paired_comparisons"][
            "full_minus_w_o_field"
        ]["metrics"]["field_coverage"]
        housing_applicability_pair = housing_ablation["paired_comparisons"][
            "full_minus_w_o_applicability"
        ]["metrics"]
        housing_baseline_pair = housing_ablation["paired_comparisons"][
            "full_minus_strongest_baseline"
        ]["metrics"]["field_coverage"]
        lines.extend(
            [
                "",
                "### `w/o Field` 与 `w/o Applicability`（HousingQA，公开专家标注，真实模型）",
                "",
                "40 个确定性复合用例包含 160 个任务字段、22 个司法辖区；每个字段的正确答案与支持法条仅用于事后评分。"
                "适用性消融只识别 2021 快照下的司法辖区过滤，不能识别法规版本替换或失效。",
                "",
                "| 版本 | Field Coverage | Citation Support Precision | Wrong-jurisdiction Rate | Answer Accuracy |",
                "|---|---:|---:|---:|---:|",
                f"| Full | {housing_full['field_coverage']:.6f} | {housing_full['citation_support_precision']:.6f} | {housing_full['wrong_jurisdiction_rate']:.6f} | {housing_full['answer_accuracy']:.6f} |",
                f"| w/o Field | {housing_without_field['field_coverage']:.6f} | {housing_without_field['citation_support_precision']:.6f} | {housing_without_field['wrong_jurisdiction_rate']:.6f} | {housing_without_field['answer_accuracy']:.6f} |",
                f"| w/o Applicability | {housing_without_applicability['field_coverage']:.6f} | {housing_without_applicability['citation_support_precision']:.6f} | {housing_without_applicability['wrong_jurisdiction_rate']:.6f} | {housing_without_applicability['answer_accuracy']:.6f} |",
                "",
                f"Full−w/o Field 的字段覆盖差值为 {housing_field_pair['mean_difference']:+.6f}，95% CI=[{housing_field_pair['ci_low']:+.6f}, {housing_field_pair['ci_high']:+.6f}]；"
                f"Full−w/o Applicability 的字段覆盖差值为 {housing_applicability_pair['field_coverage']['mean_difference']:+.6f}，"
                f"错误司法辖区率差值为 {housing_applicability_pair['wrong_jurisdiction_rate']['mean_difference']:+.6f}。",
                f"最强字段覆盖基线为 `{housing_ablation['strongest_baseline']}`；Full 相对其字段覆盖差值为 "
                f"{housing_baseline_pair['mean_difference']:+.6f}，未达到 Gate 2 要求的 +0.05。",
            ]
        )
    if (
        housing_weights["status"]
        == "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
    ):
        lines.extend(
            [
                "",
                "### 字段/角色权重敏感性（HousingQA，公开冻结真实模型分数）",
                "",
                "该单因素扫描复用同一 40 用例、160 字段、22 辖区候选池；gold 字段证据只在选择完成后评分。"
                "结果用于辨识权重行为，不用于回看结果后修改冻结参数。",
                "",
                "| 维度 | 值 | Evidence F1 | Field Coverage | Role Coverage | 相对冻结选择变化 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in housing_weights["results"]:
            aggregate = row["aggregate"]
            lines.append(
                f"| {row['dimension']} | {row['value']:.2f} | "
                f"{aggregate['evidence_f1']:.6f} | {aggregate['field_coverage']:.6f} | "
                f"{aggregate['role_coverage']:.6f} | "
                f"{row['selection_changed_cases_from_frozen']} |"
            )
        lines.extend(
            [
                "",
                "字段权重从 0 提高到冻结值 2 时 Evidence F1/字段覆盖由 0.656250 提高到 0.706250；"
                "角色权重从 0 提高到 2 时角色覆盖由 0.600000 提高到 0.633333。"
                "字段与角色权重均可辨识，但更高权重相对冻结配置的最大 Evidence F1 增益只有 0.006250，"
                "且角色标签不是 HousingQA 专家标注，因此 Gate 2 仍为 NO-GO。",
            ]
        )
    if lawshift_ablation["status"] == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL":
        lawshift_full = lawshift_ablation["aggregates"]["frc_full"]
        lawshift_without = lawshift_ablation["aggregates"]["w/o_applicability"]
        lawshift_filtered = lawshift_ablation["aggregates"][
            "applicability_filtered_cross_encoder_top1"
        ]
        lawshift_pair = lawshift_ablation["paired_comparisons"][
            "full_minus_w_o_applicability"
        ]["metrics"]
        lawshift_baseline_pair = lawshift_ablation["paired_comparisons"][
            "full_minus_strongest_baseline"
        ]["metrics"]["exact_evidence_accuracy"]
        lines.extend(
            [
                "",
                "### `w/o Applicability`（LawShift，31 类专家审阅修订，真实重排）",
                "",
                "124 个用例平衡覆盖修订前/后快照；候选池同时包含目标法条两个版本和三对词法难负例。"
                "法条版本是专家审阅的假设修订，不含权威生效或失效日期。",
                "",
                "| 版本 | Article Recall@1 | Version Accuracy | Exact Evidence Accuracy | Invalid Applicability |",
                "|---|---:|---:|---:|---:|",
                f"| Full | {lawshift_full['article_recall_at_1']:.6f} | {lawshift_full['version_accuracy']:.6f} | {lawshift_full['exact_evidence_accuracy']:.6f} | {lawshift_full['invalid_applicability_rate']:.6f} |",
                f"| w/o Applicability | {lawshift_without['article_recall_at_1']:.6f} | {lawshift_without['version_accuracy']:.6f} | {lawshift_without['exact_evidence_accuracy']:.6f} | {lawshift_without['invalid_applicability_rate']:.6f} |",
                f"| Applicability-filtered Cross-Encoder | {lawshift_filtered['article_recall_at_1']:.6f} | {lawshift_filtered['version_accuracy']:.6f} | {lawshift_filtered['exact_evidence_accuracy']:.6f} | {lawshift_filtered['invalid_applicability_rate']:.6f} |",
                "",
                f"Full−w/o Applicability 的精确版本证据差值为 {lawshift_pair['exact_evidence_accuracy']['mean_difference']:+.6f}，"
                f"95% CI=[{lawshift_pair['exact_evidence_accuracy']['ci_low']:+.6f}, {lawshift_pair['exact_evidence_accuracy']['ci_high']:+.6f}]；"
                f"错误版本率差值为 {lawshift_pair['invalid_applicability_rate']['mean_difference']:+.6f}。",
                f"Full 的 Article Recall@1 相对无过滤版本差值为 {lawshift_pair['article_recall_at_1']['mean_difference']:+.6f}；"
                f"相对公平过滤基线的精确版本证据差值为 {lawshift_baseline_pair['mean_difference']:+.6f}。"
                "因此版本过滤有效，但 FRC 独有优势未获证明。",
            ]
        )
    if eurlex_ablation["status"] == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL":
        eurlex_full = eurlex_ablation["aggregates"]["frc_full"]
        eurlex_without = eurlex_ablation["aggregates"]["w/o_applicability"]
        eurlex_filtered = eurlex_ablation["aggregates"][
            "applicability_filtered_cross_encoder_top1"
        ]
        eurlex_pair = eurlex_ablation["paired_comparisons"][
            "full_minus_w_o_applicability"
        ]["metrics"]
        eurlex_baseline_pair = eurlex_ablation["paired_comparisons"][
            "full_minus_strongest_baseline"
        ]["metrics"]["exact_evidence_accuracy"]
        lines.extend(
            [
                "",
                "### `w/o Applicability`（EUR-Lex/CELLAR，权威生效与失效边界，真实重排）",
                "",
                "30 对废止/替代法案形成 60 个边界用例，分别查询旧法最后有效日和新法首个生效日；"
                "权威 CELLAR 日期只用于适用性过滤与事后评分，查询文本不包含 gold CELEX。",
                "",
                "| 版本 | Exact Evidence Accuracy | Validity Accuracy | Invalid Applicability | Wrong Boundary Version |",
                "|---|---:|---:|---:|---:|",
                f"| Full | {eurlex_full['exact_evidence_accuracy']:.6f} | {eurlex_full['validity_accuracy']:.6f} | {eurlex_full['invalid_applicability_rate']:.6f} | {eurlex_full['wrong_boundary_version_rate']:.6f} |",
                f"| w/o Applicability | {eurlex_without['exact_evidence_accuracy']:.6f} | {eurlex_without['validity_accuracy']:.6f} | {eurlex_without['invalid_applicability_rate']:.6f} | {eurlex_without['wrong_boundary_version_rate']:.6f} |",
                f"| Applicability-filtered Cross-Encoder | {eurlex_filtered['exact_evidence_accuracy']:.6f} | {eurlex_filtered['validity_accuracy']:.6f} | {eurlex_filtered['invalid_applicability_rate']:.6f} | {eurlex_filtered['wrong_boundary_version_rate']:.6f} |",
                "",
                f"Full−w/o Applicability 的精确证据差值为 {eurlex_pair['exact_evidence_accuracy']['mean_difference']:+.6f}，"
                f"95% CI=[{eurlex_pair['exact_evidence_accuracy']['ci_low']:+.6f}, {eurlex_pair['exact_evidence_accuracy']['ci_high']:+.6f}]；"
                f"无效适用率差值为 {eurlex_pair['invalid_applicability_rate']['mean_difference']:+.6f}。",
                f"Full 相对公平过滤基线的精确证据差值为 {eurlex_baseline_pair['mean_difference']:+.6f}。"
                "因此权威日期过滤有效，但 FRC 独有优势仍未获证明，且该语料属于欧盟法律而非区县防汛文档。",
            ]
        )
    schema_audit = experiment["ablation_schema_applicability"]
    lines.extend(
        [
            "",
            "### 消融数据可识别性",
            "",
            "| 消融 | 可识别状态 | 原因 |",
            "|---|---|---|",
        ]
    )
    for variant, item in schema_audit["variants"].items():
        lines.append(f"| {variant} | {item['status']} | {item['reason']} |")
    lines.extend(
        [
            "",
            "## K 与参数敏感性审计",
            "",
            "| 数据集 | K | FRC Recall | 最强基线 | 基线 Recall | 差值 |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for dataset_name, dataset in experiment["k_sensitivity"]["datasets"].items():
        for item in dataset["summary"]:
            lines.append(
                f"| {dataset_name} | {item['k']} | {item['frc_evidence_recall']:.6f} | "
                f"{item['strongest_baseline']} | {item['baseline_evidence_recall']:.6f} | "
                f"{item['frc_minus_baseline_recall']:+.6f} |"
            )
    parameter = experiment["parameter_sensitivity"]
    best = parameter["best_by_evidence_f1"]
    lines.extend(
        [
            "",
            f"ConditionalQA 保存分数上共审计 {parameter['configuration_count']} 组 FRC 参数；最佳 Evidence F1={best['evidence_f1']:.6f}（alpha={best['alpha']}、gamma={best['gamma']}、role_threshold={best['role_threshold']}）。该扫参不是独立留出集结果，不用于事后改写冻结测试配置。",
            "",
            "| 敏感性维度 | 状态 |",
            "|---|---|",
        ]
    )
    for dimension, status in experiment["sensitivity_coverage"].items():
        lines.append(f"| {dimension} | {status} |")
    controlled = experiment["controlled_domain_sensitivity"]
    if controlled["status"] == "RUN_CONTROLLED_DOMAIN":
        lines.extend(
            [
                "",
                "### Controlled-domain role/field/conflict sensitivity",
                "",
                "This diagnostic uses a repository-constructed `SYNTHETIC` benchmark and no neural model. "
                "It verifies selector behavior but does not satisfy the public real-model Gate 2 requirement.",
                "",
                "| Dimension | Value | Evidence F1 | Gold field coverage | Selector field coverage | Role coverage | Flagged cases | Case accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in controlled["results"]:
            aggregate = item["aggregate"]
            lines.append(
                f"| {item['dimension']} | {item['value']:.2f} | {aggregate['evidence_f1']:.6f} | "
                f"{aggregate['field_coverage']:.6f} | {aggregate['selector_field_coverage']:.6f} | "
                f"{aggregate['role_coverage']:.6f} | "
                f"{aggregate['flagged_evidence_case']:.6f} | {aggregate['case_accuracy']:.6f} |"
            )
    if conflicts_ablation["status"] == "RUN_REAL_MODEL_CONFLICTS":
        lines.extend(
            [
                "",
                "### CONFLICTS conflict-threshold sensitivity (real model)",
                "",
                "Only the two conflict-disclosure role thresholds change; this is a post-hoc diagnostic.",
                "",
                "| Threshold | Accuracy | Macro F1 | Exact answer support | Newest-date retention |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for item in conflicts_ablation["conflict_threshold_sensitivity"]["results"]:
            lines.append(
                f"| {item['conflict_threshold']:.2f} | {item['classification']['accuracy']:.6f} | "
                f"{item['classification']['macro_f1']:.6f} | "
                f"{item['selection']['exact_answer_support']:.6f} | "
                f"{item['selection']['newest_date_retention']:.6f} |"
            )
    chunk_lengths = experiment["chunk_length_sensitivity"]
    if chunk_lengths["status"] == "RUN":
        lines.extend(
            [
                "",
                "### 分块长度敏感性（ConditionalQA，真实重评分）",
                "",
                "| 分块 tokens | FRC Evidence F1 | 最强基线 | 基线 F1 | 差值 | 95% CI | 重复父证据率 |",
                "|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for item in chunk_lengths["results"]:
            frc = item["metrics"]["frc_select"]
            baseline_name = item["strongest_baseline"]
            baseline = item["metrics"][baseline_name]
            paired = item["paired_frc_minus_baseline_evidence_f1"]
            lines.append(
                f"| {item['chunk_length']} | {frc['evidence_f1']:.6f} | {baseline_name} | "
                f"{baseline['evidence_f1']:.6f} | {item['frc_minus_baseline_evidence_f1']:+.6f} | "
                f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}] | "
                f"{frc['duplicate_parent_ratio']:.6f} |"
            )
    lines.extend(
        [
            "",
            "### Token 预算敏感性",
            "",
            "| 数据集 | Token 预算 | FRC Evidence F1 | 最强基线 | 基线 F1 | 差值 |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for dataset_name, dataset in experiment["token_budget_sensitivity"][
        "datasets"
    ].items():
        for item in dataset["budgets"]:
            frc = item["metrics"]["frc_select"]
            baseline_name = item["strongest_baseline"]
            baseline = item["metrics"][baseline_name]
            lines.append(
                f"| {dataset_name} | {item['token_budget']} | {frc['evidence_f1']:.6f} | "
                f"{baseline_name} | {baseline['evidence_f1']:.6f} | "
                f"{item['frc_minus_baseline_evidence_f1']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "### 多档缺失比例敏感性（ConditionalQA）",
            "",
            "| 目标缺失 | 实际缺失 | FRC Evidence F1 | 最强基线 | 差值 | FRC 错误完整声明率 |",
            "|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in experiment["missing_ratio_sensitivity"]["results"]:
        frc = item["metrics"]["frc_select"]
        lines.append(
            f"| {item['target_missing_ratio']:.0%} | {item['achieved_missing_ratio']:.2%} | "
            f"{frc['evidence_f1']:.6f} | {item['strongest_baseline']} | "
            f"{item['frc_minus_baseline_evidence_f1']:+.6f} | "
            f"{frc['false_complete_on_removed_cases']:.6f} |"
        )
    missing = report["challenge_slices"]["missing_evidence"]
    lines.extend(
        [
            "",
            "## 缺失证据挑战",
            "",
            f"协议：{missing['protocol']}。用例：{missing['cases']}。",
            "",
            "| 方法 | 可用证据 Recall | 完整可用证据集 | 角色覆盖 | 错误宣称完整率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, values in missing["metrics"].items():
        lines.append(
            f"| {method} | {values['evidence_recall']:.6f} | {values['complete_evidence_set']:.6f} | "
            f"{values['role_coverage']:.6f} | {values['false_complete']:.6f} |"
        )
    conflict = report["challenge_slices"]["conflict_and_stale"]
    lines.extend(
        [
            "",
            "## 冲突与失效证据",
            "",
            f"- 状态：`{conflict['status']}`",
            f"- 说明：{conflict.get('reason') or conflict.get('note')}",
        ]
    )
    if conflict["status"] == "RUN":
        paired = conflict["paired_frc_minus_baseline_accuracy"]
        lines.extend(
            [
                f"- 用例：{conflict['cases']}；有正确答案标注：{conflict['answer_annotated_cases']}",
                f"- 最强可复现基线：`{conflict['strongest_reproducible_baseline']}`，Accuracy={conflict['baseline_accuracy']:.6f}",
                f"- FRC Accuracy：{conflict['frc_accuracy']:.6f}",
                f"- FRC - 基线：{paired['mean_difference']:+.6f}，95% CI=[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]",
                f"- FRC 过时信息类型 Recall：{conflict['frc_outdated_recall']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- 状态：`{report['decision']['status']}`",
            f"- Gate 2：`{report['decision']['gate_2']}`",
            f"- 结论：{report['decision']['reason']}。",
            "",
            "这组结果证明真实模型、公开数据和 FRC 选择器可以形成可复现流水线，但不能证明 FRC 已经稳定优于强重排基线。",
            "",
            "## 限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
