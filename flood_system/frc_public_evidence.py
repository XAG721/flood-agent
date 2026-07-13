from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
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
PRIMARY_METRICS = ("evidence_recall", "evidence_precision", "evidence_f1", "role_coverage")

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


def load_supplemental_ablation(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
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
        raise ValueError("w/o_reranker must recompute relevance and role scores without a Cross-Encoder")
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
        if not math.isclose(value, float(claimed.get(name, float("nan"))), abs_tol=1e-6):
            raise ValueError(f"supplemental ablation aggregate mismatch for {name}: {path}")
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
            raise ValueError(f"duplicate ablation result for {variant}: {supplemental_path}")
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
        "missing_variants": [variant for variant in PLANNED_ABLATIONS if variant not in variants],
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
                    for key in ("version", "jurisdiction", "valid_from", "valid_to", "effective_at")
                ):
                    counts["candidates_with_standardized_validity_fields"] += 1
                if candidate.get("conflict_key") is not None or candidate.get("conflict_value") is not None:
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
            strongest = max(baselines, key=lambda row: float(row["evidence_recall"] or 0.0))
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
            round(float(frozen["evidence_f1"] or 0.0) - float(best["evidence_f1"] or 0.0), 6)
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
    sensitivity_coverage = {
        "k_2_3_5_8": "RUN",
        "token_budget_512_1024_2048": token_budgets["status"],
        "role_and_field_weights": "PARTIAL_ROLE_ONLY",
        "conflict_threshold": "NOT_RUN",
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
        "sensitivity_coverage": sensitivity_coverage,
        "coverage_complete": (
            ablation["status"] == "RUN"
            and all(status == "RUN" for status in sensitivity_coverage.values())
        ),
        "interpretation": (
            f"Real-model artifacts cover {len(ablation['run_variants'])} of nine planned FRC ablations, "
            "all K and token-budget values, a ConditionalQA role/redundancy parameter grid, and "
            "multi-ratio missing-evidence diagnostics. Chunk-length status is "
            f"{chunk_lengths['status']}. Schema-blocked dimensions remain unrun rather "
            "than being inferred from unrelated metrics or treated as passes."
        ),
    }


def evidence_metrics(expected_ids: Iterable[str], selected_ids: Iterable[str]) -> dict[str, float]:
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
        max((float(item.get("role_scores", {}).get(role, 0.0)) for item in selected), default=0.0)
        >= threshold
        for role in required
    )
    return covered / len(required)


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        **evidence_metrics(row.get("gold_evidence_ids", []), row.get("selected_ids", [])),
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
        raise ValueError("chunk-length sensitivity needs multiple lengths and frc_select")
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
                if not math.isclose(value, float(claimed.get(name, float("nan"))), abs_tol=1e-6):
                    raise ValueError(f"chunk-length aggregate mismatch for {key} {name}")
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
        differences = [frc[case_id][metric] - baseline[case_id][metric] for case_id in case_ids]
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        output["metrics"][metric] = {
            **paired_bootstrap(differences),
            "wins": wins,
            "ties": len(differences) - wins - losses,
            "losses": losses,
        }
    return output


def _take_with_budget(candidates: list[dict[str, Any]], *, k: int, budget: int) -> list[dict[str, Any]]:
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
            key=lambda item: (-float(item.get("scores", {}).get(score_name, 0.0)), item["id"]),
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
            key=lambda item: (-float(item.get("scores", {}).get("cross_encoder", 0.0)), item["id"]),
        )
        return selected + _take_with_budget(fill, k=k - len(selected), budget=max(0, budget - total))

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
                if old < threshold <= max(old, score):
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
            role_best[role] = max(role_best[role], float(best.get("role_scores", {}).get(role, 0.0)))
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
            "candidates": [item for item in source.get("candidates", []) if item["id"] != removed],
        }
        for method in methods:
            selected = select_precomputed(row, method)
            metrics = evidence_metrics(available_gold, [item["id"] for item in selected])
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
            method: {key: round(value / max(1, cases), 6) for key, value in values.items()}
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
    rows = [row for row in read_jsonl(path) if len(row.get("gold_evidence_ids", [])) >= 2]
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
                    item for item in source.get("candidates", []) if item["id"] not in removed
                ],
            }
            for method in methods:
                selected = select_precomputed(row, method)
                metrics = evidence_metrics(available, [item["id"] for item in selected])
                for key, value in metrics.items():
                    totals[method][key] += value
                role_coverage = selected_role_coverage({**row, "selected_evidence": selected})
                totals[method]["role_coverage"] += role_coverage
                if removed:
                    totals[method]["false_complete"] += float(role_coverage == 1.0)
        metrics = {
            method: {
                "evidence_recall": round(totals[method]["evidence_recall"] / max(1, len(rows)), 6),
                "evidence_precision": round(
                    totals[method]["evidence_precision"] / max(1, len(rows)), 6
                ),
                "evidence_f1": round(totals[method]["evidence_f1"] / max(1, len(rows)), 6),
                "complete_available_evidence_set": round(
                    totals[method]["complete_evidence_set"] / max(1, len(rows)), 6
                ),
                "role_coverage": round(totals[method]["role_coverage"] / max(1, len(rows)), 6),
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
                "empty_available_evidence_rate": round(empty_available / max(1, len(rows)), 6),
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
        if metadata.get("name") == "Google CONFLICTS FRC retrieval and conflict-classification audit":
            strongest = report["strongest_reproducible_baseline_by_accuracy"]
            return {
                "status": metadata.get("status", "RUN"),
                "cases": metadata["cases"],
                "answer_annotated_cases": metadata["answer_annotated_cases"],
                "conflict_types": metadata["conflict_type_counts"],
                "dataset_sha256": metadata["dataset_sha256"],
                "report_sha256": sha256(path),
                "strongest_reproducible_baseline": strongest,
                "baseline_accuracy": report["classification_metrics"][strongest]["accuracy"],
                "frc_accuracy": report["classification_metrics"]["frc_select"]["accuracy"],
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
        baseline_methods = [display_method(method) for method in methods if method != "frc_select"]
        strongest = max(
            baseline_methods,
            key=lambda method: float(evidence[method]["evidence_f1"] or 0.0),
        )
        source_method = "setr_style" if strongest == "coverage_greedy_proxy" else strongest
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
                "baseline_selected": sha256(selected_dir / f"{dataset}_{source_method}.jsonl"),
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
    )
    ablation_gate = experiment_audit["ablation"]["gate_required_comparison"]
    limitations = [
        "The report imports existing real-model artifacts and recomputes paired evidence metrics; it does not retrain models.",
        "The SetR paper implementation is not available in this environment; coverage_greedy_proxy is not SetR.",
        "ConditionalQA generation scores are low, so evidence-selection feasibility must not be presented as answer-generation superiority.",
        "The deterministic missing-evidence challenge removes one gold passage and reuses saved scores; it is a robustness audit, not an official dataset split.",
        "The real-model design audit remains incomplete; schema-blocked variants are not treated as run or passed, and field-weight/conflict-threshold sensitivity remains NOT_RUN.",
    ]
    if conflict_run:
        limitations.append(
            "CONFLICTS conflict-type classification is complete, but the paper's expected-behavior adherence metric and independent human judging are not reproduced."
        )
    else:
        limitations.append(
            "CONFLICTS metrics remain NOT_RUN until the official file and an equal-scoring pass are available."
        )
    return {
        "metadata": {
            "name": "FRC-Select public-dataset real-model reference audit",
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
        },
        "decision": {
            "status": "THEORETICAL_PIPELINE_FEASIBLE_BUT_SUPERIORITY_NOT_PROVEN",
            "pipeline_feasible": True,
            "evidence_f1_superiority_on_all_primary_datasets": superiority,
            "full_outperforms_w_o_role_and_w_o_field": ablation_gate["passed"],
            "design_16_2_experiment_coverage_complete": experiment_audit["coverage_complete"],
            "gate_2": "NO-GO",
            "reason": (
                "real-model FRC runs are reproducible, but paired confidence intervals do not establish "
                "consistent superiority; Full does not outperform w/o Role and w/o Field is schema-blocked on the primary public artifacts; "
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
        "# FRC-Select 公开数据真实模型参考审计",
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
    lines.extend(
        [
            "",
            "## 设计第 16.2 节消融审计",
            "",
            f"- 覆盖状态：`{ablation['status']}`；已运行 {len(ablation['run_variants'])}/{len(ablation['planned_variants'])} 组。",
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
    for dataset_name, dataset in experiment["token_budget_sensitivity"]["datasets"].items():
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
