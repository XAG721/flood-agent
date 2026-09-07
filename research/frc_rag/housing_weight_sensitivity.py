from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.housing_ablation import (
    HOUSING_SOURCE_LICENSE,
    HOUSING_SOURCE_REPOSITORY,
    HOUSING_SOURCE_REVISION,
    HOUSING_SOURCE_JSON_SHA256,
    select_housing_evidence,
    sha256,
)


SCHEMA_VERSION = "frc-housing-public-weight-sensitivity-v1"
STATUS = "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
SENSITIVITY_VALUES: dict[str, tuple[float, ...]] = {
    "field_weight": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    "role_weight": (0.0, 0.5, 1.0, 2.0, 4.0),
}
METRIC_NAMES = (
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


def read_scored_cases(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("HousingQA scored-case cache is empty")
    case_ids = [str(row.get("id", "")) for row in rows]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("HousingQA scored cases require unique non-empty identifiers")
    return rows


def _coverage(
    selected: list[dict[str, Any]],
    labels: Iterable[str],
    score_key: str,
    threshold: float,
) -> float:
    required = list(labels)
    if not required:
        return 1.0
    covered = sum(
        max(
            (
                float(candidate.get(score_key, {}).get(label, 0.0))
                for candidate in selected
            ),
            default=0.0,
        )
        >= threshold
        for label in required
    )
    return covered / len(required)


def score_weight_selection(
    case: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    field_threshold: float,
    role_threshold: float,
) -> dict[str, float | int]:
    selected_ids = {str(candidate["id"]) for candidate in selected}
    gold_ids = {str(value) for value in case.get("gold_evidence_ids", [])}
    true_positive = len(selected_ids & gold_ids)
    recall = true_positive / max(1, len(gold_ids))
    precision = true_positive / max(1, len(selected_ids))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    field_map = case.get("field_evidence_map", {})
    field_coverage = sum(
        bool(selected_ids & {str(value) for value in evidence_ids})
        for evidence_ids in field_map.values()
    ) / max(1, len(field_map))
    fields = [str(field["field_id"]) for field in case.get("fields", [])]
    roles = [str(role) for role in case.get("required_roles", [])]
    selector_field_coverage = _coverage(
        selected,
        fields,
        "field_scores",
        field_threshold,
    )
    role_coverage = _coverage(
        selected,
        roles,
        "role_scores",
        role_threshold,
    )
    wrong_jurisdiction = sum(
        candidate.get("metadata", {}).get("jurisdiction") != case.get("jurisdiction")
        for candidate in selected
    )
    return {
        "evidence_recall": round(recall, 6),
        "evidence_precision": round(precision, 6),
        "evidence_f1": round(f1, 6),
        "field_coverage": round(field_coverage, 6),
        "selector_field_coverage": round(selector_field_coverage, 6),
        "role_coverage": round(role_coverage, 6),
        "unsupported_field_rate": round(1.0 - field_coverage, 6),
        "wrong_jurisdiction_rate": round(
            wrong_jurisdiction / max(1, len(selected_ids)), 6
        ),
        "token_cost": sum(
            int(candidate.get("token_count", 1)) for candidate in selected
        ),
        "selected_count": len(selected),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("HousingQA weight sensitivity cannot aggregate an empty slice")
    return {
        "cases": len(rows),
        **{
            metric: round(
                sum(float(row["metrics"][metric]) for row in rows) / len(rows),
                6,
            )
            for metric in METRIC_NAMES
        },
    }


def evaluate_housing_weight_sensitivity(
    cases: list[dict[str, Any]],
    *,
    scored_metadata: dict[str, Any],
    scored_cases_path: Path,
    scored_metadata_path: Path,
    source_json_path: Path | None = None,
    sensitivity_values: dict[str, Iterable[float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = scored_metadata.get("config", {})
    required_config = (
        "embedding_model",
        "reranker_model",
        "score_calibration",
        "top_k",
        "token_budget",
        "field_threshold",
        "role_threshold",
        "field_weight",
        "role_weight",
    )
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise ValueError(f"HousingQA scored metadata is incomplete: {missing_config}")
    if int(scored_metadata.get("cases", -1)) != len(cases):
        raise ValueError(
            "HousingQA scored metadata case count does not match its cache"
        )
    expected_signature = scored_metadata.get("case_signature")
    if not isinstance(expected_signature, str) or len(expected_signature) != 64:
        raise ValueError("HousingQA scored metadata case signature is invalid")

    values_by_dimension = sensitivity_values or SENSITIVITY_VALUES
    normalized_values: dict[str, tuple[float, ...]] = {}
    for dimension in SENSITIVITY_VALUES:
        values = tuple(float(value) for value in values_by_dimension.get(dimension, ()))
        if (
            len(values) < 2
            or len(values) != len(set(values))
            or any(value < 0 for value in values)
        ):
            raise ValueError(
                f"HousingQA {dimension} sensitivity requires multiple unique non-negative values"
            )
        frozen_value = float(config[dimension])
        if frozen_value not in values:
            raise ValueError(
                f"HousingQA {dimension} scan must include frozen value {frozen_value}"
            )
        normalized_values[dimension] = values

    case_results: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for dimension, values in normalized_values.items():
        for value in values:
            for case in cases:
                parameters = {
                    "field_weight": float(config["field_weight"]),
                    "role_weight": float(config["role_weight"]),
                }
                parameters[dimension] = value
                selected = select_housing_evidence(
                    case,
                    "frc_full",
                    top_k=int(config["top_k"]),
                    token_budget=int(config["token_budget"]),
                    field_threshold=float(config["field_threshold"]),
                    role_threshold=float(config["role_threshold"]),
                    field_weight=parameters["field_weight"],
                    role_weight=parameters["role_weight"],
                )
                row = {
                    "dimension": dimension,
                    "value": value,
                    "case_id": str(case["id"]),
                    "selected_ids": [str(candidate["id"]) for candidate in selected],
                    "metrics": score_weight_selection(
                        case,
                        selected,
                        field_threshold=float(config["field_threshold"]),
                        role_threshold=float(config["role_threshold"]),
                    ),
                }
                case_results.append(row)
                grouped[(dimension, value)].append(row)

    summaries: list[dict[str, Any]] = []
    identifiability: dict[str, Any] = {}
    for dimension, values in normalized_values.items():
        frozen_value = float(config[dimension])
        frozen_rows = {
            row["case_id"]: row for row in grouped[(dimension, frozen_value)]
        }
        aggregates = {
            value: _aggregate(grouped[(dimension, value)]) for value in values
        }
        for value in values:
            rows = grouped[(dimension, value)]
            changed = sum(
                row["selected_ids"] != frozen_rows[row["case_id"]]["selected_ids"]
                for row in rows
            )
            summaries.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "is_frozen": value == frozen_value,
                    "selection_changed_cases_from_frozen": changed,
                    "aggregate": aggregates[value],
                }
            )
        diagnostic_metrics = (
            ("field_coverage", "evidence_f1")
            if dimension == "field_weight"
            else ("role_coverage", "evidence_f1")
        )
        ranges = {
            metric: sorted({float(aggregates[value][metric]) for value in values})
            for metric in diagnostic_metrics
        }
        if all(len(values_seen) == 1 for values_seen in ranges.values()):
            raise ValueError(f"HousingQA {dimension} sensitivity is not identifiable")
        identifiability[dimension] = {
            "status": "IDENTIFIABLE",
            "metric_values": ranges,
        }

    field_count = sum(len(case.get("fields", [])) for case in cases)
    jurisdictions = {str(case.get("jurisdiction", "")) for case in cases}
    source_hash = scored_metadata.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("HousingQA scored metadata source hash is invalid")
    if source_hash != HOUSING_SOURCE_JSON_SHA256:
        raise ValueError(
            "HousingQA scored metadata does not use the pinned public source"
        )
    if source_json_path is not None and sha256(source_json_path) != source_hash:
        raise ValueError(
            "HousingQA source JSON differs from the scored-cache provenance"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "reglab/housing_qa/questions",
        "status": STATUS,
        "metadata": {
            "data_origin": "PUBLIC",
            "is_simulated": False,
            "public_dataset": True,
            "expert_annotated_supporting_statutes": True,
            "source_repository": HOUSING_SOURCE_REPOSITORY,
            "source_revision": HOUSING_SOURCE_REVISION,
            "source_license": HOUSING_SOURCE_LICENSE,
            "source_json_sha256": source_hash,
            "scored_cases_sha256": sha256(scored_cases_path),
            "scored_metadata_sha256": sha256(scored_metadata_path),
            "case_signature": expected_signature,
            "case_count": len(cases),
            "field_count": field_count,
            "jurisdiction_count": len(jurisdictions),
            "role_count_per_case": sorted(
                {len(case.get("required_roles", [])) for case in cases}
            ),
            "candidate_occurrences": sum(
                len(case.get("candidates", [])) for case in cases
            ),
            "real_model_scores": True,
            "generator_used": False,
            "models": {
                "embedding": config["embedding_model"],
                "reranker": config["reranker_model"],
            },
            "score_calibration": config["score_calibration"],
            "frozen_parameters": {
                "top_k": int(config["top_k"]),
                "token_budget": int(config["token_budget"]),
                "field_threshold": float(config["field_threshold"]),
                "role_threshold": float(config["role_threshold"]),
                "field_weight": float(config["field_weight"]),
                "role_weight": float(config["role_weight"]),
            },
            "gold_usage": (
                "field_evidence_map and gold_evidence_ids are used only after selection for "
                "evaluation; the selector receives only frozen real-model relevance, field, "
                "role, applicability, and cost features"
            ),
            "protocol": (
                "one-factor-at-a-time descriptive sweep over the frozen HousingQA real-model "
                "candidate pool; observed results do not retune the frozen test configuration"
            ),
        },
        "dimensions": {key: list(values) for key, values in normalized_values.items()},
        "identifiability": identifiability,
        "results": summaries,
        "decision": {
            "gate_2": "NO-GO",
            "public_real_model_weight_sensitivity_complete": True,
            "frozen_parameters_changed": False,
            "reason": (
                "Both field and role weights are behaviorally identifiable on frozen public "
                "real-model scores, but this post-hoc cross-domain sensitivity does not prove "
                "FRC superiority or flood-domain validity."
            ),
        },
        "limitations": [
            "HousingQA is a public housing-law benchmark, not a flood-response benchmark.",
            "Field evidence comes from public expert question/statute mappings; the three FRC functional roles are deterministic evaluation labels rather than HousingQA expert role annotations.",
            "The sweep reuses frozen real-model scores and evaluates evidence selection only; it does not rerun answer generation for every weight setting.",
            "The sweep is descriptive and post-hoc; it cannot authorize parameter tuning or Gate 2 promotion.",
        ],
    }
    return report, case_results


def render_housing_weight_sensitivity_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# HousingQA public real-model field/role-weight sensitivity",
        "",
        f"- Status: `{report['status']}`",
        f"- Source: `{metadata['source_repository']}@{metadata['source_revision']}` "
        f"({metadata['source_license']})",
        f"- Cases / fields / jurisdictions: {metadata['case_count']} / "
        f"{metadata['field_count']} / {metadata['jurisdiction_count']}",
        f"- Models: `{metadata['models']['embedding']}` + "
        f"`{metadata['models']['reranker']}`",
        f"- Gold usage: {metadata['gold_usage']}",
        "- This is a frozen-score, one-factor descriptive sweep; it is not parameter tuning.",
        "",
        "| Dimension | Value | Frozen | Changed cases | Evidence F1 | Field coverage | "
        "Role coverage | Tokens |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        aggregate = row["aggregate"]
        lines.append(
            f"| {row['dimension']} | {row['value']:.2f} | "
            f"{'yes' if row['is_frozen'] else 'no'} | "
            f"{row['selection_changed_cases_from_frozen']} | "
            f"{aggregate['evidence_f1']:.6f} | {aggregate['field_coverage']:.6f} | "
            f"{aggregate['role_coverage']:.6f} | {aggregate['token_cost']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- Gate 2: `{report['decision']['gate_2']}`.",
            f"- {report['decision']['reason']}",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)
