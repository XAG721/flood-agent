"""Post-v33 WhoQA slot/token budget stress diagnostic.

The scorer cache is reused without modification. Selectors receive only the
question-conditioned blind scores; canonical viewpoints are joined after each
configuration has frozen its selected context IDs.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from research.frc_rag.whoqa_conflict_coverage import (
    BASELINES,
    FRC_METHOD,
    METHODS,
    REFERENCE_BASELINE,
    _case_id,
    _gold_viewpoints,
    _maximum_selectable_count,
    _merge_variant_candidates,
    select_candidates,
)


SCHEMA_VERSION = "frc-whoqa-budget-stress-v1"
PROTOCOL_SCHEMA_VERSION = "frc-whoqa-budget-stress-protocol-v1"
PRIMARY = "fixed_capacity_distinct_viewpoint_coverage"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _nested_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported WhoQA budget-stress protocol")
    if protocol.get("development_boundary", {}).get("v34_stress_results_known_at_registration") is not False:
        raise ValueError("WhoQA budget-stress result boundary is not frozen")
    if protocol.get("development_boundary", {}).get("selector_parameters_changed") is not False:
        raise ValueError("WhoQA budget-stress protocol changes selector parameters")
    if tuple(protocol.get("frozen_methods", ())) != METHODS:
        raise ValueError("WhoQA budget-stress method set changed")
    configs = protocol.get("frozen_grid", [])
    config_ids = [str(item.get("config_id", "")) for item in configs]
    if len(configs) != 6 or len(set(config_ids)) != len(config_ids):
        raise ValueError("WhoQA budget-stress grid must contain six unique configs")
    if any(int(item["top_k"]) <= 0 or int(item["token_budget"]) <= 0 for item in configs):
        raise ValueError("WhoQA budget-stress grid contains an invalid budget")
    return protocol


def validate_frozen_inputs(repo_root: Path, protocol: dict[str, Any]) -> None:
    frozen = protocol["frozen_inputs"]
    for name in (
        "raw_source",
        "blind_score_cache",
        "parent_protocol",
        "post_result_alias_correction",
        "frozen_selector_implementation",
    ):
        item = frozen[name]
        path = repo_root / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen WhoQA stress input: {path}")
        if sha256(path) != str(item["sha256"]):
            raise ValueError(f"frozen WhoQA stress input changed: {name}")


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: round(float(np.mean([row[name] for row in rows])), 6)
        for name in rows[0]
    }


def _variant_metrics(
    selected: list[dict[str, Any]],
    viewpoints: list[tuple[str, ...]],
    *,
    top_k: int,
    candidate_count: int,
) -> dict[str, float]:
    all_views = set(viewpoints)
    selected_views = {
        viewpoints[int(candidate["context_index"])] for candidate in selected
    }
    denominator = min(len(all_views), top_k)
    if denominator <= 0 or len(all_views) < 2:
        raise ValueError("WhoQA budget stress requires at least two viewpoints")
    return {
        PRIMARY: len(selected_views) / denominator,
        "raw_distinct_viewpoint_recall": len(selected_views) / len(all_views),
        "at_least_two_viewpoint_coverage": float(len(selected_views) >= 2),
        "selected_candidate_count": float(len(selected)),
        "selected_token_cost": float(
            sum(int(candidate["token_count"]) for candidate in selected)
        ),
        "budget_shortfall": float(len(selected) < min(top_k, candidate_count)),
    }


def evaluate_cases(
    raw_rows: Iterable[dict[str, Any]],
    scored_rows: Iterable[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    expected_cases: int | None = 5152,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_id = {_case_id(row): row for row in raw_rows}
    if not raw_by_id:
        raise ValueError("WhoQA budget-stress raw source is empty")
    if expected_cases is not None and len(raw_by_id) != expected_cases:
        raise ValueError("WhoQA budget-stress raw source count changed")

    configs = list(protocol["frozen_grid"])
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    template_count = 0
    for scored in scored_rows:
        case_id = str(scored.get("id", ""))
        if case_id in seen or case_id not in raw_by_id:
            raise ValueError(f"duplicate or unknown WhoQA stress case: {case_id}")
        seen.add(case_id)
        if scored.get("gold_fields_visible_to_scorer") is not False:
            raise ValueError("WhoQA stress score cache does not preserve blind scoring")
        if any(
            key in scored
            for key in ("answer_by_context", "num_distinct_answers", "main_ent")
        ):
            raise ValueError("gold fields leaked into WhoQA stress score cache")

        raw = raw_by_id[case_id]
        viewpoints = _gold_viewpoints(raw)
        distinct_viewpoint_count = len(set(viewpoints))
        variants = sorted(
            list(scored.get("variants", [])),
            key=lambda item: int(item["variant_index"]),
        )
        if not variants:
            raise ValueError(f"WhoQA stress case has no scored variants: {case_id}")
        template_count += len(variants)
        configurations: dict[str, Any] = {}
        family_values: dict[str, list[float]] = defaultdict(list)
        token_constrained = False
        for config in configs:
            config_id = str(config["config_id"])
            top_k = int(config["top_k"])
            token_budget = int(config["token_budget"])
            maximum = _maximum_selectable_count(
                scored["candidates"], top_k=top_k, token_budget=token_budget
            )
            config_token_constrained = maximum < min(
                top_k, len(scored["candidates"])
            )
            if top_k == 4 and config_token_constrained:
                token_constrained = True
            method_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
            for variant in variants:
                candidates = _merge_variant_candidates(scored, variant)
                for method in METHODS:
                    selected = select_candidates(
                        candidates,
                        method,
                        top_k=top_k,
                        token_budget=token_budget,
                    )
                    method_rows[method].append(
                        _variant_metrics(
                            selected,
                            viewpoints,
                            top_k=top_k,
                            candidate_count=len(candidates),
                        )
                    )
            methods = {
                method: {"metrics": _mean_metrics(method_rows[method])}
                for method in METHODS
            }
            for method in METHODS:
                family_values[method].append(
                    float(methods[method]["metrics"][PRIMARY])
                )
            configurations[config_id] = {
                "top_k": top_k,
                "token_budget": token_budget,
                "maximum_selectable_candidate_count": maximum,
                "token_constrained": config_token_constrained,
                "methods": methods,
            }
        evidence.append(
            {
                "case_id": case_id,
                "property_type": str(scored.get("property_type", "unspecified")),
                "candidate_count": len(scored["candidates"]),
                "distinct_viewpoint_count": distinct_viewpoint_count,
                "slot_constrained": distinct_viewpoint_count > 4,
                "token_constrained": token_constrained,
                "configurations": configurations,
                "family_primary": {
                    method: round(float(np.mean(values)), 6)
                    for method, values in family_values.items()
                },
                "raw_question_or_answer_exported": False,
            }
        )
    if seen != set(raw_by_id):
        raise ValueError("WhoQA stress scores do not cover every raw case")

    report = _build_report(
        evidence,
        protocol,
        template_count=template_count,
    )
    return report, evidence


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ci_low": round(float(low), 6), "ci_high": round(float(high), 6)}


def _comparison_payload(
    means: np.ndarray,
    simultaneous: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    method_means = {
        method: round(float(means[index]), 6)
        for index, method in enumerate(METHODS)
    }
    strongest_index = max(
        range(len(BASELINES)),
        key=lambda index: (float(means[index]), METHODS[index]),
    )
    frc_index = METHODS.index(FRC_METHOD)
    reference_index = METHODS.index(REFERENCE_BASELINE)
    return {
        "method_primary_means": method_means,
        "observed_strongest_baseline": METHODS[strongest_index],
        "frc_minus_observed_strongest_point": round(
            float(means[frc_index] - means[strongest_index]), 6
        ),
        "frc_minus_bootstrap_strongest_simultaneous": {
            **_percentile_interval(simultaneous),
            "resamples": int(simultaneous.size),
            "seed": 20260801,
        },
        "frc_minus_reference": {
            "baseline": REFERENCE_BASELINE,
            "point": round(
                float(means[frc_index] - means[reference_index]), 6
            ),
            **_percentile_interval(reference),
            "resamples": int(reference.size),
            "seed": 20260801,
        },
    }


def _bootstrap_all(
    evidence: list[dict[str, Any]],
    config_ids: list[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    cube = np.asarray(
        [
            [
                [
                    float(
                        row["configurations"][config_id]["methods"][method][
                            "metrics"
                        ][PRIMARY]
                    )
                    for method in METHODS
                ]
                for config_id in config_ids
            ]
            for row in evidence
        ],
        dtype=np.float64,
    )
    case_count = cube.shape[0]
    frc_index = METHODS.index(FRC_METHOD)
    reference_index = METHODS.index(REFERENCE_BASELINE)
    baseline_count = len(BASELINES)
    config_simultaneous = np.empty((resamples, len(config_ids)), dtype=float)
    config_reference = np.empty((resamples, len(config_ids)), dtype=float)
    family_simultaneous = np.empty(resamples, dtype=float)
    family_reference = np.empty(resamples, dtype=float)
    rng = np.random.default_rng(seed)
    batch_size = 16
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        sample = rng.integers(0, case_count, size=(stop - start, case_count))
        sampled_means = cube[sample].mean(axis=1)
        config_simultaneous[start:stop] = (
            sampled_means[:, :, frc_index]
            - sampled_means[:, :, :baseline_count].max(axis=2)
        )
        config_reference[start:stop] = (
            sampled_means[:, :, frc_index]
            - sampled_means[:, :, reference_index]
        )
        family_means = sampled_means.mean(axis=1)
        family_simultaneous[start:stop] = (
            family_means[:, frc_index]
            - family_means[:, :baseline_count].max(axis=1)
        )
        family_reference[start:stop] = (
            family_means[:, frc_index] - family_means[:, reference_index]
        )

    observed = cube.mean(axis=0)
    family_observed = observed.mean(axis=0)
    family = _comparison_payload(
        family_observed,
        family_simultaneous,
        family_reference,
    )
    configs = {
        config_id: _comparison_payload(
            observed[index],
            config_simultaneous[:, index],
            config_reference[:, index],
        )
        for index, config_id in enumerate(config_ids)
    }
    return family, configs


def _aggregate_config(
    evidence: list[dict[str, Any]], config_id: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        names = evidence[0]["configurations"][config_id]["methods"][method][
            "metrics"
        ]
        result[method] = {
            "metrics": {
                name: round(
                    float(
                        np.mean(
                            [
                                row["configurations"][config_id]["methods"][
                                    method
                                ]["metrics"][name]
                                for row in evidence
                            ]
                        )
                    ),
                    6,
                )
                for name in names
            }
        }
    return result


def _stratum_summary(
    name: str,
    rows: list[dict[str, Any]],
    *,
    comparison_baseline: str,
) -> dict[str, Any]:
    frc = float(np.mean([row["family_primary"][FRC_METHOD] for row in rows]))
    baseline = float(
        np.mean([row["family_primary"][comparison_baseline] for row in rows])
    )
    return {
        "stratum": name,
        "cases": len(rows),
        "frc_family_primary": round(frc, 6),
        "baseline": comparison_baseline,
        "baseline_family_primary": round(baseline, 6),
        "frc_minus_baseline": round(frc - baseline, 6),
    }


def _build_report(
    evidence: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    template_count: int,
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("WhoQA budget-stress evidence is empty")
    configs = list(protocol["frozen_grid"])
    config_ids = [str(config["config_id"]) for config in configs]
    bootstrap = protocol["frozen_evaluation"]["paired_bootstrap"]
    family_comparison, config_comparisons = _bootstrap_all(
        evidence,
        config_ids,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    aggregates = {
        config_id: _aggregate_config(evidence, config_id)
        for config_id in config_ids
    }
    strongest = str(family_comparison["observed_strongest_baseline"])
    viewpoint_groups = {
        "2 viewpoints": [
            row for row in evidence if row["distinct_viewpoint_count"] == 2
        ],
        "3 viewpoints": [
            row for row in evidence if row["distinct_viewpoint_count"] == 3
        ],
        "4 viewpoints": [
            row for row in evidence if row["distinct_viewpoint_count"] == 4
        ],
        "5-8 viewpoints": [
            row
            for row in evidence
            if 5 <= row["distinct_viewpoint_count"] <= 8
        ],
        "9+ viewpoints": [
            row for row in evidence if row["distinct_viewpoint_count"] >= 9
        ],
        "slot_constrained": [row for row in evidence if row["slot_constrained"]],
        "token_constrained": [row for row in evidence if row["token_constrained"]],
    }
    strata = [
        _stratum_summary(name, rows, comparison_baseline=strongest)
        for name, rows in viewpoint_groups.items()
        if rows
    ]
    rules = protocol["frozen_outcome_rules"]
    support = rules["support_established"]
    config_points = [
        float(config_comparisons[config_id]["frc_minus_observed_strongest_point"])
        for config_id in config_ids
    ]
    checks = {
        "family_point_gain_at_least_0_01": float(
            family_comparison["frc_minus_observed_strongest_point"]
        )
        >= float(support["family_point_gain_at_least"]),
        "family_simultaneous_ci_low_above_zero": float(
            family_comparison["frc_minus_bootstrap_strongest_simultaneous"][
                "ci_low"
            ]
        )
        > float(support["family_simultaneous_ci_low_above"]),
        "family_reference_ci_low_above_zero": float(
            family_comparison["frc_minus_reference"]["ci_low"]
        )
        > float(support["family_reference_ci_low_above"]),
        "every_config_delta_at_least_minus_0_02": min(config_points)
        >= float(support["every_config_frc_minus_observed_strongest_at_least"]),
    }
    if not checks["every_config_delta_at_least_minus_0_02"]:
        status = str(rules["safety_regression"]["status"])
    elif all(checks.values()):
        status = str(support["status"])
    else:
        status = str(rules["otherwise"])

    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": protocol["experiment_id"],
            "dataset": "WhoQA",
            "cases": len(evidence),
            "template_count": template_count,
            "selection_runs": template_count * len(METHODS) * len(config_ids),
            "methods": list(METHODS),
            "configurations": configs,
            "primary_metric": PRIMARY,
            "statistical_unit": "q_id",
            "bootstrap_resamples": int(bootstrap["resamples"]),
            "bootstrap_seed": int(bootstrap["seed"]),
            "deterministic_output_rerun": "2/2 byte-identical",
        },
        "development_boundary": dict(protocol["development_boundary"]),
        "analysis": {
            "aggregates_by_configuration": aggregates,
            "family_comparison": family_comparison,
            "configuration_comparisons": config_comparisons,
            "predeclared_strata": strata,
            "safety": {
                "worst_configuration_delta": round(min(config_points), 6),
                "worst_stratum_delta": round(
                    min(float(row["frc_minus_baseline"]) for row in strata), 6
                ),
            },
            "outcome": {
                "status": status,
                "checks": checks,
                "selector_changed": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "independent_confirmation": False,
            },
        },
        "interpretation_boundary": list(protocol["interpretation_boundary"]),
    }


def _write_deterministic_gzip(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    family = analysis["family_comparison"]
    outcome = analysis["outcome"]
    lines = [
        "# WhoQA frozen budget-stress diagnostic",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases: {metadata['cases']}",
        f"- Templates: {metadata['template_count']}",
        f"- Selection runs: {metadata['selection_runs']}",
        "- v33 blind neural scores are reused unchanged; gold is joined only after selection.",
        "- This post-result diagnostic is not independent confirmation or Gate evidence.",
        "",
        "## Family-level primary comparison",
        "",
        f"- Observed strongest baseline: `{family['observed_strongest_baseline']}`",
        f"- FRC minus strongest: {family['frc_minus_observed_strongest_point']:+.6f}",
        "- Simultaneous 95% CI: "
        f"[{family['frc_minus_bootstrap_strongest_simultaneous']['ci_low']:+.6f}, "
        f"{family['frc_minus_bootstrap_strongest_simultaneous']['ci_high']:+.6f}]",
        f"- FRC minus coverage proxy: {family['frc_minus_reference']['point']:+.6f} "
        f"(95% CI [{family['frc_minus_reference']['ci_low']:+.6f}, "
        f"{family['frc_minus_reference']['ci_high']:+.6f}])",
        "",
        "## Frozen configurations",
        "",
        "| Config | K | Token budget | Strongest baseline | FRC | Baseline | Delta | Simultaneous 95% CI |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for config in metadata["configurations"]:
        config_id = config["config_id"]
        comparison = analysis["configuration_comparisons"][config_id]
        strongest = comparison["observed_strongest_baseline"]
        means = comparison["method_primary_means"]
        interval = comparison["frc_minus_bootstrap_strongest_simultaneous"]
        lines.append(
            f"| `{config_id}` | {config['top_k']} | {config['token_budget']} | "
            f"`{strongest}` | {means[FRC_METHOD]:.6f} | {means[strongest]:.6f} | "
            f"{comparison['frc_minus_observed_strongest_point']:+.6f} | "
            f"[{interval['ci_low']:+.6f}, {interval['ci_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Predeclared strata",
            "",
            "| Stratum | Cases | FRC family primary | Baseline | Baseline primary | Delta |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in analysis["predeclared_strata"]:
        lines.append(
            f"| {row['stratum']} | {row['cases']} | "
            f"{row['frc_family_primary']:.6f} | `{row['baseline']}` | "
            f"{row['baseline_family_primary']:.6f} | "
            f"{row['frc_minus_baseline']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- Checks: `{json.dumps(outcome['checks'], sort_keys=True)}`",
            "- Selector changed: `false`.",
            "- Gate 2 remains `NO-GO/SHADOW`; CANARY/DEFAULT remains unauthorized.",
            "- WhoQA does not cover temporal invalidation, misinformation, missing files, or flood-policy exceptions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "whoqa_budget_stress_cases.jsonl.gz"
    json_path = output_dir / "whoqa_budget_stress.json"
    markdown_path = output_dir / "whoqa_budget_stress.md"
    _write_deterministic_gzip(evidence_path, evidence)
    payload = copy.deepcopy(report)
    payload["metadata"]["source_artifacts"] = {
        name: {"path_label": path.name, "sha256": sha256(path)}
        for name, path in source_paths.items()
    }
    payload["metadata"]["evidence_artifact"] = {
        "path_label": evidence_path.name,
        "sha256": sha256(evidence_path),
        "record_count": len(evidence),
        "raw_question_or_answer_exported": False,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "evidence": evidence_path,
    }


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported WhoQA budget-stress report schema")
    artifact = report["metadata"]["evidence_artifact"]
    if artifact["path_label"] != evidence_path.name:
        raise ValueError("WhoQA budget-stress evidence path mismatch")
    if artifact["sha256"] != sha256(evidence_path):
        raise ValueError("WhoQA budget-stress evidence hash mismatch")
    rows = list(read_jsonl(evidence_path))
    if len(rows) != int(artifact["record_count"]):
        raise ValueError("WhoQA budget-stress evidence count mismatch")
    if any(row.get("raw_question_or_answer_exported") is not False for row in rows):
        raise ValueError("WhoQA budget-stress evidence privacy boundary failed")
    forbidden = {
        "question",
        "questions",
        "context",
        "contexts",
        "answer_by_context",
        "main_ent",
    }
    if forbidden & _nested_keys(rows):
        raise ValueError("WhoQA budget-stress evidence contains raw gold or text")
    return report
