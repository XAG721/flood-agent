"""Prospective case-disjoint IIRC score-gap router experiment (v87)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.iirc_cardinality_transfer import (
    CANDIDATE_METHOD as V86_CANDIDATE_METHOD,
    CONTROL_METHODS as V86_CONTROL_METHODS,
    PARTITION_COMMITMENTS as V86_PARTITION_COMMITMENTS,
    _aggregate,
    _article_question_map,
    _case_metrics,
    _read_iirc,
    _stratum_name,
    build_partitions as build_v86_partitions,
    load_context_articles,
    prepare_case,
    requested_titles_for_stage,
)
from research.frc_rag.iirc_cardinality_transfer import (
    select_all_methods as select_v86_methods,
)
from research.frc_rag.iirc_cardinality_transfer_router import (
    load_model_artifact as load_v86_model,
)
from research.frc_rag.iirc_score_gap_router import (
    load_model_artifact as load_v87_model,
)
from research.frc_rag.iirc_score_gap_router import select_method as select_v87_method
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-IIRC-SCORE-GAP-ROUTER-PROSPECTIVE-V87"
SCHEMA_VERSION = "frc-iirc-score-gap-transfer-v87"
PARTITION_SALT = "FRC-IIRC-V87-REMAINING-PARTITION|"
STAGES = ("development", "confirmation")
STAGE_CASES = {"development": 400, "confirmation": 800}
STAGE_OFFSETS = {"development": 0, "confirmation": 400}
MAX_SOURCES = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"development": 20260809, "confirmation": 20260810}
PARTITION_COMMITMENTS = {
    "eligible": "adf053155b954245052da5854d3ba239f6f01ae8de6cdad4ba98423a7438ca2e",
    "development": "b72274afdc26ff3c817a2a8456503567f767b80f828afc7f4a3926ad2e971167",
    "confirmation": "08ff51fe0917291bcc2a1dbf1b6ebdb1d724bc968d00ca4a3accd03f05abcc86",
    "remaining": "0681233dd79b5a21d717fd9e541b404b166673b6d08b24154825b321249a2d64",
}
CANDIDATE_METHOD = "iirc_score_gap_frc_v87"
CONTROL_METHODS = (*V86_CONTROL_METHODS, V86_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
STRICT_GATES = {
    "invalid_selector_output_rate_at_most": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.62,
    "candidate_complete_evidence_recall_at_least": 0.42,
    "candidate_minus_strongest_control_f1_at_least": 0.005,
    "candidate_minus_strongest_control_ci_low_above": 0.0,
    "candidate_minus_frc_fixed2_f1_at_least": 0.005,
    "candidate_minus_frc_fixed2_ci_low_above": 0.0,
    "candidate_complete_recall_delta_vs_strongest_at_least": -0.02,
    "candidate_evidence_recall_delta_vs_strongest_at_least": -0.02,
    "candidate_mean_selected_sources_above_strongest_at_most": 0.25,
    "distinct_candidate_actions_at_least": 2,
    "second_largest_action_fraction_at_least": 0.05,
    "supported_stratum_delta_vs_strongest_at_least": -0.03,
    "supported_stratum_min_cases": 40,
}


def _commit(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_partitions(train_rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    v86 = build_v86_partitions(train_rows)
    if _commit(v86["remaining"]) != V86_PARTITION_COMMITMENTS["remaining"]:
        raise ValueError("v87 eligible pool no longer matches the v86 remainder")
    eligible = sorted(v86["remaining"])
    ordered = sorted(
        eligible,
        key=lambda value: (
            hashlib.sha256(f"{PARTITION_SALT}{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    partitions = {
        "eligible": eligible,
        "development": ordered[: STAGE_CASES["development"]],
        "confirmation": ordered[
            STAGE_OFFSETS["confirmation"] : STAGE_OFFSETS["confirmation"]
            + STAGE_CASES["confirmation"]
        ],
        "remaining": ordered[sum(STAGE_CASES.values()) :],
    }
    for name, values in partitions.items():
        if _commit(values) != PARTITION_COMMITMENTS[name]:
            raise ValueError(f"v87 partition commitment changed: {name}")
    if set(partitions["development"]) & set(partitions["confirmation"]):
        raise ValueError("v87 stages overlap")
    if set(partitions["eligible"]) & set(v86["development"] + v86["confirmation"]):
        raise ValueError("v87 eligible pool overlaps a v86 reserved partition")
    return partitions


def prepare_stage(
    train_path: Path,
    context_json: Path,
    context_index: Path,
    tokenizer: Any,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v87 stage: {stage}")
    train_rows = _read_iirc(train_path)
    selected_ids = build_partitions(train_rows)[stage]
    mapping = _article_question_map(train_rows, selected_ids)
    requested_titles = requested_titles_for_stage(train_rows, selected_ids)
    articles = load_context_articles(context_json, context_index, requested_titles)
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, int]] = []
    for qid in selected_ids:
        blind, gold, census = prepare_case(
            *mapping[qid], articles, tokenizer, stage=stage
        )
        blind["source"] = f"iirc_train_v87_{stage}"
        blind_rows.append(blind)
        gold_rows.append(gold)
        census_rows.append(census)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "selected_ids_sha256": PARTITION_COMMITMENTS[stage],
        "blind_sha256": sha256(blind_path),
        "sealed_gold_sha256": sha256(gold_path),
        "requested_unique_context_titles": len(requested_titles),
        "resolved_unique_context_titles": len(articles),
        "candidate_sources": {
            "minimum": min(row["candidate_sources"] for row in census_rows),
            "mean": round(
                float(np.mean([row["candidate_sources"] for row in census_rows])), 6
            ),
            "maximum": max(row["candidate_sources"] for row in census_rows),
        },
        "unresolved_link_sources": sum(
            row["unresolved_link_sources"] for row in census_rows
        ),
        "gold_statistics_sealed_not_reported": True,
        "overlap_with_v86_development_or_confirmation": 0,
        "development_confirmation_overlap": 0,
    }


def write_selection_outputs(
    scored_path: Path,
    v86_model_path: Path,
    v87_model_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    v86_model = load_v86_model(v86_model_path)
    v87_model = load_v87_model(v87_model_path)
    rows = read_jsonl(scored_path)
    forbidden = {
        "answer",
        "answer_type",
        "context",
        "gold_evidence_sources",
        "question_links",
    }
    if any(forbidden & set(row) for row in rows):
        raise ValueError("v87 scored cache contains a forbidden gold field")
    outputs = []
    for row in rows:
        methods = select_v86_methods(row, v86_model)
        methods[CANDIDATE_METHOD] = select_v87_method(row, v87_model)
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_sources": len(
                    {str(candidate["source"]) for candidate in row["candidates"]}
                ),
                "methods": methods,
            }
        )
    write_jsonl(output_path, outputs)
    return outputs


def _paired_bootstrap(
    candidate: np.ndarray,
    controls: dict[str, np.ndarray],
    *,
    seed: int,
    strongest_inside_resample: bool,
) -> dict[str, Any]:
    point_control = max(
        controls, key=lambda name: (float(np.mean(controls[name])), name)
    )
    point = float(np.mean(candidate) - np.mean(controls[point_control]))
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    names = sorted(controls)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(candidate), len(candidate))
        control_mean = (
            max(float(np.mean(controls[name][sample])) for name in names)
            if strongest_inside_resample
            else float(np.mean(controls[point_control][sample]))
        )
        values[index] = float(np.mean(candidate[sample])) - control_mean
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
        "point_strongest_control": point_control,
        "strongest_selected_inside_each_resample": strongest_inside_resample,
    }


def evaluate_stage(
    selection_path: Path,
    gold_path: Path,
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v87 stage: {stage}")
    selections = read_jsonl(selection_path)
    gold_rows = read_jsonl(gold_path)
    selection_by_id = {str(row["case_id"]): row for row in selections}
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    expected = STAGE_CASES[stage]
    if (
        len(selections) != expected
        or len(gold_rows) != expected
        or len(selection_by_id) != expected
        or len(gold_by_id) != expected
        or set(selection_by_id) != set(gold_by_id)
    ):
        raise ValueError("v87 selection/gold coverage changed")
    case_rows: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        methods = selection_by_id[case_id].get("methods", {})
        if set(methods) != set(METHODS):
            raise ValueError("v87 selection methods changed")
        metrics: dict[str, Any] = {}
        for method in METHODS:
            selected = [
                str(value) for value in methods[method].get("selected_sources", [])
            ]
            cardinality = int(methods[method].get("cardinality", 0))
            if (
                not 1 <= cardinality <= MAX_SOURCES
                or cardinality != len(selected)
                or len(selected) != len(set(selected))
            ):
                invalid += 1
            metrics[method] = {
                **_case_metrics(gold["gold_evidence_sources"], selected),
                "selected_sources": cardinality,
            }
        case_rows.append(
            {
                "case_id": case_id,
                "answer_type": gold["answer_type"],
                "gold_cardinality": len(gold["gold_evidence_sources"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "strata": _stratum_name(gold),
                "methods": metrics,
                "candidate_action": int(methods[CANDIDATE_METHOD]["cardinality"]),
                "candidate_predicted_utilities": methods[CANDIDATE_METHOD].get(
                    "predicted_utilities", {}
                ),
            }
        )
    aggregates = {
        method: {
            **_aggregate([row["methods"][method] for row in case_rows]),
            "mean_selected_sources": round(
                float(
                    np.mean(
                        [
                            row["methods"][method]["selected_sources"]
                            for row in case_rows
                        ]
                    )
                ),
                6,
            ),
        }
        for method in METHODS
    }
    f1_arrays = {
        method: np.asarray(
            [row["methods"][method]["evidence_f1"] for row in case_rows]
        )
        for method in METHODS
    }
    strongest = max(
        CONTROL_METHODS,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    simultaneous = _paired_bootstrap(
        f1_arrays[CANDIDATE_METHOD],
        {method: f1_arrays[method] for method in CONTROL_METHODS},
        seed=BOOTSTRAP_SEEDS[stage],
        strongest_inside_resample=True,
    )
    frc2 = _paired_bootstrap(
        f1_arrays[CANDIDATE_METHOD],
        {"frc_fixed2": f1_arrays["frc_fixed2"]},
        seed=BOOTSTRAP_SEEDS[stage] + 100,
        strongest_inside_resample=False,
    )
    strata: dict[str, Any] = {}
    for name in sorted({name for row in case_rows for name in row["strata"]}):
        local = [row for row in case_rows if name in row["strata"]]
        candidate_value = float(
            np.mean(
                [row["methods"][CANDIDATE_METHOD]["evidence_f1"] for row in local]
            )
        )
        strongest_value = float(
            np.mean([row["methods"][strongest]["evidence_f1"] for row in local])
        )
        strata[name] = {
            "cases": len(local),
            "candidate_f1": round(candidate_value, 6),
            "strongest_control_f1": round(strongest_value, 6),
            "delta": round(candidate_value - strongest_value, 6),
        }
    actions = Counter(row["candidate_action"] for row in case_rows)
    action_fractions = {
        str(key): value / expected for key, value in sorted(actions.items())
    }
    ordered_action_counts = sorted(actions.values(), reverse=True)
    second_fraction = (
        ordered_action_counts[1] / expected
        if len(ordered_action_counts) >= 2
        else 0.0
    )
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_metrics = aggregates[strongest]
    supported_deltas = [
        value["delta"]
        for value in strata.values()
        if value["cases"] >= STRICT_GATES["supported_stratum_min_cases"]
    ]
    checks = {
        "exact_cases": len(case_rows) == expected,
        "invalid_selector_output_rate_at_most_0": invalid
        / (expected * len(METHODS))
        <= 0,
        "candidate_evidence_macro_f1_at_least_0_62": candidate[
            "evidence_macro_f1"
        ]
        >= 0.62,
        "candidate_complete_evidence_recall_at_least_0_42": candidate[
            "complete_evidence_recall"
        ]
        >= 0.42,
        "candidate_minus_strongest_control_f1_at_least_0_005": simultaneous[
            "point"
        ]
        >= 0.005,
        "candidate_minus_strongest_control_ci_low_above_0": simultaneous["ci_low"]
        > 0.0,
        "candidate_minus_frc_fixed2_f1_at_least_0_005": frc2["point"] >= 0.005,
        "candidate_minus_frc_fixed2_ci_low_above_0": frc2["ci_low"] > 0.0,
        "candidate_complete_recall_delta_vs_strongest_at_least_minus_0_02": candidate[
            "complete_evidence_recall"
        ]
        - strongest_metrics["complete_evidence_recall"]
        >= -0.02,
        "candidate_evidence_recall_delta_vs_strongest_at_least_minus_0_02": candidate[
            "evidence_macro_recall"
        ]
        - strongest_metrics["evidence_macro_recall"]
        >= -0.02,
        "candidate_mean_selected_sources_above_strongest_at_most_0_25": candidate[
            "mean_selected_sources"
        ]
        - strongest_metrics["mean_selected_sources"]
        <= 0.25,
        "distinct_candidate_actions_at_least_2": len(actions) >= 2,
        "second_largest_action_fraction_at_least_0_05": second_fraction >= 0.05,
        "supported_stratum_delta_vs_strongest_at_least_minus_0_03": bool(
            supported_deltas
        )
        and min(supported_deltas) >= -0.03,
        "prior_or_stage_overlap_0": True,
    }
    all_pass = all(checks.values())
    status = (
        "IIRC_V87_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
        if stage == "development" and all_pass
        else "IIRC_V87_CASE_DISJOINT_SUPPORT_ESTABLISHED"
        if stage == "confirmation" and all_pass
        else f"IIRC_V87_{stage.upper()}_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-iirc-score-gap-transfer-result-v87",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "cases": expected,
        "aggregates": aggregates,
        "strongest_control": strongest,
        "candidate_minus_strongest_control": simultaneous,
        "candidate_minus_frc_fixed2": frc2,
        "candidate_action_counts": dict(
            sorted((str(key), value) for key, value in actions.items())
        ),
        "candidate_action_fractions": action_fractions,
        "strata": strata,
        "checks": checks,
        "all_strict_gates_pass": all_pass,
        "status": status,
        "gate_2": "NO-GO/SHADOW",
        "claim_limits": {
            "case_disjoint_same_dataset_prospective_test": True,
            "independent_cross_dataset_transfer": False,
            "answer_generation_evaluated": False,
            "setr_reproduced": False,
            "flood_domain_effectiveness_established": False,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
        },
    }
    return report, case_rows


def write_report(
    report: dict[str, Any], case_rows: Sequence[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iirc_score_gap_transfer_{report['stage']}_v87"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    cases_path = output_dir / f"{stem}_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    candidate = report["aggregates"][CANDIDATE_METHOD]
    delta = report["candidate_minus_strongest_control"]
    markdown_path.write_text(
        "\n".join(
            [
                f"# IIRC v87 {report['stage']} result",
                "",
                f"- Status: `{report['status']}`",
                f"- Cases: {report['cases']}",
                f"- Candidate F1: {candidate['evidence_macro_f1']:.6f}",
                f"- Strongest control: `{report['strongest_control']}`",
                f"- Delta: {delta['point']:.6f} "
                f"(95% CI {delta['ci_low']:.6f}, {delta['ci_high']:.6f})",
                f"- Strict gates: {'PASS' if report['all_strict_gates_pass'] else 'FAIL'}",
                "- Gate 2: `NO-GO/SHADOW`",
                "",
                "This is a case-disjoint same-dataset source-selection test, not independent cross-dataset transfer, SetR reproduction, answer-generation evidence, flood-domain validation, or production authorization.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_jsonl_gzip(cases_path, list(case_rows))
    return {"json": json_path, "markdown": markdown_path, "cases": cases_path}
