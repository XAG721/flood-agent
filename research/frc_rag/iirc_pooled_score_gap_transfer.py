"""Prospective higher-power pooled IIRC score-gap experiment (v88)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.iirc_cardinality_transfer import (
    _aggregate,
    _article_question_map,
    _case_metrics,
    _read_iirc,
    _stratum_name,
    load_context_articles,
    prepare_case,
    requested_titles_for_stage,
)
from research.frc_rag.iirc_pooled_score_gap_router import (
    load_model_artifact as load_v88_model,
)
from research.frc_rag.iirc_pooled_score_gap_router import (
    select_method as select_v88_method,
)
from research.frc_rag.iirc_score_gap_transfer import (
    CANDIDATE_METHOD as V87_CANDIDATE_METHOD,
    CONTROL_METHODS as V87_CONTROL_METHODS,
    PARTITION_COMMITMENTS as V87_PARTITION_COMMITMENTS,
    build_partitions as build_v87_partitions,
)
from research.frc_rag.iirc_score_gap_transfer import (
    write_selection_outputs as select_v87_methods_to_file,
)
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-IIRC-POOLED-SCORE-GAP-PROSPECTIVE-V88"
SCHEMA_VERSION = "frc-iirc-pooled-score-gap-transfer-v88"
PARTITION_SALT = "FRC-IIRC-V88-REMAINING-PARTITION|"
STAGES = ("development", "confirmation")
STAGE_CASES = {"development": 800, "confirmation": 1200}
STAGE_OFFSETS = {"development": 0, "confirmation": 800}
MAX_SOURCES = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"development": 20260812, "confirmation": 20260813}
PARTITION_COMMITMENTS = {
    "eligible": "8c7902360a17a233691deaf5d5b431c85af80ca15d0ff4a869c86d56cb19ca9a",
    "development": "22682d7eb60fef4500529ed21c22642764ba4b2c33431f393d646952f652a0d5",
    "confirmation": "48ad6e475eacf913d9de29bdcba6756e9bb8cd0eb228c656ecdac331280238a0",
    "remaining": "88df288fb0d01f0779dbe20783b04ff4ea41702854e71d8263b9bb35d98eb04d",
}
PRIMARY_CONTROL = "frc_fixed2"
CANDIDATE_METHOD = "iirc_pooled_score_gap_frc_v88"
CONTROL_METHODS = (*V87_CONTROL_METHODS, V87_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
STRICT_GATES = {
    "invalid_selector_output_rate_at_most": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.60,
    "candidate_complete_evidence_recall_at_least": 0.42,
    "candidate_minus_primary_control_f1_at_least": 0.005,
    "candidate_minus_primary_control_ci_low_above": 0.0,
    "candidate_min_point_delta_vs_every_control_at_least": 0.0,
    "candidate_complete_recall_delta_vs_primary_at_least": -0.02,
    "candidate_evidence_recall_delta_vs_primary_at_least": -0.02,
    "candidate_mean_selected_sources_above_primary_at_most": 0.25,
    "distinct_candidate_actions_at_least": 2,
    "second_largest_action_fraction_at_least": 0.05,
    "supported_stratum_delta_vs_primary_at_least": -0.03,
    "supported_stratum_min_cases": 60,
}


def _commit(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_partitions(train_rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    v87 = build_v87_partitions(train_rows)
    if _commit(v87["remaining"]) != V87_PARTITION_COMMITMENTS["remaining"]:
        raise ValueError("v88 eligible pool no longer matches the v87 remainder")
    eligible = sorted(v87["remaining"])
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
            raise ValueError(f"v88 partition commitment changed: {name}")
    if set(partitions["development"]) & set(partitions["confirmation"]):
        raise ValueError("v88 stages overlap")
    reserved = set(v87["development"] + v87["confirmation"])
    if set(partitions["eligible"]) & reserved:
        raise ValueError("v88 eligible pool overlaps a v87 reserved partition")
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
        raise ValueError(f"unsupported v88 stage: {stage}")
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
        blind["source"] = f"iirc_train_v88_{stage}"
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
        "overlap_with_any_prior_development_or_confirmation": 0,
        "development_confirmation_overlap": 0,
    }


def write_selection_outputs(
    scored_path: Path,
    v86_model_path: Path,
    v87_model_path: Path,
    v88_model_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    temporary = output_path.with_suffix(output_path.suffix + ".v87-controls")
    prior_rows = select_v87_methods_to_file(
        scored_path, v86_model_path, v87_model_path, temporary
    )
    v88_model = load_v88_model(v88_model_path)
    scored_by_id = {str(row["id"]): row for row in read_jsonl(scored_path)}
    outputs = []
    for prior in prior_rows:
        case_id = str(prior["case_id"])
        methods = dict(prior["methods"])
        methods[CANDIDATE_METHOD] = select_v88_method(scored_by_id[case_id], v88_model)
        outputs.append({**prior, "methods": methods})
    write_jsonl(output_path, outputs)
    temporary.unlink(missing_ok=True)
    return outputs


def _paired_bootstrap(candidate: np.ndarray, control: np.ndarray, *, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(candidate), len(candidate))
        values[index] = float(np.mean(candidate[sample] - control[sample]))
    return {
        "point": round(float(np.mean(candidate - control)), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def evaluate_stage(
    selection_path: Path,
    gold_path: Path,
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v88 stage: {stage}")
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
        raise ValueError("v88 selection/gold coverage changed")
    case_rows: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        methods = selection_by_id[case_id].get("methods", {})
        if set(methods) != set(METHODS):
            raise ValueError("v88 selection methods changed")
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
    primary = aggregates[PRIMARY_CONTROL]
    candidate = aggregates[CANDIDATE_METHOD]
    primary_delta = _paired_bootstrap(
        f1_arrays[CANDIDATE_METHOD],
        f1_arrays[PRIMARY_CONTROL],
        seed=BOOTSTRAP_SEEDS[stage],
    )
    point_deltas = {
        method: round(
            candidate["evidence_macro_f1"] - aggregates[method]["evidence_macro_f1"],
            6,
        )
        for method in CONTROL_METHODS
    }
    strongest_observed = max(
        CONTROL_METHODS,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    strata: dict[str, Any] = {}
    for name in sorted({name for row in case_rows for name in row["strata"]}):
        local = [row for row in case_rows if name in row["strata"]]
        candidate_value = float(
            np.mean(
                [row["methods"][CANDIDATE_METHOD]["evidence_f1"] for row in local]
            )
        )
        primary_value = float(
            np.mean([row["methods"][PRIMARY_CONTROL]["evidence_f1"] for row in local])
        )
        strata[name] = {
            "cases": len(local),
            "candidate_f1": round(candidate_value, 6),
            "primary_control_f1": round(primary_value, 6),
            "delta": round(candidate_value - primary_value, 6),
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
        "candidate_evidence_macro_f1_at_least_0_60": candidate[
            "evidence_macro_f1"
        ]
        >= 0.60,
        "candidate_complete_evidence_recall_at_least_0_42": candidate[
            "complete_evidence_recall"
        ]
        >= 0.42,
        "candidate_minus_primary_control_f1_at_least_0_005": primary_delta["point"]
        >= 0.005,
        "candidate_minus_primary_control_ci_low_above_0": primary_delta["ci_low"]
        > 0.0,
        "candidate_min_point_delta_vs_every_control_at_least_0": min(
            point_deltas.values()
        )
        >= 0.0,
        "candidate_complete_recall_delta_vs_primary_at_least_minus_0_02": candidate[
            "complete_evidence_recall"
        ]
        - primary["complete_evidence_recall"]
        >= -0.02,
        "candidate_evidence_recall_delta_vs_primary_at_least_minus_0_02": candidate[
            "evidence_macro_recall"
        ]
        - primary["evidence_macro_recall"]
        >= -0.02,
        "candidate_mean_selected_sources_above_primary_at_most_0_25": candidate[
            "mean_selected_sources"
        ]
        - primary["mean_selected_sources"]
        <= 0.25,
        "distinct_candidate_actions_at_least_2": len(actions) >= 2,
        "second_largest_action_fraction_at_least_0_05": second_fraction >= 0.05,
        "supported_stratum_delta_vs_primary_at_least_minus_0_03": bool(
            supported_deltas
        )
        and min(supported_deltas) >= -0.03,
        "prior_or_stage_overlap_0": True,
    }
    all_pass = all(checks.values())
    status = (
        "IIRC_V88_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
        if stage == "development" and all_pass
        else "IIRC_V88_CASE_DISJOINT_SUPPORT_ESTABLISHED"
        if stage == "confirmation" and all_pass
        else f"IIRC_V88_{stage.upper()}_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-iirc-pooled-score-gap-transfer-result-v88",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "cases": expected,
        "aggregates": aggregates,
        "primary_control": PRIMARY_CONTROL,
        "strongest_observed_control": strongest_observed,
        "candidate_minus_primary_control": primary_delta,
        "candidate_point_deltas_vs_all_controls": point_deltas,
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
    stem = f"iirc_pooled_score_gap_transfer_{report['stage']}_v88"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    cases_path = output_dir / f"{stem}_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    candidate = report["aggregates"][CANDIDATE_METHOD]
    delta = report["candidate_minus_primary_control"]
    markdown_path.write_text(
        "\n".join(
            [
                f"# IIRC v88 {report['stage']} result",
                "",
                f"- Status: `{report['status']}`",
                f"- Cases: {report['cases']}",
                f"- Candidate F1: {candidate['evidence_macro_f1']:.6f}",
                f"- Primary control: `{report['primary_control']}`",
                f"- Delta: {delta['point']:.6f} "
                f"(95% CI {delta['ci_low']:.6f}, {delta['ci_high']:.6f})",
                f"- Strongest observed control: `{report['strongest_observed_control']}`",
                f"- Strict gates: {'PASS' if report['all_strict_gates_pass'] else 'FAIL'}",
                "- Gate 2: `NO-GO/SHADOW`",
                "",
                "This is a case-disjoint same-dataset source-selection test, not independent transfer, SetR reproduction, answer-generation evidence, flood-domain validation, or production authorization.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_jsonl_gzip(cases_path, list(case_rows))
    return {"json": json_path, "markdown": markdown_path, "cases": cases_path}
