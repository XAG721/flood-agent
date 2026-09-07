"""Low-candidate robustness replication of the frozen IIRC v88 model (v89)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.iirc_cardinality_transfer import (
    CANDIDATE_METHOD as V86_CANDIDATE_METHOD,
    CONTROL_METHODS as FIXED_CONTROL_METHODS,
    ROLE_NAMES,
    _aggregate,
    _article_question_map,
    _best_chunk,
    _case_metrics,
    _read_iirc,
    _stratum_name,
    clean_text,
    load_context_articles,
    normalize_title,
    requested_titles_for_stage,
    select_all_methods as select_v86_methods,
    source_id,
)
from research.frc_rag.iirc_cardinality_transfer_router import (
    load_model_artifact as load_v86_model,
)
from research.frc_rag.iirc_cardinality_transfer_router import rank_evidence_sources
from research.frc_rag.iirc_pooled_score_gap_router import (
    load_model_artifact as load_v88_model,
)
from research.frc_rag.iirc_pooled_score_gap_router import (
    select_method as select_v88_method,
)
from research.frc_rag.iirc_pooled_score_gap_transfer import (
    PARTITION_COMMITMENTS as V88_PARTITION_COMMITMENTS,
    build_partitions as build_v88_partitions,
)
from research.frc_rag.iirc_score_gap_router import (
    load_model_artifact as load_v87_model,
)
from research.frc_rag.iirc_score_gap_router import select_method as select_v87_method
from research.frc_rag.iirc_score_gap_transfer import (
    CANDIDATE_METHOD as V87_CANDIDATE_METHOD,
)
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-IIRC-LOW-CANDIDATE-ROBUSTNESS-PROSPECTIVE-V89"
SCHEMA_VERSION = "frc-iirc-low-candidate-robustness-transfer-v89"
PARTITION_SALT = "FRC-IIRC-V89-REMAINING-PARTITION|"
STAGES = ("development", "confirmation")
STAGE_CASES = {"development": 800, "confirmation": 1200}
STAGE_OFFSETS = {"development": 0, "confirmation": 800}
MAX_SOURCES = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"development": 20260814, "confirmation": 20260815}
PARTITION_COMMITMENTS = {
    "eligible": "594ee74f7656577a28640dd6d517aff8374188dd8c54df3fc33d4a78c836de29",
    "development": "16d059c8d9cfdb561f2e2aa291df1a5319737d94761bd209b8fe2443d8e45438",
    "confirmation": "ffda02d3ecd0da07ecb72dd0790d9fd2360be034cc472a464cecaf750b88f21c",
    "remaining": "d2487d7be5b83238830bac0a50b325bb52865c11efebc7b5b13e3f14387b89c7",
}
PRIMARY_CONTROL = "frc_fixed2"
CANDIDATE_METHOD = "iirc_low_candidate_robust_frc_v89"
CONTROL_METHODS = (*FIXED_CONTROL_METHODS, V86_CANDIDATE_METHOD, V87_CANDIDATE_METHOD)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
STRICT_GATES = {
    "invalid_selector_output_rate_at_most": 0.0,
    "adapter_coverage_rate": 1.0,
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
    v88 = build_v88_partitions(train_rows)
    if _commit(v88["remaining"]) != V88_PARTITION_COMMITMENTS["remaining"]:
        raise ValueError("v89 eligible pool no longer matches the v88 remainder")
    eligible = sorted(v88["remaining"])
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
            raise ValueError(f"v89 partition commitment changed: {name}")
    if set(partitions["development"]) & set(partitions["confirmation"]):
        raise ValueError("v89 stages overlap")
    if set(partitions["eligible"]) & set(v88["development"] + v88["confirmation"]):
        raise ValueError("v89 eligible pool overlaps a v88 reserved partition")
    return partitions


def prepare_case(
    article: dict[str, Any],
    question_row: dict[str, Any],
    articles: dict[str, str],
    tokenizer: Any,
    *,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    qid = str(question_row.get("qid") or "").strip()
    question = clean_text(question_row.get("question"))
    pid = str(article.get("pid") or "").strip()
    if not qid or not question or not pid:
        raise ValueError("v89 selected row has an empty ID, question, or passage ID")
    main_source = source_id("main", pid)
    source_by_passage = {"main": main_source}
    main_chunk = _best_chunk(
        question,
        clean_text(article.get("title")),
        str(article.get("text") or ""),
        tokenizer,
    )
    if main_chunk is None:
        raise ValueError("v89 selected row has no usable main passage")
    candidates: list[dict[str, Any]] = [
        {
            "id": f"iirc::{qid}::{main_source}",
            "source": main_source,
            "text": main_chunk[0],
            "token_count": main_chunk[1],
            "metadata": {"source_kind": "main"},
        }
    ]
    requested = 0
    unresolved = 0
    seen_titles: set[str] = set()
    for link in article.get("links", []):
        title = normalize_title(link.get("target"))
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        requested += 1
        linked_source = source_id("link", title)
        source_by_passage[title] = linked_source
        raw_text = articles.get(title)
        if raw_text is None:
            unresolved += 1
            continue
        chunk = _best_chunk(question, title, raw_text, tokenizer)
        if chunk is None:
            unresolved += 1
            continue
        candidates.append(
            {
                "id": f"iirc::{qid}::{linked_source}",
                "source": linked_source,
                "text": chunk[0],
                "token_count": chunk[1],
                "metadata": {"source_kind": "linked"},
            }
        )
    candidates.sort(key=lambda item: (str(item["source"]), str(item["id"])))
    contexts = question_row.get("context")
    answer = question_row.get("answer")
    if not isinstance(contexts, list) or not isinstance(answer, dict):
        raise ValueError("v89 selected row gold fields are invalid")
    gold_passages = {
        normalize_title(context.get("passage"))
        for context in contexts
        if isinstance(context, dict)
    }
    if not gold_passages or "" in gold_passages:
        raise ValueError("v89 selected row has empty gold evidence")
    if any(passage not in source_by_passage for passage in gold_passages):
        raise ValueError("v89 gold passage is outside the original article links")
    gold_sources = sorted(source_by_passage[passage] for passage in gold_passages)
    candidate_sources = {str(candidate["source"]) for candidate in candidates}
    case_id = f"iirc::{qid}"
    blind = {
        "id": case_id,
        "dataset": "iirc",
        "source": f"iirc_train_v89_{stage}",
        "question": question,
        "required_roles": list(ROLE_NAMES),
        "candidates": candidates,
    }
    gold = {
        "id": case_id,
        "public_id": qid,
        "answer_type": str(answer.get("type") or ""),
        "gold_evidence_sources": gold_sources,
        "candidate_ceiling_complete": set(gold_sources) <= candidate_sources,
    }
    census = {
        "requested_link_sources": requested,
        "unresolved_link_sources": unresolved,
        "candidate_sources": len(candidates),
        "gold_sources": len(gold_sources),
        "gold_sources_in_candidates": len(set(gold_sources) & candidate_sources),
        "low_candidate_case": int(len(candidates) < 4),
    }
    return blind, gold, census


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
        raise ValueError(f"unsupported v89 stage: {stage}")
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
        "low_candidate_cases": sum(row["low_candidate_case"] for row in census_rows),
        "adapter_coverage_rate": 1.0,
        "unresolved_link_sources": sum(
            row["unresolved_link_sources"] for row in census_rows
        ),
        "gold_statistics_sealed_not_reported": True,
        "overlap_with_any_prior_development_or_confirmation": 0,
        "development_confirmation_overlap": 0,
    }


def _select_all_available(row: dict[str, Any]) -> dict[str, Any]:
    ordered = rank_evidence_sources(row, limit=MAX_SOURCES)
    return {
        "cardinality": len(ordered),
        "selected_sources": [str(item["source"]) for item in ordered],
        "selected_ids": [str(item["candidate_id"]) for item in ordered],
        "low_candidate_all_available": True,
    }


def _normalize_cardinality(method: dict[str, Any]) -> dict[str, Any]:
    result = dict(method)
    sources = [str(value) for value in result.get("selected_sources", [])]
    ids = [str(value) for value in result.get("selected_ids", [])]
    if len(sources) != len(ids):
        raise ValueError("v89 method source/id cardinality differs")
    result["selected_sources"] = sources
    result["selected_ids"] = ids
    result["cardinality"] = len(sources)
    return result


def write_selection_outputs(
    scored_path: Path,
    v86_model_path: Path,
    v87_model_path: Path,
    v88_model_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    v86_model = load_v86_model(v86_model_path)
    v87_model = load_v87_model(v87_model_path)
    v88_model = load_v88_model(v88_model_path)
    rows = read_jsonl(scored_path)
    forbidden = {
        "answer",
        "answer_type",
        "context",
        "gold_evidence_sources",
        "question_links",
    }
    if any(forbidden & set(row) for row in rows):
        raise ValueError("v89 scored cache contains a forbidden gold field")
    outputs = []
    for row in rows:
        candidate_count = len({str(item["source"]) for item in row["candidates"]})
        methods = {
            name: _normalize_cardinality(value)
            for name, value in select_v86_methods(row, v86_model).items()
        }
        if candidate_count < 4:
            prior_v87 = _select_all_available(row)
            candidate = _select_all_available(row)
        else:
            prior_v87 = select_v87_method(row, v87_model)
            candidate = select_v88_method(row, v88_model)
        methods[V87_CANDIDATE_METHOD] = _normalize_cardinality(prior_v87)
        methods[CANDIDATE_METHOD] = _normalize_cardinality(candidate)
        outputs.append(
            {
                "case_id": str(row["id"]),
                "candidate_sources": candidate_count,
                "low_candidate_case": candidate_count < 4,
                "methods": methods,
            }
        )
    write_jsonl(output_path, outputs)
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
        raise ValueError(f"unsupported v89 stage: {stage}")
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
        raise ValueError("v89 selection/gold coverage changed")
    case_rows: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        selection = selection_by_id[case_id]
        methods = selection.get("methods", {})
        if set(methods) != set(METHODS):
            raise ValueError("v89 selection methods changed")
        candidate_sources = int(selection.get("candidate_sources", 0))
        if candidate_sources < 1:
            raise ValueError("v89 case has no candidate sources")
        metrics: dict[str, Any] = {}
        for method in METHODS:
            selected = [
                str(value) for value in methods[method].get("selected_sources", [])
            ]
            cardinality = int(methods[method].get("cardinality", 0))
            if (
                not 1 <= cardinality <= min(MAX_SOURCES, candidate_sources)
                or cardinality != len(selected)
                or len(selected) != len(set(selected))
            ):
                invalid += 1
            metrics[method] = {
                **_case_metrics(gold["gold_evidence_sources"], selected),
                "selected_sources": cardinality,
            }
        low_candidate = candidate_sources < 4
        strata = [
            *_stratum_name(gold),
            f"candidate_sources={'lt4' if low_candidate else 'ge4'}",
        ]
        case_rows.append(
            {
                "case_id": case_id,
                "answer_type": gold["answer_type"],
                "gold_cardinality": len(gold["gold_evidence_sources"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "candidate_sources": candidate_sources,
                "low_candidate_case": low_candidate,
                "strata": strata,
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
    low_candidate_cases = sum(row["low_candidate_case"] for row in case_rows)
    checks = {
        "exact_cases": len(case_rows) == expected,
        "adapter_coverage_rate_1": len(case_rows) / expected >= 1.0,
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
        "IIRC_V89_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
        if stage == "development" and all_pass
        else "IIRC_V89_ROBUST_CASE_DISJOINT_SUPPORT_ESTABLISHED"
        if stage == "confirmation" and all_pass
        else f"IIRC_V89_{stage.upper()}_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-iirc-low-candidate-robustness-result-v89",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "cases": expected,
        "adapter_coverage_rate": 1.0,
        "low_candidate_cases": low_candidate_cases,
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
            "frozen_v88_model_robustness_replication": True,
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
    stem = f"iirc_low_candidate_robustness_{report['stage']}_v89"
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
                f"# IIRC v89 {report['stage']} result",
                "",
                f"- Status: `{report['status']}`",
                f"- Cases: {report['cases']}",
                f"- Adapter coverage: {report['adapter_coverage_rate']:.3f}",
                f"- Low-candidate cases: {report['low_candidate_cases']}",
                f"- Candidate F1: {candidate['evidence_macro_f1']:.6f}",
                f"- Primary control: `{report['primary_control']}`",
                f"- Delta: {delta['point']:.6f} "
                f"(95% CI {delta['ci_low']:.6f}, {delta['ci_high']:.6f})",
                f"- Strict gates: {'PASS' if report['all_strict_gates_pass'] else 'FAIL'}",
                "- Gate 2: `NO-GO/SHADOW`",
                "",
                "This is a frozen-model robustness replication, not independent transfer, SetR reproduction, answer-generation evidence, flood-domain validation, or production authorization.",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_jsonl_gzip(cases_path, list(case_rows))
    return {"json": json_path, "markdown": markdown_path, "cases": cases_path}
