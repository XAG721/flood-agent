"""Blind MuSiQue confirmation for the v36 dual-resource FRC selector."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.rgb_cost_aware_frc import (
    BASELINES,
    BUDGETS,
    FrozenRGBScorer,
    METHODS as RGB_METHODS,
    NEW_FRC as V35_FRC,
    OLD_FRC,
    ROLE_NAMES,
    TOP_K,
    _merge_candidates,
    _set_objective,
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    select_candidates as select_rgb_candidates,
    sha256,
    write_jsonl,
)


SCHEMA_VERSION = "frc-musique-dual-resource-v1"
PROTOCOL_SHA256 = "812cf844b1650713220f89f534bf45227f229b3fa767cdc1ff87cffc5ba505d4"
EXECUTION_SHA256 = "b5dbc90be69c0341a95a5b1b2c81b440fe6e33d9507c1ec0bac89b83d6f199c8"
SOURCE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SOURCE_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
DATASET_ID = "musique_ans_v1.0_dev"
CAPABILITY = "multi_hop_evidence_selection"
EXPECTED_CASES = 2417
EXPECTED_CHUNKS = 48656

DUAL_FRC = "frc_dual_resource_v36"
METHODS = (*RGB_METHODS, DUAL_FRC)
PRIMARY = "support_evidence_f1"
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260801
CONTENT_TOKENS = 384
STRIDE = 320

_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN = {
    "answer",
    "answer_aliases",
    "answerable",
    "question_decomposition",
    "is_supporting",
    "paragraph_idx",
    "gold_unit_ids",
    "candidate_gold",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_registration(
    protocol_path: Path,
    execution_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("MuSiQue protocol hash does not match v36 registration")
    if sha256(execution_path) != EXECUTION_SHA256:
        raise ValueError("MuSiQue execution hash does not match v36 registration")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution["registration_boundary"]["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("MuSiQue execution references another protocol")
    if protocol["frozen_methods"] != list(METHODS):
        raise ValueError("MuSiQue method registration mismatch")
    return protocol, execution


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue source is missing: {path}")
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("MuSiQue source hash mismatch")
    return list(read_jsonl(path))


def _normalize(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_row(
    row: dict[str, Any], tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    public_id = str(row.get("id", "")).strip()
    if not public_id:
        raise ValueError("empty_public_id")
    case_id = f"musique::{public_id}"
    question = _normalize(row.get("question"))
    if not question:
        raise ValueError("empty_question")
    if row.get("answerable") is not True:
        raise ValueError("non_answerable")
    paragraphs = row.get("paragraphs")
    decomposition = row.get("question_decomposition")
    if not isinstance(paragraphs, list) or not isinstance(decomposition, list):
        raise ValueError("invalid_field_types")

    records: dict[str, dict[str, Any]] = {}
    support_units: set[str] = set()
    for paragraph in paragraphs:
        if (
            not isinstance(paragraph, dict)
            or not isinstance(paragraph.get("idx"), int)
            or not isinstance(paragraph.get("is_supporting"), bool)
            or not isinstance(paragraph.get("title"), str)
            or not isinstance(paragraph.get("paragraph_text"), str)
        ):
            raise ValueError("invalid_field_types")
        title = _normalize(paragraph["title"])
        text = _normalize(paragraph["paragraph_text"])
        candidate_text = f"{title}\n{text}" if title else text
        if not candidate_text:
            continue
        digest = _digest(candidate_text)
        record = records.setdefault(
            digest,
            {"text": candidate_text, "labels": set(), "gold_units": set()},
        )
        if paragraph["is_supporting"]:
            unit = f"paragraph-{paragraph['idx']}"
            record["labels"].add("supporting")
            record["gold_units"].add(unit)
            support_units.add(unit)
        else:
            record["labels"].add("non_supporting")
    if not support_units:
        raise ValueError("no_supporting_paragraph")
    if any(len(record["labels"]) > 1 for record in records.values()):
        raise ValueError("cross_label_duplicate_conflict")
    if len(records) < 2:
        raise ValueError("fewer_than_two_unique_candidates")

    candidates: list[dict[str, Any]] = []
    candidate_gold: dict[str, dict[str, Any]] = {}
    for source_index, (_, record) in enumerate(sorted(records.items())):
        source_id = f"s{source_index:04d}"
        token_ids = list(tokenizer.encode(record["text"], add_special_tokens=False))
        if not token_ids:
            continue
        chunk_index = 0
        for start in range(0, len(token_ids), STRIDE):
            chunk_ids = token_ids[start : start + CONTENT_TOKENS]
            if not chunk_ids:
                break
            candidate_id = f"{case_id}::{source_id}::c{chunk_index:03d}"
            text = str(
                tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ).strip()
            candidates.append(
                {
                    "id": candidate_id,
                    "source_id": source_id,
                    "text": text,
                    "token_count": len(chunk_ids),
                }
            )
            candidate_gold[candidate_id] = {
                "labels": sorted(record["labels"]),
                "gold_unit_ids": sorted(record["gold_units"]),
            }
            chunk_index += 1
            if start + CONTENT_TOKENS >= len(token_ids):
                break
    if len(candidates) < 2:
        raise ValueError("fewer_than_two_unique_candidates")

    prepared = {
        "dataset_id": DATASET_ID,
        "capability": CAPABILITY,
        "id": case_id,
        "query": question,
        "required_roles": list(ROLE_NAMES),
        "candidates": candidates,
        "gold_fields_visible_to_scorer": False,
    }
    if _FORBIDDEN & set(prepared):
        raise AssertionError("gold fields leaked into MuSiQue preparation")
    gold = {
        "case_id": case_id,
        "hop_count": len(decomposition),
        "gold_unit_ids": sorted(support_units),
        "candidate_gold": candidate_gold,
    }
    return prepared, gold


def prepare_dataset(
    rows: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    allowed = {
        "empty_public_id",
        "empty_question",
        "non_answerable",
        "invalid_field_types",
        "no_supporting_paragraph",
        "fewer_than_two_unique_candidates",
        "cross_label_duplicate_conflict",
    }
    for row in rows:
        try:
            blind, labels = prepare_row(row, tokenizer)
        except ValueError as exc:
            if str(exc) not in allowed:
                raise
            exclusions[str(exc)] += 1
            continue
        prepared.append(blind)
        gold.append(labels)
    ids = [str(case["id"]) for case in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("MuSiQue contains duplicate public IDs")
    chunk_counts = [len(case["candidates"]) for case in prepared]
    token_counts = [
        int(candidate["token_count"])
        for case in prepared
        for candidate in case["candidates"]
    ]
    hops = Counter(int(case["hop_count"]) for case in gold)
    summary = {
        "source_rows": len(rows),
        "valid_rows": len(prepared),
        "exclusions": dict(sorted(exclusions.items())),
        "case_ids_sha256": canonical_json_sha256(ids),
        "candidate_chunks": {
            "total": sum(chunk_counts),
            "minimum": min(chunk_counts),
            "mean": round(float(np.mean(chunk_counts)), 6),
            "maximum": max(chunk_counts),
        },
        "chunk_token_cost": {
            "total": sum(token_counts),
            "minimum": min(token_counts),
            "mean": round(float(np.mean(token_counts)), 6),
            "maximum": max(token_counts),
        },
        "hop_distribution": {str(key): hops[key] for key in sorted(hops)},
        "gold_fields_visible_to_scorer": False,
    }
    return prepared, gold, summary


def validate_preparation_summary(
    summary: dict[str, Any], execution: dict[str, Any]
) -> None:
    expected = execution["structural_census"]
    if summary["source_rows"] != expected["source_rows"]:
        raise ValueError("MuSiQue source row count mismatch")
    if summary["valid_rows"] != expected["valid_rows"]:
        raise ValueError("MuSiQue valid row count mismatch")
    if summary["exclusions"] != expected["exclusions"]:
        raise ValueError("MuSiQue structural exclusions changed")
    if summary["case_ids_sha256"] != expected["case_ids_sha256"]:
        raise ValueError("MuSiQue case ID fingerprint mismatch")
    if summary["candidate_chunks"]["total"] != expected["candidate_chunks"][
        "total"
    ]:
        raise ValueError("MuSiQue chunk count mismatch")
    if summary["hop_distribution"] != expected["hop_distribution"]:
        raise ValueError("MuSiQue hop distribution mismatch")


class FrozenMuSiQueScorer:
    """Use the frozen v35 neural scorer and relabel only cache metadata."""

    def __init__(self, **kwargs: Any) -> None:
        self._delegate = FrozenRGBScorer(**kwargs)

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = self._delegate.score_cases(cases)
        for row in results:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return results


def _dual_resource_select(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    total = 0
    current = 0.0
    while remaining and len(selected) < top_k:
        eligible = []
        for candidate in remaining:
            token_count = int(candidate["token_count"])
            if total + token_count > token_budget:
                continue
            marginal = _set_objective([*selected, candidate]) - current
            resource_cost = token_count / token_budget + 1.0 / top_k
            eligible.append(
                (
                    marginal / resource_cost,
                    marginal,
                    float(candidate["scores"]["cross_encoder"]),
                    token_count,
                    str(candidate["id"]),
                    candidate,
                )
            )
        if not eligible:
            break
        eligible.sort(
            key=lambda value: (
                -value[0],
                -value[1],
                -value[2],
                value[3],
                value[4],
            )
        )
        best = eligible[0][5]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        current = _set_objective(selected)
    alternatives: list[list[dict[str, Any]]] = [selected]
    alternatives.extend(
        [candidate]
        for candidate in candidates
        if int(candidate["token_count"]) <= token_budget
    )
    alternatives.sort(
        key=lambda rows: (
            -_set_objective(rows),
            sum(int(item["token_count"]) for item in rows),
            tuple(sorted(str(item["id"]) for item in rows)),
        )
    )
    return alternatives[0]


def select_candidates(
    candidates: list[dict[str, Any]],
    method: str,
    *,
    top_k: int = TOP_K,
    token_budget: int,
) -> list[dict[str, Any]]:
    if method != DUAL_FRC:
        return select_rgb_candidates(
            candidates,
            method,
            top_k=top_k,
            token_budget=token_budget,
        )
    return _dual_resource_select(
        candidates,
        top_k=top_k,
        token_budget=token_budget,
    )


def _metrics(
    selected: list[dict[str, Any]],
    gold: dict[str, Any],
    *,
    token_budget: int,
) -> dict[str, float | int]:
    labels = [gold["candidate_gold"][str(item["id"])] for item in selected]
    count = len(selected)
    support_count = sum("supporting" in value["labels"] for value in labels)
    covered = {
        unit for value in labels for unit in value["gold_unit_ids"]
    }
    total_gold = len(gold["gold_unit_ids"])
    precision = support_count / count if count else 0.0
    recall = len(covered) / total_gold if total_gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    token_cost = sum(int(item["token_count"]) for item in selected)
    return {
        "support_evidence_f1": float(f1),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "complete_support_coverage": float(recall == 1.0),
        "non_support_selection_rate": (
            float((count - support_count) / count) if count else 0.0
        ),
        "selected_token_cost": token_cost,
        "budget_utilization": float(token_cost / token_budget),
        "selected_count": count,
    }


def evaluate_scored_cases(
    gold_cases: Iterable[dict[str, Any]],
    scored_cases: Iterable[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_cases}
    scored = list(scored_cases)
    if len(scored) != len(gold_by_id):
        raise ValueError("MuSiQue score cache does not cover all valid cases")
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in scored:
        case_id = str(row.get("id", ""))
        if case_id in seen or case_id not in gold_by_id:
            raise ValueError(f"MuSiQue scored ID duplicate or unknown: {case_id}")
        seen.add(case_id)
        if _FORBIDDEN & set(row):
            raise ValueError(f"MuSiQue {case_id} score cache contains gold fields")
        candidates = _merge_candidates(row)
        gold = gold_by_id[case_id]
        if set(gold["candidate_gold"]) != {
            str(candidate["id"]) for candidate in candidates
        }:
            raise ValueError(f"MuSiQue {case_id} gold/candidate coverage mismatch")
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_candidates(
                    candidates,
                    method,
                    token_budget=budget,
                )
                cost = sum(int(item["token_count"]) for item in selected)
                if len(selected) > TOP_K or cost > budget:
                    raise AssertionError(f"MuSiQue {case_id} budget violation")
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _metrics(selected, gold, token_budget=budget),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "hop_count": int(gold["hop_count"]),
                "candidate_count": len(candidates),
                "gold_unit_count": len(gold["gold_unit_ids"]),
                "configurations": configurations,
                "raw_question_answer_or_candidate_text_exported": False,
            }
        )
    if seen != set(gold_by_id):
        raise ValueError("MuSiQue scored cache omitted valid cases")
    return _build_report(evidence, resamples=resamples), evidence


def _method_budget_means(
    rows: list[dict[str, Any]], method: str, budget: int
) -> dict[str, float]:
    metrics = rows[0]["configurations"][str(budget)]["methods"][method]["metrics"]
    return {
        name: round(
            float(
                np.mean(
                    [
                        row["configurations"][str(budget)]["methods"][method][
                            "metrics"
                        ][name]
                        for row in rows
                    ]
                )
            ),
            6,
        )
        for name in metrics
    }


def _family_means(evidence: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        budget_values = [
            _method_budget_means(evidence, method, budget) for budget in BUDGETS
        ]
        result[method] = {
            name: round(
                float(np.mean([value[name] for value in budget_values])), 6
            )
            for name in budget_values[0]
        }
    return result


def _interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ci_low": round(float(low), 6), "ci_high": round(float(high), 6)}


def _bootstrap(evidence: list[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    matrix = np.asarray(
        [
            [
                [
                    float(
                        row["configurations"][str(budget)]["methods"][method][
                            "metrics"
                        ][PRIMARY]
                    )
                    for budget in BUDGETS
                ]
                for method in METHODS
            ]
            for row in evidence
        ],
        dtype=float,
    )
    observed = matrix.mean(axis=(0, 2))
    indices = {method: index for index, method in enumerate(METHODS)}
    strongest = max(
        BASELINES,
        key=lambda method: (observed[indices[method]], method),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vs_v35 = np.empty(resamples, dtype=float)
    vs_old = np.empty(resamples, dtype=float)
    simultaneous = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, len(evidence), size=len(evidence))
        means = matrix[sample].mean(axis=(0, 2))
        dual = float(means[indices[DUAL_FRC]])
        vs_v35[index] = dual - float(means[indices[V35_FRC]])
        vs_old[index] = dual - float(means[indices[OLD_FRC]])
        simultaneous[index] = dual - max(
            float(means[indices[method]]) for method in BASELINES
        )

    def comparison(other: str, values: np.ndarray) -> dict[str, Any]:
        return {
            "comparator": other,
            "point": round(
                float(observed[indices[DUAL_FRC]] - observed[indices[other]]), 6
            ),
            **_interval(values),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        }

    return {
        "observed_strongest_baseline": strongest,
        "method_primary_means": {
            method: round(float(observed[index]), 6)
            for index, method in enumerate(METHODS)
        },
        "dual_minus_v35": comparison(V35_FRC, vs_v35),
        "dual_minus_old_frc": comparison(OLD_FRC, vs_old),
        "dual_minus_observed_strongest": {
            "baseline": strongest,
            "point": round(
                float(observed[indices[DUAL_FRC]] - observed[indices[strongest]]),
                6,
            ),
        },
        "dual_minus_bootstrap_strongest_simultaneous": {
            **_interval(simultaneous),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
    }


def _build_report(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    family = _family_means(evidence)
    budget_aggregates: dict[str, Any] = {}
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        methods = {
            method: _method_budget_means(evidence, method, budget)
            for method in METHODS
        }
        strongest = max(
            BASELINES,
            key=lambda method: (methods[method][PRIMARY], method),
        )
        delta = methods[DUAL_FRC][PRIMARY] - methods[strongest][PRIMARY]
        budget_deltas[str(budget)] = round(delta, 6)
        budget_aggregates[str(budget)] = {
            "strongest_baseline": strongest,
            "dual_minus_strongest": round(delta, 6),
            "methods": methods,
        }
    hop_strata: dict[str, Any] = {}
    hop_deltas: dict[str, float] = {}
    for hop in (2, 3, 4):
        rows = [row for row in evidence if row["hop_count"] == hop]
        method_means = {
            method: round(
                float(
                    np.mean(
                        [
                            row["configurations"][str(budget)]["methods"][method][
                                "metrics"
                            ][PRIMARY]
                            for row in rows
                            for budget in BUDGETS
                        ]
                    )
                ),
                6,
            )
            for method in METHODS
        }
        strongest = max(
            BASELINES,
            key=lambda method: (method_means[method], method),
        )
        delta = method_means[DUAL_FRC] - method_means[strongest]
        hop_deltas[f"{hop}-hop"] = round(delta, 6)
        hop_strata[f"{hop}-hop"] = {
            "cases": len(rows),
            "strongest_baseline": strongest,
            "dual_minus_strongest": round(delta, 6),
            "method_primary_means": method_means,
        }
    comparison = _bootstrap(evidence, resamples=resamples)
    v35 = comparison["dual_minus_v35"]
    old = comparison["dual_minus_old_frc"]
    strongest = comparison["dual_minus_observed_strongest"]
    simultaneous = comparison["dual_minus_bootstrap_strongest_simultaneous"]
    worst_budget = min(budget_deltas.values())
    worst_hop = min(hop_deltas.values())
    checks = {
        "dual_minus_v35_point_at_least_0_01": v35["point"] >= 0.01,
        "dual_minus_v35_ci_low_above_0": v35["ci_low"] > 0.0,
        "dual_minus_old_frc_point_at_least_0_01": old["point"] >= 0.01,
        "dual_minus_old_frc_ci_low_above_0": old["ci_low"] > 0.0,
        "dual_minus_strongest_point_at_least_0_01": strongest["point"] >= 0.01,
        "dual_minus_strongest_simultaneous_ci_low_above_0": (
            simultaneous["ci_low"] > 0.0
        ),
        "every_budget_delta_at_least_minus_0_02": worst_budget >= -0.02,
        "every_hop_delta_at_least_minus_0_02": worst_hop >= -0.02,
    }
    if worst_budget < -0.02 or worst_hop < -0.02:
        status = "MUSIQUE_DUAL_RESOURCE_FRC_SAFETY_REGRESSION"
    elif all(checks.values()):
        status = "MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_ESTABLISHED"
    else:
        status = "MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_NOT_ESTABLISHED"
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "dataset": "MuSiQue-Answerable dev",
            "cases": len(evidence),
            "candidate_chunks": sum(row["candidate_count"] for row in evidence),
            "hop_distribution": {
                str(hop): sum(row["hop_count"] == hop for row in evidence)
                for hop in (2, 3, 4)
            },
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "selection_runs": len(evidence) * len(METHODS) * len(BUDGETS),
            "source_revision": SOURCE_REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "execution_sha256": EXECUTION_SHA256,
            "deterministic_output_rerun": "2/2 byte-identical",
            "raw_text_committed": False,
            "gold_visible_to_scorer": False,
        },
        "aggregates": {
            "family_equal_budget_weight": family,
            "budget": budget_aggregates,
            "hop_strata": hop_strata,
        },
        "analysis": {
            "primary_metric": PRIMARY,
            "family_comparison": comparison,
            "budget_deltas": budget_deltas,
            "hop_deltas": hop_deltas,
            "worst_budget_delta": round(worst_budget, 6),
            "worst_hop_delta": round(worst_hop, 6),
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_changed": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
            },
        },
        "development_boundary": {
            "independent_confirmation": True,
            "cross_domain_selection_evidence_only": True,
            "no_musique_tuning": True,
            "gate_evidence": False,
            "raw_question_answer_or_candidate_text_exported": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    comparison = analysis["family_comparison"]
    outcome = analysis["outcome"]
    lines = [
        "# MuSiQue 双资源 FRC 独立盲评（v36）",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 案例：{metadata['cases']}；候选分块：{metadata['candidate_chunks']}；选择运行：{metadata['selection_runs']}",
        f"- 最强非 FRC 基线：`{comparison['observed_strongest_baseline']}`",
        f"- 双资源 FRC - v35：{comparison['dual_minus_v35']['point']:+.6f}，95% CI [{comparison['dual_minus_v35']['ci_low']:+.6f}, {comparison['dual_minus_v35']['ci_high']:+.6f}]",
        f"- 双资源 FRC - 旧 FRC：{comparison['dual_minus_old_frc']['point']:+.6f}，95% CI [{comparison['dual_minus_old_frc']['ci_low']:+.6f}, {comparison['dual_minus_old_frc']['ci_high']:+.6f}]",
        f"- 双资源 FRC - 最强基线：{comparison['dual_minus_observed_strongest']['point']:+.6f}，同时 95% CI [{comparison['dual_minus_bootstrap_strongest_simultaneous']['ci_low']:+.6f}, {comparison['dual_minus_bootstrap_strongest_simultaneous']['ci_high']:+.6f}]",
        f"- 最差预算差值：{analysis['worst_budget_delta']:+.6f}；最差 hop 差值：{analysis['worst_hop_delta']:+.6f}",
        "",
        "## 等权主指标",
        "",
        "| 方法 | support evidence F1 |",
        "|---|---:|",
    ]
    means = comparison["method_primary_means"]
    lines.extend(f"| `{method}` | {means[method]:.6f} |" for method in METHODS)
    lines.extend(["", "## 预注册判定", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本实验只检验公开跨域多跳数据上的证据选择，不复现 SetR，不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、CANARY 或 DEFAULT 状态。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                payload = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                zipped.write(payload + b"\n")


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "musique_dual_resource_cases.jsonl.gz"
    json_path = output_dir / "musique_dual_resource.json"
    markdown_path = output_dir / "musique_dual_resource.md"
    _write_gzip(evidence_path, evidence)
    payload = json.loads(json.dumps(report))
    payload["metadata"]["evidence_artifact"] = {
        "path": evidence_path.name,
        "sha256": sha256(evidence_path),
        "rows": len(evidence),
        "raw_question_answer_or_candidate_text_exported": False,
    }
    payload["metadata"]["source_artifacts"] = {
        name: {"path_label": path.name, "sha256": sha256(path)}
        for name, path in sorted(source_paths.items())
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(
        render_markdown(payload),
        encoding="utf-8",
        newline="\n",
    )
    report.clear()
    report.update(payload)
    return {"json": json_path, "markdown": markdown_path, "evidence": evidence_path}


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = report["metadata"]["evidence_artifact"]
    if sha256(evidence_path) != artifact["sha256"]:
        raise ValueError("MuSiQue evidence hash mismatch")
    if len(list(read_jsonl(evidence_path))) != artifact["rows"]:
        raise ValueError("MuSiQue evidence row count mismatch")
    return report


__all__ = [
    "DUAL_FRC",
    "EXECUTION_SHA256",
    "FrozenMuSiQueScorer",
    "METHODS",
    "PRIMARY",
    "PROTOCOL_SHA256",
    "evaluate_scored_cases",
    "load_frozen_tokenizer",
    "load_registration",
    "load_report",
    "load_source_rows",
    "prepare_dataset",
    "prepare_row",
    "read_jsonl",
    "score_cases_resumable",
    "select_candidates",
    "sha256",
    "validate_preparation_summary",
    "write_jsonl",
    "write_report",
]
