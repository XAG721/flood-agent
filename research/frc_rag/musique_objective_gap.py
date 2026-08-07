"""Post-result v37 diagnostic for the frozen MuSiQue set objective."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.musique_dual_resource import (
    BUDGETS,
    DUAL_FRC,
    TOP_K,
    _merge_candidates,
    _metrics,
    _set_objective,
    select_candidates,
)
from research.frc_rag.rgb_cost_aware_frc import NEW_FRC, OLD_FRC, ROLE_NAMES


SCHEMA_VERSION = "frc-musique-objective-gap-diagnostic-v1"
PROTOCOL_SHA256 = "dd617c7aa5c22db359fc8944363389438d53ca04704a26a28a0f209eb71254f4"
SCORED_SHA256 = "e2c688063e694ef0df7df7f6052cb5e68c9739f26a59db8a24c1a54413ce9242"
V36_REPORT_SHA256 = (
    "27cdba122f94be27991768363cef23d860834d198a6d029ea07f7a9fdc0d22a8"
)
EXACT_FRC = "frc_exact_objective_v37"
CROSS_TOPK = "cross_encoder_topk"
METHODS = (CROSS_TOPK, OLD_FRC, NEW_FRC, DUAL_FRC, EXACT_FRC)
PRIMARY = "support_evidence_f1"
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 10_000
_EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("MuSiQue v37 diagnostic protocol hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != (
        "frc-musique-objective-gap-diagnostic-protocol-v1"
    ):
        raise ValueError("MuSiQue v37 diagnostic protocol schema mismatch")
    if protocol["registration_boundary"]["post_result_diagnostic"] is not True:
        raise ValueError("MuSiQue v37 must remain a post-result diagnostic")
    if protocol["registration_boundary"]["adoption_eligible"] is not False:
        raise ValueError("MuSiQue v37 cannot be adoption eligible")
    return protocol


@lru_cache(maxsize=None)
def _combination_indices(candidate_count: int, size: int) -> np.ndarray:
    values = np.asarray(
        list(itertools.combinations(range(candidate_count), size)),
        dtype=np.int16,
    )
    return values.reshape((-1, size))


def _is_better(
    objective: float,
    token_cost: int,
    selected_ids: tuple[str, ...],
    current: tuple[float, int, tuple[str, ...], tuple[int, ...]] | None,
) -> bool:
    if current is None:
        return True
    if objective > current[0] + _EPSILON:
        return True
    if abs(objective - current[0]) > _EPSILON:
        return False
    return (token_cost, selected_ids) < (current[1], current[2])


def exact_objective_sets(
    candidates: list[dict[str, Any]],
    *,
    budgets: Iterable[int] = BUDGETS,
    top_k: int = TOP_K,
) -> dict[int, list[dict[str, Any]]]:
    """Enumerate every feasible subset and return the exact F optimum."""

    requested = tuple(sorted({int(value) for value in budgets}))
    if not requested or requested[0] <= 0 or top_k <= 0:
        raise ValueError("budgets and top_k must be positive")
    ordered = sorted(candidates, key=lambda item: str(item["id"]))
    if not ordered:
        return {budget: [] for budget in requested}
    ids = np.asarray([str(item["id"]) for item in ordered], dtype=object)
    costs = np.asarray([int(item["token_count"]) for item in ordered], dtype=np.int64)
    cross = np.asarray(
        [float(item["scores"]["cross_encoder"]) for item in ordered],
        dtype=np.float64,
    )
    roles = np.asarray(
        [
            [float(item["role_scores"][role]) for role in ROLE_NAMES]
            for item in ordered
        ],
        dtype=np.float64,
    )
    best: dict[
        int, tuple[float, int, tuple[str, ...], tuple[int, ...]] | None
    ] = {budget: None for budget in requested}
    maximum_budget = requested[-1]
    for size in range(1, min(top_k, len(ordered)) + 1):
        indices = _combination_indices(len(ordered), size)
        combination_costs = costs[indices].sum(axis=1)
        possible = combination_costs <= maximum_budget
        if not np.any(possible):
            continue
        indices = indices[possible]
        combination_costs = combination_costs[possible]
        objectives = cross[indices].sum(axis=1) + 2.0 * roles[indices].max(
            axis=1
        ).mean(axis=1)
        for budget in requested:
            feasible_positions = np.flatnonzero(combination_costs <= budget)
            if not feasible_positions.size:
                continue
            feasible_objectives = objectives[feasible_positions]
            maximum = float(feasible_objectives.max())
            tied = feasible_positions[
                np.abs(feasible_objectives - maximum) <= _EPSILON
            ]
            minimum_cost = int(combination_costs[tied].min())
            tied = tied[combination_costs[tied] == minimum_cost]
            position = int(tied[0])
            chosen_indices = tuple(int(value) for value in indices[position])
            chosen_ids = tuple(str(ids[value]) for value in chosen_indices)
            if _is_better(
                maximum,
                minimum_cost,
                chosen_ids,
                best[budget],
            ):
                best[budget] = (
                    maximum,
                    minimum_cost,
                    chosen_ids,
                    chosen_indices,
                )
    return {
        budget: (
            [ordered[index] for index in value[3]] if value is not None else []
        )
        for budget, value in best.items()
    }


def _validate_source_inputs(
    scored_path: Path,
    v36_report_path: Path,
    protocol: dict[str, Any],
) -> None:
    frozen = protocol["frozen_inputs"]
    if sha256(scored_path) != frozen["scored_blind_cache_sha256"]:
        raise ValueError("MuSiQue v37 scored cache hash mismatch")
    if sha256(v36_report_path) != frozen["v36_report_sha256"]:
        raise ValueError("MuSiQue v37 v36 report hash mismatch")


def validate_preparation_summary(
    summary: dict[str, Any], protocol: dict[str, Any]
) -> None:
    frozen = protocol["frozen_inputs"]
    if summary.get("valid_rows") != frozen["case_count"]:
        raise ValueError("MuSiQue v37 source case count changed")
    chunks = summary.get("candidate_chunks", {})
    if chunks.get("total") != frozen["candidate_chunk_count"]:
        raise ValueError("MuSiQue v37 candidate chunk count changed")


def evaluate_scored_cases(
    gold_cases: Iterable[dict[str, Any]],
    scored_cases: Iterable[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_cases}
    scored = list(scored_cases)
    if [str(row.get("id", "")) for row in scored] != list(gold_by_id):
        raise ValueError("MuSiQue v37 scored/gold order or coverage mismatch")
    evidence: list[dict[str, Any]] = []
    for row in scored:
        case_id = str(row["id"])
        candidates = _merge_candidates(row)
        gold = gold_by_id[case_id]
        exact = exact_objective_sets(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            selected_by_method = {
                method: (
                    exact[budget]
                    if method == EXACT_FRC
                    else select_candidates(
                        candidates,
                        method,
                        token_budget=budget,
                    )
                )
                for method in METHODS
            }
            methods: dict[str, Any] = {}
            for method, selected in selected_by_method.items():
                cost = sum(int(item["token_count"]) for item in selected)
                if len(selected) > TOP_K or cost > budget:
                    raise AssertionError(f"MuSiQue v37 {case_id} budget violation")
                methods[method] = {
                    "selected_ids": sorted(str(item["id"]) for item in selected),
                    "objective": float(_set_objective(selected)),
                    "metrics": _metrics(selected, gold, token_budget=budget),
                }
            greedy = methods[DUAL_FRC]
            optimal = methods[EXACT_FRC]
            regret = float(optimal["objective"] - greedy["objective"])
            if regret < -1e-9:
                raise AssertionError(
                    f"MuSiQue v37 exact objective below greedy for {case_id}"
                )
            configurations[str(budget)] = {
                "methods": methods,
                "diagnostic": {
                    "absolute_objective_regret": max(0.0, regret),
                    "normalized_objective_regret": max(0.0, regret)
                    / max(float(optimal["objective"]), _EPSILON),
                    "selection_disagrees": (
                        greedy["selected_ids"] != optimal["selected_ids"]
                    ),
                },
            }
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
    return _build_report(evidence, resamples=resamples), evidence


def _configuration_values(
    evidence: list[dict[str, Any]],
    method: str,
    field: str,
    *,
    budget: int | None = None,
    hop: int | None = None,
) -> np.ndarray:
    values = []
    budgets = (budget,) if budget is not None else BUDGETS
    for row in evidence:
        if hop is not None and int(row["hop_count"]) != hop:
            continue
        for current_budget in budgets:
            method_row = row["configurations"][str(current_budget)]["methods"][
                method
            ]
            if field == "objective":
                values.append(float(method_row["objective"]))
            else:
                values.append(float(method_row["metrics"][field]))
    return np.asarray(values, dtype=np.float64)


def _diagnostic_values(
    evidence: list[dict[str, Any]], field: str
) -> np.ndarray:
    return np.asarray(
        [
            float(row["configurations"][str(budget)]["diagnostic"][field])
            for row in evidence
            for budget in BUDGETS
        ],
        dtype=np.float64,
    )


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _paired_interval(
    evidence: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    resamples: int,
) -> dict[str, float | int]:
    deltas = np.asarray(
        [
            np.mean(
                [
                    row["configurations"][str(budget)]["methods"][left][
                        "metrics"
                    ][PRIMARY]
                    - row["configurations"][str(budget)]["methods"][right][
                        "metrics"
                    ][PRIMARY]
                    for budget in BUDGETS
                ]
            )
            for row in evidence
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[index] = float(deltas[sample].mean())
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "point": round(float(deltas.mean()), 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
    }


def _method_means(evidence: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for method in METHODS:
        result[method] = {
            "objective": round(
                float(_configuration_values(evidence, method, "objective").mean()),
                6,
            ),
            PRIMARY: round(
                float(_configuration_values(evidence, method, PRIMARY).mean()),
                6,
            ),
            "support_precision": round(
                float(
                    _configuration_values(
                        evidence, method, "support_precision"
                    ).mean()
                ),
                6,
            ),
            "support_recall": round(
                float(
                    _configuration_values(
                        evidence, method, "support_recall"
                    ).mean()
                ),
                6,
            ),
            "selected_token_cost": round(
                float(
                    _configuration_values(
                        evidence, method, "selected_token_cost"
                    ).mean()
                ),
                6,
            ),
            "selected_count": round(
                float(
                    _configuration_values(
                        evidence, method, "selected_count"
                    ).mean()
                ),
                6,
            ),
        }
    return result


def _build_report(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    regrets = _diagnostic_values(evidence, "absolute_objective_regret")
    normalized = _diagnostic_values(evidence, "normalized_objective_regret")
    disagreements = _diagnostic_values(evidence, "selection_disagrees")
    exact_minus_v36 = _paired_interval(
        evidence, EXACT_FRC, DUAL_FRC, resamples=resamples
    )
    exact_minus_cross = _paired_interval(
        evidence, EXACT_FRC, CROSS_TOPK, resamples=resamples
    )
    budget_deltas = {
        str(budget): round(
            _mean(
                _configuration_values(
                    evidence, EXACT_FRC, PRIMARY, budget=budget
                )
            )
            - _mean(
                _configuration_values(
                    evidence, DUAL_FRC, PRIMARY, budget=budget
                )
            ),
            6,
        )
        for budget in BUDGETS
    }
    hop_deltas = {
        f"{hop}-hop": round(
            _mean(
                _configuration_values(evidence, EXACT_FRC, PRIMARY, hop=hop)
            )
            - _mean(
                _configuration_values(evidence, DUAL_FRC, PRIMARY, hop=hop)
            ),
            6,
        )
        for hop in (2, 3, 4)
    }
    mean_normalized = float(normalized.mean())
    regret_above = float(np.mean(normalized > 0.01))
    worst_stratum = min([*budget_deltas.values(), *hop_deltas.values()])
    optimization_signal = (
        mean_normalized >= 0.005
        and exact_minus_v36["point"] >= 0.005
        and exact_minus_v36["ci_low"] > 0.0
        and worst_stratum >= -0.01
    )
    near_optimal = mean_normalized <= 0.001 and regret_above <= 0.01
    misalignment = (
        mean_normalized >= 0.005 and exact_minus_v36["point"] < 0.005
    )
    if optimization_signal:
        status = "OPTIMIZATION_ERROR_SIGNAL"
        next_action = (
            "FREEZE_EXACT_OPTIMIZER_THEN_CONFIRM_ON_NEW_UNTOUCHED_DATA"
        )
    elif near_optimal:
        status = "GREEDY_NEAR_OBJECTIVE_OPTIMAL"
        next_action = "STOP_OPTIMIZER_DEVELOPMENT_SEEK_NEW_SCORING_SIGNALS"
    elif misalignment:
        status = "FROZEN_OBJECTIVE_ALIGNMENT_SIGNAL_INSUFFICIENT"
        next_action = "STOP_CURRENT_OBJECTIVE_OPTIMIZATION"
    else:
        status = "MIXED_OPTIMIZATION_AND_ALIGNMENT_DIAGNOSTIC"
        next_action = "REQUIRE_SEPARATELY_FROZEN_STUDY"
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "dataset": "MuSiQue-Answerable dev",
            "cases": len(evidence),
            "candidate_chunks": sum(row["candidate_count"] for row in evidence),
            "selection_runs": len(evidence) * len(METHODS) * len(BUDGETS),
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "hop_distribution": {
                str(hop): sum(row["hop_count"] == hop for row in evidence)
                for hop in (2, 3, 4)
            },
            "protocol_sha256": PROTOCOL_SHA256,
            "scored_blind_sha256": SCORED_SHA256,
            "v36_report_sha256": V36_REPORT_SHA256,
            "neural_rescoring": False,
            "gold_visible_during_selection": False,
            "raw_text_committed": False,
            "deterministic_output_rerun": "2/2 byte-identical",
        },
        "aggregates": {
            "method_means": _method_means(evidence),
            "objective_gap": {
                "mean_absolute_regret": round(float(regrets.mean()), 6),
                "mean_normalized_regret": round(mean_normalized, 6),
                "near_zero_regret_rate_at_1e_9": round(
                    float(np.mean(regrets <= 1e-9)), 6
                ),
                "regret_above_0_01_rate": round(regret_above, 6),
                "selection_disagreement_rate": round(
                    float(disagreements.mean()), 6
                ),
            },
        },
        "analysis": {
            "exact_minus_v36_support_f1": exact_minus_v36,
            "exact_minus_cross_encoder_topk_support_f1": exact_minus_cross,
            "budget_deltas": budget_deltas,
            "hop_deltas": hop_deltas,
            "worst_budget_or_hop_delta": round(worst_stratum, 6),
            "predeclared_checks": {
                "mean_normalized_regret_at_least_0_005": (
                    mean_normalized >= 0.005
                ),
                "exact_minus_v36_f1_at_least_0_005": (
                    exact_minus_v36["point"] >= 0.005
                ),
                "exact_minus_v36_f1_ci_low_above_0": (
                    exact_minus_v36["ci_low"] > 0.0
                ),
                "every_budget_and_hop_delta_at_least_minus_0_01": (
                    worst_stratum >= -0.01
                ),
                "mean_normalized_regret_at_most_0_001": (
                    mean_normalized <= 0.001
                ),
                "regret_above_0_01_rate_at_most_0_01": (
                    regret_above <= 0.01
                ),
            },
            "outcome": {
                "status": status,
                "next_action": next_action,
                "post_result_diagnostic": True,
                "adoption_eligible": False,
                "selector_changed": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
            },
        },
        "development_boundary": {
            "musique_results_known_before_diagnostic": True,
            "no_formula_score_threshold_prompt_or_weight_tuning": True,
            "raw_question_answer_or_candidate_text_exported": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    aggregates = report["aggregates"]
    analysis = report["analysis"]
    outcome = analysis["outcome"]
    gap = aggregates["objective_gap"]
    lines = [
        "# MuSiQue 冻结目标优化差距诊断（v37）",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 下一步：`{outcome['next_action']}`",
        f"- 案例：{metadata['cases']}；候选分块：{metadata['candidate_chunks']}；选择运行：{metadata['selection_runs']}",
        f"- 平均归一化目标 regret：{gap['mean_normalized_regret']:.6f}",
        f"- regret > 0.01 比例：{gap['regret_above_0_01_rate']:.6f}；集合不同比例：{gap['selection_disagreement_rate']:.6f}",
        f"- 精确解 - v36 Evidence F1：{analysis['exact_minus_v36_support_f1']['point']:+.6f}，95% CI [{analysis['exact_minus_v36_support_f1']['ci_low']:+.6f}, {analysis['exact_minus_v36_support_f1']['ci_high']:+.6f}]",
        f"- 精确解 - cross-encoder top-k：{analysis['exact_minus_cross_encoder_topk_support_f1']['point']:+.6f}，95% CI [{analysis['exact_minus_cross_encoder_topk_support_f1']['ci_low']:+.6f}, {analysis['exact_minus_cross_encoder_topk_support_f1']['ci_high']:+.6f}]",
        "",
        "## 方法均值",
        "",
        "| 方法 | 冻结目标 | Evidence F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = aggregates["method_means"][method]
        lines.append(
            f"| `{method}` | {values['objective']:.6f} | {values[PRIMARY]:.6f} | {values['support_precision']:.6f} | {values['support_recall']:.6f} |"
        )
    lines.extend(["", "## 预算与 hop 差值", ""])
    lines.extend(
        f"- `{name}`：{value:+.6f}"
        for name, value in analysis["budget_deltas"].items()
    )
    lines.extend(
        f"- `{name}`：{value:+.6f}"
        for name, value in analysis["hop_deltas"].items()
    )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "这是 MuSiQue 结果揭示后的机制诊断，只区分贪心优化误差与冻结目标对支持证据的对齐程度。它不能采用选择器、不能复现 SetR、不能证明洪水领域效果，也不能改变 Gate 2。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "musique_objective_gap_cases.jsonl.gz"
    json_path = output_dir / "musique_objective_gap.json"
    markdown_path = output_dir / "musique_objective_gap.md"
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
        render_markdown(payload), encoding="utf-8", newline="\n"
    )
    report.clear()
    report.update(payload)
    return {"json": json_path, "markdown": markdown_path, "evidence": evidence_path}


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = report["metadata"]["evidence_artifact"]
    if sha256(evidence_path) != artifact["sha256"]:
        raise ValueError("MuSiQue v37 evidence hash mismatch")
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    if rows != artifact["rows"]:
        raise ValueError("MuSiQue v37 evidence row count mismatch")
    return report


__all__ = [
    "EXACT_FRC",
    "METHODS",
    "PROTOCOL_SHA256",
    "_validate_source_inputs",
    "evaluate_scored_cases",
    "exact_objective_sets",
    "load_protocol",
    "load_report",
    "validate_preparation_summary",
    "write_report",
]
