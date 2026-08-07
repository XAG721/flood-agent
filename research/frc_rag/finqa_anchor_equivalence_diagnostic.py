"""Gold-free post-result equivalence diagnostic for FinQA v45.

The diagnostic reads only opaque ids, token costs, and frozen neural scores
from the blind score cache. It never joins supporting-fact labels and cannot
change the locked v45 result or any selector decision.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.frc_rag.finqa_anchor_guarded_atomic_roles import (
    ANCHOR_GUARDED_V45,
    GUARDED_V44,
    select_v45,
)
from research.frc_rag.hover_dynamic_atomic_roles import (
    BUDGETS,
    merge_scored_candidates,
)
from research.frc_rag.ottqa_guarded_adaptive_atomic_roles import (
    guarded_target_cardinality,
)


SCHEMA_VERSION = "frc-finqa-anchor-equivalence-diagnostic-v45"
EXPERIMENT_ID = "FRC-FINQA-ANCHOR-EQUIVALENCE-DIAGNOSTIC-V45"
PROTOCOL_SHA256 = "2364924ba8210d4d818d6493cb1a02c021bf472a7c4595ed6ec8ecfa2fce29a3"
EXPECTED_CASES = 360

_FORBIDDEN_KEYS = {
    "answer",
    "candidate_ceiling",
    "exe_ans",
    "gold",
    "gold_candidate_ids",
    "gold_count",
    "gold_inds",
    "program",
    "program_re",
    "source_mode",
}


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    configurations = len(rows)
    feasible = [row for row in rows if row["anchor_feasible"]]
    return {
        "configurations": configurations,
        "anchor_feasible_configurations": len(feasible),
        "target_cardinality_equality_rate": _rate(
            sum(bool(row["target_cardinality_equal"]) for row in rows),
            configurations,
        ),
        "ordered_selection_equality_rate": _rate(
            sum(bool(row["ordered_selection_equal"]) for row in rows),
            configurations,
        ),
        "unordered_selection_equality_rate": _rate(
            sum(bool(row["unordered_selection_equal"]) for row in rows),
            configurations,
        ),
        "v44_anchor_retention_rate_when_feasible": _rate(
            sum(bool(row["v44_anchor_retained"]) for row in feasible),
            len(feasible),
        ),
        "v44_first_selection_equals_anchor_rate_when_feasible": _rate(
            sum(bool(row["v44_first_selection_equals_anchor"]) for row in feasible),
            len(feasible),
        ),
        "mean_v44_selected_units": round(
            sum(int(row["v44_selected_count"]) for row in rows) / configurations,
            6,
        )
        if configurations
        else 0.0,
        "mean_v45_selected_units": round(
            sum(int(row["v45_selected_count"]) for row in rows) / configurations,
            6,
        )
        if configurations
        else 0.0,
    }


def analyze_anchor_equivalence(
    scored_rows: Iterable[dict[str, Any]],
    *,
    expected_cases: int | None = EXPECTED_CASES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rows = [dict(row) for row in scored_rows]
    if expected_cases is not None and len(raw_rows) != expected_cases:
        raise ValueError("FinQA diagnostic score-cache case count mismatch")
    case_ids = [str(row.get("id", "")) for row in raw_rows]
    if not all(case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("FinQA diagnostic case ids are empty or duplicated")
    for row in raw_rows:
        if _FORBIDDEN_KEYS & _nested_keys(row):
            raise ValueError("FinQA diagnostic score cache contains a gold field")

    evidence: list[dict[str, Any]] = []
    for row in raw_rows:
        candidates = merge_scored_candidates(row)
        target = guarded_target_cardinality(candidates)
        for budget in BUDGETS:
            feasible = [
                candidate
                for candidate in candidates
                if int(candidate["token_count"]) <= budget
            ]
            anchor_id = (
                str(
                    min(
                        feasible,
                        key=lambda candidate: (
                            -float(candidate["scores"]["cross_encoder"]),
                            str(candidate["id"]),
                        ),
                    )["id"]
                )
                if feasible
                else None
            )
            selected_v44 = select_v45(candidates, GUARDED_V44, token_budget=budget)
            selected_v45 = select_v45(
                candidates,
                ANCHOR_GUARDED_V45,
                token_budget=budget,
            )
            ids_v44 = [str(candidate["id"]) for candidate in selected_v44]
            ids_v45 = [str(candidate["id"]) for candidate in selected_v45]
            if len(ids_v44) > target or len(ids_v45) > target:
                raise AssertionError("FinQA diagnostic selector exceeded target cardinality")
            evidence.append(
                {
                    "case_id": str(row["id"]),
                    "budget": budget,
                    "target_cardinality_v44": target,
                    "target_cardinality_v45": target,
                    "target_cardinality_equal": True,
                    "anchor_feasible": anchor_id is not None,
                    "v44_selected_count": len(ids_v44),
                    "v45_selected_count": len(ids_v45),
                    "ordered_selection_equal": ids_v44 == ids_v45,
                    "unordered_selection_equal": set(ids_v44) == set(ids_v45),
                    "v44_anchor_retained": (
                        anchor_id in ids_v44 if anchor_id is not None else None
                    ),
                    "v44_first_selection_equals_anchor": (
                        bool(ids_v44) and ids_v44[0] == anchor_id
                        if anchor_id is not None
                        else None
                    ),
                }
            )

    overall = _summarize(evidence)
    by_budget = {
        str(budget): _summarize(
            [row for row in evidence if int(row["budget"]) == budget]
        )
        for budget in BUDGETS
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        grouped[int(row["target_cardinality_v44"])].append(row)
    by_target = {
        str(target): _summarize(grouped[target]) for target in sorted(grouped)
    }
    if overall["ordered_selection_equality_rate"] == 1.0:
        status = "FINQA_V45_ANCHOR_BEHAVIORALLY_REDUNDANT_WITH_V44_ON_FROZEN_SCORES"
    elif overall["unordered_selection_equality_rate"] == 1.0:
        status = "FINQA_V45_V44_SET_EQUIVALENT_ORDER_DIFFERENT_ON_FROZEN_SCORES"
    else:
        status = "FINQA_V45_V44_SELECTIONS_DIFFER_ZERO_METRIC_DELTA_IS_COINCIDENTAL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "cases": len(raw_rows),
            "budgets": list(BUDGETS),
            "case_budget_configurations": len(evidence),
            "gold_joined": False,
            "question_or_candidate_text_used": False,
            "post_result_diagnostic": True,
        },
        "analysis": {
            "overall": overall,
            "by_budget": by_budget,
            "by_target_cardinality": by_target,
            "outcome": {
                "status": status,
                "v45_locked_outcome_changed": False,
                "selector_adoption_authorized": False,
                "reuse_v45_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "This is a post-result descriptive diagnostic on the revealed FinQA score cache.",
            "Behavioral equivalence on this cache does not prove general equivalence on other datasets or scores.",
            "No supporting-fact labels, questions, candidate text, answers, programs, or source modes are used.",
        ],
    }
    return report, evidence


def validate_protocol(path: Path) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("FinQA anchor-equivalence protocol hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FinQA anchor-equivalence protocol experiment mismatch")
    if value.get("timing_and_scope", {}).get("diagnostic_only") is not True:
        raise ValueError("FinQA anchor-equivalence diagnostic boundary is missing")
    return value


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    for path in (json_path, markdown_path, evidence_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    overall = report["analysis"]["overall"]
    lines = [
        "# FinQA v45 anchor-equivalence diagnostic",
        "",
        f"- Status: `{report['analysis']['outcome']['status']}`",
        f"- Cases: {report['metadata']['cases']}",
        f"- Case-budget configurations: {report['metadata']['case_budget_configurations']}",
        f"- Ordered selection equality: {overall['ordered_selection_equality_rate']:.6f}",
        f"- Set equality: {overall['unordered_selection_equality_rate']:.6f}",
        f"- v44 anchor retention: {overall['v44_anchor_retention_rate_when_feasible']:.6f}",
        f"- v44 first selection equals anchor: {overall['v44_first_selection_equals_anchor_rate_when_feasible']:.6f}",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "This is a gold-free post-result diagnostic. It cannot change the locked v45 result, select a new method, or authorize tuning.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in evidence:
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


__all__ = [
    "EXPERIMENT_ID",
    "PROTOCOL_SHA256",
    "analyze_anchor_equivalence",
    "validate_protocol",
    "write_report",
]
