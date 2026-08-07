from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from research.frc_rag.musique_dual_resource import DUAL_FRC, select_candidates
from research.frc_rag.musique_objective_gap import (
    EXACT_FRC,
    PROTOCOL_SHA256,
    evaluate_scored_cases,
    exact_objective_sets,
    load_report,
    validate_preparation_summary,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/progressive_upgrade/musique_objective_gap_diagnostic_protocol.json"
)


def _roles(value: float = 0.0) -> dict[str, float]:
    return {
        "direct_answer_support": value,
        "entity_and_scope": value,
        "corroborating_evidence": value,
        "contradiction_detection": value,
    }


def _candidate(candidate_id: str, cost: int, score: float) -> dict:
    return {
        "id": candidate_id,
        "source_id": candidate_id,
        "token_count": cost,
        "scores": {
            "bm25": score,
            "dense": score,
            "hybrid": score,
            "cross_encoder": score,
        },
        "role_scores": _roles(),
    }


def test_exact_objective_finds_knapsack_combination_missed_by_greedy() -> None:
    candidates = [
        _candidate("a", 6, 1.0),
        _candidate("b", 5, 0.7),
        _candidate("c", 5, 0.7),
    ]
    greedy = select_candidates(candidates, DUAL_FRC, top_k=2, token_budget=10)
    exact = exact_objective_sets(candidates, budgets=[10], top_k=2)[10]

    assert [item["id"] for item in greedy] == ["a"]
    assert [item["id"] for item in exact] == ["b", "c"]


def test_exact_objective_uses_frozen_tie_breaks_and_constraints() -> None:
    candidates = [
        _candidate("a", 5, 0.5),
        _candidate("b", 4, 0.5),
        _candidate("c", 4, 0.5),
    ]
    result = exact_objective_sets(candidates, budgets=[4, 8], top_k=2)

    assert [item["id"] for item in result[4]] == ["b"]
    assert [item["id"] for item in result[8]] == ["b", "c"]
    assert all(
        sum(item["token_count"] for item in selected) <= budget
        and len(selected) <= 2
        for budget, selected in result.items()
    )


def _synthetic_pair() -> tuple[dict, dict]:
    case_id = "musique::objective-gap"
    candidates = [
        _candidate(f"{case_id}::a", 307, 1.0),
        _candidate(f"{case_id}::b", 256, 0.7),
        _candidate(f"{case_id}::c", 256, 0.7),
    ]
    scored = {
        "id": case_id,
        "dataset_id": "musique_ans_v1.0_dev",
        "capability": "multi_hop_evidence_selection",
        "gold_fields_visible_to_scorer": False,
        "candidates": [
            {
                "id": item["id"],
                "source_id": item["source_id"],
                "token_count": item["token_count"],
            }
            for item in candidates
        ],
        "candidate_scores": [
            {
                "id": item["id"],
                "scores": item["scores"],
                "role_scores": item["role_scores"],
            }
            for item in candidates
        ],
    }
    gold = {
        "case_id": case_id,
        "hop_count": 2,
        "gold_unit_ids": ["g0", "g1"],
        "candidate_gold": {
            candidates[0]["id"]: {
                "labels": ["non_supporting"],
                "gold_unit_ids": [],
            },
            candidates[1]["id"]: {
                "labels": ["supporting"],
                "gold_unit_ids": ["g0"],
            },
            candidates[2]["id"]: {
                "labels": ["supporting"],
                "gold_unit_ids": ["g1"],
            },
        },
    }
    return scored, gold


def test_diagnostic_is_gold_late_private_and_deterministic(tmp_path: Path) -> None:
    scored, gold = _synthetic_pair()
    report, evidence = evaluate_scored_cases([gold], [scored], resamples=32)

    assert report["metadata"]["selection_runs"] == 1 * 5 * 3
    assert report["analysis"]["exact_minus_v36_support_f1"]["point"] > 0
    assert evidence[0]["configurations"]["512"]["methods"][EXACT_FRC][
        "metrics"
    ]["support_evidence_f1"] == 1.0
    encoded = json.dumps(evidence, sort_keys=True)
    for forbidden in ("question", "answer", "text", "is_supporting"):
        assert f'"{forbidden}"' not in encoded

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    first = write_report(
        copy.deepcopy(report), evidence, tmp_path / "first", source_paths={"x": source}
    )
    second = write_report(
        copy.deepcopy(report),
        evidence,
        tmp_path / "second",
        source_paths={"x": source},
    )
    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["markdown"].read_bytes() == second["markdown"].read_bytes()
    assert first["evidence"].read_bytes() == second["evidence"].read_bytes()
    loaded = load_report(first["json"], first["evidence"])
    assert loaded["analysis"]["outcome"]["adoption_eligible"] is False
    assert loaded["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"


def test_protocol_hash_is_frozen() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256


def test_preparation_summary_uses_musique_nested_chunk_shape() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_preparation_summary(
        {"valid_rows": 2417, "candidate_chunks": {"total": 48656}},
        protocol,
    )
