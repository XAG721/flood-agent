from __future__ import annotations

from flood_system.frc_public_evidence import (
    evidence_metrics,
    paired_bootstrap,
    select_precomputed,
    selected_role_coverage,
)


def candidate(candidate_id: str, cross: float, roles: dict[str, float], *, gold: bool = False):
    return {
        "id": candidate_id,
        "token_count": 10,
        "gold": gold,
        "scores": {"bm25": cross, "dense": cross, "hybrid": cross, "cross_encoder": cross},
        "role_scores": roles,
    }


def test_precomputed_frc_balances_relevance_and_complementary_roles():
    row = {
        "required_roles": ["condition", "answer"],
        "candidates": [
            candidate("duplicate-high", 0.9, {"condition": 0.9, "answer": 0.1}),
            candidate("condition", 0.8, {"condition": 1.0, "answer": 0.1}, gold=True),
            candidate("answer", 0.7, {"condition": 0.1, "answer": 1.0}, gold=True),
        ],
    }

    selected = select_precomputed(row, "frc_select", k=2, budget=20)

    assert {item["id"] for item in selected} == {"duplicate-high", "answer"}
    assert selected_role_coverage({**row, "selected_evidence": selected}) == 1.0


def test_coverage_proxy_name_does_not_change_its_role_first_behavior():
    row = {
        "required_roles": ["condition", "answer"],
        "candidates": [
            candidate("condition", 0.4, {"condition": 0.9, "answer": 0.1}),
            candidate("answer", 0.3, {"condition": 0.1, "answer": 0.9}),
            candidate("relevant", 1.0, {"condition": 0.2, "answer": 0.2}),
        ],
    }

    selected = select_precomputed(row, "coverage_greedy_proxy", k=2, budget=20)

    assert [item["id"] for item in selected] == ["condition", "answer"]


def test_evidence_metrics_and_paired_bootstrap_are_deterministic():
    metrics = evidence_metrics(["a", "b"], ["a", "c"])
    assert metrics == {
        "evidence_recall": 0.5,
        "evidence_precision": 0.5,
        "evidence_f1": 0.5,
        "complete_evidence_set": 0.0,
    }

    first = paired_bootstrap([0.1, -0.1, 0.2], resamples=100)
    second = paired_bootstrap([0.1, -0.1, 0.2], resamples=100)
    assert first == second
    assert first["cases"] == 3
