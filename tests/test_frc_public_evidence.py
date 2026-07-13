from __future__ import annotations

from flood_system.frc_public_evidence import (
    conflict_inventory,
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


def test_conflict_inventory_imports_completed_audit_report(tmp_path):
    path = tmp_path / "conflicts_report.json"
    path.write_text(
        """{
          "metadata": {
            "name": "Google CONFLICTS FRC retrieval and conflict-classification audit",
            "status": "RUN",
            "cases": 458,
            "answer_annotated_cases": 237,
            "conflict_type_counts": {"No conflict": 161},
            "dataset_sha256": "dataset-hash"
          },
          "classification_metrics": {
            "coverage_greedy_proxy": {"accuracy": 0.34},
            "frc_select": {
              "accuracy": 0.33,
              "per_type": {"Conflict due to outdated information": {"recall": 0.56}}
            }
          },
          "strongest_reproducible_baseline_by_accuracy": "coverage_greedy_proxy",
          "paired_frc_minus_baseline_accuracy": {"ci_low": -0.04, "ci_high": 0.02},
          "decision": {"reason": "classification is not independent human judging"}
        }""",
        encoding="utf-8",
    )

    inventory = conflict_inventory(path)

    assert inventory["status"] == "RUN"
    assert inventory["cases"] == 458
    assert inventory["frc_accuracy"] == 0.33
    assert inventory["frc_outdated_recall"] == 0.56
