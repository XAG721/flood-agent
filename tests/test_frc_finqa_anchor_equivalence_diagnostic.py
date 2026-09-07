from __future__ import annotations

from pathlib import Path

from research.frc_rag.finqa_anchor_equivalence_diagnostic import (
    analyze_anchor_equivalence,
    validate_protocol,
    write_report,
)
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


def _score_row(*, differentiated: bool) -> dict:
    candidates = []
    candidate_scores = []
    for index in range(4):
        candidate_id = f"c{index}"
        candidates.append(
            {
                "id": candidate_id,
                "source_id": candidate_id,
                "text": f"synthetic {index}",
                "token_count": 20,
            }
        )
        dynamic = {role: 0.0 for role in DYNAMIC_ROLES}
        if differentiated:
            for role_index, role in enumerate(DYNAMIC_ROLES):
                if index == 1 + role_index % 2:
                    dynamic[role] = 1.0
        else:
            for role in DYNAMIC_ROLES:
                dynamic[role] = 1.0 - index * 0.1
        candidate_scores.append(
            {
                "id": candidate_id,
                "scores": {
                    "bm25": 0.1,
                    "dense": 0.1,
                    "hybrid": 0.1,
                    "cross_encoder": 0.41 if index == 0 else 0.4 - index * 0.001,
                },
                "static_role_scores": {role: 0.1 for role in STATIC_ROLES},
                "dynamic_role_scores": dynamic,
            }
        )
    return {
        "id": "synthetic-case",
        "candidates": candidates,
        "candidate_scores": candidate_scores,
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_is_registered_as_post_result_diagnostic() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/finqa_anchor_equivalence_diagnostic_protocol_v45.json"
    )
    assert protocol["timing_and_scope"]["registered_after_locked_v45_result"] is True
    assert protocol["timing_and_scope"]["confirmatory_or_causal_claim_authorized"] is False


def test_diagnostic_detects_redundant_and_non_redundant_ordering() -> None:
    redundant, _ = analyze_anchor_equivalence(
        [_score_row(differentiated=False)], expected_cases=None
    )
    assert redundant["analysis"]["overall"]["ordered_selection_equality_rate"] == 1.0

    differentiated, evidence = analyze_anchor_equivalence(
        [_score_row(differentiated=True)], expected_cases=None
    )
    assert differentiated["analysis"]["overall"][
        "v44_first_selection_equals_anchor_rate_when_feasible"
    ] < 1.0
    assert all("gold" not in key for row in evidence for key in row)


def test_diagnostic_outputs_are_deterministic(tmp_path: Path) -> None:
    report, evidence = analyze_anchor_equivalence(
        [_score_row(differentiated=False)], expected_cases=None
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        write_report(
            report,
            evidence,
            json_path=directory / "report.json",
            markdown_path=directory / "report.md",
            evidence_path=directory / "cases.jsonl.gz",
        )
    for name in ("report.json", "report.md", "cases.jsonl.gz"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
