from __future__ import annotations

from research.frc_rag.iirc_cardinality_transfer_router import (
    cardinality_from_probabilities,
    derive_training_cases,
    evidence_f1,
    rank_evidence_sources,
)


def _candidate(
    candidate_id: str,
    source: str,
    cross: float,
    roles: tuple[float, float],
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "scores": {"cross_encoder": cross},
        "role_scores": {"answer": roles[0], "condition": roles[1]},
    }


def test_rank_evidence_sources_is_unique_and_role_aware() -> None:
    row = {
        "candidates": [
            _candidate("a1", "a", 0.9, (0.9, 0.0)),
            _candidate("a2", "a", 0.8, (1.0, 0.1)),
            _candidate("b1", "b", 0.84, (0.2, 1.0)),
            _candidate("c1", "c", 0.83, (0.8, 0.2)),
        ]
    }
    ranked = rank_evidence_sources(row, limit=3)
    assert [item["source"] for item in ranked] == ["a", "b", "c"]
    assert ranked[0]["candidate_id"] == "a1"
    assert len({item["source"] for item in ranked}) == 3


def test_cardinality_thresholds_and_evidence_f1() -> None:
    thresholds = {2: 0.3, 3: 0.7, 4: 0.4}
    assert cardinality_from_probabilities({2: 0.2, 3: 0.2, 4: 0.2}, thresholds) == 1
    assert cardinality_from_probabilities({2: 0.8, 3: 0.8, 4: 0.2}, thresholds) == 3
    assert cardinality_from_probabilities({2: 0.8, 3: 0.8, 4: 0.8}, thresholds) == 4
    assert evidence_f1(["a", "b"], ["a", "c"]) == 0.5


def test_training_case_derivation_maps_chunks_to_sources_and_rejects_iirc() -> None:
    scored = [
        {
            "id": "case-1",
            "dataset": "history",
            "question": "Which evidence is needed?",
            "candidates": [
                _candidate("a1", "a", 0.9, (0.9, 0.1)),
                _candidate("a2", "a", 0.7, (0.8, 0.2)),
                _candidate("b1", "b", 0.8, (0.2, 0.9)),
            ],
        }
    ]
    gold = [{"id": "case-1", "gold_evidence_ids": ["a2", "b1"]}]
    cases = derive_training_cases(scored, gold, cohort="fixture")
    assert cases[0]["gold_sources"] == ["a", "b"]
    assert cases[0]["source_order"] == ["a", "b"]

    scored[0]["dataset"] = "iirc"
    try:
        derive_training_cases(scored, gold, cohort="fixture")
    except ValueError as error:
        assert "cannot accept IIRC" in str(error)
    else:
        raise AssertionError("IIRC model-development leakage was not rejected")
