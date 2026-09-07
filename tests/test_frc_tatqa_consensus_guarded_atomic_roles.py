from __future__ import annotations

import json
from pathlib import Path

import research.frc_rag.tatqa_consensus_guarded_atomic_roles as tatqa
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    CONSENSUS_GUARDED_V46,
    build_candidate_coverage,
    build_candidate_units,
    build_gold_rows,
    consensus_guarded_target_cardinality,
    mapped_candidate_ids,
    prepare_blind_cases,
    role_consensus_details,
    select_v46,
    validate_protocol,
)


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _candidate(identifier: str, ranks: dict[str, float], token_count: int = 10) -> dict:
    return {
        "id": identifier,
        "source_id": identifier,
        "source_kind": "table",
        "token_count": token_count,
        "scores": {
            "bm25": 0.1,
            "dense": 0.1,
            "hybrid": 0.1,
            "cross_encoder": 0.5,
        },
        "static_role_scores": {role: 0.1 for role in STATIC_ROLES},
        "dynamic_role_scores": dict(ranks),
    }


def _ranked_candidates(order_by_role: dict[str, list[str]]) -> list[dict]:
    identifiers = sorted({value for values in order_by_role.values() for value in values})
    result = []
    for identifier in identifiers:
        scores = {}
        for role in DYNAMIC_ROLES:
            ordered = order_by_role[role]
            scores[role] = float(len(ordered) - ordered.index(identifier))
        result.append(_candidate(identifier, scores))
    return result


def _orders(*values: list[str]) -> dict[str, list[str]]:
    return {role: list(values[index]) for index, role in enumerate(DYNAMIC_ROLES)}


def _context() -> dict:
    return {
        "table": {
            "uid": "synthetic-context",
            "table": [["Year", "Revenue"], ["2024", "100"]],
        },
        "paragraphs": [
            {"order": 1, "text": "Synthetic paragraph evidence."},
        ],
        "questions": [
            {
                "uid": "q-table",
                "question": "Which table value is evidence?",
                "answer": "100",
                "answer_type": "span",
                "mapping": {"table": [[1, 1]], "paragraph": {}},
            },
            {
                "uid": "q-text",
                "question": "Which paragraph is evidence?",
                "answer": "synthetic",
                "answer_type": "span",
                "mapping": {"table": [], "paragraph": {"1": "Synthetic"}},
            },
            {
                "uid": "q-hybrid",
                "question": "Which table and paragraph evidence are required?",
                "answer": "combined",
                "answer_type": "arithmetic",
                "mapping": {
                    "table": [[1, 1]],
                    "paragraph": {"1": "Synthetic"},
                },
            },
        ],
    }


def test_protocol_is_frozen_before_tatqa_data_access() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/tatqa_consensus_guarded_atomic_roles_protocol_v46.json"
    )
    assert protocol["methods"]["candidate_method"] == CONSENSUS_GUARDED_V46
    assert protocol["methods"]["token_budgets"] == [256, 512, 1024]
    assert protocol["pre_registration_access_disclosure"]["sample_content_seen"] is False


def test_all_registered_consensus_target_counterexamples() -> None:
    no_consensus = _ranked_candidates(
        _orders(
            ["a", "b", "c", "d", "e"],
            ["a", "c", "b", "d", "e"],
            ["a", "d", "b", "c", "e"],
            ["a", "e", "b", "c", "d"],
        )
    )
    shared_runner_up = _ranked_candidates(
        _orders(
            ["a", "b", "c", "d", "e"],
            ["a", "b", "d", "c", "e"],
            ["a", "c", "b", "d", "e"],
            ["a", "d", "b", "c", "e"],
        )
    )
    two_argmax = _ranked_candidates(
        _orders(
            ["a", "c", "b", "d", "e"],
            ["b", "c", "a", "d", "e"],
            ["a", "d", "b", "c", "e"],
            ["b", "c", "a", "d", "e"],
        )
    )
    four_argmax_shared = _ranked_candidates(
        _orders(
            ["a", "e", "b", "c", "d"],
            ["b", "e", "a", "c", "d"],
            ["c", "a", "b", "d", "e"],
            ["d", "b", "a", "c", "e"],
        )
    )
    four_argmax_no_shared = _ranked_candidates(
        _orders(
            ["a", "b", "c", "d", "e"],
            ["b", "c", "a", "d", "e"],
            ["c", "d", "a", "b", "e"],
            ["d", "e", "a", "b", "c"],
        )
    )
    assert consensus_guarded_target_cardinality(no_consensus) == 1
    assert consensus_guarded_target_cardinality(shared_runner_up) == 2
    assert consensus_guarded_target_cardinality(two_argmax) == 3
    assert consensus_guarded_target_cardinality(four_argmax_shared) == 5
    assert consensus_guarded_target_cardinality(four_argmax_no_shared) == 4
    assert consensus_guarded_target_cardinality([]) == 0


def test_rank_formula_is_permutation_monotone_and_source_invariant() -> None:
    candidates = _ranked_candidates(
        _orders(
            ["a", "b", "c", "d"],
            ["a", "b", "d", "c"],
            ["c", "d", "a", "b"],
            ["d", "c", "b", "a"],
        )
    )
    expected = role_consensus_details(candidates)
    assert expected == role_consensus_details(list(reversed(candidates)))

    transformed = []
    for candidate in candidates:
        row = dict(candidate)
        row["dynamic_role_scores"] = {
            role: 7.0 * value + index
            for index, (role, value) in enumerate(
                candidate["dynamic_role_scores"].items()
            )
        }
        row["source_kind"] = "text"
        transformed.append(row)
    assert expected == role_consensus_details(transformed)

    tied = [
        _candidate("b", {role: 1.0 for role in DYNAMIC_ROLES}),
        _candidate("a", {role: 1.0 for role in DYNAMIC_ROLES}),
    ]
    assert role_consensus_details(tied)["argmax_ids"] == ["a"]


def test_selection_obeys_target_budget_and_uniqueness() -> None:
    candidates = _ranked_candidates(
        _orders(
            ["a", "b", "c", "d", "e"],
            ["a", "b", "d", "c", "e"],
            ["a", "c", "b", "d", "e"],
            ["a", "d", "b", "c", "e"],
        )
    )
    narrow = select_v46(candidates, CONSENSUS_GUARDED_V46, token_budget=15)
    wide = select_v46(candidates, CONSENSUS_GUARDED_V46, token_budget=25)
    assert len(narrow) == 1
    assert len(wide) == 2
    assert len({row["id"] for row in wide}) == 2
    assert sum(row["token_count"] for row in wide) <= 25


def test_tatqa_schema_adapter_maps_cells_and_paragraphs() -> None:
    context = _context()
    units = build_candidate_units(context, Tokenizer())
    identifiers = {row["canonical_id"] for row in units}
    assert {"table_1_1", "paragraph_1"} <= identifiers
    questions = context["questions"]
    assert mapped_candidate_ids(questions[0]) == ("table_1_1",)
    assert mapped_candidate_ids(questions[1]) == ("paragraph_1",)
    assert mapped_candidate_ids(questions[2]) == ("table_1_1", "paragraph_1")


def test_blind_preparation_is_balanced_deterministic_and_gold_free(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tatqa, "TARGET_CASES_PER_MODE", 1)
    monkeypatch.setattr(tatqa, "TARGET_CASES", 3)
    monkeypatch.setattr(tatqa, "MINIMUM_TOTAL_CASES", 3)
    monkeypatch.setattr(tatqa, "MINIMUM_CASES_PER_MODE", 1)
    contexts = [_context()]
    prepared, candidate_maps, summary = prepare_blind_cases(contexts, Tokenizer())
    assert len(prepared) == 3
    assert summary["sampling"]["selected_by_source_mode"] == {
        "table_only": 1,
        "text_only": 1,
        "hybrid": 1,
    }
    serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in (
        "answer",
        "answer_type",
        "derivation",
        "mapping",
        "source_mode",
        "gold_candidate_ids",
    ):
        assert forbidden not in serialized
    assert "q-table" not in json.dumps(candidate_maps, ensure_ascii=False)
    gold_rows = build_gold_rows(contexts, candidate_maps)
    assert {row["source_mode"] for row in gold_rows} == {
        "table_only",
        "text_only",
        "hybrid",
    }
    coverage = build_candidate_coverage(gold_rows, summary["sampling"])
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["minimum_cases_and_ceiling_checks_passed"] is True
