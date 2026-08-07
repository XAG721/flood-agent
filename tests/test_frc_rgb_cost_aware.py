from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.rgb_cost_aware_frc import (
    BUDGETS,
    EXECUTION_SHA256,
    METHODS,
    NEW_FRC,
    OLD_FRC,
    POST_RESULT_CORRECTION_SHA256,
    PROTOCOL_SHA256,
    evaluate_scored_cases,
    load_report,
    prepare_row,
    read_jsonl,
    score_cases_resumable,
    select_candidates,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "docs/progressive_upgrade/rgb_cost_aware_frc_protocol.json"
EXECUTION_PATH = REPO_ROOT / "docs/progressive_upgrade/rgb_cost_aware_frc_execution.json"
CORRECTION_PATH = (
    REPO_ROOT
    / "docs/progressive_upgrade/rgb_cost_aware_frc_post_result_correction.json"
)


class FakeTokenizer:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result = []
        for word in text.split():
            if word not in self._ids:
                token_id = len(self._ids) + 1
                self._ids[word] = token_id
                self._words[token_id] = word
            result.append(self._ids[word])
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(self._words[token_id] for token_id in token_ids)


def _roles(**overrides: float) -> dict[str, float]:
    result = {
        "direct_answer_support": 0.0,
        "entity_and_scope": 0.0,
        "corroborating_evidence": 0.0,
        "contradiction_detection": 0.0,
    }
    result.update(overrides)
    return result


def _candidate(
    candidate_id: str,
    *,
    cost: int,
    cross: float,
    roles: dict[str, float] | None = None,
) -> dict:
    return {
        "id": candidate_id,
        "source_id": candidate_id,
        "token_count": cost,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": roles or _roles(),
    }


def test_prepare_row_is_blind_and_assigns_hash_sorted_neutral_ids() -> None:
    row = {
        "id": 7,
        "query": "Who won?",
        "answer": ["A"],
        "positive": ["correct evidence"],
        "negative": ["misleading evidence"],
    }
    prepared, gold = prepare_row(
        "rgb_en_refine", "noise_robustness", row, FakeTokenizer()
    )

    assert prepared["gold_fields_visible_to_scorer"] is False
    assert not ({"answer", "positive", "negative", "candidate_gold"} & set(prepared))
    assert all("label" not in json.dumps(item) for item in prepared["candidates"])
    digests = sorted(
        hashlib.sha256(text.encode()).hexdigest()
        for text in ("correct evidence", "misleading evidence")
    )
    label_by_digest = {
        hashlib.sha256("correct evidence".encode()).hexdigest(): "positive",
        hashlib.sha256("misleading evidence".encode()).hexdigest(): "negative",
    }
    labels_in_neutral_order = [
        gold["candidate_gold"][candidate["id"]]["labels"][0]
        for candidate in prepared["candidates"]
    ]
    assert labels_in_neutral_order == [label_by_digest[digest] for digest in digests]
    assert [item["source_id"] for item in prepared["candidates"]] == [
        "s0000",
        "s0001",
    ]


def test_prepare_row_excludes_cross_label_duplicate_conflict() -> None:
    row = {
        "id": 8,
        "query": "Who won?",
        "answer": ["A"],
        "positive": ["same evidence"],
        "negative": ["different evidence"],
        "positive_wrong": ["same   evidence"],
    }
    with pytest.raises(ValueError, match="cross_label_duplicate_conflict"):
        prepare_row("rgb_en_fact", "counterfactual_robustness", row, FakeTokenizer())


def test_cost_aware_selector_prefers_complementary_cheap_evidence() -> None:
    candidates = [
        _candidate(
            "long",
            cost=10,
            cross=1.0,
            roles=_roles(
                direct_answer_support=1.0,
                entity_and_scope=1.0,
                corroborating_evidence=1.0,
                contradiction_detection=1.0,
            ),
        ),
        _candidate(
            "cheap-a",
            cost=2,
            cross=0.4,
            roles=_roles(direct_answer_support=1.0),
        ),
        _candidate(
            "cheap-b",
            cost=2,
            cross=0.4,
            roles=_roles(entity_and_scope=1.0),
        ),
        _candidate(
            "cheap-c",
            cost=2,
            cross=0.4,
            roles=_roles(corroborating_evidence=1.0),
        ),
        _candidate(
            "cheap-d",
            cost=2,
            cross=0.4,
            roles=_roles(contradiction_detection=1.0),
        ),
    ]

    old = select_candidates(candidates, OLD_FRC, top_k=5, token_budget=10)
    new = select_candidates(candidates, NEW_FRC, top_k=5, token_budget=10)

    assert [item["id"] for item in old] == ["long"]
    assert {item["id"] for item in new} == {
        "cheap-a",
        "cheap-b",
        "cheap-c",
        "cheap-d",
    }
    assert sum(item["token_count"] for item in new) == 8


def test_all_selectors_respect_hard_constraints_and_knapsack_is_exact() -> None:
    candidates = [
        _candidate("a", cost=6, cross=0.8),
        _candidate("b", cost=5, cross=0.7),
        _candidate("c", cost=5, cross=0.7),
    ]
    for method in METHODS:
        selected = select_candidates(
            candidates, method, top_k=2, token_budget=10
        )
        assert len(selected) <= 2
        assert sum(item["token_count"] for item in selected) <= 10
    exact = select_candidates(
        candidates, "cross_encoder_knapsack", top_k=2, token_budget=10
    )
    assert [item["id"] for item in exact] == ["b", "c"]


def _scored_case(dataset_id: str, index: int) -> tuple[dict, dict]:
    case_id = f"{dataset_id}::{index}"
    candidates = [
        _candidate(
            f"{case_id}::s0000::c000",
            cost=10,
            cross=0.9,
            roles=_roles(direct_answer_support=0.9),
        ),
        _candidate(
            f"{case_id}::s0001::c000",
            cost=10,
            cross=0.1,
            roles=_roles(contradiction_detection=0.9),
        ),
    ]
    scored = {
        "id": case_id,
        "dataset_id": dataset_id,
        "capability": "synthetic",
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
        "dataset_id": dataset_id,
        "capability": "synthetic",
        "gold_unit_ids": ["g0"],
        "candidate_gold": {
            candidates[0]["id"]: {
                "labels": ["positive"],
                "gold_unit_ids": ["g0"],
            },
            candidates[1]["id"]: {
                "labels": ["negative"],
                "gold_unit_ids": [],
            },
        },
    }
    return scored, gold


def _synthetic_evaluation() -> tuple[dict, list[dict]]:
    pairs = [
        _scored_case("rgb_en_refine", 1),
        _scored_case("rgb_en_int", 2),
        _scored_case("rgb_en_fact", 3),
    ]
    return evaluate_scored_cases(
        [gold for _, gold in pairs],
        [scored for scored, _ in pairs],
        resamples=32,
    )


def test_evaluation_joins_gold_after_selection_and_rejects_leakage() -> None:
    report, evidence = _synthetic_evaluation()
    assert report["metadata"]["cases"] == 3
    assert report["metadata"]["selection_runs"] == 3 * len(METHODS) * len(BUDGETS)
    assert all(
        row["raw_question_answer_or_candidate_text_exported"] is False
        for row in evidence
    )
    encoded = json.dumps(evidence, sort_keys=True)
    for forbidden in ("query", "answer", "candidate_text", "source_label"):
        assert f'"{forbidden}"' not in encoded

    scored, gold = _scored_case("rgb_en_refine", 9)
    scored["answer"] = ["leak"]
    with pytest.raises(ValueError, match="gold fields"):
        evaluate_scored_cases([gold], [scored], resamples=8)


def test_resumable_scoring_and_report_artifacts_are_deterministic(
    tmp_path: Path,
) -> None:
    cases = [
        {"id": "case-1"},
        {"id": "case-2"},
        {"id": "case-3"},
    ]

    class FakeScorer:
        def score_cases(self, rows: list[dict]) -> list[dict]:
            return [{"id": row["id"], "value": 1} for row in rows]

    score_path = tmp_path / "scored.jsonl"
    first = score_cases_resumable(cases, FakeScorer(), score_path, case_batch_size=2)
    second = score_cases_resumable(cases, FakeScorer(), score_path, case_batch_size=2)
    assert first == second == list(read_jsonl(score_path))

    report, evidence = _synthetic_evaluation()
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    first_paths = write_report(
        copy.deepcopy(report), evidence, tmp_path / "first", source_paths={"x": source}
    )
    second_paths = write_report(
        copy.deepcopy(report), evidence, tmp_path / "second", source_paths={"x": source}
    )
    assert first_paths["json"].read_bytes() == second_paths["json"].read_bytes()
    assert first_paths["markdown"].read_bytes() == second_paths[
        "markdown"
    ].read_bytes()
    assert first_paths["evidence"].read_bytes() == second_paths[
        "evidence"
    ].read_bytes()
    loaded = load_report(first_paths["json"], first_paths["evidence"])
    assert loaded["metadata"]["evidence_artifact"][
        "raw_question_answer_or_candidate_text_exported"
    ] is False


def test_registration_hashes_are_stable() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256(EXECUTION_PATH.read_bytes()).hexdigest() == EXECUTION_SHA256
    assert (
        hashlib.sha256(CORRECTION_PATH.read_bytes()).hexdigest()
        == POST_RESULT_CORRECTION_SHA256
    )
