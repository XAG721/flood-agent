from __future__ import annotations

import hashlib
from pathlib import Path

from research.frc_rag.squad2_extractive_support_gate import (
    CANDIDATE,
    EXACT_ANCHOR_GATED,
    EXACT_ANCHOR_UNGATED,
    EXTRACTIVE_PROMPT,
    build_deterministic_queries,
    build_gold_rows,
    extractive_prompt,
    parse_extractive_support,
    prepare_blind_cases,
    select_disjoint_balanced_sample,
    select_v59,
    validate_protocol,
    validate_query_cache,
    validate_verifier_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def _source() -> dict:
    rows = [
        (
            "old-a",
            "Old Alpha is red. Beta is blue.",
            "What color is Old Alpha?",
            False,
            "red",
            13,
        ),
        (
            "new-a1",
            "Alpha is red. Beta is blue.",
            "What color is Alpha?",
            False,
            "red",
            9,
        ),
        (
            "new-a2",
            "Gamma is tall. Delta is short.",
            "How tall is Gamma?",
            False,
            "tall",
            9,
        ),
        (
            "old-n",
            "Old Epsilon is old. Zeta is new.",
            "Where was Old Epsilon born?",
            True,
            "",
            0,
        ),
        (
            "new-n1",
            "Epsilon is old. Zeta is new.",
            "Where was Epsilon born?",
            True,
            "",
            0,
        ),
        ("new-n2", "Eta is warm. Theta is cold.", "Who discovered Eta?", True, "", 0),
    ]
    paragraphs = []
    for raw_id, context, question, impossible, answer, start in rows:
        paragraphs.append(
            {
                "context": context,
                "qas": [
                    {
                        "id": raw_id,
                        "question": question,
                        "is_impossible": impossible,
                        "answers": (
                            []
                            if impossible
                            else [{"text": answer, "answer_start": start}]
                        ),
                        "plausible_answers": [],
                    }
                ],
            }
        )
    return {
        "version": "v2.0",
        "data": [{"title": "Synthetic", "paragraphs": paragraphs}],
    }


def _commitment(raw_id: str) -> str:
    return hashlib.sha256(raw_id.encode()).hexdigest()


def _candidate(index: int) -> dict:
    return {
        "id": f"u{index}",
        "source_id": f"u{index}",
        "token_count": 10,
        "scores": {
            "bm25": 1.0 - index * 0.1,
            "dense": 1.0 - index * 0.1,
            "hybrid": 1.0 - index * 0.1,
            "cross_encoder": 1.0 - index * 0.1,
        },
        "dynamic_role_scores": {
            "anchor": 1.0 - index * 0.1,
            "first_fact": 0.2 + index * 0.1,
            "second_fact_or_bridge": 0.6 - index * 0.05,
            "counterevidence": 0.1 + index * 0.02,
        },
        "static_role_scores": {},
    }


def test_protocol_prompt_and_disjoint_boundary_are_frozen() -> None:
    path = (
        REPO_ROOT
        / "docs/progressive_upgrade/squad2_extractive_support_gate_protocol_v59.json"
    )
    value = validate_protocol(path)
    assert value["frozen_extractive_prompt"] == EXTRACTIVE_PROMPT
    assert value["scope"]["v58_development_overlap"] == 0
    assert value["v58_boundary"]["v58_confirmation_dev_opened"] is False


def test_extractive_parser_accepts_only_abstention_or_paragraph_span() -> None:
    context = "Alpha was born in Paris in 1901."
    assert parse_extractive_support("UNSUPPORTED", context) == "UNSUPPORTED"
    assert parse_extractive_support(" paris ", context) == "SUPPORTED_SPAN"
    assert parse_extractive_support("Paris, France", context) is None
    assert parse_extractive_support("The answer is Paris", context) is None


def test_disjoint_sampler_excludes_prior_commitments() -> None:
    excluded = {_commitment("old-a"), _commitment("old-n")}
    selected, sampling = select_disjoint_balanced_sample(
        _source(),
        stage="development",
        excluded_commitments=excluded,
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    assert len(selected) == 4
    assert sampling["selected_v58_commitment_overlap"] == 0
    assert sampling["v58_commitments_present_in_source"] == 2
    assert all(_commitment(row["raw_id"]) not in excluded for row in selected)


def test_blind_preparation_and_queries_preserve_zero_overlap() -> None:
    excluded = {_commitment("old-a"), _commitment("old-n")}
    selected, _ = select_disjoint_balanced_sample(
        _source(),
        stage="development",
        excluded_commitments=excluded,
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    prepared, maps, verifier_inputs, structural = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    assert all(row["source_case_commitment"] not in excluded for row in maps)
    assert (
        structural["inherited_candidate_construction"]
        == "frozen v58 exact implementation"
    )
    queries = build_deterministic_queries(prepared)
    assert validate_query_cache(prepared, queries)["fallback_rate"] == 0.0
    assert "Answer span or UNSUPPORTED:" in extractive_prompt(verifier_inputs[0])


def test_shared_gate_candidate_and_anchor_semantics() -> None:
    candidates = [_candidate(index) for index in range(6)]
    assert (
        select_v59(
            candidates, EXACT_ANCHOR_GATED, support_passed=False, token_budget=128
        )
        == []
    )
    assert (
        select_v59(candidates, CANDIDATE, support_passed=False, token_budget=128) == []
    )
    assert select_v59(
        candidates, EXACT_ANCHOR_UNGATED, support_passed=False, token_budget=128
    )
    assert select_v59(candidates, CANDIDATE, support_passed=True, token_budget=128)


def test_verifier_cache_validation_counts_invalid_fail_closed_rows() -> None:
    inputs = [
        {"id": "a", "context": "Alpha", "question": "Q"},
        {"id": "b", "context": "Beta", "question": "Q"},
    ]
    decisions = [
        {
            "id": "a",
            "cache_key": "frozen",
            "invalid_fail_closed_used": False,
        },
        {
            "id": "b",
            "cache_key": "frozen",
            "invalid_fail_closed_used": True,
        },
    ]
    summary = validate_verifier_cache(inputs, decisions, cache_key="frozen")
    assert summary["invalid_output_count"] == 1
    assert summary["invalid_output_rate"] == 0.5


def test_gold_join_remains_post_score_and_ceiling_complete() -> None:
    excluded = {_commitment("old-a"), _commitment("old-n")}
    selected, _ = select_disjoint_balanced_sample(
        _source(),
        stage="development",
        excluded_commitments=excluded,
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    prepared, maps, _, structural = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    scored = [
        {
            "id": row["id"],
            "candidates": [
                {
                    "id": candidate["id"],
                    "source_id": candidate["source_id"],
                    "token_count": candidate["token_count"],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in prepared
    ]
    gold = build_gold_rows(_source(), maps, scored, structural_census=structural)
    assert all(row["candidate_ceiling_complete"] for row in gold)
    assert all(
        row["gold_candidate_ids"]
        for row in gold
        if row["answer_state"] == "answer_bearing"
    )
