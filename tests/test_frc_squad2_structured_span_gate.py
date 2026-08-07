from __future__ import annotations

import hashlib
from pathlib import Path

import research.frc_rag.squad2_extractive_support_gate as v59
from research.frc_rag.squad2_structured_span_gate import (
    BUDGETS,
    CANDIDATE,
    METHODS,
    STRUCTURED_PROMPT,
    build_deterministic_queries,
    load_prior_exclusion_union,
    parse_structured_support,
    prepare_blind_cases,
    select_disjoint_balanced_sample,
    structured_prompt,
    validate_protocol,
    validate_query_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def _commitment(raw_id: str) -> str:
    return hashlib.sha256(raw_id.encode()).hexdigest()


def _source() -> dict:
    rows = [
        ("old-a1", "Old Alpha is red.", "What color is Old Alpha?", False, "red", 13),
        ("old-a2", "Old Gamma is tall.", "How tall is Old Gamma?", False, "tall", 13),
        ("old-n1", "Old Epsilon is old.", "Where was Old Epsilon born?", True, "", 0),
        ("old-n2", "Old Eta is warm.", "Who discovered Old Eta?", True, "", 0),
        ("new-a1", "Alpha is red.", "What color is Alpha?", False, "red", 9),
        ("new-a2", "Gamma is tall.", "How tall is Gamma?", False, "tall", 9),
        ("new-n1", "Epsilon is old.", "Where was Epsilon born?", True, "", 0),
        ("new-n2", "Eta is warm.", "Who discovered Eta?", True, "", 0),
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


def test_protocol_prompt_and_prior_boundary_are_frozen() -> None:
    path = (
        REPO_ROOT
        / "docs/progressive_upgrade/squad2_structured_span_gate_protocol_v60.json"
    )
    value = validate_protocol(path)
    assert value["frozen_structured_prompt"] == STRUCTURED_PROMPT
    assert value["scope"]["expected_prior_exclusion_commitments"] == 1200
    assert value["prior_boundary"]["dev_content_opened"] is False


def test_structured_parser_handles_registered_tags_and_wrappers() -> None:
    context = "Alpha was born in Paris in 1901."
    assert parse_structured_support("SPAN: Paris", context) == "SUPPORTED_SPAN"
    assert parse_structured_support('SPAN: "Paris"', context) == "SUPPORTED_SPAN"
    assert parse_structured_support("ANSWER: Paris", context) == "SUPPORTED_SPAN"
    assert parse_structured_support("Paris", context) == "SUPPORTED_SPAN"
    assert parse_structured_support("Result:\n`Paris`", context) == "SUPPORTED_SPAN"


def test_structured_parser_rejects_unsupported_and_fails_closed() -> None:
    context = "Alpha was born in Paris."
    assert parse_structured_support("UNSUPPORTED", context) == "UNSUPPORTED"
    assert (
        parse_structured_support("UNSUPPORTED because the fact is absent", context)
        == "UNSUPPORTED"
    )
    assert parse_structured_support("SPAN: London", context) is None
    assert parse_structured_support("The answer is Paris", context) is None


def test_prior_exclusion_union_requires_two_disjoint_600_row_maps() -> None:
    paths = [
        REPO_ROOT
        / ".cache/benchmarks/squad2_generative_answerability_gate_v58/development/candidate_map.jsonl",
        REPO_ROOT
        / ".cache/benchmarks/squad2_extractive_support_gate_v59/development/candidate_map.jsonl",
    ]
    assert len(load_prior_exclusion_union(paths)) == 1200


def test_third_sampler_has_zero_overlap_with_prior_union() -> None:
    excluded = {
        _commitment(raw_id) for raw_id in ("old-a1", "old-a2", "old-n1", "old-n2")
    }
    selected, sampling = select_disjoint_balanced_sample(
        _source(),
        stage="development",
        excluded_commitments=excluded,
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    assert len(selected) == 4
    assert sampling["selected_prior_commitment_overlap"] == 0
    assert all(_commitment(row["raw_id"]) not in excluded for row in selected)


def test_blind_cache_queries_and_prompt_remain_deterministic() -> None:
    excluded = {
        _commitment(raw_id) for raw_id in ("old-a1", "old-a2", "old-n1", "old-n2")
    }
    selected, _ = select_disjoint_balanced_sample(
        _source(),
        stage="development",
        excluded_commitments=excluded,
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    prepared, maps, verifier_inputs, _ = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    assert all(row["source_case_commitment"] not in excluded for row in maps)
    queries = build_deterministic_queries(prepared)
    assert validate_query_cache(prepared, queries)["fallback_rate"] == 0.0
    assert structured_prompt(verifier_inputs[0]).endswith("\n\nOutput:")


def test_downstream_methods_and_budgets_are_exactly_frozen_from_v59() -> None:
    assert CANDIDATE == v59.CANDIDATE
    assert METHODS == v59.METHODS
    assert BUDGETS == v59.BUDGETS
