from __future__ import annotations

import hashlib
from pathlib import Path

import research.frc_rag.squad2_generative_answerability_gate as v58
import research.frc_rag.squad2_structured_span_gate as v60
from research.frc_rag.squad2_dual_support_union import (
    BUDGETS,
    CANDIDATE,
    combine_support_decisions,
    build_deterministic_queries,
    load_prior_exclusion_union,
    prepare_blind_cases,
    select_disjoint_balanced_sample,
    validate_protocol,
    validate_query_cache,
    validate_verifier_cache,
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


def test_protocol_reuses_exact_component_prompts_and_keeps_dev_sealed() -> None:
    path = (
        REPO_ROOT
        / "docs/progressive_upgrade/squad2_dual_support_union_protocol_v61.json"
    )
    value = validate_protocol(path)
    assert value["model_and_prompts"]["binary_prompt"] == v58.SUPPORT_PROMPT
    assert value["model_and_prompts"]["structured_prompt"] == v60.STRUCTURED_PROMPT
    assert value["prior_boundary"]["dev_content_opened"] is False


def test_dual_union_accepts_either_component_and_rejects_only_both() -> None:
    context = "Alpha was born in Paris."
    assert (
        combine_support_decisions("SUPPORTED", "UNSUPPORTED", context)["support_passed"]
        is True
    )
    assert (
        combine_support_decisions("UNSUPPORTED", "SPAN: Paris", context)[
            "support_passed"
        ]
        is True
    )
    assert (
        combine_support_decisions("UNSUPPORTED", "UNSUPPORTED", context)[
            "support_passed"
        ]
        is False
    )
    invalid = combine_support_decisions("explanation", "SPAN: London", context)
    assert invalid["support_passed"] is False
    assert invalid["binary_malformed_fail_closed_used"] is True
    assert invalid["structured_invalid_fail_closed_used"] is True


def test_prior_union_contains_all_three_disjoint_600_case_maps() -> None:
    paths = [
        REPO_ROOT
        / ".cache/benchmarks/squad2_generative_answerability_gate_v58/development/candidate_map.jsonl",
        REPO_ROOT
        / ".cache/benchmarks/squad2_extractive_support_gate_v59/development/candidate_map.jsonl",
        REPO_ROOT
        / ".cache/benchmarks/squad2_structured_span_gate_v60/development/candidate_map.jsonl",
    ]
    assert len(load_prior_exclusion_union(paths)) == 1800


def test_fourth_sampler_excludes_prior_commitments() -> None:
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


def test_blind_preparation_and_queries_remain_frozen() -> None:
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
    prepared, maps, _, _ = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    assert all(row["source_case_commitment"] not in excluded for row in maps)
    queries = build_deterministic_queries(prepared)
    assert validate_query_cache(prepared, queries)["fallback_rate"] == 0.0


def test_verifier_summary_tracks_both_component_failures() -> None:
    inputs = [{"id": "a"}, {"id": "b"}]
    decisions = [
        {
            "id": "a",
            "cache_key": "frozen",
            "binary_malformed_fail_closed_used": False,
            "structured_invalid_fail_closed_used": False,
        },
        {
            "id": "b",
            "cache_key": "frozen",
            "binary_malformed_fail_closed_used": True,
            "structured_invalid_fail_closed_used": True,
        },
    ]
    summary = validate_verifier_cache(inputs, decisions, cache_key="frozen")
    assert summary["binary_malformed_rate"] == 0.5
    assert summary["structured_invalid_rate"] == 0.5


def test_downstream_candidate_and_budgets_remain_v60_frozen() -> None:
    assert CANDIDATE == v60.CANDIDATE
    assert BUDGETS == v60.BUDGETS
