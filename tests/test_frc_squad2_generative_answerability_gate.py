from __future__ import annotations

from pathlib import Path

from research.frc_rag.squad2_generative_answerability_gate import (
    CANDIDATE,
    EXACT_ANCHOR_GATED,
    EXACT_ANCHOR_UNGATED,
    SUPPORT_PROMPT,
    build_deterministic_queries,
    build_gold_rows,
    extract_cases,
    normalize_support_decision,
    prepare_blind_cases,
    select_balanced_sample,
    select_v58,
    support_prompt,
    validate_protocol,
    validate_query_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()


def _source() -> dict:
    paragraphs = []
    rows = [
        ("a1", "Alpha is red. Beta is blue.", "What color is Alpha?", False, "red", 9),
        (
            "a2",
            "Gamma is tall. Delta is short.",
            "How tall is Gamma?",
            False,
            "tall",
            9,
        ),
        ("n1", "Epsilon is old. Zeta is new.", "Where was Epsilon born?", True, "", 0),
        ("n2", "Eta is warm. Theta is cold.", "Who discovered Eta?", True, "", 0),
    ]
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
                        "plausible_answers": [
                            {"text": "private distractor", "answer_start": 0}
                        ],
                    }
                ],
            }
        )
    return {
        "version": "v2.0",
        "data": [{"title": "Synthetic", "paragraphs": paragraphs}],
    }


def _candidate(index: int) -> dict:
    dynamic = {
        "anchor": 1.0 - index * 0.1,
        "first_fact": 0.2 + index * 0.1,
        "second_fact_or_bridge": 0.6 - index * 0.05,
        "counterevidence": 0.1 + index * 0.02,
    }
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
        "dynamic_role_scores": dynamic,
        "static_role_scores": {},
    }


def test_protocol_hash_prompt_and_registered_scope_are_frozen() -> None:
    path = (
        REPO_ROOT
        / "docs/progressive_upgrade/squad2_generative_answerability_gate_protocol_v58.json"
    )
    value = validate_protocol(path)
    assert value["scope"]["development_cases"] == 600
    assert value["frozen_support_prompt"] == SUPPORT_PROMPT
    assert (
        value["prior_result_boundary"][
            "squad2_train_or_dev_downloaded_or_opened_before_protocol_registration"
        ]
        is False
    )


def test_synthetic_schema_sampling_and_blind_caches_hide_gold() -> None:
    source = _source()
    rows, schema = extract_cases(source)
    assert len(rows) == 4
    assert schema["schema_exclusion_rate"] == 0.0
    selected, sampling = select_balanced_sample(
        source,
        stage="development",
        target_per_group=2,
        maximum_cases_per_paragraph=1,
        maximum_cases_per_article_per_state=2,
    )
    assert sampling["selected_answer_state_counts"] == {
        "answer_bearing": 2,
        "no_answer": 2,
    }
    prepared, maps, verifier_inputs, structural = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    assert len(prepared) == len(maps) == len(verifier_inputs) == 4
    assert structural["gold_fields_exported_to_blind_caches"] is False
    blind_text = repr((prepared, verifier_inputs)).lower()
    assert "is_impossible" not in blind_text
    assert "plausible_answers" not in blind_text
    assert "private distractor" not in blind_text
    assert all(row["gold_fields_visible_to_scorer"] is False for row in prepared)


def test_queries_and_support_prompt_are_deterministic() -> None:
    selected, _ = select_balanced_sample(
        _source(), stage="development", target_per_group=2
    )
    prepared, _, verifier_inputs, _ = prepare_blind_cases(
        selected, FakeTokenizer(), stage="development"
    )
    queries = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, queries)
    assert summary["fallback_rate"] == 0.0
    prompt = support_prompt(verifier_inputs[0])
    assert prompt.endswith("\n\nDecision:")
    assert "SUPPORTED or UNSUPPORTED" in prompt


def test_support_parser_is_exact_and_fail_closed() -> None:
    assert normalize_support_decision(" supported \n") == "SUPPORTED"
    assert normalize_support_decision("UNSUPPORTED") == "UNSUPPORTED"
    assert normalize_support_decision("SUPPORTED because the answer appears") is None
    assert normalize_support_decision("") is None


def test_shared_gate_changes_only_gated_methods() -> None:
    candidates = [_candidate(index) for index in range(6)]
    assert (
        select_v58(
            candidates,
            EXACT_ANCHOR_GATED,
            support_passed=False,
            token_budget=128,
        )
        == []
    )
    assert (
        select_v58(
            candidates,
            CANDIDATE,
            support_passed=False,
            token_budget=128,
        )
        == []
    )
    assert select_v58(
        candidates,
        EXACT_ANCHOR_UNGATED,
        support_passed=False,
        token_budget=128,
    )
    assert select_v58(
        candidates,
        CANDIDATE,
        support_passed=True,
        token_budget=128,
    )


def test_gold_join_uses_answer_intervals_only_after_blind_preparation() -> None:
    source = _source()
    selected, sampling = select_balanced_sample(
        source,
        stage="development",
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
    gold = build_gold_rows(source, maps, scored, structural_census=structural)
    assert len(gold) == sampling["selected_cases"]
    answer = [row for row in gold if row["answer_state"] == "answer_bearing"]
    no_answer = [row for row in gold if row["answer_state"] == "no_answer"]
    assert all(row["gold_candidate_ids"] for row in answer)
    assert all(not row["gold_candidate_ids"] for row in no_answer)
    assert all(row["candidate_ceiling_complete"] for row in gold)
