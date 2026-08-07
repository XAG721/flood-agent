from __future__ import annotations

import gzip
import json

import pytest

from research.frc_rag.whoqa_conflict_coverage import (
    METHODS,
    ROLE_NAMES,
    evaluate_scored_cases,
    load_report,
    prepare_case,
    select_candidates,
    score_cases_resumable,
    write_report,
)


def _raw_case(case_id: int = 7) -> dict:
    contexts = [
        "Alex Example was a painter in London.",
        "Alex Example was a scientist in Paris.",
        "Alex Example was a singer in Rome.",
        "Alex Example was a teacher in Delhi.",
        "Alex Example was also described as a painter.",
    ]
    answers = {
        "0": [["painter", "artist"]],
        "1": [["scientist", "researcher"]],
        "2": [["singer", "vocalist"]],
        "3": [["teacher", "educator"]],
        "4": [["painter", "creative person"]],
    }
    return {
        "q_id": case_id,
        "question_type_id": "occupation",
        "main_ent": "Alex Example",
        "questions": [
            f"What was Alex Example's occupation? Variant {index}"
            for index in range(5)
        ],
        "contexts": contexts,
        "context_ids": [f"Q{index}" for index in range(len(contexts))],
        "answer_by_context": answers,
        "num_distinct_answers": 4,
        "question_type_metadata": {},
    }


def _scored_case(raw: dict) -> dict:
    prepared = prepare_case(raw)
    variants = []
    for variant_index, question in enumerate(prepared["questions"]):
        scored = []
        for index, candidate in enumerate(prepared["candidates"]):
            relevance = 1.0 - index * 0.1
            scored.append(
                {
                    "id": candidate["id"],
                    "scores": {
                        "bm25": relevance,
                        "dense": relevance,
                        "hybrid": relevance,
                        "cross_encoder": relevance,
                    },
                    "role_scores": {
                        role: max(0.0, relevance - role_index * 0.05)
                        for role_index, role in enumerate(ROLE_NAMES)
                    },
                }
            )
        variants.append(
            {
                "variant_index": variant_index,
                "question": question,
                "candidate_scores": scored,
            }
        )
    return {
        "dataset": "WhoQA",
        "id": prepared["id"],
        "property_type": prepared["property_type"],
        "required_roles": list(ROLE_NAMES),
        "candidates": prepared["candidates"],
        "variants": variants,
        "gold_fields_visible_to_scorer": False,
    }


def test_prepare_case_excludes_gold_fields() -> None:
    prepared = prepare_case(_raw_case())
    encoded = json.dumps(prepared)
    assert len(prepared["questions"]) == 5
    assert len(prepared["candidates"]) == 5
    assert "answer_by_context" not in encoded
    assert "num_distinct_answers" not in encoded
    assert "main_ent" not in encoded


def test_frozen_selectors_respect_budget_and_ascending_ties() -> None:
    candidates = []
    for candidate_id in ("b", "a", "c"):
        candidates.append(
            {
                "id": candidate_id,
                "token_count": 6,
                "scores": {
                    "bm25": 1.0,
                    "dense": 1.0,
                    "hybrid": 1.0,
                    "cross_encoder": 1.0,
                },
                "role_scores": {role: 1.0 for role in ROLE_NAMES},
            }
        )
    for method in METHODS:
        selected = select_candidates(
            candidates, method, top_k=2, token_budget=12
        )
        assert [row["id"] for row in selected] == ["a", "b"]
        assert sum(row["token_count"] for row in selected) <= 12


def test_evaluation_joins_gold_only_after_selection_and_is_reloadable(tmp_path) -> None:
    raw = _raw_case()
    scored = _scored_case(raw)
    report, evidence = evaluate_scored_cases(
        [raw], [scored], expected_cases=None
    )
    assert report["metadata"]["cases"] == 1
    assert report["metadata"]["selection_runs"] == 30
    assert report["metadata"]["deterministic_output_rerun"] == "2/2 byte-identical"
    assert report["data_audit"]["raw_questions_or_answers_in_evidence"] is False
    assert (
        report["data_audit"][
            "public_num_distinct_answer_mismatches_against_canonical_viewpoints"
        ]
        == 0
    )
    assert evidence[0]["raw_question_or_answer_exported"] is False
    assert "questions" not in evidence[0]
    assert "answer_by_context" not in evidence[0]

    source = tmp_path / "source.json"
    source.write_text("[]\n", encoding="utf-8")
    paths = write_report(
        report,
        evidence,
        tmp_path / "out",
        source_paths={"source": source},
    )
    loaded = load_report(paths["json"], paths["evidence"])
    assert loaded["metadata"]["cases"] == 1
    with gzip.open(paths["evidence"], "rt", encoding="utf-8") as handle:
        exported = handle.read()
    assert "What was Alex" not in exported
    assert "painter" not in exported


def test_deterministic_evidence_gzip(tmp_path) -> None:
    raw = _raw_case()
    report, evidence = evaluate_scored_cases(
        [raw], [_scored_case(raw)], expected_cases=None
    )
    source = tmp_path / "source.json"
    source.write_text("[]\n", encoding="utf-8")
    first = write_report(
        report,
        evidence,
        tmp_path / "first",
        source_paths={"source": source},
    )
    second = write_report(
        report,
        evidence,
        tmp_path / "second",
        source_paths={"source": source},
    )
    assert first["evidence"].read_bytes() == second["evidence"].read_bytes()


def test_resumable_scoring_rejects_a_second_process_lock(tmp_path) -> None:
    prepared = tmp_path / "prepared.jsonl"
    output = tmp_path / "scored.jsonl"
    prepared.write_text("{}\n", encoding="utf-8")
    lock = output.with_suffix(output.suffix + ".lock")
    lock.write_text("123", encoding="ascii")
    with pytest.raises(RuntimeError, match="owns the lock"):
        score_cases_resumable(prepared, output, scorer=object())


def test_resumable_scoring_batches_cases_and_preserves_order(tmp_path) -> None:
    prepared = tmp_path / "prepared.jsonl"
    output = tmp_path / "scored.jsonl"
    rows = [{"id": str(index)} for index in range(5)]
    prepared.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    class FakeBatchScorer:
        batches: list[list[str]] = []

        def score_cases(self, cases):
            self.batches.append([row["id"] for row in cases])
            return [{"id": row["id"]} for row in cases]

    scorer = FakeBatchScorer()
    count = score_cases_resumable(
        prepared, output, scorer=scorer, case_batch_size=2
    )
    assert count == 5
    assert scorer.batches == [["0", "1"], ["2", "3"], ["4"]]
    assert [row["id"] for row in map(json.loads, output.read_text().splitlines())] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
