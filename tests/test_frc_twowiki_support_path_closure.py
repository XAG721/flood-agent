from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.twowiki_support_path_closure import (
    CANDIDATE_METHOD,
    CONTROL_METHODS,
    QUESTION_TYPES,
    build_title_graph,
    evaluate_stage,
    load_history_ids,
    prepare_case,
    read_jsonl_gzip,
    select_method,
    select_stage_ids,
    validate_finite,
    validate_registered_protocol,
    write_jsonl,
    write_jsonl_gzip,
    write_selection_outputs,
)


def _candidate(
    candidate_id: str,
    source: str,
    text: str,
    cross: float,
    *,
    tokens: int = 10,
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": text,
        "token_count": tokens,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": {
            "condition": cross,
            "attribution": cross,
            "procedure": cross,
            "answer": cross,
            "exception": cross,
        },
    }


def _scored_row(case_id: str = "case-1") -> dict[str, object]:
    return {
        "dataset": "2WikiMultiHopQA",
        "source": "synthetic",
        "id": case_id,
        "question": "How are Alpha and Beta connected?",
        "question_type": "compositional",
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": [
            _candidate("a", "Alpha", "Alpha. Alpha was founded by Beta.", 1.0),
            _candidate("b", "Beta", "Beta. Beta was founded in 1900.", 0.2),
            _candidate("c", "Gamma", "Gamma. An unrelated high score.", 0.9),
            _candidate("d", "Delta", "Delta. Another distractor.", 0.8),
            _candidate("e", "Epsilon", "Epsilon. Another distractor.", 0.7),
            _candidate("f", "Zeta", "Zeta. Another distractor.", 0.6),
        ],
    }


def test_title_graph_uses_cross_title_exact_surface_and_symmetrizes() -> None:
    row = _scored_row()
    graph = build_title_graph(row["candidates"])

    assert "b" in graph["a"]
    assert "a" in graph["b"]
    assert "c" not in graph["a"]


def test_title_graph_requires_alphanumeric_boundaries() -> None:
    candidates = [
        _candidate("a", "Alpha", "Alpha. Alphabet is not the same title.", 1.0),
        _candidate("b", "Bet", "Bet. Target.", 0.5),
    ]

    assert build_title_graph(candidates) == {"a": set(), "b": set()}


def test_candidate_closes_linked_path_before_unlinked_high_score() -> None:
    selected = select_method(_scored_row(), CANDIDATE_METHOD, k=2)

    assert [item["id"] for item in selected] == ["a", "b"]


def test_no_diversity_control_is_available_and_deterministic() -> None:
    first = select_method(
        _scored_row(),
        "cross_plus_title_link_without_source_diversity_control",
    )
    second = select_method(
        _scored_row(),
        "cross_plus_title_link_without_source_diversity_control",
    )

    assert [item["id"] for item in first] == [item["id"] for item in second]


def test_all_registered_controls_execute_on_same_scored_row() -> None:
    row = _scored_row()

    for method in CONTROL_METHODS:
        selected = select_method(row, method)
        assert 1 <= len(selected) <= 5
        assert len({item["id"] for item in selected}) == len(selected)


def test_selector_respects_token_budget() -> None:
    row = _scored_row()
    for candidate in row["candidates"]:
        candidate["token_count"] = 7

    selected = select_method(row, CANDIDATE_METHOD, token_budget=15)

    assert len(selected) == 2
    assert sum(item["token_count"] for item in selected) <= 15


def test_stage_selection_is_balanced_deterministic_and_disjoint() -> None:
    rows = [
        {"_id": f"{question_type}-{index:04d}", "type": question_type}
        for question_type in QUESTION_TYPES
        for index in range(500)
    ]
    history = {f"{question_type}-{index:04d}" for question_type in QUESTION_TYPES for index in range(10)}

    development = select_stage_ids(rows, history_ids=history, stage="development")
    repeated = select_stage_ids(rows, history_ids=history, stage="development")
    confirmation = select_stage_ids(rows, history_ids=history, stage="confirmation")

    assert development == repeated
    assert len(development) == len(set(development)) == 800
    assert len(confirmation) == len(set(confirmation)) == 800
    assert not set(development) & set(confirmation)
    assert not set(development) & history
    assert not set(confirmation) & history
    assert {
        question_type: sum(case_id.startswith(f"{question_type}-") for case_id in development)
        for question_type in QUESTION_TYPES
    } == {question_type: 200 for question_type in QUESTION_TYPES}


def test_stage_selection_rejects_insufficient_type_capacity() -> None:
    rows = [
        {"_id": f"{question_type}-{index}", "type": question_type}
        for question_type in QUESTION_TYPES
        for index in range(199)
    ]

    with pytest.raises(ValueError, match="lacks 200 eligible"):
        select_stage_ids(rows, history_ids=set(), stage="development")


def test_prepare_case_seals_gold_and_removes_answer_from_blind_row() -> None:
    row = {
        "_id": "x",
        "type": "compositional",
        "question": "Question?",
        "answer": "secret",
        "context": json.dumps(
            [["Alpha", ["Alpha cites Beta."]], ["Beta", ["Beta gives answer."]]]
        ),
        "supporting_facts": json.dumps([["Alpha", 0], ["Beta", 0]]),
    }

    blind, gold = prepare_case(row)

    assert "answer" not in blind
    assert "gold_evidence_ids" not in blind
    assert all("gold" not in candidate for candidate in blind["candidates"])
    assert len(gold["gold_evidence_ids"]) == 2


def test_prepare_case_rejects_missing_supporting_fact() -> None:
    row = {
        "_id": "x",
        "type": "compositional",
        "question": "Question?",
        "context": [["Alpha", ["Only one."]]],
        "supporting_facts": [["Alpha", 0], ["Missing", 0]],
    }

    with pytest.raises(ValueError, match="missing supporting facts"):
        prepare_case(row)


def test_history_loader_requires_exact_unique_count(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    write_jsonl(path, [{"id": "a"}, {"id": "b"}])

    assert load_history_ids(path, expected_count=2) == {"a", "b"}
    with pytest.raises(ValueError, match="requires 3 unique ids"):
        load_history_ids(path, expected_count=3)


def test_selection_output_rejects_gold_bearing_model_rows(tmp_path: Path) -> None:
    row = {**_scored_row(), "gold_evidence_ids": ["a", "b"]}

    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs([row], tmp_path / "selection.jsonl")


def test_selection_output_contains_no_question_or_candidate_text(tmp_path: Path) -> None:
    path = tmp_path / "selection.jsonl"

    rows = write_selection_outputs([_scored_row()], path)
    payload = path.read_text(encoding="utf-8")

    assert rows[0]["case_id"] == "case-1"
    assert '"question":' not in payload.lower()
    assert "founded" not in payload.lower()


def test_evaluation_writes_safe_audited_evidence(tmp_path: Path) -> None:
    scored_path = tmp_path / "scored.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    rows = []
    gold = []
    for index, question_type in enumerate(QUESTION_TYPES):
        row = _scored_row(f"case-{index}")
        row["question_type"] = question_type
        rows.append(row)
        gold.append(
            {
                "id": row["id"],
                "question_type": question_type,
                "gold_evidence_ids": ["a", "b"],
            }
        )
    write_jsonl(scored_path, rows)
    write_jsonl(gold_path, gold)

    report = evaluate_stage(
        scored_path,
        gold_path,
        selection_path,
        stage="development",
        seed=7,
        output_dir=tmp_path / "output",
        history_overlap=0,
    )
    cases = read_jsonl_gzip(tmp_path / "output/cases.jsonl.gz")

    assert report["metadata"]["gold_join_started_after_selection_output_written"] is True
    assert report["metadata"]["invalid_selector_output_count"] == 0
    candidate = report["analysis"]["methods"][CANDIDATE_METHOD]
    assert candidate["evidence_macro_f1"] == 0.571429
    assert candidate["evidence_macro_recall"] == 1.0
    assert candidate["complete_evidence_recall"] == 1.0
    assert report["analysis"]["support_checks"]["exact_cases_equals_800"] is False
    assert all("question" not in row and "answer" not in row for row in cases)


def test_gzip_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    rows = [{"case_id": "a", "value": 1.0}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    write_jsonl_gzip(first, rows)
    write_jsonl_gzip(second, rows)

    assert first.read_bytes() == second.read_bytes()


def test_protocol_validation_accepts_registered_contract() -> None:
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/progressive_upgrade/twowiki_support_path_closure_protocol_v74.json"
        ).read_text(encoding="utf-8")
    )

    validate_registered_protocol(protocol)


def test_protocol_validation_rejects_weight_drift() -> None:
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/progressive_upgrade/twowiki_support_path_closure_protocol_v74.json"
        ).read_text(encoding="utf-8")
    )
    protocol["candidate"]["link_bonus"] = 0.9

    with pytest.raises(ValueError, match="weights"):
        validate_registered_protocol(protocol)


def test_report_values_must_be_finite() -> None:
    assert validate_finite({"a": [0.0, 1.0]}) is True
    assert validate_finite({"a": [float("nan")]}) is False
