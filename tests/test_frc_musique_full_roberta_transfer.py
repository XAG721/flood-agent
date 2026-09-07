from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import research.frc_rag.musique_full_roberta_transfer as v65


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/musique_full_roberta_transfer_protocol_v65.json"
)


def _row(index: int, *, answerable: bool, overlap: bool = False) -> dict:
    question = "What is the guarded answer?" if overlap else f"Question {index}?"
    return {
        "id": f"2hop__{index}_{index + 1}",
        "question": question,
        "answerable": answerable,
        "answer": f"answer {index}",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": str(index),
                "question": f"First step {index}?",
                "answer": "first",
                "paragraph_support_idx": 0 if answerable else None,
            },
            {
                "id": str(index + 1),
                "question": f"Second step {index}?",
                "answer": f"answer {index}",
                "paragraph_support_idx": 1 if answerable else None,
            },
        ],
        "paragraphs": [
            {
                "idx": paragraph,
                "title": f"Title {paragraph}",
                "paragraph_text": f"Paragraph {paragraph} has answer {index}.",
                "is_supporting": answerable and paragraph < 2,
            }
            for paragraph in range(3)
        ],
    }


def _decision(input_row: dict, margin: float, *, invalid: bool = False) -> dict:
    return {
        "id": input_row["id"],
        "cache_key": "cache",
        "score_margin": margin,
        "span_sha256": f"span-{input_row['id']}",
        "invalid_fail_closed_used": invalid,
    }


def _support_evidence(*, passing: bool) -> list[dict]:
    rows = []
    for index in range(300):
        rows.append(
            {
                "case_id": f"a-{index}",
                "answer_state": "answerable",
                "hop_count": 2 + index % 2,
                "paragraph_count": 20,
                "local_span_trap": False,
                "support_passed": passing or index < 150,
                "invalid_fail_closed_used": False,
                "score_margin": 2.0,
                "selected_paragraph_idx": 0,
                "raw_prediction_sha256": f"a-{index}",
            }
        )
        rows.append(
            {
                "case_id": f"n-{index}",
                "answer_state": "unanswerable",
                "hop_count": 2 + index % 2,
                "paragraph_count": 20,
                "local_span_trap": index < 150,
                "support_passed": False if passing else index >= 150,
                "invalid_fail_closed_used": False,
                "score_margin": -2.0,
                "selected_paragraph_idx": 0,
                "raw_prediction_sha256": f"n-{index}",
            }
        )
    return rows


def _sampling() -> dict:
    return {
        "schema_exclusion_rate": 0.0,
        "selected_squad2_exact_question_overlap": 0,
        "selected_v36_source_id_overlap": 0,
        "selected_invalid_run_source_overlap": 0,
    }


def _structural() -> dict:
    return {"minimum_paragraphs": 20}


def test_protocol_freezes_target_before_parse_and_one_threshold() -> None:
    value = v65.validate_protocol(PROTOCOL)
    assert value["prior_boundary"]["v65_target_content_opened_or_parsed_before_this_protocol"] is False
    assert value["support_transfer"]["locked_threshold_exact"] == 0.974609375
    assert value["support_transfer"]["learned_or_adjusted_parameter_count_on_musique"] == 0


def test_question_normalization_and_v36_id_loading() -> None:
    assert v65.normalize_question("  A   QUESTION? ") == "a question?"
    assert v65.load_v36_source_ids([{"id": "musique::2hop__1_2"}]) == {
        "2hop__1_2"
    }


def test_balanced_selection_is_deterministic_and_excludes_squad_and_v36() -> None:
    rows = [
        *(_row(index, answerable=True) for index in range(30)),
        *(_row(index + 100, answerable=False) for index in range(30)),
        _row(1000, answerable=True, overlap=True),
        _row(1001, answerable=False),
    ]
    squad = {v65.normalize_question("What is the guarded answer?")}
    v36_ids = {"2hop__1001_1002"}
    first, summary = v65.select_balanced_sample(
        rows,
        squad_questions=squad,
        v36_source_ids=v36_ids,
        target_per_group=4,
    )
    second, _ = v65.select_balanced_sample(
        rows,
        squad_questions=squad,
        v36_source_ids=v36_ids,
        target_per_group=4,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert summary["selected_answer_state_counts"] == {
        "answerable": 4,
        "unanswerable": 4,
    }
    assert summary["squad2_exact_question_overlap_excluded"] == 1
    assert summary["v36_source_id_overlap_excluded"] == 1


def test_corrected_selection_assigns_paired_ids_once_and_excludes_invalid_run() -> None:
    paired = [
        row
        for index in range(60)
        for row in (
            _row(index, answerable=True),
            _row(index, answerable=False),
        )
    ]
    excluded = {v65._hash("2hop__0_1")}
    selected, summary = v65.select_balanced_sample(
        paired,
        squad_questions={"unrelated"},
        v36_source_ids={"unrelated"},
        excluded_source_commitments=excluded,
        target_per_group=10,
    )
    source_ids = [row["id"] for row in selected]
    assert len(source_ids) == len(set(source_ids)) == 20
    assert "2hop__0_1" not in source_ids
    assert summary["selected_invalid_run_source_overlap"] == 0
    assert summary["invalid_run_source_rows_excluded"] == 2


def test_blind_qa_exports_no_gold_fields() -> None:
    selected = [_row(1, answerable=True), _row(2, answerable=False)]
    prepared, maps, qa_inputs, census = v65.prepare_blind_qa(selected)
    assert len(prepared) == len(maps) == 2
    assert len(qa_inputs) == 6
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "question_decomposition",
        "is_supporting",
        "paragraph_support_idx",
    }
    assert not forbidden & v65.nested_keys([prepared, maps, qa_inputs])
    assert census["gold_fields_exported_to_blind_caches"] is False


def test_paragraph_aggregation_uses_strict_max_margin_and_fails_invalid_closed() -> None:
    selected = [_row(1, answerable=True), _row(2, answerable=False)]
    _, _, qa_inputs, _ = v65.prepare_blind_qa(selected)
    decisions = []
    for row in qa_inputs:
        paragraph = int(row["paragraph_idx"])
        margin = v65.LOCKED_THRESHOLD if paragraph == 0 else float(paragraph)
        decisions.append(_decision(row, margin))
    aggregated = v65.aggregate_paragraph_decisions(
        qa_inputs, decisions, cache_key="cache"
    )
    assert [row["selected_paragraph_idx"] for row in aggregated] == [2, 2]
    assert all(row["support_passed"] for row in aggregated)

    invalid = [_decision(row, 99.0, invalid=True) for row in qa_inputs]
    closed = v65.aggregate_paragraph_decisions(qa_inputs, invalid, cache_key="cache")
    assert all(row["invalid_fail_closed_used"] for row in closed)
    assert not any(row["support_passed"] for row in closed)


def test_support_gate_stops_retrieval_when_transfer_fails() -> None:
    report = v65.evaluate_support_gate(
        _support_evidence(passing=False),
        sampling=_sampling(),
        structural_census=_structural(),
        source_artifacts={},
    )
    outcome = report["analysis"]["outcome"]
    assert outcome["support_gate_passed"] is False
    assert outcome["retrieval_scoring_open_authorized"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"


def test_support_gate_can_open_retrieval_without_authorizing_adoption() -> None:
    report = v65.evaluate_support_gate(
        _support_evidence(passing=True),
        sampling=_sampling(),
        structural_census=_structural(),
        source_artifacts={},
    )
    outcome = report["analysis"]["outcome"]
    assert outcome["support_gate_passed"] is True
    assert outcome["retrieval_scoring_open_authorized"] is True
    assert outcome["selector_adoption_authorized"] is False
    assert outcome["canary_or_default_authorized"] is False


def test_report_archive_is_deterministic(tmp_path: Path) -> None:
    report = v65.evaluate_support_gate(
        _support_evidence(passing=True),
        sampling=_sampling(),
        structural_census=_structural(),
        source_artifacts={},
    )
    archives = []
    for index in range(2):
        archive = tmp_path / f"cases-{index}.jsonl.gz"
        v65.write_report(
            report,
            [{"case_id": "one"}],
            tmp_path / f"result-{index}.json",
            tmp_path / f"report-{index}.md",
            archive,
        )
        archives.append(archive)
    assert archives[0].read_bytes() == archives[1].read_bytes()
    with gzip.open(archives[0], "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"case_id": "one"}


def test_insufficient_balance_fails_before_qa() -> None:
    rows = [_row(index, answerable=True) for index in range(5)]
    with pytest.raises(ValueError, match="insufficient eligible balanced"):
        v65.select_balanced_sample(
            rows,
            squad_questions={"unrelated"},
            v36_source_ids={"unrelated"},
            target_per_group=2,
        )
