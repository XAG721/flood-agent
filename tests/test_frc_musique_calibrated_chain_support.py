from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag import musique_calibrated_chain_support as experiment


def _source(source_id: str, answerable: bool) -> dict:
    return {
        "id": source_id,
        "question": f"Composed question {source_id}?",
        "answerable": answerable,
        "answer": "gold",
        "answer_aliases": [],
        "question_decomposition": [
            {"id": 1, "question": f"First {source_id}?", "answer": "bridge", "paragraph_support_idx": 0},
            {"id": 2, "question": "Second #1?", "answer": "gold", "paragraph_support_idx": 1},
        ],
        "paragraphs": [
            {"idx": index, "title": f"T{index}", "paragraph_text": f"Context {source_id} {index}"}
            for index in range(3)
        ],
    }


def _balanced_rows(count: int = 30) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        source_id = f"source-{index}"
        rows.extend([_source(source_id, True), _source(source_id, False)])
    return rows


def _prepared() -> list[dict]:
    return [
        {
            "id": "m67k-case",
            "oracle_hop_templates": ["First?", "Second #1?"],
            "contexts": [
                {"paragraph_idx": 0, "context": "Alpha bridge", "context_sha256": "a"},
                {"paragraph_idx": 1, "context": "Beta answer", "context_sha256": "b"},
            ],
        }
    ]


def _evidence(answer_score: float, noanswer_score: float, count: int = 20) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        rows.append(
            {
                "case_id": f"a-{index}",
                "answer_state": "answerable",
                "hop_count": 2 + index % 3,
                "direct_score_margin": answer_score - 1.0,
                "chain_bottleneck_score": answer_score,
                "invalid_chain_output": False,
            }
        )
        rows.append(
            {
                "case_id": f"n-{index}",
                "answer_state": "unanswerable",
                "hop_count": 2 + index % 3,
                "direct_score_margin": noanswer_score + 1.0,
                "chain_bottleneck_score": noanswer_score,
                "invalid_chain_output": False,
            }
        )
    return rows


def test_select_stage_sample_is_balanced_disjoint_and_deterministic() -> None:
    rows = _balanced_rows()
    excluded_id = "source-0"
    excluded = {hashlib.sha256(excluded_id.encode()).hexdigest()}
    first, census = experiment.select_stage_sample(
        rows,
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=excluded,
        target_per_group=5,
    )
    second, second_census = experiment.select_stage_sample(
        rows,
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=excluded,
        target_per_group=5,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert census == second_census
    assert len(first) == 10
    assert len({row["id"] for row in first}) == 10
    assert sum(row["answerable"] for row in first) == 5
    assert excluded_id not in {row["id"] for row in first}


def test_select_stage_sample_rejects_split_mismatch() -> None:
    with pytest.raises(ValueError, match="stage/source split"):
        experiment.select_stage_sample(
            [],
            stage="confirmation",
            source_split="train",
            squad_questions=set(),
            excluded_source_commitments=set(),
            target_per_group=1,
        )


def test_prepare_blind_stage_removes_all_gold_fields() -> None:
    selected = [_source("alpha", True), _source("beta", False)]
    prepared, maps, direct, census = experiment.prepare_blind_stage(
        selected, stage="calibration"
    )
    keys = experiment.nested_keys([prepared, maps, direct])
    assert not {"answerable", "answer", "answer_aliases", "paragraph_support_idx"} & keys
    assert census["prepared_cases"] == 2
    assert census["direct_qa_input_rows"] == 6
    assert all(row["source_id_commitment"] for row in maps)


def test_aggregate_qa_rows_keeps_best_finite_span_without_thresholding() -> None:
    inputs = [
        {"id": "c::p0", "case_id": "c", "paragraph_idx": 0, "context": "alpha"},
        {"id": "c::p1", "case_id": "c", "paragraph_idx": 1, "context": "beta"},
    ]
    raw = [
        {
            "id": "c::p0",
            "cache_key": "k",
            "invalid_fail_closed_used": False,
            "score_margin": -2.0,
            "span_start": 0,
            "span_end": 5,
        },
        {
            "id": "c::p1",
            "cache_key": "k",
            "invalid_fail_closed_used": False,
            "score_margin": -1.0,
            "span_start": 0,
            "span_end": 4,
        },
    ]
    [result] = experiment.aggregate_qa_rows(inputs, raw, cache_key="k")
    assert result["score_margin"] == -1.0
    assert result["predicted_span"] == "beta"
    assert "support_passed" not in result


def test_below_v66_threshold_still_propagates_to_next_hop() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    [decision] = experiment.apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "m67k-case",
                "score_margin": -3.0,
                "predicted_span": "bridge",
                "predicted_span_sha256": "hash",
                "selected_paragraph_idx": 0,
                "invalid_fail_closed_used": False,
            }
        ],
        {},
        hop_index=1,
    )
    assert decision["valid_span"] is True
    assert states["m67k-case"]["alive"] is True
    assert states["m67k-case"]["predictions"][1] == "bridge"
    next_inputs, blocked = experiment.build_hop_qa_inputs(prepared, states, hop_index=2)
    assert not blocked
    assert {row["question"] for row in next_inputs} == {"Second bridge?"}


def test_invalid_span_fails_closed_and_blocks_successor() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    experiment.apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "m67k-case",
                "score_margin": None,
                "predicted_span": "",
                "predicted_span_sha256": "hash",
                "selected_paragraph_idx": None,
                "invalid_fail_closed_used": True,
            }
        ],
        {},
        hop_index=1,
    )
    inputs, blocked = experiment.build_hop_qa_inputs(prepared, states, hop_index=2)
    assert inputs == []
    assert blocked == {"m67k-case": "invalid_predecessor_output"}


def test_finalize_chain_score_uses_minimum_margin() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    states["m67k-case"].update(
        {"alive": True, "completed_hops": 2, "margins": [2.5, -0.25]}
    )
    hops = [
        {"case_id": "m67k-case", "hop_index": 1, "executed": True, "valid_span": True},
        {"case_id": "m67k-case", "hop_index": 2, "executed": True, "valid_span": True},
    ]
    [result] = experiment.finalize_chain_scores(prepared, states, hops)
    assert result["chain_complete"] is True
    assert result["chain_bottleneck_score"] == -0.25


def test_threshold_selection_obeys_strict_greater_and_safety_constraints() -> None:
    evidence = _evidence(answer_score=2.0, noanswer_score=0.0)
    result = experiment.select_threshold(evidence, score_field="chain_bottleneck_score")
    assert result["safety_constraints_met"] is True
    assert result["answerable_pass_rate"] == 1.0
    assert result["unanswerable_rejection_rate"] == 1.0
    assert experiment._decision(0.0, 0.0) is False


def test_crossfit_thresholds_never_use_held_out_fold() -> None:
    result = experiment.crossfit_threshold_diagnostic(
        _evidence(answer_score=2.0, noanswer_score=0.0, count=50),
        score_field="chain_bottleneck_score",
    )
    assert result["held_out_cases"] == 100
    assert len(result["folds"]) == 5
    assert all(row["train_cases"] + row["held_out_cases"] == 100 for row in result["folds"])
    assert result["balanced_accuracy"] == 1.0


def test_evaluate_stage_compares_one_strongest_method_and_opens_on_all_gates() -> None:
    evidence = _evidence(answer_score=0.5, noanswer_score=0.0, count=300)
    for row in evidence:
        row["direct_score_margin"] = 0.0
    calibration = experiment.fit_calibration(evidence[:400])
    report = experiment.evaluate_stage(
        evidence,
        stage="development",
        calibration=calibration,
        sampling={
            "schema_exclusion_rate": 0.0,
            "selected_excluded_source_commitment_overlap": 0,
            "selected_squad2_exact_question_overlap": 0,
        },
        structural_census={},
        source_artifacts={},
    )
    assert report["analysis"]["outcome"]["stage_gate_passed"] is True
    assert report["analysis"]["outcome"]["confirmation_open_authorized"] is True
    assert report["analysis"]["strongest_fair_baseline"]["name"] == "calibrated_direct_composed_question"


def test_write_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _evidence(answer_score=2.0, noanswer_score=0.0, count=2)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    experiment.write_evidence(first, rows)
    experiment.write_evidence(second, rows)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert len([json.loads(line) for line in handle]) == 4


def test_validate_protocol_rejects_gate_promotion(tmp_path: Path) -> None:
    source = Path("docs/progressive_upgrade/musique_calibrated_chain_support_protocol_v67.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["stopping_and_outcomes"]["gate_2"] = "GO"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate 2"):
        experiment.validate_protocol(changed)
