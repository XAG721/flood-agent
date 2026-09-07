from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from research.frc_rag.musique_sequential_chain_support import (
    LOCKED_THRESHOLD,
    _hash,
    aggregate_qa_rows,
    apply_hop_results,
    build_gold_evidence,
    build_hop_qa_inputs,
    evaluate_stage,
    finalize_chain_decisions,
    initial_chain_states,
    paired_bootstrap,
    prepare_blind_stage,
    resolve_placeholders,
    select_stage_sample,
    write_stage_report,
)


def _source(source_id: str, *, answerable: bool, hops: int = 2) -> dict:
    decomposition = []
    for hop in range(1, hops + 1):
        question = f"Who is entity {source_id}?" if hop == 1 else f"Where was #{hop - 1} born?"
        decomposition.append(
            {
                "id": f"{source_id}-{hop}",
                "question": question,
                "answer": f"gold-{hop}",
                "paragraph_support_idx": hop - 1 if answerable else None,
            }
        )
    return {
        "id": source_id,
        "question": f"Composed question {source_id}?",
        "answerable": answerable,
        "answer": "gold",
        "answer_aliases": [],
        "question_decomposition": decomposition,
        "paragraphs": [
            {
                "idx": index,
                "title": f"Title {index}",
                "paragraph_text": f"Context {source_id} paragraph {index} Ada London.",
                "is_supporting": answerable and index < hops,
            }
            for index in range(3)
        ],
    }


def _prepared(case_id: str, *, hops: int = 2) -> dict:
    return {
        "id": case_id,
        "oracle_hop_templates": [
            "Who is the person?",
            "Where was #1 born?",
            "Which country contains #2?",
            "What is the capital of #3?",
        ][:hops],
        "contexts": [
            {
                "paragraph_idx": index,
                "context": f"Context {index} Ada London England.",
                "context_sha256": _hash(f"Context {index} Ada London England."),
            }
            for index in range(3)
        ],
    }


def test_stage_sampling_is_balanced_deterministic_and_excludes_commitments() -> None:
    rows = [
        _source(source_id, answerable=answerable)
        for index in range(80)
        for source_id in [f"case-{index}"]
        for answerable in (True, False)
    ]
    excluded_id = "case-7"
    first, census = select_stage_sample(
        rows,
        stage="development",
        squad_questions=set(),
        excluded_source_commitments={_hash(excluded_id)},
        target_per_group=5,
    )
    second, _ = select_stage_sample(
        reversed(rows),
        stage="development",
        squad_questions=set(),
        excluded_source_commitments={_hash(excluded_id)},
        target_per_group=5,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == 10
    assert all(row["id"] != excluded_id for row in first)
    assert census["selected_answer_state_counts"] == {
        "answerable": 5,
        "unanswerable": 5,
    }
    assert census["selected_excluded_source_commitment_overlap"] == 0


def test_stage_sampling_excludes_exact_squad_question_overlap() -> None:
    rows = [
        _source(f"case-{index}", answerable=answerable)
        for index in range(80)
        for answerable in (True, False)
    ]
    rows[0]["question_decomposition"][0]["question"] = "Known SQuAD question?"
    _, census = select_stage_sample(
        rows,
        stage="development",
        squad_questions={"known squad question?"},
        excluded_source_commitments=set(),
        target_per_group=5,
    )
    assert census["squad2_exact_question_overlap_excluded"] == 1
    assert census["selected_squad2_exact_question_overlap"] == 0


def test_blind_preparation_keeps_oracle_questions_but_drops_gold() -> None:
    selected = [_source("case-a", answerable=True)]
    prepared, maps, direct, census = prepare_blind_stage(
        selected, stage="development"
    )
    encoded = json.dumps([prepared, maps, direct], ensure_ascii=False)
    assert "oracle_hop_templates" in encoded
    for forbidden in (
        '"answerable"',
        '"answer"',
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    ):
        assert forbidden not in encoded
    assert len(direct) == 3
    assert census["hop_count_distribution"] == {"2": 1}
    assert census["gold_fields_exported_to_blind_caches"] is False


def test_aggregate_qa_rows_uses_margin_tie_break_and_recovers_span() -> None:
    cache_key = "frozen"
    inputs = [
        {
            "id": f"case::p{index}",
            "case_id": "case",
            "paragraph_idx": index,
            "context": context,
        }
        for index, context in ((0, "Ada London"), (1, "Grace Paris"))
    ]
    decisions = [
        {
            "id": "case::p0",
            "cache_key": cache_key,
            "score_margin": LOCKED_THRESHOLD + 1,
            "span_start": 0,
            "span_end": 3,
            "invalid_fail_closed_used": False,
        },
        {
            "id": "case::p1",
            "cache_key": cache_key,
            "score_margin": LOCKED_THRESHOLD + 1,
            "span_start": 0,
            "span_end": 5,
            "invalid_fail_closed_used": False,
        },
    ]
    result = aggregate_qa_rows(inputs, decisions, cache_key=cache_key)
    assert result[0]["selected_paragraph_idx"] == 0
    assert result[0]["predicted_span"] == "Ada"
    assert result[0]["support_passed"] is True


def test_placeholder_resolution_uses_only_prior_predictions() -> None:
    resolved, missing = resolve_placeholders(
        "Where was #1 born?", {1: "Ada Lovelace"}, hop_index=2
    )
    assert resolved == "Where was Ada Lovelace born?"
    assert missing == []
    unresolved, missing = resolve_placeholders(
        "Is #2 related to #1?", {1: "Ada"}, hop_index=2
    )
    assert unresolved is None
    assert missing == [2]


def test_sequential_execution_fails_closed_after_unsupported_hop() -> None:
    prepared = [_prepared("case-a"), _prepared("case-b")]
    states = initial_chain_states(prepared)
    hop1_inputs, blocked = build_hop_qa_inputs(prepared, states, hop_index=1)
    assert len(hop1_inputs) == 6
    assert blocked == {}
    aggregated = [
        {
            "case_id": "case-a",
            "support_passed": True,
            "score_margin": 2.0,
            "selected_paragraph_idx": 0,
            "predicted_span": "Ada",
            "predicted_span_sha256": _hash("Ada"),
            "invalid_fail_closed_used": False,
        },
        {
            "case_id": "case-b",
            "support_passed": False,
            "score_margin": -1.0,
            "selected_paragraph_idx": 1,
            "predicted_span": "noise",
            "predicted_span_sha256": _hash("noise"),
            "invalid_fail_closed_used": False,
        },
    ]
    first = apply_hop_results(
        prepared, states, aggregated, blocked, hop_index=1
    )
    hop2_inputs, blocked2 = build_hop_qa_inputs(prepared, states, hop_index=2)
    assert len(hop2_inputs) == 3
    assert "Ada" in hop2_inputs[0]["question"]
    assert blocked2 == {"case-b": "predecessor_hop_unsupported"}
    second = apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "case-a",
                "support_passed": True,
                "score_margin": 3.0,
                "selected_paragraph_idx": 2,
                "predicted_span": "London",
                "predicted_span_sha256": _hash("London"),
                "invalid_fail_closed_used": False,
            }
        ],
        blocked2,
        hop_index=2,
    )
    chain = finalize_chain_decisions(prepared, states, [*first, *second])
    assert [row["support_passed"] for row in chain] == [True, False]
    assert chain[1]["executed_hops"] == 1


def _passing_evidence() -> list[dict]:
    rows: list[dict] = []
    for index in range(300):
        rows.append(
            {
                "case_id": f"a-{index}",
                "answer_state": "answerable",
                "hop_count": 2 + index % 3,
                "direct_support_passed": index < 180,
                "sequential_chain_support_passed": index < 240,
                "invalid_chain_output": False,
            }
        )
    for index in range(300):
        rows.append(
            {
                "case_id": f"u-{index}",
                "answer_state": "unanswerable",
                "hop_count": 2 + index % 3,
                "direct_support_passed": index < 240,
                "sequential_chain_support_passed": index < 30,
                "invalid_chain_output": False,
            }
        )
    return rows


def _sampling() -> dict:
    return {
        "schema_exclusion_rate": 0.0,
        "selected_excluded_source_commitment_overlap": 0,
        "selected_squad2_exact_question_overlap": 0,
    }


def test_stage_evaluation_opens_confirmation_only_when_all_gates_pass() -> None:
    report = evaluate_stage(
        _passing_evidence(),
        stage="development",
        sampling=_sampling(),
        structural_census={},
        source_artifacts={},
    )
    analysis = report["analysis"]
    assert analysis["oracle_plan_sequential_chain"]["balanced_accuracy"] == 0.85
    assert analysis["paired_correctness_delta"]["point"] == 0.45
    assert all(analysis["support_checks"].values())
    assert analysis["outcome"]["confirmation_open_authorized"] is True
    assert analysis["outcome"]["selector_adoption_authorized"] is False
    assert analysis["outcome"]["gate_2"] == "NO-GO/SHADOW"


def test_stage_evaluation_fails_without_rewriting_gate() -> None:
    rows = _passing_evidence()
    for row in rows:
        row["sequential_chain_support_passed"] = row["direct_support_passed"]
    report = evaluate_stage(
        rows,
        stage="development",
        sampling=_sampling(),
        structural_census={},
        source_artifacts={},
    )
    outcome = report["analysis"]["outcome"]
    assert outcome["stage_gate_passed"] is False
    assert outcome["confirmation_open_authorized"] is False
    assert outcome[
        "reuse_failed_stage_for_tuning_threshold_rule_gate_or_selection"
    ] is False


def test_paired_bootstrap_is_deterministic() -> None:
    candidate = np.asarray([1, 1, 1, 0, 1], dtype=float)
    baseline = np.asarray([0, 1, 0, 0, 0], dtype=float)
    first = paired_bootstrap(candidate, baseline, seed=17, resamples=1000)
    second = paired_bootstrap(candidate, baseline, seed=17, resamples=1000)
    assert first == second
    assert first["point"] == 0.6


def test_gold_join_exports_only_hashed_prediction_provenance() -> None:
    source = [_source("case-a", answerable=True)]
    prepared, _, _, _ = prepare_blind_stage(source, stage="development")
    direct = [
        {
            "case_id": prepared[0]["id"],
            "support_passed": True,
            "score_margin": 2.0,
            "decision_sha256": _hash("direct"),
            "invalid_fail_closed_used": False,
        }
    ]
    chain = [
        {
            "case_id": prepared[0]["id"],
            "support_passed": True,
            "executed_hops": 2,
            "passed_hops": 2,
            "hop_decision_sha256": _hash("hops"),
            "invalid_fail_closed_used": False,
        }
    ]
    evidence = build_gold_evidence(source, prepared, direct, chain)
    encoded = json.dumps(evidence)
    assert "gold-" not in encoded
    assert "paragraph_support_idx" not in encoded
    assert evidence[0]["answer_state"] == "answerable"


def test_stage_report_and_gzip_evidence_are_deterministic(tmp_path: Path) -> None:
    report = evaluate_stage(
        _passing_evidence(),
        stage="development",
        sampling=_sampling(),
        structural_census={},
        source_artifacts={},
    )
    result = tmp_path / "result.json"
    markdown = tmp_path / "report.md"
    evidence = tmp_path / "cases.jsonl.gz"
    write_stage_report(
        report,
        _passing_evidence(),
        result_path=result,
        report_path=markdown,
        evidence_path=evidence,
    )
    first = (result.read_bytes(), markdown.read_bytes(), evidence.read_bytes())
    write_stage_report(
        report,
        _passing_evidence(),
        result_path=result,
        report_path=markdown,
        evidence_path=evidence,
    )
    assert first == (result.read_bytes(), markdown.read_bytes(), evidence.read_bytes())
    with gzip.open(evidence, "rt", encoding="utf-8") as handle:
        assert len([line for line in handle if line.strip()]) == 600
