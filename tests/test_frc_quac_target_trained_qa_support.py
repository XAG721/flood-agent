from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

import research.frc_rag.quac_anchor_safe_consensus_slot as v57
import research.frc_rag.quac_target_trained_qa_support as v63
import scripts.run_quac_target_trained_qa_support as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/quac_target_trained_qa_support_protocol_v63.json"
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _turn(
    identifier: str,
    question: str,
    answer: str,
    context: str,
) -> dict[str, object]:
    start = context.find(answer) if answer != "CANNOTANSWER" else 0
    return {
        "id": identifier,
        "question": question,
        "orig_answer": {"text": answer, "answer_start": start},
    }


def _source(document_count: int = 12) -> dict[str, object]:
    paragraphs = []
    for index in range(document_count):
        context = (
            f"Document {index}. Paris is the prior answer. "
            f"Alice won event {index}. More context. CANNOTANSWER"
        )
        paragraphs.append(
            {
                "id": f"dialogue-{index}",
                "context": context,
                "qas": [
                    _turn(f"q-{index}-0", "Where was it?", "Paris", context),
                    _turn(f"q-{index}-1", "Who won?", "Alice", context),
                    _turn(
                        f"q-{index}-2",
                        "What score is absent?",
                        "CANNOTANSWER",
                        context,
                    ),
                ],
            }
        )
    return {
        "data": [
            {
                "title": "Synthetic title",
                "section_title": "Synthetic section",
                "paragraphs": paragraphs,
            }
        ]
    }


def _identity(rows: list[dict[str, object]]) -> list[tuple[object, object, object]]:
    return [(row["answer_state"], row["dialogue_id"], row["turn_id"]) for row in rows]


def test_protocol_freezes_target_trained_model_and_limited_claim() -> None:
    value = v63.validate_protocol(PROTOCOL)
    assert v63.PROTOCOL_SHA256 == v63._sha256(PROTOCOL)
    assert value["public_support_model"]["revision"] == v63.MODEL_REVISION
    assert (
        value["public_support_model"][
            "exact_quac_training_split_and_checkpoint_selection_disclosed"
        ]
        is False
    )
    assert (
        value["stopping_and_claim_boundary"]["strict_independent_confirmation_claimed"]
        is False
    )


def test_validation_sample_is_balanced_order_invariant_and_capped() -> None:
    source = _source(12)
    selected, summary = v63.select_balanced_validation_sample(
        source,
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    reversed_source = copy.deepcopy(source)
    reversed_source["data"][0]["paragraphs"].reverse()
    again, _ = v63.select_balanced_validation_sample(
        reversed_source,
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    assert _identity(selected) == _identity(again)
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 6,
        "no_answer": 6,
    }
    assert summary["maximum_cases_per_dialogue"] <= 2


def test_blind_preparation_exports_no_gold_to_qa_input() -> None:
    source = _source(12)
    rows, _ = v57.extract_cases(source)
    selected = [next(row for row in rows if row["turn_id"] == "q-0-1")]
    prepared, maps, documents, qa_inputs, census = v63.prepare_blind_cases(
        selected,
        source,
        FakeTokenizer(),
    )
    forbidden = {
        "answer_state",
        "answer_text",
        "answer_start",
        "answer_end",
        "gold_candidate_ids",
    }
    assert prepared[0]["id"].startswith("q63c")
    assert maps[0]["id"] == prepared[0]["id"] == qa_inputs[0]["id"]
    assert qa_inputs[0]["question"] == "Who won?"
    assert not (forbidden & set(qa_inputs[0]))
    assert len(documents) == 8
    assert census["qa_gold_fields_exported"] is False


def _support_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(300):
        rows.append(
            {
                "case_id": f"answer-{index}",
                "answer_state": "answer_bearing",
                "support_passed": index < 240,
                "score_margin": float(index),
                "invalid_fail_closed_used": False,
                "dialogue_cluster": f"d-{index}",
                "document_cluster": f"p-{index % 250}",
            }
        )
        rows.append(
            {
                "case_id": f"no-answer-{index}",
                "answer_state": "no_answer",
                "support_passed": index >= 210,
                "score_margin": float(-index),
                "invalid_fail_closed_used": False,
                "dialogue_cluster": f"n-{index}",
                "document_cluster": f"q-{index % 250}",
            }
        )
    return rows


def test_support_open_gate_requires_all_pre_registered_checks() -> None:
    sampling = {
        "schema_exclusion_rate": 0.0,
        "selected_dialogues": 300,
        "selected_documents": 250,
    }
    verifier = {
        "rows": 600,
        "invalid_output_count": 0,
        "invalid_output_rate": 0.0,
        "gold_fields_visible_to_verifier": False,
    }
    report, evidence = v63.evaluate_support_open_gate(
        _support_rows(), verifier, sampling, {"frozen": True}
    )
    outcome = report["analysis"]["outcome"]
    assert outcome["retrieval_scoring_open_authorized"] is True
    assert report["analysis"]["support_verifier"]["balanced_accuracy"] == 0.75
    assert len(evidence) == 600

    failing = _support_rows()
    for row in failing:
        if row["answer_state"] == "answer_bearing":
            row["support_passed"] = False
    closed, _ = v63.evaluate_support_open_gate(
        failing, verifier, sampling, {"frozen": True}
    )
    assert closed["analysis"]["outcome"]["retrieval_scoring_open_authorized"] is False
    assert closed["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"


def test_support_cache_fails_closed_on_invalid_output() -> None:
    inputs = [{"id": "a"}, {"id": "b"}]
    decisions = [
        {"id": "a", "cache_key": "fixed", "invalid_fail_closed_used": False},
        {"id": "b", "cache_key": "fixed", "invalid_fail_closed_used": True},
    ]
    summary = v63.validate_support_cache(inputs, decisions, cache_key="fixed")
    assert summary["invalid_output_rate"] == 0.5


def test_method_names_distinguish_target_trained_gate() -> None:
    assert (
        v63._v63_method_name("roberta_supported_cross_encoder_topk_v62")
        == "scibert_quac_supported_cross_encoder_topk_v63"
    )
    assert v63._v63_method_name("dense_topk") == "dense_topk"


def test_retrieval_authorization_requires_hash_locked_support_result(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "support-result.json"
    open_path = tmp_path / "support-open.json"
    paths = {"support_result": result_path, "support_open": open_path}
    with pytest.raises(ValueError, match="not authorized"):
        runner._validate_support_open(paths)
    result_path.write_text(
        json.dumps(
            {"analysis": {"outcome": {"retrieval_scoring_open_authorized": True}}}
        ),
        encoding="utf-8",
    )
    open_path.write_text(
        json.dumps({"support_result_sha256": runner.sha256(result_path)}),
        encoding="utf-8",
    )
    assert (
        runner._validate_support_open(paths)["analysis"]["outcome"][
            "retrieval_scoring_open_authorized"
        ]
        is True
    )


def test_report_writer_emits_deterministic_gzip_evidence(tmp_path: Path) -> None:
    sampling = {
        "schema_exclusion_rate": 0.0,
        "selected_dialogues": 300,
        "selected_documents": 250,
    }
    verifier = {
        "rows": 600,
        "invalid_output_count": 0,
        "invalid_output_rate": 0.0,
        "gold_fields_visible_to_verifier": False,
    }
    report, evidence = v63.evaluate_support_open_gate(
        _support_rows(), verifier, sampling, {"frozen": True}
    )
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    for name, target in (("first", first), ("second", second)):
        v63.write_report(
            report,
            evidence,
            tmp_path / f"{name}.json",
            tmp_path / f"{name}.md",
            target,
            title="test",
        )
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert json.loads(next(handle))["case_id"]
