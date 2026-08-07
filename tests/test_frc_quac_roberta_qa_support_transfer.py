from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.frc_rag.quac_anchor_safe_consensus_slot as v57
import research.frc_rag.quac_roberta_qa_support_transfer as v62
import scripts.run_quac_roberta_qa_support_transfer as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/quac_roberta_qa_support_transfer_protocol_v62.json"
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
    return [
        (row["answer_state"], row["dialogue_id"], row["turn_id"])
        for row in rows
    ]


def test_protocol_freezes_public_model_and_independent_boundary() -> None:
    value = v62.validate_protocol(PROTOCOL)
    assert v62.PROTOCOL_SHA256 == v62._sha256(PROTOCOL)
    assert value["public_support_model"]["revision"] == v62.MODEL_REVISION
    assert value["prior_boundary"]["v57_quac_validation_content_opened_or_parsed"] is False
    assert value["scope"]["expected_v57_exclusion_commitments"] == 600


def test_disjoint_sample_excludes_v57_commitments_and_is_order_invariant() -> None:
    source = _source(12)
    rows, _ = v57.extract_cases(source)
    excluded = {
        v62._hash(str(rows[0]["dialogue_id"]), str(rows[0]["turn_id"]))
    }
    selected, summary = v62.select_disjoint_balanced_sample(
        source,
        stage="development",
        excluded_commitments=excluded,
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    reversed_source = copy.deepcopy(source)
    reversed_source["data"][0]["paragraphs"].reverse()
    again, _ = v62.select_disjoint_balanced_sample(
        reversed_source,
        stage="development",
        excluded_commitments=excluded,
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    assert _identity(selected) == _identity(again)
    assert summary["selected_v57_commitment_overlap"] == 0
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 6,
        "no_answer": 6,
    }


def test_blind_preparation_separates_retrieval_and_qa_inputs() -> None:
    source = _source(12)
    rows, _ = v57.extract_cases(source)
    selected = [next(row for row in rows if row["turn_id"] == "q-0-1")]
    prepared, maps, documents, qa_inputs, census = v62.prepare_blind_cases(
        selected,
        source,
        FakeTokenizer(),
        stage="development",
    )
    forbidden = {
        "answer_state",
        "answer_text",
        "answer_start",
        "answer_end",
        "gold_candidate_ids",
    }
    assert prepared[0]["id"].startswith("q62c")
    assert maps[0]["id"] == prepared[0]["id"] == qa_inputs[0]["id"]
    assert set(qa_inputs[0]) == {
        "schema_version",
        "id",
        "question",
        "context",
        "gold_fields_visible_to_verifier",
    }
    assert qa_inputs[0]["question"] == "Who won?"
    assert not (forbidden & set(qa_inputs[0]))
    assert len(prepared[0]["probe_doc_keys"]) == 8
    assert len(documents) == 8
    assert census["qa_gold_fields_exported"] is False


def test_model_native_null_vs_span_rule_supports_only_strict_margin() -> None:
    context = "Alice won."
    feature = {
        "start_logits": [0.0, -1.0, 6.0, -2.0],
        "end_logits": [0.0, -1.0, 5.0, -2.0],
        "offsets": [[0, 0], [0, 0], [0, 5], [6, 10]],
        "sequence_ids": [None, 0, 1, 1],
        "cls_index": 0,
    }
    supported = v62.best_support_decision([feature], context)
    assert supported["decision"] == "SUPPORTED"
    assert supported["score_margin"] == 11.0
    assert supported["span_start"] == 0
    assert supported["span_end"] == 5

    tie = copy.deepcopy(feature)
    tie["start_logits"][0] = 6.0
    tie["end_logits"][0] = 5.0
    unsupported = v62.best_support_decision([tie], context)
    assert unsupported["decision"] == "UNSUPPORTED"
    assert unsupported["invalid_fail_closed_used"] is False


def test_nonfinite_qa_logits_fail_closed() -> None:
    feature = {
        "start_logits": [0.0, math.nan],
        "end_logits": [0.0, 1.0],
        "offsets": [[0, 0], [0, 1]],
        "sequence_ids": [None, 1],
        "cls_index": 0,
    }
    result = v62.best_support_decision([feature], "A")
    assert result["decision"] == "UNSUPPORTED"
    assert result["invalid_fail_closed_used"] is True


def test_support_cache_and_confirmation_boundary_fail_closed(tmp_path: Path) -> None:
    inputs = [{"id": "a"}, {"id": "b"}]
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
    summary = v62.validate_support_cache(inputs, decisions, cache_key="frozen")
    assert summary["invalid_output_rate"] == 0.5

    args = SimpleNamespace(stage="confirmation", result_root=tmp_path)
    with pytest.raises(ValueError, match="before development result"):
        runner._validate_confirmation_open(args)
    result_path = (
        tmp_path / "quac_roberta_qa_support_transfer_development_result_v62.json"
    )
    result_path.write_text(
        '{"analysis":{"outcome":{"support_established":false,'
        '"confirmation_open_authorized":false}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not authorize"):
        runner._validate_confirmation_open(args)
