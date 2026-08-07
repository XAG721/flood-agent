from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.frc_rag.quac_anchor_safe_consensus_slot as v57
import scripts.run_quac_anchor_safe_consensus_slot as runner


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/quac_anchor_safe_consensus_slot_protocol_v57.json"
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
        "answers": [],
        "followup": "m",
        "yesno": "x",
    }


def _source(document_count: int = 12) -> dict[str, object]:
    paragraphs = []
    for index in range(document_count):
        context = (
            f"Document {index} starts here. Paris is the prior answer. "
            f"Alice won event {index}. More context follows. CANNOTANSWER"
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
                        "What was the missing score?",
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
                "background": "unused",
                "paragraphs": paragraphs,
            }
        ]
    }


def _candidate(index: int, *, cost: int = 10) -> dict[str, object]:
    identifier = f"c{index:02d}"
    anchor = 100.0 - index
    raw = {
        "anchor": anchor,
        "first_fact": 20.0 - index,
        "second_fact_or_bridge": 20.0 - index,
        "counterevidence": 20.0 - index,
    }
    if index == 6:
        raw["first_fact"] = 200.0
        raw["second_fact_or_bridge"] = 199.0
    return {
        "id": identifier,
        "source_id": identifier,
        "token_count": cost,
        "scores": {
            "bm25": anchor,
            "dense": anchor,
            "hybrid": anchor,
            "cross_encoder": anchor,
        },
        "dynamic_role_scores": {role: max(0.0, score / 200.0) for role, score in raw.items()},
        "raw_dynamic_role_scores": raw,
    }


def test_protocol_hash_and_registered_formula_are_frozen() -> None:
    value = v57.validate_protocol(PROTOCOL)
    assert v57.PROTOCOL_SHA256 == v57._sha256(PROTOCOL)
    assert value["methods"]["candidate"] == v57.CANDIDATE
    assert value["methods"]["exact_anchor_ablation"] == v57.EXACT_ANCHOR_ABLATION
    assert value["scope"]["confirmation_cases"] == 600


def test_extract_cases_hides_current_answer_and_uses_only_prior_history() -> None:
    rows, census = v57.extract_cases(_source(1))
    current = next(row for row in rows if row["turn_id"] == "q-0-1")
    assert "Paris" in current["query"]
    assert "Alice" not in current["query"]
    assert current["history_depth"] == 1
    no_answer = next(row for row in rows if row["turn_id"] == "q-0-2")
    assert no_answer["answer_state"] == "no_answer"
    assert "CANNOTANSWER" not in no_answer["query"]
    assert census["schema_exclusion_rate"] == 0.0


def test_candidate_units_preserve_offsets_and_exclude_trailing_sentinel() -> None:
    context = "First sentence. " + " ".join(f"w{index}" for index in range(110)) + ". CANNOTANSWER"
    rows = v57.canonical_candidate_rows(context, "doc", FakeTokenizer())
    assert rows
    assert all(context[row["start"] : row["end"]] == row["text"] for row in rows)
    assert all("CANNOTANSWER" not in row["text"] for row in rows)
    assert max(row["token_count"] for row in rows) <= v57.WINDOW_TOKENS


def test_balanced_sample_is_order_invariant_and_respects_caps() -> None:
    source = _source(12)
    selected, summary = v57.select_balanced_sample(
        source,
        stage="development",
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    reversed_source = copy.deepcopy(source)
    reversed_source["data"][0]["paragraphs"].reverse()
    again, _ = v57.select_balanced_sample(
        reversed_source,
        stage="development",
        target_per_group=6,
        maximum_cases_per_dialogue=2,
        maximum_cases_per_document_per_state=1,
    )
    def identity(rows: list[dict[str, object]]) -> list[tuple[object, object, object]]:
        return [
            (row["answer_state"], row["dialogue_id"], row["turn_id"])
            for row in rows
        ]
    assert identity(selected) == identity(again)
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 6,
        "no_answer": 6,
    }
    assert summary["maximum_cases_per_dialogue"] <= 2
    assert summary["maximum_cases_per_document_per_answer_state"] <= 1


def test_blind_preparation_uses_same_quartile_decoys_and_hides_gold() -> None:
    source = _source(12)
    rows, _ = v57.extract_cases(source)
    selected = [next(row for row in rows if row["turn_id"] == "q-0-1")]
    prepared, maps, documents, census = v57.prepare_blind_cases(
        selected, source, FakeTokenizer(), stage="development"
    )
    assert len(prepared[0]["probe_doc_keys"]) == 8
    assert prepared[0]["probe_doc_keys"][0] == prepared[0]["own_doc_key"]
    assert prepared[0]["decoy_fallback_used"] is False
    assert census["decoy_fallback_rate"] == 0.0
    assert len(documents) == 8
    forbidden = {"answer_state", "answer_text", "answer_start", "answer_end", "gold_candidate_ids"}
    assert not (forbidden & set(prepared[0]))
    assert not (forbidden & set(maps[0]))
    assert all(document["gold_fields_visible_to_scorer"] is False for document in documents)


def test_single_slot_selector_changes_only_fifth_anchor_slot() -> None:
    candidates = [_candidate(index) for index in range(1, 13)]
    baseline = v57.select_v57(
        candidates,
        v57.EXACT_ANCHOR_ABLATION,
        gate_passed=True,
        token_budget=100,
    )
    selected = v57.select_v57(
        candidates, v57.CANDIDATE, gate_passed=True, token_budget=100
    )
    assert [row["id"] for row in baseline] == ["c01", "c02", "c03", "c04", "c05"]
    assert [row["id"] for row in selected] == ["c01", "c02", "c03", "c04", "c06"]
    details = v57.single_slot_consensus_details(candidates)
    assert details["winner_id"] == "c06"


def test_selector_reverts_exactly_without_consensus_or_budget() -> None:
    no_consensus = [_candidate(index) for index in range(1, 13)]
    for row in no_consensus:
        index = int(str(row["id"])[1:])
        row["raw_dynamic_role_scores"]["first_fact"] = 20.0 - index
        row["raw_dynamic_role_scores"]["second_fact_or_bridge"] = 20.0 - index
    baseline = v57.select_v57(
        no_consensus,
        v57.EXACT_ANCHOR_ABLATION,
        gate_passed=True,
        token_budget=100,
    )
    selected = v57.select_v57(
        no_consensus, v57.CANDIDATE, gate_passed=True, token_budget=100
    )
    assert selected == baseline

    expensive = [_candidate(index, cost=70 if index == 6 else 10) for index in range(1, 13)]
    expensive_baseline = v57.select_v57(
        expensive,
        v57.EXACT_ANCHOR_ABLATION,
        gate_passed=True,
        token_budget=100,
    )
    expensive_selected = v57.select_v57(
        expensive, v57.CANDIDATE, gate_passed=True, token_budget=100
    )
    assert expensive_selected == expensive_baseline
    assert v57.select_v57(
        expensive, v57.CANDIDATE, gate_passed=False, token_budget=100
    ) == []


def test_gold_interval_join_happens_after_scored_pool_is_complete() -> None:
    source = _source(8)
    rows, _ = v57.extract_cases(source)
    selected = [next(row for row in rows if row["turn_id"] == "q-0-1")]
    prepared, maps, documents, census = v57.prepare_blind_cases(
        selected, source, FakeTokenizer(), stage="development"
    )
    own = next(row for row in documents if row["id"] == prepared[0]["own_doc_key"])
    scored = [{"id": prepared[0]["id"], "candidates": own["candidates"]}]
    gold = v57.build_gold_rows(
        source,
        maps,
        scored,
        document_length_quartile_boundaries=census["document_length_quartile_boundaries"],
        turn_position_quartile_boundaries=census["dialogue_turn_position_quartile_boundaries"],
    )
    assert gold[0]["answer_state"] == "answer_bearing"
    assert gold[0]["gold_candidate_ids"]
    assert gold[0]["candidate_ceiling_complete"] is True


def test_confirmation_boundary_rejects_absent_or_failed_development(tmp_path: Path) -> None:
    args = SimpleNamespace(stage="confirmation", result_root=tmp_path)
    with pytest.raises(ValueError, match="before development result"):
        runner._validate_confirmation_open(args)
    development = tmp_path / "quac_anchor_safe_consensus_slot_development_result_v57.json"
    development.write_text(
        '{"analysis":{"outcome":{"support_established":false,"validation_open_authorized":false}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not authorize"):
        runner._validate_confirmation_open(args)
