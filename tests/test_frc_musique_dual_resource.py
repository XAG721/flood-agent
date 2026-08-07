from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.musique_dual_resource import (
    DUAL_FRC,
    EXECUTION_SHA256,
    METHODS,
    PROTOCOL_SHA256,
    evaluate_scored_cases,
    load_report,
    prepare_row,
    select_candidates,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "docs/progressive_upgrade/musique_dual_resource_protocol.json"
EXECUTION_PATH = REPO_ROOT / "docs/progressive_upgrade/musique_dual_resource_execution.json"


class FakeTokenizer:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result = []
        for word in text.split():
            if word not in self._ids:
                value = len(self._ids) + 1
                self._ids[word] = value
                self._words[value] = word
            result.append(self._ids[word])
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(self._words[value] for value in token_ids)


def _roles(**overrides: float) -> dict[str, float]:
    result = {
        "direct_answer_support": 0.0,
        "entity_and_scope": 0.0,
        "corroborating_evidence": 0.0,
        "contradiction_detection": 0.0,
    }
    result.update(overrides)
    return result


def _candidate(
    candidate_id: str,
    *,
    cost: int,
    cross: float,
    roles: dict[str, float] | None = None,
) -> dict:
    return {
        "id": candidate_id,
        "source_id": candidate_id,
        "token_count": cost,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": roles or _roles(),
    }


def test_prepare_row_is_blind_and_preserves_support_only_in_gold() -> None:
    row = {
        "id": "demo",
        "question": "Which person?",
        "answer": "A",
        "answer_aliases": [],
        "answerable": True,
        "question_decomposition": [
            {"id": 1, "question": "q1", "answer": "a1", "paragraph_support_idx": 0},
            {"id": 2, "question": "q2", "answer": "a2", "paragraph_support_idx": 1},
        ],
        "paragraphs": [
            {
                "idx": 0,
                "title": "A title",
                "paragraph_text": "supporting text",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "B title",
                "paragraph_text": "distractor text",
                "is_supporting": False,
            },
        ],
    }
    prepared, gold = prepare_row(row, FakeTokenizer())

    encoded = json.dumps(prepared, sort_keys=True)
    for forbidden in (
        "answer",
        "answerable",
        "question_decomposition",
        "is_supporting",
        "paragraph_idx",
        "gold_unit_ids",
    ):
        assert f'"{forbidden}"' not in encoded
    assert prepared["gold_fields_visible_to_scorer"] is False
    assert gold["hop_count"] == 2
    assert len(gold["gold_unit_ids"]) == 1
    assert {value["labels"][0] for value in gold["candidate_gold"].values()} == {
        "supporting",
        "non_supporting",
    }


def test_prepare_row_excludes_cross_label_duplicate() -> None:
    row = {
        "id": "conflict",
        "question": "Q",
        "answerable": True,
        "question_decomposition": [{}, {}],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Same",
                "paragraph_text": "text",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Same",
                "paragraph_text": "text",
                "is_supporting": False,
            },
        ],
    }
    with pytest.raises(ValueError, match="cross_label_duplicate_conflict"):
        prepare_row(row, FakeTokenizer())


def test_dual_resource_cost_accounts_for_top_k_slot_scarcity() -> None:
    candidates = [
        _candidate(
            "long",
            cost=10,
            cross=1.0,
            roles=_roles(
                direct_answer_support=1.0,
                entity_and_scope=1.0,
                corroborating_evidence=1.0,
                contradiction_detection=1.0,
            ),
        ),
        _candidate(
            "cheap-a",
            cost=2,
            cross=0.4,
            roles=_roles(direct_answer_support=1.0),
        ),
        _candidate(
            "cheap-b",
            cost=2,
            cross=0.4,
            roles=_roles(entity_and_scope=1.0),
        ),
        _candidate(
            "cheap-c",
            cost=2,
            cross=0.4,
            roles=_roles(corroborating_evidence=1.0),
        ),
        _candidate(
            "cheap-d",
            cost=2,
            cross=0.4,
            roles=_roles(contradiction_detection=1.0),
        ),
    ]

    v35 = select_candidates(
        candidates, "frc_cost_aware_v35", top_k=5, token_budget=10
    )
    dual = select_candidates(candidates, DUAL_FRC, top_k=5, token_budget=10)

    assert {item["id"] for item in v35} == {
        "cheap-a",
        "cheap-b",
        "cheap-c",
        "cheap-d",
    }
    assert [item["id"] for item in dual] == ["long"]


def test_all_ten_selectors_respect_dual_hard_constraints() -> None:
    candidates = [
        _candidate("a", cost=6, cross=0.8),
        _candidate("b", cost=5, cross=0.7),
        _candidate("c", cost=5, cross=0.7),
    ]
    for method in METHODS:
        selected = select_candidates(candidates, method, top_k=2, token_budget=10)
        assert len(selected) <= 2
        assert sum(item["token_count"] for item in selected) <= 10


def _scored_case(hop: int) -> tuple[dict, dict]:
    case_id = f"musique::synthetic-{hop}"
    candidates = [
        _candidate(
            f"{case_id}::s0000::c000",
            cost=10,
            cross=0.9,
            roles=_roles(direct_answer_support=0.9),
        ),
        _candidate(
            f"{case_id}::s0001::c000",
            cost=10,
            cross=0.1,
            roles=_roles(corroborating_evidence=0.9),
        ),
    ]
    scored = {
        "id": case_id,
        "dataset_id": "musique_ans_v1.0_dev",
        "capability": "synthetic",
        "gold_fields_visible_to_scorer": False,
        "candidates": [
            {
                "id": item["id"],
                "source_id": item["source_id"],
                "token_count": item["token_count"],
            }
            for item in candidates
        ],
        "candidate_scores": [
            {
                "id": item["id"],
                "scores": item["scores"],
                "role_scores": item["role_scores"],
            }
            for item in candidates
        ],
    }
    gold = {
        "case_id": case_id,
        "hop_count": hop,
        "gold_unit_ids": ["g0"],
        "candidate_gold": {
            candidates[0]["id"]: {
                "labels": ["supporting"],
                "gold_unit_ids": ["g0"],
            },
            candidates[1]["id"]: {
                "labels": ["non_supporting"],
                "gold_unit_ids": [],
            },
        },
    }
    return scored, gold


def _synthetic_report() -> tuple[dict, list[dict]]:
    pairs = [_scored_case(hop) for hop in (2, 3, 4)]
    return evaluate_scored_cases(
        [gold for _, gold in pairs],
        [scored for scored, _ in pairs],
        resamples=32,
    )


def test_evaluation_is_gold_late_private_and_deterministic(tmp_path: Path) -> None:
    report, evidence = _synthetic_report()
    assert report["metadata"]["cases"] == 3
    assert report["metadata"]["selection_runs"] == 3 * len(METHODS) * 3
    encoded = json.dumps(evidence, sort_keys=True)
    for forbidden in ("question", "answer", "text", "is_supporting"):
        assert f'"{forbidden}"' not in encoded

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    first = write_report(
        copy.deepcopy(report), evidence, tmp_path / "first", source_paths={"x": source}
    )
    second = write_report(
        copy.deepcopy(report), evidence, tmp_path / "second", source_paths={"x": source}
    )
    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["markdown"].read_bytes() == second["markdown"].read_bytes()
    assert first["evidence"].read_bytes() == second["evidence"].read_bytes()
    loaded = load_report(first["json"], first["evidence"])
    assert loaded["metadata"]["evidence_artifact"][
        "raw_question_answer_or_candidate_text_exported"
    ] is False


def test_registration_hashes_are_stable() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256(EXECUTION_PATH.read_bytes()).hexdigest() == EXECUTION_SHA256
