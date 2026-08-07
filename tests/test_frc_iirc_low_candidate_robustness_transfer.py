from __future__ import annotations

import hashlib
import json

import research.frc_rag.iirc_cardinality_transfer as v86
import research.frc_rag.iirc_low_candidate_robustness_transfer as v89
import research.frc_rag.iirc_pooled_score_gap_transfer as v88
import research.frc_rag.iirc_score_gap_transfer as v87


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(
        self,
        values: list[str],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(values)


def _commit(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _ordered(values: list[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}{value}".encode()).hexdigest(),
            value,
        ),
    )


def test_low_candidate_case_keeps_all_real_sources_without_padding() -> None:
    article = {
        "pid": "p1",
        "title": "Main",
        "text": "The main article contains evidence.",
        "links": [{"target": "Alpha"}, {"target": "Beta"}],
    }
    question = {
        "qid": "q1",
        "question": "Which sources are relevant?",
        "answer": {"type": "span"},
        "context": [{"passage": "main"}, {"passage": "alpha"}],
    }
    blind, gold, census = v89.prepare_case(
        article,
        question,
        {"alpha": "Alpha source evidence.", "beta": "Beta source evidence."},
        _Tokenizer(),
        stage="development",
    )
    assert len(blind["candidates"]) == 3
    assert census["low_candidate_case"] == 1
    assert gold["candidate_ceiling_complete"] is True
    assert not ({"answer", "answer_type", "context"} & set(blind))


def test_all_available_selection_reports_actual_cardinality() -> None:
    row = {
        "question": "Which sources?",
        "candidates": [
            {
                "id": f"c{index}",
                "source": f"s{index}",
                "scores": {"cross_encoder": 1.0 - index * 0.1},
                "role_scores": {
                    "answer": 1.0,
                    "attribution": 1.0,
                    "condition": 1.0,
                    "exception": 1.0,
                    "procedure": 1.0,
                },
            }
            for index in range(3)
        ],
    }
    result = v89._select_all_available(row)
    assert result["cardinality"] == 3
    assert len(result["selected_sources"]) == 3


def test_v89_partitions_only_use_v88_remaining(monkeypatch: object) -> None:
    ids = [f"q{index:04d}" for index in range(6400)]
    rows = [{"questions": [{"qid": value} for value in ids]}]
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_articles", 1)
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_questions", len(ids))
    order86 = _ordered(ids, v86.PARTITION_SALT)
    parts86 = {
        "all": sorted(ids),
        "development": order86[:400],
        "confirmation": order86[400:1200],
        "remaining": order86[1200:],
    }
    for name, values in parts86.items():
        monkeypatch.setitem(v86.PARTITION_COMMITMENTS, name, _commit(values))
    eligible87 = sorted(parts86["remaining"])
    order87 = _ordered(eligible87, v87.PARTITION_SALT)
    parts87 = {
        "eligible": eligible87,
        "development": order87[:400],
        "confirmation": order87[400:1200],
        "remaining": order87[1200:],
    }
    for name, values in parts87.items():
        monkeypatch.setitem(v87.PARTITION_COMMITMENTS, name, _commit(values))
    eligible88 = sorted(parts87["remaining"])
    order88 = _ordered(eligible88, v88.PARTITION_SALT)
    parts88 = {
        "eligible": eligible88,
        "development": order88[:800],
        "confirmation": order88[800:2000],
        "remaining": order88[2000:],
    }
    for name, values in parts88.items():
        monkeypatch.setitem(v88.PARTITION_COMMITMENTS, name, _commit(values))
    eligible89 = sorted(parts88["remaining"])
    order89 = _ordered(eligible89, v89.PARTITION_SALT)
    expected = {
        "eligible": eligible89,
        "development": order89[:800],
        "confirmation": order89[800:2000],
        "remaining": order89[2000:],
    }
    for name, values in expected.items():
        monkeypatch.setitem(v89.PARTITION_COMMITMENTS, name, _commit(values))
    actual = v89.build_partitions(rows)
    assert actual == expected
    assert not (set(actual["eligible"]) & set(parts88["confirmation"]))


def test_v89_method_registry_is_unique() -> None:
    assert v89.CANDIDATE_METHOD not in v89.CONTROL_METHODS
    assert len(v89.METHODS) == len(set(v89.METHODS))
    assert json.dumps(v89.METHODS)
