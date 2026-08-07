from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from research.frc_rag.hover_verification_roles import (
    ALL_ROLES,
    EXECUTION_SHA256,
    GENERIC_FRC,
    METHODS,
    PROTOCOL_SHA256,
    VERIFICATION_FRC,
    evaluate_scored_cases,
    extract_official_candidates,
    inspect_database,
    load_articles,
    load_report,
    prepare_row,
    privacy_audit,
    select_candidates,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "docs/progressive_upgrade/hover_verification_roles_protocol.json"
)
EXECUTION_PATH = (
    REPO_ROOT / "docs/progressive_upgrade/hover_verification_roles_execution.json"
)


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


def _role_scores(**overrides: float) -> dict[str, float]:
    result = {role: 0.0 for role in ALL_ROLES}
    result.update(overrides)
    return result


def _candidate(
    candidate_id: str,
    *,
    source_id: str,
    cost: int,
    cross: float,
    roles: dict[str, float] | None = None,
) -> dict:
    return {
        "id": candidate_id,
        "source_id": source_id,
        "token_count": cost,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": roles or _role_scores(),
    }


def test_extract_official_candidates_uses_only_frozen_top_twenty() -> None:
    titles = [f"Title {index}" for index in range(25)]
    rows = [
        {
            "id": "one",
            "claim": "A claim",
            "label": "SUPPORTS",
            "evidence": [["gold"]],
            "doc_retrieval_results": [[titles, list(range(25))], 1],
        }
    ]
    result, requested = extract_official_candidates(rows)

    assert result["one"] == {"claim": "A claim", "titles": titles[:20]}
    assert requested == set(titles[:20])
    assert "label" not in result["one"]
    assert "evidence" not in result["one"]


def test_sqlite_article_loader_is_read_only_and_exact(tmp_path: Path) -> None:
    database = tmp_path / "wiki.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE documents (id PRIMARY KEY, text)")
        connection.executemany(
            "INSERT INTO documents VALUES (?, ?)",
            [("A", "alpha text"), ("B", "beta text"), ("C", "")],
        )
    schema = inspect_database(database)
    articles = load_articles(database, {"A", "B", "missing"}, schema=schema)

    assert schema["quick_check"] == "ok"
    assert articles == {"A": "alpha text", "B": "beta text"}


def test_prepare_row_is_blind_and_keeps_gold_late() -> None:
    row = {
        "uid": "demo",
        "claim": "A multi-document claim",
        "supporting_facts": [["A", 0], ["B", 1]],
        "label": "SUPPORTED",
        "num_hops": 2,
    }
    retrieval = {"claim": row["claim"], "titles": ["B", "C", "A"]}
    prepared, gold = prepare_row(
        row,
        retrieval,
        {"A": "alpha text", "B": "beta text", "C": "other text"},
        FakeTokenizer(),
    )

    encoded = json.dumps(prepared, sort_keys=True)
    for forbidden in (
        "label",
        "num_hops",
        "supporting_facts",
        "gold_unit_ids",
        "candidate_gold",
    ):
        assert f'"{forbidden}"' not in encoded
    assert prepared["gold_fields_visible_to_scorer"] is False
    assert len({candidate["source_id"] for candidate in prepared["candidates"]}) == 3
    assert gold["label"] == "SUPPORTED"
    assert gold["hop_count"] == 2
    assert gold["candidate_ceiling"] == 1.0
    assert len(gold["gold_unit_ids"]) == 2


def test_role_family_is_the_only_frc_selector_difference() -> None:
    candidates = [
        _candidate(
            "generic",
            source_id="s0",
            cost=10,
            cross=0.8,
            roles=_role_scores(direct_answer_support=1.0),
        ),
        _candidate(
            "verification",
            source_id="s1",
            cost=10,
            cross=0.7,
            roles=_role_scores(claim_support=1.0),
        ),
    ]

    generic = select_candidates(
        candidates, GENERIC_FRC, top_k=1, token_budget=10
    )
    verification = select_candidates(
        candidates, VERIFICATION_FRC, top_k=1, token_budget=10
    )

    assert [item["id"] for item in generic] == ["generic"]
    assert [item["id"] for item in verification] == ["verification"]


def test_all_seven_selectors_respect_hard_constraints() -> None:
    candidates = [
        _candidate("a", source_id="s0", cost=6, cross=0.8),
        _candidate("b", source_id="s1", cost=5, cross=0.7),
        _candidate("c", source_id="s2", cost=5, cross=0.6),
    ]
    for method in METHODS:
        selected = select_candidates(candidates, method, top_k=2, token_budget=10)
        assert len(selected) <= 2
        assert sum(item["token_count"] for item in selected) <= 10


def _scored_case(
    case_index: int,
    *,
    label: str,
    hop: int,
    ceiling_complete: bool,
) -> tuple[dict, dict]:
    case_id = f"hover::synthetic-{case_index}"
    candidates = [
        _candidate(
            f"{case_id}::s0000::c000",
            source_id="s0000",
            cost=10,
            cross=0.9,
            roles=_role_scores(
                direct_answer_support=0.9,
                claim_support=0.9,
            ),
        ),
        _candidate(
            f"{case_id}::s0001::c000",
            source_id="s0001",
            cost=10,
            cross=0.1,
            roles=_role_scores(
                corroborating_evidence=0.9,
                cross_document_chain=0.9,
            ),
        ),
    ]
    scored = {
        "id": case_id,
        "dataset_id": "hover_dev_release_v1.1",
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
    gold_units = ["g0"] if ceiling_complete else ["g0", "g1"]
    gold = {
        "case_id": case_id,
        "label": label,
        "hop_count": hop,
        "gold_unit_ids": gold_units,
        "candidate_gold": {
            candidates[0]["id"]: {
                "source_id": "s0000",
                "gold_unit_ids": ["g0"],
            },
            candidates[1]["id"]: {
                "source_id": "s0001",
                "gold_unit_ids": [],
            },
        },
        "source_gold": {"s0000": ["g0"], "s0001": []},
        "candidate_ceiling": 1.0 if ceiling_complete else 0.5,
        "candidate_ceiling_complete": ceiling_complete,
        "resolved_candidate_documents": 2,
    }
    return scored, gold


def _synthetic_report() -> tuple[dict, list[dict]]:
    pairs = [
        _scored_case(0, label="SUPPORTED", hop=2, ceiling_complete=True),
        _scored_case(1, label="NOT_SUPPORTED", hop=3, ceiling_complete=True),
        _scored_case(2, label="SUPPORTED", hop=4, ceiling_complete=False),
    ]
    return evaluate_scored_cases(
        [gold for _, gold in pairs],
        [scored for scored, _ in pairs],
        resamples=32,
    )


def test_evaluation_is_private_deterministic_and_stratified(tmp_path: Path) -> None:
    report, evidence = _synthetic_report()
    assert report["metadata"]["cases"] == 3
    assert report["metadata"]["selection_runs"] == 3 * len(METHODS) * 3
    assert report["aggregates"]["strata"]["4-hop"]["cases"] == 1
    assert privacy_audit(evidence) == {"rows": 3, "leaks": [], "passed": True}

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
    assert load_report(first["json"], first["evidence"])["metadata"]["cases"] == 3


def test_registration_hashes_are_stable() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert hashlib.sha256(EXECUTION_PATH.read_bytes()).hexdigest() == EXECUTION_SHA256
