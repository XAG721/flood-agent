from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research.frc_rag.feverous_adaptive_atomic_roles import (
    ADAPTIVE_ARGMAX,
    CHALLENGES,
    DYNAMIC_RANK,
    MAXIMUM_POOL_SIZE,
    OFFICIAL_BASELINE,
    adaptive_target_cardinality,
    build_gold_rows,
    evaluate_feverous,
    extract_page_units,
    prepare_blind_cases,
    select_sample,
    select_v43,
    validate_protocol,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    def decode(self, values: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{value}" for value in values)


def _source_rows() -> tuple[list[dict], list[dict]]:
    development: list[dict] = [{"id": "", "header": "FEVEROUS"}]
    baseline: list[dict] = [{"id": "", "header": "FEVEROUS"}]
    for index, challenge in enumerate(CHALLENGES, start=1):
        case_id = 1000 + index
        page = f"Synthetic Page {case_id}"
        development.append(
            {
                "id": case_id,
                "claim": f"Claim about synthetic entity {case_id}",
                "label": "SUPPORTS",
                "challenge": challenge,
                "evidence": [
                    {
                        "content": [f"{page}_sentence_0"],
                        "context": {},
                    }
                ],
                "annotator_operations": [],
            }
        )
        predicted = (
            [f"{page}_sentence_0"]
            if index % 2
            else [[page, "sentence", "0"]]
        )
        baseline.append({"id": case_id, "predicted_evidence": predicted})
    return development, baseline


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE wiki (id TEXT PRIMARY KEY, data TEXT)")
    for index in range(1, len(CHALLENGES) + 1):
        case_id = 1000 + index
        page = f"Synthetic Page {case_id}"
        value = {
            "title": page,
            "order": ["sentence_0", "sentence_1"],
            "sentence_0": f"Synthetic entity {case_id} has the registered property.",
            "sentence_1": "This is a deterministic distractor sentence.",
        }
        connection.execute(
            "INSERT INTO wiki(id, data) VALUES (?, ?)",
            (page, json.dumps(value)),
        )
    connection.commit()
    connection.close()


def _candidate(candidate_id: str, role_scores: dict[str, float], rank: int | None):
    return {
        "id": candidate_id,
        "source_id": candidate_id,
        "token_count": 10,
        "official_rank": rank,
        "scores": {
            "bm25": 1.0 if candidate_id == "u0000" else 0.2,
            "dense": 1.0 if candidate_id == "u0000" else 0.2,
            "hybrid": 1.0 if candidate_id == "u0000" else 0.2,
            "cross_encoder": 1.0 if candidate_id == "u0000" else 0.2,
        },
        "static_role_scores": {
            "claim_support": 0.8,
            "claim_refutation": 0.7,
            "entity_bridge": 0.6,
            "cross_document_chain": 0.2,
        },
        "dynamic_role_scores": role_scores,
    }


def test_protocol_is_hash_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    value = validate_protocol(
        root
        / "docs/progressive_upgrade/feverous_adaptive_atomic_roles_protocol_v43.json",
        erratum_path=(
            root
            / "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_protocol_erratum_v43.json"
        ),
    )
    assert value["base_protocol"]["experiment_id"] == (
        "FRC-FEVEROUS-ADAPTIVE-ATOMIC-ROLES-V43"
    )
    assert value["base_protocol"]["decision_rules"]["gate_2"].startswith(
        "Always remains"
    )
    assert value["effective_erratum"]["correction"]["effective"] == 189
    assert MAXIMUM_POOL_SIZE == 189


def test_page_unit_extraction_preserves_official_element_ids() -> None:
    page = {
        "title": "Example",
        "order": ["section_0", "sentence_0", "table_0", "list_0"],
        "section_0": {"value": "Facts", "level": 1},
        "sentence_0": "A [[Target|linked value]] is present.",
        "table_0": {
            "type": "normal",
            "caption": "Registered table",
            "table": [
                [
                    {
                        "id": "header_cell_0_0_0",
                        "value": "Key",
                        "is_header": True,
                        "row_span": 1,
                        "column_span": 1,
                    },
                    {
                        "id": "cell_0_0_1",
                        "value": "Value",
                        "is_header": False,
                        "row_span": 1,
                        "column_span": 1,
                    },
                ]
            ],
        },
        "list_0": {
            "type": "unordered_list",
            "list": [{"id": "item_0_0", "value": "Item", "level": 0}],
        },
    }
    units = extract_page_units("Example Page", page)
    identifiers = {unit["evidence_id"] for unit in units}
    assert identifiers == {
        "Example Page_sentence_0",
        "Example Page_table_caption_0",
        "Example Page_header_cell_0_0_0",
        "Example Page_cell_0_0_1",
        "Example Page_item_0_0",
    }
    assert any("linked value" in unit["text"] for unit in units)


def test_sampling_and_preparation_are_deterministic_and_blind() -> None:
    development, baseline = _source_rows()
    first, first_census = select_sample(
        development,
        baseline,
        cases_per_challenge=1,
        minimum_total_cases=6,
        minimum_challenge_cases=1,
    )
    second, second_census = select_sample(
        development,
        baseline,
        cases_per_challenge=1,
        minimum_total_cases=6,
        minimum_challenge_cases=1,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert first_census == second_census

    database = (
        Path(__file__).resolve().parents[1]
        / "output/test_artifacts/feverous-v43-synthetic.db"
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    try:
        _database(database)
        prepared, candidate_maps, census = prepare_blind_cases(
            development,
            baseline,
            database,
            _Tokenizer(),
            cases_per_challenge=1,
            minimum_total_cases=6,
            minimum_challenge_cases=1,
        )
    finally:
        database.unlink(missing_ok=True)
    assert len(prepared) == len(candidate_maps) == 6
    assert census["valid_rows"] == 6
    serialized = json.dumps(prepared, sort_keys=True).lower()
    for forbidden in (
        '"challenge"',
        '"evidence"',
        '"gold"',
        '"label"',
        '"supporting_facts"',
    ):
        assert forbidden not in serialized
    assert all(case["gold_fields_visible_to_scorer"] is False for case in prepared)
    assert all(
        any(candidate["official_rank"] == 0 for candidate in case["candidates"])
        for case in prepared
    )
    assert all(len(case["candidates"]) <= MAXIMUM_POOL_SIZE for case in prepared)


def test_adaptive_cardinality_and_selection_use_distinct_role_argmax() -> None:
    candidates = [
        _candidate(
            "u0000",
            {
                "anchor": 1.0,
                "first_fact": 0.9,
                "second_fact_or_bridge": 0.1,
                "counterevidence": 0.1,
            },
            0,
        ),
        _candidate(
            "u0001",
            {
                "anchor": 0.2,
                "first_fact": 0.1,
                "second_fact_or_bridge": 1.0,
                "counterevidence": 0.9,
            },
            1,
        ),
        _candidate(
            "u0002",
            {
                "anchor": 0.1,
                "first_fact": 0.2,
                "second_fact_or_bridge": 0.2,
                "counterevidence": 0.2,
            },
            None,
        ),
    ]
    assert adaptive_target_cardinality(candidates) == 2
    selected = select_v43(candidates, ADAPTIVE_ARGMAX, token_budget=512)
    assert len(selected) == 2
    assert {row["id"] for row in selected} == {"u0000", "u0001"}
    official = select_v43(candidates, OFFICIAL_BASELINE, token_budget=512)
    assert [row["id"] for row in official] == ["u0000", "u0001"]


def test_gold_join_and_report_are_explicitly_bounded() -> None:
    development, _ = _source_rows()
    candidate_maps = []
    scored_rows = []
    for row in development[1:]:
        case_id = f"feverous-v43::{row['id']}"
        page = f"Synthetic Page {row['id']}"
        candidate_maps.append(
            {
                "id": case_id,
                "dataset_numeric_id": row["id"],
                "units": [
                    {
                        "candidate_id": "u0000",
                        "evidence_id": f"{page}_sentence_0",
                    }
                ],
            }
        )
        merged = [
            _candidate(
                "u0000",
                {
                    "anchor": 1.0,
                    "first_fact": 1.0,
                    "second_fact_or_bridge": 1.0,
                    "counterevidence": 1.0,
                },
                0,
            ),
            _candidate(
                "u0001",
                {
                    "anchor": 0.1,
                    "first_fact": 0.1,
                    "second_fact_or_bridge": 0.1,
                    "counterevidence": 0.1,
                },
                None,
            ),
        ]
        scored_rows.append(
            {
                "id": case_id,
                "candidates": [
                    {
                        "id": item["id"],
                        "source_id": item["source_id"],
                        "token_count": item["token_count"],
                        "official_rank": item["official_rank"],
                    }
                    for item in merged
                ],
                "candidate_scores": [
                    {
                        "id": item["id"],
                        "scores": item["scores"],
                        "static_role_scores": item["static_role_scores"],
                        "dynamic_role_scores": item["dynamic_role_scores"],
                    }
                    for item in merged
                ],
                "gold_fields_visible_to_scorer": False,
            }
        )
    gold = build_gold_rows(development, candidate_maps)
    report, evidence = evaluate_feverous(
        gold,
        scored_rows,
        {"fallback_rate": 0.0},
        resamples=100,
    )
    assert len(evidence) == len(CHALLENGES)
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["analysis"]["outcome"]["selector_adoption_authorized"] is False
    assert report["metadata"]["official_leaderboard_result"] is False
    assert (
        report["analysis"]["aggregates"][ADAPTIVE_ARGMAX]["mean_selected_unit_count"]
        < report["analysis"]["aggregates"][DYNAMIC_RANK]["mean_selected_unit_count"]
    )
