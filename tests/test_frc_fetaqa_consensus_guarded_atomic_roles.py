from __future__ import annotations

import json
from pathlib import Path

import research.frc_rag.fetaqa_consensus_guarded_atomic_roles as fetaqa
from research.frc_rag.fetaqa_consensus_guarded_atomic_roles import (
    build_candidate_coverage,
    build_candidate_units,
    build_gold_rows,
    evaluate_fetaqa,
    highlighted_candidate_ids,
    prepare_blind_cases,
    select_sample,
    validate_protocol,
    write_report,
)
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    CONSENSUS_GUARDED_V46,
    consensus_guarded_target_cardinality,
)


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _row(index: int, highlighted: list[list[int]]) -> dict:
    return {
        "feta_id": f"synthetic-{index}",
        "table_page_title": "Synthetic page",
        "table_section_title": "Synthetic section",
        "table_array": [
            ["Year", "Revenue", "Cost"],
            ["2023", str(100 + index), str(50 + index)],
            ["2024", str(110 + index), str(55 + index)],
        ],
        "highlighted_cell_ids": highlighted,
        "question": f"Which synthetic cells support question {index}?",
        "answer": "forbidden answer",
    }


def _scored_row(case: dict, favored: set[str]) -> dict:
    scores = []
    for index, candidate in enumerate(case["candidates"]):
        value = 1.0 if candidate["id"] in favored else 0.1 - index * 0.001
        scores.append(
            {
                "id": candidate["id"],
                "scores": {
                    "bm25": value,
                    "dense": value,
                    "hybrid": value,
                    "cross_encoder": value,
                },
                "static_role_scores": {role: value for role in STATIC_ROLES},
                "dynamic_role_scores": {
                    role: value - role_index * 0.0001
                    for role_index, role in enumerate(DYNAMIC_ROLES)
                },
            }
        )
    return {
        "id": case["id"],
        "candidates": case["candidates"],
        "candidate_scores": scores,
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_replays_v46_without_fetaqa_access() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_protocol_v47.json"
    )
    assert protocol["methods"]["candidate_method"] == CONSENSUS_GUARDED_V46
    assert protocol["development_boundary"]["candidate_formula_changed_after_tatqa_access"] is False
    assert protocol["pre_registration_access_disclosure"]["sample_content_seen"] is False


def test_schema_adapter_preserves_exact_highlight_coordinates() -> None:
    row = _row(1, [[1, 1], [2, 2]])
    units = build_candidate_units(row, Tokenizer())
    identifiers = {item["canonical_id"] for item in units}
    assert highlighted_candidate_ids(row) == ("cell_1_1", "cell_2_2")
    assert {"cell_1_1", "cell_2_2"} <= identifiers
    assert all("Synthetic page" in item["text"] for item in units)


def test_sealed_sampling_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(fetaqa, "TARGET_CASES", 3)
    monkeypatch.setattr(fetaqa, "MINIMUM_CASES", 3)
    rows = [
        _row(0, [[1, 1]]),
        _row(1, [[1, 1], [1, 2]]),
        _row(2, [[1, 1], [2, 1], [2, 2]]),
        _row(3, [[2, 2]]),
    ]
    first, first_summary = select_sample(rows)
    second, second_summary = select_sample(reversed(rows))
    assert [row["feta_id"] for row in first] == [row["feta_id"] for row in second]
    assert first_summary == second_summary
    assert first_summary["selected_rows"] == 3


def test_blind_cache_is_gold_free_and_gold_join_is_late(monkeypatch) -> None:
    monkeypatch.setattr(fetaqa, "TARGET_CASES", 4)
    monkeypatch.setattr(fetaqa, "MINIMUM_CASES", 4)
    rows = [
        _row(0, [[1, 1]]),
        _row(1, [[1, 1], [1, 2]]),
        _row(2, [[1, 1], [2, 1], [2, 2]]),
        _row(3, [[1, 2], [2, 2]]),
    ]
    prepared, candidate_maps, summary = prepare_blind_cases(rows, Tokenizer())
    assert len(prepared) == 4
    serialized = json.dumps(prepared, ensure_ascii=False)
    assert "highlighted_cell_ids" not in serialized
    assert "forbidden answer" not in serialized
    assert all("feta_id" not in item for item in candidate_maps)
    gold_rows = build_gold_rows(
        rows,
        candidate_maps,
        pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"],
    )
    assert [row["gold_count"] for row in gold_rows] == [
        len(row["gold_candidate_ids"]) for row in gold_rows
    ]
    coverage = build_candidate_coverage(
        gold_rows,
        summary["sampling"],
        summary["candidate_pool_quartile_boundaries"],
    )
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["minimum_cases_and_ceiling_checks_passed"] is True


def test_frozen_selector_and_locked_evaluation_are_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fetaqa, "TARGET_CASES", 4)
    monkeypatch.setattr(fetaqa, "MINIMUM_CASES", 4)
    monkeypatch.setattr(fetaqa, "MINIMUM_STRATUM_CASES", 1)
    rows = [
        _row(0, [[1, 1]]),
        _row(1, [[1, 1], [1, 2]]),
        _row(2, [[1, 1], [2, 1], [2, 2]]),
        _row(3, [[1, 2], [2, 2]]),
    ]
    prepared, candidate_maps, summary = prepare_blind_cases(rows, Tokenizer())
    gold_rows = build_gold_rows(
        rows,
        candidate_maps,
        pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"],
    )
    favored_by_case = {
        item["id"]: set(gold["gold_candidate_ids"])
        for item, gold in zip(candidate_maps, gold_rows, strict=True)
    }
    scored = [
        _scored_row(case, favored_by_case[case["id"]]) for case in prepared
    ]
    report, evidence = evaluate_fetaqa(
        gold_rows,
        scored,
        query_summary={"fallback_rate": 0.0},
        source_artifacts={"synthetic": True},
        resamples=50,
    )
    assert report["analysis"]["outcome"]["selector_adoption_authorized"] is False
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert len(evidence) == 4

    candidate_scores = scored[0]["candidate_scores"]
    candidates = []
    for original, score in zip(prepared[0]["candidates"], candidate_scores, strict=True):
        candidates.append(
            {
                "id": original["id"],
                "source_id": original["source_id"],
                "token_count": original["token_count"],
                "scores": score["scores"],
                "static_role_scores": score["static_role_scores"],
                "dynamic_role_scores": score["dynamic_role_scores"],
            }
        )
    assert consensus_guarded_target_cardinality(candidates) >= 1

    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "report.md"
    evidence_path = tmp_path / "cases.jsonl.gz"
    write_report(
        report,
        evidence,
        json_path=json_path,
        markdown_path=markdown_path,
        evidence_path=evidence_path,
    )
    first = (json_path.read_bytes(), markdown_path.read_bytes(), evidence_path.read_bytes())
    write_report(
        report,
        evidence,
        json_path=json_path,
        markdown_path=markdown_path,
        evidence_path=evidence_path,
    )
    second = (json_path.read_bytes(), markdown_path.read_bytes(), evidence_path.read_bytes())
    assert first == second
