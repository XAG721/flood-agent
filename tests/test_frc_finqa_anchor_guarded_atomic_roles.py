from __future__ import annotations

import json
from pathlib import Path

from research.frc_rag.finqa_anchor_guarded_atomic_roles import (
    ANCHOR_GUARDED_V45,
    build_candidate_coverage,
    build_gold_rows,
    evaluate_finqa,
    prepare_blind_cases,
    select_sample,
    select_v45_anchor_guarded,
    table_row_to_text,
    validate_protocol,
    write_report,
)
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _source_row(index: int, mode: str) -> dict:
    if mode == "table_only":
        gold = {"table_1": "table evidence"}
    elif mode == "text_only":
        gold = {"text_0": "text evidence"}
    else:
        gold = {"text_0": "text evidence", "table_1": "table evidence"}
    return {
        "id": f"synthetic/{mode}/{index}",
        "pre_text": [f"Synthetic text evidence {index}."],
        "post_text": [f"Synthetic trailing context {index}."],
        "table": [
            ["Synthetic table", "2024", "2025"],
            ["Revenue", str(100 + index), str(110 + index)],
            ["Cost", str(50 + index), str(55 + index)],
        ],
        "qa": {
            "question": f"What synthetic evidence is required for case {index}?",
            "answer": "forbidden answer",
            "program": "subtract(1, 1)",
            "program_re": "forbidden",
            "gold_inds": gold,
            "exe_ans": "0",
        },
    }


def _candidate(
    index: int,
    *,
    cross_encoder: float,
    role_winner: int | None = None,
    token_count: int = 20,
) -> dict:
    dynamic = {
        role: (1.0 if role_winner == role_index else 0.1 + index * 0.001)
        for role_index, role in enumerate(DYNAMIC_ROLES)
    }
    return {
        "id": f"c{index}",
        "source_id": f"s{index}",
        "token_count": token_count,
        "scores": {
            "bm25": 0.1,
            "dense": 0.1,
            "hybrid": 0.1,
            "cross_encoder": cross_encoder,
        },
        "static_role_scores": {role: 0.1 for role in STATIC_ROLES},
        "dynamic_role_scores": dynamic,
    }


def _candidates_with_distinct_role_winners(count: int) -> list[dict]:
    candidates = [_candidate(index, cross_encoder=0.9 - index * 0.1) for index in range(5)]
    for role_index, role in enumerate(DYNAMIC_ROLES):
        winner = min(role_index, count - 1)
        for index, candidate in enumerate(candidates):
            candidate["dynamic_role_scores"][role] = 1.0 if index == winner else 0.0
    return candidates


def _scored_row(case: dict, favored: int) -> dict:
    scores = []
    for index, candidate in enumerate(case["candidates"]):
        value = 1.0 if index == favored else 0.1 - index * 0.001
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
                    role: value - role_index * 0.001
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


def test_protocol_and_path_erratum_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    value = validate_protocol(
        root
        / "docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_protocol_v45.json",
        root
        / "docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_protocol_erratum_v45.json",
    )
    assert value["protocol"]["methods"]["candidate_method"] == ANCHOR_GUARDED_V45
    assert value["erratum"]["correction"]["effective_registered_source_code"] == (
        "code/utils/general_utils.py"
    )
    assert value["erratum"]["method_or_threshold_changed"] is False


def test_corrected_official_table_row_conversion() -> None:
    assert table_row_to_text(
        ["Report", "2024", "2025"], ["Revenue", "100", "110"]
    ) == "Report the Revenue of 2024 is 100 ; the Revenue of 2025 is 110 ;"


def test_anchor_is_retained_and_target_cardinality_is_guarded() -> None:
    candidates = [
        _candidate(0, cross_encoder=0.2, role_winner=0),
        _candidate(1, cross_encoder=0.3, role_winner=1),
        _candidate(2, cross_encoder=0.4, role_winner=2),
        _candidate(3, cross_encoder=0.5, role_winner=3),
        _candidate(4, cross_encoder=0.9),
    ]
    selected = select_v45_anchor_guarded(candidates, token_budget=512)
    assert len(selected) == 5
    assert selected[0]["id"] == "c4"
    assert len({row["id"] for row in selected}) == 5


def test_all_registered_target_budget_and_source_invariants() -> None:
    expected = {1: 3, 2: 3, 3: 4, 4: 5}
    for distinct, target in expected.items():
        candidates = _candidates_with_distinct_role_winners(distinct)
        selected = select_v45_anchor_guarded(candidates, token_budget=512)
        assert len(selected) == target
        assert selected[0]["id"] == "c0"
        assert len({row["id"] for row in selected}) == target
    assert select_v45_anchor_guarded([], token_budget=512) == []

    candidates = _candidates_with_distinct_role_winners(4)
    for index, candidate in enumerate(candidates):
        candidate["token_count"] = 60
        candidate["source_kind"] = "table" if index % 2 else "text"
    narrow = select_v45_anchor_guarded(candidates, token_budget=100)
    wide = select_v45_anchor_guarded(candidates, token_budget=512)
    assert len(narrow) == 1
    assert len(wide) == 5
    toggled = [dict(candidate, source_kind="text") for candidate in candidates]
    assert [row["id"] for row in wide] == [
        row["id"]
        for row in select_v45_anchor_guarded(toggled, token_budget=512)
    ]


def test_anchor_uses_highest_feasible_candidate_and_canonical_tie_break() -> None:
    candidates = [
        _candidate(0, cross_encoder=0.9, token_count=600),
        _candidate(2, cross_encoder=0.8),
        _candidate(1, cross_encoder=0.8),
        _candidate(3, cross_encoder=0.1),
    ]
    selected = select_v45_anchor_guarded(candidates, token_budget=100)
    assert selected[0]["id"] == "c1"
    assert sum(row["token_count"] for row in selected) <= 100


def test_sealed_sampling_is_balanced_deterministic_and_excludes_readme_ids() -> None:
    rows = [
        _source_row(index, mode)
        for mode in ("table_only", "text_only", "hybrid")
        for index in range(125)
    ]
    exposed = _source_row(999, "table_only")
    exposed["id"] = "ETR/2016/page_23.pdf-2"
    rows.append(exposed)
    first, first_summary = select_sample(rows)
    second, second_summary = select_sample(reversed(rows))
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert first_summary == second_summary
    assert first_summary["selected_by_source_mode"] == {
        "table_only": 120,
        "text_only": 120,
        "hybrid": 120,
    }
    assert first_summary["excluded"]["readme_example_id"] == 1


def test_blind_preparation_excludes_gold_and_gold_join_is_late() -> None:
    rows = [
        _source_row(index, mode)
        for mode in ("table_only", "text_only", "hybrid")
        for index in range(120)
    ]
    prepared, candidate_maps, summary = prepare_blind_cases(rows, Tokenizer())
    assert len(prepared) == 360
    assert summary["ready_for_query_generation"] is True
    serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in ("gold_inds", "program", "answer", "source_mode", "exe_ans"):
        assert forbidden not in serialized
    candidate_map_serialized = json.dumps(candidate_maps, ensure_ascii=False)
    assert "synthetic/table_only/0" not in candidate_map_serialized
    assert all("record_id" not in row for row in candidate_maps)
    assert all("record_id_sha256" in row for row in candidate_maps)
    assert prepared[0]["candidates"][0]["id"].startswith(("text_", "table_"))
    gold_rows = build_gold_rows(rows, candidate_maps)
    assert len(gold_rows) == 360
    assert {row["source_mode"] for row in gold_rows} == {
        "table_only",
        "text_only",
        "hybrid",
    }
    coverage = build_candidate_coverage(gold_rows, summary["sampling"])
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["minimum_cases_and_ceiling_checks_passed"] is True


def test_locked_evaluation_and_outputs_are_deterministic(tmp_path: Path) -> None:
    rows = [
        _source_row(index, mode)
        for mode in ("table_only", "text_only", "hybrid")
        for index in range(120)
    ]
    prepared, candidate_maps, _ = prepare_blind_cases(rows, Tokenizer())
    gold_rows = build_gold_rows(rows, candidate_maps)
    canonical_by_case = {
        row["id"]: {
            unit["canonical_id"]: index for index, unit in enumerate(row["units"])
        }
        for row in candidate_maps
    }
    scored = [
        _scored_row(
            case,
            canonical_by_case[case["id"]][
                "table_1" if gold["source_mode"] != "text_only" else "text_0"
            ],
        )
        for case, gold in zip(prepared, gold_rows, strict=True)
    ]
    query_summary = {
        "rows": 360,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
    }
    report, evidence = evaluate_finqa(
        gold_rows, scored, query_summary, resamples=20
    )
    assert report["metadata"]["official_leaderboard_result"] is False
    assert report["analysis"]["anchor_retention_rate"] == 1.0
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        write_report(
            report,
            evidence,
            json_path=directory / "report.json",
            markdown_path=directory / "report.md",
            evidence_path=directory / "cases.jsonl.gz",
        )
    for name in ("report.json", "report.md", "cases.jsonl.gz"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
