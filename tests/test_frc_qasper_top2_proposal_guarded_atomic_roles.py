from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import research.frc_rag.qasper_top2_proposal_guarded_atomic_roles as qasper
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES
from research.frc_rag.qasper_top2_proposal_guarded_atomic_roles import (
    TOP2_PROPOSAL_GUARDED_V48,
    build_candidate_coverage,
    build_candidate_units,
    build_gold_rows,
    evaluate_qasper,
    mapped_answer_references,
    prepare_blind_cases,
    select_sample,
    select_v48,
    top2_proposal_details,
    validate_protocol,
    write_report,
)


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _answer(evidence: list[str], *, kind: str = "extractive") -> dict:
    payload = {
        "unanswerable": False,
        "extractive_spans": ["value"] if kind == "extractive" else [],
        "free_form_answer": "value" if kind == "free_form" else "",
        "yes_no": True if kind == "yes_no" else None,
        "evidence": evidence,
        "highlighted_evidence": evidence,
    }
    return {"annotation_id": "forbidden", "worker_id": "forbidden", "answer": payload}


def _paper(index: int, *, ambiguous: bool = False) -> dict:
    first = f"Alpha evidence {index}."
    second = first if ambiguous else f"Beta evidence {index}."
    return {
        "title": f"Synthetic paper {index}",
        "full_text": [
            {"section_name": "Introduction", "paragraphs": [first, second]},
            {"section_name": "Results", "paragraphs": [f"Gamma evidence {index}."]},
        ],
        "figures_and_tables": [{"caption": f"Accuracy results {index}."}],
        "qas": [
            {
                "question_id": f"question-{index}",
                "question": f"What supports result {index}?",
                "answers": [
                    _answer([first]),
                    _answer([second, f"FLOAT SELECTED: Accuracy results {index}."], kind="free_form"),
                ],
            }
        ],
    }


def _scored_candidate(identifier: str, role_scores: dict[str, float], index: int) -> dict:
    cross = 1.0 - index * 0.01
    return {
        "id": identifier,
        "source_id": identifier,
        "source_kind": "paragraph",
        "token_count": 5,
        "scores": {"bm25": cross, "dense": cross, "hybrid": cross, "cross_encoder": cross},
        "static_role_scores": {role: cross for role in STATIC_ROLES},
        "dynamic_role_scores": role_scores,
    }


def _proposal_candidates(pair_starts: list[int], count: int = 8) -> list[dict]:
    scores = [{role: 0.0 for role in DYNAMIC_ROLES} for _ in range(count)]
    for role_index, role in enumerate(DYNAMIC_ROLES):
        start = pair_starts[role_index]
        scores[start][role] = 10.0
        scores[start + 1][role] = 9.0
        for index in range(count):
            scores[index][role] += (count - index) * 0.001
    return [_scored_candidate(f"c{index}", scores[index], index) for index in range(count)]


def _scored_row(case: dict) -> dict:
    candidate_scores = []
    for index, candidate in enumerate(case["candidates"]):
        value = 1.0 - index * 0.01
        candidate_scores.append(
            {
                "id": candidate["id"],
                "scores": {"bm25": value, "dense": value, "hybrid": value, "cross_encoder": value},
                "static_role_scores": {role: value for role in STATIC_ROLES},
                "dynamic_role_scores": {role: value - role_index * 0.0001 for role_index, role in enumerate(DYNAMIC_ROLES)},
            }
        )
    return {
        "id": case["id"],
        "candidates": case["candidates"],
        "candidate_scores": candidate_scores,
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_freezes_candidate_and_permanently_excludes_train() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(root / "docs/progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_protocol_v48.json")
    assert protocol["methods"]["candidate_method"] == TOP2_PROPOSAL_GUARDED_V48
    assert protocol["development_boundary"]["qasper_train_split_permanently_excluded"] is True
    assert protocol["pre_registration_access_disclosure"]["validation_row_id_question_answer_evidence_paragraph_or_caption_seen"] is False


def test_schema_adapter_maps_alternative_paragraph_and_float_references() -> None:
    paper = _paper(1)
    units = build_candidate_units(paper, Tokenizer())
    identifiers = {item["canonical_id"] for item in units}
    assert identifiers == {"paragraph_0_0", "paragraph_0_1", "paragraph_1_0", "float_0"}
    references, answer_type, complete = mapped_answer_references(paper["qas"][0], units)
    assert references == (("float_0", "paragraph_0_1"), ("paragraph_0_0",))
    assert answer_type == "mixed_or_unknown"
    assert complete is True


def test_duplicate_normalized_candidate_text_is_ineligible(monkeypatch) -> None:
    monkeypatch.setattr(qasper, "TARGET_CASES", 1)
    monkeypatch.setattr(qasper, "MINIMUM_CASES", 1)
    selected, summary = select_sample({"paper": _paper(1, ambiguous=True)})
    assert selected == []
    assert summary["excluded"] == {"missing_unambiguous_evidence_reference": 1}


def test_sealed_sampling_is_order_invariant(monkeypatch) -> None:
    monkeypatch.setattr(qasper, "TARGET_CASES", 3)
    monkeypatch.setattr(qasper, "MINIMUM_CASES", 3)
    papers = {f"paper-{index}": _paper(index) for index in range(5)}
    first, first_summary = select_sample(papers)
    second, second_summary = select_sample(reversed(list(papers.items())))
    assert [row["question_id"] for row in first] == [row["question_id"] for row in second]
    assert first_summary == second_summary


def test_blind_cache_is_gold_free_and_multi_reference_gold_join_is_late(monkeypatch) -> None:
    monkeypatch.setattr(qasper, "TARGET_CASES", 4)
    monkeypatch.setattr(qasper, "MINIMUM_CASES", 4)
    papers = {f"paper-{index}": _paper(index) for index in range(4)}
    prepared, maps, summary = prepare_blind_cases(papers, Tokenizer())
    serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in ("answers", "annotation_id", "worker_id", "evidence", "free_form_answer", "yes_no"):
        assert f'"{forbidden}":' not in serialized
    assert len(prepared) == 4
    assert all("question_id" not in item and "paper_id" not in item for item in maps)
    gold = build_gold_rows(papers, maps, pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"])
    assert all(row["reference_count"] == 2 for row in gold)
    assert all(row["evidence_source_mode"] == "includes_float" for row in gold)
    coverage = build_candidate_coverage(gold, summary["sampling"], summary["candidate_pool_quartile_boundaries"])
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["minimum_cases_and_ceiling_checks_passed"] is True


def test_top2_proposal_targets_and_preservation_invariants() -> None:
    shared = _proposal_candidates([0, 0, 0, 0])
    paired = _proposal_candidates([0, 0, 2, 2])
    disjoint = _proposal_candidates([0, 2, 4, 6])
    assert top2_proposal_details(shared)["proposal_count"] == 2
    assert top2_proposal_details(paired)["proposal_count"] == 4
    assert top2_proposal_details(disjoint)["proposal_count"] == 8
    assert [top2_proposal_details(value)["target_cardinality"] for value in (shared, paired, disjoint)] == [2, 4, 5]
    for candidates in (shared, paired, disjoint):
        details = top2_proposal_details(candidates)
        selected = select_v48(candidates, TOP2_PROPOSAL_GUARDED_V48, token_budget=1024)
        selected_ids = {item["id"] for item in selected}
        assert selected_ids <= set(details["proposal_ids"])
        assert len(selected_ids) == details["target_cardinality"]
        if details["proposal_count"] <= 5:
            assert selected_ids == set(details["proposal_ids"])
        assert len(selected_ids) == len(selected)
        assert sum(item["token_count"] for item in selected) <= 1024


def test_top2_selector_is_permutation_monotone_and_source_kind_invariant() -> None:
    candidates = _proposal_candidates([0, 0, 2, 2])
    expected = {item["id"] for item in select_v48(candidates, TOP2_PROPOSAL_GUARDED_V48, token_budget=1024)}
    assert expected == {item["id"] for item in select_v48(list(reversed(candidates)), TOP2_PROPOSAL_GUARDED_V48, token_budget=1024)}
    transformed = deepcopy(candidates)
    for item in transformed:
        item["dynamic_role_scores"] = {role: 3.0 * value + 7.0 for role, value in item["dynamic_role_scores"].items()}
    assert expected == {item["id"] for item in select_v48(transformed, TOP2_PROPOSAL_GUARDED_V48, token_budget=1024)}
    changed_sources = deepcopy(candidates)
    for index, item in enumerate(changed_sources):
        item["source_kind"] = "float" if index % 2 else "paragraph"
    assert expected == {item["id"] for item in select_v48(changed_sources, TOP2_PROPOSAL_GUARDED_V48, token_budget=1024)}
    constrained = select_v48(candidates, TOP2_PROPOSAL_GUARDED_V48, token_budget=10)
    assert len(constrained) <= len(expected)


def test_locked_evaluation_and_report_are_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qasper, "TARGET_CASES", 4)
    monkeypatch.setattr(qasper, "MINIMUM_CASES", 4)
    monkeypatch.setattr(qasper, "MINIMUM_STRATUM_CASES", 1)
    papers = {f"paper-{index}": _paper(index) for index in range(4)}
    prepared, maps, summary = prepare_blind_cases(papers, Tokenizer())
    gold = build_gold_rows(papers, maps, pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"])
    scored = [_scored_row(case) for case in prepared]
    report, evidence = evaluate_qasper(gold, scored, query_summary={"fallback_rate": 0.0}, source_artifacts={"synthetic": True}, resamples=30)
    assert report["analysis"]["outcome"]["selector_adoption_authorized"] is False
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["metadata"]["multi_reference_gold_not_unioned"] is True
    assert len(evidence) == 4
    paths = (tmp_path / "result.json", tmp_path / "report.md", tmp_path / "cases.jsonl.gz")
    write_report(report, evidence, json_path=paths[0], markdown_path=paths[1], evidence_path=paths[2])
    first = tuple(path.read_bytes() for path in paths)
    write_report(report, evidence, json_path=paths[0], markdown_path=paths[1], evidence_path=paths[2])
    assert first == tuple(path.read_bytes() for path in paths)
