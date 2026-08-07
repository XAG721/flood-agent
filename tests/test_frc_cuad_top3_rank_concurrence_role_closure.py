from __future__ import annotations

import gzip
import json
import types
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np

import research.frc_rag.cuad_top3_rank_concurrence_role_closure as cuad
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _source(contract_count: int = 4) -> dict:
    data = []
    for index in range(contract_count):
        answer = f"Alpha obligation {index}."
        context = f"{answer}\nBeta exception {index}; Gamma condition {index}."
        data.append(
            {
                "title": f"contract-{index}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": f"q-{index}-answer__Governing-Law",
                                "question": f"What is alpha obligation {index}?",
                                "answers": [{"text": answer, "answer_start": 0}],
                                "is_impossible": False,
                            },
                            {
                                "id": f"q-{index}-none__Parties",
                                "question": f"Is delta stated {index}?",
                                "answers": [],
                                "is_impossible": True,
                            },
                        ],
                    }
                ],
            }
        )
    return {"version": "synthetic", "data": data}


def _candidate(identifier: str, index: int) -> dict:
    descending = 1.0 - 0.1 * index
    raw = {
        "anchor": float((3, 2, 1, 4, 5, 6)[index]),
        "first_fact": float((5, 4, 3, 2, 1, 6)[index]),
        "second_fact_or_bridge": float((6, 5, 4, 3, 2, 1)[index]),
        "counterevidence": float((3, 4, 1, 2, 6, 5)[index]),
    }
    dynamic = {
        role: float((index + role_index) % 6) / 5.0
        for role_index, role in enumerate(DYNAMIC_ROLES)
    }
    return {
        "id": identifier,
        "source_id": identifier,
        "token_count": 16,
        "scores": {
            "bm25": descending,
            "dense": descending,
            "hybrid": descending,
            "cross_encoder": descending,
        },
        "static_role_scores": {role: descending for role in STATIC_ROLES},
        "dynamic_role_scores": dynamic,
        "raw_dynamic_role_scores": raw,
    }


def _candidates() -> list[dict]:
    return [_candidate(f"span_{index:04d}", index) for index in range(6)]


def _scored_row(case_id: str) -> dict:
    candidates = _candidates()
    return {
        "id": case_id,
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
                "static_role_scores": item["static_role_scores"],
                "dynamic_role_scores": item["dynamic_role_scores"],
                "raw_dynamic_role_scores": item["raw_dynamic_role_scores"],
            }
            for item in candidates
        ],
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_and_source_registration_preserve_prospective_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = cuad.validate_protocol(
        root
        / "docs/progressive_upgrade/cuad_top3_rank_concurrence_role_closure_protocol_v53.json"
    )
    source = json.loads(
        (root / "docs/progressive_upgrade/cuad_source_registration_v53.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        protocol["source_access_boundary_at_registration"]["cuad_train_content_opened"]
        is False
    )
    assert (
        protocol["source_access_boundary_at_registration"]["cuad_test_content_opened"]
        is False
    )
    assert (
        protocol["confirmation_rule"][
            "test_content_open_authorized_only_if_every_development_gate_passes"
        ]
        is True
    )
    assert source["data_archive"]["member_names_listed_without_member_content_read"]
    assert source["sample_distribution_or_gold_read_before_registration"] is False


def test_reader_opens_only_requested_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "cuad.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("train_separate_questions.json", json.dumps(_source()))
        archive.writestr("test.json", b"not-json-test")
        archive.writestr("CUADv1.json", b"not-json-full")
    assert cuad.read_source_member(archive_path, "development") == _source()


def test_balanced_sample_is_order_invariant_and_contract_capped() -> None:
    source = _source(6)
    selected, summary = cuad.select_balanced_sample(
        source,
        stage="development",
        target_per_group=4,
        maximum_cases_per_contract=2,
    )
    reversed_source = deepcopy(source)
    reversed_source["data"].reverse()
    second, second_summary = cuad.select_balanced_sample(
        reversed_source,
        stage="development",
        target_per_group=4,
        maximum_cases_per_contract=2,
    )
    assert [row["question_id"] for row in selected] == [
        row["question_id"] for row in second
    ]
    assert summary == second_summary
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 4,
        "no_answer": 4,
    }
    assert summary["maximum_cases_per_contract"] <= 2


def test_candidate_construction_preserves_exact_offsets_and_is_deterministic() -> None:
    context = "Alpha obligation. Beta exception; Gamma condition.\nDelta clause."
    candidates = cuad.build_source_candidates(context, Tokenizer())
    second = cuad.build_source_candidates(context, Tokenizer())
    assert candidates == second
    assert len({row["id"] for row in candidates}) == len(candidates)
    assert all(context[row["start"] : row["end"]] == row["text"] for row in candidates)
    assert any(row["text"] == "Alpha obligation." for row in candidates)
    assert any(row["text"] == "Beta exception;" for row in candidates)
    assert all(row["id"].startswith("span_") for row in candidates)


def test_blind_cache_queries_and_maps_do_not_export_gold() -> None:
    selected, _ = cuad.select_balanced_sample(
        _source(2),
        stage="development",
        target_per_group=2,
        maximum_cases_per_contract=2,
    )
    prepared, maps, census = cuad.prepare_blind_cases(
        selected, Tokenizer(), stage="development"
    )
    queries = cuad.build_deterministic_queries(prepared)
    assert cuad.validate_query_cache(prepared, queries)["fallback_rate"] == 0.0
    blind_serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in (
        '"answers"',
        '"answer_start"',
        '"answer_state"',
        '"contract_key"',
        '"question_id"',
        '"start"',
        '"end"',
    ):
        assert forbidden not in blind_serialized
    assert all("contract_id_sha256" in row for row in maps)
    assert census["gold_fields_exported_to_blind_cache"] is False


def test_retrieval_pool_is_order_invariant_and_bounded() -> None:
    candidates = [
        {
            "id": f"span_{index:04d}",
            "source_id": f"span_{index:04d}",
            "text": f"alpha clause token {index}",
            "token_count": 4,
        }
        for index in range(120)
    ]
    dense = np.linspace(0.0, 1.0, num=len(candidates))
    first = cuad.build_retrieval_pool(
        candidates, query="alpha clause", dense_scores=dense
    )
    second = cuad.build_retrieval_pool(
        list(reversed(candidates)),
        query="alpha clause",
        dense_scores=list(reversed(dense)),
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) <= cuad.RETRIEVAL_POOL_MAXIMUM


def test_source_embeddings_are_cached_without_gold(monkeypatch) -> None:
    calls: list[list[str]] = []
    scorer = cuad.FrozenCuadScorer.__new__(cuad.FrozenCuadScorer)
    scorer._source_embedding_cache = {}

    def fake_encode(self, texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.asarray(
            [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    scorer._encode = types.MethodType(fake_encode, scorer)

    def fake_parent(self, cases: list[dict]) -> list[dict]:
        return [
            {
                "id": row["id"],
                "candidates": row["candidates"],
                "candidate_scores": [],
                "gold_fields_visible_to_scorer": False,
            }
            for row in cases
        ]

    monkeypatch.setattr(cuad.v50.FrozenContractNliScorer, "score_cases", fake_parent)
    base_candidates = [
        {
            "id": f"span_{index}",
            "source_id": f"span_{index}",
            "text": f"clause {index}",
            "token_count": 2,
        }
        for index in range(4)
    ]
    cases = [
        {
            "id": f"case-{index}",
            "query": f"query {index}",
            "candidates": base_candidates,
            "gold_fields_visible_to_scorer": False,
        }
        for index in range(2)
    ]
    rows = scorer.score_cases(cases)
    assert len(rows) == 2
    assert sum(call == [row["text"] for row in base_candidates] for call in calls) == 1
    assert calls[-2:] == [["query 0"], ["query 1"]]
    assert len(scorer._source_embedding_cache) == 1


def test_top3_gate_and_role_closure_are_rank_invariant() -> None:
    candidates = _candidates()
    details = cuad.top3_rank_concurrence_details(candidates)
    assert details["top3_rank_concurrence_passed"] is True
    assert details == cuad.top3_rank_concurrence_details(list(reversed(candidates)))

    transformed = deepcopy(candidates)
    transforms = {
        "anchor": (2.0, 3.0),
        "first_fact": (4.0, -5.0),
        "second_fact_or_bridge": (0.25, 11.0),
    }
    for candidate in transformed:
        for role, (scale, shift) in transforms.items():
            value = candidate["raw_dynamic_role_scores"][role]
            candidate["raw_dynamic_role_scores"][role] = scale * value + shift
    assert cuad.top3_rank_concurrence_details(transformed) == details
    candidate_ids = [
        row["id"]
        for row in cuad.select_v53(candidates, cuad.CANDIDATE, token_budget=256)
    ]
    ablation_ids = [
        row["id"]
        for row in cuad.select_v53(
            candidates, cuad.SAME_GATE_FRC_ABLATION, token_budget=256
        )
    ]
    assert (
        candidate_ids[: len(details["concurrence_nucleus_ids"])]
        == details["concurrence_nucleus_ids"]
    )
    assert candidate_ids != ablation_ids


def test_official_answer_match_and_gold_join_follow_registered_rule() -> None:
    assert cuad.official_answer_match(
        "Alpha payment clause", "Alpha payment clause applies", parties=False
    )
    assert not cuad.official_answer_match(
        "Alpha payment clause", "Unrelated exception", parties=False
    )
    assert cuad.official_answer_match(
        "Acme Holdings", "Acme Holdings and Beta LLC", parties=True
    )

    selected, _ = cuad.select_balanced_sample(
        _source(1),
        stage="development",
        target_per_group=1,
        maximum_cases_per_contract=2,
    )
    prepared, maps, census = cuad.prepare_blind_cases(
        selected, Tokenizer(), stage="development"
    )
    scored = [
        {
            "id": row["id"],
            "candidates": [
                {
                    "id": candidate["id"],
                    "source_id": candidate["source_id"],
                    "token_count": candidate["token_count"],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in prepared
    ]
    gold = cuad.build_gold_rows(
        _source(1),
        maps,
        scored,
        contract_length_quartile_boundaries=census[
            "contract_length_quartile_boundaries"
        ],
    )
    assert {row["answer_state"] for row in gold} == {"answer_bearing", "no_answer"}
    assert all(row["candidate_ceiling_complete"] for row in gold)


def test_evaluation_and_report_are_deterministic_and_gold_minimal(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cuad, "BOOTSTRAP_RESAMPLES", 50)
    scored = [_scored_row(f"case-{index}") for index in range(4)]
    gold = [
        {
            "case_id": f"case-{index}",
            "contract_cluster": f"contract-{index // 2}",
            "answer_state": "answer_bearing" if index < 2 else "no_answer",
            "answer_groups": [["span_0000"]] if index < 2 else [],
            "answer_count": 1 if index < 2 else 0,
            "answer_count_group": "one" if index < 2 else "none",
            "category": "synthetic",
            "candidate_unit_count": 6,
            "candidate_pool_quartile": "q1",
            "contract_length_quartile": "q1",
            "candidate_ceiling_complete": True,
        }
        for index in range(4)
    ]
    query_summary = {"fallback_rate": 0.0}
    artifacts = {"score_fallback_rate": 0.0}
    report, evidence = cuad.evaluate_stage(
        gold, scored, query_summary, artifacts, stage="development"
    )
    second_report, second_evidence = cuad.evaluate_stage(
        gold, scored, query_summary, artifacts, stage="development"
    )
    assert report == second_report
    assert evidence == second_evidence
    assert report["analysis"]["outcome"]["test_open_authorized"] is False
    serialized = json.dumps(evidence, ensure_ascii=False)
    for forbidden in (
        '"answer_groups"',
        '"gold_candidate_ids"',
        '"question"',
        '"context"',
        '"answer_start"',
    ):
        assert forbidden not in serialized

    cuad.write_report(
        report,
        evidence,
        tmp_path / "result.json",
        tmp_path / "result.md",
        tmp_path / "evidence.jsonl.gz",
    )
    with gzip.open(tmp_path / "evidence.jsonl.gz", "rt", encoding="utf-8") as handle:
        published = [json.loads(line) for line in handle]
    assert published == cuad._round_for_display(evidence)
