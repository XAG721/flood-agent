from __future__ import annotations

import gzip
import json
import types
from copy import deepcopy
from pathlib import Path

import numpy as np

import research.frc_rag.doc2dial_document_contrastive_role_closure as d2d
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _document_source(count: int = 12) -> dict:
    documents = {}
    for index in range(count):
        parts = [
            f"Eligibility condition {index}.",
            f"Solution action {index}.",
            f"Exception rule {index}.",
            f"Deadline detail {index}.",
        ]
        text = " ".join(parts)
        spans = {}
        cursor = 0
        for span_index, part in enumerate(parts):
            start = text.index(part, cursor)
            end = start + len(part)
            cursor = end
            spans[f"sp-{index}-{span_index}"] = {
                "sp_id": f"sp-{index}-{span_index}",
                "start_sp": start,
                "end_sp": end,
                "text_sp": part,
                "label": "must-remain-hidden",
            }
        documents[f"doc-{index}"] = {
            "doc_id": f"doc-{index}",
            "domain": "dmv",
            "title": f"Document {index}",
            "doc_text": text,
            "spans": spans,
        }
    return {"doc_data": {"dmv": documents}}


def _dialogue_source(count: int = 12) -> dict:
    dialogues = []
    for index in range(count):
        dialogues.append(
            {
                "dial_id": f"dial-{index}",
                "doc_id": f"doc-{index}",
                "domain": "dmv",
                "turns": [
                    {
                        "turn_id": "0",
                        "role": "user",
                        "utterance": f"What is eligibility {index}?",
                        "references": [],
                    },
                    {
                        "turn_id": "1",
                        "role": "agent",
                        "utterance": f"Eligibility response {index}",
                        "references": [
                            {
                                "sp_id": f"sp-{index}-0",
                                "label": "solution",
                            }
                        ],
                    },
                    {
                        "turn_id": "2",
                        "role": "user",
                        "utterance": f"What about unrelated weather {index}?",
                        "references": [],
                    },
                    {
                        "turn_id": "3",
                        "role": "agent",
                        "utterance": "Irrelevant",
                        "references": [],
                    },
                ],
            }
        )
    return {"dial_data": {"dmv": dialogues}}


def _candidate(identifier: str, index: int) -> dict:
    descending = 1.0 - 0.1 * index
    raw = {
        "anchor": float((9, 8, 7, 6)[index]),
        "first_fact": float((5, 9, 7, 6)[index]),
        "second_fact_or_bridge": float((5, 6, 9, 7)[index]),
        "counterevidence": float((5, 6, 7, 9)[index]),
    }
    dynamic = {
        role: float((index + role_index) % 4) / 3.0
        for role_index, role in enumerate(DYNAMIC_ROLES)
    }
    return {
        "id": identifier,
        "source_id": identifier,
        "token_count": 12,
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
    return [_candidate(f"span_{index:04d}", index) for index in range(4)]


def _scored_row(case_id: str, own_rank_one: bool = True) -> dict:
    candidates = _candidates()
    own = "doc-own"
    other = "doc-other"
    ranking = [own, other] if own_rank_one else [other, own]
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
        "document_contrastive_gate": {
            "assigned_document_rank": 1 if own_rank_one else 2,
            "assigned_document_rank_one": own_rank_one,
            "probe_document_count": 2,
            "randomization_p_value_if_passed": 0.5 if own_rank_one else None,
            "document_ranking": ranking,
            "document_scores": {own: 2.0, other: 1.0},
            "decoy_fallback_used": False,
        },
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_and_source_registration_preserve_prospective_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "docs/progressive_upgrade/doc2dial_document_contrastive_role_closure_protocol_v54.json"
    )
    source_path = root / "docs/progressive_upgrade/doc2dial_source_registration_v54.json"
    protocol = d2d.validate_protocol(protocol_path)
    source = d2d.validate_source_registration(
        source_path,
        protocol_path=protocol_path,
        source_root=root / ".cache/benchmarks/doc2dial/v54_source",
    )
    assert protocol["source_access_boundary_at_registration"][
        "dialogue_json_content_opened"
    ] is False
    assert protocol["confirmation_rule"][
        "validation_content_open_authorized_only_if_every_development_gate_passes"
    ] is True
    assert source["dialogue_or_document_json_content_read_before_registration"] is False
    assert source["validation_content_read"] is False


def test_reader_opens_only_requested_dialogue_split(tmp_path: Path) -> None:
    (tmp_path / d2d.SOURCE_FILES["documents"]).write_text(
        json.dumps(_document_source()), encoding="utf-8"
    )
    (tmp_path / d2d.SOURCE_FILES["development"]).write_text(
        json.dumps(_dialogue_source()), encoding="utf-8"
    )
    (tmp_path / d2d.SOURCE_FILES["confirmation"]).write_bytes(b"not-json")
    documents, dialogues = d2d.read_stage_sources(tmp_path, "development")
    assert documents == _document_source()
    assert dialogues == _dialogue_source()


def test_case_extraction_and_balanced_sampling_are_order_invariant() -> None:
    documents = _document_source()
    dialogues = _dialogue_source()
    rows, schema = d2d.extract_cases(documents, dialogues)
    assert len(rows) == 24
    assert schema["schema_exclusion_rate"] == 0.0
    selected, summary = d2d.select_balanced_sample(
        documents,
        dialogues,
        stage="development",
        target_per_group=6,
    )
    reversed_dialogues = deepcopy(dialogues)
    reversed_dialogues["dial_data"]["dmv"].reverse()
    second, second_summary = d2d.select_balanced_sample(
        documents,
        reversed_dialogues,
        stage="development",
        target_per_group=6,
    )
    assert [
        (row["dial_id"], row["target_turn_id"]) for row in selected
    ] == [(row["dial_id"], row["target_turn_id"]) for row in second]
    assert summary == second_summary
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 6,
        "no_answer": 6,
    }
    assert summary["maximum_cases_per_dialogue"] <= 2
    assert summary["maximum_cases_per_document_per_answer_state"] <= 4


def test_blind_preparation_hides_references_labels_and_raw_ids() -> None:
    documents = _document_source()
    selected, _ = d2d.select_balanced_sample(
        documents,
        _dialogue_source(),
        stage="development",
        target_per_group=4,
    )
    prepared, maps, store, census = d2d.prepare_blind_cases(
        selected, documents, Tokenizer(), stage="development"
    )
    assert len(prepared) == 8
    assert len(store) == 12
    assert census["decoy_fallback_rate"] == 0.0
    assert all(len(row["probe_doc_keys"]) == 8 for row in prepared)
    blind = json.dumps({"cases": prepared, "documents": store}, ensure_ascii=False)
    for forbidden in (
        '"references"',
        '"reference_ids"',
        '"sp_id"',
        '"label"',
        '"dial_id"',
        '"doc_id"',
        '"answer_state"',
    ):
        assert forbidden not in blind
    assert all("span_aliases" in row for row in maps)
    assert all(
        "must-remain-hidden" not in candidate["text"]
        for row in store
        for candidate in row["candidates"]
    )


def test_decoys_and_queries_are_deterministic_under_document_permutation() -> None:
    documents = _document_source()
    selected, _ = d2d.select_balanced_sample(
        documents,
        _dialogue_source(),
        stage="development",
        target_per_group=2,
    )
    first, _, _, _ = d2d.prepare_blind_cases(
        selected, documents, Tokenizer(), stage="development"
    )
    reversed_documents = deepcopy(documents)
    reversed_documents["doc_data"]["dmv"] = dict(
        reversed(list(reversed_documents["doc_data"]["dmv"].items()))
    )
    second, _, _, _ = d2d.prepare_blind_cases(
        selected, reversed_documents, Tokenizer(), stage="development"
    )
    assert first == second
    queries = d2d.build_deterministic_queries(first)
    assert d2d.validate_query_cache(first, queries)["fallback_rate"] == 0.0
    assert set(queries[0]["atomic_queries"]) == set(DYNAMIC_ROLES)


def test_retrieval_pool_is_order_invariant_and_bounded() -> None:
    candidates = [
        {
            "id": f"span_{index:04d}",
            "source_id": f"span_{index:04d}",
            "text": f"eligibility clause token {index}",
            "token_count": 4,
        }
        for index in range(120)
    ]
    dense = np.linspace(0.0, 1.0, num=len(candidates))
    first = d2d.build_retrieval_pool(
        candidates, query="eligibility clause", dense_scores=dense
    )
    second = d2d.build_retrieval_pool(
        list(reversed(candidates)),
        query="eligibility clause",
        dense_scores=list(reversed(dense)),
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) <= d2d.RETRIEVAL_POOL_MAXIMUM


def test_scorer_uses_shared_store_and_ranks_assigned_document(monkeypatch) -> None:
    documents = []
    keys = [f"doc-{index}" for index in range(8)]
    for index, key in enumerate(keys):
        documents.append(
            {
                "id": key,
                "candidates": [
                    {
                        "id": f"candidate-{index}",
                        "source_id": f"candidate-{index}",
                        "text": "own decisive answer" if index == 0 else f"decoy {index}",
                        "token_count": 3,
                    }
                ],
            }
        )
    scorer = d2d.FrozenDoc2DialScorer.__new__(d2d.FrozenDoc2DialScorer)
    scorer._documents = {row["id"]: row for row in documents}
    scorer._source_embedding_cache = {}

    def fake_encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def fake_predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(
            [10.0 if "own decisive" in text else float(index) / 100.0 for index, (_, text) in enumerate(pairs)],
            dtype=float,
        )

    scorer._encode = types.MethodType(fake_encode, scorer)
    scorer._predict = types.MethodType(fake_predict, scorer)

    def fake_parent(self, cases: list[dict]) -> list[dict]:
        return [
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
                "candidate_scores": [],
                "gold_fields_visible_to_scorer": False,
            }
            for row in cases
        ]

    monkeypatch.setattr(
        d2d.v50.FrozenContractNliScorer, "score_cases", fake_parent
    )
    rows = scorer.score_cases(
        [
            {
                "id": "case-1",
                "query": "what is the answer",
                "own_doc_key": keys[0],
                "probe_doc_keys": keys,
                "decoy_fallback_used": False,
                "gold_fields_visible_to_scorer": False,
            }
        ]
    )
    assert rows[0]["document_contrastive_gate"]["assigned_document_rank_one"]
    assert rows[0]["document_contrastive_gate"]["assigned_document_rank"] == 1
    assert len(scorer._source_embedding_cache) == 8


def test_rank_concurrent_role_closure_and_gate_are_separate() -> None:
    candidates = _candidates()
    details = d2d.rank_concurrent_role_closure_details(candidates)
    assert details["anchor_frontier_ids"] == [
        "span_0000",
        "span_0001",
        "span_0002",
    ]
    assert details["closure_ids"] == [
        "span_0000",
        "span_0001",
        "span_0002",
    ]
    selected = d2d.select_v54(
        candidates, d2d.CANDIDATE, gate_passed=True, token_budget=128
    )
    assert [row["id"] for row in selected] == details["closure_ids"]
    assert (
        d2d.select_v54(
            candidates, d2d.CANDIDATE, gate_passed=False, token_budget=128
        )
        == []
    )


def test_gold_join_uses_exact_span_ids_after_scoring() -> None:
    documents = _document_source()
    dialogues = _dialogue_source()
    selected, _ = d2d.select_balanced_sample(
        documents, dialogues, stage="development", target_per_group=1
    )
    prepared, maps, store, census = d2d.prepare_blind_cases(
        selected, documents, Tokenizer(), stage="development"
    )
    by_document = {row["id"]: row for row in store}
    scored = [
        {
            "id": row["id"],
            "candidates": [
                {
                    "id": candidate["id"],
                    "source_id": candidate["source_id"],
                    "token_count": candidate["token_count"],
                }
                for candidate in by_document[row["own_doc_key"]]["candidates"]
            ],
        }
        for row in prepared
    ]
    gold = d2d.build_gold_rows(
        documents,
        dialogues,
        maps,
        scored,
        document_length_quartile_boundaries=census[
            "document_length_quartile_boundaries"
        ],
        turn_position_quartile_boundaries=census[
            "dialogue_turn_position_quartile_boundaries"
        ],
    )
    assert {row["answer_state"] for row in gold} == {
        "answer_bearing",
        "no_answer",
    }
    assert all(row["candidate_ceiling_complete"] for row in gold)
    assert sorted(len(row["gold_candidate_ids"]) for row in gold) == [0, 1]


def test_evaluation_and_report_are_deterministic_and_gold_minimal(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(d2d, "BOOTSTRAP_RESAMPLES", 50)
    scored = [
        _scored_row(f"case-{index}", own_rank_one=index % 2 == 0)
        for index in range(4)
    ]
    gold = [
        {
            "case_id": f"case-{index}",
            "document_cluster": f"document-{index // 2}",
            "dialogue_cluster": f"dialogue-{index}",
            "answer_state": "answer_bearing" if index < 2 else "no_answer",
            "gold_candidate_ids": ["span_0000"] if index < 2 else [],
            "reference_count": 1 if index < 2 else 0,
            "reference_count_group": "one" if index < 2 else "none",
            "domain": "synthetic",
            "candidate_unit_count": 4,
            "candidate_pool_quartile": "q1",
            "document_length_quartile": "q1",
            "dialogue_turn_position_quartile": "q1",
            "candidate_ceiling_complete": True,
        }
        for index in range(4)
    ]
    artifacts = {
        "score_fallback_rate": 0.0,
        "sampling": {
            "maximum_cases_per_dialogue": 1,
            "maximum_cases_per_document_per_answer_state": 1,
            "schema_exclusion_rate": 0.0,
        },
        "structural_census": {"decoy_fallback_rate": 0.0},
    }
    report, evidence = d2d.evaluate_stage(
        gold,
        scored,
        {"fallback_rate": 0.0},
        artifacts,
        stage="development",
    )
    second_report, second_evidence = d2d.evaluate_stage(
        gold,
        scored,
        {"fallback_rate": 0.0},
        artifacts,
        stage="development",
    )
    assert report == second_report
    assert evidence == second_evidence
    assert report["analysis"]["outcome"]["validation_open_authorized"] is False
    serialized = json.dumps(evidence, ensure_ascii=False)
    for forbidden in (
        '"gold_candidate_ids"',
        '"references"',
        '"query"',
        '"utterance"',
        '"sp_id"',
    ):
        assert forbidden not in serialized

    d2d.write_report(
        report,
        evidence,
        tmp_path / "result.json",
        tmp_path / "result.md",
        tmp_path / "evidence.jsonl.gz",
    )
    first = (tmp_path / "evidence.jsonl.gz").read_bytes()
    d2d.write_report(
        report,
        evidence,
        tmp_path / "result.json",
        tmp_path / "result.md",
        tmp_path / "evidence.jsonl.gz",
    )
    assert first == (tmp_path / "evidence.jsonl.gz").read_bytes()
    with gzip.open(tmp_path / "evidence.jsonl.gz", "rt", encoding="utf-8") as handle:
        published = [json.loads(line) for line in handle]
    assert published == d2d._round_for_display(evidence)
