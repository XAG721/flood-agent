from __future__ import annotations

import json
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import research.frc_rag.doc2dial_wood_document_contrastive_transfer as wood
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _documents(count: int = 12) -> dict:
    values = {}
    for index in range(count):
        parts = [
            f"Eligibility {index}.",
            f"Solution {index}.",
            f"Exception {index}.",
        ]
        text = " ".join(parts)
        spans = {}
        cursor = 0
        for position, part in enumerate(parts):
            start = text.index(part, cursor)
            end = start + len(part)
            cursor = end
            sp_id = f"sp-{index}-{position}"
            spans[sp_id] = {
                "id_sp": sp_id,
                "start_sp": start,
                "end_sp": end,
                "text_sp": part,
                "tag": "hidden-label",
            }
        values[f"doc-{index}"] = {
            "title": f"Document {index}",
            "doc_text": text,
            "spans": spans,
        }
    return {"doc_data": {"dmv": values}}


def _dialogues(count: int = 12) -> dict:
    documents = {}
    for index in range(count):
        documents[f"doc-{index}"] = [
            {
                "dial_id": f"dial-{index}",
                "turns": [
                    {
                        "turn_id": 0,
                        "role": "user",
                        "da": "request/query/open",
                        "utterance": f"What is eligibility {index}?",
                        "reference": [],
                    },
                    {
                        "turn_id": 1,
                        "role": "agent",
                        "da": "respond/reply/open",
                        "utterance": f"Eligibility {index}.",
                        "reference": [
                            {"id_sp": f"sp-{index}-0", "label": "solution"}
                        ],
                    },
                    {
                        "turn_id": 2,
                        "role": "user",
                        "da": "request/query/ood",
                        "utterance": f"Tell me unrelated weather {index}",
                        "reference": [],
                    },
                    {
                        "turn_id": 3,
                        "role": "agent",
                        "da": "respond/reply/ood",
                        "utterance": "Irrelevant",
                        "reference": [],
                    },
                ],
            }
        ]
    return {"dial_data": {"dmv": documents}}


def _archive(path: Path, *, invalid_confirmation: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(wood.SOURCE_MEMBERS["documents"], json.dumps(_documents()))
        archive.writestr(
            wood.SOURCE_MEMBERS["development"], json.dumps(_dialogues())
        )
        archive.writestr(
            wood.SOURCE_MEMBERS["confirmation"],
            b"not-json" if invalid_confirmation else json.dumps(_dialogues()),
        )
        archive.writestr(
            "doc2dial/v0.9/data/woOOD/doc2dial_dial_train.json", b"not-source"
        )


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


def _scored(case_id: str, pass_gate: bool) -> dict:
    candidates = [_candidate(f"span_{index:04d}", index) for index in range(4)]
    own, other = "own", "other"
    return {
        "id": case_id,
        "candidates": [
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "token_count": row["token_count"],
            }
            for row in candidates
        ],
        "candidate_scores": [
            {
                "id": row["id"],
                "scores": row["scores"],
                "static_role_scores": row["static_role_scores"],
                "dynamic_role_scores": row["dynamic_role_scores"],
                "raw_dynamic_role_scores": row["raw_dynamic_role_scores"],
            }
            for row in candidates
        ],
        "document_contrastive_gate": {
            "assigned_document_rank": 1 if pass_gate else 2,
            "assigned_document_rank_one": pass_gate,
            "probe_document_count": 2,
            "randomization_p_value_if_passed": 0.5 if pass_gate else None,
            "document_ranking": [own, other] if pass_gate else [other, own],
            "document_scores": {own: 2.0, other: 1.0},
            "decoy_fallback_used": False,
        },
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_source_and_member_boundaries_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "docs/progressive_upgrade/doc2dial_wood_document_contrastive_transfer_protocol_v55.json"
    )
    source_path = root / "docs/progressive_upgrade/doc2dial_wood_source_registration_v55.json"
    protocol = wood.validate_protocol(protocol_path)
    source = wood.validate_source_registration(
        source_path,
        protocol_path=protocol_path,
        source_archive=root
        / ".cache/benchmarks/doc2dial/v55_source/doc2dial_v0.9.zip",
    )
    assert protocol["source_access_boundary_at_registration"][
        "v0_9_document_or_dialogue_content_opened"
    ] is False
    assert source["member_resolution_checks"]["wood_train_member_unique"]
    assert source["confirmation_content_opened"] is False


def test_reader_opens_only_requested_wood_split(tmp_path: Path) -> None:
    archive_path = tmp_path / "doc2dial.zip"
    _archive(archive_path, invalid_confirmation=True)
    documents, dialogues = wood.read_stage_sources(archive_path, "development")
    assert documents == _documents()
    rows, schema = wood.extract_cases(documents, dialogues)
    assert len(rows) == 24
    assert schema["schema_exclusion_rate"] == 0.0


def test_ood_requires_empty_reference_and_adjacent_ood_act() -> None:
    dialogues = _dialogues(1)
    raw = dialogues["dial_data"]["dmv"]["doc-0"][0]["turns"]
    raw[2]["da"] = "request/query/open"
    raw[3]["da"] = "respond/reply/open"
    normalised = wood._normalise_dialogue_source(dialogues)
    rows, schema = wood.extract_cases(_documents(1), normalised)
    assert Counter(row["answer_state"] for row in rows) == {"answer_bearing": 1}
    assert schema["schema_exclusion_reasons"] == {
        "empty_reference_without_adjacent_ood_act": 1
    }


def test_balanced_sampling_is_order_invariant_and_capped() -> None:
    documents = _documents()
    normalised = wood._normalise_dialogue_source(_dialogues())
    selected, summary = wood.select_balanced_sample(
        documents, normalised, stage="development", target_per_group=6
    )
    reversed_source = deepcopy(normalised)
    reversed_source["dial_data"]["dmv"] = dict(
        reversed(list(reversed_source["dial_data"]["dmv"].items()))
    )
    second, second_summary = wood.select_balanced_sample(
        documents, reversed_source, stage="development", target_per_group=6
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


def test_blind_cache_hides_old_schema_gold_and_uses_v55_decoys() -> None:
    documents = _documents()
    dialogues = wood._normalise_dialogue_source(_dialogues())
    selected, _ = wood.select_balanced_sample(
        documents, dialogues, stage="development", target_per_group=4
    )
    prepared, maps, store, census = wood.prepare_blind_cases(
        selected, documents, Tokenizer(), stage="development"
    )
    assert len(prepared) == 8
    assert len(store) == 12
    assert census["decoy_fallback_rate"] == 0.0
    assert all(row["id"].startswith("w55c") for row in prepared)
    blind = json.dumps({"prepared": prepared, "store": store}, ensure_ascii=False)
    for forbidden in (
        '"reference"',
        '"references"',
        '"id_sp"',
        '"sp_id"',
        '"da"',
        '"answer_state"',
        "hidden-label",
    ):
        assert forbidden not in blind
    assert all("span_aliases" in row for row in maps)


def test_query_and_selector_transfer_are_exact_and_gate_shared() -> None:
    prepared = [
        {
            "schema_version": wood.SCHEMA_VERSION,
            "dataset_id": wood.DATASET_ID,
            "capability": wood.CAPABILITY,
            "stage": "development",
            "id": "case-1",
            "query": "What is eligibility?",
            "own_doc_key": "own",
            "probe_doc_keys": ["own", *[f"d{i}" for i in range(7)]],
            "decoy_fallback_used": False,
            "gold_fields_visible_to_scorer": False,
        }
    ]
    queries = wood.build_deterministic_queries(prepared)
    assert wood.validate_query_cache(prepared, queries)["fallback_rate"] == 0.0
    assert set(queries[0]["atomic_queries"]) == set(DYNAMIC_ROLES)
    candidates = [_candidate(f"span_{index:04d}", index) for index in range(4)]
    selected = wood.select_v55(
        candidates, wood.CANDIDATE, gate_passed=True, token_budget=128
    )
    assert [row["id"] for row in selected] == [
        "span_0000",
        "span_0001",
        "span_0002",
    ]
    assert (
        wood.select_v55(
            candidates, wood.CANDIDATE, gate_passed=False, token_budget=128
        )
        == []
    )


def test_gold_join_maps_v0_9_id_sp_after_complete_scores() -> None:
    documents = _documents()
    dialogues = wood._normalise_dialogue_source(_dialogues())
    selected, _ = wood.select_balanced_sample(
        documents, dialogues, stage="development", target_per_group=1
    )
    prepared, maps, store, census = wood.prepare_blind_cases(
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
    gold = wood.build_gold_rows(
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
    assert sorted(len(row["gold_candidate_ids"]) for row in gold) == [0, 1]
    assert all(row["candidate_ceiling_complete"] for row in gold)


def test_evaluation_is_deterministic_gold_minimal_and_uses_v55_seed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(wood, "BOOTSTRAP_RESAMPLES", 50)
    scored = [_scored(f"case-{index}", index % 2 == 0) for index in range(4)]
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
    report, evidence = wood.evaluate_stage(
        gold,
        scored,
        {"fallback_rate": 0.0},
        artifacts,
        stage="development",
    )
    second, second_evidence = wood.evaluate_stage(
        gold,
        scored,
        {"fallback_rate": 0.0},
        artifacts,
        stage="development",
    )
    assert report == second
    assert evidence == second_evidence
    assert report["analysis"]["family_comparison"][
        "candidate_minus_ungated_v49"
    ]["seed"] == wood.BOOTSTRAP_SEED
    assert report["analysis"]["outcome"]["confirmation_open_authorized"] is False
    serialized = json.dumps(evidence, ensure_ascii=False)
    for forbidden in ('"gold_candidate_ids"', '"query"', '"reference"', '"sp_id"'):
        assert forbidden not in serialized
