from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import research.frc_rag.doc2dial_wood_schema_corrected_transfer as schema
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _documents(count: int = 12) -> dict:
    result = {}
    for index in range(count):
        first = f"Eligibility {index}."
        second = f"Solution {index}."
        text = f"{first} {second}"
        result[f"doc-{index}"] = {
            "title": f"Document {index}",
            "doc_text": text,
            "spans": {
                f"sp-{index}-0": {
                    "id_sp": f"sp-{index}-0",
                    "start_sp": 0,
                    "end_sp": len(first),
                    "text_sp": first,
                },
                f"sp-{index}-1": {
                    "id_sp": f"sp-{index}-1",
                    "start_sp": len(first) + 1,
                    "end_sp": len(text),
                    "text_sp": second,
                },
            },
        }
    return {"doc_data": {"dmv": result}}


def _turns(index: int) -> list[dict]:
    return [
        {
            "turn_id": 0,
            "role": "user",
            "da": "request/query/open",
            "utterance": f"Eligibility question {index}",
            "reference": [],
        },
        {
            "turn_id": 1,
            "role": "agent",
            "da": "respond/reply/open",
            "utterance": f"Eligibility {index}.",
            "reference": [{"id_sp": f"sp-{index}-0", "label": "solution"}],
        },
        {
            "turn_id": 2,
            "role": "user",
            "da": "request/query/open",
            "utterance": f"Unanswerable but not OOD act {index}",
            "reference": [],
        },
        {
            "turn_id": 3,
            "role": "agent",
            "da": "respond/noSolution",
            "utterance": "No solution",
            "reference": [],
        },
        {
            "turn_id": 4,
            "role": "user",
            "da": "request/query/ood",
            "utterance": f"Weather {index}",
            "reference": [],
        },
        {
            "turn_id": 5,
            "role": "agent",
            "da": "respond/reply/ood",
            "utterance": "Irrelevant",
            "reference": [],
        },
    ]


def _dialogues(count: int = 12, *, keyed: bool = False) -> dict:
    values = {}
    for index in range(count):
        turns: list[dict] | dict[str, dict] = _turns(index)
        if keyed and index == 0:
            turns = {str(turn["turn_id"]): turn for turn in reversed(turns)}
        values[f"doc-{index}"] = [
            {"dial_id": f"dial-{index}", "turns": turns}
        ]
    return {"dial_data": {"dmv": values}}


def _candidate(identifier: str, index: int) -> dict:
    raw = {
        "anchor": float((9, 8, 7, 6)[index]),
        "first_fact": float((5, 9, 7, 6)[index]),
        "second_fact_or_bridge": float((5, 6, 9, 7)[index]),
        "counterevidence": float((5, 6, 7, 9)[index]),
    }
    score = 1.0 - index / 10.0
    return {
        "id": identifier,
        "source_id": identifier,
        "token_count": 8,
        "scores": {
            "bm25": score,
            "dense": score,
            "hybrid": score,
            "cross_encoder": score,
        },
        "static_role_scores": {role: score for role in STATIC_ROLES},
        "dynamic_role_scores": {
            role: float((index + role_index) % 4) / 3.0
            for role_index, role in enumerate(DYNAMIC_ROLES)
        },
        "raw_dynamic_role_scores": raw,
    }


def _scored(case_id: str, passed: bool) -> dict:
    values = [_candidate(f"span_{index:04d}", index) for index in range(4)]
    return {
        "id": case_id,
        "candidates": [
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "token_count": row["token_count"],
            }
            for row in values
        ],
        "candidate_scores": [
            {
                "id": row["id"],
                "scores": row["scores"],
                "static_role_scores": row["static_role_scores"],
                "dynamic_role_scores": row["dynamic_role_scores"],
                "raw_dynamic_role_scores": row["raw_dynamic_role_scores"],
            }
            for row in values
        ],
        "document_contrastive_gate": {
            "assigned_document_rank": 1 if passed else 2,
            "assigned_document_rank_one": passed,
            "probe_document_count": 2,
            "randomization_p_value_if_passed": 0.5 if passed else None,
            "document_ranking": ["own", "other"] if passed else ["other", "own"],
            "document_scores": {"own": 2.0, "other": 1.0},
            "decoy_fallback_used": False,
        },
        "gold_fields_visible_to_scorer": False,
    }


def test_protocol_and_source_disclose_schema_only_access() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_protocol_v56.json"
    )
    source_path = root / "docs/progressive_upgrade/doc2dial_wood_source_registration_v56.json"
    protocol = schema.validate_protocol(protocol_path)
    source = schema.validate_source_registration(
        source_path,
        protocol_path=protocol_path,
        source_archive=root
        / ".cache/benchmarks/doc2dial/v55_source/doc2dial_v0.9.zip",
    )
    assert protocol["pre_registration_source_access_disclosure"][
        "shared_document_and_wood_train_members_parsed_during_failed_v55_adapter_check"
    ]
    assert protocol["pre_registration_source_access_disclosure"][
        "wood_dev_confirmation_member_opened"
    ] is False
    assert source["pre_registration_access"]["metrics_computed"] is False


def test_keyed_turns_are_sorted_numerically_and_noninteger_keys_fail() -> None:
    normalised = schema._normalise_dialogue_source(_dialogues(1, keyed=True))
    turns = normalised["dial_data"]["dmv"]["doc-0"][0]["turns"]
    assert [turn["turn_id"] for turn in turns] == [0, 1, 2, 3, 4, 5]
    broken = _dialogues(1, keyed=True)
    raw = broken["dial_data"]["dmv"]["doc-0"][0]["turns"]
    raw["not-an-integer"] = raw.pop("5")
    try:
        schema._normalise_dialogue_source(broken)
    except ValueError as exc:
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("non-integer keyed turn was accepted")


def test_all_empty_references_are_no_answer_with_descriptive_subtypes() -> None:
    dialogues = schema._normalise_dialogue_source(_dialogues(1, keyed=True))
    rows, census = schema.extract_cases(_documents(1), dialogues)
    assert [row["answer_state"] for row in rows] == [
        "answer_bearing",
        "no_answer",
        "no_answer",
    ]
    assert [row["no_answer_subtype"] for row in rows] == [
        "answer_bearing",
        "other_empty",
        "ood_act",
    ]
    assert census["schema_exclusion_rate"] == 0.0


def test_sampling_and_blind_preparation_hide_state_subtype_and_acts() -> None:
    documents = _documents()
    dialogues = schema._normalise_dialogue_source(_dialogues(keyed=True))
    selected, summary = schema.select_balanced_sample(
        documents, dialogues, stage="development", target_per_group=6
    )
    reversed_dialogues = deepcopy(dialogues)
    reversed_dialogues["dial_data"]["dmv"] = dict(
        reversed(list(reversed_dialogues["dial_data"]["dmv"].items()))
    )
    second, _ = schema.select_balanced_sample(
        documents, reversed_dialogues, stage="development", target_per_group=6
    )
    assert [
        (row["dial_id"], row["target_turn_id"]) for row in selected
    ] == [(row["dial_id"], row["target_turn_id"]) for row in second]
    prepared, maps, store, structural = schema.prepare_blind_cases(
        selected, documents, Tokenizer(), stage="development"
    )
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 6,
        "no_answer": 6,
    }
    assert structural["decoy_fallback_rate"] == 0.0
    blind = json.dumps({"prepared": prepared, "store": store}, ensure_ascii=False)
    for forbidden in (
        '"answer_state"',
        '"no_answer_subtype"',
        '"da"',
        '"reference"',
        '"sp_id"',
    ):
        assert forbidden not in blind
    assert all(row["id"].startswith("w56c") for row in prepared)
    assert all("span_aliases" in row for row in maps)


def test_queries_and_transferred_selector_are_unchanged() -> None:
    prepared = [
        {
            "schema_version": schema.SCHEMA_VERSION,
            "dataset_id": schema.DATASET_ID,
            "capability": schema.CAPABILITY,
            "stage": "development",
            "id": "case",
            "query": "Eligibility?",
            "own_doc_key": "own",
            "probe_doc_keys": ["own", *[f"d{i}" for i in range(7)]],
            "decoy_fallback_used": False,
            "gold_fields_visible_to_scorer": False,
        }
    ]
    queries = schema.build_deterministic_queries(prepared)
    assert schema.validate_query_cache(prepared, queries)["fallback_rate"] == 0.0
    candidates = [_candidate(f"span_{index:04d}", index) for index in range(4)]
    selected = schema.select_v56(
        candidates, schema.CANDIDATE, gate_passed=True, token_budget=128
    )
    assert [row["id"] for row in selected] == [
        "span_0000",
        "span_0001",
        "span_0002",
    ]


def test_evaluation_adds_subtype_strata_and_v56_seed(monkeypatch) -> None:
    monkeypatch.setattr(schema, "BOOTSTRAP_RESAMPLES", 50)
    scored = [_scored(f"case-{index}", index % 2 == 0) for index in range(4)]
    gold = [
        {
            "case_id": f"case-{index}",
            "document_cluster": f"document-{index // 2}",
            "dialogue_cluster": f"dialogue-{index}",
            "answer_state": "answer_bearing" if index < 2 else "no_answer",
            "no_answer_subtype": (
                "answer_bearing" if index < 2 else ("ood_act" if index == 2 else "other_empty")
            ),
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
    report, evidence = schema.evaluate_stage(
        gold,
        scored,
        {"fallback_rate": 0.0},
        artifacts,
        stage="development",
    )
    second, second_evidence = schema.evaluate_stage(
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
    ]["seed"] == schema.BOOTSTRAP_SEED
    assert report["analysis"]["schema_corrections"][
        "retrieval_gate_selector_changed"
    ] is False
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert '"gold_candidate_ids"' not in serialized
