from __future__ import annotations

import gzip
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import research.frc_rag.contractnli_native_zero_consensus_abstention as cnli
from research.frc_rag.contractnli_native_zero_consensus_abstention import (
    ANCHOR_GATE_FRC_V50,
    GATED_NON_FRC_METHODS,
    NATIVE_ZERO_CONSENSUS_FRC_V50,
    QUERY_TEMPLATES,
    build_deterministic_queries,
    build_gold_rows,
    build_span_units,
    evaluate_contractnli,
    native_zero_gate_details,
    prepare_blind_cases,
    read_test_source,
    select_sample,
    select_v50,
    validate_protocol,
    validate_query_cache,
    write_report,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _source(document_count: int = 3) -> dict:
    choices = (
        "Entailment",
        "Contradiction",
        "NotMentioned",
        "Entailment",
        "Contradiction",
        "NotMentioned",
    )
    labels = {
        f"hypothesis-{index}": {
            "hypothesis": f"Registered synthetic hypothesis {index}."
        }
        for index in range(len(choices))
    }
    documents = []
    for document_index in range(document_count):
        first = f"Alpha clause {document_index}."
        second = f"Beta exception {document_index}."
        text = f"{first}\n{second}"
        annotations = {}
        for hypothesis_index, choice in enumerate(choices):
            annotations[f"hypothesis-{hypothesis_index}"] = {
                "choice": choice,
                "spans": []
                if choice == "NotMentioned"
                else [hypothesis_index % 2],
            }
        documents.append(
            {
                "id": f"document-{document_index}",
                "document_type": "synthetic-nda",
                "text": text,
                "spans": [
                    [0, len(first)],
                    [len(first) + 1, len(text)],
                ],
                "annotation_sets": [{"annotations": annotations}],
            }
        )
    return {"labels": labels, "documents": documents}


def _candidate(
    identifier: str,
    index: int,
    *,
    anchor: float,
    support: float,
    contradiction: float,
    exception: float = -1.0,
) -> dict:
    rank = 1.0 - index * 0.1
    dynamic = {
        "anchor": rank,
        "first_fact": rank - 0.01,
        "second_fact_or_bridge": rank - 0.02,
        "counterevidence": rank - 0.03,
    }
    return {
        "id": identifier,
        "source_id": identifier,
        "source_kind": "contract_span",
        "text": f"Synthetic candidate {identifier}.",
        "token_count": 5,
        "scores": {
            "bm25": rank,
            "dense": rank,
            "hybrid": rank,
            "cross_encoder": rank,
        },
        "static_role_scores": {role: rank for role in STATIC_ROLES},
        "dynamic_role_scores": dynamic,
        "raw_dynamic_role_scores": {
            "anchor": anchor,
            "first_fact": support,
            "second_fact_or_bridge": contradiction,
            "counterevidence": exception,
        },
    }


def _scored_row(case_id: str, *, consensus: bool) -> dict:
    candidates = [
        _candidate(
            "span_0000",
            0,
            anchor=1.0,
            support=1.0 if consensus else -1.0,
            contradiction=-1.0,
        ),
        _candidate(
            "span_0001",
            1,
            anchor=-1.0,
            support=-1.0,
            contradiction=-1.0,
        ),
        _candidate(
            "span_0002",
            2,
            anchor=-2.0,
            support=-2.0,
            contradiction=-2.0,
        ),
    ]
    return {
        "id": case_id,
        "candidates": [
            {
                "id": item["id"],
                "source_id": item["source_id"],
                "source_kind": item["source_kind"],
                "text": item["text"],
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


def test_protocol_freezes_native_zero_formula_and_split_exclusions() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_protocol_v50.json"
    )
    assert protocol["methods"]["candidate_method"] == NATIVE_ZERO_CONSENSUS_FRC_V50
    assert protocol["query_templates"] == {
        **QUERY_TEMPLATES,
        "query_text_or_template_may_not_change_after_registration": True,
    }
    assert protocol["development_boundary"][
        "contractnli_train_split_permanently_excluded"
    ] is True
    assert protocol["development_boundary"][
        "contractnli_dev_split_permanently_excluded"
    ] is True


def test_archive_reader_opens_only_test_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "contract-nli.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("contract-nli/train.json", b"not-json-train")
        archive.writestr("contract-nli/dev.json", b"not-json-dev")
        archive.writestr(
            "contract-nli/test.json",
            json.dumps(_source(1), ensure_ascii=False).encode(),
        )
    assert read_test_source(archive_path) == _source(1)


def test_official_half_open_spans_and_late_gold_join_map_exactly(monkeypatch) -> None:
    source = _source(1)
    units = build_span_units(source["documents"][0], Tokenizer())
    text = source["documents"][0]["text"]
    assert [text[item["start"] : item["end"]] for item in units] == [
        "Alpha clause 0.",
        "Beta exception 0.",
    ]

    monkeypatch.setattr(cnli, "TARGET_CASES", 6)
    monkeypatch.setattr(cnli, "TARGET_PER_LABEL", 2)
    selected, _ = select_sample(source)
    prepared, maps, census = prepare_blind_cases(selected, Tokenizer())
    gold = build_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=census["candidate_pool_quartile_boundaries"],
        document_length_quartile_boundaries=census[
            "document_length_quartile_boundaries"
        ],
    )
    assert len(prepared) == len(gold) == 6
    assert all(row["candidate_ceiling_complete"] for row in gold)
    assert all(
        not row["gold_candidate_ids"]
        for row in gold
        if row["label_group"] == "NotMentioned"
    )


def test_whitespace_only_official_spans_are_skipped_without_renumbering() -> None:
    document = {
        "text": "Alpha. \nBeta.",
        "spans": [[0, 6], [6, 8], [8, 13]],
    }
    units = build_span_units(document, Tokenizer())
    assert [item["canonical_id"] for item in units] == [
        "span_0000",
        "span_0002",
    ]
    assert [item["text"] for item in units] == ["Alpha.", "Beta."]


def test_balanced_sampling_is_order_invariant_and_respects_document_cap() -> None:
    source = _source(100)
    first, first_summary = select_sample(source)
    reversed_source = {
        "labels": dict(reversed(list(source["labels"].items()))),
        "documents": list(reversed(source["documents"])),
    }
    second, second_summary = select_sample(reversed_source)
    first_keys = [
        (row["document_id"], row["hypothesis_key"], row["label"]) for row in first
    ]
    second_keys = [
        (row["document_id"], row["hypothesis_key"], row["label"]) for row in second
    ]
    assert first_keys == second_keys
    assert first_summary == second_summary
    assert first_summary["selected_label_counts"] == {
        "Entailment": 200,
        "Contradiction": 200,
        "NotMentioned": 200,
    }
    assert first_summary["maximum_cases_per_document"] == 6


def test_blind_cache_and_deterministic_queries_remain_gold_free(monkeypatch) -> None:
    monkeypatch.setattr(cnli, "TARGET_CASES", 6)
    monkeypatch.setattr(cnli, "TARGET_PER_LABEL", 2)
    selected, _ = select_sample(_source(1))
    prepared, maps, _ = prepare_blind_cases(selected, Tokenizer())
    serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in (
        '"annotation_sets"',
        '"choice"',
        '"document_id"',
        '"gold_candidate_ids"',
        '"label"',
        '"spans"',
        '"start"',
        '"end"',
    ):
        assert forbidden not in serialized
    assert all("document_id" not in item and "hypothesis_key" not in item for item in maps)
    queries = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, queries)
    assert summary["fallback_rate"] == 0.0
    for prepared_row, query_row in zip(prepared, queries, strict=True):
        assert query_row["atomic_queries"]["anchor"] == prepared_row["query"]


def test_native_zero_gate_requires_same_span_dual_signal() -> None:
    negative = [
        _candidate("span_0000", 0, anchor=-1.0, support=-1.0, contradiction=-1.0)
    ]
    assert native_zero_gate_details(negative)["anchor_gate_passed"] is False
    assert native_zero_gate_details(negative)["consensus_gate_passed"] is False

    anchor_only = [
        _candidate("span_0000", 0, anchor=1.0, support=-1.0, contradiction=-1.0)
    ]
    assert native_zero_gate_details(anchor_only)["anchor_gate_passed"] is True
    assert native_zero_gate_details(anchor_only)["consensus_gate_passed"] is False

    split_signal = [
        _candidate("span_0000", 0, anchor=1.0, support=-1.0, contradiction=-1.0),
        _candidate("span_0001", 1, anchor=-1.0, support=1.0, contradiction=-1.0),
    ]
    assert native_zero_gate_details(split_signal)["consensus_gate_passed"] is False

    same_span = deepcopy(split_signal)
    same_span[0]["raw_dynamic_role_scores"]["second_fact_or_bridge"] = 0.01
    assert native_zero_gate_details(same_span)["consensus_gate_passed"] is True


def test_guarded_selectors_share_gate_and_candidate_reuses_v43_selector() -> None:
    anchor_only = [
        _candidate("span_0000", 0, anchor=1.0, support=-1.0, contradiction=-1.0),
        _candidate("span_0001", 1, anchor=-1.0, support=-1.0, contradiction=-1.0),
    ]
    assert select_v50(
        anchor_only, NATIVE_ZERO_CONSENSUS_FRC_V50, token_budget=256
    ) == []
    assert select_v50(anchor_only, GATED_NON_FRC_METHODS[0], token_budget=256) == []
    assert select_v50(anchor_only, ANCHOR_GATE_FRC_V50, token_budget=256)

    passing = deepcopy(anchor_only)
    passing[0]["raw_dynamic_role_scores"]["first_fact"] = 0.01
    candidate_ids = [
        item["id"]
        for item in select_v50(
            passing, NATIVE_ZERO_CONSENSUS_FRC_V50, token_budget=256
        )
    ]
    frozen_v43_ids = [
        item["id"]
        for item in select_v50(passing, ADAPTIVE_ARGMAX, token_budget=256)
    ]
    assert candidate_ids == frozen_v43_ids
    assert candidate_ids == [
        item["id"]
        for item in select_v50(
            list(reversed(passing)),
            NATIVE_ZERO_CONSENSUS_FRC_V50,
            token_budget=256,
        )
    ]
    exact_zero = deepcopy(passing)
    exact_zero[0]["raw_dynamic_role_scores"]["first_fact"] = 0.0
    assert select_v50(
        exact_zero, NATIVE_ZERO_CONSENSUS_FRC_V50, token_budget=256
    ) == []


def test_locked_evaluation_is_deterministic_and_publishes_gold_free_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cnli, "BOOTSTRAP_RESAMPLES", 100)
    monkeypatch.setattr(cnli, "MINIMUM_CASES", 6)
    monkeypatch.setattr(cnli, "MINIMUM_STRATUM_CASES", 99)
    labels = (
        "Entailment",
        "Contradiction",
        "NotMentioned",
        "Entailment",
        "Contradiction",
        "NotMentioned",
    )
    gold = [
        {
            "case_id": f"case-{index}",
            "document_cluster": f"document-{index}",
            "label_group": label,
            "gold_candidate_ids": [] if label == "NotMentioned" else ["span_0000"],
            "candidate_unit_count": 3,
            "candidate_pool_quartile": "q1",
            "document_length_quartile": "q1",
            "document_type": "synthetic-nda",
            "candidate_ceiling_complete": True,
        }
        for index, label in enumerate(labels)
    ]
    scored = [
        _scored_row(
            f"case-{index}",
            consensus=label != "NotMentioned",
        )
        for index, label in enumerate(labels)
    ]
    query_summary = {"fallback_rate": 0.0, "rows": 6}
    first_report, first_evidence = evaluate_contractnli(
        gold, scored, query_summary, {"synthetic": True}
    )
    second_report, second_evidence = evaluate_contractnli(
        gold, scored, query_summary, {"synthetic": True}
    )
    assert first_report == second_report
    assert first_evidence == second_evidence
    assert first_report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert first_report["analysis"]["outcome"]["selector_adoption_authorized"] is False

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    evidence_path = tmp_path / "evidence.jsonl.gz"
    write_report(first_report, first_evidence, json_path, markdown_path, evidence_path)
    serialized = gzip.decompress(evidence_path.read_bytes()).decode()
    assert "Synthetic candidate" not in serialized
    assert "gold_candidate_ids" not in serialized
    assert "NO-GO/SHADOW" in markdown_path.read_text(encoding="utf-8")
