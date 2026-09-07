from __future__ import annotations

import gzip
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import research.frc_rag.contractnli_rank_concurrence_confirmation as rank52
from research.frc_rag.hover_dynamic_atomic_roles import STATIC_ROLES
from research.frc_rag.contractnli_rank_concurrence_confirmation import (
    CANDIDATE,
    GATED_NON_FRC_METHODS,
    build_candidate_coverage,
    build_deterministic_queries,
    evaluate_confirmation,
    prepare_blind_cases,
    rank_concurrence_gate_details,
    read_train_source,
    select_balanced_sample,
    select_v52,
    validate_protocol,
    validate_query_cache,
    write_report,
)


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _source(document_count: int = 100) -> dict:
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
        annotations = {
            f"hypothesis-{index}": {
                "choice": choice,
                "spans": [] if choice == "NotMentioned" else [index % 2],
            }
            for index, choice in enumerate(choices)
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
) -> dict:
    rank = 1.0 - index * 0.1
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
        "dynamic_role_scores": {
            "anchor": rank,
            "first_fact": rank - 0.01,
            "second_fact_or_bridge": rank - 0.02,
            "counterevidence": rank - 0.03,
        },
        "raw_dynamic_role_scores": {
            "anchor": anchor,
            "first_fact": support,
            "second_fact_or_bridge": contradiction,
            "counterevidence": 0.5,
        },
    }


def _candidates(*, passing: bool) -> list[dict]:
    if passing:
        return [
            _candidate("span_0000", 0, anchor=4.0, support=4.0, contradiction=2.0),
            _candidate("span_0001", 1, anchor=3.0, support=2.0, contradiction=4.0),
            _candidate("span_0002", 2, anchor=2.0, support=3.0, contradiction=3.0),
        ]
    return [
        _candidate("span_0000", 0, anchor=4.0, support=2.0, contradiction=2.0),
        _candidate("span_0001", 1, anchor=3.0, support=4.0, contradiction=3.0),
        _candidate("span_0002", 2, anchor=2.0, support=3.0, contradiction=4.0),
    ]


def _scored_row(case_id: str, *, passing: bool) -> dict:
    candidates = _candidates(passing=passing)
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


def test_protocol_freezes_untouched_train_and_parameter_free_rank_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_protocol_v52.json"
    )
    boundary = protocol["development_history_boundary"]
    assert boundary["official_train_member_content_opened_before_v52_registration"] is False
    assert boundary["v52_has_no_learned_threshold_or_dataset_fitted_parameter"] is True
    assert boundary[
        "v50_test_and_v51_dev_permanently_excluded_from_v52_fitting_selection_and_evaluation"
    ] is True
    gate = protocol["rank_concurrence_gate"]
    assert gate["learned_threshold_temperature_margin_prior_or_candidate_count_feature"] is False
    assert gate["selector_after_pass"] == "unchanged low_core_divergence_guarded_frc_v49"


def test_archive_reader_opens_only_train_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "contract-nli.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("contract-nli/dev.json", b"not-json-dev")
        archive.writestr("contract-nli/test.json", b"not-json-test")
        archive.writestr("contract-nli/train.json", json.dumps(_source(1)).encode())
    assert read_train_source(archive_path) == _source(1)


def test_balanced_train_sample_is_order_invariant_and_document_capped() -> None:
    source = _source()
    selected, summary = select_balanced_sample(source)
    reversed_source = {
        "labels": dict(reversed(list(source["labels"].items()))),
        "documents": list(reversed(source["documents"])),
    }
    second, second_summary = select_balanced_sample(reversed_source)
    keys = [(row["document_id"], row["hypothesis_key"]) for row in selected]
    second_keys = [(row["document_id"], row["hypothesis_key"]) for row in second]
    assert keys == second_keys
    assert summary == second_summary
    assert summary["selected_cases"] == 600
    assert summary["selected_documents"] == 100
    assert summary["selected_label_counts"] == {
        "Entailment": 200,
        "Contradiction": 200,
        "NotMentioned": 200,
    }
    assert summary["maximum_cases_per_document"] == 6


def test_blind_cache_and_queries_export_no_gold() -> None:
    selected, _ = select_balanced_sample(_source(1), target_per_label=2)
    prepared, maps, _ = prepare_blind_cases(selected, Tokenizer())
    serialized = json.dumps(prepared, ensure_ascii=False)
    for forbidden in (
        '"annotation_sets"',
        '"choice"',
        '"document_id"',
        '"gold_candidate_ids"',
        '"label"',
        '"spans"',
    ):
        assert forbidden not in serialized
    assert all("document_id" not in row and "hypothesis_key" not in row for row in maps)
    queries = build_deterministic_queries(prepared)
    assert validate_query_cache(prepared, queries)["fallback_rate"] == 0.0


def test_rank_concurrence_requires_same_top_span_and_is_invariant() -> None:
    passing = _candidates(passing=True)
    failing = _candidates(passing=False)
    assert rank_concurrence_gate_details(passing)["rank_concurrence_passed"] is True
    assert rank_concurrence_gate_details(failing)["rank_concurrence_passed"] is False
    assert rank_concurrence_gate_details(list(reversed(passing))) == (
        rank_concurrence_gate_details(passing)
    )

    transformed = deepcopy(passing)
    transforms = {
        "anchor": (3.0, 7.0),
        "first_fact": (2.0, -11.0),
        "second_fact_or_bridge": (5.0, 0.25),
    }
    for candidate in transformed:
        for role, (scale, shift) in transforms.items():
            value = candidate["raw_dynamic_role_scores"][role]
            candidate["raw_dynamic_role_scores"][role] = scale * value + shift
    assert rank_concurrence_gate_details(transformed) == rank_concurrence_gate_details(
        passing
    )


def test_rank_gate_ties_and_shared_gate_are_deterministic() -> None:
    tied = _candidates(passing=False)
    for candidate in tied:
        candidate["raw_dynamic_role_scores"]["anchor"] = 1.0
        candidate["raw_dynamic_role_scores"]["first_fact"] = 1.0
    details = rank_concurrence_gate_details(tied)
    assert details["anchor_winner_id"] == "span_0000"
    assert details["support_winner_id"] == "span_0000"
    assert details["rank_concurrence_passed"] is True

    failing = _candidates(passing=False)
    assert select_v52(failing, CANDIDATE, token_budget=256) == []
    assert select_v52(failing, GATED_NON_FRC_METHODS[0], token_budget=256) == []
    passing = _candidates(passing=True)
    assert select_v52(passing, CANDIDATE, token_budget=256)
    assert select_v52(list(reversed(passing)), CANDIDATE, token_budget=256) == (
        select_v52(passing, CANDIDATE, token_budget=256)
    )


def test_candidate_coverage_requires_exact_registered_confirmation() -> None:
    labels = ("Entailment", "Contradiction", "NotMentioned")
    gold = []
    for index in range(600):
        label = labels[index % 3]
        gold.append(
            {
                "case_id": f"case-{index}",
                "document_cluster": f"doc-{index // 6}",
                "label_group": label,
                "gold_candidate_ids": []
                if label == "NotMentioned"
                else ["span_0000"],
                "candidate_unit_count": 3,
                "candidate_pool_quartile": "q1",
                "document_length_quartile": "q1",
                "document_type": "synthetic-nda",
                "candidate_ceiling_complete": True,
            }
        )
    report = build_candidate_coverage(
        gold,
        {"selected_cases": 600, "maximum_cases_per_document": 6},
    )
    assert report["all_checks_passed"] is True
    assert report["documents"] == 100
    assert report["candidate_ceiling_complete_rate_on_evidence_cases"] == 1.0
    assert report["not_mentioned_official_empty_evidence_rate"] == 1.0


def test_confirmation_evaluation_is_deterministic_and_publishes_no_gold(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rank52, "TARGET_CASES", 30)
    monkeypatch.setattr(rank52, "MINIMUM_STRATUM_CASES", 99)
    monkeypatch.setattr(rank52, "BOOTSTRAP_RESAMPLES", 100)
    labels = ("Entailment", "Contradiction", "NotMentioned")
    gold = []
    scored = []
    for index in range(30):
        label = labels[index % 3]
        gold.append(
            {
                "case_id": f"case-{index:02d}",
                "document_cluster": f"doc-{index // 3:02d}",
                "label_group": label,
                "gold_candidate_ids": []
                if label == "NotMentioned"
                else ["span_0000"],
                "candidate_unit_count": 3,
                "candidate_pool_quartile": "q1",
                "document_length_quartile": "q1",
                "document_type": "synthetic-nda",
                "candidate_ceiling_complete": True,
            }
        )
        scored.append(
            _scored_row(f"case-{index:02d}", passing=label != "NotMentioned")
        )
    first_report, first_evidence = evaluate_confirmation(
        gold, scored, {"fallback_rate": 0.0, "rows": 30}, {"synthetic": True}
    )
    second_report, second_evidence = evaluate_confirmation(
        gold, scored, {"fallback_rate": 0.0, "rows": 30}, {"synthetic": True}
    )
    assert first_report == second_report
    assert first_evidence == second_evidence
    outcome = first_report["analysis"]["outcome"]
    assert outcome["selector_adoption_authorized"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"
    assert first_report["analysis"]["aggregates"][CANDIDATE][
        "not_mentioned_abstention_accuracy"
    ] == 1.0

    result = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    evidence = tmp_path / "cases.jsonl.gz"
    write_report(first_report, first_evidence, result, markdown, evidence)
    published = gzip.decompress(evidence.read_bytes()).decode()
    assert "gold_candidate_ids" not in published
    assert "Synthetic candidate" not in published
    assert "Registered synthetic hypothesis" not in published
    assert "NO-GO/SHADOW" in markdown.read_text(encoding="utf-8")
