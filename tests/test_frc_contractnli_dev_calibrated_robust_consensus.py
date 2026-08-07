from __future__ import annotations

import gzip
import json
import math
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from research.frc_rag.contractnli_dev_calibrated_robust_consensus import (
    FOLD_COUNT,
    MAX_CASES_PER_DOCUMENT,
    TARGET_CASES,
    TARGET_PER_LABEL,
    aggregate_records,
    build_candidate_coverage,
    build_deterministic_queries,
    calibrate_development,
    fit_threshold,
    fold_for_document,
    prepare_blind_cases,
    read_split_source,
    robust_consensus_details,
    select_balanced_sample,
    threshold_candidates,
    validate_protocol,
    validate_query_cache,
    write_development_report,
)


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _source(document_count: int = 40) -> dict:
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


def _raw_candidate(identifier: str, anchor: float, support: float, oppose: float) -> dict:
    return {
        "id": identifier,
        "raw_dynamic_role_scores": {
            "anchor": anchor,
            "first_fact": support,
            "second_fact_or_bridge": oppose,
        },
    }


def _metric(*, utility: float, evidence: float | None, abstained: bool) -> dict:
    return {
        "utility_f1": utility,
        "evidence_f1": evidence,
        "precision": evidence,
        "recall": evidence,
        "complete_recall": bool(evidence),
        "gold_unit_hits": int(bool(evidence)),
        "selected_unit_count": 0 if abstained else 1,
        "selected_token_cost": 0 if abstained else 8,
        "abstained": abstained,
    }


def _development_record(index: int) -> dict:
    label = ("Entailment", "Contradiction", "NotMentioned")[index % 3]
    missing = label == "NotMentioned"
    document_index = index // MAX_CASES_PER_DOCUMENT
    return {
        "case_id": f"case-{index:03d}",
        "document_cluster": f"document-commitment-{document_index:02d}",
        "development_fold": document_index % FOLD_COUNT,
        "label_group": label,
        "candidate_unit_count": 4,
        "candidate_pool_quartile": f"q{index % 4 + 1}",
        "document_length_quartile": f"q{index % 4 + 1}",
        "document_type": "synthetic-nda",
        "candidate_ceiling_complete": True,
        "robust_score": -2.0 if missing else 2.0,
        "robust_score_hex": (-2.0 if missing else 2.0).hex(),
        "configurations": {
            str(budget): {
                "pass_metrics": _metric(
                    utility=0.0 if missing else 1.0,
                    evidence=None if missing else 1.0,
                    abstained=False,
                ),
                "abstain_metrics": _metric(
                    utility=1.0 if missing else 0.0,
                    evidence=None if missing else 0.0,
                    abstained=True,
                ),
            }
            for budget in (256, 512, 1024)
        },
    }


def test_protocol_freezes_development_boundary_formula_and_split_roles() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_protocol_v51.json"
    )
    boundary = protocol["development_boundary"]
    assert boundary["official_dev_member_content_opened_before_v51_registration"] is False
    assert boundary["official_train_member_content_opened_before_v51_registration"] is False
    assert boundary["official_test_split_permanently_excluded_from_v51"] is True
    assert protocol["development_scope"]["target_cases"] == TARGET_CASES
    assert protocol["confirmation_scope"]["split"].startswith("official train")
    assert protocol["robust_consensus_score"]["case_score"].startswith("maximum")


def test_archive_reader_opens_only_requested_registered_split(tmp_path: Path) -> None:
    dev_archive = tmp_path / "dev-only.zip"
    with zipfile.ZipFile(dev_archive, mode="w") as archive:
        archive.writestr("contract-nli/train.json", b"not-json-train")
        archive.writestr("contract-nli/test.json", b"not-json-test")
        archive.writestr("contract-nli/dev.json", json.dumps(_source(1)).encode())
    assert read_split_source(dev_archive, "dev") == _source(1)

    train_archive = tmp_path / "train-only.zip"
    with zipfile.ZipFile(train_archive, mode="w") as archive:
        archive.writestr("contract-nli/dev.json", b"not-json-dev")
        archive.writestr("contract-nli/test.json", b"not-json-test")
        archive.writestr("contract-nli/train.json", json.dumps(_source(1)).encode())
    assert read_split_source(train_archive, "train") == _source(1)
    with pytest.raises(ValueError, match="only dev or train"):
        read_split_source(train_archive, "test")


def test_balanced_development_sample_is_order_invariant_and_capped() -> None:
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
    assert len(selected) == TARGET_CASES
    assert summary["selected_label_counts"] == {
        "Entailment": TARGET_PER_LABEL,
        "Contradiction": TARGET_PER_LABEL,
        "NotMentioned": TARGET_PER_LABEL,
    }
    assert summary["maximum_cases_per_document"] == MAX_CASES_PER_DOCUMENT
    assert summary["selected_documents"] == 40


def test_blind_cache_and_deterministic_queries_export_no_gold() -> None:
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
        '"development_fold"',
    ):
        assert forbidden not in serialized
    assert all("document_id" not in row and "hypothesis_key" not in row for row in maps)
    queries = build_deterministic_queries(prepared)
    assert validate_query_cache(prepared, queries)["fallback_rate"] == 0.0


def test_robust_consensus_is_same_span_permutation_and_affine_invariant() -> None:
    candidates = [
        _raw_candidate("span_0000", 1.0, 1.0, 1.0),
        _raw_candidate("span_0001", 2.0, 2.0, 2.0),
        _raw_candidate("span_0002", 3.0, 3.0, 3.0),
        _raw_candidate("span_0003", 4.0, 4.0, 4.0),
    ]
    original = robust_consensus_details(candidates)
    assert original["winning_candidate_id"] == "span_0003"
    assert original["case_score"] == pytest.approx(1.0)
    assert robust_consensus_details(list(reversed(candidates)))["case_score"] == pytest.approx(
        original["case_score"]
    )
    transformed = deepcopy(candidates)
    for candidate in transformed:
        candidate["raw_dynamic_role_scores"] = {
            role: value * 3.0 + 7.0
            for role, value in candidate["raw_dynamic_role_scores"].items()
        }
    assert robust_consensus_details(transformed)["case_score"] == pytest.approx(
        original["case_score"]
    )

    split_signal = [
        _raw_candidate("anchor", 4.0, 1.0, 1.0),
        _raw_candidate("polarity", 1.0, 4.0, 4.0),
        _raw_candidate("low-1", 2.0, 2.0, 2.0),
        _raw_candidate("low-2", 3.0, 3.0, 3.0),
    ]
    same_span = deepcopy(split_signal)
    same_span[0]["raw_dynamic_role_scores"]["first_fact"] = 4.0
    same_span[0]["raw_dynamic_role_scores"]["second_fact_or_bridge"] = 4.0
    assert robust_consensus_details(same_span)["case_score"] > robust_consensus_details(
        split_signal
    )["case_score"]


def test_threshold_grid_and_fit_are_deterministic_and_feasible() -> None:
    records = [_development_record(index) for index in range(30)]
    candidates = threshold_candidates(records)
    assert len(candidates) == 3
    assert candidates[0] < -2.0
    assert candidates[1] == 0.0
    assert candidates[2] > 2.0
    fit = fit_threshold(records)
    assert fit["threshold_hex"] == 0.0.hex()
    assert fit["aggregate"]["evidence_or_abstention_macro_f1"] == 1.0
    assert fit["aggregate"]["evidence_bearing_macro_f1"] == 1.0
    assert fit["aggregate"]["not_mentioned_abstention_accuracy"] == 1.0
    assert fit["aggregate"]["abstention_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert fit == fit_threshold(list(reversed(records)))


def test_document_fold_assignment_is_stable_and_bounded() -> None:
    first = [fold_for_document(f"document-{index}") for index in range(100)]
    second = [fold_for_document(f"document-{index}") for index in range(100)]
    assert first == second
    assert set(first) == set(range(FOLD_COUNT))
    assert all(0 <= fold < FOLD_COUNT for fold in first)


def test_development_coverage_requires_exact_registered_structure() -> None:
    records = [_development_record(index) for index in range(TARGET_CASES)]
    gold = [
        {
            **row,
            "gold_candidate_ids": []
            if row["label_group"] == "NotMentioned"
            else ["span_0000"],
        }
        for row in records
    ]
    sampling = {
        "selected_cases": TARGET_CASES,
        "maximum_cases_per_document": MAX_CASES_PER_DOCUMENT,
    }
    report = build_candidate_coverage(gold, sampling)
    assert report["minimum_cases_and_ceiling_checks_passed"] is True
    assert report["development_documents"] == 40
    assert set(report["development_fold_counts"]) == {"0", "1", "2", "3", "4"}
    assert report["label_groups"] == {
        "Entailment": TARGET_PER_LABEL,
        "Contradiction": TARGET_PER_LABEL,
        "NotMentioned": TARGET_PER_LABEL,
    }


def test_oof_calibration_is_deterministic_and_gold_free_in_published_evidence(
    tmp_path: Path,
) -> None:
    records = [_development_record(index) for index in range(TARGET_CASES)]
    query_summary = {"fallback_rate": 0.0, "rows": TARGET_CASES}
    first_report, first_evidence = calibrate_development(
        records, query_summary, {"synthetic": True}
    )
    second_report, second_evidence = calibrate_development(
        list(reversed(records)), query_summary, {"synthetic": True}
    )
    assert first_report == second_report
    assert sorted(first_evidence, key=lambda row: row["case_id"]) == sorted(
        second_evidence, key=lambda row: row["case_id"]
    )
    outcome = first_report["analysis"]["outcome"]
    assert outcome["train_open_authorized"] is True
    assert outcome["selector_adoption_authorized"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"
    assert first_report["analysis"]["oof_utility_gain_vs_ungated_v49"] == pytest.approx(
        1 / 3, abs=1e-6
    )
    assert aggregate_records(records, threshold=None)["abstention_rate"] == 0.0

    result = tmp_path / "development.json"
    markdown = tmp_path / "development.md"
    evidence = tmp_path / "development.jsonl.gz"
    write_development_report(first_report, first_evidence, result, markdown, evidence)
    published = gzip.decompress(evidence.read_bytes()).decode()
    assert "gold_candidate_ids" not in published
    assert "Registered synthetic hypothesis" not in published
    assert "Alpha clause" not in published
    assert "NO-GO/SHADOW" in markdown.read_text(encoding="utf-8")
    assert math.isfinite(float.fromhex(outcome["final_threshold_hex"]))
