from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_ROLES,
    STATIC_ROLES,
)
from research.frc_rag.scifact_dynamic_atomic_roles import (
    PROTOCOL_SHA256,
    CorpusBM25Index,
    build_gold_rows,
    evaluate_scifact,
    parse_corpus,
    prepare_blind_cases,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def __init__(self) -> None:
        self.text_by_id: dict[int, str] = {}
        self.id_by_text: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result = []
        for token in text.split():
            if token not in self.id_by_text:
                token_id = len(self.id_by_text) + 1
                self.id_by_text[token] = token_id
                self.text_by_id[token_id] = token
            result.append(self.id_by_text[token])
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(self.text_by_id[token_id] for token_id in token_ids)


def _corpus(count: int = 25) -> list[dict]:
    return [
        {
            "doc_id": index,
            "title": f"Document {index}",
            "abstract": [
                "alpha special evidence" if index == 1 else "unrelated material"
            ],
            "structured": False,
        }
        for index in range(1, count + 1)
    ]


def _claim(
    claim_id: int,
    *,
    doc_id: int | None,
    label: str = "SUPPORT",
    text: str = "alpha special claim",
) -> dict:
    evidence = {}
    if doc_id is not None:
        evidence[str(doc_id)] = [{"label": label, "sentences": [0]}]
    return {"id": claim_id, "claim": text, "evidence": evidence}


def _scored_row(case_id: str, candidate_count: int = 3) -> dict:
    candidates = [
        {
            "id": f"{case_id}::s{index:04d}::c000",
            "source_id": f"s{index:04d}",
            "token_count": 50,
        }
        for index in range(candidate_count)
    ]
    scores = []
    for index, candidate in enumerate(candidates):
        value = 1.0 - index / candidate_count
        scores.append(
            {
                "id": candidate["id"],
                "scores": {
                    "bm25": value,
                    "dense": value,
                    "hybrid": value,
                    "cross_encoder": value,
                },
                "static_role_scores": {
                    role: value for role in STATIC_ROLES
                },
                "dynamic_role_scores": {
                    role: (index + role_index + 1) / 10
                    for role_index, role in enumerate(DYNAMIC_ROLES)
                },
            }
        )
    return {
        "schema_version": "frc-scifact-dynamic-atomic-roles-v42",
        "dataset_id": "scifact_claims_dev_v42",
        "capability": "scientific_claim_document_evidence_selection",
        "id": case_id,
        "candidates": candidates,
        "candidate_scores": scores,
        "gold_fields_visible_to_scorer": False,
    }


def test_scifact_v42_protocol_is_frozen() -> None:
    path = (
        ROOT
        / "docs/progressive_upgrade/"
        "scifact_dynamic_atomic_roles_protocol_v42.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROTOCOL_SHA256
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["research_boundary"]["scifact_data_downloaded_before_registration"] is False
    assert value["decision_and_stopping_rule"]["no_scifact_tuning_after_any_result"] is True


def test_corpus_bm25_is_corpus_wide_stable_and_preserves_query_multiplicity() -> None:
    documents = parse_corpus(
        [
            {"doc_id": 2, "title": "Beta", "abstract": ["plain"], "structured": False},
            {"doc_id": 1, "title": "Alpha", "abstract": ["alpha alpha"], "structured": False},
            {"doc_id": 3, "title": "Gamma", "abstract": ["plain"], "structured": False},
        ]
    )
    index = CorpusBM25Index(documents)
    assert index.rank("alpha", depth=3) == [1, 2, 3]
    assert index.rank("alpha alpha", depth=3) == [1, 2, 3]
    assert index.rank("unknown", depth=3) == [1, 2, 3]


def test_blind_preparation_excludes_documentation_examples_and_empty_evidence() -> None:
    claims = [
        _claim(3, doc_id=1),
        _claim(4, doc_id=25),
        _claim(6, doc_id=None),
    ]
    prepared, census = prepare_blind_cases(
        _corpus(), claims, FakeTokenizer(), minimum_cases=1
    )
    assert [row["id"] for row in prepared] == ["scifact-v42::4"]
    assert census["documentation_examples_excluded"] == 1
    assert census["empty_evidence_structurally_non_evaluable"] == 1
    serialized = json.dumps(prepared, ensure_ascii=False)
    assert '"doc_id":' not in serialized
    assert '"evidence":' not in serialized
    assert "SUPPORT" not in serialized

    gold = build_gold_rows(_corpus(), claims, prepared, minimum_cases=1)
    assert gold[0]["candidate_ceiling"] == 0.0
    assert gold[0]["gold_sources"] == []


def test_missing_gold_document_fails_instead_of_substituting() -> None:
    with pytest.raises(ValueError, match="missing documents"):
        prepare_blind_cases(
            _corpus(5),
            [_claim(4, doc_id=99)],
            FakeTokenizer(),
            minimum_cases=1,
        )


def test_mixed_label_and_multiple_gold_documents_are_reconstructed_post_score() -> None:
    claim = _claim(4, doc_id=1)
    claim["evidence"]["2"] = [
        {"label": "CONTRADICT", "sentences": [0]}
    ]
    prepared, _ = prepare_blind_cases(
        _corpus(), [claim], FakeTokenizer(), minimum_cases=1
    )
    gold = build_gold_rows(_corpus(), [claim], prepared, minimum_cases=1)
    assert gold[0]["label"] == "MIXED"
    assert gold[0]["gold_document_count"] == 2
    assert gold[0]["candidate_ceiling_complete"] is True


def test_evaluation_has_frozen_thresholds_and_exports_no_raw_text() -> None:
    case_id = "scifact-v42::4"
    scored = [_scored_row(case_id)]
    gold = [
        {
            "case_id": case_id,
            "label": "SUPPORT",
            "gold_document_count": 1,
            "gold_sources": ["s0000"],
            "candidate_ceiling": 1.0,
            "candidate_ceiling_complete": True,
        }
    ]
    query_summary = {
        "rows": 1,
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "fallback_reasons": {},
        "query_length_codepoints": {"minimum": 3, "mean": 4.0, "maximum": 5},
        "gold_fields_visible_to_generator": False,
    }
    report, evidence = evaluate_scifact(
        gold, scored, query_summary, resamples=20
    )
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["analysis"]["outcome"]["scifact_reuse_for_tuning_authorized"] is False
    assert report["analysis"]["support_checks"]["no_forbidden_field_leak"] is True
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert '"claim"' not in serialized
    assert '"doc_id"' not in serialized

    output_path = ROOT / "output/test_artifacts/scifact_v42"
    paths = write_report(report, evidence, output_path)
    first = {name: path.read_bytes() for name, path in paths.items()}
    paths = write_report(report, evidence, output_path)
    second = {name: path.read_bytes() for name, path in paths.items()}
    assert first == second
    with gzip.open(paths["evidence"], "rt", encoding="utf-8") as handle:
        exported = json.loads(handle.readline())
    assert exported["raw_document_ids_exported"] is False


def test_scored_cache_rejects_gold_label() -> None:
    row = _scored_row("scifact-v42::4")
    row["label"] = "SUPPORT"
    with pytest.raises(ValueError, match="forbidden"):
        evaluate_scifact(
            [
                {
                    "case_id": "scifact-v42::4",
                    "label": "SUPPORT",
                    "gold_document_count": 1,
                    "gold_sources": ["s0000"],
                    "candidate_ceiling": 1.0,
                    "candidate_ceiling_complete": True,
                }
            ],
            [row],
            {"fallback_rate": 0.0},
            resamples=5,
        )
