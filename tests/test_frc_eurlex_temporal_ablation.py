from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.eurlex_temporal_ablation import (
    EURLEX_METHODS,
    EURLEX_SOURCE_SCHEMA,
    _is_applicable,
    build_eurlex_ablation_report,
    build_eurlex_temporal_cases,
    extract_eurlex_html,
    load_eurlex_source,
    parse_eurlex_repeal_pairs,
    select_eurlex_evidence,
    select_eurlex_methods,
    sha256_text,
)
from research.frc_rag.public_evidence import load_eurlex_temporal_ablation


def _binding(value: str) -> dict[str, str]:
    return {"type": "literal", "value": value}


def _source_row(
    *,
    old: str,
    new: str,
    old_from: str,
    old_to: str,
    new_from: str,
    new_to: str = "9999-12-31",
    old_type: str = "REG",
    new_type: str = "REG",
) -> dict:
    authority = "http://publications.europa.eu/resource/authority/resource-type/"
    return {
        "old": _binding(f"http://example.test/{old}"),
        "oldcelex": _binding(old),
        "oldtype": _binding(authority + old_type),
        "olddateforce": _binding(old_from),
        "olddateend": _binding(old_to),
        "new": _binding(f"http://example.test/{new}"),
        "newcelex": _binding(new),
        "newtype": _binding(authority + new_type),
        "newdateforce": _binding(new_from),
        "newdateend": _binding(new_to),
    }


def _side(celex: str, title: str, valid_from: str, valid_to: str) -> dict:
    return {
        "celex": celex,
        "title": title,
        "text": (title + " energy reporting obligations evidence ") * 20,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "content_url": f"https://example.test/{celex}",
    }


def _pair(index: int, *, family: str = "regulation") -> dict:
    return {
        "pair_id": f"OLD{index}--NEW{index}",
        "family": family,
        "old": _side(
            f"OLD{index}",
            f"Earlier energy reporting measure {index}",
            "2018-01-01",
            "2023-12-31",
        ),
        "new": _side(
            f"NEW{index}",
            f"Replacement energy reporting measure {index}",
            "2024-01-01",
            "9999-12-31",
        ),
    }


def test_html_parser_supports_formex_and_legacy_eurlex_layouts() -> None:
    modern = """
    <html><body><div class="eli-main-title">
    <p class="oj-doc-ti">Directive (EU) 2024/1</p>
    <p class="oj-doc-ti">on energy reporting</p></div>
    <p class="oj-normal">Article 1 establishes reporting duties.</p>
    <p class="oj-normal">This text is deliberately long enough for extraction.</p>
    </body></html>
    """ + "".join(
        f"<p class='oj-normal'>Additional legal evidence paragraph {index}.</p>"
        for index in range(10)
    )
    legacy = """
    <html><head><meta name="DC.description" content="Legacy Regulation on reporting"></head>
    <body><p>Legacy legal paragraph with binding reporting requirements.</p>
    </body></html>
    """ + "".join(
        f"<p>Additional legacy legal evidence paragraph {index}.</p>"
        for index in range(10)
    )
    modern_result = extract_eurlex_html(modern)
    legacy_result = extract_eurlex_html(legacy)
    assert modern_result["title"] == "Directive (EU) 2024/1 on energy reporting"
    assert legacy_result["title"] == "Legacy Regulation on reporting"
    assert "Article 1" in modern_result["text"]
    assert "binding reporting requirements" in legacy_result["text"]


def test_sparql_parser_keeps_only_unique_adjacent_boundaries() -> None:
    valid = _source_row(
        old="32020R0001",
        new="32024R0002",
        old_from="2020-01-01",
        old_to="2024-05-31",
        new_from="2024-06-01",
    )
    valid_later_entry = _source_row(
        old="32020R0001",
        new="32024R0002",
        old_from="2021-01-01",
        old_to="2024-05-31",
        new_from="2024-06-01",
    )
    ambiguous_one = _source_row(
        old="32020D0003",
        new="32024D0004",
        old_from="2020-01-01",
        old_to="2024-06-29",
        new_from="2024-06-30",
        new_type="DEC",
        old_type="DEC",
    )
    ambiguous_two = _source_row(
        old="32020D0003",
        new="32024D0004",
        old_from="2020-01-01",
        old_to="2024-07-30",
        new_from="2024-07-31",
        new_type="DEC",
        old_type="DEC",
    )
    payload = {
        "results": {
            "bindings": [valid, valid_later_entry, ambiguous_one, ambiguous_two]
        }
    }
    pairs = parse_eurlex_repeal_pairs(payload)
    assert len(pairs) == 1
    assert pairs[0]["pair_id"] == "32020R0001--32024R0002"
    assert pairs[0]["old"]["valid_from"] == "2020-01-01"
    assert pairs[0]["old"]["valid_to"] == "2024-05-31"
    assert pairs[0]["new"]["valid_from"] == "2024-06-01"


def test_temporal_cases_keep_labels_out_of_questions_and_dates_applicable() -> None:
    cases = build_eurlex_temporal_cases(
        [_pair(1), _pair(2)], distractor_pairs=1
    )
    assert len(cases) == 4
    for case in cases:
        assert case["gold_evidence_id"] not in case["question"]
        gold = next(
            item
            for item in case["candidates"]
            if item["id"] == case["gold_evidence_id"]
        )
        counterpart = next(
            item
            for item in case["candidates"]
            if item["id"] == case["superseded_pair_evidence_id"]
        )
        assert _is_applicable(case, gold)
        assert not _is_applicable(case, counterpart)
        assert len(case["candidates"]) == 4


def _scored_cases() -> list[dict]:
    cases = build_eurlex_temporal_cases(
        [_pair(1), _pair(2)], distractor_pairs=1
    )
    for case in cases:
        for candidate in case["candidates"]:
            if candidate["id"] == case["superseded_pair_evidence_id"]:
                score = 1.0
            elif candidate["id"] == case["gold_evidence_id"]:
                score = 0.8
            else:
                score = 0.1
            candidate["scores"] = {"bm25": score, "cross_encoder": score}
    return cases


def test_applicability_ablation_removes_only_date_filter() -> None:
    case = _scored_cases()[0]
    full = select_eurlex_evidence(case, "frc_full")
    fair = select_eurlex_evidence(
        case, "applicability_filtered_cross_encoder_top1"
    )
    without = select_eurlex_evidence(case, "w/o_applicability")
    assert [item["id"] for item in full] == [case["gold_evidence_id"]]
    assert [item["id"] for item in fair] == [case["gold_evidence_id"]]
    assert [item["id"] for item in without] == [
        case["superseded_pair_evidence_id"]
    ]


def test_report_uses_fair_filtered_baseline_and_preserves_no_go(
    tmp_path: Path,
) -> None:
    cases = _scored_cases()
    selected = select_eurlex_methods(cases)
    manifest_path = tmp_path / "manifest.json"
    scored_path = tmp_path / "scored.jsonl"
    manifest = {
        "schema_version": EURLEX_SOURCE_SCHEMA,
        "metadata": {
            "snapshot_date": "2026-07-14",
            "sparql_query_sha256": "a" * 64,
        },
        "pairs": [_pair(1), _pair(2)],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    scored_path.write_text("{}\n", encoding="utf-8")
    report = build_eurlex_ablation_report(
        cases=cases,
        selected_rows=selected,
        config={
            "reranker_model": "test-reranker",
            "top_k": 1,
            "token_budget": 512,
            "distractor_pairs": 1,
            "seed": 20260714,
        },
        source_manifest=manifest,
        source_manifest_path=manifest_path,
        scored_path=scored_path,
    )
    assert set(report["aggregates"]) == set(EURLEX_METHODS)
    assert report["aggregates"]["frc_full"]["exact_evidence_accuracy"] == 1.0
    assert report["aggregates"]["w/o_applicability"][
        "exact_evidence_accuracy"
    ] == 0.0
    assert (
        report["strongest_baseline_by_exact_evidence_accuracy"]
        == "applicability_filtered_cross_encoder_top1"
    )
    assert report["decision"][
        "full_exact_gain_over_strongest_baseline_at_least_0_05"
    ] is False
    assert report["decision"]["gate_2"] == "NO-GO"


def test_source_loader_fails_closed_on_canonical_text_drift(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    pair = _pair(1)
    manifest_pair = {"pair_id": pair["pair_id"], "family": pair["family"]}
    for side in ("old", "new"):
        source = pair[side]
        text_hash = sha256_text(source["text"])
        raw_hash = "b" * 64
        document = {
            "celex": source["celex"],
            "title": source["title"],
            "text": source["text"],
            "raw_html_sha256": raw_hash,
            "canonical_text_sha256": text_hash,
        }
        (documents / f"{source['celex']}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        manifest_pair[side] = {
            **source,
            "raw_html_sha256": raw_hash,
            "canonical_text_sha256": text_hash,
        }
    manifest = {
        "schema_version": EURLEX_SOURCE_SCHEMA,
        "metadata": {},
        "pairs": [manifest_pair],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, loaded = load_eurlex_source(manifest_path, documents)
    assert loaded[0]["old"]["text"] == pair["old"]["text"]
    changed = json.loads(
        (documents / f"{pair['old']['celex']}.json").read_text(encoding="utf-8")
    )
    changed["text"] += " tampered"
    (documents / f"{pair['old']['celex']}.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source drift"):
        load_eurlex_source(manifest_path, documents)


def test_public_loader_reaggregates_committed_eurlex_ablation(
    tmp_path: Path,
) -> None:
    source = Path(
        "output/rag_evaluation/eurlex_temporal_ablation/"
        "eurlex_temporal_ablation.json"
    )
    loaded = load_eurlex_temporal_ablation(source)

    assert loaded["status"] == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
    assert loaded["metadata"]["pair_count"] == 30
    assert loaded["metadata"]["case_count"] == 60
    assert loaded["strongest_baseline"] == (
        "applicability_filtered_cross_encoder_top1"
    )
    assert loaded["decision"]["gate_2"] == "NO-GO"

    report = json.loads(source.read_text(encoding="utf-8"))
    report["aggregates"]["frc_full"]["exact_evidence_accuracy"] = 0.123456
    case_file = report["case_results_artifact"]["file"]
    (tmp_path / case_file).write_bytes((source.parent / case_file).read_bytes())
    artifact = tmp_path / "tampered-eurlex-ablation.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="EUR-Lex aggregate mismatch"):
        load_eurlex_temporal_ablation(artifact)
