from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    PROTOCOL_SHA256,
    STATIC_RANK,
    STATIC_ROLES,
    build_partitions,
    evaluate_confirmation,
    evaluate_mechanism_gate,
    merge_scored_candidates,
    parse_atomic_queries,
    partition_commitments,
    select_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate(index: int, *, source: int | None = None) -> dict:
    value = (10 - index) / 10
    return {
        "id": f"c{index:02d}",
        "source_id": f"s{source if source is not None else index:02d}",
        "token_count": 50,
        "scores": {
            "bm25": value,
            "dense": value,
            "hybrid": value,
            "cross_encoder": value,
        },
        "static_role_scores": {role: value for role in STATIC_ROLES},
        "dynamic_role_scores": {
            role: (10 - ((index - role_index * 2) % 10)) / 10
            for role_index, role in enumerate(DYNAMIC_ROLES)
        },
    }


def _scored_row(case_id: str = "hover-v41::case") -> dict:
    candidates = [_candidate(index) for index in range(10)]
    return {
        "schema_version": "frc-hover-dynamic-atomic-roles-v41",
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
            }
            for row in candidates
        ],
        "gold_fields_visible_to_scorer": False,
    }


def test_v41_protocol_and_v40_closure_are_frozen() -> None:
    protocol = ROOT / "docs/progressive_upgrade/hover_dynamic_atomic_roles_protocol_v41.json"
    closure = ROOT / "docs/progressive_upgrade/hover_dynamic_atomic_roles_v40_closure.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    closure_value = json.loads(closure.read_text(encoding="utf-8"))
    assert closure_value["status"] == "REGISTRATION_BOUNDARY_VIOLATED_BEFORE_SCORING"
    assert closure_value["mandatory_action"]["v40_pilot_or_confirmation_may_be_scored"] is False


def test_v39_mechanism_diagnostic_pins_role_collapse() -> None:
    path = (
        ROOT
        / "output/rag_evaluation/hover_verification_roles/"
        "hover_role_mechanism_diagnostic.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["interpretation"]["status"] == "STATIC_ROLE_SIGNAL_COLLAPSE_OBSERVED"
    assert report["role_signal"]["rank_correlations"]["generic"]["mean"] == 0.947723
    assert (
        report["role_signal"]["rank_correlations"]["verification"]["mean"]
        == 0.946191
    )
    assert report["selection_collapse"]["1500"]["verification_cross_set_jaccard"] == 0.99825
    assert report["metadata"]["gold_used_for_role_or_selection_diagnostic"] is False


def test_partition_recovery_is_deterministic_and_disjoint() -> None:
    rows = [{"uid": f"uid-{index:05d}"} for index in range(6000)]
    first = build_partitions(rows)
    second = build_partitions(list(reversed(rows)))
    assert first == second
    assert len(first["v40_excluded"]) == 2512
    assert len(first["pilot"]) == 512
    assert len(first["confirmation"]) == 2000
    assert not (set(first["v40_excluded"]) & set(first["pilot"]))
    assert not (set(first["pilot"]) & set(first["confirmation"]))
    assert partition_commitments(first) == partition_commitments(second)


def test_atomic_query_parser_is_strict_and_has_frozen_fallback() -> None:
    response = json.dumps(
        {
            "anchor": "entity identity",
            "first_fact": "first relation",
            "second_fact_or_bridge": "bridge relation",
            "counterevidence": "contradicting relation",
        }
    )
    parsed, fallback, reason = parse_atomic_queries(response, "A claim")
    assert list(parsed) == list(DYNAMIC_ROLES)
    assert fallback is False
    assert reason == ""

    parsed, fallback, reason = parse_atomic_queries("not json", "A claim")
    assert fallback is True
    assert reason
    assert all("A claim" in parsed[role] for role in DYNAMIC_ROLES)


def test_rank_coverage_selector_enforces_budget_and_unique_source() -> None:
    candidates = [_candidate(index) for index in range(10)]
    candidates[1]["source_id"] = candidates[0]["source_id"]
    static = select_candidates(candidates, STATIC_RANK, token_budget=512)
    dynamic = select_candidates(candidates, DYNAMIC_RANK, token_budget=512)
    for selected in (static, dynamic):
        assert len(selected) <= 5
        assert sum(row["token_count"] for row in selected) <= 512
        assert len({row["source_id"] for row in selected}) == len(selected)
    assert [row["id"] for row in static] != [row["id"] for row in dynamic]


def test_gold_free_mechanism_gate_detects_role_decorrelation() -> None:
    rows = [_scored_row(f"hover-v41::{index}") for index in range(8)]
    report = evaluate_mechanism_gate(
        rows,
        {
            "rows": 8,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "fallback_reasons": {},
            "query_length_codepoints": {"minimum": 3, "mean": 5.0, "maximum": 8},
            "gold_fields_visible_to_generator": False,
        },
    )
    assert report["metadata"]["gold_fields_joined"] is False
    assert report["checks"]["dynamic_spearman_reduction_at_least_0_05"] is True
    assert report["checks"]["dynamic_distinct_argmax_gain_at_least_0_25"] is True
    assert report["outcome"]["confirmation_open_authorized"] is True


def test_scored_cache_rejects_gold_field() -> None:
    row = _scored_row()
    row["label"] = "SUPPORTED"
    with pytest.raises(ValueError, match="forbidden"):
        merge_scored_candidates(row)


def test_confirmation_evaluation_keeps_raw_text_out() -> None:
    row = _scored_row()
    gold = [
        {
            "case_id": row["id"],
            "label": "SUPPORTED",
            "hop_count": 2,
            "gold_document_count": 2,
            "gold_sources": ["s00", "s02"],
            "candidate_ceiling": 1.0,
            "candidate_ceiling_complete": True,
        }
    ]
    report, evidence = evaluate_confirmation(gold, [row], resamples=10)
    assert report["metadata"]["cases"] == 1
    assert evidence[0]["raw_claim_title_article_or_evidence_text_exported"] is False
    assert '"claim":' not in json.dumps(evidence[0])
