from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag.whoqa_budget_stress import (
    PRIMARY,
    evaluate_cases,
    load_protocol,
    load_report,
    read_jsonl,
    sha256,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "docs/progressive_upgrade/whoqa_budget_stress_protocol.json"
)
REPORT_PATH = (
    REPO_ROOT
    / "output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress.json"
)
EVIDENCE_PATH = (
    REPO_ROOT
    / "output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress_cases.jsonl.gz"
)


def _protocol() -> dict:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["frozen_evaluation"]["paired_bootstrap"]["resamples"] = 64
    return protocol


def _raw_case() -> dict:
    return {
        "q_id": "synthetic-1",
        "contexts": ["first", "duplicate", "alternative"],
        "answer_by_context": {
            "0": [["alpha"]],
            "1": [["alpha"]],
            "2": [["beta"]],
        },
        "num_distinct_answers": 2,
    }


def _scores(
    bm25: float,
    dense: float,
    hybrid: float,
    cross_encoder: float,
) -> dict[str, float]:
    return {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "cross_encoder": cross_encoder,
    }


def _roles(answer: float, alternative: float) -> dict[str, float]:
    return {
        "answer_claim": answer,
        "entity_attribution": 0.1,
        "alternative_claim": alternative,
        "ambiguity_disclosure": 0.1,
    }


def _scored_case() -> dict:
    candidates = [
        {"id": "c0", "context_index": 0, "token_count": 300},
        {"id": "c1", "context_index": 1, "token_count": 300},
        {"id": "c2", "context_index": 2, "token_count": 300},
    ]
    candidate_scores = [
        {
            "id": "c0",
            "scores": _scores(0.9, 0.9, 0.9, 0.9),
            "role_scores": _roles(0.9, 0.1),
        },
        {
            "id": "c1",
            "scores": _scores(0.8, 0.8, 0.8, 0.8),
            "role_scores": _roles(0.8, 0.1),
        },
        {
            "id": "c2",
            "scores": _scores(0.7, 0.7, 0.7, 0.7),
            "role_scores": _roles(0.1, 0.9),
        },
    ]
    return {
        "id": "synthetic-1",
        "property_type": "demo",
        "gold_fields_visible_to_scorer": False,
        "candidates": candidates,
        "variants": [
            {
                "variant_index": 0,
                "candidate_scores": candidate_scores,
            }
        ],
    }


def test_protocol_freezes_six_configs_and_selector_source_hash() -> None:
    protocol = load_protocol(PROTOCOL_PATH)

    assert [item["config_id"] for item in protocol["frozen_grid"]] == [
        "k2_b1500",
        "k3_b1500",
        "k4_b256",
        "k4_b512",
        "k4_b1024",
        "k4_b1500",
    ]
    selector = REPO_ROOT / protocol["frozen_inputs"][
        "frozen_selector_implementation"
    ]["path"]
    assert sha256(selector) == protocol["frozen_inputs"][
        "frozen_selector_implementation"
    ]["sha256"]
    assert protocol["development_boundary"]["independent_confirmation"] is False
    assert protocol["development_boundary"]["gate_evidence"] is False


def test_stress_metric_penalizes_token_shortfall_and_rewards_view_diversity() -> None:
    report, evidence = evaluate_cases(
        [_raw_case()],
        [_scored_case()],
        _protocol(),
        expected_cases=None,
    )

    row = evidence[0]
    assert row["configurations"]["k4_b256"]["methods"]["frc_select"][
        "metrics"
    ][PRIMARY] == 0.0
    assert row["configurations"]["k2_b1500"]["methods"]["bm25_topk"][
        "metrics"
    ][PRIMARY] == 0.5
    assert row["configurations"]["k2_b1500"]["methods"]["frc_select"][
        "metrics"
    ][PRIMARY] == 1.0
    assert report["metadata"]["selection_runs"] == 36
    assert report["development_boundary"]["gate_evidence"] is False
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"


def test_gold_fields_in_blind_score_cache_are_rejected() -> None:
    scored = _scored_case()
    scored["answer_by_context"] = {"0": [["alpha"]]}

    with pytest.raises(ValueError, match="gold fields leaked"):
        evaluate_cases(
            [_raw_case()],
            [scored],
            _protocol(),
            expected_cases=None,
        )


def test_report_is_deterministic_private_and_tamper_evident(tmp_path: Path) -> None:
    report, evidence = evaluate_cases(
        [_raw_case()],
        [_scored_case()],
        _protocol(),
        expected_cases=None,
    )
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    first = write_report(report, evidence, tmp_path / "first", source_paths={"x": source})
    second = write_report(
        report,
        evidence,
        tmp_path / "second",
        source_paths={"x": source},
    )

    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["markdown"].read_bytes() == second["markdown"].read_bytes()
    assert first["evidence"].read_bytes() == second["evidence"].read_bytes()
    loaded = load_report(first["json"], first["evidence"])
    assert loaded["metadata"]["evidence_artifact"][
        "raw_question_or_answer_exported"
    ] is False
    assert not (
        {"question", "contexts", "answer_by_context", "main_ent"}
        & set(json.dumps(evidence, sort_keys=True).split('"'))
    )

    first["evidence"].write_bytes(first["evidence"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        load_report(first["json"], first["evidence"])


def test_protocol_file_hash_is_stable_for_result_provenance() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == (
        "2c120be37eac8c5955c8170d828dfbf0638325b37fee51ae42494c4b08a6fbec"
    )


def test_committed_full_report_preserves_the_frozen_negative_result() -> None:
    report = load_report(REPORT_PATH, EVIDENCE_PATH)
    metadata = report["metadata"]
    analysis = report["analysis"]
    family = analysis["family_comparison"]
    outcome = analysis["outcome"]

    assert metadata["cases"] == 5152
    assert metadata["template_count"] == 26957
    assert metadata["selection_runs"] == 970452
    assert metadata["deterministic_output_rerun"] == "2/2 byte-identical"
    assert family["observed_strongest_baseline"] == "dense_topk"
    assert family["method_primary_means"]["dense_topk"] == 0.890719
    assert family["method_primary_means"]["frc_select"] == 0.887793
    assert family["frc_minus_observed_strongest_point"] == -0.002926
    assert family["frc_minus_bootstrap_strongest_simultaneous"]["ci_low"] == (
        -0.003858
    )
    assert family["frc_minus_bootstrap_strongest_simultaneous"]["ci_high"] == (
        -0.002025
    )
    assert analysis["configuration_comparisons"]["k4_b256"][
        "frc_minus_observed_strongest_point"
    ] == -0.016657
    assert analysis["safety"] == {
        "worst_configuration_delta": -0.016657,
        "worst_stratum_delta": -0.012057,
    }
    assert outcome["status"] == "WHOQA_STRESS_SUPPORT_NOT_ESTABLISHED"
    assert outcome["selector_changed"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"
    assert outcome["canary_or_default_authorized"] is False
    assert outcome["independent_confirmation"] is False

    evidence = list(read_jsonl(EVIDENCE_PATH))
    assert len(evidence) == 5152
    assert len({row["case_id"] for row in evidence}) == 5152
    encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        '"question"',
        '"questions"',
        '"contexts"',
        '"answer_by_context"',
        '"main_ent"',
    ):
        assert forbidden not in encoded
