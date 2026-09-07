from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag import musique_multisignal_chain_support as experiment


def _source(source_id: str, answerable: bool, *, hops: int = 2) -> dict:
    decomposition = []
    for hop in range(1, hops + 1):
        decomposition.append(
            {
                "id": hop,
                "question": f"Hop {hop} #1?" if hop > 1 else f"Hop 1 {source_id}?",
                "answer": "gold" if hop == hops else f"bridge-{hop}",
                "paragraph_support_idx": hop - 1,
            }
        )
    return {
        "id": source_id,
        "question": f"Composed question {source_id}?",
        "answerable": answerable,
        "answer": "gold",
        "answer_aliases": [],
        "question_decomposition": decomposition,
        "paragraphs": [
            {
                "idx": index,
                "title": f"T{index}",
                "paragraph_text": f"Context {source_id} {index}",
            }
            for index in range(max(3, hops))
        ],
    }


def _balanced_rows(count: int = 80) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        source_id = f"source-{index}"
        rows.extend([_source(source_id, True), _source(source_id, False)])
    return rows


def _prepared(*, hops: int = 2) -> list[dict]:
    return [
        {
            "id": "m68k-case",
            "oracle_hop_templates": [
                "First?",
                "Second #1?",
                "Third #2?",
                "Fourth #3?",
            ][:hops],
            "contexts": [
                {
                    "paragraph_idx": index,
                    "context": context,
                    "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                }
                for index, context in enumerate(
                    [
                        "Alpha bridge is introduced here.",
                        "Alpha bridge leads to Beta answer.",
                        "Beta answer identifies Gamma result.",
                        "Gamma result completes the chain.",
                    ][: max(2, hops)]
                )
            ],
        }
    ]


def _feature_row(
    case_id: str,
    *,
    answerable: bool,
    signal: float,
    hop_count: int,
) -> dict:
    values = {name: 0.0 for name in experiment.FEATURE_NAMES}
    values["chain_mean_margin_scaled"] = signal
    values["hop_count_is_3"] = float(hop_count == 3)
    values["hop_count_is_4"] = float(hop_count == 4)
    return {
        "case_id": case_id,
        "answer_state": "answerable" if answerable else "unanswerable",
        "hop_count": hop_count,
        "direct_score_margin": 0.0,
        "chain_bottleneck_score": 0.0,
        "features": values,
        "invalid_feature_output": False,
    }


def _separable_evidence(count: int = 60) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        hop_count = 2 + index % 3
        rows.append(
            _feature_row(
                f"a-{index}", answerable=True, signal=1.0, hop_count=hop_count
            )
        )
        rows.append(
            _feature_row(
                f"n-{index}", answerable=False, signal=0.0, hop_count=hop_count
            )
        )
    return rows


def _frozen_calibration() -> dict:
    zero_model = {
        "feature_names": list(experiment.TWO_SIGNAL_FEATURE_NAMES),
        "coefficients": [0.0, 0.0],
        "intercept": 0.0,
    }
    candidate_model = {
        "feature_names": list(experiment.FEATURE_NAMES),
        "coefficients": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "intercept": -5.0,
    }
    threshold = {"threshold_exact": 0.5}
    raw_threshold = {"threshold_exact": 0.0}
    return {
        "calibrated_direct_composed_question": {"pooled": raw_threshold},
        "calibrated_chain_bottleneck": {"pooled": raw_threshold},
        "two_signal_logistic_control": {
            "model": zero_model,
            "pooled": threshold,
        },
        "multisignal_logistic_candidate": {
            "model": candidate_model,
            "pooled": threshold,
        },
    }


def test_select_stage_sample_is_balanced_disjoint_and_deterministic() -> None:
    rows = _balanced_rows()
    excluded_id = "source-0"
    excluded = {hashlib.sha256(excluded_id.encode()).hexdigest()}
    first, census = experiment.select_stage_sample(
        rows,
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=excluded,
        target_per_group=5,
    )
    second, second_census = experiment.select_stage_sample(
        rows,
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=excluded,
        target_per_group=5,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert census == second_census
    assert len(first) == 10
    assert len({row["id"] for row in first}) == 10
    assert sum(row["answerable"] for row in first) == 5
    assert excluded_id not in {row["id"] for row in first}


def test_select_stage_sample_rejects_split_mismatch() -> None:
    with pytest.raises(ValueError, match="stage/source split"):
        experiment.select_stage_sample(
            [],
            stage="confirmation",
            source_split="train",
            squad_questions=set(),
            excluded_source_commitments=set(),
            target_per_group=1,
        )


def test_prepare_blind_stage_removes_all_gold_fields() -> None:
    selected = [_source("alpha", True), _source("beta", False)]
    prepared, maps, direct, census = experiment.prepare_blind_stage(
        selected, stage="calibration"
    )
    keys = experiment.nested_keys([prepared, maps, direct])
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    }
    assert not forbidden & keys
    assert census["prepared_cases"] == 2
    assert census["direct_qa_input_rows"] == 6
    assert all(row["source_id_commitment"] for row in maps)


def test_entity_matching_is_normalized_and_rejects_short_spans() -> None:
    assert experiment.normalize_entity("  Alpha-BRIDGE! ") == "alpha bridge"
    assert experiment.entity_appears_in_context(
        "Alpha-Bridge", "The ALPHA bridge leads onward."
    )
    assert not experiment.entity_appears_in_context("AI", "AI is present")


def test_below_fixed_threshold_still_propagates_and_records_transition() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    [first] = experiment.apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "m68k-case",
                "score_margin": -3.0,
                "predicted_span": "Alpha bridge",
                "predicted_span_sha256": "hash-1",
                "selected_paragraph_idx": 0,
                "invalid_fail_closed_used": False,
            }
        ],
        {},
        hop_index=1,
    )
    assert first["valid_span"] is True
    assert states["m68k-case"]["alive"] is True
    next_inputs, blocked = experiment.build_hop_qa_inputs(
        prepared, states, hop_index=2
    )
    assert not blocked
    assert {row["question"] for row in next_inputs} == {"Second Alpha bridge?"}
    [second] = experiment.apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "m68k-case",
                "score_margin": -2.0,
                "predicted_span": "Beta answer",
                "predicted_span_sha256": "hash-2",
                "selected_paragraph_idx": 1,
                "invalid_fail_closed_used": False,
            }
        ],
        {},
        hop_index=2,
    )
    assert second["transition_entity_match"] is True


def test_finalize_feature_decision_computes_frozen_nine_signal_vector() -> None:
    prepared = _prepared(hops=3)
    states = experiment.initial_chain_states(prepared)
    states["m68k-case"].update(
        {
            "alive": True,
            "completed_hops": 3,
            "margins": [2.0, 1.0, -1.0],
            "selected_paragraphs": [0, 1, 2],
            "transition_matches": [True, False],
        }
    )
    direct = [
        {
            "case_id": "m68k-case",
            "score_margin": 4.0,
            "selected_paragraph_idx": 2,
        }
    ]
    hops = [
        {
            "case_id": "m68k-case",
            "hop_index": index,
            "executed": True,
            "valid_span": True,
        }
        for index in range(1, 4)
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops
    )
    features = result["features"]
    assert result["feature_complete"] is True
    assert features["direct_margin_scaled"] == pytest.approx(0.4)
    assert features["chain_min_margin_scaled"] == pytest.approx(-0.1)
    assert features["chain_mean_margin_scaled"] == pytest.approx(2.0 / 30.0)
    assert features["high_margin_fraction"] == pytest.approx(2.0 / 3.0)
    assert features["transition_entity_match_fraction"] == 0.5
    assert features["distinct_selected_paragraph_ratio"] == 1.0
    assert features["direct_final_paragraph_agreement"] == 1.0
    assert features["hop_count_is_3"] == 1.0
    assert features["hop_count_is_4"] == 0.0


def test_invalid_chain_output_fails_closed_without_partial_features() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    states["m68k-case"].update({"alive": False, "invalid_output": True})
    direct = [
        {
            "case_id": "m68k-case",
            "score_margin": 1.0,
            "selected_paragraph_idx": 0,
        }
    ]
    hops = [
        {
            "case_id": "m68k-case",
            "hop_index": 1,
            "executed": True,
            "valid_span": False,
        },
        {
            "case_id": "m68k-case",
            "hop_index": 2,
            "executed": False,
            "valid_span": False,
        },
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops
    )
    assert result["feature_complete"] is False
    assert result["invalid_fail_closed_used"] is True
    assert set(result["features"]) == set(experiment.FEATURE_NAMES)
    assert all(value is None for value in result["features"].values())


def test_projected_logistic_is_deterministic_and_monotone_constrained() -> None:
    evidence = _separable_evidence(count=30)
    first = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.FEATURE_NAMES
    )
    second = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.FEATURE_NAMES
    )
    assert first == second
    coefficients = dict(zip(first["feature_names"], first["coefficients"], strict=True))
    assert coefficients["chain_mean_margin_scaled"] > 0.0
    assert all(coefficients[name] >= 0.0 for name in experiment.MONOTONE_FEATURE_NAMES)
    scores = experiment.predict_projected_logistic(evidence, first)
    assert np.mean(scores[::2]) > np.mean(scores[1::2])


def test_crossfit_model_covers_each_case_once_without_held_out_training() -> None:
    evidence = _separable_evidence(count=50)
    result = experiment._crossfit_model(
        evidence, feature_names=experiment.TWO_SIGNAL_FEATURE_NAMES
    )
    assert result["held_out_cases"] == len(evidence)
    assert len(result["folds"]) == 5
    assert sum(row["held_out_cases"] for row in result["folds"]) == len(evidence)
    assert all(
        row["train_cases"] + row["held_out_cases"] == len(evidence)
        for row in result["folds"]
    )


def test_threshold_selection_obeys_strict_greater_and_safety_constraints() -> None:
    labels = np.asarray([1.0, 1.0, 0.0, 0.0])
    scores = np.asarray([1.0, 1.0, 0.0, 0.0])
    result = experiment.select_threshold_from_scores(labels, scores)
    assert result["threshold_exact"] == 0.0
    assert result["safety_constraints_met"] is True
    assert experiment.score_metrics(labels, scores, 1.0)["answerable_pass_rate"] == 0.0


def test_evaluate_stage_uses_strongest_fair_baseline_and_opens_all_gates() -> None:
    evidence = _separable_evidence(count=300)
    report = experiment.evaluate_stage(
        evidence,
        stage="development",
        calibration=_frozen_calibration(),
        sampling={
            "schema_exclusion_rate": 0.0,
            "selected_excluded_source_commitment_overlap": 0,
            "selected_squad2_exact_question_overlap": 0,
        },
        structural_census={},
        source_artifacts={},
    )
    outcome = report["analysis"]["outcome"]
    assert report["analysis"]["strongest_fair_baseline"]["name"] == (
        "calibrated_direct_composed_question"
    )
    assert report["analysis"]["paired_correctness_delta"]["point"] == 0.5
    assert outcome["stage_gate_passed"] is True
    assert outcome["confirmation_open_authorized"] is True
    assert outcome["canary_or_default_authorized"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"


def test_write_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _separable_evidence(count=2)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    experiment.write_evidence(first, rows)
    experiment.write_evidence(second, rows)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert len([json.loads(line) for line in handle]) == 4


def test_validate_protocol_rejects_gate_promotion(tmp_path: Path) -> None:
    source = Path(
        "docs/progressive_upgrade/musique_multisignal_chain_support_protocol_v68.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["stopping_and_outcomes"]["gate_2"] = "GO"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate 2"):
        experiment.validate_protocol(changed)
