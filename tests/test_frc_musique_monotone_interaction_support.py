from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag import musique_monotone_interaction_support as experiment


def _source(source_id: str, answerable: bool, *, hops: int = 2) -> dict:
    decomposition = [
        {
            "id": hop,
            "question": f"Hop {hop} #1?" if hop > 1 else f"Hop 1 {source_id}?",
            "answer": "gold" if hop == hops else f"bridge-{hop}",
            "paragraph_support_idx": hop - 1,
        }
        for hop in range(1, hops + 1)
    ]
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


def _balanced_rows(count: int = 100) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        source_id = f"source-{index}"
        rows.extend([_source(source_id, True), _source(source_id, False)])
    return rows


def _prepared(*, hops: int = 2) -> list[dict]:
    contexts = [
        "Alpha bridge is introduced here.",
        "Alpha bridge leads to Beta answer.",
        "Beta answer identifies Gamma result.",
        "Gamma result completes the chain.",
    ][: max(2, hops)]
    return [
        {
            "id": "m69k-case",
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
                for index, context in enumerate(contexts)
            ],
        }
    ]


def _base_features(*, answerable: bool, hop_count: int) -> dict[str, float]:
    values = {name: 0.0 for name in experiment.BASE_FEATURE_NAMES}
    if answerable:
        values["chain_min_margin_scaled"] = 1.0
        values["chain_mean_margin_scaled"] = 1.0
        values["transition_entity_match_fraction"] = 1.0
        values["distinct_selected_paragraph_ratio"] = 1.0
    values["hop_count_is_3"] = float(hop_count == 3)
    values["hop_count_is_4"] = float(hop_count == 4)
    return values


def _feature_row(
    case_id: str,
    *,
    answerable: bool,
    hop_count: int,
) -> dict:
    return {
        "case_id": case_id,
        "answer_state": "answerable" if answerable else "unanswerable",
        "hop_count": hop_count,
        "direct_score_margin": 0.0,
        "chain_bottleneck_score": 0.0,
        "features": experiment.expand_feature_vector(
            _base_features(answerable=answerable, hop_count=hop_count)
        ),
        "invalid_feature_output": False,
    }


def _separable_evidence(count: int = 60) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        hop_count = 2 + index % 3
        rows.append(
            _feature_row(f"a-{index}", answerable=True, hop_count=hop_count)
        )
        rows.append(
            _feature_row(f"n-{index}", answerable=False, hop_count=hop_count)
        )
    return rows


def _zero_model(feature_names: tuple[str, ...]) -> dict:
    return {
        "feature_names": list(feature_names),
        "coefficients": [0.0] * len(feature_names),
        "intercept": 0.0,
    }


def _frozen_calibration() -> dict:
    candidate_coefficients = [0.0] * len(experiment.CANDIDATE_FEATURE_NAMES)
    candidate_coefficients[
        experiment.CANDIDATE_FEATURE_NAMES.index(
            "weakest_link_x_transition_consistency"
        )
    ] = 10.0
    candidate = {
        "feature_names": list(experiment.CANDIDATE_FEATURE_NAMES),
        "coefficients": candidate_coefficients,
        "intercept": -5.0,
    }
    model_threshold = {"threshold_exact": 0.5}
    raw_threshold = {"threshold_exact": 0.0}
    return {
        "calibrated_direct_composed_question": {"pooled": raw_threshold},
        "calibrated_chain_bottleneck": {"pooled": raw_threshold},
        "linear_nine_signal_control": {
            "model": _zero_model(experiment.LINEAR_FEATURE_NAMES),
            "pooled": model_threshold,
        },
        "monotone_additive_spline_control": {
            "model": _zero_model(experiment.ADDITIVE_FEATURE_NAMES),
            "pooled": model_threshold,
        },
        "monotone_interaction_candidate": {
            "model": candidate,
            "pooled": model_threshold,
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


def test_prepare_blind_stage_removes_all_gold_fields() -> None:
    prepared, maps, direct, census = experiment.prepare_blind_stage(
        [_source("alpha", True), _source("beta", False)],
        stage="calibration",
    )
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    }
    assert not forbidden & experiment.nested_keys([prepared, maps, direct])
    assert census["prepared_cases"] == 2
    assert census["direct_qa_input_rows"] == 6
    assert all(row["id"].startswith("m69k-") for row in prepared)


def test_expand_feature_vector_uses_frozen_hinges_and_interactions() -> None:
    base = {
        "direct_margin_scaled": 0.4,
        "chain_min_margin_scaled": -0.2,
        "chain_mean_margin_scaled": 0.2,
        "high_margin_fraction": 0.75,
        "transition_entity_match_fraction": 0.5,
        "distinct_selected_paragraph_ratio": 1.0,
        "direct_final_paragraph_agreement": 1.0,
        "hop_count_is_3": 1.0,
        "hop_count_is_4": 0.0,
    }
    values = experiment.expand_feature_vector(base)
    assert tuple(values) == experiment.CANDIDATE_FEATURE_NAMES
    assert values["direct_margin_scaled_hinge_m0_5"] == pytest.approx(0.6)
    assert values["chain_min_margin_scaled_hinge_m0_5"] == pytest.approx(0.2)
    assert values["chain_mean_margin_scaled_hinge_0_0"] == pytest.approx(0.2)
    assert values["high_margin_fraction_hinge_p0_25"] == pytest.approx(2 / 3)
    assert values["transition_entity_match_fraction_hinge_p0_25"] == (
        pytest.approx(1 / 3)
    )
    assert values["distinct_selected_paragraph_ratio_hinge_p0_75"] == 1.0
    assert values["weakest_link_x_transition_consistency"] == pytest.approx(0.2)
    assert values["mean_confidence_x_transition_consistency"] == pytest.approx(0.3)
    assert values["strong_hop_x_paragraph_coverage"] == 0.75
    assert values["transition_x_paragraph_coverage"] == 0.5
    assert values["direct_confidence_x_final_alignment"] == pytest.approx(0.7)


def test_hinge_is_continuous_zero_at_knot_and_normalized_at_one() -> None:
    assert experiment._hinge(0.25, 0.25) == 0.0
    assert experiment._hinge(1.0, 0.25) == 1.0
    assert experiment._hinge(-0.5, -0.5) == 0.0
    assert experiment._hinge(1.0, -0.5) == 1.0


def test_below_fixed_threshold_still_propagates_and_records_transition() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    experiment.apply_hop_results(
        prepared,
        states,
        [
            {
                "case_id": "m69k-case",
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
                "case_id": "m69k-case",
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


def test_finalize_feature_decision_expands_complete_base_vector() -> None:
    prepared = _prepared(hops=3)
    states = experiment.initial_chain_states(prepared)
    states["m69k-case"].update(
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
            "case_id": "m69k-case",
            "score_margin": 4.0,
            "selected_paragraph_idx": 2,
        }
    ]
    hops = [
        {
            "case_id": "m69k-case",
            "hop_index": index,
            "executed": True,
            "valid_span": True,
        }
        for index in range(1, 4)
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops
    )
    assert result["feature_complete"] is True
    assert tuple(result["features"]) == experiment.CANDIDATE_FEATURE_NAMES
    assert result["features"]["chain_min_margin_scaled"] == pytest.approx(-0.1)
    assert result["features"]["hop_count_is_3"] == 1.0


def test_invalid_chain_output_fails_closed_without_partial_features() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    states["m69k-case"].update({"alive": False, "invalid_output": True})
    direct = [
        {
            "case_id": "m69k-case",
            "score_margin": 1.0,
            "selected_paragraph_idx": 0,
        }
    ]
    hops = [
        {
            "case_id": "m69k-case",
            "hop_index": index,
            "executed": index == 1,
            "valid_span": False,
        }
        for index in range(1, 3)
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops
    )
    assert result["feature_complete"] is False
    assert result["invalid_fail_closed_used"] is True
    assert all(value is None for value in result["features"].values())


def test_gold_join_imputes_invalid_blind_output_with_frozen_fail_closed_values() -> None:
    source = _source("case", True, hops=3)
    prepared = _prepared(hops=3)
    feature = {
        "case_id": "m69k-case",
        "direct_score_margin": None,
        "chain_bottleneck_score": None,
        "chain_mean_score": None,
        "features": {name: None for name in experiment.CANDIDATE_FEATURE_NAMES},
        "feature_sha256": "blind-null-feature-hash",
        "feature_complete": False,
        "executed_hops": 1,
        "valid_hops": 0,
        "hop_decision_sha256": "hop-hash",
        "invalid_fail_closed_used": True,
    }
    [evidence] = experiment.build_gold_evidence([source], prepared, [feature])
    assert evidence["schema_version"] == "frc-musique-v69-feature-evidence-v2"
    assert evidence["direct_score_margin"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert evidence["chain_bottleneck_score"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert evidence["chain_mean_score"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert evidence["features"]["direct_margin_scaled"] == -1.0
    assert evidence["features"]["hop_count_is_3"] == 1.0
    assert evidence["features"]["hop_count_is_4"] == 0.0
    assert all(value is not None for value in evidence["features"].values())
    assert evidence["blind_feature_sha256"] == "blind-null-feature-hash"
    assert evidence["invalid_feature_output"] is True


def test_projected_logistic_is_deterministic_and_nonnegative_constrained() -> None:
    evidence = _separable_evidence(count=30)
    first = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    second = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    assert first == second
    coefficients = dict(zip(first["feature_names"], first["coefficients"], strict=True))
    assert coefficients["weakest_link_x_transition_consistency"] > 0.0
    assert all(
        coefficients[name] >= 0.0
        for name in experiment.NONNEGATIVE_FEATURE_NAMES
    )


def test_feature_contracts_have_frozen_nested_capacities() -> None:
    assert len(experiment.LINEAR_FEATURE_NAMES) == 9
    assert len(experiment.HINGE_FEATURE_NAMES) == 18
    assert len(experiment.ADDITIVE_FEATURE_NAMES) == 27
    assert len(experiment.INTERACTION_FEATURE_NAMES) == 5
    assert len(experiment.CANDIDATE_FEATURE_NAMES) == 32
    assert experiment.LINEAR_FEATURE_NAMES == experiment.BASE_FEATURE_NAMES
    assert experiment.ADDITIVE_FEATURE_NAMES[:9] == experiment.BASE_FEATURE_NAMES
    assert experiment.CANDIDATE_FEATURE_NAMES[:27] == (
        experiment.ADDITIVE_FEATURE_NAMES
    )


def test_crossfit_model_covers_each_case_once_without_held_out_training() -> None:
    evidence = _separable_evidence(count=50)
    result = experiment._crossfit_model(
        evidence, feature_names=experiment.LINEAR_FEATURE_NAMES
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


def test_evaluate_stage_requires_primary_and_interaction_mechanism_gates() -> None:
    evidence = _separable_evidence(count=400)
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
    analysis = report["analysis"]
    outcome = analysis["outcome"]
    assert analysis["strongest_fair_baseline"]["name"] == (
        "calibrated_direct_composed_question"
    )
    assert analysis["paired_correctness_delta"]["point"] == 0.5
    assert analysis["interaction_mechanism_delta_vs_additive"]["point"] == 0.5
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
        "docs/progressive_upgrade/"
        "musique_monotone_interaction_support_protocol_v69.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["stopping_and_outcomes"]["gate_2"] = "GO"
    changed = tmp_path / "protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate 2"):
        experiment.validate_protocol(changed)
