from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.frc_rag import musique_bridge_counterfactual_dependence as experiment


def _source(source_id: str, answerable: bool, *, hops: int = 2) -> dict:
    return {
        "id": source_id,
        "question": f"Composed question {source_id}?",
        "answerable": answerable,
        "answer": "gold",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": hop,
                "question": f"Hop {hop} #{hop - 1}?"
                if hop > 1
                else f"Hop 1 {source_id}?",
                "answer": "gold" if hop == hops else f"bridge-{hop}",
                "paragraph_support_idx": hop - 1,
            }
            for hop in range(1, hops + 1)
        ],
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
            "id": "m71k-case",
            "oracle_hop_templates": [
                "First?",
                "Second #1?",
                "Third #2 and #1?",
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


def _complete_state(*, hops: int = 2) -> dict[str, dict]:
    return {
        "m71k-case": {
            "alive": True,
            "predictions": {
                index: ("Alpha bridge" if index == 1 else f"Answer {index}")
                for index in range(1, hops + 1)
            },
            "margins": [4.0 - index for index in range(hops)],
            "selected_paragraphs": list(range(hops)),
            "transition_matches": [True] * (hops - 1),
            "competition": [
                {"top1_top2_gap_scaled": 0.4, "top1_softmax_mass": 0.7}
                for _ in range(hops)
            ],
            "completed_hops": hops,
            "invalid_output": False,
        }
    }


def _all_features(*, answerable: bool, hop_count: int) -> dict[str, float]:
    base = {name: 0.0 for name in experiment.BASE_FEATURE_NAMES}
    if answerable:
        base["chain_min_margin_scaled"] = 1.0
        base["chain_mean_margin_scaled"] = 1.0
        base["transition_entity_match_fraction"] = 1.0
        base["distinct_selected_paragraph_ratio"] = 1.0
    base["hop_count_is_3"] = float(hop_count == 3)
    base["hop_count_is_4"] = float(hop_count == 4)
    prior = experiment.v70.v69.expand_feature_vector(base)
    competition = {
        name: float(answerable) for name in experiment.COMPETITION_FEATURE_NAMES
    }
    counterfactual = {
        name: float(answerable)
        for name in experiment.COUNTERFACTUAL_FEATURE_NAMES
    }
    return {**prior, **competition, **counterfactual}


def _separable_evidence(count: int = 30) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        hop_count = 2 + index % 3
        for answerable, prefix in ((True, "a"), (False, "n")):
            rows.append(
                {
                    "case_id": f"{prefix}-{index}",
                    "answer_state": "answerable"
                    if answerable
                    else "unanswerable",
                    "hop_count": hop_count,
                    "direct_score_margin": 0.0,
                    "chain_bottleneck_score": 0.0,
                    "features": _all_features(
                        answerable=answerable, hop_count=hop_count
                    ),
                    "invalid_feature_output": False,
                }
            )
    return rows


def _zero_model(feature_names: tuple[str, ...]) -> dict:
    return {
        "feature_names": list(feature_names),
        "coefficients": [0.0] * len(feature_names),
        "intercept": 0.0,
    }


def _frozen_calibration(*, counterfactual_only_perfect: bool = False) -> dict:
    candidate = _zero_model(experiment.CANDIDATE_FEATURE_NAMES)
    candidate["coefficients"][0 - len(experiment.COUNTERFACTUAL_FEATURE_NAMES)] = 10.0
    candidate["intercept"] = -5.0
    counterfactual_only = _zero_model(experiment.COUNTERFACTUAL_FEATURE_NAMES)
    if counterfactual_only_perfect:
        counterfactual_only["coefficients"][0] = 10.0
        counterfactual_only["intercept"] = -5.0
    raw_threshold = {"threshold_exact": 0.0}
    model_threshold = {"threshold_exact": 0.5}
    result = {
        "calibrated_direct_composed_question": {"pooled": raw_threshold},
        "calibrated_chain_bottleneck": {"pooled": raw_threshold},
        "bridge_counterfactual_candidate": {
            "model": candidate,
            "pooled": model_threshold,
        },
    }
    models = {
        "linear_nine_signal_control": _zero_model(experiment.BASE_FEATURE_NAMES),
        "monotone_additive_spline_control": _zero_model(
            experiment.ADDITIVE_FEATURE_NAMES
        ),
        "monotone_interaction_control": _zero_model(
            experiment.INTERACTION_FEATURE_NAMES
        ),
        "paragraph_competition_control": _zero_model(
            experiment.PARAGRAPH_COMPETITION_FEATURE_NAMES
        ),
        "bridge_counterfactual_only_control": counterfactual_only,
    }
    result.update(
        {
            name: {"model": model, "pooled": model_threshold}
            for name, model in models.items()
        }
    )
    return result


def _direct_and_hops(*, hops: int = 2) -> tuple[list[dict], list[dict]]:
    direct = [
        {
            "case_id": "m71k-case",
            "score_margin": 4.0,
            "selected_paragraph_idx": hops - 1,
            "competition_complete": True,
            "competition": {
                "top1_top2_gap_scaled": 0.5,
                "top1_softmax_mass": 0.8,
            },
        }
    ]
    decisions = [
        {
            "case_id": "m71k-case",
            "hop_index": index,
            "executed": True,
            "valid_span": True,
        }
        for index in range(1, hops + 1)
    ]
    return direct, decisions


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
    assert sum(row["answerable"] for row in first) == 5
    assert excluded_id not in {row["id"] for row in first}


def test_prepare_blind_stage_uses_v71_ids_and_removes_gold() -> None:
    prepared, maps, direct, census = experiment.prepare_blind_stage(
        [_source("alpha", True), _source("beta", False)], stage="calibration"
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
    assert all(row["id"].startswith("m71k-") for row in prepared)


def test_counterfactual_inputs_replace_every_placeholder_without_predictions() -> None:
    prepared = _prepared(hops=3)
    inputs = experiment.build_counterfactual_qa_inputs(
        prepared, _complete_state(hops=3)
    )
    questions = {row["hop_index"]: row["question"] for row in inputs}
    assert questions[2] == "Second unknown bridge entity?"
    assert questions[3] == (
        "Third unknown bridge entity and unknown bridge entity?"
    )
    assert all("Alpha bridge" not in row["question"] for row in inputs)
    assert all("#" not in row["question"] for row in inputs)


def test_counterfactual_inputs_skip_incomplete_factual_chain() -> None:
    states = _complete_state()
    states["m71k-case"]["alive"] = False
    assert experiment.build_counterfactual_qa_inputs(_prepared(), states) == []


def test_downstream_question_without_placeholder_is_unchanged_zero_substitution() -> None:
    prepared = _prepared(hops=2)
    prepared[0]["oracle_hop_templates"][1] = "Independent downstream question?"
    inputs = experiment.build_counterfactual_qa_inputs(
        prepared, _complete_state(hops=2)
    )
    assert {row["question"] for row in inputs} == {
        "Independent downstream question?"
    }
    assert {row["placeholder_substitutions"] for row in inputs} == {0}


def test_aggregate_counterfactual_rows_preserves_root_and_hop() -> None:
    inputs = [
        {
            "id": f"transition::p{index}",
            "case_id": "transition",
            "root_case_id": "m71k-case",
            "hop_index": 2,
            "paragraph_idx": index,
            "context": text,
        }
        for index, text in enumerate(("Alpha", "Beta", "Gamma"))
    ]
    decisions = [
        {
            "id": row["id"],
            "cache_key": "key",
            "score_margin": float(3 - index),
            "span_start": 0,
            "span_end": len(row["context"]),
            "invalid_fail_closed_used": False,
        }
        for index, row in enumerate(inputs)
    ]
    [result] = experiment.aggregate_counterfactual_qa_rows(
        inputs, decisions, cache_key="key"
    )
    assert result["root_case_id"] == "m71k-case"
    assert result["hop_index"] == 2
    assert result["score_margin"] == 3.0


def test_finalize_features_compute_fixed_counterfactual_statistics() -> None:
    prepared = _prepared(hops=2)
    states = _complete_state(hops=2)
    direct, hops = _direct_and_hops(hops=2)
    counterfactual = [
        {
            "root_case_id": "m71k-case",
            "hop_index": 2,
            "score_margin": 0.0,
            "selected_paragraph_idx": 0,
            "predicted_span": "Different",
            "predicted_span_sha256": "span-hash",
            "invalid_fail_closed_used": False,
        }
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops, counterfactual
    )
    assert result["feature_complete"] is True
    assert tuple(result["features"]) == experiment.ALL_FEATURE_NAMES
    assert result["features"]["bridge_margin_drop_min"] == pytest.approx(0.3)
    assert result["features"]["bridge_paragraph_change_min"] == 1.0
    assert result["features"]["bridge_span_change_min"] == 1.0


def test_counterfactual_no_span_is_maximum_dependence_not_invalid() -> None:
    prepared = _prepared(hops=2)
    states = _complete_state(hops=2)
    direct, hops = _direct_and_hops(hops=2)
    counterfactual = [
        {
            "root_case_id": "m71k-case",
            "hop_index": 2,
            "score_margin": -3.0,
            "selected_paragraph_idx": 0,
            "predicted_span": "",
            "predicted_span_sha256": hashlib.sha256(b"").hexdigest(),
            "invalid_fail_closed_used": False,
        }
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops, counterfactual
    )
    assert result["feature_complete"] is True
    assert all(
        result["features"][name] == 1.0
        for name in experiment.COUNTERFACTUAL_FEATURE_NAMES
    )


def test_missing_counterfactual_transition_fails_case_closed() -> None:
    prepared = _prepared(hops=2)
    direct, hops = _direct_and_hops(hops=2)
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, _complete_state(hops=2), hops, []
    )
    assert result["feature_complete"] is False
    assert result["invalid_fail_closed_used"] is True


def test_gold_join_imputes_invalid_counterfactual_features() -> None:
    feature = {
        "case_id": "m71k-case",
        "direct_score_margin": None,
        "chain_bottleneck_score": None,
        "chain_mean_score": None,
        "features": {name: None for name in experiment.ALL_FEATURE_NAMES},
        "feature_sha256": "blind-null",
        "feature_complete": False,
        "executed_hops": 1,
        "valid_hops": 1,
        "counterfactual_transitions": 0,
        "counterfactual_decision_sha256": "counterfactual-hash",
        "hop_decision_sha256": "hop-hash",
        "invalid_fail_closed_used": True,
    }
    [evidence] = experiment.build_gold_evidence(
        [_source("case", True)], _prepared(), [feature]
    )
    assert evidence["direct_score_margin"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert all(
        evidence["features"][name] == 0.0
        for name in experiment.COUNTERFACTUAL_FEATURE_NAMES
    )
    assert evidence["invalid_feature_output"] is True


def test_projected_logistic_is_deterministic_and_counterfactual_nonnegative() -> None:
    evidence = _separable_evidence()
    first = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    second = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    assert first == second
    coefficients = dict(
        zip(first["feature_names"], first["coefficients"], strict=True)
    )
    assert all(
        coefficients[name] >= 0.0
        for name in experiment.COUNTERFACTUAL_FEATURE_NAMES
    )


def test_feature_contracts_and_total_capacity_are_frozen() -> None:
    assert len(experiment.BASE_FEATURE_NAMES) == 9
    assert len(experiment.INTERACTION_FEATURE_NAMES) == 32
    assert len(experiment.COMPETITION_FEATURE_NAMES) == 6
    assert len(experiment.COUNTERFACTUAL_FEATURE_NAMES) == 6
    assert len(experiment.CANDIDATE_FEATURE_NAMES) == 15
    assert len(experiment.ALL_FEATURE_NAMES) == 44
    assert sum((2, 11, 29, 34, 17, 8, 17)) == 118


def test_crossfit_model_covers_each_case_once() -> None:
    evidence = _separable_evidence()
    result = experiment._crossfit_model(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    assert result["held_out_cases"] == len(evidence)
    assert sum(row["held_out_cases"] for row in result["folds"]) == len(evidence)


def test_evaluate_stage_requires_all_three_incremental_gates() -> None:
    evidence = _separable_evidence(count=400)
    sampling = {
        "schema_exclusion_rate": 0.0,
        "selected_excluded_source_commitment_overlap": 0,
        "selected_squad2_exact_question_overlap": 0,
    }
    passed = experiment.evaluate_stage(
        evidence,
        stage="development",
        calibration=_frozen_calibration(),
        sampling=sampling,
        structural_census={},
        source_artifacts={},
    )
    assert passed["analysis"]["outcome"]["stage_gate_passed"] is True
    failed = experiment.evaluate_stage(
        evidence,
        stage="development",
        calibration=_frozen_calibration(counterfactual_only_perfect=True),
        sampling=sampling,
        structural_census={},
        source_artifacts={},
    )
    assert failed["analysis"]["support_checks"][
        "candidate_minus_counterfactual_only_control_at_least_0_02"
    ] is False
    assert failed["analysis"]["outcome"]["stage_gate_passed"] is False


def test_write_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _separable_evidence(count=2)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    experiment.write_evidence(first, rows)
    experiment.write_evidence(second, rows)
    assert first.read_bytes() == second.read_bytes()


def test_validate_protocol_rejects_gate_promotion(tmp_path: Path) -> None:
    source = Path(
        "docs/progressive_upgrade/"
        "musique_bridge_counterfactual_dependence_protocol_v71.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value["stopping_and_outcomes"]["gate_2"] = "GO"
    target = tmp_path / "protocol.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate 2"):
        experiment.validate_protocol(target)
