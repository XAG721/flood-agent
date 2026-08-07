from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.frc_rag import musique_paragraph_competition_support as experiment
from scripts import run_musique_paragraph_competition_support as runner


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
                "question": f"Hop {hop} #1?" if hop > 1 else f"Hop 1 {source_id}?",
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
            "id": "m70k-case",
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


def _all_features(*, answerable: bool, hop_count: int) -> dict[str, float]:
    base = {name: 0.0 for name in experiment.BASE_FEATURE_NAMES}
    if answerable:
        base["chain_min_margin_scaled"] = 1.0
        base["chain_mean_margin_scaled"] = 1.0
        base["transition_entity_match_fraction"] = 1.0
        base["distinct_selected_paragraph_ratio"] = 1.0
    base["hop_count_is_3"] = float(hop_count == 3)
    base["hop_count_is_4"] = float(hop_count == 4)
    prior = experiment.v69.expand_feature_vector(base)
    competition = {
        name: float(answerable) for name in experiment.COMPETITION_FEATURE_NAMES
    }
    return {**prior, **competition}


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
        "features": _all_features(answerable=answerable, hop_count=hop_count),
        "invalid_feature_output": False,
    }


def _separable_evidence(count: int = 30) -> list[dict]:
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


def _frozen_calibration(*, competition_only_perfect: bool = False) -> dict:
    candidate_coefficients = [0.0] * len(experiment.CANDIDATE_FEATURE_NAMES)
    candidate_coefficients[
        experiment.CANDIDATE_FEATURE_NAMES.index(
            "direct_top1_top2_gap_scaled"
        )
    ] = 10.0
    candidate = {
        "feature_names": list(experiment.CANDIDATE_FEATURE_NAMES),
        "coefficients": candidate_coefficients,
        "intercept": -5.0,
    }
    competition_only = _zero_model(experiment.COMPETITION_FEATURE_NAMES)
    if competition_only_perfect:
        competition_only["coefficients"][0] = 10.0
        competition_only["intercept"] = -5.0
    raw_threshold = {"threshold_exact": 0.0}
    model_threshold = {"threshold_exact": 0.5}
    result = {
        "calibrated_direct_composed_question": {"pooled": raw_threshold},
        "calibrated_chain_bottleneck": {"pooled": raw_threshold},
        "paragraph_competition_candidate": {
            "model": candidate,
            "pooled": model_threshold,
        },
    }
    models = {
        "linear_nine_signal_control": _zero_model(
            experiment.BASE_FEATURE_NAMES
        ),
        "monotone_additive_spline_control": _zero_model(
            experiment.ADDITIVE_FEATURE_NAMES
        ),
        "monotone_interaction_control": _zero_model(
            experiment.INTERACTION_FEATURE_NAMES
        ),
        "paragraph_competition_only_control": competition_only,
    }
    result.update(
        {
            name: {"model": model, "pooled": model_threshold}
            for name, model in models.items()
        }
    )
    return result


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
    assert all(row["id"].startswith("m70k-") for row in prepared)


def test_paragraph_competition_statistics_use_fixed_gap_and_softmax() -> None:
    result = experiment.paragraph_competition_statistics([3.0, 1.0, 0.0])
    assert result["complete"] is True
    assert result["valid_paragraph_rows"] == 3
    assert result["top1_top2_gap_scaled"] == pytest.approx(0.2)
    expected_mass = 1.0 / (1.0 + np.exp(-2.0) + np.exp(-3.0))
    assert result["top1_softmax_mass"] == pytest.approx(expected_mass)


def test_paragraph_competition_tie_has_zero_gap_and_half_or_less_mass() -> None:
    result = experiment.paragraph_competition_statistics([2.0, 2.0, -5.0])
    assert result["top1_top2_gap_scaled"] == 0.0
    assert result["top1_softmax_mass"] <= 0.5
    incomplete = experiment.paragraph_competition_statistics([1.0])
    assert incomplete["complete"] is False


def test_aggregate_qa_rows_adds_gold_free_competition_summary() -> None:
    inputs = [
        {
            "id": f"m70k-case::p{index}",
            "case_id": "m70k-case",
            "paragraph_idx": index,
            "context": text,
        }
        for index, text in enumerate(("Alpha", "Beta", "Gamma"))
    ]
    margins = (3.0, 1.0, 0.0)
    decisions = [
        {
            "id": row["id"],
            "cache_key": "key",
            "score_margin": margins[index],
            "span_start": 0,
            "span_end": len(row["context"]),
            "invalid_fail_closed_used": False,
        }
        for index, row in enumerate(inputs)
    ]
    [result] = experiment.aggregate_qa_rows(inputs, decisions, cache_key="key")
    assert result["score_margin"] == 3.0
    assert result["competition_complete"] is True
    assert result["competition"]["top1_top2_gap_scaled"] == pytest.approx(0.2)
    assert "predicted_span" not in result["competition"]


def test_below_fixed_threshold_still_propagates_with_competition() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    aggregated = [
        {
            "case_id": "m70k-case",
            "score_margin": -3.0,
            "predicted_span": "Alpha bridge",
            "predicted_span_sha256": "hash-1",
            "selected_paragraph_idx": 0,
            "invalid_fail_closed_used": False,
            "competition_complete": True,
            "competition": {
                "valid_paragraph_rows": 3,
                "top1_top2_gap_scaled": 0.1,
                "top1_softmax_mass": 0.5,
            },
            "competition_sha256": "competition-1",
        }
    ]
    experiment.apply_hop_results(
        prepared, states, aggregated, {}, hop_index=1
    )
    next_inputs, blocked = experiment.build_hop_qa_inputs(
        prepared, states, hop_index=2
    )
    assert not blocked
    assert {row["question"] for row in next_inputs} == {"Second Alpha bridge?"}
    assert len(states["m70k-case"]["competition"]) == 1


def test_incomplete_competition_fails_chain_closed() -> None:
    prepared = _prepared()
    states = experiment.initial_chain_states(prepared)
    aggregated = [
        {
            "case_id": "m70k-case",
            "score_margin": 3.0,
            "predicted_span": "Alpha bridge",
            "predicted_span_sha256": "hash-1",
            "selected_paragraph_idx": 0,
            "invalid_fail_closed_used": False,
            "competition_complete": False,
            "competition": {
                "valid_paragraph_rows": 1,
                "top1_top2_gap_scaled": None,
                "top1_softmax_mass": None,
            },
            "competition_sha256": "competition-1",
        }
    ]
    [decision] = experiment.apply_hop_results(
        prepared, states, aggregated, {}, hop_index=1
    )
    assert decision["valid_span"] is False
    assert states["m70k-case"]["alive"] is False
    assert states["m70k-case"]["invalid_output"] is True


def test_finalize_feature_decision_builds_frozen_feature_union() -> None:
    prepared = _prepared(hops=2)
    states = experiment.initial_chain_states(prepared)
    states["m70k-case"].update(
        {
            "alive": True,
            "completed_hops": 2,
            "margins": [2.0, 1.0],
            "selected_paragraphs": [0, 1],
            "transition_matches": [True],
            "competition": [
                {"top1_top2_gap_scaled": 0.4, "top1_softmax_mass": 0.7},
                {"top1_top2_gap_scaled": 0.2, "top1_softmax_mass": 0.6},
            ],
        }
    )
    direct = [
        {
            "case_id": "m70k-case",
            "score_margin": 4.0,
            "selected_paragraph_idx": 1,
            "competition_complete": True,
            "competition": {
                "top1_top2_gap_scaled": 0.5,
                "top1_softmax_mass": 0.8,
            },
        }
    ]
    hops = [
        {
            "case_id": "m70k-case",
            "hop_index": index,
            "executed": True,
            "valid_span": True,
        }
        for index in range(1, 3)
    ]
    [result] = experiment.finalize_feature_decisions(
        prepared, direct, states, hops
    )
    assert result["feature_complete"] is True
    assert tuple(result["features"]) == experiment.ALL_FEATURE_NAMES
    assert result["features"]["chain_min_top1_top2_gap_scaled"] == 0.2
    assert result["features"]["chain_mean_top1_softmax_mass"] == pytest.approx(0.65)


def test_gold_join_imputes_invalid_output_with_frozen_fail_closed_values() -> None:
    source = _source("case", True, hops=3)
    prepared = _prepared(hops=3)
    feature = {
        "case_id": "m70k-case",
        "direct_score_margin": None,
        "chain_bottleneck_score": None,
        "chain_mean_score": None,
        "features": {name: None for name in experiment.ALL_FEATURE_NAMES},
        "feature_sha256": "blind-null-feature-hash",
        "feature_complete": False,
        "executed_hops": 1,
        "valid_hops": 0,
        "hop_decision_sha256": "hop-hash",
        "invalid_fail_closed_used": True,
    }
    [evidence] = experiment.build_gold_evidence([source], prepared, [feature])
    assert evidence["direct_score_margin"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert evidence["features"]["direct_margin_scaled"] == -1.0
    assert evidence["features"]["hop_count_is_3"] == 1.0
    assert all(
        evidence["features"][name] == 0.0
        for name in experiment.COMPETITION_FEATURE_NAMES
    )
    assert evidence["invalid_feature_output"] is True


def test_projected_logistic_is_deterministic_and_nonnegative_constrained() -> None:
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
        for name in first["monotone_nonnegative_feature_names"]
    )


def test_feature_contracts_have_frozen_capacities() -> None:
    assert len(experiment.BASE_FEATURE_NAMES) == 9
    assert len(experiment.ADDITIVE_FEATURE_NAMES) == 27
    assert len(experiment.INTERACTION_FEATURE_NAMES) == 32
    assert len(experiment.COMPETITION_FEATURE_NAMES) == 6
    assert len(experiment.CANDIDATE_FEATURE_NAMES) == 15
    assert sum((2, 11, 29, 34, 8, 17)) == 101


def test_crossfit_model_covers_each_case_once_without_held_out_training() -> None:
    evidence = _separable_evidence()
    result = experiment._crossfit_model(
        evidence,
        feature_names=experiment.CANDIDATE_FEATURE_NAMES,
    )
    assert result["held_out_cases"] == len(evidence)
    assert sum(row["held_out_cases"] for row in result["folds"]) == len(evidence)
    assert all(
        row["train_cases"] + row["held_out_cases"] == len(evidence)
        for row in result["folds"]
    )


def test_threshold_selection_obeys_strict_greater_and_safety_constraints() -> None:
    labels = np.asarray([1.0, 1.0, 0.0, 0.0])
    scores = np.asarray([0.8, 0.5, 0.5, 0.1])
    selected = experiment.select_threshold_from_scores(labels, scores)
    threshold = float(selected["threshold_exact"])
    metrics = experiment.score_metrics(labels, scores, threshold)
    assert metrics["threshold_exact"] == threshold
    assert bool((scores > threshold)[1]) is False


def test_evaluate_stage_requires_all_three_incremental_gates() -> None:
    # The formal development gate also freezes the exact 400/400 census, so
    # exercise the complete gate contract rather than only the metric clauses.
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
        calibration=_frozen_calibration(competition_only_perfect=True),
        sampling=sampling,
        structural_census={},
        source_artifacts={},
    )
    assert failed["analysis"]["support_checks"][
        "candidate_minus_competition_only_control_at_least_0_02"
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
        "musique_paragraph_competition_support_protocol_v70.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value["stopping_and_outcomes"]["gate_2"] = "GO"
    target = tmp_path / "protocol.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate 2"):
        experiment.validate_protocol(target)


def test_runner_select_parses_squad2_json_before_question_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    squad_path = tmp_path / "squad.json"
    squad_path.write_text('{"data": []}', encoding="utf-8")
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        runner, "_stage_target", lambda _args: (Path("unused.jsonl"), "train")
    )
    monkeypatch.setattr(runner, "_load_excluded", lambda _args: set())

    def extract(source: dict) -> set[str]:
        seen["source"] = source
        return {"registered question"}

    monkeypatch.setattr(runner.experiment, "extract_squad2_questions", extract)
    monkeypatch.setattr(
        runner.experiment,
        "select_stage_sample",
        lambda _rows, **_kwargs: ([], {}),
    )
    args = SimpleNamespace(squad2_train=squad_path, stage="calibration")

    assert runner._select(args) == ([], {})
    assert seen["source"] == {"data": []}
