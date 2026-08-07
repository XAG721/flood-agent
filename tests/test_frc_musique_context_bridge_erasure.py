from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from research.frc_rag import musique_context_bridge_erasure as experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepared(
    *,
    template: str = "Where was #1 born?",
    contexts: tuple[str, ...] = (
        "Ada Lovelace was born in London.",
        "London is the capital of England.",
    ),
) -> list[dict[str, object]]:
    return [
        {
            "id": "m73c-case",
            "oracle_hop_templates": ["Who is the person?", template],
            "contexts": [
                {"paragraph_idx": index, "context": context}
                for index, context in enumerate(contexts)
            ],
        }
    ]


def _states() -> dict[str, dict[str, object]]:
    return {
        "m73c-case": {
            "alive": True,
            "completed_hops": 2,
            "predictions": {1: "Ada Lovelace", 2: "London"},
            "margins": [3.0, 2.0],
            "selected_paragraphs": [0, 0],
        }
    }


def _prior_feature_decision(*, complete: bool = True) -> dict[str, object]:
    features = {name: 0.0 for name in experiment.v72.ALL_FEATURE_NAMES}
    return {
        "case_id": "m73c-case",
        "hop_count": 2,
        "executed_hops": 2,
        "valid_hops": 2,
        "sentinel_transitions": 1,
        "rival_dependency_count": 1,
        "rival_eligible_count": 1,
        "rival_transitions": 1,
        "feature_complete": complete,
        "direct_score_margin": 1.0 if complete else None,
        "chain_bottleneck_score": 1.0 if complete else None,
        "chain_mean_score": 1.5 if complete else None,
        "features": features if complete else {name: None for name in features},
        "feature_sha256": "prior-feature",
        "factual_feature_sha256": "factual-feature",
        "sentinel_decision_sha256": "sentinel",
        "rival_decision_sha256": "rival",
        "invalid_fail_closed_used": not complete,
        "hop_decision_sha256": "hops",
    }


def _erasure_decision(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "root_case_id": "m73c-case",
        "hop_index": 2,
        "dependency_hop_index": 1,
        "score_margin": 1.0,
        "selected_paragraph_idx": 1,
        "predicted_span": "Paris",
        "predicted_span_sha256": "span",
        "invalid_fail_closed_used": False,
        "matched_occurrences": 1,
        "matched_paragraphs": 1,
    }
    value.update(overrides)
    return value


def test_protocol_freezes_v73_feature_capacity_and_no_go() -> None:
    protocol = experiment.validate_protocol(
        REPO_ROOT
        / "docs/progressive_upgrade/musique_context_bridge_erasure_protocol_v73.json"
    )

    assert protocol["experiment_id"] == experiment.EXPERIMENT_ID
    assert tuple(protocol["feature_contract"]["fixed_order"]) == (
        experiment.ERASURE_FEATURE_NAMES
    )
    assert protocol["model_contract"][
        "total_learned_scalars_across_all_reported_methods"
    ] == 152
    assert protocol["sampling"][
        "expected_prior_source_commitment_union_count"
    ] == 14500
    assert protocol["stopping_and_outcomes"]["gate_2"] == "NO-GO/SHADOW"


def test_feature_orders_preserve_all_prior_features() -> None:
    assert len(experiment.BASE_FEATURE_NAMES) == 9
    assert len(experiment.ERASURE_FEATURE_NAMES) == 6
    assert len(experiment.CANDIDATE_FEATURE_NAMES) == 15
    assert len(experiment.ALL_FEATURE_NAMES) == 56
    assert experiment.ALL_FEATURE_NAMES[: len(experiment.v72.ALL_FEATURE_NAMES)] == (
        experiment.v72.ALL_FEATURE_NAMES
    )


def test_registered_prior_paths_are_loaded_as_jsonl_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        '{"source_id_commitment":"a"}\n'
        '{"source_id_commitment":"b"}\n',
        encoding="utf-8",
    )
    second.write_text(
        '{"source_id_commitment":"b"}\n'
        '{"source_id_commitment":"c"}\n',
        encoding="utf-8",
    )

    commitments = experiment.load_source_commitments_from_paths([first, second])

    assert commitments == {"a", "b", "c"}


def test_erasure_is_case_insensitive_and_uses_mask() -> None:
    text, count = experiment._erase_bounded_mentions(
        "ADA LOVELACE met Ada Lovelace.", "Ada Lovelace"
    )

    assert text == "<mask> met <mask>."
    assert count == 2


def test_erasure_rejects_partial_alphanumeric_matches() -> None:
    text, count = experiment._erase_bounded_mentions(
        "US policy, RUS and USA", "US"
    )

    assert text == "<mask> policy, RUS and USA"
    assert count == 1


def test_erasure_allows_punctuation_bounded_surface() -> None:
    text, count = experiment._erase_bounded_mentions(
        "The C++ language differs from C.", "C++"
    )

    assert text == "The <mask> language differs from C."
    assert count == 1


def test_empty_surface_is_never_erased() -> None:
    assert experiment._erase_bounded_mentions("evidence", "  ") == (
        "evidence",
        0,
    )


def test_build_erasure_inputs_holds_factual_question_fixed() -> None:
    manifest, inputs = experiment.build_context_erasure_qa_inputs(
        _prepared(), _states()
    )

    assert len(manifest) == 1
    assert manifest[0]["matched_occurrences"] == 1
    assert manifest[0]["matched_paragraphs"] == 1
    assert len(inputs) == 2
    assert {row["question"] for row in inputs} == {
        "Where was Ada Lovelace born?"
    }
    assert inputs[0]["context"] == "<mask> was born in London."
    assert inputs[1]["context"] == "London is the capital of England."


def test_no_context_occurrence_creates_manifest_without_qa_inputs() -> None:
    manifest, inputs = experiment.build_context_erasure_qa_inputs(
        _prepared(contexts=("No matching person appears here.",)), _states()
    )

    assert len(manifest) == 1
    assert manifest[0]["matched_occurrences"] == 0
    assert inputs == []


def test_each_dependency_receives_a_separate_erasure_transition() -> None:
    prepared = _prepared(
        template="Did #1 meet #2?",
        contexts=("Ada Lovelace met Charles Babbage.",),
    )
    states = _states()
    states["m73c-case"]["predictions"] = {
        1: "Ada Lovelace",
        2: "Charles Babbage",
        3: "yes",
    }
    states["m73c-case"]["completed_hops"] = 2

    manifest, inputs = experiment.build_context_erasure_qa_inputs(
        prepared, states
    )

    assert [row["dependency_hop_index"] for row in manifest] == [1]
    assert len(inputs) == 1


def test_erasure_inputs_and_manifest_do_not_export_gold() -> None:
    manifest, inputs = experiment.build_context_erasure_qa_inputs(
        _prepared(), _states()
    )

    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    }
    assert not forbidden & experiment.nested_keys([*manifest, *inputs])


def test_aggregate_erasure_rows_preserves_transition_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        {
            "id": "transition::p0",
            "case_id": "transition",
            "root_case_id": "m73c-case",
            "hop_index": 2,
            "dependency_hop_index": 1,
            "matched_occurrences": 2,
            "matched_paragraphs": 1,
        }
    ]
    aggregated = [
        {
            "case_id": "transition",
            "score_margin": 1.0,
            "selected_paragraph_idx": 0,
        }
    ]
    monkeypatch.setattr(experiment, "aggregate_qa_rows", lambda *args, **kwargs: aggregated)

    result = experiment.aggregate_erasure_qa_rows(
        inputs, [], cache_key="cache"
    )

    assert result[0]["root_case_id"] == "m73c-case"
    assert result[0]["matched_occurrences"] == 2
    assert result[0]["schema_version"] == "frc-musique-v73-erasure-decision-v1"


def test_finalize_computes_directional_erasure_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment.v72,
        "finalize_feature_decisions",
        lambda *args, **kwargs: [_prior_feature_decision()],
    )
    manifest = [
        {
            "root_case_id": "m73c-case",
            "hop_index": 2,
            "dependency_hop_index": 1,
            "matched_occurrences": 1,
        }
    ]

    result = experiment.finalize_feature_decisions(
        _prepared(),
        [],
        _states(),
        [],
        [],
        [],
        [],
        manifest,
        [_erasure_decision()],
    )[0]

    assert result["feature_complete"] is True
    assert result["features"]["erasure_dependency_coverage_fraction"] == 1.0
    assert result["features"]["erasure_margin_drop_min"] == 0.1
    assert result["features"]["erasure_factual_win_fraction"] == 1.0
    assert result["features"]["erasure_paragraph_change_fraction"] == 1.0
    assert result["features"]["erasure_span_change_fraction"] == 1.0


def test_finalize_no_occurrence_uses_frozen_zero_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment.v72,
        "finalize_feature_decisions",
        lambda *args, **kwargs: [_prior_feature_decision()],
    )
    manifest = [
        {
            "root_case_id": "m73c-case",
            "hop_index": 2,
            "dependency_hop_index": 1,
            "matched_occurrences": 0,
        }
    ]

    result = experiment.finalize_feature_decisions(
        _prepared(), [], _states(), [], [], [], [], manifest, []
    )[0]

    assert result["feature_complete"] is True
    assert [result["features"][name] for name in experiment.ERASURE_FEATURE_NAMES] == [
        0.0
    ] * 6


def test_finalize_requires_every_eligible_erasure_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment.v72,
        "finalize_feature_decisions",
        lambda *args, **kwargs: [_prior_feature_decision()],
    )
    manifest = [
        {
            "root_case_id": "m73c-case",
            "hop_index": 2,
            "dependency_hop_index": 1,
            "matched_occurrences": 1,
        }
    ]

    result = experiment.finalize_feature_decisions(
        _prepared(), [], _states(), [], [], [], [], manifest, []
    )[0]

    assert result["feature_complete"] is False
    assert result["invalid_fail_closed_used"] is True


def test_gold_evidence_fail_closes_incomplete_features() -> None:
    selected = [
        {
            "answerable": False,
            "question_decomposition": [{}, {}],
            "paragraphs": [{}, {}],
        }
    ]
    feature = {
        **_prior_feature_decision(complete=False),
        "erasure_dependency_count": 1,
        "erasure_eligible_count": 1,
        "erasure_transitions": 0,
        "erasure_manifest_sha256": "manifest",
        "erasure_decision_sha256": "erasure",
    }

    evidence = experiment.build_gold_evidence(
        selected, _prepared(), [feature]
    )[0]

    assert evidence["invalid_feature_output"] is True
    assert evidence["direct_score_margin"] == experiment.FAIL_CLOSED_RAW_SCORE
    assert set(evidence["features"]) == set(experiment.ALL_FEATURE_NAMES)


def test_projected_logistic_keeps_erasure_coefficients_nonnegative() -> None:
    evidence = [
        {
            "answer_state": "answerable" if index % 2 else "unanswerable",
            "features": {
                name: float((index + offset) % 3) / 2.0
                for offset, name in enumerate(experiment.CANDIDATE_FEATURE_NAMES)
            },
        }
        for index in range(20)
    ]

    model = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    coefficients = dict(
        zip(model["feature_names"], model["coefficients"], strict=True)
    )

    assert all(
        coefficients[name] >= 0.0 for name in experiment.ERASURE_FEATURE_NAMES
    )


def test_evaluation_fails_closed_when_erasure_coverage_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = []
    for index in range(800):
        features = {name: 0.0 for name in experiment.ALL_FEATURE_NAMES}
        evidence.append(
            {
                "case_id": f"case-{index}",
                "answer_state": "answerable" if index < 400 else "unanswerable",
                "hop_count": 2,
                "direct_score_margin": 0.0,
                "chain_bottleneck_score": 0.0,
                "features": features,
                "invalid_feature_output": False,
            }
        )
    calibration: dict[str, object] = {
        "calibrated_direct_composed_question": {"pooled": {"threshold_exact": 0.0}},
        "calibrated_chain_bottleneck": {"pooled": {"threshold_exact": 0.0}},
    }
    for name, feature_names in experiment.MODEL_FEATURE_CONTRACTS.items():
        calibration[name] = {
            "pooled": {"threshold_exact": 0.5},
            "model": {
                "feature_names": list(feature_names),
                "coefficients": [0.0] * len(feature_names),
                "intercept": 0.0,
            },
        }
    monkeypatch.setattr(
        experiment,
        "paired_correctness_interval",
        lambda *args, **kwargs: {
            "point": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "resamples": 10000,
            "seed": kwargs["seed"],
        },
    )

    result = experiment.evaluate_stage(
        evidence,
        stage="development",
        calibration=calibration,
        sampling={
            "schema_exclusion_rate": 0.0,
            "selected_excluded_source_commitment_overlap": 0,
            "selected_squad2_exact_question_overlap": 0,
        },
        structural_census={},
        source_artifacts={},
    )

    assert result["analysis"]["mean_erasure_dependency_coverage_fraction"] == 0.0
    assert result["analysis"]["support_checks"][
        "mean_erasure_dependency_coverage_fraction_at_least_0_50"
    ] is False
    assert result["analysis"]["outcome"]["stage_gate_passed"] is False


def test_write_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    rows = [{"case_id": "case", "features": {"x": 1.0}}]

    experiment.write_evidence(first, rows)
    experiment.write_evidence(second, rows)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == rows[0]
