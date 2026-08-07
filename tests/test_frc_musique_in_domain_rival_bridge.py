from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research.frc_rag import musique_in_domain_rival_bridge as experiment


def _state(case_id: str = "m72k-case") -> dict[str, dict[str, object]]:
    return {
        case_id: {
            "alive": True,
            "completed_hops": 3,
            "predictions": {1: "Alice", 2: "Paris", 3: "France"},
            "margins": [2.0, 3.0, 4.0],
            "selected_paragraphs": [0, 1, 2],
        }
    }


def _prepared(case_id: str = "m72k-case") -> dict[str, object]:
    return {
        "id": case_id,
        "oracle_hop_templates": [
            "Who founded the group?",
            "Where was #1 born?",
            "Which country contains #2 and is associated with #1?",
        ],
        "contexts": [
            {"paragraph_idx": 0, "context": "Alice founded the group."},
            {"paragraph_idx": 1, "context": "Bob was born in Paris."},
            {"paragraph_idx": 2, "context": "France contains Paris."},
        ],
    }


def _raw(
    row_id: str,
    context: str,
    span: str,
    margin: float,
    *,
    cache_key: str = "cache",
) -> dict[str, object]:
    start = context.index(span)
    return {
        "id": row_id,
        "schema_version": "frc-musique-v72-raw-qa-v1",
        "cache_key": cache_key,
        "score_margin": margin,
        "span_start": start,
        "span_end": start + len(span),
        "span_sha256": experiment._hash(span),
        "invalid_fail_closed_used": False,
    }


def _decision(
    *,
    case_id: str = "m72k-case",
    hop_index: int = 1,
    paragraph_idx: int = 0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "hop_index": hop_index,
        "valid_span": True,
        "selected_paragraph_idx": paragraph_idx,
    }


def _counterfactual_decision(
    *,
    root_case_id: str = "m72k-case",
    hop_index: int,
    margin: float,
    paragraph_idx: int,
    span: str,
    dependency_hop_index: int | None = None,
) -> dict[str, object]:
    return {
        "root_case_id": root_case_id,
        "hop_index": hop_index,
        "dependency_hop_index": dependency_hop_index,
        "score_margin": margin,
        "selected_paragraph_idx": paragraph_idx,
        "predicted_span": span,
        "predicted_span_sha256": experiment._hash(span),
        "invalid_fail_closed_used": False,
    }


def test_frozen_feature_and_capacity_contract() -> None:
    assert len(experiment.BASE_FEATURE_NAMES) == 9
    assert len(experiment.SENTINEL_ONLY_FEATURE_NAMES) == 6
    assert len(experiment.RIVAL_FEATURE_NAMES) == 6
    assert len(experiment.CANDIDATE_FEATURE_NAMES) == 15
    assert len(experiment.ALL_FEATURE_NAMES) == 50
    assert sum((2, 11, 29, 34, 17, 17, 8, 17)) == 135


def test_protocol_validation_accepts_frozen_registration() -> None:
    protocol = experiment.validate_protocol(
        Path(
            "docs/progressive_upgrade/"
            "musique_in_domain_rival_bridge_protocol_v72.json"
        )
    )
    assert protocol["experiment_id"] == experiment.EXPERIMENT_ID
    assert protocol["sampling"]["expected_prior_source_commitment_union_count"] == 12100


def test_protocol_validation_rejects_capacity_change(tmp_path: Path) -> None:
    source = Path(
        "docs/progressive_upgrade/musique_in_domain_rival_bridge_protocol_v72.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value["model_contract"]["total_learned_scalars_across_all_reported_methods"] = 134
    target = tmp_path / "protocol.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="capacity"):
        experiment.validate_protocol(target)


def test_placeholder_indices_are_unique_ordered_and_prior_only() -> None:
    assert experiment._placeholder_indices("#2 with #1 and #2 then #4", 4) == [1, 2]


def test_rival_catalog_selects_highest_margin_different_paragraph_span() -> None:
    case_id = "m72k-case"
    inputs = [
        {
            "id": f"{case_id}::hop1::p1",
            "case_id": case_id,
            "paragraph_idx": 1,
            "context": "Bob was born in Paris.",
        },
        {
            "id": f"{case_id}::hop1::p2",
            "case_id": case_id,
            "paragraph_idx": 2,
            "context": "Carol lives in Rome.",
        },
    ]
    raw = [
        _raw(inputs[0]["id"], inputs[0]["context"], "Bob", 1.5),
        _raw(inputs[1]["id"], inputs[1]["context"], "Carol", 2.5),
    ]
    result = experiment.build_rival_catalog(
        inputs,
        raw,
        [_decision()],
        _state(),
        hop_index=1,
    )
    assert len(result) == 1
    assert result[0]["rival_span"] == "Carol"
    assert result[0]["paragraph_idx"] == 2


def test_rival_catalog_validates_raw_offset_hash_before_trimming() -> None:
    case_id = "m72k-case"
    context = "Bob\n"
    row_id = f"{case_id}::hop1::p1"
    result = experiment.build_rival_catalog(
        [
            {
                "id": row_id,
                "case_id": case_id,
                "paragraph_idx": 1,
                "context": context,
            }
        ],
        [_raw(row_id, context, "Bob\n", 2.0)],
        [_decision()],
        _state(),
        hop_index=1,
    )
    assert result[0]["rival_span"] == "Bob"
    assert result[0]["rival_span_sha256"] == experiment._hash("Bob")


def test_rival_catalog_rejects_normalized_factual_duplicate() -> None:
    case_id = "m72k-case"
    context = "ALICE appears again."
    inputs = [
        {
            "id": f"{case_id}::hop1::p1",
            "case_id": case_id,
            "paragraph_idx": 1,
            "context": context,
        }
    ]
    result = experiment.build_rival_catalog(
        inputs,
        [_raw(inputs[0]["id"], context, "ALICE", 5.0)],
        [_decision()],
        _state(),
        hop_index=1,
    )
    assert result == []


def test_rival_catalog_rejects_selected_factual_paragraph() -> None:
    case_id = "m72k-case"
    context = "Bob appears here."
    inputs = [
        {
            "id": f"{case_id}::hop1::p0",
            "case_id": case_id,
            "paragraph_idx": 0,
            "context": context,
        }
    ]
    result = experiment.build_rival_catalog(
        inputs,
        [_raw(inputs[0]["id"], context, "Bob", 5.0)],
        [_decision()],
        _state(),
        hop_index=1,
    )
    assert result == []


def test_rival_question_changes_one_dependency_and_keeps_others_factual() -> None:
    catalog = [
        {
            "case_id": "m72k-case",
            "hop_index": 1,
            "paragraph_idx": 2,
            "rival_span": "Carol",
            "rival_span_sha256": experiment._hash("Carol"),
        },
        {
            "case_id": "m72k-case",
            "hop_index": 2,
            "paragraph_idx": 0,
            "rival_span": "Rome",
            "rival_span_sha256": experiment._hash("Rome"),
        },
    ]
    result = experiment.build_rival_qa_inputs([_prepared()], _state(), catalog)
    hop3 = [row for row in result if row["hop_index"] == 3]
    assert len(hop3) == 6
    questions = {row["dependency_hop_index"]: row["question"] for row in hop3}
    assert questions[1] == "Which country contains Paris and is associated with Carol?"
    assert questions[2] == "Which country contains Rome and is associated with Alice?"


def test_no_rival_means_no_rival_query() -> None:
    assert experiment.build_rival_qa_inputs([_prepared()], _state(), []) == []


def test_sentinel_question_retains_zero_substitution_semantics() -> None:
    prepared = _prepared()
    prepared["oracle_hop_templates"][1] = "Question with no placeholder"
    result = experiment.build_sentinel_qa_inputs([prepared], _state())
    hop2 = [row for row in result if row["hop_index"] == 2]
    assert len(hop2) == 3
    assert {row["question"] for row in hop2} == {"Question with no placeholder"}
    assert {row["placeholder_substitutions"] for row in hop2} == {0}


def test_invalid_rival_output_is_maximum_dependence() -> None:
    value = {
        "score_margin": None,
        "predicted_span": "",
        "invalid_fail_closed_used": True,
    }
    assert experiment._counterfactual_statistics(
        value,
        factual_margin=2.0,
        factual_paragraph=0,
        factual_span="Alice",
    ) == (1.0, 1.0, 1.0, 1.0, False)


def test_valid_rival_statistics_are_directional_and_bounded() -> None:
    value = {
        "score_margin": -20.0,
        "predicted_span": "Bob",
        "selected_paragraph_idx": 2,
        "invalid_fail_closed_used": False,
    }
    result = experiment._counterfactual_statistics(
        value,
        factual_margin=5.0,
        factual_paragraph=0,
        factual_span="Alice",
    )
    assert result == (1.0, 1.0, 1.0, 1.0, True)


def test_finalize_features_combines_sentinel_and_rival_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    base_features = {name: 0.0 for name in experiment.v71.v70.ALL_FEATURE_NAMES}
    base = {
        "feature_complete": True,
        "features": base_features,
        "feature_sha256": experiment._hash("base"),
        "hop_count": 3,
        "executed_hops": 3,
        "valid_hops": 3,
        "direct_score_margin": 1.0,
        "chain_bottleneck_score": 2.0,
        "chain_mean_score": 3.0,
        "invalid_fail_closed_used": False,
        "hop_decision_sha256": experiment._hash("hops"),
    }
    monkeypatch.setattr(
        experiment.v71.v70,
        "finalize_feature_decisions",
        lambda *args, **kwargs: [base],
    )
    sentinels = [
        _counterfactual_decision(
            hop_index=2, margin=2.0, paragraph_idx=2, span="Lyon"
        ),
        _counterfactual_decision(
            hop_index=3, margin=3.0, paragraph_idx=0, span="Spain"
        ),
    ]
    rivals = [
        _counterfactual_decision(
            hop_index=2,
            dependency_hop_index=1,
            margin=1.0,
            paragraph_idx=2,
            span="Lyon",
        ),
        _counterfactual_decision(
            hop_index=3,
            dependency_hop_index=1,
            margin=2.0,
            paragraph_idx=0,
            span="Spain",
        ),
        _counterfactual_decision(
            hop_index=3,
            dependency_hop_index=2,
            margin=5.0,
            paragraph_idx=2,
            span="France",
        ),
    ]
    catalog = [
        {"case_id": "m72k-case", "hop_index": 1},
        {"case_id": "m72k-case", "hop_index": 2},
    ]
    result = experiment.finalize_feature_decisions(
        [_prepared()], [], _state(), [], sentinels, rivals, catalog
    )[0]
    assert result["feature_complete"] is True
    assert result["rival_dependency_count"] == 3
    assert result["rival_eligible_count"] == 3
    assert result["features"]["rival_dependency_coverage_fraction"] == 1.0
    assert result["features"]["rival_margin_advantage_min"] == 0.0
    assert math.isclose(
        result["features"]["rival_margin_advantage_mean"],
        (0.2 + 0.2 + 0.0) / 3.0,
    )
    assert result["features"]["rival_factual_win_fraction"] == pytest.approx(2 / 3)


def test_fail_closed_vector_covers_all_fifty_features() -> None:
    features = experiment._fail_closed_features(3)
    assert tuple(features) == experiment.ALL_FEATURE_NAMES
    assert len(features) == 50
    assert all(math.isfinite(value) for value in features.values())


def test_projected_logistic_keeps_monotone_coefficients_nonnegative() -> None:
    evidence = []
    for index in range(12):
        features = {name: 0.0 for name in experiment.ALL_FEATURE_NAMES}
        features["rival_margin_advantage_mean"] = index / 11
        evidence.append(
            {
                "answer_state": "answerable" if index >= 6 else "unanswerable",
                "features": features,
            }
        )
    model = experiment.fit_projected_logistic(
        evidence, feature_names=experiment.CANDIDATE_FEATURE_NAMES
    )
    coefficients = dict(zip(model["feature_names"], model["coefficients"], strict=True))
    assert all(
        coefficients[name] >= 0.0
        for name in experiment.CANDIDATE_FEATURE_NAMES
        if name not in experiment.UNCONSTRAINED_FEATURE_NAMES
    )


def test_sample_selection_is_balanced_disjoint_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        experiment.v71.v70.v66, "_valid_row", lambda row: (True, "ok")
    )
    monkeypatch.setattr(
        experiment.v71.v70.v66,
        "_has_squad_overlap",
        lambda row, questions: False,
    )
    rows = [
        {"id": f"a-{index}", "answerable": True} for index in range(40)
    ] + [{"id": f"u-{index}", "answerable": False} for index in range(40)]
    first, report = experiment.select_stage_sample(
        rows,
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=set(),
        target_per_group=3,
    )
    second, _ = experiment.select_stage_sample(
        reversed(rows),
        stage="calibration",
        source_split="train",
        squad_questions=set(),
        excluded_source_commitments=set(),
        target_per_group=3,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert report["selected_answer_state_counts"] == {
        "answerable": 3,
        "unanswerable": 3,
    }
