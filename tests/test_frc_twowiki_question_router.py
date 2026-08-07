from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    CANDIDATE_METHOD,
    CONTROL_METHODS,
    FEATURE_DIMENSION,
    HAND_TERMS,
    ROUTER_THRESHOLD,
    SOFT_METHOD,
    fit_logistic,
    load_model_artifact,
    predict_logistic,
    question_features,
    select_method,
    select_stage_ids,
    serialize_model,
    validate_finite,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_support_path_closure import (
    QUESTION_TYPES,
    canonical_json_sha256,
    read_jsonl_gzip,
    sha256,
    write_jsonl,
)
from research.frc_rag.twowiki_question_router import evaluate_stage


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
PROTOCOL_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_protocol_v75.json"
)
IMPLEMENTATION_LOCK_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_implementation_v75.json"
)


def _candidate(
    candidate_id: str,
    source: str,
    text: str,
    cross: float,
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": text,
        "token_count": 10,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": {
            "condition": cross,
            "attribution": cross,
            "procedure": cross,
            "answer": cross,
            "exception": cross,
        },
    }


def _scored_row(case_id: str = "case-1") -> dict[str, object]:
    return {
        "dataset": "2WikiMultiHopQA",
        "source": "synthetic",
        "id": case_id,
        "question": "Which was founded earlier, Alpha or Beta?",
        "question_type": "comparison",
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": [
            _candidate("a", "Alpha", "Alpha. Alpha cites Beta.", 1.0),
            _candidate("b", "Beta", "Beta. Beta was founded in 1900.", 0.2),
            _candidate("c", "Gamma", "Gamma. Unrelated high score.", 0.9),
            _candidate("d", "Delta", "Delta. Another distractor.", 0.8),
            _candidate("e", "Epsilon", "Epsilon. Another distractor.", 0.7),
            _candidate("f", "Zeta", "Zeta. Another distractor.", 0.6),
        ],
    }


def _which_router_model() -> dict[str, object]:
    feature_count = len(question_features("Which one?"))
    weights = np.zeros(feature_count, dtype=np.float64)
    weights[2] = 10.0
    return {
        "means": np.zeros(feature_count, dtype=np.float64),
        "scales": np.ones(feature_count, dtype=np.float64),
        "intercept": -5.0,
        "weights": weights,
        "l2": 8.0,
        "iterations": 1,
        "converged": True,
    }


def test_question_features_are_deterministic_and_have_frozen_shape() -> None:
    first = question_features("Which film was released earlier?")
    second = question_features("Which film was released earlier?")

    assert np.array_equal(first, second)
    assert len(first) == 2 + 6 + len(HAND_TERMS) + FEATURE_DIMENSION
    assert first[2] == 1.0


def test_question_features_reject_nonpositive_hash_dimension() -> None:
    with pytest.raises(ValueError, match="positive"):
        question_features("Question?", dimension=0)


def test_logistic_fit_converges_and_predicts_two_classes() -> None:
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])

    model = fit_logistic(features, labels, l2=0.5)
    predictions = predict_logistic(model, features)

    assert model["converged"] is True
    assert np.all(predictions[:2] < 0.5)
    assert np.all(predictions[2:] > 0.5)


def test_logistic_fit_rejects_single_class_labels() -> None:
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(np.asarray([[0.0], [1.0]]), np.asarray([1.0, 1.0]))


def test_serialized_model_records_actual_hash_dimension() -> None:
    model = _which_router_model()
    extra = np.zeros(64, dtype=np.float64)
    for key in ("means", "scales", "weights"):
        model[key] = np.concatenate([model[key], extra])

    payload = serialize_model(model)

    assert payload["feature_dimension"] == 128
    assert payload["total_feature_count"] == len(model["weights"])


def test_frozen_model_artifact_loads_and_hashes_match() -> None:
    artifact, model = load_model_artifact(MODEL_PATH)

    assert artifact["history_cases"] == 800
    assert artifact["selected_crossfit"]["crossfit_router_balanced_accuracy"] == 1.0
    assert artifact["model_sha256"] == canonical_json_sha256(artifact["model"])
    assert len(model["weights"]) == artifact["model"]["total_feature_count"]


def test_model_artifact_rejects_inconsistent_arrays(tmp_path: Path) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["model"]["weights"] = artifact["model"]["weights"][:-1]
    artifact["model_sha256"] = canonical_json_sha256(artifact["model"])
    path = tmp_path / "bad-model.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="feature arrays"):
        load_model_artifact(path)


def test_stage_selection_is_balanced_deterministic_and_disjoint() -> None:
    rows = [
        {"_id": f"{question_type}-{index:03d}", "type": question_type}
        for question_type in QUESTION_TYPES
        for index in range(12)
    ]
    excluded = {f"{question_type}-000" for question_type in QUESTION_TYPES}

    development = select_stage_ids(
        rows, excluded_ids=excluded, stage="development", quota_per_type=3
    )
    repeated = select_stage_ids(
        rows, excluded_ids=excluded, stage="development", quota_per_type=3
    )
    confirmation = select_stage_ids(
        rows, excluded_ids=excluded, stage="confirmation", quota_per_type=3
    )

    assert development == repeated
    assert len(development) == len(confirmation) == 12
    assert not set(development) & excluded
    assert not set(confirmation) & excluded
    assert not set(development) & set(confirmation)
    assert {
        name: sum(case_id.startswith(f"{name}-") for case_id in development)
        for name in QUESTION_TYPES
    } == {name: 3 for name in QUESTION_TYPES}


def test_stage_selection_rejects_duplicate_ids() -> None:
    rows = [{"_id": "same", "type": "comparison"}] * 2

    with pytest.raises(ValueError, match="duplicated"):
        select_stage_ids(
            rows, excluded_ids=set(), stage="development", quota_per_type=1
        )


def test_runtime_route_depends_on_question_not_official_type() -> None:
    model = _which_router_model()
    comparison_row = _scored_row()
    mislabeled_row = {**comparison_row, "question_type": "inference"}

    _, first = select_method(comparison_row, CANDIDATE_METHOD, model)
    _, second = select_method(mislabeled_row, CANDIDATE_METHOD, model)

    assert first == second
    assert first["route"] == ANCHOR_METHOD
    assert first["comparison_probability"] >= ROUTER_THRESHOLD


def test_runtime_router_uses_soft_route_for_non_which_question() -> None:
    row = {**_scored_row(), "question": "Where was Alpha founded?"}

    _, routing = select_method(row, CANDIDATE_METHOD, _which_router_model())

    assert routing["route"] == SOFT_METHOD
    assert routing["comparison_probability"] < ROUTER_THRESHOLD


def test_all_frozen_controls_execute() -> None:
    row = _scored_row()
    model = _which_router_model()

    for method in CONTROL_METHODS:
        selected, routing = select_method(row, method, model)
        assert 1 <= len(selected) <= 5
        assert routing["route"]


def test_selection_output_rejects_gold_bearing_rows(tmp_path: Path) -> None:
    row = {**_scored_row(), "answer": "secret"}

    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs([row], _which_router_model(), tmp_path / "out.jsonl")


def test_selection_output_contains_no_question_type_or_candidate_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.jsonl"

    write_selection_outputs([_scored_row()], _which_router_model(), path)
    payload = path.read_text(encoding="utf-8").lower()

    assert '"question"' not in payload
    assert '"question_type"' not in payload
    assert "founded" not in payload


def test_small_evaluation_writes_safe_evidence_and_fails_size_gate(
    tmp_path: Path,
) -> None:
    scored_path = tmp_path / "scored.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    rows = []
    gold = []
    for index, question_type in enumerate(QUESTION_TYPES):
        row = _scored_row(f"case-{index}")
        row["question_type"] = question_type
        rows.append(row)
        gold.append(
            {
                "id": row["id"],
                "question_type": question_type,
                "gold_evidence_ids": ["a", "b"],
            }
        )
    write_jsonl(scored_path, rows)
    write_jsonl(gold_path, gold)

    report = evaluate_stage(
        scored_path,
        gold_path,
        selection_path,
        MODEL_PATH,
        stage="development",
        seed=7,
        output_dir=tmp_path / "output",
        prior_overlap=0,
    )
    cases = read_jsonl_gzip(tmp_path / "output/cases.jsonl.gz")

    assert report["metadata"]["selection_written_before_gold_join"] is True
    assert report["metadata"]["official_type_used_by_runtime_router"] is False
    assert report["analysis"]["support_checks"]["exact_cases_equals_800"] is False
    assert all("question" not in row and "answer" not in row for row in cases)


def test_registered_protocol_matches_implementation() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    validate_registered_protocol(protocol)


def test_implementation_lock_hashes_all_frozen_files() -> None:
    implementation_lock = json.loads(
        IMPLEMENTATION_LOCK_PATH.read_text(encoding="utf-8")
    )

    for contract in implementation_lock["files"].values():
        assert sha256(ROOT / contract["path"]) == contract["sha256"]


def test_protocol_validation_rejects_gate_drift() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["development_gates"]["candidate_evidence_macro_f1_at_least"] = 0.52

    with pytest.raises(ValueError, match="gates drifted"):
        validate_registered_protocol(protocol)


def test_values_must_be_finite() -> None:
    assert validate_finite({"values": [0.0, 1.0]}) is True
    assert validate_finite({"values": [float("nan")]}) is False
