from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.hotpot_graph_router import (
    CANDIDATE_METHOD,
    COMMON_RESOURCES,
    CONTROL_METHODS,
    DEVELOPMENT_GATES,
    FEATURE_NAMES,
    SOFT_METHOD,
    ZERO_SHOT_METHOD,
    blind_history_row,
    evaluate_stage,
    graph_features,
    load_router_artifact,
    predict_gbr,
    prepare_case,
    select_method,
    select_stage_ids,
    validate_finite,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl_gzip,
    sha256,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/hotpot_graph_router_model_development_v76.json"
)
V75_MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
PROTOCOL_PATH = ROOT / "docs/progressive_upgrade/hotpot_graph_router_protocol_v76.json"
IMPLEMENTATION_LOCK_PATH = (
    ROOT / "docs/progressive_upgrade/hotpot_graph_router_implementation_v76.json"
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
        "metadata": {"title": source},
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
    specifications = (
        ("a", "Alpha", "Alpha. Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta. Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Gamma. Unrelated high score.", 0.9),
        ("d", "Delta", "Delta. Another distractor.", 0.8),
        ("e", "Epsilon", "Epsilon. Another distractor.", 0.7),
        ("f", "Zeta", "Zeta. Another distractor.", 0.6),
    )
    return {
        "dataset": "HotpotQA",
        "source": "synthetic",
        "id": case_id,
        "question": "How is Alpha connected to Beta?",
        "question_type": "bridge",
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*specification) for specification in specifications],
    }


def _constant_router(prediction: float) -> dict[str, object]:
    return {
        "model": {
            "feature_count": len(FEATURE_NAMES),
            "learning_rate": 0.1,
            "init_prediction": prediction,
            "trees": [],
        }
    }


def _v75_model() -> dict[str, object]:
    return load_v75_model_artifact(V75_MODEL_PATH)[1]


def test_graph_features_are_deterministic_finite_and_uniquely_named() -> None:
    row = _scored_row()

    first = graph_features(row, _v75_model())
    second = graph_features(row, _v75_model())

    assert np.array_equal(first, second)
    assert len(first) == len(FEATURE_NAMES) == 71
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert np.isfinite(first).all()


def test_graph_features_ignore_official_question_type_and_gold() -> None:
    row = _scored_row()
    altered = {
        **row,
        "question_type": "comparison",
        "answer": "secret",
        "gold_evidence_ids": ["a", "b"],
    }

    assert np.array_equal(
        graph_features(row, _v75_model()), graph_features(altered, _v75_model())
    )


def test_graph_features_reject_duplicate_candidate_ids() -> None:
    row = _scored_row()
    row["candidates"] = [row["candidates"][0], row["candidates"][0]]

    with pytest.raises(ValueError, match="duplicated"):
        graph_features(row, _v75_model())


def test_blind_history_row_strips_all_gold_fields() -> None:
    row = {
        **_scored_row(),
        "answer": "secret",
        "gold_evidence_ids": ["a"],
    }
    row["candidates"][0]["gold"] = True
    row["candidates"][0]["gold_roles"] = ["answer"]

    blind = blind_history_row(row)
    payload = json.dumps(blind, sort_keys=True)

    assert "secret" not in payload
    assert "gold_evidence_ids" not in payload
    assert '"gold"' not in payload
    assert "gold_roles" not in payload


def test_exported_tree_prediction_uses_frozen_branch_semantics() -> None:
    payload = {
        "feature_count": 2,
        "learning_rate": 0.5,
        "init_prediction": 1.0,
        "trees": [
            {
                "children_left": [1, -1, -1],
                "children_right": [2, -1, -1],
                "features": [0, -2, -2],
                "thresholds": [0.5, -2.0, -2.0],
                "values": [0.0, -2.0, 4.0],
            }
        ],
    }

    predictions = predict_gbr(payload, np.asarray([[0.5, 9.0], [0.6, 9.0]]))

    assert predictions.tolist() == [0.0, 3.0]
    with pytest.raises(ValueError, match="dimension"):
        predict_gbr(payload, np.asarray([[0.5]]))


def test_frozen_router_artifact_loads_and_hashes_match() -> None:
    artifact = load_router_artifact(MODEL_PATH)

    assert artifact["history"]["cases"] == 1000
    assert artifact["crossfit"]["candidate"]["evidence_macro_f1"] == 0.562448
    assert artifact["crossfit"]["candidate_minus_cross_encoder"]["ci_low"] > 0
    assert artifact["model_sha256"] == canonical_json_sha256(artifact["model"])
    assert artifact["selected_configuration"]["feature_names"] == list(FEATURE_NAMES)


def test_router_artifact_rejects_model_and_feature_contract_tampering(
    tmp_path: Path,
) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["model"]["init_prediction"] = 99.0
    bad_model_path = tmp_path / "bad-model.json"
    bad_model_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash"):
        load_router_artifact(bad_model_path)

    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["selected_configuration"]["feature_names"][0] = "changed"
    bad_features_path = tmp_path / "bad-features.json"
    bad_features_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="feature contract"):
        load_router_artifact(bad_features_path)


def test_stage_selection_is_balanced_deterministic_and_disjoint() -> None:
    rows = [
        {"id": f"{question_type}-{index:03d}", "type": question_type}
        for question_type in ("bridge", "comparison")
        for index in range(12)
    ]
    history = {"bridge-000", "comparison-000"}

    development = select_stage_ids(
        rows, history_ids=history, stage="development", quota_per_type=3
    )
    repeated = select_stage_ids(
        rows, history_ids=history, stage="development", quota_per_type=3
    )
    confirmation = select_stage_ids(
        rows, history_ids=history, stage="confirmation", quota_per_type=3
    )

    assert development == repeated
    assert len(development) == len(confirmation) == 6
    assert not set(development) & history
    assert not set(confirmation) & history
    assert not set(development) & set(confirmation)
    assert {
        name: sum(case_id.startswith(f"{name}-") for case_id in development)
        for name in ("bridge", "comparison")
    } == {"bridge": 3, "comparison": 3}


@pytest.mark.parametrize(
    ("rows", "stage", "message"),
    [
        ([{"id": "same", "type": "bridge"}] * 2, "development", "duplicate"),
        ([{"id": "x", "type": "unknown"}], "development", "invalid"),
        ([{"id": "x", "type": "bridge"}], "unknown", "unsupported"),
    ],
)
def test_stage_selection_rejects_invalid_contracts(
    rows: list[dict[str, str]], stage: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_stage_ids(rows, history_ids=set(), stage=stage, quota_per_type=1)


def test_stage_selection_rejects_insufficient_type_capacity() -> None:
    rows = [
        {"id": "bridge-1", "type": "bridge"},
        {"id": "comparison-1", "type": "comparison"},
    ]

    with pytest.raises(ValueError, match="lacks 2 eligible bridge"):
        select_stage_ids(rows, history_ids=set(), stage="development", quota_per_type=2)


def test_prepare_hotpot_case_seals_gold_and_preserves_candidate_contract() -> None:
    blind, gold = prepare_case(
        {
            "id": "hp-1",
            "type": "bridge",
            "question": "How are Alpha and Beta related?",
            "answer": "secret",
            "context": {
                "title": ["Alpha", "Beta"],
                "sentences": [["Alpha cites Beta."], ["Beta explains Alpha."]],
            },
            "supporting_facts": {
                "title": ["Alpha", "Beta"],
                "sent_id": [0, 0],
            },
        }
    )

    assert blind["id"] == gold["id"] == "hp-1"
    assert [candidate["id"] for candidate in blind["candidates"]] == [
        "hotpot-0::Alpha::0",
        "hotpot-1::Beta::0",
    ]
    assert gold["gold_evidence_ids"] == [
        "hotpot-0::Alpha::0",
        "hotpot-1::Beta::0",
    ]
    assert "secret" not in json.dumps(blind, sort_keys=True)
    assert "gold" not in json.dumps(blind, sort_keys=True).lower()


def test_prepare_hotpot_case_rejects_unmatched_evidence() -> None:
    with pytest.raises(ValueError, match="missing supporting evidence"):
        prepare_case(
            {
                "id": "hp-bad",
                "type": "bridge",
                "question": "Question?",
                "context": {"title": ["Alpha"], "sentences": [["Only one."]]},
                "supporting_facts": {
                    "title": ["Alpha", "Missing"],
                    "sent_id": [0, 0],
                },
            }
        )


def test_candidate_router_uses_soft_only_for_positive_predicted_gain() -> None:
    row = _scored_row()
    model = _v75_model()

    _, positive = select_method(row, CANDIDATE_METHOD, _constant_router(1.0), model)
    _, zero = select_method(row, CANDIDATE_METHOD, _constant_router(0.0), model)

    assert positive["route"] == SOFT_METHOD
    assert positive["predicted_soft_gain"] == 1.0
    assert zero["route"] == "cross_encoder_topk"
    assert zero["predicted_soft_gain"] == 0.0


def test_all_registered_controls_execute() -> None:
    artifact = load_router_artifact(MODEL_PATH)
    row = _scored_row()

    for method in CONTROL_METHODS:
        selected, routing = select_method(row, method, artifact, _v75_model())
        assert 1 <= len(selected) <= 5
        assert routing["route"]
    assert ZERO_SHOT_METHOD in CONTROL_METHODS


@pytest.mark.parametrize("location", ["row", "candidate"])
def test_selection_output_rejects_gold_bearing_rows(
    tmp_path: Path, location: str
) -> None:
    row = _scored_row()
    if location == "row":
        row["answer"] = "secret"
    else:
        row["candidates"][0]["gold"] = True

    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs(
            [row],
            load_router_artifact(MODEL_PATH),
            _v75_model(),
            tmp_path / "out.jsonl",
        )


def test_selection_output_is_safe_and_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.jsonl"
    row = _scored_row()
    artifact = load_router_artifact(MODEL_PATH)

    write_selection_outputs([row], artifact, _v75_model(), path)
    payload = path.read_text(encoding="utf-8").lower()

    assert '"question"' not in payload
    assert '"question_type"' not in payload
    assert "connected" not in payload
    assert '"text"' not in payload
    with pytest.raises(ValueError, match="unique non-empty"):
        write_selection_outputs([row, row], artifact, _v75_model(), path)


def test_registered_protocol_is_exact_and_detects_gate_or_resource_drift() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["common_resources"] == COMMON_RESOURCES
    assert protocol["development_gates"] == DEVELOPMENT_GATES

    protocol["development_gates"]["candidate_evidence_macro_f1_at_least"] = 0.0
    with pytest.raises(ValueError, match="gates drifted"):
        validate_registered_protocol(protocol)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["common_resources"]["top_k"] = 6
    with pytest.raises(ValueError, match="resources drifted"):
        validate_registered_protocol(protocol)


def test_implementation_lock_hashes_every_frozen_file() -> None:
    lock = json.loads(IMPLEMENTATION_LOCK_PATH.read_text(encoding="utf-8"))

    for contract in lock["files"].values():
        path = ROOT / contract["path"]
        assert path.stat().st_size == contract["bytes"]
        assert sha256(path) == contract["sha256"]


def test_small_evaluation_writes_safe_evidence_and_fails_size_gate(
    tmp_path: Path,
) -> None:
    scored_path = tmp_path / "scored.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    rows = []
    gold = []
    for index, question_type in enumerate(("bridge", "comparison")):
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
        V75_MODEL_PATH,
        stage="development",
        seed=7,
        output_dir=tmp_path / "output",
        history_overlap=0,
    )
    cases = read_jsonl_gzip(tmp_path / "output/cases.jsonl.gz")

    assert report["metadata"]["selection_written_before_gold_join"] is True
    assert (
        report["metadata"]["official_type_answer_or_gold_used_by_runtime_router"]
        is False
    )
    assert report["analysis"]["support_checks"]["exact_cases_equals_800"] is False
    assert report["analysis"]["outcome"]["confirmation_open_authorized"] is False
    audited_payload = json.dumps(cases).lower()
    assert '"question":' not in audited_payload
    assert '"candidate_ids":' not in audited_payload
    assert '"text":' not in audited_payload
    assert sha256(selection_path) == report["metadata"]["selection_output_sha256"]


def test_validation_recursively_rejects_non_finite_values() -> None:
    assert validate_finite({"nested": [0.0, 1.0]}) is True
    assert validate_finite({"nested": [float("nan")]}) is False
