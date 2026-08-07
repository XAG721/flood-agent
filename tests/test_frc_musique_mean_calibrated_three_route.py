from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.hotpot_graph_router import (
    CANDIDATE_METHOD as V76_CANDIDATE_METHOD,
)
from research.frc_rag.hotpot_three_route_router import (
    ANCHOR_METHOD,
    MODEL_CONFIGURATIONS,
    ROUTE_POLICY_CONFIGURATIONS,
    SOFT_METHOD,
    load_router_artifact,
    model_configurations,
    route_predictions,
)
from research.frc_rag.musique_graph_router_transfer import source_id_commitment
from research.frc_rag.musique_mean_calibrated_three_route import (
    CANDIDATE_METHOD,
    CONTROL_METHODS,
    DEVELOPMENT_GATES,
    LEARNED_ROUTES,
    NONINFERIORITY_ENVELOPE,
    adjusted_route_gains,
    evaluate_stage,
    load_calibration,
    select_stage_ids,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl,
    read_jsonl_gzip,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "docs/progressive_upgrade/hotpot_three_route_router_model_development_v78.json"
)
V76_MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/hotpot_graph_router_model_development_v76.json"
)
V75_MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
CALIBRATION_PATH = (
    ROOT / "docs/progressive_upgrade/musique_three_route_mean_calibration_v78.json"
)
PROTOCOL_PATH = (
    ROOT / "docs/progressive_upgrade/musique_mean_calibrated_three_route_protocol_v78.json"
)


def _metadata(count: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prefix in ("2hop", "3hop1", "4hop1"):
        for index in range(count):
            source_id = f"{prefix}__case-{index:03d}"
            rows.extend(
                [
                    {"id": source_id, "answerable": True},
                    {"id": source_id, "answerable": False},
                ]
            )
    return rows


def _calibration_ids() -> set[str]:
    return {f"v77-calibration-{index:03d}" for index in range(800)}


def _candidate(
    candidate_id: str, source: str, text: str, cross: float
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


def _scored_row(case_id: str = "2hop__synthetic") -> dict[str, object]:
    specs = (
        ("a", "Alpha", "Alpha. Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta. Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Gamma. Unrelated high score.", 0.9),
        ("d", "Delta", "Delta. Another distractor.", 0.8),
        ("e", "Epsilon", "Epsilon. Another distractor.", 0.7),
        ("f", "Zeta", "Zeta. Another distractor.", 0.6),
    )
    return {
        "dataset": "MuSiQue",
        "source": "synthetic",
        "id": case_id,
        "question": "How is Alpha connected to Beta?",
        "hop_count": 2,
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*spec) for spec in specs],
    }


def test_finite_model_grid_is_complete_unique_and_disclosed() -> None:
    configurations = model_configurations()
    assert len(configurations) == MODEL_CONFIGURATIONS == 108
    assert ROUTE_POLICY_CONFIGURATIONS == 432
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 108


def test_route_predictions_choose_best_strictly_positive_gain() -> None:
    routes, gains = route_predictions(
        np.asarray([0.1, -0.1, 0.02, 0.01]),
        np.asarray([0.05, -0.2, 0.03, 0.01]),
        threshold=0.01,
    )
    assert routes.tolist() == [SOFT_METHOD, "cross_encoder_topk", ANCHOR_METHOD, "cross_encoder_topk"]
    assert gains.tolist() == pytest.approx([0.1, -0.1, 0.03, 0.01])


def test_route_predictions_reject_nonfinite_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="aligned"):
        route_predictions(np.asarray([1.0]), np.asarray([1.0, 2.0]), threshold=0.0)
    with pytest.raises(ValueError, match="non-finite"):
        route_predictions(np.asarray([np.nan]), np.asarray([0.0]), threshold=0.0)


def test_real_history_model_reproduces_registered_search_and_v76_control() -> None:
    artifact = load_router_artifact(MODEL_PATH)
    assert artifact["search"]["route_policy_configurations"] == 432
    assert artifact["crossfit"]["candidate"]["evidence_macro_f1"] == 0.563008
    assert artifact["crossfit"]["v76_candidate"]["evidence_macro_f1"] == 0.562448
    assert artifact["crossfit"]["candidate_minus_v76"]["point"] == 0.00056
    assert artifact["crossfit"]["route_counts"] == {
        "cross_encoder_topk": 602,
        SOFT_METHOD: 398,
    }
    assert artifact["crossfit"]["oof_prediction_mean_by_learned_route"] == {
        SOFT_METHOD: -0.031033214877,
        ANCHOR_METHOD: 0.0,
    }


def test_history_model_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["models"][SOFT_METHOD]["payload"]["learning_rate"] = 0.25
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_router_artifact(path)


def test_registered_mean_calibration_is_single_configuration_and_exact() -> None:
    router = load_router_artifact(MODEL_PATH)
    calibration = load_calibration(CALIBRATION_PATH, router)
    adjusted = adjusted_route_gains(
        {SOFT_METHOD: -0.021033214877, ANCHOR_METHOD: 0.0},
        router,
        calibration,
    )
    assert adjusted == pytest.approx(
        {SOFT_METHOD: 0.000566, ANCHOR_METHOD: 0.013651}
    )
    assert calibration["frozen_formula"]["calibration_configuration_count"] == 1
    assert calibration["prospective_boundary"]["v78_target_ids_selected_before_calibration_lock"] == 0


def test_calibration_rejects_history_center_tampering(tmp_path: Path) -> None:
    router = load_router_artifact(MODEL_PATH)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    calibration["history_router"]["oof_prediction_mean_by_learned_route"][SOFT_METHOD] = 0.0
    path = tmp_path / "tampered-calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="centers drifted"):
        load_calibration(path, router)


def test_stage_selection_is_deterministic_balanced_disjoint_and_excluded() -> None:
    rows = _metadata()
    excluded_id = "2hop__case-000"
    calibration_ids = _calibration_ids() | {"3hop1__case-000"}
    calibration_ids.remove("v77-calibration-000")
    quotas = {2: 2, 3: 1, 4: 1}
    kwargs = {
        "excluded_source_commitments": {source_id_commitment(excluded_id)},
        "calibration_source_ids": calibration_ids,
        "hop_quotas": quotas,
    }
    development = select_stage_ids(rows, stage="development", **kwargs)
    repeated = select_stage_ids(rows, stage="development", **kwargs)
    confirmation = select_stage_ids(rows, stage="confirmation", **kwargs)
    assert development == repeated
    assert len(development) == len(confirmation) == 4
    assert excluded_id not in development + confirmation
    assert "3hop1__case-000" not in development + confirmation
    assert not set(development) & set(confirmation)


def test_stage_selection_requires_exact_v77_calibration_count() -> None:
    with pytest.raises(ValueError, match="exactly 800"):
        select_stage_ids(
            _metadata(),
            excluded_source_commitments=set(),
            calibration_source_ids=set(),
            stage="development",
            hop_quotas={2: 1},
        )


def test_selection_output_contains_all_controls_and_calibrated_route(
    tmp_path: Path,
) -> None:
    history_router = load_router_artifact(MODEL_PATH)
    v76_router = json.loads(V76_MODEL_PATH.read_text(encoding="utf-8"))
    calibration = load_calibration(CALIBRATION_PATH, history_router)
    _, v75_model = load_v75_model_artifact(V75_MODEL_PATH)
    path = tmp_path / "selection.jsonl"
    outputs = write_selection_outputs(
        [_scored_row()],
        history_router,
        v76_router,
        calibration,
        v75_model,
        path,
    )
    assert read_jsonl(path) == outputs
    assert set(outputs[0]["methods"]) == {*CONTROL_METHODS, CANDIDATE_METHOD}
    assert V76_CANDIDATE_METHOD in outputs[0]["methods"]
    candidate = outputs[0]["methods"][CANDIDATE_METHOD]
    assert candidate["route"] in {"cross_encoder_topk", *LEARNED_ROUTES}
    assert candidate["adjusted_soft_gain"] is not None
    assert candidate["adjusted_anchor_gain"] is not None


def test_registered_protocol_matches_code_and_rejects_gate_tampering() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["development_gates"] == DEVELOPMENT_GATES
    assert protocol["noninferiority_envelope"] == NONINFERIORITY_ENVELOPE
    protocol["development_gates"]["candidate_evidence_macro_f1_at_least"] = 0.0
    with pytest.raises(ValueError, match="gates drifted"):
        validate_registered_protocol(protocol)


def test_synthetic_evaluation_writes_reaggregatable_case_evidence(
    tmp_path: Path,
) -> None:
    scored_path = tmp_path / "scored.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    output_dir = tmp_path / "output"
    rows = [
        _scored_row("2hop__synthetic"),
        {**_scored_row("3hop1__synthetic"), "hop_count": 3},
        {**_scored_row("4hop1__synthetic"), "hop_count": 4},
    ]
    write_jsonl(scored_path, rows)
    write_jsonl(
        gold_path,
        [
            {
                "id": row["id"],
                "hop_count": row["hop_count"],
                "gold_evidence_ids": ["a", "b"],
            }
            for row in rows
        ],
    )
    report = evaluate_stage(
        scored_path,
        gold_path,
        selection_path,
        MODEL_PATH,
        V76_MODEL_PATH,
        CALIBRATION_PATH,
        V75_MODEL_PATH,
        stage="development",
        seed=7,
        output_dir=output_dir,
        prior_overlap=0,
        calibration_overlap=0,
    )
    assert report["metadata"]["cases"] == 3
    assert report["analysis"]["support_checks"]["exact_cases_equals_600"] is False
    cases = read_jsonl_gzip(output_dir / "cases.jsonl.gz")
    assert len(cases) == 3
    assert set(cases[0]["methods"]) == {*CONTROL_METHODS, CANDIDATE_METHOD}


def test_adjusted_gain_contract_rejects_missing_route() -> None:
    router = load_router_artifact(MODEL_PATH)
    calibration = load_calibration(CALIBRATION_PATH, router)
    with pytest.raises(ValueError, match="contract drifted"):
        adjusted_route_gains({SOFT_METHOD: 0.0}, router, calibration)


def test_calibration_tampering_does_not_mutate_loaded_router() -> None:
    router = load_router_artifact(MODEL_PATH)
    before = copy.deepcopy(router)
    load_calibration(CALIBRATION_PATH, router)
    assert router == before
