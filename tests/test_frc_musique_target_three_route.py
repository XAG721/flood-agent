from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.hotpot_three_route_router import (
    ANCHOR_METHOD,
    DEFAULT_ROUTE,
    SOFT_METHOD,
)
from research.frc_rag.musique_graph_router_transfer import source_id_commitment
from research.frc_rag.musique_mean_calibrated_three_route import (
    CANDIDATE_METHOD as V78_CANDIDATE_METHOD,
    load_calibration,
)
from research.frc_rag.musique_target_three_route_router import (
    CANDIDATE_METHOD,
    MODEL_CONFIGURATIONS,
    ROUTE_POLICY_CONFIGURATIONS,
    THRESHOLD_GRID,
    load_router_artifact,
    model_configurations,
)
from research.frc_rag.musique_target_three_route_transfer import (
    CONTROL_METHODS,
    DEVELOPMENT_GATES,
    NONINFERIORITY_ENVELOPE,
    evaluate_stage,
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
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
TARGET_MODEL_PATH = (
    DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
)
HISTORY_MODEL_PATH = (
    DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
)
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
CALIBRATION_PATH = DOCS_ROOT / "musique_three_route_mean_calibration_v78.json"
PROTOCOL_PATH = DOCS_ROOT / "musique_target_three_route_protocol_v79.json"


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


def _exclusion_ids(prefix: str, count: int) -> set[str]:
    return {f"{prefix}-{index:03d}" for index in range(count)}


def test_v79_finite_model_grid_is_complete_unique_and_disclosed() -> None:
    configurations = model_configurations()
    assert len(configurations) == MODEL_CONFIGURATIONS == 108
    assert ROUTE_POLICY_CONFIGURATIONS == 432
    assert len(THRESHOLD_GRID) == 4
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 108


def test_real_v79_model_reproduces_registered_model_selection_diagnostic() -> None:
    artifact = load_router_artifact(TARGET_MODEL_PATH)
    selected = artifact["selected_configuration"]
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    assert selected["model_configuration_index"] == 95
    assert selected["threshold"] == 0.005
    assert diagnostic["candidate"]["evidence_macro_f1"] == 0.551733
    assert diagnostic["candidate_minus_v78"]["point"] == 0.005351
    assert diagnostic["route_counts"] == {
        ANCHOR_METHOD: 215,
        DEFAULT_ROUTE: 215,
        SOFT_METHOD: 170,
    }
    assert diagnostic["candidate_minus_v78"]["ci_low"] <= 0.0
    assert artifact["prospective_boundary"]["v79_target_training_or_tuning_cases"] == 0


def test_v79_model_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    artifact = json.loads(TARGET_MODEL_PATH.read_text(encoding="utf-8"))
    artifact["models"][SOFT_METHOD]["payload"]["learning_rate"] = 0.25
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_router_artifact(path)


def test_stage_selection_is_deterministic_balanced_disjoint_and_excluded() -> None:
    excluded_id = "2hop__case-000"
    v77_ids = _exclusion_ids("v77", 800)
    v78_ids = _exclusion_ids("v78", 600)
    v77_ids.remove("v77-000")
    v77_ids.add("3hop1__case-000")
    v78_ids.remove("v78-000")
    v78_ids.add("4hop1__case-000")
    kwargs = {
        "excluded_source_commitments": {source_id_commitment(excluded_id)},
        "v77_calibration_source_ids": v77_ids,
        "v78_training_source_ids": v78_ids,
        "hop_quotas": {2: 2, 3: 1, 4: 1},
    }
    development = select_stage_ids(_metadata(), stage="development", **kwargs)
    repeated = select_stage_ids(_metadata(), stage="development", **kwargs)
    confirmation = select_stage_ids(_metadata(), stage="confirmation", **kwargs)
    assert development == repeated
    assert len(development) == len(confirmation) == 4
    assert excluded_id not in development + confirmation
    assert "3hop1__case-000" not in development + confirmation
    assert "4hop1__case-000" not in development + confirmation
    assert not set(development) & set(confirmation)


@pytest.mark.parametrize(
    ("v77_count", "v78_count", "message"),
    [(0, 600, "exactly 800"), (800, 0, "exactly 600")],
)
def test_stage_selection_requires_exact_registered_exclusion_counts(
    v77_count: int, v78_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_stage_ids(
            _metadata(),
            excluded_source_commitments=set(),
            v77_calibration_source_ids=_exclusion_ids("v77", v77_count),
            v78_training_source_ids=_exclusion_ids("v78", v78_count),
            stage="development",
            hop_quotas={2: 1},
        )


def test_selection_output_contains_all_controls_and_frozen_routes(
    tmp_path: Path,
) -> None:
    target_router = load_router_artifact(TARGET_MODEL_PATH)
    from research.frc_rag.hotpot_three_route_router import load_router_artifact as load_history

    history_router = load_history(HISTORY_MODEL_PATH)
    v76_router = json.loads(V76_MODEL_PATH.read_text(encoding="utf-8"))
    calibration = load_calibration(CALIBRATION_PATH, history_router)
    _, v75_model = load_v75_model_artifact(V75_MODEL_PATH)
    path = tmp_path / "selection.jsonl"
    outputs = write_selection_outputs(
        [_scored_row()],
        target_router,
        history_router,
        v76_router,
        calibration,
        v75_model,
        path,
    )
    assert read_jsonl(path) == outputs
    assert set(outputs[0]["methods"]) == {*CONTROL_METHODS, CANDIDATE_METHOD}
    assert V78_CANDIDATE_METHOD in outputs[0]["methods"]
    candidate = outputs[0]["methods"][CANDIDATE_METHOD]
    assert candidate["route"] in {DEFAULT_ROUTE, SOFT_METHOD, ANCHOR_METHOD}
    assert candidate["predicted_soft_gain"] is not None
    assert candidate["predicted_anchor_gain"] is not None


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
        TARGET_MODEL_PATH,
        HISTORY_MODEL_PATH,
        V76_MODEL_PATH,
        CALIBRATION_PATH,
        V75_MODEL_PATH,
        stage="development",
        seed=7,
        output_dir=output_dir,
        prior_overlap=0,
        v77_overlap=0,
        v78_overlap=0,
    )
    assert report["metadata"]["cases"] == 3
    assert report["analysis"]["support_checks"]["exact_cases_equals_470"] is False
    cases = read_jsonl_gzip(output_dir / "cases.jsonl.gz")
    assert len(cases) == 3
    assert set(cases[0]["methods"]) == {*CONTROL_METHODS, CANDIDATE_METHOD}
