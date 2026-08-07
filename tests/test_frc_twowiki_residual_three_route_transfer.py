from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.hotpot_graph_router import (
    load_router_artifact as load_v76_router_artifact,
)
from research.frc_rag.hotpot_three_route_router import (
    load_router_artifact as load_v78_router_artifact,
)
from research.frc_rag.musique_anchor_default_router import (
    load_router_artifact as load_v80_router_artifact,
)
from research.frc_rag.musique_target_three_route_router import (
    load_router_artifact as load_v79_router_artifact,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    CANDIDATE_METHOD,
    CROSS_ACTION,
    CROSS_METHOD,
    DEFAULT_ACTION,
    FEATURE_NAMES,
    FLIP_ACTION,
    MODEL_CONFIGURATIONS,
    ROUTE_POLICY_CONFIGURATIONS,
    SOFT_METHOD,
    THRESHOLD_GRID,
    WEIGHT_MODE_GRID,
    _predict_model,
    load_router_artifact,
    model_configurations,
    route_actions,
    routes_for_actions,
)
from research.frc_rag.twowiki_residual_three_route_transfer import (
    CONTROL_METHODS,
    METHODS,
    NONINFERIORITY_ENVELOPE,
    STRICT_GATES,
    select_stage_ids,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_support_path_closure import QUESTION_TYPES, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
V78_MODEL_PATH = DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
V79_MODEL_PATH = DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
V80_MODEL_PATH = DOCS_ROOT / "musique_anchor_default_router_model_development_v80.json"
V81_MODEL_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)
PROTOCOL_PATH = DOCS_ROOT / "twowiki_residual_three_route_protocol_v81.json"


def _candidate(
    candidate_id: str, source: str, text: str, score: float
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": text,
        "metadata": {"title": source},
        "token_count": 10,
        "scores": {
            "bm25": score,
            "dense": score,
            "hybrid": score,
            "cross_encoder": score,
        },
        "role_scores": {
            "condition": score,
            "attribution": score,
            "procedure": score,
            "answer": score,
            "exception": score,
        },
    }


def _scored_row(case_id: str = "synthetic-v81") -> dict[str, object]:
    specs = (
        ("a", "Alpha", "Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Unrelated high score.", 0.9),
        ("d", "Delta", "Another distractor.", 0.8),
        ("e", "Epsilon", "Another distractor.", 0.7),
        ("f", "Zeta", "Another distractor.", 0.6),
    )
    return {
        "dataset": "2WikiMultiHopQA",
        "source": "synthetic",
        "id": case_id,
        "question": "Which happened first, Alpha or Beta?",
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*spec) for spec in specs],
    }


def test_v81_finite_model_grid_is_complete_unique_and_disclosed() -> None:
    configurations = model_configurations()
    assert len(configurations) == MODEL_CONFIGURATIONS == 216
    assert ROUTE_POLICY_CONFIGURATIONS == 1512
    assert len(THRESHOLD_GRID) == 7
    assert WEIGHT_MODE_GRID == ("uniform", "nonzero_x4", "magnitude_x20")
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 216


def test_portable_tree_predictor_preserves_float64_threshold_semantics() -> None:
    feature_index = 0
    value = np.float32(0.0846666693687439)
    threshold = 0.0846666656434536
    assert value <= threshold  # numpy scalar rounds the threshold to float32
    assert not (float(value) <= threshold)
    payload = {
        "feature_count": len(FEATURE_NAMES),
        "init_prediction": 0.0,
        "learning_rate": 1.0,
        "trees": [
            {
                "children_left": [1, -1, -1],
                "children_right": [2, -1, -1],
                "features": [feature_index, -2, -2],
                "thresholds": [threshold, -2.0, -2.0],
                "values": [0.0, 1.0, 2.0],
            }
        ],
    }
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    features[feature_index] = float(value)
    assert _predict_model(payload, features).tolist() == [2.0]


def test_route_actions_and_actual_routes_keep_v75_as_default() -> None:
    actions, gains = route_actions(
        np.asarray([0.001, 0.02, 0.01]),
        np.asarray([0.002, 0.01, 0.03]),
        threshold=0.005,
    )
    assert actions.tolist() == [DEFAULT_ACTION, CROSS_ACTION, FLIP_ACTION]
    assert gains.tolist() == [0.002, 0.02, 0.03]
    assert routes_for_actions(
        actions, [SOFT_METHOD, SOFT_METHOD, SOFT_METHOD]
    ).tolist() == [SOFT_METHOD, CROSS_METHOD, "alternating_anchor_link_control"]
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        route_actions(np.asarray([0.1]), np.asarray([0.1, 0.2]), threshold=0.0)


def test_real_v81_model_reproduces_registered_oof_diagnostic() -> None:
    artifact = load_router_artifact(V81_MODEL_PATH)
    selected = artifact["selected_configuration"]
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    assert selected == {
        **selected,
        "estimators": 100,
        "max_depth": 2,
        "min_samples_leaf": 25,
        "learning_rate": 0.1,
        "loss": "huber",
        "weight_mode": "magnitude_x20",
        "model_configuration_index": 188,
        "threshold": 0.0025,
    }
    assert diagnostic["candidate"]["evidence_macro_f1"] == 0.540164
    assert diagnostic["v75_default"]["evidence_macro_f1"] == 0.534251
    assert diagnostic["candidate_minus_v75"] == {
        "point": 0.005913,
        "ci_low": 0.004067,
        "ci_high": 0.007857,
        "resamples": 10000,
        "seed": 20261171,
    }
    assert diagnostic["action_counts"] == {
        CROSS_ACTION: 134,
        FLIP_ACTION: 376,
        DEFAULT_ACTION: 1090,
    }
    assert all(
        row["candidate_minus_v75"] >= 0.0
        for row in diagnostic["per_question_type"].values()
    )
    assert artifact["prospective_boundary"]["v81_target_training_or_tuning_cases"] == 0


def test_v81_model_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    artifact = json.loads(V81_MODEL_PATH.read_text(encoding="utf-8"))
    artifact["models"][CROSS_ACTION]["payload"]["learning_rate"] = 0.25
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_router_artifact(path)


def test_target_selection_is_deterministic_balanced_and_prior_disjoint() -> None:
    metadata = [
        {"_id": f"{question_type}-{index:03d}", "type": question_type}
        for question_type in QUESTION_TYPES
        for index in range(8)
    ]
    excluded = {f"{question_type}-000" for question_type in QUESTION_TYPES}
    quotas = {question_type: 2 for question_type in QUESTION_TYPES}
    development = select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="development",
        question_type_quotas=quotas,
    )
    confirmation = select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="confirmation",
        question_type_quotas=quotas,
    )
    assert development == select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="development",
        question_type_quotas=quotas,
    )
    assert len(development) == len(confirmation) == 8
    assert not (set(development) & set(confirmation))
    assert not (set(development) & excluded)
    assert not (set(confirmation) & excluded)


def test_selection_outputs_execute_every_registered_control(tmp_path: Path) -> None:
    v81_router = load_router_artifact(V81_MODEL_PATH)
    v80_router = load_v80_router_artifact(V80_MODEL_PATH)
    v79_router = load_v79_router_artifact(V79_MODEL_PATH)
    v78_router = load_v78_router_artifact(V78_MODEL_PATH)
    v76_router = load_v76_router_artifact(V76_MODEL_PATH)
    _, v75_model = load_v75_model_artifact(V75_MODEL_PATH)
    path = tmp_path / "selection.jsonl"
    outputs = write_selection_outputs(
        [_scored_row()],
        v81_router=v81_router,
        v80_router=v80_router,
        v79_router=v79_router,
        v78_router=v78_router,
        v76_router=v76_router,
        v75_model=v75_model,
        path=path,
    )
    assert outputs == read_jsonl(path)
    assert set(outputs[0]["methods"]) == set(METHODS)
    assert set(CONTROL_METHODS) < set(METHODS)
    candidate = outputs[0]["methods"][CANDIDATE_METHOD]
    assert candidate["action"] in {CROSS_ACTION, FLIP_ACTION, DEFAULT_ACTION}
    assert candidate["route"] in {
        CROSS_METHOD,
        SOFT_METHOD,
        "alternating_anchor_link_control",
    }


def test_selection_outputs_reject_gold_fields(tmp_path: Path) -> None:
    v81_router = load_router_artifact(V81_MODEL_PATH)
    v80_router = load_v80_router_artifact(V80_MODEL_PATH)
    v79_router = load_v79_router_artifact(V79_MODEL_PATH)
    v78_router = load_v78_router_artifact(V78_MODEL_PATH)
    v76_router = load_v76_router_artifact(V76_MODEL_PATH)
    _, v75_model = load_v75_model_artifact(V75_MODEL_PATH)
    row = _scored_row()
    row["answer"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs(
            [row],
            v81_router=v81_router,
            v80_router=v80_router,
            v79_router=v79_router,
            v78_router=v78_router,
            v76_router=v76_router,
            v75_model=v75_model,
            path=tmp_path / "selection.jsonl",
        )


def test_registered_protocol_matches_code_constants() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["strict_gates"] == STRICT_GATES
    assert protocol["noninferiority_envelope"] == NONINFERIORITY_ENVELOPE
