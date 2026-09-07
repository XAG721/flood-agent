from __future__ import annotations

import json
from pathlib import Path

from research.frc_rag.musique_graph_router_transfer import load_source_commitments
from research.frc_rag.musique_target_three_route_router import load_router_artifact
from research.frc_rag.musique_target_three_route_transfer import (
    PRIOR_COMMITMENT_UNION,
    TOTAL_EXCLUDED_TARGET_IDS,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_support_path_closure import read_jsonl, sha256


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
LOCK_PATH = DOCS_ROOT / "musique_target_three_route_implementation_v79.json"
SOURCE_PATH = DOCS_ROOT / "musique_target_three_route_source_registration_v79.json"
PROTOCOL_PATH = DOCS_ROOT / "musique_target_three_route_protocol_v79.json"
MODEL_PATH = (
    DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
)


def _assert_contract(contract: dict[str, object]) -> Path:
    path = ROOT / str(contract["path"])
    assert path.stat().st_size == int(contract["bytes"])
    assert sha256(path) == str(contract["sha256"])
    return path


def test_v79_implementation_lock_covers_every_registered_file() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["pre_target_verification"]["v79_target_ids_selected"] == 0
    assert lock["target_model_development"]["finite_route_policy_configurations"] == 432
    assert lock["target_model_development"]["selected_threshold"] == 0.005
    assert lock["registered_target_capacity"]["total_excluded_ids"] == (
        TOTAL_EXCLUDED_TARGET_IDS
    )
    for contract in lock["files"].values():
        _assert_contract(contract)


def test_v79_source_registry_revalidates_nested_maps_and_dependencies() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    _assert_contract(source["target"])
    registry_path = _assert_contract(source["prior_exclusion_registry"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for contract in registry["prior_exclusion_sources"].values():
        if not isinstance(contract, dict) or "path" not in contract:
            continue
        path = _assert_contract(contract)
        assert len(load_source_commitments([path])) == int(
            contract["expected_unique_source_commitments"]
        )
        paths.append(path)
    assert len(load_source_commitments(paths)) == PRIOR_COMMITMENT_UNION
    v77_path = _assert_contract(source["v77_target_domain_calibration_ids"])
    v78_path = _assert_contract(source["v78_target_domain_training_ids"])
    v77_ids = {str(row["id"]) for row in read_jsonl(v77_path)}
    v78_ids = {str(row["id"]) for row in read_jsonl(v78_path)}
    assert len(v77_ids) == 800
    assert len(v78_ids) == 600
    assert not v77_ids & v78_ids
    dependencies = source["frozen_dependencies"]
    for prefix in (
        "target_router",
        "history_router",
        "mean_calibration",
        "v76_router",
        "v75_model",
        "target_training_scored",
        "target_training_cases",
    ):
        _assert_contract(
            {
                "path": dependencies[f"{prefix}_path"],
                "bytes": dependencies[f"{prefix}_bytes"],
                "sha256": dependencies[f"{prefix}_sha256"],
            }
        )


def test_v79_protocol_and_target_model_are_mutually_consistent() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    model = load_router_artifact(MODEL_PATH)
    selected = model["selected_configuration"]
    assert protocol["candidate"]["feature_count"] == len(selected["feature_names"])
    assert protocol["candidate"]["selected_model_configuration_index"] == selected[
        "model_configuration_index"
    ]
    assert protocol["candidate"]["selected_threshold"] == selected["threshold"]
    assert protocol["candidate"]["selected_oof_model_selection_f1"] == model[
        "crossfit_model_selection_diagnostic"
    ]["candidate"]["evidence_macro_f1"]
