from __future__ import annotations

import json
from pathlib import Path

from research.frc_rag.hotpot_three_route_router import load_router_artifact
from research.frc_rag.musique_graph_router_transfer import load_source_commitments
from research.frc_rag.musique_mean_calibrated_three_route import (
    PRIOR_COMMITMENT_UNION,
    load_calibration,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_support_path_closure import sha256


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
LOCK_PATH = DOCS_ROOT / "musique_mean_calibrated_three_route_implementation_v78.json"
SOURCE_PATH = DOCS_ROOT / "musique_mean_calibrated_three_route_source_registration_v78.json"
PROTOCOL_PATH = DOCS_ROOT / "musique_mean_calibrated_three_route_protocol_v78.json"
MODEL_PATH = DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
CALIBRATION_PATH = DOCS_ROOT / "musique_three_route_mean_calibration_v78.json"


def _assert_contract(contract: dict[str, object]) -> Path:
    path = ROOT / str(contract["path"])
    assert path.stat().st_size == int(contract["bytes"])
    assert sha256(path) == str(contract["sha256"])
    return path


def test_v78_implementation_lock_covers_every_registered_file() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["pre_target_verification"]["v78_target_ids_selected"] == 0
    assert lock["history_model_development"]["finite_route_policy_configurations"] == 432
    assert lock["target_calibration"]["configuration_count"] == 1
    for contract in lock["files"].values():
        _assert_contract(contract)


def test_v78_source_registry_revalidates_nested_prior_maps_and_dependencies() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    target = source["target"]
    _assert_contract(target)
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
    v77_rows = [
        json.loads(line)
        for line in v77_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(v77_rows) == len({str(row["id"]) for row in v77_rows}) == 800
    dependencies = source["frozen_dependencies"]
    for prefix in (
        "history_source",
        "history_router",
        "mean_calibration",
        "v76_router",
        "v75_model",
    ):
        _assert_contract(
            {
                "path": dependencies[f"{prefix}_path"],
                "bytes": dependencies[f"{prefix}_bytes"],
                "sha256": dependencies[f"{prefix}_sha256"],
            }
        )


def test_v78_protocol_model_and_calibration_are_mutually_consistent() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    router = load_router_artifact(MODEL_PATH)
    calibration = load_calibration(CALIBRATION_PATH, router)
    assert protocol["candidate"]["calibration_configuration_count"] == 1
    assert calibration["frozen_formula"]["calibration_configuration_count"] == 1
    assert protocol["candidate"]["feature_count"] == len(
        router["selected_configuration"]["feature_names"]
    )
