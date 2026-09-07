from __future__ import annotations

import json
from pathlib import Path

from research.frc_rag.musique_anchor_default_router import load_router_artifact
from research.frc_rag.musique_anchor_default_transfer import (
    PRIOR_COMMITMENT_UNION,
    TOTAL_EXCLUDED_TARGET_IDS,
    validate_registered_protocol,
)
from research.frc_rag.musique_graph_router_transfer import load_source_commitments
from research.frc_rag.twowiki_support_path_closure import read_jsonl, sha256


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
LOCK_PATH = (
    DOCS_ROOT / "musique_anchor_default_terminal_holdout_implementation_v80.json"
)
SOURCE_PATH = (
    DOCS_ROOT / "musique_anchor_default_terminal_holdout_source_registration_v80.json"
)
PROTOCOL_PATH = (
    DOCS_ROOT / "musique_anchor_default_terminal_holdout_protocol_v80.json"
)
MODEL_PATH = (
    DOCS_ROOT / "musique_anchor_default_router_model_development_v80.json"
)


def _assert_contract(contract: dict[str, object]) -> Path:
    path = ROOT / str(contract["path"])
    assert path.stat().st_size == int(contract["bytes"])
    assert sha256(path) == str(contract["sha256"])
    return path


def test_v80_implementation_lock_covers_every_registered_file() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["pre_target_verification"]["v80_target_ids_selected"] == 0
    assert lock["target_model_development"]["finite_route_policy_configurations"] == (
        1512
    )
    assert lock["target_model_development"]["selected_configuration"][
        "threshold"
    ] == 0.0025
    assert lock["registered_target_capacity"]["total_excluded_ids"] == (
        TOTAL_EXCLUDED_TARGET_IDS
    )
    assert (
        lock["registered_target_capacity"][
            "second_comparable_same_source_stage_supported"
        ]
        is False
    )
    for contract in lock["files"].values():
        _assert_contract(contract)


def test_v80_source_registry_revalidates_nested_maps_ids_and_dependencies() -> None:
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
    registered = (
        ("v77_target_domain_calibration_ids", 800),
        ("v78_target_domain_training_ids", 600),
        ("v79_target_domain_training_ids", 470),
    )
    id_sets: list[set[str]] = []
    for key, expected in registered:
        path = _assert_contract(source[key])
        rows = read_jsonl(path)
        values = {str(row["id"]) for row in rows}
        assert len(rows) == len(values) == expected
        id_sets.append(values)
    assert not id_sets[0] & id_sets[1]
    assert not id_sets[0] & id_sets[2]
    assert not id_sets[1] & id_sets[2]
    dependencies = source["frozen_dependencies"]
    for prefix in (
        "anchor_router",
        "v79_router",
        "history_router",
        "mean_calibration",
        "v76_router",
        "v75_model",
        "v78_training_scored",
        "v78_training_cases",
        "v79_training_scored",
        "v79_training_cases",
    ):
        _assert_contract(
            {
                "path": dependencies[f"{prefix}_path"],
                "bytes": dependencies[f"{prefix}_bytes"],
                "sha256": dependencies[f"{prefix}_sha256"],
            }
        )


def test_v80_protocol_model_and_terminal_policy_are_mutually_consistent() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    model = load_router_artifact(MODEL_PATH)
    selected = model["selected_configuration"]
    candidate = protocol["candidate"]
    assert candidate["feature_count"] == len(selected["feature_names"])
    assert candidate["selected_model_configuration_index"] == selected[
        "model_configuration_index"
    ]
    assert candidate["selected_threshold"] == selected["threshold"]
    assert candidate["selected_oof_model_selection_f1"] == model[
        "crossfit_model_selection_diagnostic"
    ]["candidate"]["evidence_macro_f1"]
    assert protocol["stage"]["further_same_source_confirmation_authorized"] is False
    assert protocol["claim_limits"]["selector_canary_default_or_gate2_authorized"] is (
        False
    )


def test_v80_runner_registration_verification_passes_before_target_selection() -> None:
    from scripts.run_musique_anchor_default_transfer import _verify_registration

    protocol, source, exclusion_paths = _verify_registration()
    assert protocol["stage"]["name"] == "terminal_holdout"
    assert source["pre_registration_access_boundary"]["v80_target_ids_selected"] == 0
    assert exclusion_paths
