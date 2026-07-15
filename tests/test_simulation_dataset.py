from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from flood_system.simulation_dataset import (
    SimulatedExternalGateway,
    SimulatedGatewayTimeout,
    SimulationDispatchScenario,
    build_candidate_evaluation_payload,
    build_floodagent_bench,
)
from flood_system.scenario_acceptance import build_scenario_acceptance_report


def test_floodagent_bench_is_deterministic_traceable_and_family_isolated():
    first = build_floodagent_bench(seed=42)
    second = build_floodagent_bench(seed=42)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["objects"]) == 30
    assert len(first["events"]) == 24
    assert len(first["scenario_catalog"]) == 24
    assert first["data_card"]["checksum"] == first["manifest_sha256"]
    assert all(
        item["is_simulated"] and item["source_version"] and item["data_origin"] == "SYNTHETIC"
        for item in first["objects"]
    )
    assert all(item["warning"]["is_simulated"] for item in first["events"])
    scenario_fields = {
        "input_data",
        "expected_candidate_objects",
        "expected_fields_and_evidence",
        "expected_conflicts_or_missing",
        "legal_state_path",
        "rejected_operations",
        "expected_error_codes",
        "acceptance_assertions",
    }
    assert all(scenario_fields <= scenario.keys() for scenario in first["scenario_catalog"])
    assert all(
        scenario["is_simulated"] and scenario["data_origin"] == "SYNTHETIC"
        for scenario in first["scenario_catalog"]
    )
    families: dict[str, set[str]] = {}
    for event in first["events"]:
        families.setdefault(event["template_family"], set()).add(event["split"])
    assert all(len(splits) == 1 for splits in families.values())
    evaluation = build_candidate_evaluation_payload(first)
    assert len(evaluation["full_cases"]) == 24
    assert all(case["is_simulated"] for case in evaluation["full_cases"])
    evaluation_families: dict[str, set[str]] = {}
    for case in evaluation["full_cases"]:
        evaluation_families.setdefault(case["template_family"], set()).add(case["split"])
    assert all(len(splits) == 1 for splits in evaluation_families.values())


def test_simulation_gateway_cannot_contact_real_endpoints():
    gateway = SimulatedExternalGateway()
    accepted = gateway.dispatch("simulated://member-unit", {"task_id": "TASK-1"})
    assert accepted["status"] == "accepted"
    assert accepted["is_simulated"] is True
    with pytest.raises(ValueError, match="refuses non-simulated"):
        gateway.dispatch("https://real.example.gov/callback", {"task_id": "TASK-1"})


@pytest.mark.parametrize(
    ("scenario", "status", "callback_versions"),
    [
        (SimulationDispatchScenario.NORMAL, "accepted", [1, 2]),
        (SimulationDispatchScenario.REJECT, "rejected", [1]),
        (SimulationDispatchScenario.PARTIAL_SUCCESS, "partial_success", [1]),
        (SimulationDispatchScenario.DUPLICATE_CALLBACK, "accepted", [1, 1, 2]),
        (SimulationDispatchScenario.OUT_OF_ORDER_CALLBACK, "accepted", [2, 1]),
    ],
)
def test_simulation_gateway_fault_matrix_is_deterministic_and_traceable(scenario, status, callback_versions):
    gateway = SimulatedExternalGateway()
    kwargs = {
        "scenario": scenario,
        "request_id": "SIMREQ-1",
        "trace_id": "SIMTRACE-1",
        "idempotency_key": "dispatch:TASK-1:v1",
        "event_time": datetime(2026, 7, 13, tzinfo=timezone.utc),
    }

    first = gateway.dispatch("simulated://member-unit", {"task_id": "TASK-1"}, **kwargs)
    second = gateway.dispatch("simulated://member-unit", {"task_id": "TASK-1"}, **kwargs)

    assert first == second
    assert first["status"] == status
    assert first["request_id"] == "SIMREQ-1"
    assert first["trace_id"] == "SIMTRACE-1"
    assert first["is_simulated"] is True
    assert [item["version"] for item in first["callbacks"]] == callback_versions
    assert all(item["is_simulated"] for item in first["callbacks"])


def test_simulation_gateway_timeout_is_explicit_and_retryable_by_caller():
    with pytest.raises(SimulatedGatewayTimeout, match="SIMREQ-timeout"):
        SimulatedExternalGateway().dispatch(
            "simulated://member-unit",
            {"task_id": "TASK-1"},
            scenario=SimulationDispatchScenario.TIMEOUT,
            request_id="SIMREQ-timeout",
        )


def test_all_24_simulation_scenarios_have_existing_automated_evidence():
    repo_root = Path(__file__).resolve().parents[1]
    report = build_scenario_acceptance_report(repo_root)

    assert report["summary"] == {
        "scenario_count": 24,
        "covered_count": 24,
        "missing_evidence_count": 0,
        "status": "PASS",
    }
    assert all(item["automated_evidence"] for item in report["scenarios"])
    assert all(not item["missing_evidence_refs"] for item in report["scenarios"])
