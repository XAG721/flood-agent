from __future__ import annotations

import pytest

from flood_system.simulation_dataset import (
    SimulatedExternalGateway,
    build_candidate_evaluation_payload,
    build_floodagent_bench,
)


def test_floodagent_bench_is_deterministic_traceable_and_family_isolated():
    first = build_floodagent_bench(seed=42)
    second = build_floodagent_bench(seed=42)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["objects"]) == 30
    assert len(first["events"]) == 24
    assert all(item["is_simulated"] and item["source_version"] for item in first["objects"])
    assert all(item["warning"]["is_simulated"] for item in first["events"])
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
