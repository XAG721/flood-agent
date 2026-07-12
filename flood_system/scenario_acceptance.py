from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .simulation_dataset import DEFAULT_SIMULATION_SEED, build_floodagent_bench


SCENARIO_TEST_EVIDENCE: dict[str, tuple[str, ...]] = {
    "SIMSCENARIO-01": ("tests/test_response_workflow.py::test_full_deterministic_response_loop_and_close",),
    "SIMSCENARIO-02": (
        "tests/test_response_workflow.py::test_candidate_discovery_excludes_missing_coordinates_with_explicit_limitation",
    ),
    "SIMSCENARIO-03": (
        "tests/test_response_policies.py::test_geo_polygon_rejects_wrong_crs_and_out_of_bounds_coordinates",
    ),
    "SIMSCENARIO-04": (
        "tests/test_response_workflow.py::test_risk_object_alias_duplicate_is_merged_and_expired_registry_is_blocked",
    ),
    "SIMSCENARIO-05": ("tests/test_response_policies.py::test_point_in_polygon_includes_boundary_and_excludes_outside_point",),
    "SIMSCENARIO-06": (
        "tests/test_response_workflow.py::test_risk_object_alias_duplicate_is_merged_and_expired_registry_is_blocked",
    ),
    "SIMSCENARIO-07": (
        "tests/test_response_workflow.py::test_missing_registry_object_can_be_manually_added_and_confirmed",
    ),
    "SIMSCENARIO-08": (
        "tests/test_response_workflow.py::test_document_versions_are_clause_indexed_and_superseded_versions_leave_current_retrieval",
    ),
    "SIMSCENARIO-09": (
        "tests/test_response_workflow.py::test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution",
    ),
    "SIMSCENARIO-10": ("tests/test_frc_evaluation.py::test_conflict_detection_benchmark_reports_gate_two_metrics",),
    "SIMSCENARIO-11": (
        "tests/test_response_workflow.py::test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution",
    ),
    "SIMSCENARIO-12": (
        "tests/test_response_workflow.py::test_grounded_task_draft_covers_all_roles_and_keeps_plan_attribution",
    ),
    "SIMSCENARIO-13": (
        "tests/test_response_workflow.py::test_grounded_task_draft_is_blocked_when_policy_evidence_is_missing",
    ),
    "SIMSCENARIO-14": (
        "tests/test_response_workflow.py::test_document_parse_and_index_dependency_failures_are_isolated",
    ),
    "SIMSCENARIO-15": (
        "tests/test_response_workflow.py::test_document_parse_and_index_dependency_failures_are_isolated",
        "tests/test_system.py::test_supervisor_loop_retries_and_records_warning",
    ),
    "SIMSCENARIO-16": ("tests/test_system.py::test_platform_api_returns_explicit_llm_errors_without_rule_fallback",),
    "SIMSCENARIO-17": (
        "tests/test_response_workflow.py::test_task_versions_keep_complete_immutable_snapshots_and_approval_history",
    ),
    "SIMSCENARIO-18": (
        "tests/test_response_workflow.py::test_rule_engine_hard_blocks_ungrounded_ai_draft_and_optimistic_lock_conflict",
    ),
    "SIMSCENARIO-19": (
        "tests/test_response_workflow.py::test_simulated_dispatch_fault_matrix_is_audited_without_mutating_task_state",
        "tests/test_response_workflow.py::test_external_service_callback_is_token_bound_idempotent_and_state_safe",
    ),
    "SIMSCENARIO-20": (
        "tests/test_response_workflow.py::test_simulated_dispatch_fault_matrix_is_audited_without_mutating_task_state",
    ),
    "SIMSCENARIO-21": (
        "tests/test_response_workflow.py::test_liaison_assignment_and_reassignment_control_the_field_executor",
        "tests/test_response_workflow.py::test_deadline_extension_is_independently_approved_without_mutating_approved_payload",
        "tests/test_response_workflow.py::test_resource_shortage_feedback_creates_escalation_with_alternatives",
    ),
    "SIMSCENARIO-22": (
        "tests/test_response_workflow.py::test_all_four_deadline_types_escalate_with_explicit_audit",
    ),
    "SIMSCENARIO-23": (
        "tests/test_response_workflow.py::test_verification_can_return_completion_for_rework_without_losing_feedback",
    ),
    "SIMSCENARIO-24": (
        "tests/test_response_workflow.py::test_manual_takeover_and_commander_cancellation_are_audited_terminal_branches",
        "tests/test_response_workflow.py::test_simulated_dispatch_timeout_stays_pending_and_recovers_idempotently",
    ),
}


def _python_test_functions(repo_root: Path) -> set[str]:
    found: set[str] = set()
    for path in sorted((repo_root / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        relative = path.relative_to(repo_root).as_posix()
        found.update(
            f"{relative}::{node.name}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        )
    return found


def build_scenario_acceptance_report(
    repo_root: Path,
    *,
    seed: int = DEFAULT_SIMULATION_SEED,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    bench = build_floodagent_bench(seed=seed)
    known_tests = _python_test_functions(repo_root)
    results = []
    for scenario in bench["scenario_catalog"]:
        refs = list(SCENARIO_TEST_EVIDENCE.get(scenario["scenario_id"], ()))
        missing_refs = [ref for ref in refs if ref not in known_tests]
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "name": scenario["name"],
                "expected_error_codes": scenario["expected_error_codes"],
                "automated_evidence": refs,
                "missing_evidence_refs": missing_refs,
                "status": "covered" if refs and not missing_refs else "missing_evidence",
            }
        )
    covered = sum(item["status"] == "covered" for item in results)
    return {
        "metadata": {
            "dataset_id": bench["data_card"]["dataset_id"],
            "dataset_version": bench["data_card"]["version"],
            "manifest_sha256": bench["manifest_sha256"],
            "seed": seed,
            "generated_at": bench["data_card"]["generated_at"],
            "test_command": "python -m pytest -q",
            "interpretation": "covered 表示场景已绑定到当前存在且由全量 CI 执行的自动测试；测试运行结果以 CI 为准。",
        },
        "summary": {
            "scenario_count": len(results),
            "covered_count": covered,
            "missing_evidence_count": len(results) - covered,
            "status": "PASS" if covered == len(results) == 24 else "FAIL",
        },
        "scenarios": results,
    }


def write_scenario_acceptance_report(
    repo_root: Path,
    json_target: Path,
    markdown_target: Path,
    *,
    seed: int = DEFAULT_SIMULATION_SEED,
) -> tuple[Path, Path]:
    report = build_scenario_acceptance_report(repo_root, seed=seed)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FloodAgent-Bench 24 场景自动化验收追踪",
        "",
        f"- 数据集版本：`{report['metadata']['dataset_version']}`",
        f"- 场景覆盖：{report['summary']['covered_count']}/{report['summary']['scenario_count']}",
        f"- 判定：`{report['summary']['status']}`",
        "- 说明：覆盖表示场景已绑定到全量 CI 执行的自动测试；不替代真实岗位或生产环境验收。",
        "",
        "| 场景 | 名称 | 自动化证据 | 状态 |",
        "|---|---|---|---|",
    ]
    for item in report["scenarios"]:
        evidence = "<br>".join(f"`{ref}`" for ref in item["automated_evidence"])
        lines.append(f"| {item['scenario_id']} | {item['name']} | {evidence} | {item['status']} |")
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_target, markdown_target
