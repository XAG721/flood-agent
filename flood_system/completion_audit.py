from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from flood_system.design_contract_audit import build_design_contract_audit


AUDIT_VERSION = "progressive-completion-audit-v9"

SOFTWARE_DELIVERABLES: dict[str, tuple[str, ...]] = {
    "web_and_cesium": (
        "frontend/src/pages/ResponseWorkflowPage.tsx",
        "3D_visual/src/App.tsx",
    ),
    "core_api": (
        "flood_system/api.py",
        "flood_system/http/idempotency.py",
        "flood_system/http/response_router.py",
    ),
    "worker": ("scripts/run_response_worker.py",),
    "candidate_service": ("flood_system/response_workflow/candidate_discovery.py",),
    "document_governance": (
        "flood_system/response_workflow/document_governance.py",
        "tests/test_document_governance.py",
    ),
    "frc_rag_service": (
        "flood_system/rag.py",
        "flood_system/response_workflow/service.py",
    ),
    "draft_rules_state_machine": (
        "flood_system/response_workflow/service.py",
        "flood_system/response_workflow/state_machine.py",
    ),
    "simulation": (
        "flood_system/simulation_dataset.py",
        "scripts/run_simulation_scenario_acceptance.py",
    ),
    "migration": (
        "scripts/migrate_response_schema.py",
        "scripts/migrate_response_to_postgis.py",
        "infra/postgis/001_shadow_projection.sql",
    ),
    "deployment": ("Dockerfile", "frontend/Dockerfile", "docker-compose.yml"),
    "contract_audit": (
        "flood_system/design_contract_audit.py",
        "scripts/run_design_contract_audit.py",
        "scripts/freeze_legacy_baseline.py",
    ),
}

DOCUMENT_DELIVERABLES: dict[str, tuple[str, ...]] = {
    "inventory_and_reuse_matrix": (
        "docs/progressive_upgrade/asset_inventory.md",
        "docs/progressive_upgrade/implementation_acceptance_matrix.md",
    ),
    "contract_and_migration_mapping": (
        "docs/progressive_upgrade/api_compatibility_mapping.md",
        "docs/progressive_upgrade/contracts_and_data_dictionary.md",
        "infra/migrations/README.md",
    ),
    "requirements_and_design": (
        "洪水预警响应系统_渐进式迭代开发与升级设计.md",
        "面向区县防办的洪水预警响应系统_升级设计说明V3.md",
    ),
    "api_and_acceptance": (
        "docs/openapi.json",
        "docs/V3_upgrade_acceptance_matrix.md",
        "docs/progressive_upgrade/completion_traceability_audit.md",
    ),
    "operations_and_users": (
        "docs/progressive_upgrade/operations_and_rollback.md",
        "docs/progressive_upgrade/user_manual.md",
    ),
    "security_and_incidents": (
        "docs/progressive_upgrade/security_and_incident_manual.md",
        "docs/progressive_upgrade/dependency_security_audit.md",
    ),
    "experiments": (
        "docs/progressive_upgrade/evaluation_and_gate_report.md",
        "docs/progressive_upgrade/frc_public_evaluation_protocol.md",
    ),
    "contract_traceability": (
        "docs/progressive_upgrade/design_contract_evidence_policy.json",
        "output/acceptance/legacy_baseline_manifest.md",
        "output/acceptance/design_contract_audit.md",
    ),
}

DATA_EXPERIMENT_DELIVERABLES: dict[str, tuple[str, ...]] = {
    "floodagent_bench": (
        "output/floodagent_bench/floodagent_bench.json",
        "output/floodagent_bench/scenario_acceptance_report.json",
    ),
    "candidate_and_performance": (
        "output/candidate_evaluation/report.json",
        "output/performance/controlled_performance_report.json",
        "benchmarks/controlled_performance_budget.json",
    ),
    "rag_comparison_and_challenges": (
        "output/rag_evaluation/rag_evaluation_report.json",
        "output/rag_evaluation/full_public/public_benchmark_report.md",
        "output/rag_evaluation/public_frc_reference/public_frc_reference_report.json",
        "output/rag_evaluation/conflicts_frc/conflicts_frc_report.json",
        "output/rag_evaluation/controlled_domain_sensitivity/controlled_domain_sensitivity.json",
        "output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.json",
        "output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation_cases.jsonl.gz",
        "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.json",
        "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation_cases.jsonl.gz",
        "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.json",
        "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md",
        "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity_cases.jsonl.gz",
        "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation.json",
        "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation_cases.jsonl.gz",
        "benchmarks/eurlex_temporal_selection.json",
        "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.json",
        "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.md",
        "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation_cases.jsonl.gz",
    ),
    "reproduction_entrypoints": (
        "scripts/run_candidate_evaluation.py",
        "scripts/run_rag_evaluation.py",
        "scripts/run_public_rag_benchmarks.py",
        "scripts/run_conflicts_frc_evaluation.py",
        "scripts/run_frc_controlled_sensitivity.py",
        "scripts/run_conflicts_frc_ablation.py",
        "scripts/run_housing_frc_ablation.py",
        "scripts/run_housing_weight_sensitivity.py",
        "scripts/run_lawshift_temporal_ablation.py",
        "scripts/prepare_eurlex_temporal_source.py",
        "scripts/run_eurlex_temporal_ablation.py",
        "flood_system/frc_eurlex_temporal_ablation.py",
        "flood_system/frc_housing_weight_sensitivity.py",
    ),
    "contract_and_legacy_baseline_evidence": (
        "output/acceptance/legacy_baseline_manifest.json",
        "output/acceptance/design_contract_audit.json",
        "tests/test_design_contract_audit.py",
    ),
}

REQUIRED_RESPONSE_PATHS = {
    "/response/events",
    "/response/events/{event_id}",
    "/response/events/{event_id}/risk-objects/discover",
    "/response/events/{event_id}/risk-objects/{object_id}/verify",
    "/response/events/{event_id}/candidate-object-lists",
    "/response/events/{event_id}/candidate-object-lists/freeze",
    "/response/risk-objects",
    "/response/risk-objects/versions",
    "/response/risk-objects/imports",
    "/response/risk-objects/file-imports",
    "/response/documents",
    "/response/events/{event_id}/evidence-packages",
    "/response/evidence-packages/{package_id}/conflicts/{conflict_id}/resolve",
    "/response/evidence-packages/{package_id}/freeze",
    "/response/events/{event_id}/risk-objects/{object_id}/task-draft",
    "/response/events/{event_id}/review-draft",
    "/response/tasks/{task_id}/decision",
    "/response/tasks/{task_id}/assign",
    "/response/tasks/{task_id}/acknowledge",
    "/response/tasks/{task_id}/start",
    "/response/tasks/{task_id}/feedback",
    "/response/tasks/{task_id}/deadline-extensions",
    "/response/tasks/{task_id}/submit",
    "/response/tasks/{task_id}/verify-completion",
    "/response/events/{event_id}/close",
    "/response/dispatch/outbox/process",
    "/response/simulation/dispatch-callbacks",
    "/response/events/{event_id}/replay",
    "/response/events/{event_id}/audit-trail",
    "/response/legacy/events/{legacy_event_id}",
}

EVIDENCE_FILES = (
    "output/floodagent_bench/scenario_acceptance_report.json",
    "output/candidate_evaluation/report.json",
    "output/performance/controlled_performance_report.json",
    "output/rag_evaluation/public_frc_reference/public_frc_reference_report.json",
    "docs/openapi.json",
    "docker-compose.yml",
    "洪水预警响应系统_渐进式迭代开发与升级设计.md",
    "面向区县防办的洪水预警响应系统_升级设计说明V3.md",
    "docs/progressive_upgrade/completion_traceability_audit.md",
    "docs/V3_upgrade_acceptance_matrix.md",
    "output/rag_evaluation/controlled_domain_sensitivity/controlled_domain_sensitivity.json",
    "output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.json",
    "output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation_cases.jsonl.gz",
    "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.json",
    "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation_cases.jsonl.gz",
    "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.json",
    "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md",
    "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity_cases.jsonl.gz",
    "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation.json",
    "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation_cases.jsonl.gz",
    "benchmarks/eurlex_temporal_selection.json",
    "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.json",
    "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.md",
    "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation_cases.jsonl.gz",
    "docs/progressive_upgrade/design_contract_evidence_policy.json",
    "output/acceptance/legacy_baseline_manifest.json",
    "output/acceptance/legacy_baseline_manifest.md",
    "output/acceptance/design_contract_audit.json",
    "output/acceptance/design_contract_audit.md",
)

EXTERNAL_NO_GO_ITEMS = (
    "authoritative_warning_gis_and_object_data",
    "production_postgresql_authority_switch",
    "production_idp_mfa_tls_kms",
    "real_cross_department_dispatch",
    "independent_expert_annotation_and_adjudication",
    "third_party_penetration_test",
    "remote_host_disaster_recovery",
    "real_role_uat",
    "legacy_production_traffic_retirement",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 evidence after newline normalization for cross-platform CI."""
    text = path.read_text(encoding="utf-8-sig")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _evidence_sha256(path: Path) -> str:
    """Hash text canonically and binary evidence as its exact stored bytes."""
    if path.suffix.lower() in {".gz", ".zip"}:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return _canonical_text_sha256(path)


def _check(
    requirement_id: str,
    title: str,
    passed: bool,
    evidence: Iterable[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "title": title,
        "status": "PASS" if passed else "FAIL",
        "evidence": list(evidence),
        "details": details or {},
    }


def check_artifact_groups(
    repo_root: Path,
    requirement_id: str,
    title: str,
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    missing = {
        group: [relative for relative in paths if not (repo_root / relative).is_file()]
        for group, paths in groups.items()
    }
    missing = {group: paths for group, paths in missing.items() if paths}
    evidence = list(
        dict.fromkeys(relative for paths in groups.values() for relative in paths)
    )
    return _check(
        requirement_id,
        title,
        not missing,
        evidence,
        {"group_count": len(groups), "missing": missing},
    )


def check_gate_one(candidate_report: dict[str, Any]) -> dict[str, Any]:
    baseline = candidate_report["baseline"]
    stress = candidate_report["missing_feature_stress"]
    calibration_improvement = float(candidate_report["calibration_improvement"])
    passed = (
        float(baseline["recall_at_k"]) >= 0.95
        and float(baseline["critical_miss_rate"]) <= 0.05
        and float(baseline["ece"]) <= 0.10
        and calibration_improvement > 0
        and float(stress["recall_drop_percentage_points"]) <= 10.0
    )
    return _check(
        "gate_1_candidate_quality",
        "Gate 1 候选质量达到冻结的受控阈值",
        passed,
        ["output/candidate_evaluation/report.json"],
        {
            "recall_at_5": baseline["recall_at_k"],
            "critical_miss_rate": baseline["critical_miss_rate"],
            "ece": baseline["ece"],
            "calibration_improvement": calibration_improvement,
            "missing_feature_recall_drop_percentage_points": stress[
                "recall_drop_percentage_points"
            ],
        },
    )


def _compose_services(compose_text: str) -> set[str]:
    services_block = compose_text.split("services:", 1)[1].split("\nvolumes:", 1)[0]
    return set(re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", services_block))


def build_progressive_completion_audit(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    scenario_report = _load_json(repo_root / EVIDENCE_FILES[0])
    candidate_report = _load_json(repo_root / EVIDENCE_FILES[1])
    performance_report = _load_json(repo_root / EVIDENCE_FILES[2])
    rag_report = _load_json(repo_root / EVIDENCE_FILES[3])
    openapi = _load_json(repo_root / EVIDENCE_FILES[4])
    compose_text = (repo_root / EVIDENCE_FILES[5]).read_text(encoding="utf-8-sig")
    committed_design_audit = _load_json(
        repo_root / "output/acceptance/design_contract_audit.json"
    )
    rebuilt_design_audit = build_design_contract_audit(repo_root)
    design_summary = rebuilt_design_audit["summary"]
    design_contract_passed = (
        committed_design_audit == rebuilt_design_audit
        and design_summary["item_count"] == 89
        and design_summary["controlled_or_local_pass_count"] == 87
        and design_summary["external_no_go_ids"] == ["18.4-10", "22.4-10"]
        and design_summary["all_items_accounted"] is True
        and design_summary["controlled_scope_complete"] is True
        and design_summary["full_production_complete"] is False
        and rebuilt_design_audit["legacy_baseline"]["offline_reproducible"] is True
    )

    scenario_summary = scenario_report["summary"]
    scenario_passed = (
        scenario_summary["status"] == "PASS"
        and int(scenario_summary["scenario_count"]) == 24
        and int(scenario_summary["covered_count"]) == 24
        and int(scenario_summary["missing_evidence_count"]) == 0
    )

    metadata = candidate_report["metadata"]
    provenance_passed = (
        metadata["data_origin"] == "SYNTHETIC"
        and metadata["is_simulated"] is True
        and isinstance(metadata["seed"], int)
        and bool(metadata["manifest_sha256"])
    )

    openapi_paths = set(openapi["paths"])
    missing_paths = sorted(REQUIRED_RESPONSE_PATHS - openapi_paths)
    compose_services = _compose_services(compose_text)
    missing_services = sorted(
        {"backend", "frontend", "worker", "postgis"} - compose_services
    )

    rag_decision = rag_report["decision"]
    conflict_slice = rag_report["challenge_slices"]["conflict_and_stale"]
    experiment_audit = rag_report["design_16_2_experiment_audit"]
    housing_ablation = experiment_audit["housing_real_model_ablation"]
    housing_weights = experiment_audit["housing_public_weight_sensitivity"]
    lawshift_ablation = experiment_audit["lawshift_temporal_ablation"]
    eurlex_ablation = experiment_audit["eurlex_temporal_ablation"]
    gate_two_is_safely_held = (
        rag_decision["gate_2"] == "NO-GO"
        and rag_decision["pipeline_feasible"] is True
        and rag_decision["evidence_f1_superiority_on_all_primary_datasets"] is False
        and conflict_slice["status"] == "RUN"
        and int(conflict_slice["cases"]) == 458
        and conflict_slice["strongest_reproducible_baseline"] == "coverage_greedy_proxy"
        and experiment_audit["status"] == "PARTIAL"
        and rag_decision["full_outperforms_w_o_role_and_w_o_field"] is False
        and experiment_audit["token_budget_sensitivity"]["status"] == "RUN"
        and experiment_audit["token_budget_sensitivity"]["all_methods_within_budget"]
        is True
        and experiment_audit["missing_ratio_sensitivity"]["status"] == "RUN"
        and experiment_audit["controlled_domain_sensitivity"]["status"]
        == "RUN_CONTROLLED_DOMAIN"
        and experiment_audit["sensitivity_coverage"]["role_and_field_weights"]
        == "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
        and housing_weights["status"]
        == "RUN_PUBLIC_EXPERT_FIELD_REAL_MODEL_ROLE_DIAGNOSTIC"
        and housing_weights["decision"]["public_real_model_weight_sensitivity_complete"]
        is True
        and housing_weights["decision"]["frozen_parameters_changed"] is False
        and housing_weights["decision"]["gate_2"] == "NO-GO"
        and experiment_audit["technical_sensitivity_matrix_complete"] is True
        and experiment_audit["flood_domain_expert_validation_complete"] is False
        and experiment_audit["sensitivity_coverage"]["conflict_threshold"]
        == "RUN_REAL_MODEL_CONFLICTS"
        and experiment_audit["conflicts_real_model_ablation"]["status"]
        == "RUN_REAL_MODEL_CONFLICTS"
        and housing_ablation["status"]
        == "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021"
        and housing_ablation["decision"]["gate_2"] == "NO-GO"
        and housing_ablation["decision"]["full_strictly_better_than_w_o_field"] is True
        and housing_ablation["decision"][
            "full_field_coverage_gain_over_strongest_baseline_at_least_0_05"
        ]
        is False
        and lawshift_ablation["status"]
        == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
        and lawshift_ablation["decision"]["gate_2"] == "NO-GO"
        and lawshift_ablation["decision"]["full_strictly_better_than_w_o_applicability"]
        is True
        and lawshift_ablation["decision"][
            "full_exact_gain_over_strongest_baseline_at_least_0_05"
        ]
        is False
        and eurlex_ablation["status"]
        == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
        and eurlex_ablation["decision"]["gate_2"] == "NO-GO"
        and eurlex_ablation["decision"]["full_strictly_better_than_w_o_applicability"]
        is True
        and eurlex_ablation["decision"][
            "full_exact_gain_over_strongest_baseline_at_least_0_05"
        ]
        is False
        and experiment_audit["applicability_identifiability"]["version_replacement"]
        == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
        and experiment_audit["applicability_identifiability"][
            "effective_or_expiry_dates"
        ]
        == "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL"
        and experiment_audit["applicability_identifiability"]["complete"] is True
        and int(experiment_audit["combined_ablation_coverage"]["run_count"]) == 9
        and experiment_audit["combined_ablation_coverage"]["missing_variants"] == []
        and rag_decision["design_16_2_experiment_coverage_complete"] is False
    )
    service_source = (
        repo_root / "flood_system/response_workflow/service.py"
    ).read_text(encoding="utf-8")
    gate_two_is_safely_held = gate_two_is_safely_held and all(
        marker in service_source
        for marker in (
            "feature.frc_rag_formal_enabled",
            "FRC-RAG formal retrieval is gated off",
            "RetrievalMode.BASELINE_ONLY",
            "RetrievalMode.SHADOW",
        )
    )

    performance_decision = performance_report["decision"]
    performance_passed = performance_decision["status"] == "PASS" and all(
        item["passed"] for item in performance_decision["checks"]
    )

    boundary_document = (
        repo_root / "docs/progressive_upgrade/completion_traceability_audit.md"
    ).read_text(encoding="utf-8-sig")
    boundary_markers = (
        "真实预警/GIS/对象主数据",
        "生产 PostgreSQL",
        "真实 IdP/MFA",
        "生产 TLS/KMS",
        "真实跨部门下发",
        "第三方渗透测试",
        "异地主机恢复",
        "两名独立业务专家",
        "真实岗位 UAT",
        "生产旧流量归零",
    )
    missing_boundary_markers = [
        marker for marker in boundary_markers if marker not in boundary_document
    ]

    requirements = [
        check_artifact_groups(
            repo_root,
            "software_deliverables",
            "第 20.1 节软件交付物齐备",
            SOFTWARE_DELIVERABLES,
        ),
        check_artifact_groups(
            repo_root,
            "document_deliverables",
            "第 20.3 节文档交付物齐备",
            DOCUMENT_DELIVERABLES,
        ),
        check_artifact_groups(
            repo_root,
            "data_experiment_deliverables",
            "第 20.2 节数据与实验交付物齐备",
            DATA_EXPERIMENT_DELIVERABLES,
        ),
        _check(
            "openapi_contract",
            "受控响应工作流通过版本化 Core API 暴露",
            not missing_paths,
            ["docs/openapi.json"],
            {
                "required_path_count": len(REQUIRED_RESPONSE_PATHS),
                "missing_paths": missing_paths,
            },
        ),
        _check(
            "compose_environment",
            "完整受控环境声明 Web、API、Worker 和 PostGIS 服务",
            not missing_services,
            ["docker-compose.yml"],
            {
                "services": sorted(compose_services),
                "missing_services": missing_services,
            },
        ),
        _check(
            "simulation_provenance",
            "Gate 0 模拟数据具有来源、版本、种子和清单溯源",
            provenance_passed,
            ["output/candidate_evaluation/report.json"],
            {
                "data_origin": metadata["data_origin"],
                "is_simulated": metadata["is_simulated"],
                "seed": metadata["seed"],
                "manifest_sha256": metadata["manifest_sha256"],
            },
        ),
        _check(
            "scenario_acceptance",
            "24 个受控正常与故障场景均具有可执行测试证据",
            scenario_passed,
            ["output/floodagent_bench/scenario_acceptance_report.json"],
            dict(scenario_summary),
        ),
        check_gate_one(candidate_report),
        _check(
            "gate_2_safe_fallback",
            "Gate 2 使用真实证据且未获支持的 FRC 正式切流保持阻断",
            gate_two_is_safely_held,
            [
                "output/rag_evaluation/public_frc_reference/public_frc_reference_report.json",
                "output/rag_evaluation/controlled_domain_sensitivity/controlled_domain_sensitivity.json",
                "output/rag_evaluation/conflicts_frc_ablation/conflicts_frc_ablation.json",
                "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation.json",
                "output/rag_evaluation/housing_frc_ablation/housing_frc_ablation_cases.jsonl.gz",
                "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.json",
                "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity_cases.jsonl.gz",
                "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation.json",
                "output/rag_evaluation/lawshift_temporal_ablation/lawshift_temporal_ablation_cases.jsonl.gz",
                "benchmarks/eurlex_temporal_selection.json",
                "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation.json",
                "output/rag_evaluation/eurlex_temporal_ablation/eurlex_temporal_ablation_cases.jsonl.gz",
                "flood_system/response_workflow/service.py",
            ],
            {
                "gate_2": rag_decision["gate_2"],
                "pipeline_feasible": rag_decision["pipeline_feasible"],
                "conflicts_cases": conflict_slice["cases"],
                "strongest_reproducible_baseline": conflict_slice[
                    "strongest_reproducible_baseline"
                ],
                "baseline_accuracy": conflict_slice["baseline_accuracy"],
                "frc_accuracy": conflict_slice["frc_accuracy"],
                "ablation_variants_planned": experiment_audit[
                    "combined_ablation_coverage"
                ]["planned_count"],
                "experiment_coverage_complete": experiment_audit["coverage_complete"],
                "token_budget_sensitivity": experiment_audit[
                    "token_budget_sensitivity"
                ]["status"],
                "missing_ratio_sensitivity": experiment_audit[
                    "missing_ratio_sensitivity"
                ]["status"],
                "controlled_domain_sensitivity": experiment_audit[
                    "controlled_domain_sensitivity"
                ]["status"],
                "role_field_weight_sensitivity": experiment_audit[
                    "sensitivity_coverage"
                ]["role_and_field_weights"],
                "conflict_threshold_sensitivity": experiment_audit[
                    "sensitivity_coverage"
                ]["conflict_threshold"],
                "conflicts_real_model_ablation": experiment_audit[
                    "conflicts_real_model_ablation"
                ]["status"],
                "housing_real_model_ablation": housing_ablation["status"],
                "housing_full_field_coverage": housing_ablation["aggregates"][
                    "frc_full"
                ]["field_coverage"],
                "housing_strongest_baseline": housing_ablation["strongest_baseline"],
                "housing_weight_sensitivity": housing_weights["status"],
                "housing_weight_field_values": housing_weights["dimensions"][
                    "field_weight"
                ],
                "housing_weight_role_values": housing_weights["dimensions"][
                    "role_weight"
                ],
                "technical_sensitivity_matrix_complete": experiment_audit[
                    "technical_sensitivity_matrix_complete"
                ],
                "flood_domain_expert_validation_complete": experiment_audit[
                    "flood_domain_expert_validation_complete"
                ],
                "lawshift_temporal_ablation": lawshift_ablation["status"],
                "lawshift_full_exact_evidence_accuracy": lawshift_ablation[
                    "aggregates"
                ]["frc_full"]["exact_evidence_accuracy"],
                "lawshift_strongest_baseline": lawshift_ablation["strongest_baseline"],
                "eurlex_temporal_ablation": eurlex_ablation["status"],
                "eurlex_full_exact_evidence_accuracy": eurlex_ablation["aggregates"][
                    "frc_full"
                ]["exact_evidence_accuracy"],
                "eurlex_strongest_baseline": eurlex_ablation["strongest_baseline"],
                "version_replacement_applicability": experiment_audit[
                    "applicability_identifiability"
                ]["version_replacement"],
                "effective_or_expiry_dates": experiment_audit[
                    "applicability_identifiability"
                ]["effective_or_expiry_dates"],
                "ablation_variants_run": experiment_audit["combined_ablation_coverage"][
                    "run_count"
                ],
            },
        ),
        _check(
            "controlled_performance",
            "全部冻结的受控 API 性能预算通过",
            performance_passed,
            ["output/performance/controlled_performance_report.json"],
            {
                "status": performance_decision["status"],
                "check_count": len(performance_decision["checks"]),
            },
        ),
        _check(
            "external_boundaries",
            "生产、真实数据与独立人员条件明确保持外部 No-Go",
            not missing_boundary_markers,
            ["docs/progressive_upgrade/completion_traceability_audit.md"],
            {"missing_boundary_markers": missing_boundary_markers},
        ),
        _check(
            "design_contract_traceability",
            "第 18.3—18.6、22 和 23 节的 89 条显式合同逐条归属且外部条件不冒充完成",
            design_contract_passed,
            [
                "docs/progressive_upgrade/design_contract_evidence_policy.json",
                "output/acceptance/legacy_baseline_manifest.json",
                "output/acceptance/design_contract_audit.json",
                "output/acceptance/design_contract_audit.md",
                "tests/test_design_contract_audit.py",
            ],
            {
                "committed_report_matches_rebuild": committed_design_audit
                == rebuilt_design_audit,
                "item_count": design_summary["item_count"],
                "controlled_or_local_pass_count": design_summary[
                    "controlled_or_local_pass_count"
                ],
                "external_no_go_ids": design_summary["external_no_go_ids"],
                "legacy_baseline_source_commit": rebuilt_design_audit[
                    "legacy_baseline"
                ]["source_commit"],
                "legacy_database_sha256": rebuilt_design_audit["legacy_baseline"][
                    "database"
                ]["sha256"],
                "legacy_rag_index_sha256": rebuilt_design_audit["legacy_baseline"][
                    "rag_index"
                ]["sha256"],
            },
        ),
    ]

    passed_count = sum(item["status"] == "PASS" for item in requirements)
    evidence_hashes = {
        relative: _evidence_sha256(repo_root / relative) for relative in EVIDENCE_FILES
    }
    controlled_status = "PASS" if passed_count == len(requirements) else "FAIL"
    return {
        "metadata": {
            "audit_version": AUDIT_VERSION,
            "scope": "single-district controlled simulation",
            "contract": "洪水预警响应系统_渐进式迭代开发与升级设计.md sections 18-23",
            "evidence_hash_canonicalization": (
                "UTF-8 text without BOM with CRLF/CR normalized to LF; binary archives use raw bytes"
            ),
        },
        "summary": {
            "controlled_first_iteration": controlled_status,
            "passed_local_requirements": passed_count,
            "local_requirement_count": len(requirements),
            "frc_gate_2": rag_decision["gate_2"],
            "production_readiness": "NO-GO_EXTERNAL",
            "overall": (
                "CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO"
                if controlled_status == "PASS"
                else "CONTROLLED_SCOPE_INCOMPLETE"
            ),
        },
        "requirements": requirements,
        "evidence_sha256": evidence_hashes,
        "external_no_go": [
            {
                "item": item,
                "status": "NO-GO_EXTERNAL",
                "reason": "requires authoritative organization, production environment, or independent human evidence",
            }
            for item in EXTERNAL_NO_GO_ITEMS
        ],
        "interpretation": (
            "PASS proves the repository's controlled first-iteration contract only. "
            "It does not convert Gate 2 or any external production/UAT item into GO."
        ),
    }


def write_progressive_completion_audit(
    repo_root: Path,
    json_target: Path,
    markdown_target: Path,
) -> tuple[Path, Path]:
    report = build_progressive_completion_audit(repo_root)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = report["summary"]
    lines = [
        "# 渐进式升级机器可读完成性审计",
        "",
        f"- 受控首期迭代：`{summary['controlled_first_iteration']}`",
        f"- 本地要求：{summary['passed_local_requirements']}/{summary['local_requirement_count']}",
        f"- FRC-RAG Gate 2：`{summary['frc_gate_2']}`",
        f"- 生产就绪：`{summary['production_readiness']}`",
        f"- 总体边界：`{summary['overall']}`",
        "",
        "> PASS 只证明单一区域受控模拟首期闭环；不会把 Gate 2 或真实生产/UAT 外部条件改写为 GO。",
        "",
        "## 本地要求",
        "",
        "| 要求 | 状态 | 证据 |",
        "|---|---|---|",
    ]
    for requirement in report["requirements"]:
        evidence = "<br>".join(f"`{item}`" for item in requirement["evidence"])
        lines.append(
            f"| {requirement['title']} | {requirement['status']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## 外部 No-Go",
            "",
            "| 项目 | 状态 |",
            "|---|---|",
        ]
    )
    for item in report["external_no_go"]:
        lines.append(f"| `{item['item']}` | {item['status']} |")
    markdown_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_target, markdown_target
