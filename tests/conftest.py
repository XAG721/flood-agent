from __future__ import annotations

import pytest


LOCAL_EVIDENCE_TESTS = {
    "test_completion_audit.py::test_completion_audit_proves_controlled_scope_without_overstating_production",
    "test_completion_audit.py::test_hotpot_v84_cardinality_boundary_is_audited",
    "test_completion_audit.py::test_completion_audit_output_is_deterministic",
    "test_frc_doc2dial_document_contrastive_role_closure.py::test_protocol_and_source_registration_preserve_prospective_boundary",
    "test_frc_doc2dial_wood_document_contrastive_transfer.py::test_protocol_source_and_member_boundaries_are_registered",
    "test_frc_doc2dial_wood_schema_corrected_transfer.py::test_protocol_and_source_disclose_schema_only_access",
    "test_frc_musique_anchor_default_transfer_lock.py::test_v80_source_registry_revalidates_nested_maps_ids_and_dependencies",
    "test_frc_musique_anchor_default_transfer_lock.py::test_v80_runner_registration_verification_passes_before_target_selection",
    "test_frc_musique_mean_calibrated_three_route_lock.py::test_v78_source_registry_revalidates_nested_prior_maps_and_dependencies",
    "test_frc_musique_target_three_route_lock.py::test_v79_source_registry_revalidates_nested_maps_and_dependencies",
    "test_frc_squad2_dual_support_union.py::test_prior_union_contains_all_three_disjoint_600_case_maps",
    "test_frc_squad2_structured_span_gate.py::test_prior_exclusion_union_requires_two_disjoint_600_row_maps",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    local_evidence = pytest.mark.local_evidence
    for item in items:
        relative_nodeid = item.nodeid.removeprefix("tests/")
        if relative_nodeid in LOCAL_EVIDENCE_TESTS:
            item.add_marker(local_evidence)
