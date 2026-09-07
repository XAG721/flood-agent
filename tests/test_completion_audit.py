from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flood_system.completion_audit import (
    _canonical_text_sha256,
    _evidence_sha256,
    build_progressive_completion_audit,
    check_artifact_groups,
    check_gate_one,
    write_progressive_completion_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_completion_audit_proves_controlled_scope_without_overstating_production() -> (
    None
):
    report = build_progressive_completion_audit(REPO_ROOT)

    assert report["summary"] == {
        "controlled_first_iteration": "PASS",
        "passed_local_requirements": 67,
        "local_requirement_count": 67,
        "frc_gate_2": "NO-GO",
        "production_readiness": "NO-GO_EXTERNAL",
        "overall": "CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO",
    }
    assert all(item["status"] == "PASS" for item in report["requirements"])
    assert len(report["external_no_go"]) == 9
    assert all(item["status"] == "NO-GO_EXTERNAL" for item in report["external_no_go"])
    gate_two = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "gate_2_safe_fallback"
    )
    assert gate_two["details"]["qasc_scoring_optimization_status"] == (
        "KEEP_FROZEN_PER_CASE_SCORING_ENTRYPOINT"
    )
    assert gate_two["details"]["qasc_scoring_optimization_speedups"] == {
        "8": 1.03091,
        "16": 1.036633,
        "32": 1.023087,
    }
    assert gate_two["details"]["qasc_scoring_optimization_equivalence"] == {
        "8": True,
        "16": False,
        "32": False,
    }
    assert gate_two["details"]["qasc_scoring_reference_rewritten"] is False
    assert gate_two["details"]["conflicts_expected_behavior_status"] == (
        "GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION"
    )
    assert gate_two["details"]["conflicts_expected_behavior_case_count"] == 458
    assert gate_two["details"]["conflicts_expected_behavior_response_count"] == 916
    assert (
        gate_two["details"]["conflicts_expected_behavior_human_evidence_complete"]
        is False
    )
    behavior_generation = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "conflicts_expected_behavior_generation"
    )
    assert behavior_generation["details"]["methods_hidden_from_reviewers"] is True
    assert behavior_generation["details"]["human_evidence_complete"] is False
    assert (
        behavior_generation["details"]["blind_mapping_published_with_package"] is False
    )
    annotation_operations = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "conflicts_annotation_operations_preparation"
    )
    assert annotation_operations["details"]["reviewer_slots"] == [
        "reviewer_1",
        "reviewer_2",
        "adjudicator",
    ]
    assert annotation_operations["details"]["batch_count_per_slot"] == 8
    assert annotation_operations["details"]["primary_task_count_per_slot"] == 458
    assert annotation_operations["details"]["repeat_task_count_per_slot"] == 23
    assert annotation_operations["details"]["total_task_count_per_slot"] == 481
    assert annotation_operations["details"]["public_file_count"] == 48
    assert annotation_operations["details"]["routing_published"] is False
    assert (
        annotation_operations["details"]["method_identity_present_in_public_files"]
        is False
    )
    assert annotation_operations["details"]["human_evidence_complete"] is False
    workstation = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "conflicts_annotation_workstation_readiness"
    )
    assert workstation["details"]["status"] == ("LOCAL_REVIEW_UI_READY_AWAITING_HUMANS")
    assert workstation["details"]["reviewer_slots"] == [
        "reviewer_1",
        "reviewer_2",
        "adjudicator",
    ]
    assert workstation["details"]["batch_count_per_slot"] == 8
    assert workstation["details"]["task_count_per_slot"] == 481
    assert workstation["details"]["implementation_file_count"] == 6
    assert workstation["details"]["loopback_only"] is True
    assert workstation["details"]["draft_storage_git_ignored"] is True
    assert workstation["details"]["private_routing_loaded"] is False
    assert workstation["details"]["method_identity_loaded"] is False
    assert workstation["details"]["browser_probe_status"] == (
        "PASS_PRIVATE_NON_HUMAN_PROBE_REMOVED"
    )
    assert workstation["details"]["human_evidence_complete"] is False
    assert workstation["details"]["gate_2"] == "NO-GO/SHADOW"
    collection = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "conflicts_annotation_collection_readiness"
    )
    assert collection["details"]["status"] == (
        "PRIVATE_COLLECTION_ORCHESTRATION_READY_AWAITING_HUMANS"
    )
    assert collection["details"]["expected_finalized_batch_count"] == 24
    assert collection["details"]["minimum_exact_agreement"] == 0.8
    assert collection["details"]["implementation_file_count"] == 3
    assert collection["details"]["real_private_status"] == (
        "AWAITING_INDEPENDENT_HUMAN_BATCHES"
    )
    assert collection["details"]["real_present_batch_count"] == 0
    assert collection["details"]["real_finalized_batch_count"] == 0
    assert collection["details"]["blind_mapping_loaded"] is False
    assert collection["details"]["synthetic_test_data_is_human_evidence"] is False
    assert collection["details"]["human_evidence_complete"] is False
    assert collection["details"]["gate_2"] == "NO-GO/SHADOW"
    router = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "conflicts_selector_router_discovery_boundary"
    )
    assert router["details"]["status"] == (
        "DISCOVERY_ROUTING_SIGNAL_NOT_ADOPTED_FRC_CONTRIBUTION_NOT_ESTABLISHED"
    )
    assert router["details"]["retrospective_discovery"] is True
    assert router["details"]["pre_registered_confirmation"] is False
    assert router["details"]["case_count"] == 458
    assert router["details"]["feature_counts"] == {
        "all_methods": 47,
        "without_frc": 45,
    }
    assert router["details"]["static_accuracy"] == 0.344978
    assert router["details"]["all_methods_accuracy"] == 0.395197
    assert router["details"]["without_frc_accuracy"] == 0.393013
    assert router["details"]["all_minus_static"]["mean_difference"] == 0.050218
    assert router["details"]["all_minus_static"]["ci_low"] == 0.008734
    assert router["details"]["all_minus_static"]["ci_high"] == 0.091703
    assert router["details"]["all_minus_without_frc"]["mean_difference"] == (0.002183)
    assert router["details"]["frc_unique_oracle_correct_cases"] == 2
    assert router["details"]["outdated_conflict_recall_delta"] == -0.435483
    assert router["details"]["folds_with_nonnegative_accuracy_gain"] == 3
    assert router["details"]["requires_all_method_predictions_at_inference"] is True
    assert router["details"]["router_adopted"] is False
    assert router["details"]["frc_selector_replaced"] is False
    assert router["details"]["independent_confirmation_required"] is True
    assert router["details"]["gate_2"] == "NO-GO/SHADOW"
    whoqa = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "whoqa_independent_selection_boundary"
    )
    assert whoqa["details"]["status"] == ("WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED")
    assert whoqa["details"]["case_count"] == 5152
    assert whoqa["details"]["template_count_distribution"] == {
        "5": 4753,
        "8": 399,
    }
    assert whoqa["details"]["selection_runs"] == 161742
    assert whoqa["details"]["strongest_baseline"] == "bm25_topk"
    assert whoqa["details"]["bm25_primary_coverage"] == 0.995203
    assert whoqa["details"]["frc_primary_coverage"] == 0.994919
    assert whoqa["details"]["frc_minus_strongest"] == -0.000283
    assert whoqa["details"]["frc_minus_strongest_simultaneous_ci"]["ci_low"] == -0.0011
    assert (
        whoqa["details"]["frc_minus_strongest_simultaneous_ci"]["ci_high"] == 0.000039
    )
    assert whoqa["details"]["frc_minus_coverage_proxy"]["point"] == -0.00011
    assert whoqa["details"]["official_viewpoint_count_mismatches"] == 0
    assert whoqa["details"]["worst_viewpoint_count_delta"] == -0.002485
    assert whoqa["details"]["worst_property_delta"] == -0.002156
    assert whoqa["details"]["deterministic_output_rerun"] == ("2/2 byte-identical")
    assert whoqa["details"]["gate_2"] == "NO-GO/SHADOW"
    assert whoqa["details"]["canary_or_default_authorized"] is False
    whoqa_stress = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "whoqa_budget_stress_boundary"
    )
    assert whoqa_stress["details"]["status"] == ("WHOQA_STRESS_SUPPORT_NOT_ESTABLISHED")
    assert whoqa_stress["details"]["case_count"] == 5152
    assert whoqa_stress["details"]["template_count"] == 26957
    assert whoqa_stress["details"]["selection_runs"] == 970452
    assert whoqa_stress["details"]["configuration_count"] == 6
    assert whoqa_stress["details"]["strongest_baseline"] == "dense_topk"
    assert whoqa_stress["details"]["dense_family_primary"] == 0.890719
    assert whoqa_stress["details"]["frc_family_primary"] == 0.887793
    assert whoqa_stress["details"]["frc_minus_strongest"] == -0.002926
    assert (
        whoqa_stress["details"]["frc_minus_strongest_simultaneous_ci"]["ci_low"]
        == -0.003858
    )
    assert (
        whoqa_stress["details"]["frc_minus_strongest_simultaneous_ci"]["ci_high"]
        == -0.002025
    )
    assert whoqa_stress["details"]["configuration_deltas"] == {
        "k2_b1500": -0.000633,
        "k3_b1500": -0.00016,
        "k4_b256": -0.016657,
        "k4_b512": -0.004657,
        "k4_b1024": -0.000341,
        "k4_b1500": -0.000283,
    }
    assert whoqa_stress["details"]["worst_configuration_delta"] == -0.016657
    assert whoqa_stress["details"]["worst_stratum_delta"] == -0.012057
    assert whoqa_stress["details"]["selector_changed"] is False
    assert whoqa_stress["details"]["independent_confirmation"] is False
    assert whoqa_stress["details"]["deterministic_output_rerun"] == (
        "2/2 byte-identical"
    )
    assert whoqa_stress["details"]["gate_2"] == "NO-GO/SHADOW"
    assert whoqa_stress["details"]["canary_or_default_authorized"] is False
    rgb = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "rgb_cost_aware_frc_boundary"
    )
    assert rgb["details"]["status"] == "RGB_COST_AWARE_FRC_SAFETY_REGRESSION"
    assert rgb["details"]["case_count"] == 498
    assert rgb["details"]["candidate_chunks"] == 14580
    assert rgb["details"]["selection_runs"] == 13446
    assert rgb["details"]["strongest_baseline"] == "coverage_greedy_proxy"
    assert rgb["details"]["cost_aware_primary"] == 0.507421
    assert rgb["details"]["old_frc_primary"] == 0.551199
    assert rgb["details"]["cost_aware_minus_old_frc"]["point"] == -0.043778
    assert rgb["details"]["cost_aware_minus_strongest"]["point"] == -0.048719
    assert (
        rgb["details"]["cost_aware_minus_strongest_simultaneous_ci"]["ci_low"]
        == -0.066307
    )
    assert (
        rgb["details"]["cost_aware_minus_strongest_simultaneous_ci"]["ci_high"]
        == -0.032655
    )
    assert rgb["details"]["worst_dataset_budget_delta"] == -0.127688
    assert rgb["details"]["positive_wrong_rate_delta"] == -0.022449
    assert rgb["details"]["selector_changed"] is False
    assert rgb["details"]["deterministic_output_rerun"] == "2/2 byte-identical"
    assert rgb["details"]["gate_2"] == "NO-GO/SHADOW"
    assert rgb["details"]["canary_or_default_authorized"] is False
    musique = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "musique_dual_resource_frc_boundary"
    )
    assert musique["status"] == "PASS"
    assert (
        musique["details"]["status"]
        == "MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_NOT_ESTABLISHED"
    )
    assert musique["details"]["case_count"] == 2417
    assert musique["details"]["candidate_chunks"] == 48656
    assert musique["details"]["selection_runs"] == 72510
    assert musique["details"]["hop_distribution"] == {
        "2": 1252,
        "3": 760,
        "4": 405,
    }
    assert musique["details"]["strongest_baseline"] == "cross_encoder_topk"
    assert musique["details"]["dual_resource_primary"] == 0.528545
    assert musique["details"]["old_frc_primary"] == 0.530801
    assert musique["details"]["dual_minus_v35"]["point"] == 0.004402
    assert musique["details"]["dual_minus_v35"]["ci_low"] == 0.002244
    assert musique["details"]["dual_minus_old_frc"]["point"] == -0.002256
    assert musique["details"]["dual_minus_strongest"]["point"] == -0.001854
    assert (
        musique["details"]["dual_minus_strongest_simultaneous_ci"]["ci_low"]
        == -0.003784
    )
    assert (
        musique["details"]["dual_minus_strongest_simultaneous_ci"]["ci_high"]
        == -0.00002
    )
    assert musique["details"]["budget_deltas"] == {
        "512": -0.00457,
        "1024": -0.00092,
        "1500": -0.000194,
    }
    assert musique["details"]["hop_deltas"] == {
        "2-hop": -0.004806,
        "3-hop": 0.003234,
        "4-hop": -0.002514,
    }
    assert musique["details"]["selector_changed"] is False
    assert musique["details"]["deterministic_output_rerun"] == ("2/2 byte-identical")
    assert musique["details"]["gate_2"] == "NO-GO/SHADOW"
    assert musique["details"]["canary_or_default_authorized"] is False
    gap = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "musique_objective_gap_boundary"
    )
    assert gap["status"] == "PASS"
    assert gap["details"]["status"] == ("MIXED_OPTIMIZATION_AND_ALIGNMENT_DIAGNOSTIC")
    assert gap["details"]["objective_gap"]["mean_normalized_regret"] == 0.000809
    assert gap["details"]["objective_gap"]["near_zero_regret_rate_at_1e_9"] == (0.8494)
    assert gap["details"]["exact_primary"] == 0.529945
    assert gap["details"]["exact_minus_v36"]["point"] == 0.001399
    assert gap["details"]["exact_minus_v36"]["ci_low"] == -0.000208
    assert gap["details"]["exact_minus_cross_encoder_topk"]["point"] == (-0.000455)
    assert gap["details"]["adoption_eligible"] is False
    assert gap["details"]["selector_changed"] is False
    assert gap["details"]["gate_2"] == "NO-GO/SHADOW"
    crossfit = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "musique_crossfit_support_boundary"
    )
    assert crossfit["status"] == "PASS"
    assert crossfit["details"]["status"] == ("CROSSFIT_SUPPORT_SIGNAL_NOT_ESTABLISHED")
    assert crossfit["details"]["base_primary"] == 0.537234
    assert crossfit["details"]["full_primary"] == 0.537782
    assert crossfit["details"]["full_minus_cross_encoder_topk"]["point"] == (0.007383)
    assert crossfit["details"]["full_minus_cross_encoder_topk"]["ci_low"] == (0.003425)
    assert crossfit["details"]["full_minus_base"]["point"] == 0.000548
    assert crossfit["details"]["full_minus_base"]["ci_low"] == -0.000744
    assert crossfit["details"]["adoption_eligible"] is False
    assert crossfit["details"]["independent_confirmation"] is False
    assert crossfit["details"]["selector_changed"] is False
    assert crossfit["details"]["gate_2"] == "NO-GO/SHADOW"
    hover = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "hover_verification_roles_boundary"
    )
    assert hover["status"] == "PASS"
    assert hover["details"]["status"] == (
        "HOVER_VERIFICATION_ROLE_SUPPORT_NOT_ESTABLISHED"
    )
    assert hover["details"]["case_count"] == 4000
    assert hover["details"]["candidate_documents"] == 80000
    assert hover["details"]["candidate_chunks"] == 81224
    assert hover["details"]["selection_runs"] == 84000
    assert hover["details"]["candidate_ceiling"] == {
        "mean": 0.6085,
        "complete_cases": 1060,
        "incomplete_cases": 2940,
    }
    assert hover["details"]["generic_roles_primary"] == 0.433321
    assert hover["details"]["verification_roles_primary"] == 0.433286
    assert hover["details"]["verification_minus_generic"]["point"] == -0.000035
    assert hover["details"]["verification_minus_generic"]["ci_low"] == -0.000187
    assert hover["details"]["verification_minus_strongest"]["point"] == 0.000267
    assert hover["details"]["selector_changed"] is False
    assert hover["details"]["gate_2"] == "NO-GO/SHADOW"
    mechanism = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "hover_static_role_mechanism_diagnostic"
    )
    assert mechanism["status"] == "PASS"
    assert mechanism["details"]["status"] == "STATIC_ROLE_SIGNAL_COLLAPSE_OBSERVED"
    assert mechanism["details"]["post_result_diagnostic"] is True
    assert mechanism["details"]["gold_used"] is False
    assert mechanism["details"]["generic_mean_spearman"] == 0.947723
    assert mechanism["details"]["verification_mean_spearman"] == 0.946191
    assert mechanism["details"]["causal_claim"] is False
    assert mechanism["details"]["gate_2"] == "NO-GO/SHADOW"
    dynamic = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "hover_dynamic_atomic_roles_boundary"
    )
    assert dynamic["status"] == "PASS"
    assert dynamic["details"]["v40_status"] == (
        "REGISTRATION_BOUNDARY_VIOLATED_BEFORE_SCORING"
    )
    assert dynamic["details"]["v40_excluded_ids"] == 2512
    assert dynamic["details"]["pilot_status"] == (
        "MECHANISM_ESTABLISHED_OPEN_CONFIRMATION"
    )
    assert dynamic["details"]["confirmation_status"] == (
        "DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED"
    )
    assert dynamic["details"]["confirmation_cases"] == 2000
    assert dynamic["details"]["generation_fallback_rate"] == 0.0145
    assert dynamic["details"]["strongest_non_frc"] == "cross_encoder_topk"
    assert (
        dynamic["details"]["method_primary_means"]["dynamic_rank_coverage_frc_v41"]
        == 0.428851
    )
    assert dynamic["details"]["dynamic_minus_static"] == {
        "point": 0.001593,
        "ci_low": 0.000453,
        "ci_high": 0.002762,
        "resamples": 10000,
        "seed": 20260801,
    }
    assert dynamic["details"]["dynamic_minus_strongest_non_frc"] == {
        "point": 0.001463,
        "ci_low": 0.000308,
        "ci_high": 0.002688,
        "resamples": 10000,
        "seed": 20260801,
    }
    assert dynamic["details"]["support_checks"] == {
        "dynamic_minus_static_point_at_least_0_005": False,
        "dynamic_minus_static_ci_low_above_0": True,
        "dynamic_minus_strongest_point_at_least_0_01": False,
        "dynamic_minus_strongest_ci_low_above_0": True,
        "every_budget_and_stratum_delta_at_least_minus_0_02": True,
    }
    assert dynamic["details"]["deterministic_output_rerun"] == ("2/2 byte-identical")
    assert dynamic["details"]["selector_adoption_authorized"] is False
    assert dynamic["details"]["gate_2"] == "NO-GO/SHADOW"
    scifact = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "scifact_dynamic_atomic_roles_external_boundary"
    )
    assert scifact["status"] == "PASS"
    assert scifact["details"]["status"] == (
        "SCIFACT_DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED"
    )
    assert scifact["details"]["cases"] == 187
    assert scifact["details"]["generation_fallback_rate"] == 0.005348
    assert (
        scifact["details"]["gold_free_mechanism"]["all_v41_mechanism_checks_passed"]
        is True
    )
    assert scifact["details"]["strongest_non_frc"] == "dense_topk"
    assert (
        scifact["details"]["method_primary_means"]["dynamic_rank_coverage_frc_v41"]
        == 0.403909
    )
    assert scifact["details"]["method_primary_means"]["dense_topk"] == 0.460616
    assert scifact["details"]["dynamic_minus_static"] == {
        "point": 0.002821,
        "ci_low": -0.007003,
        "ci_high": 0.012092,
        "resamples": 10000,
        "seed": 20260802,
    }
    assert scifact["details"]["dynamic_minus_strongest_non_frc"] == {
        "point": -0.056707,
        "ci_low": -0.078099,
        "ci_high": -0.036516,
        "resamples": 10000,
        "seed": 20260802,
    }
    assert scifact["details"]["support_checks"] == {
        "dynamic_minus_static_ci_low_above_0": False,
        "dynamic_minus_static_point_at_least_0_005": False,
        "dynamic_minus_strongest_ci_low_above_0": False,
        "dynamic_minus_strongest_point_at_least_0_01": False,
        "every_budget_and_supported_stratum_at_least_minus_0_02": False,
        "generation_fallback_rate_at_most_0_05": True,
        "no_forbidden_field_leak": True,
    }
    assert scifact["details"]["selector_adoption_authorized"] is False
    assert scifact["details"]["scifact_reuse_for_tuning_authorized"] is False
    assert scifact["details"]["gate_2"] == "NO-GO/SHADOW"
    feverous = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "feverous_adaptive_atomic_roles_external_boundary"
    )
    assert feverous["status"] == "PASS"
    assert feverous["details"]["status"] == "FEVEROUS_BOUNDED_POOL_INCONCLUSIVE"
    assert feverous["details"]["cases"] == 240
    assert feverous["details"]["generation_fallback_rate"] == 0.029167
    assert feverous["details"]["candidate_ceiling_complete_rate"] == 0.4625
    assert feverous["details"]["strongest_non_frc"] == "cross_encoder_topk"
    assert (
        feverous["details"]["method_aggregates"]["adaptive_argmax_cardinality_frc_v43"][
            "evidence_macro_f1"
        ]
        == 0.172505
    )
    assert (
        feverous["details"]["method_aggregates"]["dynamic_rank_coverage_frc_v41"][
            "evidence_macro_f1"
        ]
        == 0.218379
    )
    assert feverous["details"]["adaptive_minus_dynamic"] == {
        "point": -0.045874,
        "ci_low": -0.079872,
        "ci_high": -0.016021,
    }
    assert feverous["details"]["adaptive_minus_strongest_non_frc"] == {
        "point": -0.03678,
        "ci_low": -0.073102,
        "ci_high": -0.005646,
    }
    assert feverous["details"]["mean_selected_unit_reduction"] == 2.227778
    assert feverous["details"]["complete_evidence_recall_drop"] == 0.0875
    assert feverous["details"]["locked_evaluation_schema_fix"] == {
        "gold_metrics_computed_before_fix": True,
        "metric_values_observed_before_fix": False,
        "method_or_threshold_changed": False,
    }
    assert feverous["details"]["selector_adoption_authorized"] is False
    assert feverous["details"]["feverous_v43_reuse_for_tuning_authorized"] is False
    assert feverous["details"]["gate_2"] == "NO-GO/SHADOW"
    ottqa = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "ottqa_guarded_adaptive_atomic_roles_external_boundary"
    )
    assert ottqa["status"] == "PASS"
    assert (
        ottqa["details"]["status"] == "OTTQA_GUARDED_ADAPTIVE_SUPPORT_NOT_ESTABLISHED"
    )
    assert ottqa["details"]["cases"] == 240
    assert ottqa["details"]["source_modes"] == {"passage": 120, "table": 120}
    assert ottqa["details"]["generation_fallback_rate"] == 0.029167
    assert ottqa["details"]["candidate_ceiling_complete_rate"] == 0.995833
    assert ottqa["details"]["strongest_non_frc"] == "official_anchor_topk"
    assert (
        ottqa["details"]["method_aggregates"]["guarded_adaptive_cardinality_frc_v44"][
            "answer_evidence_macro_f1"
        ]
        == 0.18996
    )
    assert ottqa["details"]["guarded_minus_dynamic"] == {
        "point": 0.016093,
        "ci_low": 0.001488,
        "ci_high": 0.029713,
    }
    assert ottqa["details"]["guarded_minus_strongest_non_frc"] == {
        "point": -0.015061,
        "ci_low": -0.058018,
        "ci_high": 0.01418,
    }
    assert ottqa["details"]["source_mode_deltas"]["passage"]["delta"] == 0.029029
    assert ottqa["details"]["source_mode_deltas"]["table"]["delta"] == -0.158409
    assert ottqa["details"]["answer_evidence_recall_drop"] == 0.066667
    assert ottqa["details"]["selector_adoption_authorized"] is False
    assert ottqa["details"]["ottqa_v44_reuse_for_tuning_authorized"] is False
    assert ottqa["details"]["official_leaderboard_result"] is False
    assert ottqa["details"]["gate_2"] == "NO-GO/SHADOW"
    finqa = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "finqa_anchor_guarded_atomic_roles_external_boundary"
    )
    assert finqa["status"] == "PASS"
    assert finqa["details"]["status"] == "FINQA_ANCHOR_GUARDED_SUPPORT_NOT_ESTABLISHED"
    assert finqa["details"]["cases"] == 360
    assert finqa["details"]["source_modes"] == {
        "table_only": 120,
        "text_only": 120,
        "hybrid": 120,
    }
    assert finqa["details"]["gold_count_strata"] == {
        "single_gold_fact": 151,
        "multiple_gold_facts": 209,
    }
    assert finqa["details"]["generation_fallback_rate"] == 0.033333
    assert finqa["details"]["candidate_ceiling_complete_rate"] == 1.0
    assert finqa["details"]["strongest_non_frc"] == "cross_encoder_knapsack"
    assert (
        finqa["details"]["method_aggregates"][
            "anchor_guarded_adaptive_cardinality_frc_v45"
        ]["supporting_fact_macro_f1"]
        == 0.510429
    )
    assert (
        finqa["details"]["method_aggregates"]["adaptive_argmax_cardinality_frc_v43"][
            "supporting_fact_macro_f1"
        ]
        == 0.623946
    )
    assert finqa["details"]["v45_minus_v44"] == {
        "point": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }
    assert finqa["details"]["v45_minus_dynamic"] == {
        "point": 0.098177,
        "ci_low": 0.084628,
        "ci_high": 0.111273,
    }
    assert finqa["details"]["v45_minus_strongest_non_frc"] == {
        "point": 0.100927,
        "ci_low": 0.085851,
        "ci_high": 0.113962,
    }
    assert finqa["details"]["mean_selected_unit_reduction"] == 1.762037
    assert finqa["details"]["supporting_fact_recall_drop"] == 0.059937
    assert finqa["details"]["selector_adoption_authorized"] is False
    assert finqa["details"]["finqa_v45_reuse_for_tuning_authorized"] is False
    diagnostic = finqa["details"]["anchor_equivalence_diagnostic"]
    assert (
        diagnostic["status"]
        == "FINQA_V45_V44_SET_EQUIVALENT_ORDER_DIFFERENT_ON_FROZEN_SCORES"
    )
    assert diagnostic["configurations"] == 1080
    assert diagnostic["target_cardinality_equality_rate"] == 1.0
    assert diagnostic["ordered_selection_equality_rate"] == 0.866667
    assert diagnostic["unordered_selection_equality_rate"] == 1.0
    assert diagnostic["v44_anchor_retention_rate_when_feasible"] == 1.0
    assert (
        diagnostic["v44_first_selection_equals_anchor_rate_when_feasible"] == 0.866667
    )
    assert {
        target: details["configurations"]
        for target, details in diagnostic["by_target_cardinality"].items()
    } == {"3": 861, "4": 183, "5": 36}
    assert diagnostic["v45_locked_outcome_changed"] is False
    assert diagnostic["selector_adoption_authorized"] is False
    assert diagnostic["reuse_v45_for_tuning_or_selection"] is False
    assert diagnostic["gate_2"] == "NO-GO/SHADOW"
    assert finqa["details"]["official_leaderboard_result"] is False
    assert finqa["details"]["gate_2"] == "NO-GO/SHADOW"
    consensus = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "tatqa_fetaqa_qasper_adaptive_atomic_roles_external_boundary"
    )
    assert consensus["status"] == "PASS"
    tatqa = consensus["details"]["tatqa"]
    assert tatqa == {
        "status": "TATQA_FULL_CONTEXT_POOL_INCONCLUSIVE",
        "stage_reached": "POST_ACCESS_STRUCTURAL_CENSUS_PRE_QUERY",
        "official_questions": 1668,
        "exact_mapping_eligible_questions": 0,
        "pseudo_gold_created": False,
        "metrics_computed": False,
        "selector_adoption_authorized": False,
        "reuse_v46_for_tuning_or_selection": False,
        "gate_2": "NO-GO/SHADOW",
    }
    fetaqa = consensus["details"]["fetaqa"]
    assert fetaqa["status"] == "FETAQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED"
    assert fetaqa["cases"] == 600
    assert fetaqa["generation_fallback_rate"] == 0.011667
    assert fetaqa["candidate_ceiling_complete_rate"] == 1.0
    assert fetaqa["strongest_frozen_frc_control"] == (
        "guarded_adaptive_cardinality_frc_v44"
    )
    assert fetaqa["strongest_non_frc"] == "cross_encoder_knapsack"
    assert (
        fetaqa["method_aggregates"]["consensus_guarded_adaptive_cardinality_frc_v46"][
            "supporting_cell_macro_f1"
        ]
        == 0.358926
    )
    assert (
        fetaqa["method_aggregates"]["guarded_adaptive_cardinality_frc_v44"][
            "supporting_cell_macro_f1"
        ]
        == 0.375153
    )
    assert (
        fetaqa["method_aggregates"]["cross_encoder_knapsack"][
            "supporting_cell_macro_f1"
        ]
        == 0.366367
    )
    assert fetaqa["v47_minus_strongest_frozen_frc_control"] == {
        "point": -0.016227,
        "ci_low": -0.022628,
        "ci_high": -0.010334,
    }
    assert fetaqa["v47_minus_strongest_non_frc"] == {
        "point": -0.007441,
        "ci_low": -0.025087,
        "ci_high": 0.00067,
    }
    assert fetaqa["mean_selected_unit_reduction"] == 1.843333
    assert fetaqa["supporting_cell_recall_drop"] == 0.085533
    assert fetaqa["consensus_trigger_rate"] == 0.496667
    assert fetaqa["target_cardinality_distribution"] == {
        "1": 2,
        "2": 123,
        "3": 276,
        "4": 179,
        "5": 20,
    }
    assert (
        fetaqa["support_checks"]["mean_selected_unit_reduction_at_least_0_50"] is True
    )
    assert fetaqa["support_checks"]["supporting_cell_recall_drop_at_most_0_02"] is False
    assert fetaqa["selector_adoption_authorized"] is False
    assert fetaqa["reuse_v47_for_tuning_or_selection"] is False
    assert fetaqa["official_leaderboard_result"] is False
    assert fetaqa["gate_2"] == "NO-GO/SHADOW"
    qasper = consensus["details"]["qasper"]
    assert qasper["status"] == ("QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_NOT_ESTABLISHED")
    assert qasper["cases"] == 600
    assert qasper["generation_fallback_rate"] == 0.005
    assert qasper["candidate_ceiling_complete_rate"] == 1.0
    assert qasper["strongest_frozen_frc_control"] == (
        "adaptive_argmax_cardinality_frc_v43"
    )
    assert qasper["strongest_non_frc"] == "cross_encoder_topk"
    assert (
        qasper["method_aggregates"]["top2_proposal_guarded_frc_v48"][
            "evidence_macro_f1"
        ]
        == 0.296239
    )
    assert (
        qasper["method_aggregates"]["adaptive_argmax_cardinality_frc_v43"][
            "evidence_macro_f1"
        ]
        == 0.316885
    )
    assert (
        qasper["method_aggregates"]["cross_encoder_topk"]["evidence_macro_f1"]
        == 0.247513
    )
    assert qasper["v48_minus_strongest_frozen_frc_control"] == {
        "point": -0.020645,
        "ci_low": -0.034871,
        "ci_high": -0.006266,
    }
    assert qasper["v48_minus_strongest_non_frc"] == {
        "point": 0.048726,
        "ci_low": 0.034631,
        "ci_high": 0.062649,
    }
    assert qasper["mean_selected_unit_reduction"] == 1.263333
    assert qasper["evidence_recall_drop"] == 0.043625
    assert qasper["adaptive_target_below_five_rate"] == 0.613333
    assert qasper["target_cardinality_distribution"] == {
        "2": 42,
        "3": 134,
        "4": 192,
        "5": 232,
    }
    assert (
        qasper["support_checks"]["v48_minus_strongest_non_frc_ci_low_above_0"] is True
    )
    assert (
        qasper["support_checks"]["v48_minus_strongest_frc_point_at_least_0_005"]
        is False
    )
    assert qasper["support_checks"]["evidence_recall_drop_at_most_0_01"] is False
    assert qasper["selector_adoption_authorized"] is False
    assert qasper["reuse_v48_for_tuning_or_selection"] is False
    assert qasper["official_leaderboard_result"] is False
    assert qasper["gate_2"] == "NO-GO/SHADOW"
    evidence_inference = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "evidence_inference_low_core_divergence_external_boundary"
    )
    assert evidence_inference["status"] == "PASS"
    evidence_inference_details = evidence_inference["details"]
    assert evidence_inference_details["status"] == (
        "EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED"
    )
    assert evidence_inference_details["cases"] == 600
    assert evidence_inference_details["generation_fallback_rate"] == 0.026667
    assert evidence_inference_details["candidate_ceiling_complete_rate"] == 1.0
    assert evidence_inference_details["strongest_frozen_frc_control"] == (
        "adaptive_argmax_cardinality_frc_v43"
    )
    assert evidence_inference_details["strongest_non_frc"] == "cross_encoder_topk"
    assert (
        evidence_inference_details["method_aggregates"][
            "low_core_divergence_guarded_frc_v49"
        ]["evidence_macro_f1"]
        == 0.220684
    )
    assert (
        evidence_inference_details["method_aggregates"][
            "adaptive_argmax_cardinality_frc_v43"
        ]["evidence_macro_f1"]
        == 0.223942
    )
    assert (
        evidence_inference_details["method_aggregates"]["cross_encoder_topk"][
            "evidence_macro_f1"
        ]
        == 0.186549
    )
    assert evidence_inference_details["v49_minus_strongest_frozen_frc_control"] == {
        "point": -0.003258,
        "ci_low": -0.008086,
        "ci_high": 0.001747,
    }
    assert evidence_inference_details["v49_minus_strongest_non_frc"] == {
        "point": 0.034135,
        "ci_low": 0.021521,
        "ci_high": 0.04682,
    }
    assert evidence_inference_details["mean_selected_unit_reduction"] == 1.952222
    assert evidence_inference_details["macro_recall_improvement_vs_v43"] == 0.00871
    assert evidence_inference_details["expansion_trigger_rate"] == 0.15
    assert evidence_inference_details["selection_set_difference_rate_vs_v43"] == (
        0.149444
    )
    assert evidence_inference_details["target_cardinality_distribution"] == {
        "1": 36,
        "2": 96,
        "3": 302,
        "4": 166,
    }
    assert (
        evidence_inference_details["support_checks"][
            "v49_minus_strongest_non_frc_ci_low_above_0"
        ]
        is True
    )
    assert (
        evidence_inference_details["support_checks"][
            "v49_minus_strongest_frc_ci_low_above_0"
        ]
        is False
    )
    assert (
        evidence_inference_details["support_checks"][
            "macro_recall_improvement_vs_v43_at_least_0_01"
        ]
        is False
    )
    assert evidence_inference_details["selector_adoption_authorized"] is False
    assert evidence_inference_details["reuse_v49_for_tuning_or_selection"] is False
    assert evidence_inference_details["official_leaderboard_result"] is False
    assert evidence_inference_details["eraser_result"] is False
    assert evidence_inference_details["gate_2"] == "NO-GO/SHADOW"
    contractnli = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "contractnli_native_zero_consensus_external_boundary"
    )
    assert contractnli["status"] == "PASS"
    contractnli_details = contractnli["details"]
    assert contractnli_details["status"] == (
        "CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED"
    )
    assert contractnli_details["cases"] == 600
    assert contractnli_details["label_counts"] == {
        "Entailment": 200,
        "Contradiction": 200,
        "NotMentioned": 200,
    }
    assert contractnli_details["query_fallback_rate"] == 0.0
    assert contractnli_details["strongest_shared_gate_non_frc"] == (
        "native_zero_consensus_cross_encoder_topk_v50"
    )
    assert contractnli_details["strongest_ungated_frozen_frc"] == (
        "low_core_divergence_guarded_frc_v49"
    )
    assert (
        contractnli_details["candidate_aggregate"]["evidence_or_abstention_macro_f1"]
        == 0.165173
    )
    assert (
        contractnli_details["candidate_aggregate"]["evidence_bearing_macro_f1"]
        == 0.24776
    )
    assert (
        contractnli_details["candidate_aggregate"]["not_mentioned_abstention_accuracy"]
        == 0.0
    )
    assert contractnli_details["candidate_aggregate"]["abstention_rate"] == 0.0
    assert (
        contractnli_details["family_comparison"][
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        == 0.00443
    )
    assert (
        contractnli_details["family_comparison"][
            "candidate_minus_strongest_ungated_frozen_frc"
        ]["point"]
        == -0.001423
    )
    assert (
        contractnli_details["family_comparison"]["candidate_minus_anchor_gate_frc"][
            "point"
        ]
        == 0.0
    )
    assert contractnli_details["all_cases_passed_anchor_gate"] is True
    assert contractnli_details["all_cases_passed_consensus_gate"] is True
    assert (
        contractnli_details["support_checks"]["candidate_abstention_rate_at_least_0_05"]
        is False
    )
    assert (
        contractnli_details["schema_repair_confirmation_not_completely_untouched"]
        is True
    )
    assert contractnli_details["selector_adoption_authorized"] is False
    assert contractnli_details["reuse_v50_for_tuning_or_selection"] is False
    assert contractnli_details["official_leaderboard_result"] is False
    assert contractnli_details["gate_2"] == "NO-GO/SHADOW"
    contractnli_v51 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "contractnli_dev_robust_consensus_stop_before_train_boundary"
    )
    assert contractnli_v51["status"] == "PASS"
    v51_details = contractnli_v51["details"]
    assert v51_details["status"] == (
        "CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN"
    )
    assert v51_details["cases"] == 240
    assert v51_details["documents"] == 60
    assert v51_details["label_counts"] == {
        "Entailment": 80,
        "Contradiction": 80,
        "NotMentioned": 80,
    }
    assert v51_details["query_fallback_rate"] == 0.0
    assert v51_details["candidate_ceiling_complete_rate"] == 1.0
    assert (
        v51_details["ungated_v49_aggregate"]["evidence_or_abstention_macro_f1"]
        == 0.154504
    )
    assert (
        v51_details["oof_candidate_aggregate"]["evidence_or_abstention_macro_f1"]
        == 0.191845
    )
    assert (
        v51_details["oof_candidate_aggregate"]["not_mentioned_abstention_accuracy"]
        == 0.1875
    )
    assert v51_details["oof_utility_gain_vs_ungated_v49"] == 0.037341
    assert v51_details["oof_evidence_f1_drop_vs_ungated_v49"] == 0.037738
    assert (
        v51_details["development_checks"]["oof_evidence_bearing_f1_drop_at_most_0_03"]
        is False
    )
    assert (
        v51_details["development_checks"][
            "oof_not_mentioned_abstention_accuracy_at_least_0_2"
        ]
        is False
    )
    assert v51_details["final_threshold_hex"] == "0x1.3c887b71d53bcp+3"
    assert v51_details["execution_order_erratum_disclosed"] is True
    assert v51_details["train_member_opened"] is False
    assert v51_details["train_open_authorized"] is False
    assert v51_details["selector_adoption_authorized"] is False
    assert v51_details["official_leaderboard_result"] is False
    assert v51_details["gate_2"] == "NO-GO/SHADOW"
    contractnli_v52 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "contractnli_rank_concurrence_confirmation_boundary"
    )
    assert contractnli_v52["status"] == "PASS"
    v52_details = contractnli_v52["details"]
    assert v52_details["status"] == (
        "CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED"
    )
    assert v52_details["cases"] == 600
    assert v52_details["documents"] == 319
    assert v52_details["label_counts"] == {
        "Entailment": 200,
        "Contradiction": 200,
        "NotMentioned": 200,
    }
    assert v52_details["query_fallback_rate"] == 0.0
    assert v52_details["candidate_ceiling_complete_rate"] == 1.0
    assert (
        v52_details["candidate_aggregate"]["evidence_or_abstention_macro_f1"]
        == 0.250304
    )
    assert (
        v52_details["candidate_aggregate"]["not_mentioned_abstention_accuracy"] == 0.41
    )
    assert v52_details["candidate_aggregate"]["abstention_rate"] == 0.373333
    assert v52_details["strongest_shared_gate_non_frc"] == (
        "rank_concurrence_hybrid_topk_v52"
    )
    assert v52_details["family_comparison"][
        "candidate_minus_strongest_shared_gate_non_frc"
    ] == {
        "point": 0.009038,
        "ci_low": -0.006814,
        "ci_high": 0.024925,
        "clusters": 319,
        "resamples": 10000,
        "seed": 20260812,
    }
    assert v52_details["evidence_f1_drop_vs_ungated_v49"] == 0.082325
    assert v52_details["minimum_budget_or_supported_stratum_delta"] == -0.027922
    assert (
        v52_details["support_checks"][
            "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0"
        ]
        is False
    )
    assert (
        v52_details["support_checks"][
            "evidence_bearing_macro_f1_drop_vs_ungated_v49_at_most_0_03"
        ]
        is False
    )
    assert v52_details["parameter_free"] is True
    assert v52_details["v50_test_or_v51_dev_reused"] is False
    assert v52_details["confirmation_train_reuse_for_tuning_or_selection"] is False
    assert v52_details["selector_adoption_authorized"] is False
    assert v52_details["official_leaderboard_result"] is False
    assert v52_details["gate_2"] == "NO-GO/SHADOW"
    cuad_v53 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "cuad_top3_rank_concurrence_role_closure_boundary"
    )
    assert cuad_v53["status"] == "PASS"
    v53_details = cuad_v53["details"]
    assert v53_details["status"] == (
        "CUAD_V53_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TEST"
    )
    assert v53_details["stage"] == "development"
    assert v53_details["cases"] == 400
    assert v53_details["contracts"] == 259
    assert v53_details["answer_state_counts"] == {
        "answer_bearing": 200,
        "no_answer": 200,
    }
    assert v53_details["query_fallback_rate"] == 0.0
    assert v53_details["score_fallback_rate"] == 0.0
    assert v53_details["candidate_ceiling_complete_rate"] == 0.635
    assert v53_details["top3_gate_passed_cases"] == 397
    assert (
        v53_details["candidate_aggregate"]["answer_or_abstention_macro_f1"] == 0.046032
    )
    assert v53_details["candidate_aggregate"]["no_answer_abstention_accuracy"] == 0.005
    assert v53_details["candidate_aggregate"]["abstention_rate"] == 0.0075
    assert v53_details["strongest_shared_gate_non_frc"] == (
        "top3_rank_concurrence_hybrid_topk_v53"
    )
    assert v53_details["family_comparison"][
        "candidate_minus_strongest_shared_gate_non_frc"
    ] == {
        "point": -0.023731,
        "ci_low": -0.040711,
        "ci_high": -0.006903,
        "clusters": 259,
        "resamples": 10000,
        "seed": 20260813,
    }
    assert v53_details["answer_recall_drop_vs_ungated_v49"] == -0.02
    assert v53_details["minimum_budget_or_supported_stratum_delta"] == -0.050796
    assert (
        v53_details["support_checks"]["candidate_ceiling_complete_rate_at_least_0_9"]
        is False
    )
    assert (
        v53_details["support_checks"]["no_answer_abstention_accuracy_at_least_0_2"]
        is False
    )
    assert v53_details["train_reuse_for_tuning_or_selection"] is False
    assert v53_details["test_content_opened"] is False
    assert v53_details["test_open_authorized"] is False
    assert v53_details["post_result_formatting_erratum_disclosed"] is True
    assert v53_details["official_leaderboard_result"] is False
    assert v53_details["selector_adoption_authorized"] is False
    assert v53_details["gate_2"] == "NO-GO/SHADOW"
    doc2dial_v54 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "doc2dial_v54_source_quota_closure_boundary"
    )
    assert doc2dial_v54["status"] == "PASS"
    assert doc2dial_v54["details"]["status"] == (
        "DOC2DIAL_V54_SCHEMA_INCONCLUSIVE_STOP"
    )
    assert doc2dial_v54["details"]["answer_state_counts"] == {
        "answer_bearing": 20431,
        "no_answer": 0,
    }
    assert doc2dial_v54["details"]["confirmation_opened"] is False
    assert doc2dial_v54["details"]["neural_scoring_started"] is False
    assert doc2dial_v54["details"]["gate_2"] == "NO-GO/SHADOW"
    doc2dial_v55 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "doc2dial_wood_v55_schema_closure_boundary"
    )
    assert doc2dial_v55["status"] == "PASS"
    assert doc2dial_v55["details"]["status"] == (
        "DOC2DIAL_WOOD_V55_SCHEMA_INCONCLUSIVE_STOP"
    )
    assert doc2dial_v55["details"]["turn_container_shapes"] == {
        "list": 3459,
        "dict": 12,
    }
    assert doc2dial_v55["details"]["schema_exclusion_rate"] == 0.018618
    assert doc2dial_v55["details"]["confirmation_opened"] is False
    assert doc2dial_v55["details"]["woood_sibling_opened"] is False
    assert doc2dial_v55["details"]["gate_2"] == "NO-GO/SHADOW"
    doc2dial_v56 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "doc2dial_wood_v56_schema_corrected_transfer_boundary"
    )
    assert doc2dial_v56["status"] == "PASS"
    v56_details = doc2dial_v56["details"]
    assert v56_details["status"] == (
        "DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v56_details["cases"] == 400
    assert v56_details["documents"] == 261
    assert v56_details["dialogues"] == 386
    assert v56_details["answer_state_counts"] == {
        "answer_bearing": 200,
        "no_answer": 200,
    }
    assert v56_details["no_answer_subtypes"] == {
        "ood_act": 176,
        "other_empty": 24,
    }
    assert v56_details["candidate_ceiling_complete_rate"] == 0.805
    assert v56_details["candidate_aggregate"]["answer_or_abstention_macro_f1"] == (
        0.517417
    )
    assert v56_details["candidate_aggregate"]["no_answer_abstention_accuracy"] == (
        0.875
    )
    assert v56_details["candidate_aggregate"]["abstention_rate"] == 0.6125
    assert (
        v56_details["family_comparison"][
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        == -0.008827
    )
    assert v56_details["minimum_budget_or_supported_stratum_delta"] == -0.038932
    assert v56_details["pre_outcome_schema_access_disclosed"] is True
    assert v56_details["confirmation_opened"] is False
    assert v56_details["confirmation_open_authorized"] is False
    assert v56_details["selector_adoption_authorized"] is False
    assert v56_details["gate_2"] == "NO-GO/SHADOW"
    quac_v57 = next(
        item
        for item in report["requirements"]
        if item["requirement_id"] == "quac_v57_anchor_safe_consensus_slot_boundary"
    )
    assert quac_v57["status"] == "PASS"
    v57_details = quac_v57["details"]
    assert v57_details["status"] == (
        "QUAC_V57_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION"
    )
    assert v57_details["cases"] == 600
    assert v57_details["documents"] == 554
    assert v57_details["dialogues"] == 578
    assert v57_details["answer_state_counts"] == {
        "answer_bearing": 300,
        "no_answer": 300,
    }
    assert v57_details["candidate_ceiling_complete_rate"] == 1.0
    assert v57_details["single_slot_trigger_rate"] == 0.085
    assert (
        v57_details["candidate_aggregate"]["answer_or_abstention_macro_f1"] == 0.114109
    )
    assert (
        v57_details["candidate_aggregate"]["no_answer_abstention_accuracy"] == 0.066667
    )
    assert v57_details["candidate_aggregate"]["abstention_rate"] == 0.075
    assert (
        v57_details["family_comparison"]["candidate_minus_exact_anchor_ablation"][
            "point"
        ]
        == 0.000317
    )
    assert (
        v57_details["family_comparison"]["candidate_minus_strongest_same_gate_frc"][
            "point"
        ]
        == 0.014943
    )
    assert v57_details["minimum_budget_or_supported_stratum_delta"] == -0.012086
    assert v57_details["validation_opened"] is False
    assert v57_details["validation_open_authorized"] is False
    assert v57_details["selector_adoption_authorized"] is False
    assert v57_details["gate_2"] == "NO-GO/SHADOW"

    squad2_expectations = {
        58: {
            "slug": "generative_answerability_gate",
            "articles": 299,
            "paragraphs": 590,
            "balanced_accuracy": 0.71,
            "answer_rate_key": "answer_bearing_pass_rate",
            "answer_rate": 0.56,
            "no_answer_rate": 0.86,
            "invalid_key": "malformed_or_fallback_rate",
            "invalid_rate": 0.0,
            "utility": 0.672593,
            "answer_f1": 0.485185,
            "answer_recall": 0.516667,
            "non_frc_delta": 0.107759,
            "same_gate_frc_delta": -0.004444,
        },
        59: {
            "slug": "extractive_support_gate",
            "articles": 309,
            "paragraphs": 592,
            "balanced_accuracy": 0.663333,
            "answer_rate_key": "answer_bearing_span_pass_rate",
            "answer_rate": 0.426667,
            "no_answer_rate": 0.9,
            "invalid_key": "invalid_output_rate",
            "invalid_rate": 0.503333,
            "utility": 0.631667,
            "answer_f1": 0.363333,
            "answer_recall": 0.386667,
            "non_frc_delta": 0.08023,
            "same_gate_frc_delta": 0.001389,
        },
        60: {
            "slug": "structured_span_gate",
            "articles": 317,
            "paragraphs": 584,
            "balanced_accuracy": 0.736667,
            "answer_rate_key": "answer_bearing_span_pass_rate",
            "answer_rate": 0.693333,
            "no_answer_rate": 0.78,
            "invalid_key": "invalid_output_rate",
            "invalid_rate": 0.016667,
            "utility": 0.691019,
            "answer_f1": 0.602037,
            "answer_recall": 0.633889,
            "non_frc_delta": 0.140103,
            "same_gate_frc_delta": 0.005,
        },
        61: {
            "slug": "dual_support_union",
            "articles": 309,
            "paragraphs": 590,
            "balanced_accuracy": 0.748333,
            "answer_rate_key": "answer_bearing_span_pass_rate",
            "answer_rate": 0.723333,
            "no_answer_rate": 0.773333,
            "invalid_key": "invalid_output_rate",
            "invalid_rate": 0.011667,
            "utility": 0.701389,
            "answer_f1": 0.629444,
            "answer_recall": 0.655,
            "non_frc_delta": 0.143495,
            "same_gate_frc_delta": 0.007778,
        },
    }
    for version, expected in squad2_expectations.items():
        item = next(
            requirement
            for requirement in report["requirements"]
            if requirement["requirement_id"]
            == f"squad2_v{version}_{expected['slug']}_boundary"
        )
        assert item["status"] == "PASS"
        details = item["details"]
        assert details["cases"] == 600
        assert details["articles"] == expected["articles"]
        assert details["paragraphs"] == expected["paragraphs"]
        assert details["answer_state_counts"] == {
            "answer_bearing": 300,
            "no_answer": 300,
        }
        verifier = details["support_verifier"]
        assert verifier["balanced_accuracy"] == expected["balanced_accuracy"]
        assert verifier[expected["answer_rate_key"]] == expected["answer_rate"]
        assert verifier["no_answer_rejection_rate"] == expected["no_answer_rate"]
        assert verifier[expected["invalid_key"]] == expected["invalid_rate"]
        aggregate = details["candidate_aggregate"]
        assert aggregate["answer_or_abstention_macro_f1"] == expected["utility"]
        assert aggregate["answer_bearing_macro_f1"] == expected["answer_f1"]
        assert aggregate["answer_macro_recall"] == expected["answer_recall"]
        comparison = details["family_comparison"]
        assert (
            comparison["candidate_minus_strongest_shared_gate_non_frc"]["point"]
            == expected["non_frc_delta"]
        )
        assert (
            comparison["candidate_minus_strongest_same_gate_frc"]["point"]
            == expected["same_gate_frc_delta"]
        )
        assert details["confirmation_opened"] is False
        assert details["confirmation_open_authorized"] is False
        assert details["selector_adoption_authorized"] is False
        assert details["gate_2"] == "NO-GO/SHADOW"

    quac_v62 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "quac_v62_roberta_qa_support_transfer_boundary"
    )
    assert quac_v62["status"] == "PASS"
    v62_details = quac_v62["details"]
    assert v62_details["cases"] == 600
    assert v62_details["dialogues"] == 587
    assert v62_details["documents"] == 566
    assert v62_details["answer_state_counts"] == {
        "answer_bearing": 300,
        "no_answer": 300,
    }
    assert v62_details["selected_v57_commitment_overlap"] == 0
    assert v62_details["support_verifier"]["balanced_accuracy"] == 0.59
    assert (
        v62_details["support_verifier"]["answer_bearing_support_pass_rate"] == 0.686667
    )
    assert v62_details["support_verifier"]["no_answer_rejection_rate"] == 0.493333
    assert (
        v62_details["candidate_aggregate"]["answer_or_abstention_macro_f1"] == 0.308444
    )
    assert v62_details["candidate_aggregate"]["answer_macro_recall"] == 0.139722
    assert (
        v62_details["family_comparison"][
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        == -0.013077
    )
    assert (
        v62_details["family_comparison"]["candidate_minus_strongest_same_gate_frc"][
            "point"
        ]
        == -0.013272
    )
    assert v62_details["validation_opened"] is False
    assert v62_details["validation_open_authorized"] is False
    assert v62_details["selector_adoption_authorized"] is False
    assert v62_details["gate_2"] == "NO-GO/SHADOW"

    quac_v63 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "quac_v63_target_trained_qa_support_boundary"
    )
    assert quac_v63["status"] == "PASS"
    v63_details = quac_v63["details"]
    assert v63_details["cases"] == 600
    assert v63_details["dialogues"] == 477
    assert v63_details["documents"] == 477
    assert v63_details["answer_state_counts"] == {
        "answer_bearing": 300,
        "no_answer": 300,
    }
    assert v63_details["selected_v57_commitment_overlap"] == 0
    assert v63_details["support_verifier"] == {
        "rows": 600,
        "gold_fields_visible_to_verifier": False,
        "answer_bearing_support_pass_rate": 0.486667,
        "no_answer_rejection_rate": 0.5,
        "balanced_accuracy": 0.493333,
        "invalid_output_count": 0,
        "invalid_output_rate": 0.0,
    }
    assert v63_details["query_generation_started"] is False
    assert v63_details["retrieval_scoring_started"] is False
    assert v63_details["retrieval_scoring_open_authorized"] is False
    assert v63_details["strict_independent_confirmation_claimed"] is False
    assert v63_details["validation_reuse_for_tuning_or_selection"] is False
    assert v63_details["selector_adoption_authorized"] is False
    assert v63_details["gate_2"] == "NO-GO/SHADOW"

    squad2_v64 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "squad2_v64_calibrated_roberta_component_feasibility"
    )
    assert squad2_v64["status"] == "PASS"
    v64_details = squad2_v64["details"]
    assert v64_details["calibration_cases"] == 2000
    assert v64_details["calibration_articles"] == 426
    assert v64_details["calibration_oof_metrics"]["balanced_accuracy"] == 0.944
    assert v64_details["locked_threshold_exact"] == 0.974609375
    assert v64_details["confirmation_cases"] == 600
    assert v64_details["confirmation_articles"] == 35
    assert v64_details["confirmation_support_verifier"]["balanced_accuracy"] == (
        0.873333
    )
    assert v64_details["candidate_aggregate"]["answer_or_abstention_macro_f1"] == (
        0.807944
    )
    assert (
        v64_details["family_comparison"][
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        == 0.171069
    )
    assert v64_details["self_domain_held_out_component_feasibility_only"] is True
    assert v64_details["independent_model_training_confirmation_claimed"] is False
    assert v64_details["selector_adoption_authorized"] is False
    assert v64_details["canary_or_default_authorized"] is False
    assert v64_details["gate_2"] == "NO-GO/SHADOW"

    musique_v65 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "musique_full_v65_roberta_transfer_boundary"
    )
    assert musique_v65["status"] == "PASS"
    v65_details = musique_v65["details"]
    assert v65_details["status"] == (
        "MUSIQUE_FULL_V65_ROBERTA_TRANSFER_SUPPORT_NOT_ESTABLISHED_"
        "STOP_BEFORE_RETRIEVAL"
    )
    assert v65_details["invalid_run_status"] == (
        "MUSIQUE_FULL_V65_PROCEDURAL_INVALID_DUPLICATE_SOURCE_ID_STOP_BEFORE_METRICS"
    )
    assert v65_details["invalid_run_gold_or_metrics_computed"] is False
    assert v65_details["invalid_run_unique_source_commitments"] == 300
    assert v65_details["corrected_cases"] == 600
    assert v65_details["corrected_unique_case_ids"] == 600
    assert v65_details["corrected_invalid_run_source_overlap"] == 0
    assert v65_details["corrected_squad2_exact_question_overlap"] == 0
    assert v65_details["corrected_v36_source_id_overlap"] == 0
    assert v65_details["support_verifier"] == {
        "answer_bearing_support_pass_rate": 0.63,
        "balanced_accuracy": 0.563333,
        "gold_fields_visible_to_verifier": False,
        "invalid_output_count": 0,
        "invalid_output_rate": 0.0,
        "locked_threshold": 0.974609,
        "no_answer_rejection_rate": 0.496667,
        "rows": 600,
    }
    assert v65_details["local_span_trap"] == {
        "cases": 69,
        "unanswerable_rejection_rate": 0.449275,
    }
    assert v65_details["query_generation_started"] is False
    assert v65_details["neural_retrieval_scoring_started"] is False
    assert (
        v65_details["strict_independent_model_training_confirmation_claimed"] is False
    )
    assert v65_details["reuse_v65_for_tuning_threshold_or_selection"] is False
    assert v65_details["selector_adoption_authorized"] is False
    assert v65_details["canary_or_default_authorized"] is False
    assert v65_details["gate_2"] == "NO-GO/SHADOW"

    musique_v66 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v66_sequential_chain_support_boundary"
    )
    assert musique_v66["status"] == "PASS"
    v66_details = musique_v66["details"]
    assert v66_details["status"] == (
        "MUSIQUE_V66_SEQUENTIAL_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_"
        "STOP_BEFORE_CONFIRMATION"
    )
    assert v66_details["cases"] == 600
    assert v66_details["answer_state_counts"] == {
        "answerable": 300,
        "unanswerable": 300,
    }
    assert v66_details["v65_source_commitment_union"] == 900
    assert v66_details["selected_v65_source_overlap"] == 0
    assert v66_details["selected_squad2_exact_question_overlap"] == 0
    assert v66_details["direct_composed_question"]["balanced_accuracy"] == 0.568333
    assert v66_details["oracle_plan_sequential_chain"] == {
        "rows": 600,
        "answerable_rows": 300,
        "unanswerable_rows": 300,
        "answerable_support_pass_rate": 0.39,
        "unanswerable_rejection_rate": 0.816667,
        "balanced_accuracy": 0.603333,
    }
    assert v66_details["paired_correctness_delta"] == {
        "point": 0.035,
        "ci_low": -0.015,
        "ci_high": 0.085,
        "resamples": 10000,
        "seed": 20260820,
    }
    assert v66_details["direct_passed_unanswerable_traps"] == {
        "cases": 142,
        "sequential_rejection_rate": 0.697183,
    }
    assert v66_details["confirmation_opened"] is False
    assert v66_details["automatic_decomposer_result"] is False
    assert (
        v66_details["strict_independent_model_training_confirmation_claimed"] is False
    )
    assert (
        v66_details["reuse_failed_stage_for_tuning_threshold_rule_gate_or_selection"]
        is False
    )
    assert v66_details["selector_adoption_authorized"] is False
    assert v66_details["canary_or_default_authorized"] is False
    assert v66_details["gate_2"] == "NO-GO/SHADOW"

    musique_v67 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v67_calibrated_chain_support_boundary"
    )
    assert musique_v67["status"] == "PASS"
    v67_details = musique_v67["details"]
    assert v67_details["status"] == (
        "MUSIQUE_V67_CALIBRATED_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_"
        "STOP_BEFORE_CONFIRMATION"
    )
    assert v67_details["prior_source_commitment_union"] == 1500
    assert v67_details["calibration_cases"] == 1000
    assert v67_details["development_cases"] == 600
    assert v67_details["calibration_thresholds"] == {
        "direct": 1.216796875,
        "chain_bottleneck": -1.0703125,
    }
    assert v67_details["calibration_crossfit"]["direct"]["balanced_accuracy"] == 0.593
    assert (
        v67_details["calibration_crossfit"]["chain_bottleneck"]["balanced_accuracy"]
        == 0.662
    )
    assert (
        v67_details["development_methods"]["calibrated_direct"]["balanced_accuracy"]
        == 0.568333
    )
    assert v67_details["development_methods"]["fixed_v66_full_chain"] == {
        "rows": 600,
        "answerable_rows": 300,
        "unanswerable_rows": 300,
        "threshold_exact": 0.974609375,
        "answerable_pass_rate": 0.393333,
        "unanswerable_rejection_rate": 0.836667,
        "balanced_accuracy": 0.615,
    }
    assert v67_details["development_methods"]["calibrated_chain_bottleneck"] == {
        "rows": 600,
        "answerable_rows": 300,
        "unanswerable_rows": 300,
        "threshold_exact": -1.0703125,
        "answerable_pass_rate": 0.553333,
        "unanswerable_rejection_rate": 0.7,
        "balanced_accuracy": 0.626667,
    }
    assert v67_details["paired_correctness_delta"] == {
        "point": 0.011667,
        "ci_low": -0.018333,
        "ci_high": 0.041667,
        "resamples": 10000,
        "seed": 20260823,
    }
    assert v67_details["unanswerable_hop_strata"] == {
        "2": {"cases": 240, "rejection_rate": 0.654167},
        "3": {"cases": 48, "rejection_rate": 0.854167},
        "4": {"cases": 12, "rejection_rate": 1.0},
    }
    assert v67_details["confirmation_opened"] is False
    assert v67_details["automatic_decomposer_result"] is False
    assert (
        v67_details["reuse_failed_stage_for_tuning_threshold_score_gate_or_selection"]
        is False
    )
    assert v67_details["selector_adoption_authorized"] is False
    assert v67_details["canary_or_default_authorized"] is False
    assert v67_details["gate_2"] == "NO-GO/SHADOW"

    musique_v68 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v68_multisignal_chain_support_boundary"
    )
    assert musique_v68["status"] == "PASS"
    v68_details = musique_v68["details"]
    assert v68_details["status"] == (
        "MUSIQUE_V68_MULTISIGNAL_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_"
        "STOP_BEFORE_CONFIRMATION"
    )
    assert v68_details["prior_source_commitment_union"] == 3100
    assert v68_details["calibration_cases"] == 1200
    assert v68_details["development_cases"] == 600
    assert v68_details["total_learned_scalars"] == 17
    assert (
        v68_details["calibration_crossfit"]["multisignal_logistic_candidate"][
            "balanced_accuracy"
        ]
        == 0.691667
    )
    assert v68_details["development_methods"]["multisignal_logistic_candidate"] == {
        "rows": 600,
        "answerable_rows": 300,
        "unanswerable_rows": 300,
        "threshold_exact": 0.5476630638951829,
        "answerable_pass_rate": 0.573333,
        "unanswerable_rejection_rate": 0.776667,
        "balanced_accuracy": 0.675,
    }
    assert v68_details["strongest_fair_baseline"]["name"] == (
        "calibrated_chain_bottleneck"
    )
    assert v68_details["paired_correctness_delta"] == {
        "point": 0.036667,
        "ci_low": 0.01,
        "ci_high": 0.063333,
        "resamples": 10000,
        "seed": 20260826,
    }
    assert v68_details["unanswerable_hop_strata"] == {
        "2": {"cases": 222, "rejection_rate": 0.761261},
        "3": {"cases": 57, "rejection_rate": 0.807018},
        "4": {"cases": 21, "rejection_rate": 0.857143},
    }
    assert v68_details["support_checks"]["paired_correctness_ci_low_above_0"] is True
    assert (
        v68_details["support_checks"][
            "multisignal_minus_strongest_fair_baseline_at_least_0_05"
        ]
        is False
    )
    assert v68_details["confirmation_opened"] is False
    assert v68_details["automatic_decomposer_result"] is False
    assert (
        v68_details[
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v68_details["selector_adoption_authorized"] is False
    assert v68_details["canary_or_default_authorized"] is False
    assert v68_details["gate_2"] == "NO-GO/SHADOW"

    musique_v69 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v69_monotone_interaction_support_boundary"
    )
    assert musique_v69["status"] == "PASS"
    v69_details = musique_v69["details"]
    assert v69_details["status"] == (
        "MUSIQUE_V69_MONOTONE_INTERACTION_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v69_details["prior_source_commitment_union"] == 4900
    assert v69_details["calibration_cases"] == 1600
    assert v69_details["development_cases"] == 800
    assert v69_details["base_feature_count"] == 9
    assert v69_details["hinge_feature_count"] == 18
    assert v69_details["interaction_feature_count"] == 5
    assert v69_details["candidate_feature_count"] == 32
    assert v69_details["total_learned_scalars"] == 76
    assert v69_details["calibration_invalid_fail_closed_count"] == 1
    assert v69_details["implementation_erratum_disclosed"] is True
    assert (
        v69_details["calibration_crossfit"]["monotone_interaction_candidate"][
            "balanced_accuracy"
        ]
        == 0.704375
    )
    assert v69_details["development_methods"]["monotone_interaction_candidate"] == {
        "rows": 800,
        "answerable_rows": 400,
        "unanswerable_rows": 400,
        "threshold_exact": 0.4970355610538546,
        "answerable_pass_rate": 0.62,
        "unanswerable_rejection_rate": 0.755,
        "balanced_accuracy": 0.6875,
    }
    assert v69_details["strongest_fair_baseline"]["name"] == (
        "monotone_additive_spline_control"
    )
    assert v69_details["paired_correctness_delta"] == {
        "point": -0.0025,
        "ci_low": -0.01,
        "ci_high": 0.005,
        "resamples": 10000,
        "seed": 20260829,
    }
    assert v69_details["interaction_mechanism_delta_vs_additive"] == {
        "point": -0.0025,
        "ci_low": -0.01,
        "ci_high": 0.005,
        "resamples": 10000,
        "seed": 20261829,
    }
    assert v69_details["unanswerable_hop_strata"] == {
        "2": {"cases": 302, "rejection_rate": 0.741722},
        "3": {"cases": 80, "rejection_rate": 0.7625},
        "4": {"cases": 18, "rejection_rate": 0.944444},
    }
    assert (
        v69_details["support_checks"]["candidate_answerable_pass_rate_at_least_0_60"]
        is True
    )
    assert (
        v69_details["support_checks"]["candidate_minus_additive_control_at_least_0_02"]
        is False
    )
    assert v69_details["confirmation_opened"] is False
    assert v69_details["automatic_decomposer_result"] is False
    assert (
        v69_details[
            "reuse_failed_stage_for_basis_interaction_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v69_details["selector_adoption_authorized"] is False
    assert v69_details["canary_or_default_authorized"] is False
    assert v69_details["gate_2"] == "NO-GO/SHADOW"

    musique_v70 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v70_paragraph_competition_support_boundary"
    )
    assert musique_v70["status"] == "PASS"
    v70_details = musique_v70["details"]
    assert v70_details["status"] == (
        "MUSIQUE_V70_PARAGRAPH_COMPETITION_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v70_details["prior_source_commitment_union"] == 7300
    assert v70_details["calibration_cases"] == 1600
    assert v70_details["development_cases"] == 800
    assert v70_details["base_feature_count"] == 9
    assert v70_details["prior_feature_count"] == 32
    assert v70_details["competition_feature_count"] == 6
    assert v70_details["candidate_feature_count"] == 15
    assert v70_details["audited_feature_count"] == 38
    assert v70_details["total_learned_scalars"] == 101
    assert v70_details["calibration_invalid_fail_closed_count"] == 0
    assert v70_details["development_invalid_fail_closed_count"] == 2
    assert v70_details["implementation_erratum_disclosed"] is True
    assert (
        v70_details["calibration_crossfit"]["paragraph_competition_candidate"][
            "balanced_accuracy"
        ]
        == 0.68375
    )
    assert v70_details["development_methods"]["paragraph_competition_candidate"] == {
        "rows": 800,
        "answerable_rows": 400,
        "unanswerable_rows": 400,
        "threshold_exact": 0.5384328444913514,
        "answerable_pass_rate": 0.5725,
        "unanswerable_rejection_rate": 0.7575,
        "balanced_accuracy": 0.665,
    }
    assert v70_details["strongest_fair_baseline"]["name"] == (
        "monotone_interaction_control"
    )
    assert v70_details["strongest_prior_feature_control"]["name"] == (
        "monotone_interaction_control"
    )
    assert v70_details["paired_correctness_delta"] == {
        "point": -0.02625,
        "ci_low": -0.04875,
        "ci_high": -0.00375,
        "resamples": 10000,
        "seed": 20260831,
    }
    assert v70_details["incremental_delta_vs_strongest_prior_feature_control"] == {
        "point": -0.02625,
        "ci_low": -0.04875,
        "ci_high": -0.004969,
        "resamples": 10000,
        "seed": 20261831,
    }
    assert v70_details["incremental_delta_vs_competition_only_control"] == {
        "point": 0.09625,
        "ci_low": 0.06,
        "ci_high": 0.1325,
        "resamples": 10000,
        "seed": 20262831,
    }
    assert v70_details["unanswerable_hop_strata"] == {
        "2": {"cases": 303, "rejection_rate": 0.742574},
        "3": {"cases": 76, "rejection_rate": 0.789474},
        "4": {"cases": 21, "rejection_rate": 0.857143},
    }
    assert (
        v70_details["support_checks"][
            "candidate_minus_competition_only_control_at_least_0_02"
        ]
        is True
    )
    assert (
        v70_details["support_checks"][
            "candidate_minus_strongest_prior_feature_control_at_least_0_02"
        ]
        is False
    )
    assert v70_details["confirmation_opened"] is False
    assert v70_details["automatic_decomposer_result"] is False
    assert (
        v70_details[
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v70_details["selector_adoption_authorized"] is False
    assert v70_details["canary_or_default_authorized"] is False
    assert v70_details["gate_2"] == "NO-GO/SHADOW"

    musique_v71 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v71_bridge_counterfactual_dependence_boundary"
    )
    assert musique_v71["status"] == "PASS"
    v71_details = musique_v71["details"]
    assert v71_details["status"] == (
        "MUSIQUE_V71_BRIDGE_COUNTERFACTUAL_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v71_details["prior_source_commitment_union"] == 9700
    assert v71_details["calibration_cases"] == 1600
    assert v71_details["development_cases"] == 800
    assert v71_details["base_feature_count"] == 9
    assert v71_details["prior_feature_count"] == 32
    assert v71_details["competition_feature_count"] == 6
    assert v71_details["counterfactual_feature_count"] == 6
    assert v71_details["candidate_feature_count"] == 15
    assert v71_details["audited_feature_count"] == 44
    assert v71_details["total_learned_scalars"] == 118
    assert v71_details["calibration_invalid_fail_closed_count"] == 0
    assert v71_details["development_invalid_fail_closed_count"] == 0
    assert v71_details["implementation_erratum_disclosed"] is True
    assert (
        v71_details["calibration_crossfit"]["bridge_counterfactual_candidate"][
            "balanced_accuracy"
        ]
        == 0.689375
    )
    assert v71_details["development_methods"]["bridge_counterfactual_candidate"] == {
        "rows": 800,
        "answerable_rows": 400,
        "unanswerable_rows": 400,
        "threshold_exact": 0.546911023516025,
        "answerable_pass_rate": 0.6225,
        "unanswerable_rejection_rate": 0.79,
        "balanced_accuracy": 0.70625,
    }
    assert v71_details["strongest_fair_baseline"]["name"] == (
        "monotone_additive_spline_control"
    )
    assert v71_details["strongest_prior_feature_control"]["name"] == (
        "monotone_additive_spline_control"
    )
    assert v71_details["paired_correctness_delta"] == {
        "point": -0.01625,
        "ci_low": -0.03375,
        "ci_high": 0.0025,
        "resamples": 10000,
        "seed": 20260903,
    }
    assert v71_details["incremental_delta_vs_strongest_prior_feature_control"] == {
        "point": -0.01625,
        "ci_low": -0.035,
        "ci_high": 0.0025,
        "resamples": 10000,
        "seed": 20261903,
    }
    assert v71_details["incremental_delta_vs_counterfactual_only_control"] == {
        "point": 0.09125,
        "ci_low": 0.05625,
        "ci_high": 0.12625,
        "resamples": 10000,
        "seed": 20262903,
    }
    assert v71_details["unanswerable_hop_strata"] == {
        "2": {"cases": 304, "rejection_rate": 0.769737},
        "3": {"cases": 76, "rejection_rate": 0.828947},
        "4": {"cases": 20, "rejection_rate": 0.95},
    }
    assert (
        v71_details["support_checks"][
            "candidate_minus_counterfactual_only_control_at_least_0_02"
        ]
        is True
    )
    assert (
        v71_details["support_checks"][
            "candidate_minus_strongest_prior_feature_control_at_least_0_02"
        ]
        is False
    )
    assert v71_details["confirmation_opened"] is False
    assert v71_details["automatic_decomposer_result"] is False
    assert (
        v71_details[
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v71_details["selector_adoption_authorized"] is False
    assert v71_details["canary_or_default_authorized"] is False
    assert v71_details["gate_2"] == "NO-GO/SHADOW"

    musique_v72 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v72_in_domain_rival_bridge_boundary"
    )
    assert musique_v72["status"] == "PASS"
    v72_details = musique_v72["details"]
    assert v72_details["status"] == (
        "MUSIQUE_V72_IN_DOMAIN_RIVAL_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v72_details["prior_source_commitment_union"] == 12100
    assert v72_details["calibration_cases"] == 1600
    assert v72_details["development_cases"] == 800
    assert v72_details["base_feature_count"] == 9
    assert v72_details["prior_feature_count"] == 32
    assert v72_details["competition_feature_count"] == 6
    assert v72_details["sentinel_feature_count"] == 6
    assert v72_details["rival_feature_count"] == 6
    assert v72_details["candidate_feature_count"] == 15
    assert v72_details["audited_feature_count"] == 50
    assert v72_details["total_learned_scalars"] == 135
    assert v72_details["calibration_invalid_fail_closed_count"] == 0
    assert v72_details["development_invalid_fail_closed_count"] == 1
    assert v72_details["implementation_erratum_disclosed"] is True
    assert (
        v72_details["calibration_crossfit"]["in_domain_rival_candidate"][
            "balanced_accuracy"
        ]
        == 0.6875
    )
    assert v72_details["development_methods"]["in_domain_rival_candidate"] == {
        "rows": 800,
        "answerable_rows": 400,
        "unanswerable_rows": 400,
        "threshold_exact": 0.5324149828721157,
        "answerable_pass_rate": 0.6175,
        "unanswerable_rejection_rate": 0.795,
        "balanced_accuracy": 0.70625,
    }
    assert v72_details["strongest_fair_baseline"]["name"] == (
        "linear_nine_signal_control"
    )
    assert v72_details["strongest_prior_feature_control"]["name"] == (
        "linear_nine_signal_control"
    )
    assert v72_details["paired_correctness_delta"] == {
        "point": -0.00125,
        "ci_low": -0.015,
        "ci_high": 0.0125,
        "resamples": 10000,
        "seed": 20260906,
    }
    assert v72_details["incremental_delta_vs_strongest_prior_feature_control"] == {
        "point": -0.00125,
        "ci_low": -0.015,
        "ci_high": 0.0125,
        "resamples": 10000,
        "seed": 20261906,
    }
    assert v72_details["incremental_delta_vs_rival_only_control"] == {
        "point": 0.1125,
        "ci_low": 0.08,
        "ci_high": 0.14625,
        "resamples": 10000,
        "seed": 20262906,
    }
    assert v72_details["unanswerable_hop_strata"] == {
        "2": {"cases": 313, "rejection_rate": 0.763578},
        "3": {"cases": 77, "rejection_rate": 0.896104},
        "4": {"cases": 10, "rejection_rate": 1.0},
    }
    assert (
        v72_details["support_checks"][
            "candidate_minus_rival_only_control_at_least_0_02"
        ]
        is True
    )
    assert (
        v72_details["support_checks"][
            "candidate_minus_strongest_prior_feature_control_at_least_0_02"
        ]
        is False
    )
    assert (
        v72_details["support_checks"]["invalid_feature_output_rate_equals_0"] is False
    )
    assert v72_details["confirmation_opened"] is False
    assert v72_details["automatic_decomposer_result"] is False
    assert (
        v72_details[
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v72_details["selector_adoption_authorized"] is False
    assert v72_details["canary_or_default_authorized"] is False
    assert v72_details["gate_2"] == "NO-GO/SHADOW"

    musique_v73 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v73_context_bridge_erasure_boundary"
    )
    assert musique_v73["status"] == "PASS"
    v73_details = musique_v73["details"]
    assert v73_details["status"] == (
        "MUSIQUE_V73_CONTEXT_BRIDGE_ERASURE_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v73_details["prior_source_commitment_union"] == 14500
    assert v73_details["calibration_cases"] == 1600
    assert v73_details["development_cases"] == 800
    assert v73_details["base_feature_count"] == 9
    assert v73_details["prior_feature_count"] == 32
    assert v73_details["competition_feature_count"] == 6
    assert v73_details["sentinel_feature_count"] == 6
    assert v73_details["rival_feature_count"] == 6
    assert v73_details["erasure_feature_count"] == 6
    assert v73_details["candidate_feature_count"] == 15
    assert v73_details["audited_feature_count"] == 56
    assert v73_details["total_learned_scalars"] == 152
    assert v73_details["calibration_invalid_fail_closed_count"] == 3
    assert v73_details["development_invalid_fail_closed_count"] == 0
    assert v73_details["implementation_erratum_disclosed"] is True
    assert (
        v73_details["calibration_crossfit"]["context_erasure_candidate"][
            "balanced_accuracy"
        ]
        == 0.71625
    )
    assert v73_details["development_methods"]["context_erasure_candidate"] == {
        "rows": 800,
        "answerable_rows": 400,
        "unanswerable_rows": 400,
        "threshold_exact": 0.5592693586391849,
        "answerable_pass_rate": 0.5875,
        "unanswerable_rejection_rate": 0.8175,
        "balanced_accuracy": 0.7025,
    }
    assert v73_details["strongest_fair_baseline"]["name"] == ("in_domain_rival_control")
    assert v73_details["strongest_prior_feature_control"]["name"] == (
        "in_domain_rival_control"
    )
    assert v73_details["paired_correctness_delta"] == {
        "point": 0.0,
        "ci_low": -0.01375,
        "ci_high": 0.01375,
        "resamples": 10000,
        "seed": 20260909,
    }
    assert v73_details["incremental_delta_vs_strongest_prior_feature_control"] == {
        "point": 0.0,
        "ci_low": -0.01375,
        "ci_high": 0.01375,
        "resamples": 10000,
        "seed": 20261909,
    }
    assert v73_details["incremental_delta_vs_erasure_only_control"] == {
        "point": 0.13875,
        "ci_low": 0.10375,
        "ci_high": 0.175,
        "resamples": 10000,
        "seed": 20262909,
    }
    assert v73_details["mean_erasure_dependency_coverage_fraction"] == 0.99875
    assert v73_details["unanswerable_hop_strata"] == {
        "2": {"cases": 292, "rejection_rate": 0.797945},
        "3": {"cases": 87, "rejection_rate": 0.850575},
        "4": {"cases": 21, "rejection_rate": 0.952381},
    }
    assert (
        v73_details["support_checks"]["candidate_answerable_pass_rate_at_least_0_60"]
        is False
    )
    assert (
        v73_details["support_checks"]["candidate_balanced_accuracy_at_least_0_72"]
        is False
    )
    assert (
        v73_details["support_checks"][
            "candidate_minus_strongest_fair_baseline_at_least_0_05"
        ]
        is False
    )
    assert (
        v73_details["support_checks"][
            "candidate_minus_strongest_prior_feature_control_at_least_0_02"
        ]
        is False
    )
    assert (
        v73_details["support_checks"][
            "candidate_unanswerable_rejection_rate_at_least_0_80"
        ]
        is True
    )
    assert (
        v73_details["support_checks"][
            "candidate_minus_erasure_only_control_at_least_0_02"
        ]
        is True
    )
    assert (
        v73_details["support_checks"][
            "erasure_only_control_paired_correctness_ci_low_above_0"
        ]
        is True
    )
    assert (
        v73_details["support_checks"][
            "every_observed_unanswerable_hop_stratum_rejection_rate_at_least_0_75"
        ]
        is True
    )
    assert v73_details["support_checks"]["invalid_feature_output_rate_equals_0"] is True
    assert v73_details["confirmation_opened"] is False
    assert v73_details["automatic_decomposer_result"] is False
    assert (
        v73_details[
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection"
        ]
        is False
    )
    assert v73_details["selector_adoption_authorized"] is False
    assert v73_details["canary_or_default_authorized"] is False
    assert v73_details["gate_2"] == "NO-GO/SHADOW"

    twowiki_v74 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "twowiki_v74_support_path_closure_boundary"
    )
    assert twowiki_v74["status"] == "PASS"
    v74_details = twowiki_v74["details"]
    assert v74_details["status"] == (
        "2WIKI_V74_SUPPORT_PATH_CLOSURE_DEVELOPMENT_SUPPORT_NOT_"
        "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v74_details["history_cases_excluded"] == 1000
    assert v74_details["development_cases"] == 800
    assert v74_details["question_type_counts"] == {
        "bridge_comparison": 200,
        "comparison": 200,
        "compositional": 200,
        "inference": 200,
    }
    assert v74_details["candidate_count"] == 24354
    assert v74_details["history_overlap"] == 0
    assert v74_details["stage_overlap"] == 0
    assert v74_details["invalid_selector_output_count"] == 0
    assert v74_details["methods"]["soft_title_link_support_path_closure_v74"] == {
        "cases": 800,
        "evidence_macro_f1": 0.518988,
        "evidence_macro_recall": 0.809375,
        "complete_evidence_recall": 0.6075,
        "mean_selected_tokens": 126.10875,
    }
    assert v74_details["strongest_same_resource_control"] == {
        "name": "alternating_anchor_link_control",
        "cases": 800,
        "evidence_macro_f1": 0.51727,
        "evidence_macro_recall": 0.799437,
        "complete_evidence_recall": 0.55125,
        "mean_selected_tokens": 124.9675,
    }
    assert v74_details["candidate_minus_strongest_control"] == {
        "point": 0.001718,
        "ci_low": -0.008036,
        "ci_high": 0.011401,
        "resamples": 10000,
        "seed": 20261010,
    }
    assert v74_details["candidate_minus_frc_select"] == {
        "point": 0.076095,
        "ci_low": 0.063476,
        "ci_high": 0.088913,
        "resamples": 10000,
        "seed": 20261006,
    }
    assert v74_details["per_question_type_delta_vs_strongest_control"] == {
        "bridge_comparison": {
            "cases": 200,
            "candidate_evidence_macro_f1": 0.631667,
            "strongest_control_evidence_macro_f1": 0.676222,
            "delta": -0.044556,
        },
        "comparison": {
            "cases": 200,
            "candidate_evidence_macro_f1": 0.507143,
            "strongest_control_evidence_macro_f1": 0.551429,
            "delta": -0.044286,
        },
        "compositional": {
            "cases": 200,
            "candidate_evidence_macro_f1": 0.481429,
            "strongest_control_evidence_macro_f1": 0.444286,
            "delta": 0.037143,
        },
        "inference": {
            "cases": 200,
            "candidate_evidence_macro_f1": 0.455714,
            "strongest_control_evidence_macro_f1": 0.397143,
            "delta": 0.058571,
        },
    }
    assert (
        v74_details["support_checks"]["candidate_minus_frc_select_ci_low_above_0"]
        is True
    )
    assert (
        v74_details["support_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_01"
        ]
        is False
    )
    assert (
        v74_details["support_checks"][
            "candidate_minus_strongest_control_ci_low_above_0"
        ]
        is False
    )
    assert (
        v74_details["support_checks"]["every_question_type_delta_at_least_minus_0_02"]
        is False
    )
    assert v74_details["confirmation_opened"] is False
    assert v74_details["official_2wiki_leaderboard_result"] is False
    assert v74_details["strict_independent_model_training_confirmation"] is False
    assert (
        v74_details["reuse_target_stage_for_method_weight_gate_or_selection"] is False
    )
    assert v74_details["selector_adoption_authorized"] is False
    assert v74_details["canary_or_default_authorized"] is False
    assert v74_details["gate_2"] == "NO-GO/SHADOW"

    twowiki_v75 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "twowiki_v75_question_router_boundary"
    )
    assert twowiki_v75["status"] == "PASS"
    v75_details = twowiki_v75["details"]
    assert v75_details["status"] == (
        "2WIKI_V75_QUESTION_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED"
    )
    assert v75_details["prior_cases_excluded"] == 1800
    assert v75_details["development_cases"] == 800
    assert v75_details["confirmation_cases"] == 800
    assert v75_details["development_confirmation_overlap"] == 0
    assert v75_details["development_candidate_count"] == 24993
    assert v75_details["confirmation_candidate_count"] == 25096
    assert v75_details["development_methods"][
        "learned_question_routed_support_path_v75"
    ] == {
        "cases": 800,
        "evidence_macro_f1": 0.537489,
        "evidence_macro_recall": 0.837687,
        "complete_evidence_recall": 0.635,
        "mean_selected_tokens": 126.44625,
    }
    assert v75_details["confirmation_methods"][
        "learned_question_routed_support_path_v75"
    ] == {
        "cases": 800,
        "evidence_macro_f1": 0.531013,
        "evidence_macro_recall": 0.8245,
        "complete_evidence_recall": 0.595,
        "mean_selected_tokens": 128.12625,
    }
    assert v75_details["development_candidate_minus_strongest_static"] == {
        "point": 0.019575,
        "ci_low": 0.013056,
        "ci_high": 0.026464,
        "resamples": 10000,
        "seed": 20261013,
    }
    assert v75_details["confirmation_candidate_minus_strongest_static"] == {
        "point": 0.013571,
        "ci_low": 0.008214,
        "ci_high": 0.018929,
        "resamples": 10000,
        "seed": 20261015,
    }
    assert v75_details["development_router"]["balanced_accuracy"] == 1.0
    assert v75_details["confirmation_router"]["balanced_accuracy"] == 0.99875
    assert v75_details["confirmation_candidate_minus_lexical_router"] == {
        "point": 0.0,
        "ci_low": -0.001071,
        "ci_high": 0.001071,
        "resamples": 10000,
        "seed": 20261016,
    }
    assert (
        v75_details["learned_router_superiority_over_lexical_control_established"]
        is False
    )
    assert v75_details["independent_dataset_confirmation"] is False
    assert v75_details["official_2wiki_leaderboard_result"] is False
    assert v75_details["selector_adoption_authorized"] is False
    assert v75_details["canary_or_default_authorized"] is False
    assert v75_details["gate_2"] == "NO-GO/SHADOW"

    hotpot_v76 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "hotpot_v76_graph_router_boundary"
    )
    assert hotpot_v76["status"] == "PASS"
    v76_details = hotpot_v76["details"]
    assert v76_details["status"] == (
        "HOTPOT_V76_GRAPH_ROUTER_CASE_DISJOINT_SUPPORT_ESTABLISHED"
    )
    assert v76_details["history_cases_excluded"] == 1000
    assert v76_details["historical_configurations"] == 1089
    assert v76_details["development_cases"] == 800
    assert v76_details["confirmation_cases"] == 800
    assert v76_details["development_confirmation_overlap"] == 0
    assert v76_details["development_candidate_count"] == 32022
    assert v76_details["confirmation_candidate_count"] == 32407
    assert v76_details["development_methods"]["graph_gbr_soft_override_router_v76"] == {
        "cases": 800,
        "evidence_macro_f1": 0.568959,
        "evidence_macro_recall": 0.904979,
        "complete_evidence_recall": 0.775,
        "mean_selected_tokens": 131.31875,
    }
    assert v76_details["confirmation_methods"][
        "graph_gbr_soft_override_router_v76"
    ] == {
        "cases": 800,
        "evidence_macro_f1": 0.570916,
        "evidence_macro_recall": 0.906979,
        "complete_evidence_recall": 0.7875,
        "mean_selected_tokens": 130.5875,
    }
    assert v76_details["development_candidate_minus_strongest_control"] == {
        "point": 0.008119,
        "ci_low": 0.003859,
        "ci_high": 0.012623,
        "resamples": 10000,
        "seed": 20261031,
    }
    assert v76_details["confirmation_candidate_minus_strongest_control"] == {
        "point": 0.00646,
        "ci_low": 0.002201,
        "ci_high": 0.010864,
        "resamples": 10000,
        "seed": 20261032,
    }
    assert v76_details["development_router"] == {
        "soft_override_rate": 0.22,
        "history_crossfit_override_rate": 0.346,
    }
    assert v76_details["confirmation_router"] == {
        "soft_override_rate": 0.23625,
        "history_crossfit_override_rate": 0.346,
    }
    assert v76_details["confirmation_per_question_type"] == {
        "bridge": {
            "cases": 400,
            "best_control": "zero_shot_v75_question_router_control",
            "candidate_evidence_macro_f1": 0.54864,
            "best_control_evidence_macro_f1": 0.543378,
            "delta": 0.005262,
        },
        "comparison": {
            "cases": 400,
            "best_control": "cross_encoder_topk",
            "candidate_evidence_macro_f1": 0.593193,
            "best_control_evidence_macro_f1": 0.594532,
            "delta": -0.001339,
        },
    }
    assert v76_details["independent_public_dataset_from_v75_2wiki"] is True
    assert v76_details["independent_dataset_from_v76_hotpot_history"] is False
    assert v76_details["strict_independent_model_training_confirmation"] is False
    assert v76_details["official_hotpotqa_leaderboard_result"] is False
    assert v76_details["selector_adoption_authorized"] is False
    assert v76_details["canary_or_default_authorized"] is False
    assert v76_details["gate_2"] == "NO-GO/SHADOW"

    musique_v77 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v77_frozen_router_transfer_boundary"
    )
    assert musique_v77["status"] == "PASS"
    v77_details = musique_v77["details"]
    assert v77_details["status"] == (
        "MUSIQUE_V77_FROZEN_V76_ROUTER_TRANSFER_ADVANTAGE_"
        "NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v77_details["prior_source_commitments_excluded"] == 16900
    assert v77_details["development_cases"] == 800
    assert v77_details["hop_count_distribution"] == {
        "2": 400,
        "3": 250,
        "4": 150,
    }
    assert v77_details["candidate_count"] == 15999
    assert v77_details["methods"]["graph_gbr_soft_override_router_v76"] == {
        "cases": 800,
        "evidence_macro_f1": 0.535779,
        "evidence_macro_recall": 0.798333,
        "complete_evidence_recall": 0.5425,
        "mean_selected_tokens": 542.12125,
    }
    assert v77_details["strongest_registered_control"]["name"] == (
        "alternating_anchor_link_control"
    )
    assert v77_details["candidate_minus_strongest_control"] == {
        "point": -0.006225,
        "ci_low": -0.012733,
        "ci_high": 0.000164,
        "resamples": 10000,
        "seed": 20261104,
    }
    assert v77_details["candidate_minus_cross_encoder"] == {
        "point": 0.007426,
        "ci_low": 0.002535,
        "ci_high": 0.012416,
        "resamples": 10000,
        "seed": 20261101,
    }
    assert v77_details["per_hop"]["2"]["delta"] == -0.000714
    assert v77_details["per_hop"]["3"]["delta"] == -0.009
    assert v77_details["per_hop"]["4"]["delta"] == -0.016296
    assert v77_details["router"] == {
        "soft_override_rate": 0.4375,
        "v76_history_crossfit_override_rate": 0.346,
    }
    assert (
        v77_details["support_checks"]["candidate_minus_cross_encoder_ci_low_above_0"]
        is True
    )
    assert (
        v77_details["support_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_005"
        ]
        is False
    )
    assert all(v77_details["noninferiority_checks"].values()) is False
    assert v77_details["confirmation_opened"] is False
    assert v77_details["target_training_or_tuning_cases"] == 0
    assert v77_details["broader_research_program_independent_of_musique"] is False
    assert v77_details["strict_independent_model_training_confirmation"] is False
    assert v77_details["official_musique_leaderboard_result"] is False
    assert v77_details["selector_adoption_authorized"] is False
    assert v77_details["canary_or_default_authorized"] is False
    assert v77_details["gate_2"] == "NO-GO/SHADOW"

    musique_v78 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v78_mean_calibrated_three_route_boundary"
    )
    assert musique_v78["status"] == "PASS"
    v78_details = musique_v78["details"]
    assert v78_details["status"] == (
        "MUSIQUE_V78_MEAN_CALIBRATED_THREE_ROUTE_ADVANTAGE_"
        "NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v78_details["history_route_policy_configurations"] == 432
    assert v78_details["history_candidate_minus_v76"]["point"] == 0.00056
    assert v78_details["v77_calibration_cases"] == 800
    assert v78_details["v77_calibration_configuration_count"] == 1
    assert v78_details["development_cases"] == 600
    assert v78_details["hop_count_distribution"] == {
        "2": 300,
        "3": 200,
        "4": 100,
    }
    assert v78_details["candidate_count"] == 12000
    assert v78_details["methods"]["mean_calibrated_three_route_router_v78"] == {
        "cases": 600,
        "evidence_macro_f1": 0.546382,
        "evidence_macro_recall": 0.814444,
        "complete_evidence_recall": 0.563333,
        "mean_selected_tokens": 539.298333,
    }
    assert v78_details["strongest_registered_control"]["name"] == (
        "alternating_anchor_link_control"
    )
    assert v78_details["candidate_minus_strongest_control"] == {
        "point": -0.0017,
        "ci_low": -0.006865,
        "ci_high": 0.003572,
        "resamples": 10000,
        "seed": 20261124,
    }
    assert v78_details["candidate_minus_v76"] == {
        "point": 0.000496,
        "ci_low": -0.006369,
        "ci_high": 0.007388,
        "resamples": 10000,
        "seed": 20261127,
    }
    assert v78_details["per_hop"]["2"]["delta"] == -0.000952
    assert v78_details["per_hop"]["3"]["delta"] == 0.00125
    assert v78_details["per_hop"]["4"]["delta"] == -0.015556
    assert v78_details["router"]["route_counts"] == {
        "alternating_anchor_link_control": 316,
        "soft_title_link_support_path_closure_v74": 284,
    }
    assert (
        v78_details["support_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_005"
        ]
        is False
    )
    assert all(v78_details["noninferiority_checks"].values()) is True
    assert v78_details["confirmation_opened"] is False
    assert v78_details["v78_target_training_or_tuning_cases"] == 0
    assert v78_details["strict_zero_target_tuning"] is False
    assert v78_details["strict_independent_model_training_confirmation"] is False
    assert v78_details["official_musique_leaderboard_result"] is False
    assert v78_details["selector_adoption_authorized"] is False
    assert v78_details["canary_or_default_authorized"] is False
    assert v78_details["gate_2"] == "NO-GO/SHADOW"

    musique_v79 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "musique_v79_target_three_route_boundary"
    )
    assert musique_v79["status"] == "PASS"
    v79_details = musique_v79["details"]
    assert v79_details["status"] == (
        "MUSIQUE_V79_TARGET_THREE_ROUTE_ADVANTAGE_"
        "NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
    )
    assert v79_details["route_policy_configurations"] == 432
    assert v79_details["v78_training_cases"] == 600
    assert (
        v79_details["oof_model_selection_diagnostic"]["evidence_macro_f1"] == 0.551733
    )
    assert v79_details["development_cases"] == 470
    assert v79_details["hop_count_distribution"] == {
        "2": 250,
        "3": 150,
        "4": 70,
    }
    assert v79_details["candidate_count"] == 9400
    assert v79_details["methods"]["musique_target_trained_three_route_router_v79"] == {
        "cases": 470,
        "evidence_macro_f1": 0.553723,
        "evidence_macro_recall": 0.831383,
        "complete_evidence_recall": 0.597872,
        "mean_selected_tokens": 534.485106,
    }
    assert v79_details["strongest_registered_control"]["name"] == (
        "alternating_anchor_link_control"
    )
    assert v79_details["candidate_minus_strongest_control"] == {
        "point": 0.002871,
        "ci_low": -0.002187,
        "ci_high": 0.007844,
        "resamples": 10000,
        "seed": 20261144,
    }
    assert v79_details["candidate_minus_v78"] == {
        "point": 0.005961,
        "ci_low": -0.00038,
        "ci_high": 0.012606,
        "resamples": 10000,
        "seed": 20261148,
    }
    assert v79_details["per_hop"]["2"]["delta"] == 0.001143
    assert v79_details["per_hop"]["3"]["delta"] == 0.006667
    assert v79_details["per_hop"]["4"]["delta"] == -0.003175
    assert v79_details["router"]["route_counts"] == {
        "alternating_anchor_link_control": 164,
        "cross_encoder_topk": 193,
        "soft_title_link_support_path_closure_v74": 113,
    }
    assert (
        v79_details["support_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_005"
        ]
        is False
    )
    assert (
        v79_details["support_checks"]["candidate_minus_frozen_v78_ci_low_above_0"]
        is False
    )
    assert all(v79_details["noninferiority_checks"].values()) is True
    assert v79_details["confirmation_opened"] is False
    assert v79_details["v79_target_training_or_tuning_cases"] == 0
    assert v79_details["strict_zero_target_tuning"] is False
    assert v79_details["strict_independent_model_training_confirmation"] is False
    assert v79_details["official_musique_leaderboard_result"] is False
    assert v79_details["selector_adoption_authorized"] is False
    assert v79_details["canary_or_default_authorized"] is False
    assert v79_details["gate_2"] == "NO-GO/SHADOW"

    musique_v80 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "musique_v80_anchor_default_terminal_boundary"
    )
    assert musique_v80["status"] == "PASS"
    v80_details = musique_v80["details"]
    assert v80_details["status"] == (
        "MUSIQUE_V80_ANCHOR_DEFAULT_TERMINAL_HOLDOUT_ADVANTAGE_NOT_ESTABLISHED"
    )
    assert v80_details["route_policy_configurations"] == 1512
    assert v80_details["target_domain_training_cases"] == 1070
    assert (
        v80_details["oof_model_selection_diagnostic"]["evidence_macro_f1"] == 0.554843
    )
    assert v80_details["terminal_holdout_cases"] == 600
    assert v80_details["hop_count_distribution"] == {
        "2": 400,
        "3": 150,
        "4": 50,
    }
    assert v80_details["candidate_count"] == 11998
    assert v80_details["methods"]["musique_anchor_default_three_route_router_v80"] == {
        "cases": 600,
        "evidence_macro_f1": 0.527817,
        "evidence_macro_recall": 0.835833,
        "complete_evidence_recall": 0.643333,
        "mean_selected_tokens": 551.835,
    }
    assert v80_details["strongest_registered_control"]["name"] == (
        "musique_target_trained_three_route_router_v79"
    )
    assert v80_details["candidate_minus_v79"] == {
        "point": 0.001561,
        "ci_low": -0.002533,
        "ci_high": 0.005774,
        "resamples": 10000,
        "seed": 20261169,
    }
    assert v80_details["per_hop"]["2"]["delta"] == 0.003571
    assert v80_details["per_hop"]["3"]["delta"] == -0.006667
    assert v80_details["per_hop"]["4"]["delta"] == 0.004444
    assert v80_details["router"]["route_counts"] == {
        "alternating_anchor_link_control": 273,
        "cross_encoder_topk": 186,
        "soft_title_link_support_path_closure_v74": 141,
    }
    assert (
        v80_details["support_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_005"
        ]
        is False
    )
    assert (
        v80_details["support_checks"][
            "every_hop_delta_vs_best_control_at_least_minus_0_005"
        ]
        is False
    )
    assert all(v80_details["noninferiority_checks"].values()) is True
    assert v80_details["further_same_source_confirmation_authorized"] is False
    assert v80_details["holdout_reuse_for_tuning_or_selection"] is False
    assert v80_details["v80_target_training_or_tuning_cases"] == 0
    assert v80_details["strict_zero_target_tuning"] is False
    assert v80_details["strict_independent_model_training_confirmation"] is False
    assert v80_details["official_musique_leaderboard_result"] is False
    assert v80_details["selector_adoption_authorized"] is False
    assert v80_details["canary_or_default_authorized"] is False
    assert v80_details["gate_2"] == "NO-GO/SHADOW"

    twowiki_v81 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"] == "twowiki_v81_residual_three_route_boundary"
    )
    assert twowiki_v81["status"] == "PASS"
    v81_details = twowiki_v81["details"]
    assert v81_details["status"] == (
        "2WIKI_V81_RESIDUAL_ROUTER_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP"
    )
    assert v81_details["route_policy_configurations"] == 1512
    assert v81_details["training_cases"] == 1600
    assert (
        v81_details["oof_model_selection_diagnostic"]["evidence_macro_f1"] == 0.540164
    )
    assert v81_details["development_cases"] == 800
    assert v81_details["question_type_distribution"] == {
        "bridge_comparison": 200,
        "comparison": 200,
        "compositional": 200,
        "inference": 200,
    }
    assert v81_details["candidate_count"] == 25875
    assert v81_details["methods"]["twowiki_residual_three_route_router_v81"] == {
        "cases": 800,
        "evidence_macro_f1": 0.547195,
        "evidence_macro_recall": 0.851,
        "complete_evidence_recall": 0.65875,
        "mean_selected_tokens": 126.43125,
    }
    assert v81_details["strongest_registered_control"]["name"] == (
        "learned_question_routed_support_path_v75"
    )
    assert v81_details["candidate_minus_v75"] == {
        "point": 0.004444,
        "ci_low": 0.001706,
        "ci_high": 0.007381,
        "resamples": 10000,
        "seed": 20261186,
    }
    assert v81_details["per_question_type"]["comparison"]["delta"] == -0.008571
    assert v81_details["router"]["action_counts"] == {
        "cross_override": 80,
        "soft_anchor_flip_override": 202,
        "v75_default": 518,
    }
    assert (
        v81_details["strict_gate_checks"][
            "candidate_minus_strongest_control_f1_at_least_0_005"
        ]
        is False
    )
    assert (
        v81_details["strict_gate_checks"][
            "every_question_type_delta_vs_best_control_at_least_minus_0_005"
        ]
        is False
    )
    assert all(v81_details["noninferiority_checks"].values()) is True
    assert v81_details["confirmation_opened"] is False
    assert v81_details["target_training_or_tuning_cases"] == 0
    assert v81_details["strict_independent_dataset_confirmation"] is False
    assert v81_details["official_2wiki_leaderboard_result"] is False
    assert v81_details["selector_adoption_authorized"] is False
    assert v81_details["canary_or_default_authorized"] is False
    assert v81_details["gate_2"] == "NO-GO/SHADOW"

    twowiki_v82 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "twowiki_v82_cascaded_style_residual_boundary"
    )
    assert twowiki_v82["status"] == "PASS"
    v82_details = twowiki_v82["details"]
    assert v82_details["status"] == (
        "2WIKI_V82_CASCADED_ROUTER_DEVELOPMENT_ADVANTAGE_NOT_ESTABLISHED_STOP"
    )
    assert v82_details["route_policy_configurations"] == 560
    assert v82_details["training_cases"] == 2400
    assert v82_details["selected_configuration"]["classifier_configuration_index"] == 11
    assert v82_details["selected_configuration"]["hash_dimension"] == 256
    assert v82_details["selected_configuration"]["l2"] == 32.0
    assert v82_details["selected_configuration"]["direct_threshold"] == 0.8
    assert v82_details["selected_configuration"]["flip_threshold"] == 0.015
    assert (
        v82_details["oof_model_selection_diagnostic"]["evidence_macro_f1"] == 0.543817
    )
    assert v82_details["development_cases"] == 800
    assert v82_details["question_type_distribution"] == {
        "bridge_comparison": 200,
        "comparison": 200,
        "compositional": 200,
        "inference": 200,
    }
    assert v82_details["candidate_count"] == 25305
    assert v82_details["methods"]["twowiki_cascaded_style_residual_router_v82"] == {
        "cases": 800,
        "evidence_macro_f1": 0.544792,
        "evidence_macro_recall": 0.844271,
        "complete_evidence_recall": 0.64375,
        "mean_selected_tokens": 124.53,
    }
    assert v82_details["strongest_registered_control"]["name"] == (
        "twowiki_residual_three_route_router_v81"
    )
    assert v82_details["candidate_minus_v81"] == {
        "point": 0.002143,
        "ci_low": 0.000357,
        "ci_high": 0.004286,
        "resamples": 10000,
        "seed": 20261202,
    }
    assert v82_details["candidate_minus_v75"] == {
        "point": 0.006508,
        "ci_low": 0.003373,
        "ci_high": 0.009762,
        "resamples": 10000,
        "seed": 20261197,
    }
    assert v82_details["per_question_type"]["bridge_comparison"]["delta"] == 0.0
    assert v82_details["per_question_type"]["comparison"]["delta"] == 0.0
    assert v82_details["per_question_type"]["compositional"]["delta"] == (-0.001429)
    assert v82_details["per_question_type"]["inference"]["delta"] == -0.002857
    assert v82_details["router"]["action_counts"] == {
        "direct_comparison_cross_override": 168,
        "v75_default": 478,
        "v81_safe_flip_override": 154,
    }
    assert v82_details["router"]["direct_comparison_balanced_accuracy"] == 0.92
    failed_v82_strict_gates = [
        name for name, passed in v82_details["strict_gate_checks"].items() if not passed
    ]
    assert failed_v82_strict_gates == [
        "candidate_minus_strongest_control_f1_at_least_0_005"
    ]
    assert all(v82_details["noninferiority_checks"].values()) is True
    assert v82_details["confirmation_opened"] is False
    assert v82_details["target_training_or_tuning_cases"] == 0
    assert v82_details["strict_independent_dataset_confirmation"] is False
    assert v82_details["official_2wiki_leaderboard_result"] is False
    assert v82_details["selector_adoption_authorized"] is False
    assert v82_details["canary_or_default_authorized"] is False
    assert v82_details["gate_2"] == "NO-GO/SHADOW"

    twowiki_v83 = next(
        requirement
        for requirement in report["requirements"]
        if requirement["requirement_id"]
        == "twowiki_v83_bridge_aware_precision_trim_boundary"
    )
    assert twowiki_v83["status"] == "PASS"
    v83_details = twowiki_v83["details"]
    assert v83_details["status"] == (
        "2WIKI_V83_PRECISION_TRIM_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED"
    )
    assert v83_details["training_cases"] == 3200
    assert v83_details["classifier_configurations"] == 16
    assert v83_details["policy_configurations"] == 112
    assert v83_details["selected_configuration"][
        "classifier_configuration_index"
    ] == 3
    assert v83_details["selected_configuration"]["hash_dimension"] == 64
    assert v83_details["selected_configuration"]["l2"] == 32.0
    assert v83_details["selected_configuration"]["bridge_threshold"] == 0.3
    assert v83_details["development_cases"] == 800
    assert v83_details["confirmation_cases"] == 800
    assert v83_details["stage_overlap"] == 0
    development = v83_details["development"]
    confirmation = v83_details["confirmation"]
    assert development["candidate_count"] == 25741
    assert development["methods"]["twowiki_bridge_aware_precision_trim_v83"] == {
        "cases": 800,
        "evidence_macro_f1": 0.556329,
        "evidence_macro_recall": 0.836437,
        "complete_evidence_recall": 0.6325,
        "mean_selected_tokens": 119.7375,
    }
    assert development["candidate_minus_v81"] == {
        "point": 0.011373,
        "ci_low": 0.007802,
        "ci_high": 0.014985,
        "resamples": 10000,
        "seed": 20261214,
    }
    assert development["candidate_minus_v82"] == {
        "point": 0.011651,
        "ci_low": 0.008184,
        "ci_high": 0.015118,
        "resamples": 10000,
        "seed": 20261215,
    }
    assert all(development["strict_gate_checks"].values()) is True
    assert development["confirmation_opened"] is True
    assert confirmation["candidate_count"] == 25174
    assert confirmation["methods"]["twowiki_bridge_aware_precision_trim_v83"] == {
        "cases": 800,
        "evidence_macro_f1": 0.54942,
        "evidence_macro_recall": 0.828125,
        "complete_evidence_recall": 0.61625,
        "mean_selected_tokens": 118.68875,
    }
    assert confirmation["candidate_minus_v82"] == {
        "point": 0.006434,
        "ci_low": 0.002465,
        "ci_high": 0.010253,
        "resamples": 10000,
        "seed": 20261216,
    }
    failed_v83_confirmation_gates = [
        name
        for name, passed in confirmation["strict_gate_checks"].items()
        if not passed
    ]
    assert failed_v83_confirmation_gates == [
        "candidate_evidence_macro_f1_at_least_0_55"
    ]
    assert all(confirmation["noninferiority_checks"].values()) is True
    assert confirmation["precision_trim"]["bridge_classifier_balanced_accuracy"] == (
        0.989167
    )
    assert confirmation["precision_trim"]["action_counts"] == {
        "bridge_aware_trim_to_four": 209,
        "keep_frozen_v82_selection": 591,
    }
    assert v83_details["target_training_or_tuning_cases"] == 0
    assert v83_details["strict_independent_dataset_confirmation"] is False
    assert v83_details["official_2wiki_leaderboard_result"] is False
    assert v83_details["selector_adoption_authorized"] is False
    assert v83_details["canary_or_default_authorized"] is False
    assert v83_details["gate_2"] == "NO-GO/SHADOW"


def test_hotpot_v84_cardinality_boundary_is_audited() -> None:
    report = build_progressive_completion_audit(REPO_ROOT)
    check = next(
        item
        for item in report["requirements"]
        if item["requirement_id"]
        == "hotpot_v84_question_type_cardinality_boundary"
    )
    assert check["status"] == "PASS"
    details = check["details"]
    assert details["status"] == (
        "HOTPOT_V84_CARDINALITY_CASE_DISJOINT_CONSTRAINED_ADVANTAGE_ESTABLISHED"
    )
    assert details["training_cases"] == 2600
    assert details["classifier_configurations"] == 16
    assert details["policy_configurations"] == 112
    assert details["selected_configuration"]["hash_dimension"] == 512
    assert details["selected_configuration"]["l2"] == 32.0
    assert details["selected_configuration"]["comparison_threshold"] == 0.3
    assert details["development_cases"] == 600
    assert details["confirmation_cases"] == 600
    assert details["stage_overlap"] == 0
    development = details["development"]
    confirmation = details["confirmation"]
    candidate = "hotpot_question_type_cardinality_router_v84"
    prefix3 = "frozen_v76_prefix3_cardinality_control"
    prefix4 = "frozen_v76_prefix4_cardinality_control"
    assert development["methods"][candidate] == {
        "evidence_macro_f1": 0.644873,
        "evidence_macro_recall": 0.832607,
        "complete_evidence_recall": 0.643333,
        "mean_selected_tokens": 97.513333,
    }
    assert confirmation["methods"][candidate] == {
        "evidence_macro_f1": 0.639898,
        "evidence_macro_recall": 0.82452,
        "complete_evidence_recall": 0.625,
        "mean_selected_tokens": 96.92,
    }
    assert development["raw_strongest_registered_control"]["name"] == prefix3
    assert development["raw_strongest_registered_control"]["safety_eligible"] is False
    assert confirmation["raw_strongest_registered_control"]["name"] == prefix3
    assert confirmation["raw_strongest_registered_control"]["safety_eligible"] is False
    assert development["strongest_safety_eligible_control"]["name"] == prefix4
    assert confirmation["strongest_safety_eligible_control"]["name"] == prefix4
    assert development["candidate_minus_prefix4"] == {
        "point": 0.029039,
        "ci_low": 0.022617,
        "ci_high": 0.035716,
        "resamples": 10000,
        "seed": 20261215,
    }
    assert confirmation["candidate_minus_prefix4"] == {
        "point": 0.026492,
        "ci_low": 0.019841,
        "ci_high": 0.033051,
        "resamples": 10000,
        "seed": 20261216,
    }
    assert all(development["strict_gate_checks"].values()) is True
    assert all(confirmation["strict_gate_checks"].values()) is True
    assert details["target_training_or_tuning_cases"] == 0
    assert details["strict_independent_dataset_confirmation"] is False
    assert details["official_hotpotqa_leaderboard_result"] is False
    assert details[
        "raw_f1_superiority_over_every_registered_control_established"
    ] is False
    assert details["selector_adoption_authorized"] is False
    assert details["canary_or_default_authorized"] is False
    assert details["gate_2"] == "NO-GO/SHADOW"


def test_artifact_check_reports_the_exact_missing_deliverable(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("evidence", encoding="utf-8")

    result = check_artifact_groups(
        tmp_path,
        "demo",
        "demo artifacts",
        {"group": ("present.txt", "missing.txt")},
    )

    assert result["status"] == "FAIL"
    assert result["details"]["missing"] == {"group": ["missing.txt"]}


def test_gate_one_check_rejects_threshold_regression() -> None:
    report = {
        "baseline": {"recall_at_k": 0.94, "critical_miss_rate": 0.06, "ece": 0.11},
        "calibration_improvement": -0.01,
        "missing_feature_stress": {"recall_drop_percentage_points": 11.0},
    }

    result = check_gate_one(report)

    assert result["status"] == "FAIL"
    assert result["details"]["recall_at_5"] == 0.94


def test_completion_audit_output_is_deterministic(tmp_path: Path) -> None:
    first_json = tmp_path / "first.json"
    first_md = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_md = tmp_path / "second.md"

    write_progressive_completion_audit(REPO_ROOT, first_json, first_md)
    write_progressive_completion_audit(REPO_ROOT, second_json, second_md)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()
    report = json.loads(first_json.read_text(encoding="utf-8"))
    assert report["metadata"]["audit_version"] == "progressive-completion-audit-v84"


def test_evidence_hash_is_independent_of_platform_newlines(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes("第一行\n第二行\n".encode())
    crlf.write_bytes("第一行\r\n第二行\r\n".encode())

    assert _canonical_text_sha256(lf) == _canonical_text_sha256(crlf)


def test_binary_evidence_hash_uses_exact_archive_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "cases.jsonl.gz"
    payload = b"\x1f\x8b\x08\x00binary-evidence"
    archive.write_bytes(payload)

    assert _evidence_sha256(archive) == hashlib.sha256(payload).hexdigest()
