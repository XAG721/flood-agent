from __future__ import annotations

import json

from flood_system.frc_public_evidence import (
    _stable_missing_evidence_ids,
    aggregate_precomputed_selectors,
    audit_public_ablation_schema,
    build_design_experiment_audit,
    conflict_inventory,
    evidence_metrics,
    load_supplemental_ablation,
    paired_bootstrap,
    select_precomputed,
    selected_role_coverage,
)


def candidate(candidate_id: str, cross: float, roles: dict[str, float], *, gold: bool = False):
    return {
        "id": candidate_id,
        "token_count": 10,
        "gold": gold,
        "scores": {"bm25": cross, "dense": cross, "hybrid": cross, "cross_encoder": cross},
        "role_scores": roles,
    }


def test_precomputed_frc_balances_relevance_and_complementary_roles():
    row = {
        "required_roles": ["condition", "answer"],
        "candidates": [
            candidate("duplicate-high", 0.9, {"condition": 0.9, "answer": 0.1}),
            candidate("condition", 0.8, {"condition": 1.0, "answer": 0.1}, gold=True),
            candidate("answer", 0.7, {"condition": 0.1, "answer": 1.0}, gold=True),
        ],
    }

    selected = select_precomputed(row, "frc_select", k=2, budget=20)

    assert {item["id"] for item in selected} == {"duplicate-high", "answer"}
    assert selected_role_coverage({**row, "selected_evidence": selected}) == 1.0


def test_coverage_proxy_name_does_not_change_its_role_first_behavior():
    row = {
        "required_roles": ["condition", "answer"],
        "candidates": [
            candidate("condition", 0.4, {"condition": 0.9, "answer": 0.1}),
            candidate("answer", 0.3, {"condition": 0.1, "answer": 0.9}),
            candidate("relevant", 1.0, {"condition": 0.2, "answer": 0.2}),
        ],
    }

    selected = select_precomputed(row, "coverage_greedy_proxy", k=2, budget=20)

    assert [item["id"] for item in selected] == ["condition", "answer"]


def test_evidence_metrics_and_paired_bootstrap_are_deterministic():
    metrics = evidence_metrics(["a", "b"], ["a", "c"])
    assert metrics == {
        "evidence_recall": 0.5,
        "evidence_precision": 0.5,
        "evidence_f1": 0.5,
        "complete_evidence_set": 0.0,
    }

    first = paired_bootstrap([0.1, -0.1, 0.2], resamples=100)
    second = paired_bootstrap([0.1, -0.1, 0.2], resamples=100)
    assert first == second
    assert first["cases"] == 3


def test_conflict_inventory_imports_completed_audit_report(tmp_path):
    path = tmp_path / "conflicts_report.json"
    path.write_text(
        """{
          "metadata": {
            "name": "Google CONFLICTS FRC retrieval and conflict-classification audit",
            "status": "RUN",
            "cases": 458,
            "answer_annotated_cases": 237,
            "conflict_type_counts": {"No conflict": 161},
            "dataset_sha256": "dataset-hash"
          },
          "classification_metrics": {
            "coverage_greedy_proxy": {"accuracy": 0.34},
            "frc_select": {
              "accuracy": 0.33,
              "per_type": {"Conflict due to outdated information": {"recall": 0.56}}
            }
          },
          "strongest_reproducible_baseline_by_accuracy": "coverage_greedy_proxy",
          "paired_frc_minus_baseline_accuracy": {"ci_low": -0.04, "ci_high": 0.02},
          "decision": {"reason": "classification is not independent human judging"}
        }""",
        encoding="utf-8",
    )

    inventory = conflict_inventory(path)

    assert inventory["status"] == "RUN"
    assert inventory["cases"] == 458
    assert inventory["frc_accuracy"] == 0.33
    assert inventory["frc_outdated_recall"] == 0.56


def test_design_experiment_audit_does_not_treat_missing_ablations_as_passed(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "ablation_conditionalqa.csv").write_text(
        "Dataset,Method,Evidence Recall@5,Evidence Precision@5,Evidence F1@5,Role Coverage@5,Condition Coverage,Redundancy,Token Cost,Cases\n"
        "conditionalqa,frc_full,0.70,0.80,0.71,1.0,0.98,0.20,900,10\n"
        "conditionalqa,w/o Role,0.71,0.81,0.72,0.97,0.98,0.21,905,10\n"
        "conditionalqa,w/o Redundancy,0.70,0.80,0.71,1.0,0.98,0.20,900,10\n"
        "conditionalqa,Role-only,0.68,0.78,0.69,1.0,0.97,0.25,910,10\n"
        "conditionalqa,random_role,0.69,0.79,0.70,0.99,0.97,0.22,902,10\n",
        encoding="utf-8",
    )
    k_header = "Dataset,K,Method,Evidence Recall@K,Role Coverage@K,Condition Coverage@K,Answer F1@K,Token Cost@K,Cases\n"
    for dataset in ("conditionalqa", "multihoprag", "hotpotqa"):
        rows = []
        for k in (2, 3, 5, 8):
            rows.extend(
                [
                    f"{dataset},{k},cross_encoder_topk,0.6,0.9,0.8,0.1,500,10",
                    f"{dataset},{k},mmr,0.5,0.9,0.8,0.1,500,10",
                    f"{dataset},{k},setr_style,0.59,1.0,0.8,0.1,500,10",
                    f"{dataset},{k},frc_select,0.61,1.0,0.8,0.1,500,10",
                ]
            )
        (metrics / f"k_sensitivity_{dataset}.csv").write_text(
            k_header + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    (metrics / "frc_param_sweep_conditionalqa.csv").write_text(
        "Method,alpha,gamma,role_threshold,role_mix,Evidence Recall@5,Evidence F1@5,Role Coverage@5,Condition Coverage,Redundancy,Token Cost,Cases\n"
        "frc_select,2.0,0.0,0.55,0.15,0.70,0.71,1.0,0.98,0.20,900,10\n",
        encoding="utf-8",
    )
    role_scores = tmp_path / "role_scores"
    role_scores.mkdir()
    for dataset in ("conditionalqa", "multihoprag", "hotpotqa"):
        row = {
            "id": f"{dataset}-case",
            "required_roles": ["answer", "condition"],
            "gold_evidence_ids": ["gold-a", "gold-b"],
            "candidates": [
                candidate("gold-a", 1.0, {"answer": 1.0, "condition": 0.1}, gold=True),
                candidate("gold-b", 0.9, {"answer": 0.1, "condition": 1.0}, gold=True),
                candidate("other", 0.8, {"answer": 0.2, "condition": 0.2}),
            ],
        }
        (role_scores / f"role_scores_{dataset}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    audit = build_design_experiment_audit(metrics)

    assert audit["status"] == "PARTIAL"
    assert audit["ablation"]["missing_variants"] == [
        "w/o_field",
        "w/o_applicability",
        "w/o_conflict",
        "w/o_reranker",
    ]
    assert audit["ablation"]["gate_required_comparison"]["passed"] is False
    assert audit["k_sensitivity"]["status"] == "RUN"
    assert audit["parameter_sensitivity"]["configuration_count"] == 1
    assert audit["token_budget_sensitivity"]["status"] == "RUN"
    assert audit["missing_ratio_sensitivity"]["status"] == "RUN"


def test_supplemental_real_model_ablation_is_reaggregated_from_cases(tmp_path):
    path = tmp_path / "wo-reranker.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "frc-real-model-ablation-v1",
                "variant": "w/o_reranker",
                "metadata": {
                    "cross_encoder_used": False,
                    "scoring_backend": "bge_biencoder_no_cross_encoder",
                },
                "aggregate": {
                    "evidence_recall": 0.75,
                    "evidence_precision": 0.5,
                    "evidence_f1": 0.6,
                    "role_coverage": 0.875,
                    "token_cost": 100.0,
                },
                "case_results": [
                    {
                        "metrics": {
                            "evidence_recall": 0.5,
                            "evidence_precision": 0.5,
                            "evidence_f1": 0.5,
                            "role_coverage": 0.75,
                            "token_cost": 90,
                        }
                    },
                    {
                        "metrics": {
                            "evidence_recall": 1.0,
                            "evidence_precision": 0.5,
                            "evidence_f1": 0.7,
                            "role_coverage": 1.0,
                            "token_cost": 110,
                        }
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    variant, metrics, provenance = load_supplemental_ablation(path)

    assert variant == "w/o_reranker"
    assert metrics["evidence_f1"] == 0.6
    assert metrics["cases"] == 2
    assert provenance["metadata"]["cross_encoder_used"] is False


def test_public_schema_audit_blocks_unidentifiable_domain_ablations(tmp_path):
    for dataset in ("conditionalqa", "multihoprag", "hotpotqa"):
        row = {
            "id": f"{dataset}-case",
            "required_roles": ["answer"],
            "candidates": [candidate("evidence", 1.0, {"answer": 1.0})],
        }
        (tmp_path / f"role_scores_{dataset}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    audit = audit_public_ablation_schema(tmp_path)

    assert audit["totals"]["cases"] == 3
    assert audit["variants"]["w/o_field"]["status"] == "SCHEMA_BLOCKED"
    assert audit["variants"]["w/o_applicability"]["status"] == "SCHEMA_BLOCKED"
    assert audit["variants"]["w/o_conflict"]["status"] == "SCHEMA_BLOCKED"
    assert audit["variants"]["w/o_reranker"]["status"] == "REQUIRES_REAL_RESCORING"


def test_k_sensitivity_uses_coverage_proxy_name_instead_of_setr(tmp_path):
    metrics = tmp_path
    header = "Dataset,K,Method,Evidence Recall@K,Role Coverage@K,Condition Coverage@K,Answer F1@K,Token Cost@K,Cases\n"
    for dataset in ("conditionalqa", "multihoprag", "hotpotqa"):
        rows = []
        for k in (2, 3, 5, 8):
            for method, recall in (
                ("cross_encoder_topk", 0.6),
                ("mmr", 0.5),
                ("setr_style", 0.61),
                ("frc_select", 0.62),
            ):
                rows.append(f"{dataset},{k},{method},{recall},1.0,0.8,0.1,500,10")
        (metrics / f"k_sensitivity_{dataset}.csv").write_text(
            header + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    from flood_system.frc_public_evidence import load_k_sensitivity_audit

    audit = load_k_sensitivity_audit(metrics)

    assert audit["datasets"]["hotpotqa"]["summary"][0]["strongest_baseline"] == "coverage_greedy_proxy"


def test_precomputed_token_budget_is_strict():
    row = {
        "required_roles": ["answer"],
        "gold_evidence_ids": ["too-large"],
        "candidates": [candidate("too-large", 1.0, {"answer": 1.0}, gold=True) | {"token_count": 11}],
    }

    metrics = aggregate_precomputed_selectors(
        [row],
        ["cross_encoder_topk", "coverage_greedy_proxy", "frc_select"],
        k=5,
        budget=10,
    )

    assert all(values["token_cost"] == 0.0 for values in metrics.values())
    assert all(values["budget_violation"] == 0.0 for values in metrics.values())


def test_missing_ratio_removal_is_deterministic_and_nested():
    gold = [f"evidence-{index}" for index in range(100)]

    removed_25_first = _stable_missing_evidence_ids("case-1", gold, 0.25)
    removed_25_second = _stable_missing_evidence_ids("case-1", gold, 0.25)
    removed_50 = _stable_missing_evidence_ids("case-1", gold, 0.5)
    removed_75 = _stable_missing_evidence_ids("case-1", gold, 0.75)

    assert removed_25_first == removed_25_second
    assert removed_25_first <= removed_50 <= removed_75
    assert 15 <= len(removed_25_first) <= 35
