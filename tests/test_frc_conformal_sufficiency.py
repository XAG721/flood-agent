from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.conformal_robustness import (
    SCHEMA_VERSION as ROBUSTNESS_SCHEMA_VERSION,
    evaluate_conformal_robustness,
    load_conformal_robustness,
    write_conformal_robustness,
)
from research.frc_rag.conformal_model_selection import (
    SCHEMA_VERSION as MODEL_SELECTION_SCHEMA_VERSION,
    evaluate_nested_model_selection,
    load_nested_model_selection,
    write_nested_model_selection,
)
from research.frc_rag.conformal_cross_dataset import (
    SCHEMA_VERSION as CROSS_DATASET_SCHEMA_VERSION,
    evaluate_cross_dataset_confirmation,
    evaluate_cross_dataset_series,
    load_cross_dataset_confirmation,
    load_cross_dataset_series,
    write_cross_dataset_confirmation,
    write_cross_dataset_series,
)
from research.frc_rag.conformal_subgroup_audit import (
    evaluate_conformal_subgroup_audit,
    load_conformal_subgroup_audit,
    write_conformal_subgroup_audit,
)
from research.frc_rag.conformal_mondrian import (
    SCHEMA_VERSION as MONDRIAN_SCHEMA_VERSION,
    evaluate_hierarchical_mondrian,
    load_hierarchical_mondrian,
    write_hierarchical_mondrian,
)
from research.frc_rag.conformal_multi_axis_mondrian import (
    FALLBACK_ORDER as MULTI_AXIS_FALLBACK_ORDER,
    SCHEMA_VERSION as MULTI_AXIS_MONDRIAN_SCHEMA_VERSION,
    evaluate_multi_axis_mondrian,
    load_multi_axis_mondrian,
    write_multi_axis_mondrian,
)
from research.frc_rag.conformal_head_transfer import (
    SCHEMA_VERSION as HEAD_TRANSFER_SCHEMA_VERSION,
    evaluate_cross_dataset_head_transfer,
    fit_transferred_head,
    load_cross_dataset_head_transfer,
    write_cross_dataset_head_transfer,
)
from research.frc_rag.conformal_transfer_admission import (
    evaluate_transfer_admission_guard,
    issue_transfer_certificate,
    load_transfer_admission_guard,
    write_transfer_admission_guard,
)
from research.frc_rag.conformal_contextual import (
    SCHEMA_VERSION as CONTEXTUAL_SCHEMA_VERSION,
    evaluate_contextual_conformal,
    load_contextual_conformal,
    write_contextual_conformal,
)
from research.frc_rag.conformal_score_stability import (
    AUGMENTED_FEATURE_NAMES,
    SCHEMA_VERSION as SCORE_STABILITY_SCHEMA_VERSION,
    evaluate_score_stability_conformal,
    extract_score_stability_features,
    load_score_stability_conformal,
    write_score_stability_conformal,
)
from research.frc_rag.conformal_review_ranking import (
    SCHEMA_VERSION as REVIEW_RANKING_SCHEMA_VERSION,
    evaluate_review_ranking,
    load_review_ranking,
    write_review_ranking,
)
from research.frc_rag.conformal_sufficiency import (
    SCHEMA_VERSION,
    _mcnemar_exact,
    build_variants,
    evaluate_conformal_sufficiency,
    extract_features,
    load_conformal_sufficiency,
    prepare_conformal_source,
    write_conformal_sufficiency,
)
from research.frc_rag.public_evidence import select_precomputed
from scripts.benchmark_conformal_source_preparation import benchmark


def _candidate(
    candidate_id: str,
    cross: float,
    condition: float,
    answer: float,
    *,
    gold: bool,
) -> dict:
    return {
        "id": candidate_id,
        "token_count": 20,
        "gold": gold,
        "gold_roles": ["condition"] if condition > answer else ["answer"],
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": {"condition": condition, "answer": answer},
    }


def _rows(count: int = 120) -> list[dict]:
    rows = []
    for index in range(count):
        adjustment = (index % 7) * 0.01
        prefix = f"case-{index:03d}"
        rows.append(
            {
                "id": prefix,
                "dataset": "synthetic-test",
                "required_roles": ["condition", "answer"],
                "gold_evidence_ids": [f"{prefix}-condition", f"{prefix}-answer"],
                "candidates": [
                    _candidate(
                        f"{prefix}-condition",
                        0.82 - adjustment,
                        0.96,
                        0.08,
                        gold=True,
                    ),
                    _candidate(
                        f"{prefix}-answer",
                        0.78 - adjustment,
                        0.06,
                        0.95,
                        gold=True,
                    ),
                    _candidate(
                        f"{prefix}-distractor-a",
                        0.55 + adjustment,
                        0.42,
                        0.38,
                        gold=False,
                    ),
                    _candidate(
                        f"{prefix}-distractor-b",
                        0.35 + adjustment,
                        0.31,
                        0.44,
                        gold=False,
                    ),
                ],
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_features_do_not_read_gold_labels() -> None:
    row = _rows(1)[0]
    selected = select_precomputed(row, "frc_select", k=3, budget=100)
    original = extract_features(row, selected, k=3, token_budget=100)
    mutated = {
        **row,
        "gold_evidence_ids": ["invented"],
        "candidates": [
            {**candidate, "gold": not candidate["gold"], "gold_roles": ["invented"]}
            for candidate in row["candidates"]
        ],
    }
    mutated_selected = [
        next(
            candidate
            for candidate in mutated["candidates"]
            if candidate["id"] == selected_candidate["id"]
        )
        for selected_candidate in selected
    ]

    assert np.array_equal(
        original,
        extract_features(mutated, mutated_selected, k=3, token_budget=100),
    )


def test_grouped_conformal_experiment_is_deterministic_and_has_no_case_leakage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())

    first, first_cases = evaluate_conformal_sufficiency(source, k=3, token_budget=100)
    second, second_cases = evaluate_conformal_sufficiency(source, k=3, token_budget=100)
    prepared = prepare_conformal_source(source, k=3, token_budget=100)
    prepared_report, prepared_cases = evaluate_conformal_sufficiency(
        source,
        k=3,
        token_budget=100,
        prepared_source=prepared,
    )

    assert first == second
    assert first_cases == second_cases
    assert prepared_report == first
    assert prepared_cases == first_cases
    assert first["metadata"]["schema_version"] == SCHEMA_VERSION
    assert first["metadata"]["eligible_cases"] == 120
    assert first["metadata"]["variant_count"] == 480
    assert first["calibration"]["case_level_calibration_units"] > 10
    primary_calibration = first["calibration"]["alphas"]["0.1"]
    assert primary_calibration["finite_sample_nominal_case_error_upper_bound"] <= 0.1
    assert primary_calibration["threshold_order_statistic_rank"] > 0
    assert (
        first["evaluation"]["split_conformal"]["false_complete_case_family_rate"] <= 0.2
    )
    split_by_case: dict[str, set[str]] = {}
    for item in first_cases:
        split_by_case.setdefault(item["case_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in split_by_case.values())
    with pytest.raises(ValueError, match="token_budget"):
        evaluate_conformal_sufficiency(
            source,
            k=3,
            token_budget=101,
            prepared_source=prepared,
        )


def test_writer_and_loader_reaggregate_case_artifact(tmp_path: Path) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())
    report, cases = evaluate_conformal_sufficiency(source, k=3, token_budget=100)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    cases_path = tmp_path / "cases.jsonl.gz"

    write_conformal_sufficiency(
        report,
        cases,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )

    assert load_conformal_sufficiency(json_path, cases_path) == report
    assert "NO-GO/SHADOW" in markdown_path.read_text(encoding="utf-8")

    report["evaluation"]["split_conformal"]["declarations"] += 1
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="aggregate metrics"):
        load_conformal_sufficiency(json_path, cases_path)


def test_build_variants_rejects_duplicate_case_ids() -> None:
    row = _rows(1)[0]

    with pytest.raises(ValueError, match="duplicate case id"):
        build_variants([row, row])


def test_reference_run_provenance_is_hashed_and_generation_is_not_used(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    reference = tmp_path / "run_report.md"
    _write_rows(source, _rows())
    reference.write_text(
        "\n".join(
            [
                "- scoring backend: real",
                "- embedding model: test-embedder",
                "- reranker model: test-reranker",
                "- score calibration: per_case_minmax",
                "- role relevance mix: 0.15",
                "- FRC alpha/beta/gamma: 2.0/1.0/0.0",
            ]
        ),
        encoding="utf-8",
    )

    report, _ = evaluate_conformal_sufficiency(
        source,
        reference_run_report=reference,
        k=3,
        token_budget=100,
    )

    provenance = report["metadata"]["reference_run"]
    assert provenance["status"] == "VERIFIED_BY_HASHED_REFERENCE_REPORT"
    assert provenance["embedding_model"] == "test-embedder"
    assert provenance["generation_outputs_used"] is False
    assert len(provenance["report_sha256"]) == 64


def test_repeated_group_split_robustness_is_deterministic_and_reaggregated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())
    split_versions = (
        "test-repeat-00",
        "test-repeat-01",
        "test-repeat-02",
    )

    first = evaluate_conformal_robustness(
        source,
        split_versions=split_versions,
        k=3,
        token_budget=100,
    )
    second = evaluate_conformal_robustness(
        source,
        split_versions=split_versions,
        k=3,
        token_budget=100,
    )

    assert first == second
    assert first["metadata"]["schema_version"] == ROBUSTNESS_SCHEMA_VERSION
    assert first["metadata"]["repeat_count"] == 3
    assert [item["split_version"] for item in first["repeats"]] == list(split_versions)
    assert first["decision"]["gate_2"] == "NO-GO/SHADOW"

    json_path = tmp_path / "robustness.json"
    markdown_path = tmp_path / "robustness.md"
    write_conformal_robustness(
        first,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert load_conformal_robustness(json_path) == first
    assert "重复分组" in markdown_path.read_text(encoding="utf-8")

    first["aggregate"]["alphas"]["0.1"]["metrics"]["abstention_rate"]["mean"] += 0.01
    json_path.write_text(
        json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="aggregate"):
        load_conformal_robustness(json_path)


def test_repeated_group_split_rejects_duplicate_split_versions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())

    with pytest.raises(ValueError, match="unique"):
        evaluate_conformal_robustness(
            source,
            split_versions=("duplicate", "duplicate"),
            k=3,
            token_budget=100,
        )


def test_nested_model_selection_is_train_only_deterministic_and_reaggregated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())
    split_versions = (
        "nested-outer-00",
        "nested-outer-01",
        "nested-outer-02",
    )
    candidates = (
        {
            "candidate_id": "linear-l2-1",
            "feature_transform": "linear",
            "l2": 1.0,
        },
        {
            "candidate_id": "sqrt-l2-4",
            "feature_transform": "sqrt",
            "l2": 4.0,
        },
        {
            "candidate_id": "quadratic-l2-16",
            "feature_transform": "quadratic",
            "l2": 16.0,
        },
    )

    first = evaluate_nested_model_selection(
        source,
        split_versions=split_versions,
        candidates=candidates,
        inner_repeats=2,
        k=3,
        token_budget=100,
    )
    second = evaluate_nested_model_selection(
        source,
        split_versions=split_versions,
        candidates=candidates,
        inner_repeats=2,
        k=3,
        token_budget=100,
    )

    assert first == second
    assert first["metadata"]["schema_version"] == MODEL_SELECTION_SCHEMA_VERSION
    assert first["metadata"]["outer_repeat_count"] == 3
    assert first["metadata"]["inner_repeats"] == 2
    assert first["decision"]["gate_2"] == "NO-GO/SHADOW"
    candidate_ids = {item["candidate_id"] for item in candidates}
    for repeat in first["outer_repeats"]:
        assert repeat["selected_candidate_id"] in candidate_ids
        assert (
            len(repeat["inner_selection"]["selected_candidate"]["inner_repeats"]) == 2
        )
        assert (
            repeat["inner_selection"]["selected_candidate"]["candidate_id"]
            == repeat["selected_candidate_id"]
        )

    json_path = tmp_path / "selection.json"
    markdown_path = tmp_path / "selection.md"
    write_nested_model_selection(
        first,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert load_nested_model_selection(json_path) == first
    assert "outer evaluation" in markdown_path.read_text(encoding="utf-8").lower()

    first["aggregate"]["nested_selected"]["abstention_rate"]["mean"] += 0.01
    json_path.write_text(
        json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="aggregate"):
        load_nested_model_selection(json_path)


def test_nested_model_selection_rejects_unregistered_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())

    with pytest.raises(ValueError, match="unsupported feature transform"):
        evaluate_nested_model_selection(
            source,
            split_versions=("outer-a", "outer-b"),
            candidates=(
                {
                    "candidate_id": "invalid",
                    "feature_transform": "evaluation-tuned",
                    "l2": 1.0,
                },
            ),
            inner_repeats=2,
            k=3,
            token_budget=100,
        )


def test_cross_dataset_confirmation_is_untuned_deterministic_and_reaggregated(
    tmp_path: Path,
) -> None:
    reference_source = tmp_path / "reference_scores.jsonl"
    confirmation_source = tmp_path / "confirmation_scores.jsonl"
    reference_rows = [{**row, "dataset": "ReferenceQA"} for row in _rows()]
    confirmation_rows = [{**row, "dataset": "ConfirmationQA"} for row in _rows()]
    _write_rows(reference_source, reference_rows)
    _write_rows(confirmation_source, confirmation_rows)
    split_versions = (
        "cross-dataset-00",
        "cross-dataset-01",
        "cross-dataset-02",
    )
    reference_report = evaluate_conformal_robustness(
        reference_source,
        split_versions=split_versions,
        k=3,
        token_budget=100,
    )
    reference_json = tmp_path / "reference_robustness.json"
    reference_markdown = tmp_path / "reference_robustness.md"
    write_conformal_robustness(
        reference_report,
        json_path=reference_json,
        markdown_path=reference_markdown,
    )

    first = evaluate_cross_dataset_confirmation(
        reference_json,
        confirmation_source,
        split_versions=split_versions,
        alpha_control_fraction=2 / 3,
        max_mean_case_family_rate=1.0,
    )
    second = evaluate_cross_dataset_confirmation(
        reference_json,
        confirmation_source,
        split_versions=split_versions,
        alpha_control_fraction=2 / 3,
        max_mean_case_family_rate=1.0,
    )

    assert first == second
    assert first["metadata"]["schema_version"] == CROSS_DATASET_SCHEMA_VERSION
    assert first["metadata"]["reference_dataset"] == "ReferenceQA"
    assert first["metadata"]["confirmation_dataset"] == "ConfirmationQA"
    assert (
        first["pre_registered_hypotheses"]["no_confirmation_feature_or_model_selection"]
        is True
    )
    assert (
        first["outcome"]["checks"]["confirmation_dataset_differs_from_reference"]
        is True
    )

    json_path = tmp_path / "confirmation.json"
    markdown_path = tmp_path / "confirmation.md"
    write_cross_dataset_confirmation(
        first,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    assert (
        load_cross_dataset_confirmation(
            json_path, reference_robustness_path=reference_json
        )
        == first
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "跨数据集" in markdown
    assert "ConfirmationQA" in markdown
    assert (
        first["metadata"]["status"]
        == "RUN_PUBLIC_REAL_MODEL_CONFIRMATIONQA_CONFIRMATION"
    )

    first["outcome"]["cross_dataset_safety_signal_confirmed"] = not first["outcome"][
        "cross_dataset_safety_signal_confirmed"
    ]
    json_path.write_text(
        json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outcome"):
        load_cross_dataset_confirmation(
            json_path, reference_robustness_path=reference_json
        )


def test_mcnemar_exact_is_stable_for_large_discordant_counts() -> None:
    variants = [{"complete": False} for _ in range(2400)]
    baseline = np.ones(2400, dtype=bool)
    conformal = np.zeros(2400, dtype=bool)

    large = _mcnemar_exact(variants, baseline, conformal)
    assert large["discordant_pairs"] == 2400
    assert math.isfinite(large["two_sided_exact_p_value"])
    assert 0.0 <= large["two_sided_exact_p_value"] <= 1.0

    small_variants = [{"complete": False} for _ in range(10)]
    small_baseline = np.asarray([True] * 8 + [False] * 2)
    small_conformal = np.asarray([False] * 8 + [True] * 2)
    small = _mcnemar_exact(
        small_variants,
        small_baseline,
        small_conformal,
    )
    expected = 2.0 * sum(math.comb(10, k) for k in range(3)) / (2**10)
    assert small["two_sided_exact_p_value"] == pytest.approx(expected)


def test_cross_dataset_series_recomputes_source_report_hashes(
    tmp_path: Path,
) -> None:
    split_versions = ("series-00", "series-01", "series-02")
    reference_source = tmp_path / "reference.jsonl"
    _write_rows(
        reference_source,
        [{**row, "dataset": "ReferenceQA"} for row in _rows()],
    )
    reference_report = evaluate_conformal_robustness(
        reference_source,
        split_versions=split_versions,
        k=3,
        token_budget=100,
    )
    reference_json = tmp_path / "reference.json"
    write_conformal_robustness(
        reference_report,
        json_path=reference_json,
        markdown_path=tmp_path / "reference.md",
    )

    confirmation_paths: list[Path] = []
    for dataset in ("ConfirmationA", "ConfirmationB"):
        source = tmp_path / f"{dataset}.jsonl"
        _write_rows(
            source,
            [{**row, "dataset": dataset} for row in _rows()],
        )
        confirmation = evaluate_cross_dataset_confirmation(
            reference_json,
            source,
            split_versions=split_versions,
            alpha_control_fraction=2 / 3,
            max_mean_case_family_rate=1.0,
        )
        confirmation_path = tmp_path / f"{dataset}.json"
        write_cross_dataset_confirmation(
            confirmation,
            json_path=confirmation_path,
            markdown_path=tmp_path / f"{dataset}.md",
        )
        confirmation_paths.append(confirmation_path)

    path_tuple = tuple(confirmation_paths)
    series = evaluate_cross_dataset_series(
        path_tuple,
        reference_robustness_path=reference_json,
    )
    assert series["metadata"]["confirmation_dataset_count"] == 2
    assert [item["dataset"] for item in series["confirmations"]] == [
        "ConfirmationA",
        "ConfirmationB",
    ]
    series_json = tmp_path / "series.json"
    write_cross_dataset_series(
        series,
        json_path=series_json,
        markdown_path=tmp_path / "series.md",
    )
    assert (
        load_cross_dataset_series(
            series_json,
            confirmation_paths=path_tuple,
            reference_robustness_path=reference_json,
        )
        == series
    )

    tampered = json.loads(confirmation_paths[1].read_text(encoding="utf-8"))
    tampered["decision"]["finding"] = "tampered"
    confirmation_paths[1].write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="series"):
        load_cross_dataset_series(
            series_json,
            confirmation_paths=path_tuple,
            reference_robustness_path=reference_json,
        )


def test_source_preparation_benchmark_requires_exact_equivalence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scores.jsonl"
    _write_rows(source, _rows())

    report = benchmark(source, reference_run_report=None)

    comparison = report["comparison"]
    assert comparison["results_identical"] is True
    assert comparison["legacy"]["jsonl_parse_passes"] == 10
    assert comparison["single_parse"]["jsonl_parse_passes"] == 1
    assert (
        comparison["legacy"]["canonical_repeat_sha256"]
        == comparison["single_parse"]["canonical_repeat_sha256"]
    )


def test_observable_subgroup_audit_reaggregates_case_artifact(
    tmp_path: Path,
) -> None:
    split_versions = ("subgroup-00", "subgroup-01", "subgroup-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset in ("DatasetA", "DatasetB"):
        source = tmp_path / f"{dataset}.jsonl"
        rows = [
            {
                **row,
                "dataset": dataset,
                "question_type": ("comparison" if index % 2 else "inference"),
            }
            for index, row in enumerate(_rows())
        ]
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    report, records = evaluate_conformal_subgroup_audit(
        sources,
        expected,
        split_versions=split_versions,
        min_incomplete_case_families=5,
        min_eligible_repeats=2,
        alpha=0.1,
    )
    assert report["metadata"]["subgroup_dimensions"] == [
        "question_type",
        "required_role_count",
        "candidate_count_bucket",
    ]
    assert (
        report["theoretical_scope"]["gold_fields_used_for_subgroup_assignment"] is False
    )
    assert report["outcome"]["conditional_subgroup_guarantee_claimed"] is False
    assert all(
        item["robustness_repeats_exactly_reproduced"] for item in report["provenance"]
    )
    assert records
    assert (
        not {
            "gold_evidence_ids",
            "gold_roles",
            "removed_gold_count",
        }
        & records[0].keys()
    )

    json_path = tmp_path / "subgroup.json"
    markdown_path = tmp_path / "subgroup.md"
    cases_path = tmp_path / "subgroup-cases.jsonl.gz"
    write_conformal_subgroup_audit(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_conformal_subgroup_audit(json_path, cases_path) == report
    assert "可观察子群" in markdown_path.read_text(encoding="utf-8")
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }
    write_conformal_subgroup_audit(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert first_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_conformal_subgroup_audit(json_path, cases_path)


def test_hierarchical_mondrian_is_calibration_only_and_reaggregated(
    tmp_path: Path,
) -> None:
    split_versions = ("mondrian-00", "mondrian-01", "mondrian-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset in ("DatasetA", "DatasetB"):
        source = tmp_path / f"{dataset}.jsonl"
        rows = [
            {
                **row,
                "dataset": dataset,
                "question_type": ("comparison" if index % 2 else "inference"),
            }
            for index, row in enumerate(_rows())
        ]
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    report, records = evaluate_hierarchical_mondrian(
        sources,
        expected,
        split_versions=split_versions,
        min_group_calibration_units=5,
        min_incomplete_case_families=5,
        min_eligible_repeats=2,
    )

    assert report["metadata"]["schema_version"] == MONDRIAN_SCHEMA_VERSION
    assert report["development_protocol"]["post_hoc"] is True
    assert report["development_protocol"]["independent_confirmation"] is False
    assert (
        report["development_protocol"][
            "evaluation_used_for_threshold_or_fallback_selection"
        ]
        is False
    )
    assert report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["outcome"]["checks"]["all_threshold_sources_calibration_only"][
        "passed"
    ]
    assert records
    assert all(
        record["mondrian_threshold_source"]
        in {"joint", "question_type", "required_role_count", "global"}
        for record in records
    )
    assert (
        not {
            "gold_evidence_ids",
            "gold_roles",
            "removed_gold_count",
        }
        & records[0].keys()
    )

    json_path = tmp_path / "mondrian.json"
    markdown_path = tmp_path / "mondrian.md"
    cases_path = tmp_path / "mondrian-cases.jsonl.gz"
    write_hierarchical_mondrian(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_hierarchical_mondrian(json_path, cases_path) == report
    assert "事后方法开发" in markdown_path.read_text(encoding="utf-8")
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }
    write_hierarchical_mondrian(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert first_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_hierarchical_mondrian(json_path, cases_path)


def test_multi_axis_mondrian_uses_frozen_hierarchy_and_reaggregates(
    tmp_path: Path,
) -> None:
    split_versions = ("multi-axis-00", "multi-axis-01", "multi-axis-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset in ("DatasetA", "DatasetB"):
        source = tmp_path / f"{dataset}.jsonl"
        rows = [
            {
                **row,
                "dataset": dataset,
                "question_type": ("comparison" if index % 2 else "inference"),
            }
            for index, row in enumerate(_rows())
        ]
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    report, records = evaluate_multi_axis_mondrian(
        sources,
        expected,
        split_versions=split_versions,
        min_group_calibration_units=1,
        min_incomplete_case_families=5,
        min_eligible_repeats=2,
    )

    assert report["metadata"]["schema_version"] == MULTI_AXIS_MONDRIAN_SCHEMA_VERSION
    assert report["metadata"]["fallback_order"] == list(MULTI_AXIS_FALLBACK_ORDER)
    assert report["development_protocol"]["candidate_size_aware"] is True
    assert report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert records
    assert all(
        record["mondrian_threshold_source"] == MULTI_AXIS_FALLBACK_ORDER[0]
        for record in records
    )
    assert all(
        "candidate_count_bucket=" in record["mondrian_threshold_key"]
        for record in records
    )

    json_path = tmp_path / "multi-axis.json"
    markdown_path = tmp_path / "multi-axis.md"
    cases_path = tmp_path / "multi-axis-cases.jsonl.gz"
    write_multi_axis_mondrian(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_multi_axis_mondrian(json_path, cases_path) == report
    assert "多轴 Mondrian" in markdown_path.read_text(encoding="utf-8")

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_multi_axis_mondrian(json_path, cases_path)


def test_cross_dataset_head_transfer_excludes_target_train_and_reaggregates(
    tmp_path: Path,
) -> None:
    split_versions = ("transfer-00", "transfer-01", "transfer-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset_index, dataset in enumerate(
        ("DatasetA", "DatasetB", "DatasetC", "DatasetD")
    ):
        source = tmp_path / f"{dataset}.jsonl"
        rows = [
            {
                **row,
                "dataset": dataset,
                "question_type": (
                    "comparison" if (index + dataset_index) % 2 else "inference"
                ),
            }
            for index, row in enumerate(_rows())
        ]
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    prepared = {
        dataset: prepare_conformal_source(
            source,
            k=3,
            token_budget=100,
        )
        for dataset, source in sources.items()
    }
    original_model, original_audit = fit_transferred_head(
        prepared,
        target_dataset="DatasetA",
        split_version=split_versions[0],
    )
    mutated = dict(prepared)
    mutated["DatasetA"] = {
        **prepared["DatasetA"],
        "variants": [
            {
                **item,
                "complete": not bool(item["complete"]),
                "features": np.full_like(item["features"], 999.0),
            }
            for item in prepared["DatasetA"]["variants"]
        ],
    }
    mutated_model, mutated_audit = fit_transferred_head(
        mutated,
        target_dataset="DatasetA",
        split_version=split_versions[0],
    )
    for key in ("means", "scales", "weights"):
        np.testing.assert_array_equal(original_model[key], mutated_model[key])
    assert original_model["intercept"] == mutated_model["intercept"]
    assert original_audit == mutated_audit
    assert original_audit["target_train_labels_used"] is False
    assert original_audit["target_train_features_used"] is False

    report, records = evaluate_cross_dataset_head_transfer(
        sources,
        expected,
        split_versions=split_versions,
    )
    assert report["metadata"]["schema_version"] == HEAD_TRANSFER_SCHEMA_VERSION
    assert report["metadata"]["dataset_count"] == 4
    assert (
        report["development_protocol"][
            "target_train_labels_or_features_used_for_transferred_head"
        ]
        is False
    )
    assert report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert records
    assert all(record["target_train_used_for_head"] is False for record in records)

    json_path = tmp_path / "head-transfer.json"
    markdown_path = tmp_path / "head-transfer.md"
    cases_path = tmp_path / "head-transfer-cases.jsonl.gz"
    write_cross_dataset_head_transfer(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_cross_dataset_head_transfer(json_path, cases_path) == report
    assert "目标 train" in markdown_path.read_text(encoding="utf-8")

    certificate = issue_transfer_certificate(
        {
            "incomplete_case_families": 40,
            "complete_variants": 35,
            "incomplete_variants": 120,
            "auc": 0.8,
            "threshold": 0.7,
            "complete_recall_at_threshold": 0.2,
        }
    )
    assert certificate["certified"] is True
    assert certificate["evaluation_fields_used"] is False
    assert certificate["target_train_fields_used"] is False
    assert not {
        "false_complete_case_family_rate",
        "true_complete_declaration_rate",
    } & set(certificate["input_fields"])

    guard_report, guard_evidence = evaluate_transfer_admission_guard(
        json_path,
        cases_path,
    )
    assert len(guard_evidence) == 12
    assert all(
        item["certificate"]["evaluation_fields_used"] is False
        and item["certificate"]["target_train_fields_used"] is False
        for item in guard_evidence
    )
    guard_json = tmp_path / "guard.json"
    guard_markdown = tmp_path / "guard.md"
    guard_cases = tmp_path / "guard-evidence.jsonl.gz"
    write_transfer_admission_guard(
        guard_report,
        guard_evidence,
        json_path=guard_json,
        markdown_path=guard_markdown,
        evidence_path=guard_cases,
    )
    assert load_transfer_admission_guard(guard_json, guard_cases) == guard_report
    assert "准入守卫" in guard_markdown.read_text(encoding="utf-8")

    guard_report["outcome"]["status"] = "tampered"
    guard_json.write_text(
        json.dumps(guard_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_transfer_admission_guard(guard_json, guard_cases)

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_cross_dataset_head_transfer(json_path, cases_path)


def test_contextual_head_uses_train_vocabulary_and_reaggregates(
    tmp_path: Path,
) -> None:
    split_versions = ("context-00", "context-01", "context-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset in ("DatasetA", "DatasetB"):
        source = tmp_path / f"{dataset}.jsonl"
        rows = [
            {
                **row,
                "dataset": dataset,
                "question_type": ("comparison" if index % 2 else "inference"),
            }
            for index, row in enumerate(_rows())
        ]
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    report, records = evaluate_contextual_conformal(
        sources,
        expected,
        split_versions=split_versions,
        min_train_category_cases=5,
        min_incomplete_case_families=5,
        min_eligible_repeats=2,
    )

    assert report["metadata"]["schema_version"] == CONTEXTUAL_SCHEMA_VERSION
    assert report["development_protocol"]["post_hoc"] is True
    assert report["development_protocol"]["independent_confirmation"] is False
    assert (
        report["development_protocol"]["context_vocabulary_source"]
        == "train_case_families_only"
    )
    assert (
        report["development_protocol"][
            "context_feature_or_model_selection_on_evaluation"
        ]
        is False
    )
    assert report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["outcome"]["checks"]["context_vocabulary_train_only"]["passed"]
    assert records
    assert (
        not {
            "gold_evidence_ids",
            "gold_roles",
            "removed_gold_count",
        }
        & records[0].keys()
    )
    for dataset_repeats in report["training_repeats"].values():
        for repeat in dataset_repeats:
            assert repeat["context_encoder"]["source_split"] == "train"
            assert repeat["calibration"]["threshold_uses_calibration_only"]

    json_path = tmp_path / "contextual.json"
    markdown_path = tmp_path / "contextual.md"
    cases_path = tmp_path / "contextual-cases.jsonl.gz"
    write_contextual_conformal(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_contextual_conformal(json_path, cases_path) == report
    assert "train-only" in markdown_path.read_text(encoding="utf-8")
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }
    write_contextual_conformal(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert first_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_path, markdown_path, cases_path)
    }

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_contextual_conformal(json_path, cases_path)


def test_score_stability_head_is_gold_free_target_fitted_and_reaggregates(
    tmp_path: Path,
) -> None:
    row = _rows(1)[0]
    selected = select_precomputed(row, "frc_select", k=3, budget=100)
    original = extract_score_stability_features(row, selected, k=3, token_budget=100)
    mutated = {
        **row,
        "gold_evidence_ids": ["invented"],
        "candidates": [
            {**candidate, "gold": not candidate["gold"], "gold_roles": ["invented"]}
            for candidate in row["candidates"]
        ],
    }
    mutated_selected = [
        next(
            candidate
            for candidate in mutated["candidates"]
            if candidate["id"] == selected_candidate["id"]
        )
        for selected_candidate in selected
    ]
    assert np.array_equal(
        original,
        extract_score_stability_features(
            mutated, mutated_selected, k=3, token_budget=100
        ),
    )
    assert len(original) == len(AUGMENTED_FEATURE_NAMES) == 64

    split_versions = ("stability-00", "stability-01", "stability-02")
    sources: dict[str, Path] = {}
    expected: dict[str, tuple[Path, dict]] = {}
    for dataset_index, dataset in enumerate(("DatasetA", "DatasetB")):
        source = tmp_path / f"{dataset}.jsonl"
        rows = []
        for index, source_row in enumerate(_rows()):
            candidates = []
            for candidate_index, candidate in enumerate(source_row["candidates"]):
                candidates.append(
                    {
                        **candidate,
                        "scores": {
                            **candidate["scores"],
                            "bm25": candidate["scores"]["bm25"]
                            + (candidate_index % 2) * 0.03,
                            "dense": candidate["scores"]["dense"]
                            + ((index + dataset_index) % 3) * 0.01,
                            "hybrid": candidate["scores"]["hybrid"]
                            - (candidate_index % 3) * 0.02,
                        },
                    }
                )
            rows.append(
                {
                    **source_row,
                    "dataset": dataset,
                    "question_type": ("comparison" if index % 2 else "inference"),
                    "candidates": candidates,
                }
            )
        _write_rows(source, rows)
        robustness = evaluate_conformal_robustness(
            source,
            split_versions=split_versions,
            k=3,
            token_budget=100,
        )
        robustness_path = tmp_path / f"{dataset}-robustness.json"
        write_conformal_robustness(
            robustness,
            json_path=robustness_path,
            markdown_path=tmp_path / f"{dataset}-robustness.md",
        )
        sources[dataset] = source
        expected[dataset] = (robustness_path, robustness)

    report, records = evaluate_score_stability_conformal(
        sources,
        expected,
        split_versions=split_versions,
        min_incomplete_case_families=5,
        min_eligible_repeats=2,
    )
    assert report["metadata"]["schema_version"] == SCORE_STABILITY_SCHEMA_VERSION
    assert report["metadata"]["base_feature_count"] == 37
    assert report["metadata"]["score_stability_feature_count"] == 27
    assert report["metadata"]["augmented_feature_count"] == 64
    assert report["development_protocol"]["target_fitted_head_required"] is True
    assert (
        report["development_protocol"][
            "feature_or_hyperparameter_selection_on_evaluation"
        ]
        is False
    )
    assert report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert records
    assert (
        not {
            "gold",
            "gold_evidence_ids",
            "gold_roles",
            "removed_gold_count",
        }
        & records[0].keys()
    )
    assert all(record["target_train_fitted"] for record in records)
    assert all(record["threshold_calibration_only"] for record in records)
    for dataset_repeats in report["training_repeats"].values():
        for repeat in dataset_repeats:
            assert repeat["model"]["scaler_source"] == "target train split only"
            assert repeat["calibration"]["threshold_uses_calibration_only"]

    json_path = tmp_path / "score-stability.json"
    markdown_path = tmp_path / "score-stability.md"
    cases_path = tmp_path / "score-stability-cases.jsonl.gz"
    write_score_stability_conformal(
        report,
        records,
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_score_stability_conformal(json_path, cases_path) == report

    review_report, review_evidence = evaluate_review_ranking(
        report,
        records,
        source_artifact={
            "report_path_label": json_path.name,
            "report_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
            "cases_path_label": cases_path.name,
            "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "case_record_count": len(records),
        },
        min_subgroup_pool_variants=5,
        min_subgroup_complete_variants=1,
        min_subgroup_supported_repeats=2,
    )
    assert review_report["metadata"]["schema_version"] == REVIEW_RANKING_SCHEMA_VERSION
    assert review_report["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert review_report["development_protocol"]["review_only"] is True
    assert review_evidence
    assert all(
        item["automatic_decision_before"] == item["automatic_decision_after"]
        for item in review_evidence
    )
    assert all(
        not item["review_pool"]
        and not any(any(methods.values()) for methods in item["reviewed"].values())
        for item in review_evidence
        if item["automatic_decision_before"]
    )
    for dataset in review_report["datasets"]:
        for repeat in dataset["per_repeat"]:
            for budget in repeat["budgets"].values():
                assert (
                    budget["baseline"]["reviewed_count"]
                    == budget["score_stability"]["reviewed_count"]
                )

    review_json = tmp_path / "review-ranking.json"
    review_markdown = tmp_path / "review-ranking.md"
    review_cases = tmp_path / "review-ranking-evidence.jsonl.gz"
    write_review_ranking(
        review_report,
        review_evidence,
        json_path=review_json,
        markdown_path=review_markdown,
        evidence_path=review_cases,
    )
    assert load_review_ranking(review_json, review_cases) == review_report
    assert "review-only" in review_markdown.read_text(encoding="utf-8")
    review_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (review_json, review_markdown, review_cases)
    }
    write_review_ranking(
        review_report,
        review_evidence,
        json_path=review_json,
        markdown_path=review_markdown,
        evidence_path=review_cases,
    )
    assert review_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (review_json, review_markdown, review_cases)
    }
    tampered_record = next(item for item in review_evidence if item["review_pool"])
    original_rank = tampered_record["baseline_review_rank"]
    tampered_record["baseline_review_rank"] = int(original_rank) + 1
    write_review_ranking(
        review_report,
        review_evidence,
        json_path=review_json,
        markdown_path=review_markdown,
        evidence_path=review_cases,
    )
    with pytest.raises(ValueError, match="frozen boundaries"):
        load_review_ranking(review_json, review_cases)
    tampered_record["baseline_review_rank"] = original_rank
    write_review_ranking(
        review_report,
        review_evidence,
        json_path=review_json,
        markdown_path=review_markdown,
        evidence_path=review_cases,
    )
    review_report["outcome"]["status"] = "tampered"
    review_json.write_text(
        json.dumps(review_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_review_ranking(review_json, review_cases)
    assert "分数稳定性" in markdown_path.read_text(encoding="utf-8")

    report["outcome"]["status"] = "tampered"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregates"):
        load_score_stability_conformal(json_path, cases_path)
