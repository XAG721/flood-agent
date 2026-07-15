from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.lawshift_temporal_ablation import (
    LAWSHIFT_METHODS,
    build_lawshift_ablation_report,
    build_lawshift_temporal_cases,
    build_source_manifest,
    select_lawshift_evidence,
    select_lawshift_methods,
)
from research.frc_rag.public_evidence import load_lawshift_temporal_ablation


def _write_fixture(root: Path) -> Path:
    revision = root / "toy_revision"
    revision.mkdir(parents=True)
    original_articles = {
        "1-0-0": "旧法规定，红色行为属于本罪。",
        "2-0-0": "其他法条处理蓝色行为。",
        "3-0-0": "另一个法条处理绿色行为。",
    }
    revised_articles = {
        **original_articles,
        "1-0-0": "新法规定，黄色行为属于本罪。",
    }
    original_cases = [
        {
            "fact": "被告实施红色行为。",
            "relevant_articles": ["1-0-0"],
            "charge": "不可进入检索提示的标签",
            "prison_time": 1,
        }
    ]
    revised_cases = [
        {
            "fact": "被告实施黄色行为。",
            "relevant_articles": ["1-0-0"],
            "charge": "不可进入检索提示的标签",
            "prison_time": 1,
        }
    ]
    for name, payload in (
        ("articles_original.json", original_articles),
        ("articles_poisoned.json", revised_articles),
        ("original.json", original_cases),
        ("poisoned.json", revised_cases),
    ):
        (revision / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return root


def test_lawshift_pairs_preserve_labels_without_charge_leakage(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)

    cases = build_lawshift_temporal_cases(
        root,
        revision_types=("toy_revision",),
        pairs_per_revision=1,
        distractor_articles=1,
        seed=7,
    )

    assert len(cases) == 2
    assert {case["as_of_version"] for case in cases} == {"original", "revised"}
    assert all(len(case["candidates"]) == 4 for case in cases)
    assert all(case["gold_article_id"] == "1-0-0" for case in cases)
    assert all("不可进入检索提示的标签" not in case["question"] for case in cases)
    assert all(
        {candidate["version"] for candidate in case["candidates"]}
        == {"original", "revised"}
        for case in cases
    )


def _scored_cases(cases: list[dict]) -> list[dict]:
    output = []
    for case in cases:
        candidates = []
        wrong_version = "revised" if case["as_of_version"] == "original" else "original"
        for candidate in case["candidates"]:
            target = candidate["article_id"] == case["gold_article_id"]
            cross = 0.1
            if target and candidate["version"] == wrong_version:
                cross = 1.0
            elif target:
                cross = 0.9
            candidates.append(
                {
                    **candidate,
                    "scores": {
                        "char_bm25": cross,
                        "cross_encoder": cross,
                    },
                }
            )
        output.append({**case, "candidates": candidates})
    return output


def test_temporal_ablation_removes_only_version_filter(tmp_path: Path) -> None:
    cases = _scored_cases(
        build_lawshift_temporal_cases(
            _write_fixture(tmp_path),
            revision_types=("toy_revision",),
            pairs_per_revision=1,
            distractor_articles=1,
            seed=7,
        )
    )

    for case in cases:
        full = select_lawshift_evidence(case, "frc_full")
        filtered = select_lawshift_evidence(
            case, "applicability_filtered_cross_encoder_top1"
        )
        without = select_lawshift_evidence(case, "w/o_applicability")
        assert full[0]["id"] == case["gold_evidence_id"]
        assert filtered[0]["id"] == case["gold_evidence_id"]
        assert without[0]["article_id"] == case["gold_article_id"]
        assert without[0]["version"] != case["as_of_version"]


def test_report_keeps_fair_filtered_baseline_and_no_go(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    cases = _scored_cases(
        build_lawshift_temporal_cases(
            root,
            revision_types=("toy_revision",),
            pairs_per_revision=1,
            distractor_articles=1,
            seed=7,
        )
    )
    selected = select_lawshift_methods(cases)
    scored_path = tmp_path / "scored.jsonl"
    scored_path.write_text("{}\n", encoding="utf-8")

    report = build_lawshift_ablation_report(
        cases=cases,
        selected_rows=selected,
        config={
            "reranker_model": "fixture-reranker",
            "top_k": 1,
            "token_budget": 512,
            "pairs_per_revision": 1,
            "distractor_articles": 1,
            "seed": 7,
        },
        source_manifest=build_source_manifest(
            root, revision_types=("toy_revision",)
        ),
        scored_path=scored_path,
    )

    assert set(report["aggregates"]) == set(LAWSHIFT_METHODS)
    assert report["aggregates"]["frc_full"]["exact_evidence_accuracy"] == 1.0
    assert report["aggregates"]["w/o_applicability"]["version_accuracy"] == 0.0
    assert report["strongest_baseline_by_exact_evidence_accuracy"] == (
        "applicability_filtered_cross_encoder_top1"
    )
    assert report["decision"]["full_strictly_better_than_w_o_applicability"] is True
    assert report["decision"]["full_exact_gain_over_strongest_baseline_at_least_0_05"] is False
    assert report["decision"]["gate_2"] == "NO-GO"


def test_public_loader_reaggregates_committed_lawshift_ablation(tmp_path: Path) -> None:
    source = Path(
        "output/rag_evaluation/lawshift_temporal_ablation/"
        "lawshift_temporal_ablation.json"
    )
    loaded = load_lawshift_temporal_ablation(source)

    assert loaded["status"] == "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL"
    assert loaded["metadata"]["case_count"] == 124
    assert loaded["metadata"]["revision_type_count"] == 31
    assert loaded["strongest_baseline"] == (
        "applicability_filtered_cross_encoder_top1"
    )
    assert loaded["decision"]["gate_2"] == "NO-GO"

    report = json.loads(source.read_text(encoding="utf-8"))
    report["aggregates"]["frc_full"]["exact_evidence_accuracy"] = 0.123456
    case_file = report["case_results_artifact"]["file"]
    (tmp_path / case_file).write_bytes((source.parent / case_file).read_bytes())
    artifact = tmp_path / "tampered-lawshift-ablation.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="LawShift aggregate mismatch"):
        load_lawshift_temporal_ablation(artifact)
