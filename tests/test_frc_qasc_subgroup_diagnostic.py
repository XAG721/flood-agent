from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.qasc_subgroup_diagnostic import (
    analyze_qasc_question_types,
    load_qasc_subgroup_diagnostic,
    write_qasc_subgroup_diagnostic,
)


SPLITS = tuple(f"split-{index:02d}" for index in range(10))


def _records() -> list[dict]:
    records = []
    for split_version in SPLITS:
        for question_type, false_complete_count in (
            ("qasc_what", 2),
            ("qasc_other", 8),
        ):
            for index in range(40):
                records.append(
                    {
                        "split_version": split_version,
                        "case_id": f"{question_type}-{index:02d}",
                        "question_type": question_type,
                        "complete": False,
                        "conformal_declared_complete": (
                            index < false_complete_count
                        ),
                    }
                )
    return records


def test_qasc_question_type_diagnostic_is_descriptive_and_deterministic() -> None:
    first = analyze_qasc_question_types(_records(), split_versions=SPLITS)
    second = analyze_qasc_question_types(
        list(reversed(_records())), split_versions=SPLITS
    )
    assert first == second
    assert first["outcome"]["status"] == "DESCRIPTIVE_HETEROGENEITY_DETECTED"
    assert first["outcome"]["eligible_question_type_count"] == 2
    assert first["outcome"]["worst_eligible_question_type"][
        "question_type"
    ] == "qasc_other"
    assert first["outcome"]["worst_eligible_question_type"][
        "false_complete_case_family_rate_mean"
    ] == 0.2


def test_qasc_question_type_artifact_reaggregates_and_detects_tampering(
    tmp_path: Path,
) -> None:
    analysis = analyze_qasc_question_types(_records(), split_versions=SPLITS)
    report = {
        "metadata": {
            "schema_version": "frc-qasc-question-type-diagnostic-v1",
            "alpha": 0.1,
            "split_versions": list(SPLITS),
            "minimum_incomplete_case_families": 30,
            "minimum_eligible_repeats": 7,
            "case_artifact": {},
        },
        "analysis": analysis,
        "decision": {"gate_2": "NO-GO/SHADOW"},
    }
    json_path = tmp_path / "diagnostic.json"
    markdown_path = tmp_path / "diagnostic.md"
    cases_path = tmp_path / "cases.jsonl.gz"
    write_qasc_subgroup_diagnostic(
        report,
        _records(),
        json_path=json_path,
        markdown_path=markdown_path,
        cases_path=cases_path,
    )
    assert load_qasc_subgroup_diagnostic(json_path, cases_path) == report
    tampered = json.loads(json_path.read_text(encoding="utf-8"))
    tampered["analysis"]["outcome"]["status"] = "TAMPERED"
    json_path.write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not reaggregate"):
        load_qasc_subgroup_diagnostic(json_path, cases_path)
