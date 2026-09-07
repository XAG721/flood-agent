from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.conflicts_expected_behavior import (
    SUBMISSION_SCHEMA_VERSION,
    answer_generation_prompt,
    blank_submission_rows,
    build_blinded_package,
    canonical_json_sha256,
    compare_submissions,
    finalize_adjudication,
    generate_answers,
    validate_submission,
    write_gzip_jsonl,
)


def _selected_rows() -> list[dict]:
    common = {
        "source": "demo",
        "conflict_type": "Conflict due to outdated information",
        "correct_answer": "new answer",
        "candidate_count": 3,
    }
    evidence = {
        "id": "doc-1",
        "title": "Source one",
        "url": "https://example.test/one",
        "date": "2025-01-01",
        "text": "The current answer is new answer.",
        "scores": {"cross_encoder": 0.99},
        "role_scores": {"temporal_validity": 0.91},
    }
    rows = []
    for case_id, question in (("case-1", "What is current?"), ("case-2", "What changed?")):
        for method in ("coverage_greedy_proxy", "frc_select"):
            selected = [evidence]
            if case_id == "case-2" and method == "frc_select":
                selected = [
                    {
                        **evidence,
                        "id": "doc-2",
                        "title": "Source two",
                        "url": "https://example.test/two",
                    }
                ]
            rows.append(
                {
                    **common,
                    "case_id": case_id,
                    "question": question,
                    "method": method,
                    "selected_ids": [item["id"] for item in selected],
                    "selected_evidence": selected,
                }
            )
    return rows


class _FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_prompts(self, prompts: list[str]) -> list[str]:
        self.prompts.extend(prompts)
        return [f"Grounded response {len(self.prompts) - len(prompts) + index} [1]" for index in range(len(prompts))]


def _completed_submission(package_id: str, items: list[dict], annotator: str) -> dict:
    decisions = blank_submission_rows(items)
    for decision in decisions:
        for rating in decision["ratings"].values():
            rating.update(
                {
                    "expected_behavior_adherence": "PASS",
                    "factual_grounding": "PASS",
                    "citation_correctness": "PASS",
                    "answer_correctness": "PASS",
                    "rationale": "The response follows the evidence and expected behavior.",
                }
            )
        decision["preference"] = "TIE"
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "package_id": package_id,
        "annotator_id": annotator,
        "decisions": decisions,
    }


def test_generation_prompt_is_gold_and_method_blind() -> None:
    row = _selected_rows()[0]
    changed = {
        **row,
        "method": "hidden-other-method",
        "conflict_type": "No conflict",
        "correct_answer": "secret gold answer",
    }

    prompt = answer_generation_prompt(row)

    assert prompt == answer_generation_prompt(changed)
    assert "Conflict due to outdated information" not in prompt
    assert "new answer" in prompt  # It appears only in the public selected source text.
    assert "coverage_greedy_proxy" not in prompt
    assert "cross_encoder" not in prompt
    assert "temporal_validity" not in prompt


def test_generation_deduplicates_equal_selections_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "generations.jsonl"
    generator = _FakeGenerator()

    first = generate_answers(
        _selected_rows(),
        generator=generator,
        output_path=output,
        cache_key="frozen-cache-key",
        prompt_batch_size=2,
    )

    assert len(first) == 4
    assert len(generator.prompts) == 3
    assert first[1]["reused_from_equivalent_selection"] is True
    assert all(item["citation_indices"] == [1] for item in first)
    assert all("conflict_type" not in item and "correct_answer" not in item for item in first)

    resumed_generator = _FakeGenerator()
    second = generate_answers(
        _selected_rows(),
        generator=resumed_generator,
        output_path=output,
        cache_key="frozen-cache-key",
        prompt_batch_size=2,
    )
    assert resumed_generator.prompts == []
    assert second == first


def test_blinded_package_is_deterministic_and_removes_method_scores(tmp_path: Path) -> None:
    generator = _FakeGenerator()
    generations = generate_answers(
        _selected_rows(),
        generator=generator,
        output_path=tmp_path / "generation.jsonl",
        cache_key="key",
        prompt_batch_size=8,
    )

    first = build_blinded_package(_selected_rows(), generations, blind_seed="blind-v1")
    second = build_blinded_package(_selected_rows(), generations, blind_seed="blind-v1")

    assert first == second
    package_id, items, mapping = first
    encoded_items = json.dumps(items, ensure_ascii=False)
    assert package_id.startswith("CONFLICTS-BEHAVIOR-")
    assert "coverage_greedy_proxy" not in encoded_items
    assert "frc_select" not in encoded_items
    assert '"scores"' not in encoded_items
    assert '"role_scores"' not in encoded_items
    assert {value["method"] for item in mapping["items"] for value in item["aliases"].values()} == {
        "coverage_greedy_proxy",
        "frc_select",
    }

    left = write_gzip_jsonl(tmp_path / "first.jsonl.gz", items)
    right = write_gzip_jsonl(tmp_path / "second.jsonl.gz", items)
    assert left.read_bytes() == right.read_bytes()


def test_independent_comparison_and_adjudication_unblind_only_at_finalize(
    tmp_path: Path,
) -> None:
    generations = generate_answers(
        _selected_rows(),
        generator=_FakeGenerator(),
        output_path=tmp_path / "generation.jsonl",
        cache_key="key",
        prompt_batch_size=8,
    )
    package_id, items, mapping = build_blinded_package(
        _selected_rows(), generations, blind_seed="blind-v1"
    )
    first = _completed_submission(package_id, items, "reviewer-alpha")
    second = _completed_submission(package_id, items, "reviewer-beta")
    adjudication = _completed_submission(package_id, items, "reviewer-gamma")

    comparison = compare_submissions(items, first, second)

    assert comparison["disagreement_count"] == 0
    assert all(value == 1.0 for value in comparison["cohen_kappa"].values())
    assert "method_metrics" not in comparison

    report = finalize_adjudication(
        items,
        {
            "package_id": package_id,
            "blind_mapping_sha256": canonical_json_sha256(mapping),
        },
        mapping,
        first,
        second,
        adjudication,
    )
    assert report["status"] == "COMPLETED_INDEPENDENT_ADJUDICATED_CROSS_DOMAIN_EVALUATION"
    assert report["method_metrics"]["frc_select"]["expected_behavior_adherence"]["rate"] == 1.0
    assert report["method_metrics"]["coverage_greedy_proxy"]["factual_grounding"]["rate"] == 1.0
    assert report["paired_frc_minus_baseline"]["expected_behavior_adherence"] == {
        "case_count": 2,
        "mean_difference": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "wins": 0,
        "ties": 2,
        "losses": 0,
        "resamples": 10000,
        "seed": 20260731,
    }
    assert report["decision"]["gate_2"] == "NO-GO/SHADOW"

    placeholder = _completed_submission(package_id, items, "reviewer-delta")
    placeholder["annotator_id"] = "REPLACE_WITH_REVIEWER"
    with pytest.raises(ValueError, match="non-placeholder"):
        validate_submission(items, placeholder)
