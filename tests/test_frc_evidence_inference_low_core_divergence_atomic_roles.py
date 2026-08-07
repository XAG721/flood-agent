from __future__ import annotations

import csv
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import research.frc_rag.evidence_inference_low_core_divergence_atomic_roles as ei
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    build_candidate_coverage,
    build_gold_rows,
    build_sentence_units,
    evaluate_evidence_inference,
    low_core_divergence_details,
    prepare_blind_cases,
    read_validation_source,
    select_sample,
    select_v49,
    validate_protocol,
    write_report,
)
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _article(index: int) -> str:
    return (
        f"Trial {index}\n"
        f"Background statement {index}. "
        f"Primary evidence sentence {index}. "
        f"Secondary evidence sentence {index}. "
        f"Safety context sentence {index}. "
        f"Conclusion sentence {index}."
    )


def _annotation(
    prompt_id: int,
    pmcid: int,
    user_id: str,
    article: str,
    evidence: str,
    *,
    label: str = "significantly increased",
    in_abstract: str = "0",
) -> dict[str, str]:
    start = article.index(evidence)
    return {
        "UserID": user_id,
        "PromptID": str(prompt_id),
        "PMCID": str(pmcid),
        "Valid Label": "1",
        "Valid Reasoning": "1",
        "Label": label,
        "Annotations": evidence,
        "Label Code": "1",
        "In Abstract": in_abstract,
        "Evidence Start": str(start),
        "Evidence End": str(start + len(evidence) - 1),
    }


def _source(count: int = 4) -> dict:
    prompts = []
    annotations = []
    articles = {}
    validation_ids = []
    for index in range(count):
        prompt_id = 20_000 + index
        pmcid = 30_000 + index
        article = _article(index)
        validation_ids.append(pmcid)
        articles[pmcid] = article
        prompts.append(
            {
                "PromptID": str(prompt_id),
                "PMCID": str(pmcid),
                "Intervention": f"intervention {index}",
                "Comparator": f"comparator {index}",
                "Outcome": f"outcome {index}",
            }
        )
        annotations.append(
            _annotation(
                prompt_id,
                pmcid,
                "doctor-a",
                article,
                f"Primary evidence sentence {index}.",
            )
        )
        annotations.extend(
            [
                _annotation(
                    prompt_id,
                    pmcid,
                    "doctor-b",
                    article,
                    f"Secondary evidence sentence {index}.",
                ),
                _annotation(
                    prompt_id,
                    pmcid,
                    "doctor-b",
                    article,
                    f"Safety context sentence {index}.",
                ),
            ]
        )
    return {
        "validation_article_ids": validation_ids,
        "prompts": prompts,
        "annotations": annotations,
        "articles": articles,
    }


def _scored_candidate(
    identifier: str, role_scores: dict[str, float], index: int
) -> dict:
    cross = 1.0 - index * 0.01
    return {
        "id": identifier,
        "source_id": identifier,
        "source_kind": "sentence",
        "text": f"candidate {identifier}",
        "token_count": 5,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "static_role_scores": {role: cross for role in STATIC_ROLES},
        "dynamic_role_scores": role_scores,
    }


def _ranked_candidates(pairs: list[tuple[int, int]], count: int = 8) -> list[dict]:
    scores = [{role: 0.0 for role in DYNAMIC_ROLES} for _ in range(count)]
    for role_index, role in enumerate(DYNAMIC_ROLES):
        first, second = pairs[role_index]
        scores[first][role] = 10.0
        scores[second][role] = 9.0
        for index in range(count):
            scores[index][role] += (count - index) * 0.001
    return [_scored_candidate(f"c{index}", scores[index], index) for index in range(count)]


def _scored_row(case: dict) -> dict:
    candidate_scores = []
    for index, candidate in enumerate(case["candidates"]):
        value = 1.0 - index * 0.01
        candidate_scores.append(
            {
                "id": candidate["id"],
                "scores": {
                    "bm25": value,
                    "dense": value,
                    "hybrid": value,
                    "cross_encoder": value,
                },
                "static_role_scores": {role: value for role in STATIC_ROLES},
                "dynamic_role_scores": {
                    role: value - role_index * 0.0001
                    for role_index, role in enumerate(DYNAMIC_ROLES)
                },
            }
        )
    return {
        "id": case["id"],
        "candidates": case["candidates"],
        "candidate_scores": candidate_scores,
        "gold_fields_visible_to_scorer": False,
    }


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _add_tar_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def test_protocol_freezes_source_formula_and_split_exclusions() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(
        root
        / "docs/progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_protocol_v49.json"
    )
    assert protocol["methods"]["candidate_method"] == LOW_CORE_DIVERGENCE_V49
    assert (
        protocol["development_boundary"][
            "evidence_inference_train_split_permanently_excluded"
        ]
        is True
    )
    assert (
        protocol["development_boundary"][
            "evidence_inference_test_split_permanently_excluded"
        ]
        is True
    )
    assert protocol["pre_registration_access_disclosure"][
        "validation_article_id_prompt_annotation_evidence_label_or_article_text_seen"
    ] is False


def test_sentence_split_retains_offsets_and_inclusive_evidence_maps_exactly() -> None:
    article = _article(1)
    units = build_sentence_units(article, Tokenizer())
    assert len(units) == 5
    assert article[units[1]["start"] : units[1]["end"]] == "Primary evidence sentence 1."
    annotation = _annotation(
        20_001,
        30_001,
        "doctor-a",
        article,
        "Primary evidence sentence 1.",
    )
    references, label, location, complete = ei._reference_metadata(
        [annotation], units, article_length=len(article)
    )
    assert references == (("sentence_0001",),)
    assert label == "significantly increased"
    assert location == "includes_non_abstract"
    assert complete is True


def test_archive_reader_retains_only_validation_rows_and_article(tmp_path: Path) -> None:
    val_article = _article(1)
    train_article = _article(2)
    val_prompt = {
        "PromptID": "20001",
        "PMCID": "30001",
        "Intervention": "A",
        "Comparator": "B",
        "Outcome": "C",
    }
    train_prompt = {**val_prompt, "PromptID": "20002", "PMCID": "30002"}
    val_annotation = _annotation(
        20_001,
        30_001,
        "doctor-a",
        val_article,
        "Primary evidence sentence 1.",
    )
    train_annotation = _annotation(
        20_002,
        30_002,
        "doctor-a",
        train_article,
        "Primary evidence sentence 2.",
    )
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, mode="w:gz") as archive:
        _add_tar_bytes(archive, "v2/splits/validation_article_ids.txt", b"30001\n")
        _add_tar_bytes(archive, "v2/splits/train_article_ids.txt", b"30002\n")
        _add_tar_bytes(
            archive,
            "v2/prompts_merged.csv",
            _csv_bytes([val_prompt, train_prompt]),
        )
        _add_tar_bytes(
            archive,
            "v2/annotations_merged.csv",
            _csv_bytes([val_annotation, train_annotation]),
        )
        _add_tar_bytes(archive, "v2/txt_files/PMC30001.txt", val_article.encode())
        _add_tar_bytes(archive, "v2/txt_files/PMC30002.txt", train_article.encode())
    source = read_validation_source(path)
    assert source["validation_article_ids"] == [30_001]
    assert [_prompt_id for _prompt_id in (row["PromptID"] for row in source["prompts"])] == [
        "20001"
    ]
    assert [row["PromptID"] for row in source["annotations"]] == ["20001"]
    assert source["articles"] == {30_001: val_article}


def test_sealed_sampling_is_order_invariant(monkeypatch) -> None:
    monkeypatch.setattr(ei, "TARGET_CASES", 3)
    monkeypatch.setattr(ei, "MINIMUM_CASES", 3)
    source = _source(5)
    first, first_summary = select_sample(source)
    reversed_source = {
        **source,
        "prompts": list(reversed(source["prompts"])),
        "annotations": list(reversed(source["annotations"])),
    }
    second, second_summary = select_sample(reversed_source)
    assert [row["prompt_id"] for row in first] == [row["prompt_id"] for row in second]
    assert first_summary == second_summary


def test_blind_cache_is_gold_free_and_annotator_references_join_late(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ei, "TARGET_CASES", 4)
    monkeypatch.setattr(ei, "MINIMUM_CASES", 4)
    source = _source(4)
    prepared, maps, summary = prepare_blind_cases(source, Tokenizer())
    serialized = json.dumps(prepared, ensure_ascii=False).lower()
    for forbidden in (
        "promptid",
        "pmcid",
        "userid",
        "evidence start",
        "evidence end",
        "gold_references",
        '"label":',
    ):
        assert forbidden not in serialized
    assert len(prepared) == 4
    assert all("prompt_id" not in item and "pmcid" not in item for item in maps)
    gold = build_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"],
    )
    assert all(row["reference_count"] == 2 for row in gold)
    assert all(row["smallest_reference_count"] == 1 for row in gold)
    coverage = build_candidate_coverage(
        gold, summary["sampling"], summary["candidate_pool_quartile_boundaries"]
    )
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["minimum_cases_and_ceiling_checks_passed"] is True


def test_low_core_divergence_targets_match_registered_synthetic_invariants() -> None:
    shared = _ranked_candidates([(0, 1), (0, 1), (0, 1), (0, 1)])
    one_core_divergent = _ranked_candidates([(0, 1), (0, 2), (0, 3), (0, 4)])
    two_core_divergent = _ranked_candidates([(0, 2), (0, 3), (1, 4), (1, 5)])
    three_core = _ranked_candidates([(0, 3), (1, 4), (2, 5), (0, 6)])
    assert low_core_divergence_details(shared)["expansion_triggered"] is False
    assert low_core_divergence_details(shared)["target_cardinality"] == 1
    assert low_core_divergence_details(one_core_divergent)["expansion_triggered"] is True
    assert low_core_divergence_details(one_core_divergent)["target_cardinality"] == 2
    assert low_core_divergence_details(two_core_divergent)["expansion_triggered"] is True
    assert low_core_divergence_details(two_core_divergent)["target_cardinality"] == 3
    assert low_core_divergence_details(three_core)["expansion_triggered"] is False
    assert low_core_divergence_details(three_core)["target_cardinality"] == 3
    for candidates in (shared, one_core_divergent, two_core_divergent, three_core):
        details = low_core_divergence_details(candidates)
        selected = select_v49(
            candidates, LOW_CORE_DIVERGENCE_V49, token_budget=1024
        )
        assert len(selected) == details["target_cardinality"]
        assert len({item["id"] for item in selected}) == len(selected)
        assert sum(item["token_count"] for item in selected) <= 1024


def test_selector_is_permutation_monotone_and_text_position_invariant() -> None:
    candidates = _ranked_candidates([(0, 1), (0, 2), (0, 3), (0, 4)])
    expected = {
        item["id"]
        for item in select_v49(
            candidates, LOW_CORE_DIVERGENCE_V49, token_budget=1024
        )
    }
    reversed_selected = {
        item["id"]
        for item in select_v49(
            list(reversed(candidates)), LOW_CORE_DIVERGENCE_V49, token_budget=1024
        )
    }
    assert expected == reversed_selected
    transformed = deepcopy(candidates)
    for item in transformed:
        item["dynamic_role_scores"] = {
            role: 3.0 * value + 7.0
            for role, value in item["dynamic_role_scores"].items()
        }
    assert expected == {
        item["id"]
        for item in select_v49(
            transformed, LOW_CORE_DIVERGENCE_V49, token_budget=1024
        )
    }
    changed_text = deepcopy(candidates)
    for index, item in enumerate(changed_text):
        item["text"] = f"changed position-like text {999 - index}"
        item["source_kind"] = "other"
    assert expected == {
        item["id"]
        for item in select_v49(
            changed_text, LOW_CORE_DIVERGENCE_V49, token_budget=1024
        )
    }
    constrained = select_v49(
        candidates, LOW_CORE_DIVERGENCE_V49, token_budget=5
    )
    assert len(constrained) <= len(expected)


def test_locked_evaluation_and_report_are_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ei, "TARGET_CASES", 4)
    monkeypatch.setattr(ei, "MINIMUM_CASES", 4)
    monkeypatch.setattr(ei, "MINIMUM_STRATUM_CASES", 1)
    source = _source(4)
    prepared, maps, summary = prepare_blind_cases(source, Tokenizer())
    gold = build_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=summary["candidate_pool_quartile_boundaries"],
    )
    scored = [_scored_row(case) for case in prepared]
    report, evidence = evaluate_evidence_inference(
        gold,
        scored,
        query_summary={"fallback_rate": 0.0},
        source_artifacts={"synthetic": True},
        resamples=30,
    )
    assert report["analysis"]["outcome"]["selector_adoption_authorized"] is False
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["metadata"]["multi_reference_gold_not_unioned"] is True
    assert report["metadata"]["eraser_result"] is False
    assert len(evidence) == 4
    paths = (
        tmp_path / "result.json",
        tmp_path / "report.md",
        tmp_path / "cases.jsonl.gz",
    )
    write_report(
        report,
        evidence,
        json_path=paths[0],
        markdown_path=paths[1],
        evidence_path=paths[2],
    )
    first = tuple(path.read_bytes() for path in paths)
    write_report(
        report,
        evidence,
        json_path=paths[0],
        markdown_path=paths[1],
        evidence_path=paths[2],
    )
    assert first == tuple(path.read_bytes() for path in paths)
