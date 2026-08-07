from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES, STATIC_ROLES
from research.frc_rag.ottqa_guarded_adaptive_atomic_roles import (
    ADAPTIVE_V43,
    GUARDED_V44,
    MAXIMUM_POOL_SIZE,
    adaptive_v43_target_cardinality,
    archive_directory_git_manifest,
    archive_git_blob_sha1,
    build_candidate_coverage,
    build_gold_rows,
    directory_git_manifest,
    evaluate_ottqa,
    guarded_target_cardinality,
    prepare_blind_cases,
    select_sample,
    select_v44,
    validate_protocol,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        for token in text.split():
            if token not in self.values:
                index = len(self.values) + 1
                self.values[token] = index
                self.reverse[index] = token
            result.append(self.values[token])
        return result

    def decode(self, values: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self.reverse[value] for value in values)


def _linked_row(
    question_id: str,
    table_id: str,
    *,
    mode: str = "table",
) -> dict:
    answer_node = (
        ["Answer", [0, 1], None, "table"]
        if mode == "table"
        else ["Alpha", [0, 0], "/wiki/Alpha", "passage"]
    )
    return {
        "question_id": question_id,
        "question": "Which Alpha record contains the Answer?",
        "table_id": table_id,
        "answer-text": "Answer",
        "answer-node": [answer_node],
        "tf-idf": [["Alpha", [0, 0], "/wiki/Alpha", "alpha", 0.1]],
        "string-overlap": [],
        "links": [["Alpha", [0, 0], None, "string match", 1.0]],
    }


def _write_sources(root: Path, table_id: str) -> tuple[Path, Path]:
    tables = root / "tables"
    passages = root / "passages"
    tables.mkdir(parents=True, exist_ok=True)
    passages.mkdir(parents=True, exist_ok=True)
    table = {
        "title": "Synthetic table",
        "section_title": "Test section",
        "header": [["Name", []], ["Value", []]],
        "data": [
            [["Alpha", ["/wiki/Alpha"]], ["Answer", []]],
            [["Beta", ["/wiki/Beta"]], ["Distractor", []]],
        ],
    }
    (tables / f"{table_id}.json").write_text(json.dumps(table), encoding="utf-8")
    (passages / f"{table_id}.json").write_text(
        json.dumps(
            {
                "/wiki/Alpha": "The Alpha passage contains Answer and supporting context.",
                "/wiki/Beta": "The Beta passage is a distractor.",
            }
        ),
        encoding="utf-8",
    )
    return tables, passages


def _candidate(
    identifier: str,
    role_scores: dict[str, float],
    *,
    official_rank: int | None = None,
    cross_encoder: float = 0.5,
) -> dict:
    return {
        "id": identifier,
        "source_id": f"source-{identifier}",
        "token_count": 50,
        "official_rank": official_rank,
        "scores": {
            "bm25": cross_encoder,
            "dense": cross_encoder,
            "hybrid": cross_encoder,
            "cross_encoder": cross_encoder,
        },
        "static_role_scores": {role: cross_encoder for role in STATIC_ROLES},
        "dynamic_role_scores": role_scores,
    }


def _role_candidates(distinct: int) -> list[dict]:
    identifiers = ["a", "b", "c", "d", "e"]
    assignments = [identifiers[index % distinct] for index in range(len(DYNAMIC_ROLES))]
    result: list[dict] = []
    for candidate_index, identifier in enumerate(identifiers):
        scores = {
            role: (
                1.0
                if assignments[role_index] == identifier
                else 0.1 - candidate_index * 0.01
            )
            for role_index, role in enumerate(DYNAMIC_ROLES)
        }
        result.append(
            _candidate(
                identifier,
                scores,
                official_rank=candidate_index,
                cross_encoder=1.0 - candidate_index * 0.1,
            )
        )
    return result


def test_protocol_is_hash_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    value = validate_protocol(
        root
        / "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_v44.json",
        erratum_path=(
            root
            / "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_erratum_v44.json"
        ),
        erratum2_path=(
            root
            / "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_erratum2_v44.json"
        ),
        erratum3_path=(
            root
            / "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_erratum3_v44.json"
        ),
    )
    protocol = value["base_protocol"]
    assert protocol["methods"]["candidate_method"] == GUARDED_V44
    assert (
        protocol["development_boundary"][
            "ottqa_question_answer_or_evidence_content_inspected_before_registration"
        ]
        is False
    )
    assert len(value["storage_errata"]) == 3
    assert all(
        erratum["method_or_threshold_changed"] is False
        for erratum in value["storage_errata"]
    )


def test_guarded_formula_matches_every_registered_synthetic_invariant() -> None:
    assert guarded_target_cardinality([]) == 0
    assert adaptive_v43_target_cardinality([]) == 0
    expected = {1: 3, 2: 3, 3: 4, 4: 5}
    for distinct, target in expected.items():
        candidates = _role_candidates(distinct)
        assert guarded_target_cardinality(candidates) == target
        assert adaptive_v43_target_cardinality(candidates) == distinct


def test_guarded_selection_is_deterministic_and_budget_can_only_reduce_count() -> None:
    candidates = _role_candidates(3)
    first = select_v44(candidates, GUARDED_V44, token_budget=512)
    second = select_v44(list(reversed(candidates)), GUARDED_V44, token_budget=512)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4
    assert len(select_v44(candidates, GUARDED_V44, token_budget=100)) <= len(first)
    replay = select_v44(candidates, ADAPTIVE_V43, token_budget=512)
    assert len(replay) == 3


def test_sampling_is_deterministic_balanced_and_does_not_export_mode() -> None:
    rows = [
        _linked_row(f"t-{index:03d}", f"table-t-{index:03d}", mode="table")
        for index in range(130)
    ]
    rows.extend(
        _linked_row(f"p-{index:03d}", f"table-p-{index:03d}", mode="passage")
        for index in range(130)
    )
    first, census = select_sample(rows)
    second, _ = select_sample(list(reversed(rows)))
    assert [row["question_id"] for row in first] == [
        row["question_id"] for row in second
    ]
    assert len(first) == 240
    assert census["selected_by_source_mode"] == {"table": 120, "passage": 120}
    assert census["selection_exports_source_mode_to_blind_cache"] is False


def test_preparation_is_blind_bounded_and_gold_join_is_late(tmp_path: Path) -> None:
    table_id = "synthetic_0"
    tables, passages = _write_sources(tmp_path, table_id)
    linked = [
        _linked_row("table-question", table_id, mode="table"),
        _linked_row("passage-question", table_id, mode="passage"),
    ]
    prepared, maps, census = prepare_blind_cases(linked, tables, passages, _Tokenizer())
    assert len(prepared) == len(maps) == 2
    assert census["gold_fields_exported_to_blind_cache"] is False
    assert all(len(row["candidates"]) <= MAXIMUM_POOL_SIZE for row in prepared)
    serialized = json.dumps(prepared, sort_keys=True)
    for forbidden in (
        '"answer-node"',
        '"answer-text"',
        '"source_mode"',
        '"table_id"',
        '"where"',
    ):
        assert forbidden not in serialized
    gold = build_gold_rows(linked, maps)
    assert all(row["candidate_ceiling_complete"] for row in gold)
    coverage = build_candidate_coverage(gold, census)
    assert coverage["candidate_ceiling_complete_rate"] == 1.0
    assert coverage["candidate_method_or_threshold_changed"] is False


def test_directory_manifest_is_stable(tmp_path: Path) -> None:
    directory = tmp_path / "values"
    directory.mkdir()
    (directory / "b.txt").write_text("beta", encoding="utf-8")
    (directory / "a.txt").write_text("alpha", encoding="utf-8")
    first = directory_git_manifest(directory, "data/example")
    second = directory_git_manifest(directory, "data/example")
    assert first == second
    assert first["files"] == 2
    assert first["bytes"] == 9


def test_tar_source_supports_windows_invalid_git_names_without_extraction(
    tmp_path: Path,
) -> None:
    table_id = 'unsafe:"table*0'
    linked = [_linked_row("archive-question", table_id, mode="table")]
    table = {
        "title": "Archived table",
        "header": [["Name", []], ["Value", []]],
        "data": [[["Alpha", ["/wiki/Alpha"]], ["Answer", []]]],
    }
    passages = {"/wiki/Alpha": "Alpha passage contains Answer."}
    archive_path = tmp_path / "source.tar"
    members = {
        "LICENSE": b"MIT\n",
        "preprocessed_data/dev_linked.json": json.dumps(linked).encode("utf-8"),
        f"data/traindev_tables_tok/{table_id}.json": json.dumps(table).encode("utf-8"),
        f"data/traindev_request_tok/{table_id}.json": json.dumps(passages).encode(
            "utf-8"
        ),
    }
    with tarfile.open(archive_path, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    assert archive_git_blob_sha1(archive_path, "LICENSE")
    manifest = archive_directory_git_manifest(archive_path, "data/traindev_tables_tok")
    assert manifest["files"] == 1
    prepared, maps, _ = prepare_blind_cases(
        linked,
        None,
        None,
        _Tokenizer(),
        source_archive=archive_path,
    )
    assert len(prepared) == len(maps) == 1
    assert maps[0]["table_id"] == table_id


def test_locked_evaluation_reports_oracle_boundary_and_gate() -> None:
    candidates = _role_candidates(4)
    scored = {
        "id": "case-1",
        "gold_fields_visible_to_scorer": False,
        "candidates": [
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "token_count": row["token_count"],
                "official_rank": row["official_rank"],
            }
            for row in candidates
        ],
        "candidate_scores": [
            {
                "id": row["id"],
                "scores": row["scores"],
                "static_role_scores": row["static_role_scores"],
                "dynamic_role_scores": row["dynamic_role_scores"],
            }
            for row in candidates
        ],
    }
    gold = {
        "case_id": "case-1",
        "question_id": "question-1",
        "source_mode": "table",
        "answer_node_count": 1,
        "represented_answer_node_count": 1,
        "gold_candidate_ids": ["a"],
        "candidate_ceiling_complete": True,
    }
    report, evidence = evaluate_ottqa(
        [gold],
        [scored],
        {"fallback_rate": 0.0},
        resamples=50,
    )
    assert len(evidence) == 1
    assert report["metadata"]["oracle_table_bounded_pool"] is True
    assert report["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"
    assert report["analysis"]["outcome"]["canary_or_default_authorized"] is False
