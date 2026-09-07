from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.hotpot_graph_router import CANDIDATE_METHOD
from research.frc_rag.musique_graph_router_transfer import (
    DEVELOPMENT_GATES,
    HOP_QUOTAS,
    NONINFERIORITY_ENVELOPE,
    evaluate_stage,
    hop_count_from_id,
    load_source_commitments,
    prepare_case,
    select_stage_ids,
    source_id_commitment,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl_gzip,
    sha256,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/hotpot_graph_router_model_development_v76.json"
)
V75_MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
PROTOCOL_PATH = (
    ROOT / "docs/progressive_upgrade/musique_graph_router_transfer_protocol_v77.json"
)
IMPLEMENTATION_LOCK_PATH = (
    ROOT / "docs/progressive_upgrade/musique_graph_router_transfer_implementation_v77.json"
)


def _metadata(count: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prefix in ("2hop", "3hop1", "4hop1"):
        for index in range(count):
            source_id = f"{prefix}__case-{index:03d}"
            rows.extend(
                [
                    {"id": source_id, "answerable": True},
                    {"id": source_id, "answerable": False},
                ]
            )
    return rows


def _paragraph(index: int, title: str, supporting: bool) -> dict[str, object]:
    return {
        "idx": index,
        "title": title,
        "paragraph_text": f"{title} contains evidence.",
        "is_supporting": supporting,
    }


def _candidate(
    candidate_id: str, source: str, text: str, cross: float
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": text,
        "metadata": {"title": source},
        "token_count": 10,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": {
            "condition": cross,
            "attribution": cross,
            "procedure": cross,
            "answer": cross,
            "exception": cross,
        },
    }


def _scored_row(case_id: str, hop_count: int) -> dict[str, object]:
    specs = (
        ("a", "Alpha", "Alpha. Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta. Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Gamma. Unrelated high score.", 0.9),
        ("d", "Delta", "Delta. Another distractor.", 0.8),
        ("e", "Epsilon", "Epsilon. Another distractor.", 0.7),
        ("f", "Zeta", "Zeta. Another distractor.", 0.6),
    )
    return {
        "dataset": "MuSiQue",
        "source": "synthetic",
        "id": case_id,
        "question": "How is Alpha connected to Beta?",
        "hop_count": hop_count,
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*spec) for spec in specs],
    }


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [("2hop__a", 2), ("3hop1__b", 3), ("4hop3__c", 4)],
)
def test_hop_count_uses_only_registered_id_prefix(
    source_id: str, expected: int
) -> None:
    assert hop_count_from_id(source_id) == expected


def test_hop_count_rejects_unregistered_prefix() -> None:
    with pytest.raises(ValueError, match="hop prefix"):
        hop_count_from_id("5hop__bad")


def test_stage_selection_is_deterministic_balanced_disjoint_and_excluded() -> None:
    rows = _metadata()
    excluded_id = "2hop__case-000"
    excluded = {source_id_commitment(excluded_id)}
    quotas = {2: 2, 3: 1, 4: 1}

    development = select_stage_ids(
        rows,
        excluded_source_commitments=excluded,
        stage="development",
        hop_quotas=quotas,
    )
    repeated = select_stage_ids(
        rows,
        excluded_source_commitments=excluded,
        stage="development",
        hop_quotas=quotas,
    )
    confirmation = select_stage_ids(
        rows,
        excluded_source_commitments=excluded,
        stage="confirmation",
        hop_quotas=quotas,
    )

    assert development == repeated
    assert len(development) == len(confirmation) == 4
    assert excluded_id not in development + confirmation
    assert not set(development) & set(confirmation)
    assert {
        hop: sum(hop_count_from_id(source_id) == hop for source_id in development)
        for hop in quotas
    } == quotas


def test_stage_selection_rejects_duplicate_answerable_ids() -> None:
    rows = [{"id": "2hop__same", "answerable": True}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        select_stage_ids(
            rows,
            excluded_source_commitments=set(),
            stage="development",
            hop_quotas={2: 1},
        )


def test_stage_selection_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="lacks 2 eligible untouched 2-hop"):
        select_stage_ids(
            [{"id": "2hop__one", "answerable": True}],
            excluded_source_commitments=set(),
            stage="development",
            hop_quotas={2: 2},
        )


def test_commitment_loader_unions_only_hashed_source_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    alpha = source_id_commitment("alpha")
    beta = source_id_commitment("beta")
    write_jsonl(first, [{"source_id_commitment": alpha}])
    write_jsonl(
        second,
        [
            {"source_id_commitment": alpha},
            {"source_id_commitment": beta},
        ],
    )

    assert load_source_commitments([first, second]) == {alpha, beta}


def test_commitment_loader_rejects_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"source_id_commitment": "not-a-sha256"}])
    with pytest.raises(ValueError, match="invalid commitment"):
        load_source_commitments([path])


def test_prepare_case_seals_gold_and_preserves_paragraph_contract() -> None:
    blind, gold = prepare_case(
        {
            "id": "2hop__case",
            "answerable": True,
            "question": "How are Alpha and Beta related?",
            "answer": "secret",
            "question_decomposition": [{"answer": "hidden"}],
            "paragraphs": [
                _paragraph(0, "Alpha", True),
                _paragraph(1, "Beta", True),
                _paragraph(2, "Gamma", False),
            ],
        }
    )

    payload = json.dumps(blind, sort_keys=True).lower()
    assert blind["id"] == gold["id"] == "2hop__case"
    assert blind["hop_count"] == gold["hop_count"] == 2
    assert [candidate["id"] for candidate in blind["candidates"]] == [
        "musique-p00",
        "musique-p01",
        "musique-p02",
    ]
    assert gold["gold_evidence_ids"] == ["musique-p00", "musique-p01"]
    assert "secret" not in payload
    assert "question_decomposition" not in payload
    assert "is_supporting" not in payload
    assert "gold" not in payload


@pytest.mark.parametrize(
    ("row_update", "message"),
    [
        ({"answerable": False}, "answerable"),
        ({"question": ""}, "question or paragraphs"),
        (
            {"paragraphs": [_paragraph(0, "Alpha", False), _paragraph(1, "Beta", False)]},
            "no supporting",
        ),
    ],
)
def test_prepare_case_rejects_invalid_contracts(
    row_update: dict[str, object], message: str
) -> None:
    row = {
        "id": "2hop__bad",
        "answerable": True,
        "question": "Question?",
        "paragraphs": [_paragraph(0, "Alpha", True), _paragraph(1, "Beta", True)],
        **row_update,
    }
    with pytest.raises(ValueError, match=message):
        prepare_case(row)


def test_registered_protocol_is_exact_and_detects_drift() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["development_gates"] == DEVELOPMENT_GATES
    assert protocol["noninferiority_envelope"] == NONINFERIORITY_ENVELOPE
    assert protocol["stages"]["development"]["hop_quota"] == {
        str(key): value for key, value in HOP_QUOTAS.items()
    }

    protocol["development_gates"]["candidate_evidence_macro_f1_at_least"] = 0.0
    with pytest.raises(ValueError, match="gates drifted"):
        validate_registered_protocol(protocol)


def test_implementation_lock_hashes_every_frozen_file() -> None:
    lock = json.loads(IMPLEMENTATION_LOCK_PATH.read_text(encoding="utf-8"))
    for contract in lock["files"].values():
        path = ROOT / contract["path"]
        assert path.stat().st_size == contract["bytes"]
        assert sha256(path) == contract["sha256"]


def test_small_evaluation_writes_safe_evidence_and_fails_size_gate(
    tmp_path: Path,
) -> None:
    scored_path = tmp_path / "scored.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    rows = [
        _scored_row(f"{hop}hop__case", hop) for hop in HOP_QUOTAS
    ]
    gold = [
        {
            "id": row["id"],
            "hop_count": row["hop_count"],
            "gold_evidence_ids": ["a", "b"],
        }
        for row in rows
    ]
    write_jsonl(scored_path, rows)
    write_jsonl(gold_path, gold)

    report = evaluate_stage(
        scored_path,
        gold_path,
        selection_path,
        MODEL_PATH,
        V75_MODEL_PATH,
        stage="development",
        seed=7,
        output_dir=tmp_path / "output",
        prior_overlap=0,
    )
    cases = read_jsonl_gzip(tmp_path / "output/cases.jsonl.gz")

    assert report["metadata"]["selection_written_before_gold_join"] is True
    assert report["analysis"]["support_checks"]["exact_cases_equals_800"] is False
    assert report["analysis"]["outcome"]["confirmation_open_authorized"] is False
    assert CANDIDATE_METHOD in report["analysis"]["methods"]
    payload = json.dumps(cases).lower()
    assert '"question":' not in payload
    assert '"candidate_ids":' not in payload
    assert '"text":' not in payload
    assert sha256(selection_path) == report["metadata"]["selection_output_sha256"]
