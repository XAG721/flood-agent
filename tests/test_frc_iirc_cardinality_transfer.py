from __future__ import annotations

import json
from pathlib import Path

from research.frc_rag.iirc_cardinality_transfer import (
    PARTITION_COMMITMENTS,
    build_context_offset_index,
    build_partitions,
    clean_text,
    load_context_articles,
    normalize_title,
    select_all_methods,
    source_id,
)


def _candidate(candidate_id: str, source: str, cross: float) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": candidate_id,
        "scores": {
            "bm25": cross,
            "dense": cross,
            "hybrid": cross,
            "cross_encoder": cross,
        },
        "role_scores": {
            "answer": cross,
            "attribution": cross,
            "condition": cross,
            "exception": cross,
            "procedure": cross,
        },
    }


def test_normalization_and_source_identity_are_deterministic() -> None:
    assert normalize_title("New_York%20City") == "new york city"
    assert clean_text("<doc>Hello&nbsp; world</doc>") == "Hello world"
    assert source_id("link", "new york city") == source_id("link", "new york city")
    assert source_id("link", "new york city") != source_id("main", "new york city")


def test_context_offset_index_round_trip(tmp_path: Path, monkeypatch: object) -> None:
    source = tmp_path / "context.json"
    source.write_text(
        '{\n  "alpha": "First <b>article</b>",\n  "beta": "Second article"\n}\n',
        encoding="utf-8",
    )
    import research.frc_rag.iirc_cardinality_transfer as module

    monkeypatch.setitem(module.SOURCE_HASHES, "context_json", module.sha256(source))
    monkeypatch.setitem(module.SOURCE_SIZES, "context_json", source.stat().st_size)
    monkeypatch.setitem(module.SOURCE_COUNTS, "context_articles", 2)
    index = tmp_path / "context.sqlite"
    assert build_context_offset_index(source, index)["articles"] == 2
    assert load_context_articles(source, index, ["beta", "missing"]) == {
        "beta": "Second article"
    }


def test_partitions_follow_registered_hash_order(monkeypatch: object) -> None:
    rows = [
        {
            "questions": [
                {"qid": f"q{index:04d}"} for index in range(1200)
            ]
        }
    ]
    import research.frc_rag.iirc_cardinality_transfer as module

    monkeypatch.setitem(module.SOURCE_COUNTS, "train_articles", 1)
    monkeypatch.setitem(module.SOURCE_COUNTS, "train_questions", 1200)
    ids = [f"q{index:04d}" for index in range(1200)]
    ordered = sorted(
        ids,
        key=lambda value: (
            module.hashlib.sha256(f"{module.PARTITION_SALT}{value}".encode()).hexdigest(),
            value,
        ),
    )
    expected = {
        "all": sorted(ids),
        "development": ordered[:400],
        "confirmation": ordered[400:1200],
        "remaining": [],
    }
    for name, values in expected.items():
        monkeypatch.setitem(
            PARTITION_COMMITMENTS,
            name,
            module.hashlib.sha256("\n".join(values).encode()).hexdigest(),
        )
    assert build_partitions(rows) == expected


def test_all_methods_share_blind_candidates() -> None:
    row = {
        "question": "Which sources are required?",
        "candidates": [
            _candidate("a", "sa", 0.9),
            _candidate("b", "sb", 0.8),
            _candidate("c", "sc", 0.7),
            _candidate("d", "sd", 0.6),
        ],
    }
    # Use a minimal artifact whose probabilities all stay below their thresholds.
    from research.frc_rag.twowiki_question_router import fit_logistic, question_features, serialize_model
    import numpy as np

    features = np.vstack(
        [question_features("one", dimension=64), question_features("many", dimension=64)]
    )
    artifact = {
        "model_selection": {
            "selected_hash_dimension": 64,
            "selected_thresholds": {"2": 1.1, "3": 1.1, "4": 1.1},
        },
        "ordinal_models": {
            str(target): serialize_model(fit_logistic(features, np.asarray([0, 1]), l2=8.0))
            for target in (2, 3, 4)
        },
    }
    methods = select_all_methods(row, artifact)
    assert methods["cross_dataset_question_cardinality_frc_v86"]["cardinality"] == 1
    assert methods["frc_fixed4"]["selected_sources"] == ["sa", "sb", "sc", "sd"]
    assert json.dumps(methods, sort_keys=True)
