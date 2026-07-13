from __future__ import annotations

from flood_system.frc_conflicts_evaluation import (
    answer_support,
    classification_metrics,
    load_conflicts_cases,
    local_model_snapshot,
    normalize_conflict_label,
    parse_date,
    selection_diagnostics,
    selection_signature,
)


def test_load_conflicts_cases_preserves_labels_and_uses_short_text(tmp_path):
    path = tmp_path / "conflicts.jsonl"
    path.write_text(
        '{"source":"freshqa","question":"When?","search_results":'
        '[{"title":"A","url":"https://example.com/a","snippet":"fallback",'
        '"date":"2025-01-01","response_str":"long","short_text":"short evidence"}],'
        '"conflict_type":"Conflict due to outdated information","correct_answer":"2025"}\n',
        encoding="utf-8",
    )

    cases = load_conflicts_cases(path)

    assert cases[0]["conflict_type"] == "Conflict due to outdated information"
    assert cases[0]["candidates"][0]["text"] == "short evidence"
    assert cases[0]["required_roles"] == [
        "answer_claim",
        "source_attribution",
        "alternative_claim",
        "temporal_validity",
    ]


def test_conflict_label_normalization_and_metrics_handle_unparsed_output():
    assert normalize_conflict_label("Conflict due to outdated information") == (
        "Conflict due to outdated information"
    )
    assert normalize_conflict_label("Label: No conflict.") == "No conflict"
    assert normalize_conflict_label("I cannot decide") is None

    metrics = classification_metrics(
        [
            {
                "gold_label": "No conflict",
                "predicted_label": "No conflict",
            },
            {
                "gold_label": "Conflict due to outdated information",
                "predicted_label": None,
            },
        ]
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["parse_rate"] == 0.5
    assert metrics["per_type"]["No conflict"]["recall"] == 1.0
    assert metrics["per_type"]["Conflict due to outdated information"]["support"] == 1
    assert metrics["per_type"]["Conflict due to outdated information"]["recall"] == 0.0


def test_date_and_selection_diagnostics_measure_temporal_endpoints():
    assert parse_date("2025-03-31").year == 2025
    assert parse_date("Dec 10, 2024").year == 2024
    assert parse_date("NA") is None

    candidates = [
        {
            "id": "old",
            "text": "The answer was four accounts in 2024.",
            "url": "https://old.example/a",
            "date": "2024-01-01",
            "token_count": 8,
        },
        {
            "id": "new",
            "text": "The answer is five accounts in 2025.",
            "url": "https://new.example/b",
            "date": "2025-01-01",
            "token_count": 8,
        },
    ]
    case = {"candidates": candidates}
    row = {
        "selected_evidence": candidates,
        "selected_ids": ["old", "new"],
        "correct_answer": "five accounts",
    }

    diagnostics = selection_diagnostics(row, case)

    assert diagnostics["newest_date_retention"] == 1.0
    assert diagnostics["temporal_endpoint_coverage"] == 1.0
    assert diagnostics["exact_answer_support"] == 1.0
    assert diagnostics["answer_token_recall"] == 1.0


def test_answer_support_excludes_unannotated_answers():
    assert answer_support("", [{"text": "anything"}]) == (0.0, 0.0)


def test_local_model_snapshot_resolves_cached_main_revision(tmp_path):
    repository = tmp_path / "hub" / "models--BAAI--model"
    snapshot = repository / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("revision-1\n", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert local_model_snapshot(tmp_path, "BAAI/model") == snapshot


def test_selection_signature_reuses_only_identical_ordered_evidence_for_same_case():
    first = {"case_id": "case-1", "selected_ids": ["a", "b"]}
    same = {"case_id": "case-1", "selected_ids": ["a", "b"]}
    reordered = {"case_id": "case-1", "selected_ids": ["b", "a"]}
    other_case = {"case_id": "case-2", "selected_ids": ["a", "b"]}

    assert selection_signature(first) == selection_signature(same)
    assert selection_signature(first) != selection_signature(reordered)
    assert selection_signature(first) != selection_signature(other_case)
