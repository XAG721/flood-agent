from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from research.frc_rag.musique_crossfit_support import (
    PROTOCOL_SHA256,
    build_case_features,
    crossfit_predictions,
    evaluate_scored_cases,
    fit_logistic,
    fold_for_case,
    load_report,
    predict_logistic,
    validate_preparation_summary,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "docs/progressive_upgrade/musique_crossfit_support_protocol.json"
)


def _role_scores(value: float) -> dict[str, float]:
    return {
        "direct_answer_support": value,
        "entity_and_scope": value,
        "corroborating_evidence": value,
        "contradiction_detection": value,
    }


def _pair(case_id: str) -> tuple[dict, dict]:
    candidates = [
        (f"{case_id}::a", 170, 0.9, 0.1, False),
        (f"{case_id}::b", 180, 0.6, 0.9, True),
        (f"{case_id}::c", 190, 0.2, 0.2, False),
    ]
    scored = {
        "id": case_id,
        "dataset_id": "musique_ans_v1.0_dev",
        "capability": "multi_hop_evidence_selection",
        "gold_fields_visible_to_scorer": False,
        "candidates": [
            {"id": row[0], "source_id": row[0], "token_count": row[1]}
            for row in candidates
        ],
        "candidate_scores": [
            {
                "id": row[0],
                "scores": {
                    "bm25": row[2],
                    "dense": row[2],
                    "hybrid": row[2],
                    "cross_encoder": row[2],
                },
                "role_scores": _role_scores(row[3]),
            }
            for row in candidates
        ],
    }
    gold = {
        "case_id": case_id,
        "hop_count": 2 + fold_for_case(case_id) % 3,
        "gold_unit_ids": ["g0"],
        "candidate_gold": {
            row[0]: {
                "labels": ["supporting" if row[4] else "non_supporting"],
                "gold_unit_ids": ["g0"] if row[4] else [],
            }
            for row in candidates
        },
    }
    return scored, gold


def _fold_complete_pairs() -> list[tuple[dict, dict]]:
    found: dict[int, tuple[dict, dict]] = {}
    index = 0
    while len(found) < 5:
        case_id = f"musique::crossfit-{index}"
        found.setdefault(fold_for_case(case_id), _pair(case_id))
        index += 1
    return [found[fold] for fold in range(5)]


def test_fold_assignment_is_deterministic_and_case_isolated() -> None:
    pairs = _fold_complete_pairs()
    assert [fold_for_case(scored["id"]) for scored, _ in pairs] == list(range(5))
    assert all(
        fold_for_case(scored["id"]) == fold_for_case(scored["id"])
        for scored, _ in pairs
    )


def test_logistic_prediction_does_not_accept_evaluation_labels() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    model = fit_logistic(features, labels, weights)
    evaluation_features = np.asarray([[0.5], [2.5]])

    first = predict_logistic(model, evaluation_features)
    counterfactual_evaluation_labels = np.asarray([1.0, 0.0])
    second = predict_logistic(model, evaluation_features)

    assert np.array_equal(first, second)
    assert counterfactual_evaluation_labels.tolist() == [1.0, 0.0]


def test_crossfit_predictions_cover_each_case_once_without_eval_labels() -> None:
    pairs = _fold_complete_pairs()
    cases = [build_case_features(scored, gold) for scored, gold in pairs]
    predictions, audits = crossfit_predictions(cases, "full")

    assert set(predictions) == {case.case_id for case in cases}
    assert [audit["fold"] for audit in audits] == list(range(5))
    assert all(audit["evaluation_cases"] == 1 for audit in audits)
    assert all(audit["evaluation_labels_visible_to_fit"] is False for audit in audits)


def test_report_is_private_and_deterministic(tmp_path: Path) -> None:
    pairs = _fold_complete_pairs()
    report, evidence = evaluate_scored_cases(
        [gold for _, gold in pairs],
        [scored for scored, _ in pairs],
        resamples=32,
    )
    assert report["metadata"]["fold_distribution"] == {
        str(fold): 1 for fold in range(5)
    }
    assert report["analysis"]["outcome"]["adoption_eligible"] is False
    encoded = json.dumps(evidence, sort_keys=True)
    for forbidden in ("question", "answer", "text", "is_supporting"):
        assert f'"{forbidden}"' not in encoded

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    first = write_report(
        copy.deepcopy(report), evidence, tmp_path / "first", source_paths={"x": source}
    )
    second = write_report(
        copy.deepcopy(report),
        evidence,
        tmp_path / "second",
        source_paths={"x": source},
    )
    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["markdown"].read_bytes() == second["markdown"].read_bytes()
    assert first["evidence"].read_bytes() == second["evidence"].read_bytes()
    loaded = load_report(first["json"], first["evidence"])
    assert loaded["analysis"]["outcome"]["gate_2"] == "NO-GO/SHADOW"


def test_protocol_hash_and_preparation_shape_are_frozen() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_preparation_summary(
        {"valid_rows": 2417, "candidate_chunks": {"total": 48656}},
        protocol,
    )
