from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.hotpot_graph_router import (
    CANDIDATE_METHOD as V76_CANDIDATE_METHOD,
    load_router_artifact as load_v76_router_artifact,
)
from research.frc_rag.hotpot_question_type_cardinality_router import (
    ACTIONS,
    CANDIDATE_METHOD,
    load_router_artifact as load_v84_router_artifact,
)
from research.frc_rag.hotpot_question_type_cardinality_transfer import (
    CONTROL_METHODS,
    METHODS,
    NONINFERIORITY_ENVELOPE,
    PREFIX3_METHOD,
    PREFIX4_METHOD,
    SAFETY_ELIGIBILITY,
    STRICT_GATES,
    select_stage_ids,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
V84_MODEL_PATH = (
    DOCS_ROOT / "hotpot_question_type_cardinality_router_model_development_v84.json"
)
PROTOCOL_PATH = DOCS_ROOT / "hotpot_question_type_cardinality_protocol_v84.json"


def _candidate(
    candidate_id: str, source: str, text: str, score: float
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "source": source,
        "text": text,
        "metadata": {"title": source},
        "token_count": 10,
        "scores": {
            "bm25": score,
            "dense": score,
            "hybrid": score,
            "cross_encoder": score,
        },
        "role_scores": {
            "condition": score,
            "attribution": score,
            "procedure": score,
            "answer": score,
            "exception": score,
        },
    }


def _scored_row(case_id: str = "synthetic-v84") -> dict[str, object]:
    specs = (
        ("a", "Alpha", "Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Unrelated high score.", 0.9),
        ("d", "Delta", "Another distractor.", 0.8),
        ("e", "Epsilon", "Another distractor.", 0.7),
        ("f", "Zeta", "Another distractor.", 0.6),
    )
    return {
        "dataset": "HotpotQA",
        "source": "synthetic",
        "id": case_id,
        "question": "Which film has the director who was born later, Alpha or Beta?",
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*spec) for spec in specs],
    }


def _models() -> dict[str, object]:
    v84_router, v84_classifier = load_v84_router_artifact(V84_MODEL_PATH)
    return {
        "v84_router": v84_router,
        "v84_classifier": v84_classifier,
        "v76_router": load_v76_router_artifact(V76_MODEL_PATH),
        "v75_model": load_v75_model_artifact(V75_MODEL_PATH)[1],
    }


def test_v84_target_selection_is_deterministic_quoted_and_prior_disjoint() -> None:
    metadata = [
        {"id": f"{question_type}-{index:03d}", "type": question_type}
        for question_type in ("bridge", "comparison")
        for index in range(8)
    ]
    excluded = {"bridge-000", "comparison-000"}
    quotas = {"bridge": 3, "comparison": 2}
    development = select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="development",
        question_type_quotas=quotas,
    )
    confirmation = select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="confirmation",
        question_type_quotas=quotas,
    )
    assert development == select_stage_ids(
        metadata,
        excluded_ids=excluded,
        stage="development",
        question_type_quotas=quotas,
    )
    assert len(development) == len(confirmation) == 5
    assert not (set(development) & set(confirmation))
    assert not (set(development) & excluded)
    assert not (set(confirmation) & excluded)


def test_selection_outputs_execute_v84_and_all_registered_controls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.jsonl"
    outputs = write_selection_outputs([_scored_row()], path=path, **_models())
    assert outputs == read_jsonl(path)
    assert tuple(outputs[0]["methods"]) == METHODS
    assert set(CONTROL_METHODS) < set(METHODS)
    assert V76_CANDIDATE_METHOD in CONTROL_METHODS
    assert PREFIX3_METHOD in CONTROL_METHODS
    assert PREFIX4_METHOD in CONTROL_METHODS
    candidate = outputs[0]["methods"][CANDIDATE_METHOD]
    assert candidate["action"] in ACTIONS
    assert candidate["comparison_probability"] is not None
    assert candidate["retained_count"] in {3, 4}
    assert (
        candidate["selected_ids"]
        == outputs[0]["methods"][V76_CANDIDATE_METHOD]["selected_ids"][
            : candidate["retained_count"]
        ]
    )


def test_v84_selection_outputs_reject_gold_fields(tmp_path: Path) -> None:
    row = _scored_row()
    row["answer"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs([row], path=tmp_path / "selection.jsonl", **_models())


def test_v84_registered_protocol_matches_code_constants() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["strict_gates"] == STRICT_GATES
    assert protocol["safety_eligibility"] == SAFETY_ELIGIBILITY
    assert protocol["noninferiority_envelope"] == NONINFERIORITY_ENVELOPE
