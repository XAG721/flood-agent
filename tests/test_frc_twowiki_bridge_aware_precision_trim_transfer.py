from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.hotpot_graph_router import (
    load_router_artifact as load_v76_router_artifact,
)
from research.frc_rag.hotpot_three_route_router import (
    load_router_artifact as load_v78_router_artifact,
)
from research.frc_rag.musique_anchor_default_router import (
    load_router_artifact as load_v80_router_artifact,
)
from research.frc_rag.musique_target_three_route_router import (
    load_router_artifact as load_v79_router_artifact,
)
from research.frc_rag.twowiki_bridge_aware_precision_trim_router import (
    ACTIONS,
    CANDIDATE_METHOD,
    load_router_artifact as load_v83_router_artifact,
)
from research.frc_rag.twowiki_bridge_aware_precision_trim_transfer import (
    CONTROL_METHODS,
    METHODS,
    NONINFERIORITY_ENVELOPE,
    STRICT_GATES,
    select_stage_ids,
    validate_registered_protocol,
    write_selection_outputs,
)
from research.frc_rag.twowiki_cascaded_style_residual_router import (
    CANDIDATE_METHOD as V82_CANDIDATE_METHOD,
    load_router_artifact as load_v82_router_artifact,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    load_router_artifact as load_v81_router_artifact,
)
from research.frc_rag.twowiki_support_path_closure import QUESTION_TYPES, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs/progressive_upgrade"
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
V78_MODEL_PATH = DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
V79_MODEL_PATH = (
    DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
)
V80_MODEL_PATH = DOCS_ROOT / "musique_anchor_default_router_model_development_v80.json"
V81_MODEL_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)
V82_MODEL_PATH = (
    DOCS_ROOT / "twowiki_cascaded_style_residual_router_model_development_v82.json"
)
V83_MODEL_PATH = (
    DOCS_ROOT / "twowiki_bridge_aware_precision_trim_router_model_development_v83.json"
)
PROTOCOL_PATH = DOCS_ROOT / "twowiki_bridge_aware_precision_trim_protocol_v83.json"


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


def _scored_row(case_id: str = "synthetic-v83") -> dict[str, object]:
    specs = (
        ("a", "Alpha", "Alpha cites Beta.", 1.0),
        ("b", "Beta", "Beta cites Alpha.", 0.2),
        ("c", "Gamma", "Unrelated high score.", 0.9),
        ("d", "Delta", "Another distractor.", 0.8),
        ("e", "Epsilon", "Another distractor.", 0.7),
        ("f", "Zeta", "Another distractor.", 0.6),
    )
    return {
        "dataset": "2WikiMultiHopQA",
        "source": "synthetic",
        "id": case_id,
        "question": ("Which film has the director who was born later, Alpha or Beta?"),
        "required_roles": ["procedure", "answer"],
        "candidates": [_candidate(*spec) for spec in specs],
    }


def _models() -> dict[str, object]:
    v83_router, v83_classifier = load_v83_router_artifact(V83_MODEL_PATH)
    v82_router, v82_classifier = load_v82_router_artifact(V82_MODEL_PATH)
    return {
        "v83_router": v83_router,
        "v83_classifier": v83_classifier,
        "v82_router": v82_router,
        "v82_classifier": v82_classifier,
        "v81_router": load_v81_router_artifact(V81_MODEL_PATH),
        "v80_router": load_v80_router_artifact(V80_MODEL_PATH),
        "v79_router": load_v79_router_artifact(V79_MODEL_PATH),
        "v78_router": load_v78_router_artifact(V78_MODEL_PATH),
        "v76_router": load_v76_router_artifact(V76_MODEL_PATH),
        "v75_model": load_v75_model_artifact(V75_MODEL_PATH)[1],
    }


def test_v83_target_selection_is_deterministic_balanced_and_prior_disjoint() -> None:
    metadata = [
        {"_id": f"{question_type}-{index:03d}", "type": question_type}
        for question_type in QUESTION_TYPES
        for index in range(8)
    ]
    excluded = {f"{question_type}-000" for question_type in QUESTION_TYPES}
    quotas = {question_type: 2 for question_type in QUESTION_TYPES}
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
    assert len(development) == len(confirmation) == 8
    assert not (set(development) & set(confirmation))
    assert not (set(development) & excluded)
    assert not (set(confirmation) & excluded)


def test_selection_outputs_execute_v83_and_every_registered_control(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.jsonl"
    outputs = write_selection_outputs([_scored_row()], path=path, **_models())
    assert outputs == read_jsonl(path)
    assert tuple(outputs[0]["methods"]) == METHODS
    assert set(CONTROL_METHODS) < set(METHODS)
    assert V82_CANDIDATE_METHOD in CONTROL_METHODS
    candidate = outputs[0]["methods"][CANDIDATE_METHOD]
    assert candidate["action"] in ACTIONS
    assert candidate["bridge_probability"] is not None
    assert candidate["retained_count"] in {4, 5}
    assert (
        candidate["selected_ids"]
        == outputs[0]["methods"][V82_CANDIDATE_METHOD]["selected_ids"][
            : candidate["retained_count"]
        ]
    )


def test_v83_selection_outputs_reject_gold_fields(tmp_path: Path) -> None:
    row = _scored_row()
    row["answer"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden gold fields"):
        write_selection_outputs([row], path=tmp_path / "selection.jsonl", **_models())


def test_v83_registered_protocol_matches_code_constants() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_registered_protocol(protocol)
    assert protocol["strict_gates"] == STRICT_GATES
    assert protocol["noninferiority_envelope"] == NONINFERIORITY_ENVELOPE
