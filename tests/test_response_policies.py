from types import SimpleNamespace

from flood_system.response_workflow.candidate_discovery import point_in_polygon
from flood_system.response_workflow.feedback_policy import (
    build_review_recommendations,
    classify_feedback,
    feedback_dedupe_key,
    missing_evidence_types,
)
from flood_system.response_workflow.models import (
    FeedbackCategory,
    FeedbackRequest,
    OperatorRole,
)


def feedback_request(**updates) -> FeedbackRequest:
    payload = {
        "operator_id": "field-1",
        "operator_role": OperatorRole.FIELD_OPERATOR,
        "terminal_id": "terminal-1",
        "summary": "现场处置更新",
    }
    payload.update(updates)
    return FeedbackRequest(**payload)


def test_point_in_polygon_includes_boundary_and_excludes_outside_point():
    ring = [(108.0, 34.0), (109.0, 34.0), (109.0, 35.0), (108.0, 35.0), (108.0, 34.0)]

    assert point_in_polygon(108.5, 34.5, ring) is True
    assert point_in_polygon(108.0, 34.5, ring) is True
    assert point_in_polygon(109.5, 34.5, ring) is False


def test_feedback_classification_uses_structured_fields_before_keywords():
    category, reason = classify_feedback(
        feedback_request(summary="需要协调支援", resource_gap="缺少排水泵")
    )

    assert category == FeedbackCategory.RESOURCE_SHORTAGE
    assert "资源缺口" in reason


def test_feedback_dedupe_key_normalizes_whitespace():
    first = feedback_request(summary="道路 已 封控", evidence=[{"type": "现场照片", "value": "photo-1"}])
    second = feedback_request(summary="道路已封控", evidence=[{"value": "photo-1", "type": "现场照片"}])

    assert feedback_dedupe_key(first) == feedback_dedupe_key(second)


def test_missing_evidence_and_review_recommendations_are_deterministic():
    task = SimpleNamespace(task_id="TASK-1", event_id="EVENT-1", required_evidence=["现场照片", "积水深度"])
    missing = missing_evidence_types(task, [{"type": "现场照片", "value": "photo-1"}])
    feedback = SimpleNamespace(category=FeedbackCategory.RESOURCE_SHORTAGE)

    assert missing == ["积水深度"]
    recommendations = build_review_recommendations(
        [feedback],
        has_escalations=True,
        has_duplicates=False,
        has_unresolved_tasks=True,
    )
    assert len(recommendations) == 3
    assert any("资源台账" in item for item in recommendations)
