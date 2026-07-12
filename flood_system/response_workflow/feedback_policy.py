from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from .models import FeedbackCategory, FeedbackRequest, ResponseTask, TaskFeedback, TimelineEntry


def classify_feedback(request: FeedbackRequest) -> tuple[FeedbackCategory, str]:
    text = " ".join(filter(None, [request.summary, request.blocked_reason, request.resource_gap]))
    if request.resource_gap:
        return FeedbackCategory.RESOURCE_SHORTAGE, "检测到资源缺口字段，按资源不足反馈分类。"
    if request.blocked_reason:
        return FeedbackCategory.BLOCKED, "检测到受阻原因字段，按执行受阻反馈分类。"
    if any(keyword in text for keyword in ("协同", "支援", "增援", "协调")):
        return FeedbackCategory.COORDINATION_REQUEST, "反馈包含协同或支援请求。"
    if request.completion_percent is not None and request.completion_percent < 100:
        return FeedbackCategory.PARTIAL_COMPLETION, "完成比例低于 100%，记录为部分完成并保持任务可继续执行。"
    if request.evidence:
        return FeedbackCategory.COMPLETION, "反馈包含结构化完成证据，进入待核实流程。"
    return FeedbackCategory.SITUATION_UPDATE, "未检测到完成、受阻或协同信号，作为现场态势更新。"


def feedback_dedupe_key(request: FeedbackRequest) -> str:
    payload = {
        "summary": re.sub(r"\s+", "", request.summary).lower(),
        "blocked_reason": re.sub(r"\s+", "", request.blocked_reason or "").lower(),
        "resource_gap": re.sub(r"\s+", "", request.resource_gap or "").lower(),
        "completion_percent": request.completion_percent,
        "evidence": sorted(
            (str(item.get("type", "")).strip(), str(item.get("value", item.get("url", ""))).strip())
            for item in request.evidence
        ),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recommend_feedback_alternatives(category: FeedbackCategory) -> list[str]:
    if category == FeedbackCategory.RESOURCE_SHORTAGE:
        return [
            "核查本部门同类资源和事件级资源台账，优先内部调剂。",
            "向相邻成员单位或区级资源协调岗位申请补充资源。",
            "在不降低安全要求的前提下调整执行顺序并重新确认时限。",
        ]
    if category == FeedbackCategory.BLOCKED:
        return [
            "核实受阻位置和现场安全条件，必要时暂停原路径。",
            "由成员单位联络员协调替代路线、替代人员或替代设备。",
            "重大方案变化提交重新审批后再恢复执行。",
        ]
    return ["提交防办审核员协调相关成员单位。", "保留现场证据并明确下一次反馈时限。"]


def recommend_deadline_alternatives(reason: str) -> list[str]:
    if "未确认" in reason:
        return ["通知成员单位联络员二次确认。", "核对责任人联系方式并准备重新分派。"]
    return ["核查执行受阻原因和资源缺口。", "协调增援或重新分派并更新完成时限。"]


def review_timeline_line(entry: TimelineEntry) -> str:
    labels = {
        "task_approved_and_issued": "任务批准并下发",
        "task_rejected_to_draft": "任务退回修改",
        "task_waived": "任务批准豁免",
        "event_closed": "事件授权关闭",
    }
    return f"{entry.created_at.isoformat()}：{labels.get(entry.action, entry.action)}（{entry.actor_id}）"


def build_review_recommendations(
    feedback: Iterable[TaskFeedback],
    *,
    has_escalations: bool,
    has_duplicates: bool,
    has_unresolved_tasks: bool,
) -> list[str]:
    recommendations: list[str] = []
    if has_escalations:
        recommendations.append("复核确认时限和完成时限配置，并针对高频受阻原因预置跨部门资源调剂方案。")
    if any(item.category == FeedbackCategory.RESOURCE_SHORTAGE for item in feedback):
        recommendations.append("更新事件级资源台账和相邻单位支援清单，缩短资源缺口协调时间。")
    if has_duplicates:
        recommendations.append("保持反馈去重规则，同时要求现场人员在状态变化时补充新的时间、定位或证据值。")
    if has_unresolved_tasks:
        recommendations.append("事件关闭前逐项处理未完成、未核实或未批准豁免的任务。")
    if not recommendations:
        recommendations.append("保持当前审批、证据核实和事件台账机制，并在后续演练中复测确认时效。")
    return recommendations


def missing_evidence_types(task: ResponseTask, evidence: list[dict]) -> list[str]:
    supplied = {str(item.get("type", "")).strip() for item in evidence}
    return [required for required in task.required_evidence if required not in supplied]
