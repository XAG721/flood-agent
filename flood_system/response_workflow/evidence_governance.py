from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Protocol, Sequence


TASK_FIELD_QUERIES: dict[str, str] = {
    "trigger_condition": "触发条件、预警等级、阈值和适用范围",
    "risk_object": "风险对象、位置、脆弱性和影响范围",
    "responsible_party": "责任单位、责任岗位、权限和协同主体",
    "action": "处置动作、执行步骤、先后顺序和操作要求",
    "deadline": "接收、开始、完成、核验时限和数值阈值",
    "resource_dependency": "人员、车辆、设备、物资和资源依赖",
    "feedback_requirement": "反馈内容、附件、位置、时间和证据要求",
    "escalation_condition": "催办、升级、改派和人工接管条件",
    "exception_condition": "例外、受阻、终止、替代和特殊情况",
}

# These fields are required before an AI-assisted task draft may be approved. Resource
# and exception fields remain visible as MISSING when the source corpus legitimately has
# no applicable provision; they must never be silently fabricated.
REQUIRED_TASK_FIELDS: tuple[str, ...] = (
    "trigger_condition",
    "risk_object",
    "responsible_party",
    "action",
    "deadline",
    "feedback_requirement",
    "escalation_condition",
)

ROLE_DEFAULT_FIELDS: dict[str, tuple[str, ...]] = {
    "condition": ("trigger_condition",),
    "object": ("risk_object",),
    "responsibility": ("responsible_party",),
    "procedure": ("action",),
    "exception": ("exception_condition",),
}

FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "trigger_condition": ("当", "达到", "触发", "预警", "阈值", "when", "trigger"),
    "risk_object": ("对象", "通道", "道路", "学校", "社区", "地铁", "object"),
    "responsible_party": ("责任", "部门", "岗位", "防办", "住建", "交警", "responsible"),
    "action": ("应", "需", "执行", "核查", "组织", "启动", "封控", "execute", "procedure"),
    "deadline": ("时限", "截止", "分钟", "小时", "日内", "within", "deadline", "minute", "hour"),
    "resource_dependency": ("资源", "人员", "车辆", "设备", "物资", "泵", "工具", "resource", "equipment"),
    "feedback_requirement": ("反馈", "回报", "上报", "照片", "附件", "记录", "feedback", "report"),
    "escalation_condition": ("升级", "催办", "改派", "接管", "超时", "escalate", "takeover"),
    "exception_condition": ("若", "如", "例外", "受阻", "终止", "替代", "不足", "无法", "except", "insufficient"),
}

SLOT_SEPARATOR = "｜"


def task_field_slots() -> list[str]:
    return [f"{field}{SLOT_SEPARATOR}{query}" for field, query in TASK_FIELD_QUERIES.items()]


def infer_task_field_support(
    text: str,
    roles: Iterable[str],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, float]:
    metadata = metadata or {}
    support: dict[str, float] = {}
    explicit = metadata.get("task_field_support")
    if isinstance(explicit, Mapping):
        for field, raw_score in explicit.items():
            if str(field) not in TASK_FIELD_QUERIES:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            support[str(field)] = max(0.0, min(1.0, score))

    selection = metadata.get("_evidence_selection")
    if isinstance(selection, Mapping):
        covered_slots = selection.get("covered_slots", [])
        if isinstance(covered_slots, Sequence) and not isinstance(
            covered_slots, (str, bytes)
        ):
            for slot in covered_slots:
                if not isinstance(slot, Mapping):
                    continue
                slot_text = str(slot.get("text", ""))
                field = slot_text.split(SLOT_SEPARATOR, 1)[0].strip()
                if field not in TASK_FIELD_QUERIES:
                    continue
                try:
                    score = float(slot.get("support_score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                support[field] = max(support.get(field, 0.0), min(1.0, score))

    normalized_roles = {str(role) for role in roles}
    for role in normalized_roles:
        for field in ROLE_DEFAULT_FIELDS.get(role, ()):
            support[field] = max(support.get(field, 0.0), 0.6)

    lowered = text.casefold()
    for field, keywords in FIELD_KEYWORDS.items():
        if any(keyword.casefold() in lowered for keyword in keywords):
            support[field] = max(support.get(field, 0.0), 0.75)
    return {field: round(score, 4) for field, score in sorted(support.items())}


def build_field_evidence_map(evidence: Sequence[object]) -> dict[str, list[str]]:
    mapping = {field: [] for field in TASK_FIELD_QUERIES}
    for item in evidence:
        source_id = str(getattr(item, "source_id", ""))
        support = getattr(item, "field_support", {})
        if not source_id or not isinstance(support, Mapping):
            continue
        for field, score in support.items():
            if field in mapping and isinstance(score, (int, float)) and score >= 0.18:
                mapping[field].append(source_id)
    return {
        field: list(dict.fromkeys(source_ids))
        for field, source_ids in mapping.items()
    }


@dataclass(frozen=True, slots=True)
class NliResult:
    relation: str
    confidence: float
    model_version: str
    status: str = "completed"
    error: str = ""


class NliAdapter(Protocol):
    model_version: str

    def classify(self, premise: str, hypothesis: str) -> NliResult: ...


class UnavailableNliAdapter:
    """Fail-explicit default; production must inject a versioned NLI adapter."""

    model_version = "nli-unavailable"

    def classify(self, premise: str, hypothesis: str) -> NliResult:
        return NliResult(
            relation="unavailable",
            confidence=0.0,
            model_version=self.model_version,
            status="unavailable",
            error="No versioned NLI model adapter is configured; deterministic conflict rules remain active.",
        )


def candidate_nli_pairs(evidence: Sequence[object], *, limit: int = 100):
    pairs = []
    for left, right in combinations(evidence, 2):
        if getattr(left, "source_type", "") == "workflow_contract" or getattr(
            right, "source_type", ""
        ) == "workflow_contract":
            continue
        left_fields = {
            field
            for field, score in getattr(left, "field_support", {}).items()
            if isinstance(score, (int, float)) and score >= 0.18
        }
        right_fields = {
            field
            for field, score in getattr(right, "field_support", {}).items()
            if isinstance(score, (int, float)) and score >= 0.18
        }
        shared_fields = sorted(left_fields & right_fields)
        if not shared_fields:
            continue
        pairs.append((left, right, shared_fields))
        if len(pairs) >= limit:
            break
    return pairs
