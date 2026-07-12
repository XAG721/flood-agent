from __future__ import annotations

from .models import TaskStatus


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
)

ALLOWED_TASK_TRANSITIONS = frozenset(
    {
        (TaskStatus.DRAFT, TaskStatus.PENDING_APPROVAL),
        (TaskStatus.PENDING_APPROVAL, TaskStatus.DRAFT),
        (TaskStatus.PENDING_APPROVAL, TaskStatus.ISSUED),
        (TaskStatus.ISSUED, TaskStatus.ACKNOWLEDGED),
        (TaskStatus.ISSUED, TaskStatus.BLOCKED),
        (TaskStatus.ISSUED, TaskStatus.ESCALATED),
        (TaskStatus.ACKNOWLEDGED, TaskStatus.IN_PROGRESS),
        (TaskStatus.ACKNOWLEDGED, TaskStatus.BLOCKED),
        (TaskStatus.ACKNOWLEDGED, TaskStatus.ESCALATED),
        (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
        (TaskStatus.IN_PROGRESS, TaskStatus.PARTIALLY_COMPLETED),
        (TaskStatus.IN_PROGRESS, TaskStatus.PENDING_VERIFICATION),
        (TaskStatus.IN_PROGRESS, TaskStatus.ESCALATED),
        (TaskStatus.BLOCKED, TaskStatus.ACKNOWLEDGED),
        (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
        (TaskStatus.BLOCKED, TaskStatus.ESCALATED),
        (TaskStatus.PARTIALLY_COMPLETED, TaskStatus.PENDING_VERIFICATION),
        (TaskStatus.PARTIALLY_COMPLETED, TaskStatus.BLOCKED),
        (TaskStatus.PARTIALLY_COMPLETED, TaskStatus.ESCALATED),
        (TaskStatus.PENDING_VERIFICATION, TaskStatus.COMPLETED),
        (TaskStatus.PENDING_VERIFICATION, TaskStatus.IN_PROGRESS),
        (TaskStatus.PENDING_VERIFICATION, TaskStatus.ESCALATED),
        (TaskStatus.ESCALATED, TaskStatus.ACKNOWLEDGED),
        (TaskStatus.ESCALATED, TaskStatus.IN_PROGRESS),
        (TaskStatus.ESCALATED, TaskStatus.BLOCKED),
        (TaskStatus.TAKEN_OVER, TaskStatus.PENDING_VERIFICATION),
        (TaskStatus.TAKEN_OVER, TaskStatus.BLOCKED),
        *(
            (status, TaskStatus.CANCELLED)
            for status in TaskStatus
            if status not in TERMINAL_TASK_STATUSES
        ),
        *(
            (status, TaskStatus.TAKEN_OVER)
            for status in {
                TaskStatus.ISSUED,
                TaskStatus.ACKNOWLEDGED,
                TaskStatus.IN_PROGRESS,
                TaskStatus.BLOCKED,
                TaskStatus.ESCALATED,
                TaskStatus.PARTIALLY_COMPLETED,
            }
        ),
        *(
            (status, TaskStatus.WAIVED)
            for status in TaskStatus
            if status not in TERMINAL_TASK_STATUSES
        ),
    }
)


def can_transition(source: TaskStatus, target: TaskStatus) -> bool:
    return (source, target) in ALLOWED_TASK_TRANSITIONS


def ensure_transition(source: TaskStatus, target: TaskStatus) -> None:
    if not can_transition(source, target):
        raise ValueError(f"cannot transition task from {source.value} to {target.value}")
