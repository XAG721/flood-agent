from __future__ import annotations


ROLE_RANK = {
    "observer": 0,
    "street_operator": 1,
    "district_operator": 2,
    "commander": 3,
}

ACTION_MIN_ROLE = {
    "event_create": "district_operator",
    "event_ingest": "district_operator",
    "simulation_ingest": "district_operator",
    "proposal_resolve": "commander",
    "proposal_draft_edit": "commander",
    "runtime_admin_write": "commander",
    "dataset_manage": "commander",
    "supervisor_control": "commander",
    "agent_replay": "commander",
    "archive_run": "commander",
    "evaluation_run": "district_operator",
}

ACTION_LABELS = {
    "event_create": "创建事件",
    "event_ingest": "导入监测数据",
    "simulation_ingest": "导入模拟结果",
    "proposal_resolve": "处理区域请示",
    "proposal_draft_edit": "编辑请示草稿",
    "runtime_admin_write": "修改运行期数据",
    "dataset_manage": "管理数据集任务",
    "supervisor_control": "控制后台巡检",
    "agent_replay": "重放智能体任务",
    "archive_run": "执行归档清理",
    "evaluation_run": "运行评测任务",
}


class AuthorizationError(PermissionError):
    pass


def normalize_operator_role(role: str | None, *, default: str = "commander") -> str:
    value = (role or default).strip().lower()
    if value not in ROLE_RANK:
        raise AuthorizationError(f"Unknown operator_role: {role}")
    return value


def ensure_operator_role(action: str, role: str | None) -> str:
    normalized = normalize_operator_role(role)
    minimum_role = ACTION_MIN_ROLE.get(action)
    if minimum_role is None:
        return normalized
    if ROLE_RANK[normalized] < ROLE_RANK[minimum_role]:
        raise AuthorizationError(
            f"{action} requires {minimum_role} privileges, but the current operator role is {normalized}."
        )
    return normalized


def list_operator_capabilities(role: str | None) -> dict[str, object]:
    normalized = normalize_operator_role(role)
    capabilities: dict[str, bool] = {}
    for action, minimum_role in ACTION_MIN_ROLE.items():
        capabilities[action] = ROLE_RANK[normalized] >= ROLE_RANK[minimum_role]
    return {
        "operator_role": normalized,
        "role_rank": ROLE_RANK[normalized],
        "capabilities": capabilities,
        "action_labels": ACTION_LABELS,
    }
