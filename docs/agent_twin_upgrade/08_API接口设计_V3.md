# AgentTwin Flood V3 API 接口设计

## 1. 设计原则

1. 新接口统一使用 `/api/v3/*`。
2. 与当前 `/v2/*` 并行存在。
3. `V3` 负责分析与生成前链路，`V2` 继续承接 proposal、通知、执行和审计闭环。
4. 接口设计应支持“数字孪生智能体洪水预警系统前端”的主屏联动与对话控制。

## 2. 天气事件解析

### `POST /api/v3/weather-events/parse`

请求：

```json
{
  "text": "未来3小时A街道有强降雨，局地可能出现城市内涝。",
  "operator_area": "A街道"
}
```

响应：

```json
{
  "weather_event_id": "WEA-001",
  "hazard_type": "heavy_rain",
  "warning_level": "yellow",
  "affected_area": ["A街道"],
  "time_window": "未来3小时",
  "certainty": "possible"
}
```

## 3. 影响链生成

### `POST /api/v3/events/{event_id}/impact-chains/generate`

响应：

```json
{
  "event_id": "EVT-001",
  "impact_chains": [
    {
      "chain_id": "CHAIN-001",
      "object_id": "underpass_001",
      "impact_type": "车辆滞留",
      "secondary_impact": "交通中断",
      "recommended_action": "安排巡查并准备临时警戒",
      "confidence": 0.84,
      "evidence_refs": ["case_01", "policy_02"]
    }
  ]
}
```

## 4. 情景方案生成

### `POST /api/v3/events/{event_id}/scenarios/generate`

响应：

```json
{
  "scenarios": [
    {
      "scenario_id": "SCN-001",
      "name": "标准防御方案",
      "score": 0.84,
      "score_breakdown": {
        "timeliness": 0.88,
        "resource_fit": 0.79,
        "public_disruption": 0.72
      },
      "actions": []
    }
  ]
}
```

## 5. 多智能体会商

### `POST /api/v3/events/{event_id}/agent-council/run`

响应：

```json
{
  "council_id": "COUNCIL-001",
  "agent_outputs": [],
  "final_recommendation": {},
  "audit_result": {}
}
```

## 6. proposal 生成与桥接

### `POST /api/v3/events/{event_id}/proposals/generate`

说明：

- 将 `V3` 分析结果落成 `ActionProposal`
- 最终仍写入当前 proposal 队列

响应：

```json
{
  "proposal_id": "PRP-001",
  "status": "pending",
  "bridged_to_v2": true
}
```

## 7. 分众预警生成

### `POST /api/v3/proposals/{proposal_id}/warnings/generate`

响应：

```json
{
  "leader_message": "...",
  "department_tasks": [],
  "community_message": "...",
  "public_message": "...",
  "cap_like_alert": {}
}
```

## 8. 复盘接口

### `POST /api/v3/events/{event_id}/postmortem/generate`

说明：

- 生成事件复盘摘要
- 更新经验与知识库

## 9. 面向前端的接口要求

为了支撑“数字孪生智能体洪水预警系统前端”，接口还应满足以下要求：

1. 能支持总览主屏加载态势、对象和风险层数据。
2. 能支持点击对象后的详情联动。
3. 能支持智能体对话控制请求与返回。
4. 能支持 proposal 状态实时刷新。
5. 能支持分众预警内容回显与版本切换。

## 10. SSE / WebSocket 事件建议

### SSE 事件类型

- `impact_chain_ready`
- `scenario_ready`
- `agent_council_progress`
- `proposal_generated`
- `warnings_ready`
- `audit_blocked`
- `twin_overview_updated`
- `focus_object_updated`

其中：

- `twin_overview_updated` 用于刷新数字孪生主屏态势
- `focus_object_updated` 用于刷新当前焦点对象联动区域

## 11. 错误码建议

| 错误码 | 含义 |
|---|---|
| `V3_PARSE_FAILED` | 天气事件解析失败 |
| `V3_IMPACT_INSUFFICIENT_DATA` | 影响链所需数据不足 |
| `V3_AUDIT_BLOCKED` | 审计阻断 |
| `V3_PERMISSION_BLOCKED` | 权限边界阻断 |
| `V3_INCONSISTENT_MESSAGES` | 分众消息不一致 |

## 12. 与现有 V2 的兼容建议

1. 审批仍走现有：
   - `POST /v2/proposals/{proposal_id}/approve`
   - `POST /v2/proposals/{proposal_id}/reject`
2. 通知与执行日志仍落当前：
   - `NotificationDraft`
   - `ExecutionLogEntry`
3. `V3` API 只扩展，不破坏当前控制台运行。
