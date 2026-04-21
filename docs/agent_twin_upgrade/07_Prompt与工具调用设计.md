# AgentTwin Flood V3 Prompt 与工具调用设计

## 1. 设计目标

Prompt 设计的目标不是让模型“自由发挥”，而是约束模型稳定输出结构化、可审计、可落地的结果。

## 2. Prompt 分层

## 2.1 系统层 Prompt

定义全局规则：

- 只能基于提供的事实与证据作答
- 不得虚构对象、地点、资源、政策
- 高风险动作必须标注审批要求
- 输出必须符合 JSON Schema

## 2.2 角色层 Prompt

每个智能体单独定义：

- 职责
- 允许使用的输入
- 输出字段
- 禁止行为

## 2.3 任务层 Prompt

针对具体任务生成约束，例如：

- 影响链生成
- 多方案推演
- 分众消息生成
- 审计校验

## 3. 工具调用原则

1. 优先调用系统已有数据接口，不直接猜测。
2. 工具调用失败时显式返回错误，不 silently fallback。
3. 审计智能体必须能看到其他智能体调用过的证据摘要。

## 4. 推荐工具清单

| 工具名 | 用途 |
|---|---|
| `get_weather_event_context` | 获取天气解析上下文 |
| `search_impact_graph` | 查询知识图谱关系 |
| `retrieve_rag_evidence` | 检索政策、案例、画像 |
| `get_resource_status` | 获取资源状态 |
| `get_operator_policy` | 获取审批与权限边界 |
| `list_action_templates` | 获取动作模板 |
| `build_cap_like_alert` | 生成统一 Alert Object |
| `check_message_consistency` | 校验分众消息一致性 |

## 5. 输出约束

### 5.1 影响链生成

```json
{
  "impact_chains": [
    {
      "object_id": "underpass_001",
      "impact_type": "车辆滞留",
      "secondary_impact": "交通中断",
      "recommended_action": "安排巡查并准备临时警戒",
      "confidence": 0.84,
      "evidence_refs": ["case_underpass", "policy_route_control"]
    }
  ]
}
```

### 5.2 方案生成

```json
{
  "scenarios": [
    {
      "name": "标准防御方案",
      "score": 0.82,
      "actions": [],
      "tradeoff_summary": "执行成本较低，但对脆弱群体保护一般"
    }
  ]
}
```

### 5.3 审计输出

```json
{
  "audit_passed": false,
  "issues": [
    {
      "type": "missing_evidence",
      "message": "建议封路，但未引用任何政策或历史依据"
    }
  ]
}
```

## 6. 与当前 V2 Prompt 的关系

当前 `flood_system/v2/prompt_profiles.json` 已经定义了：

- `object_advisory`
- `copilot_chat`
- `regional_decision`
- `proposal_draft`
- `regional_analysis_package`
- `execution_bundle`
- `execution_summary`

`V3` 建议在此基础上新增：

- `weather_event_parse`
- `impact_chain_generate`
- `scenario_generate`
- `agent_council_summarize`
- `audience_warning_generate`
- `audit_verify`

## 7. 失败兜底策略

1. 解析失败：返回结构化错误和待补充字段。
2. 证据不足：生成保守建议，不允许输出高风险 proposal。
3. 消息冲突：阻断发布，仅返回冲突报告。
