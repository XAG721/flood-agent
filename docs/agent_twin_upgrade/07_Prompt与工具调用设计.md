# AgentTwin Flood AgentTwin Prompt 与工具调用设计

## 1. 设计目标

Prompt 设计的目标不是让模型自由发挥，而是约束模型稳定输出结构化、可审计、可落地、可被前端直接消费的结果。

这里的“前端直接消费”特指：

- 数字孪生城市洪涝态势主屏
- 智能体对话控制台
- 行动处置页
- 审计复盘页

## 2. Prompt 分层

## 2.1 系统层 Prompt

定义全局规则：

- 只能基于提供的事实与证据作答
- 不得虚构对象、地点、资源、政策
- 高风险动作必须标注审批要求
- 输出必须符合 JSON Schema
- 输出应尽量可被前端主屏和对话控制台直接解释与展示

## 2.2 角色层 Prompt

每个智能体单独定义：

- 职责
- 允许使用的输入
- 输出字段
- 禁止行为
- 前端呈现目标

所谓“前端呈现目标”，是指该 Prompt 的输出最终将主要显示在哪个界面，例如：

- 主屏态势摘要
- 对象风险解释
- 方案比较区域
- 审批依据说明

## 2.3 任务层 Prompt

针对具体任务生成约束，例如：

- 影响链生成
- 多方案推演
- 分众消息生成
- 审计校验
- 对话追问解释

其中，“对话追问解释”是 `AgentTwin` 新前端体验中的重点任务类型。

## 3. 工具调用原则

1. 优先调用系统已有数据接口，不直接猜测。
2. 工具调用失败时显式返回错误，不允许 silently fallback。
3. 审计智能体必须能看到其他智能体调用过的证据摘要。
4. 面向前端的回答应包含足够的摘要字段，方便直接展示在主屏或对话控制台中。

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

补充要求：

- 应包含可映射到前端对象卡片或地图节点的 `object_id`
- 应包含可用于主屏摘要或对话解释的 `recommended_action`

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

补充要求：

- `tradeoff_summary` 应能直接显示在前端方案比较区
- `score` 和 `actions` 应能支持行动处置页和对话控制台联动

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

补充要求：

- 审计消息应可直接用于前端审批阻断提示
- `issues.message` 应使用适合人工阅读的表述

### 5.4 对话控制输出

建议增加统一对话响应结构：

```json
{
  "answer": "当前学校周边道路存在积水扩散风险，建议优先排查接送通道。",
  "summary": "学校接送通道风险上升",
  "recommended_actions": ["排查通道", "设置临时警示", "准备绕行方案"],
  "follow_up_prompts": ["为什么判定这里风险升高？", "如果现在需要转移，首要动作是什么？"],
  "evidence_refs": ["monitor_001", "case_school_route"]
}
```

该结构应可直接用于智能体对话控制台展示。

## 6. 与当前 现有平台 Prompt 的关系

当前 `flood_system/platform/prompt_profiles.json` 已定义：

- `object_advisory`
- `copilot_chat`
- `regional_decision`
- `proposal_draft`
- `regional_analysis_package`
- `execution_bundle`
- `execution_summary`

`AgentTwin` 建议在此基础上新增：

- `weather_event_parse`
- `impact_chain_generate`
- `scenario_generate`
- `agent_council_summarize`
- `audience_warning_generate`
- `audit_verify`
- `agent_dialog_explain`
- `focus_object_follow_up`

## 7. 失败兜底策略

1. 解析失败：返回结构化错误和待补充字段。
2. 证据不足：生成保守建议，不允许输出高风险 proposal。
3. 消息冲突：阻断发布，仅返回冲突报告。
4. 对话上下文缺失：提示用户当前缺失哪些对象或事件上下文，不直接编造答案。
