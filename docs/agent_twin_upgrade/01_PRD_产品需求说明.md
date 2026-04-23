# AgentTwin Flood 产品需求说明

## 1. 产品定位

`AgentTwin Flood` 是一个面向甲方演示、联调验证和后续产品化扩展的 **数字孪生智能体洪水预警生产级 demo 测试品**。

它不是毕业设计原型，也不是最终生产系统。当前阶段的目标是在有限演示区域和固定主链路内，展示一套可信、可操作、可降级、可追溯的智能体洪水预警闭环。

核心价值：

- 把天气或洪水事件转化为对象级影响链。
- 把影响链转化为可审批的行动 proposal。
- 把 approved proposal 转化为分众 warning drafts。
- 通过数字孪生主屏、智能体会商和审计中心向甲方展示系统能力。

## 2. 目标用户

- 甲方演示观众：关注系统能力、视觉效果、链路完整性和可产品化潜力。
- 指挥员：关注是否能看清态势、审批动作、生成预警和追溯依据。
- 值班员：关注是否能快速选择对象、追问智能体、查看 proposal 和执行状态。
- 技术联调人员：关注接口边界、数据结构、SSE、降级和日志审计。

## 3. 本期目标

### 3.1 产品目标

- 形成一个可运行、可演示、可联调的 V3 demo。
- 首页呈现数字孪生智能体指挥台，而不是普通后台。
- 主屏内完成 `对象选择 -> 智能体追问 -> proposal 生成 -> 人工审批 -> warning 生成`。
- 二级页面支持会商透明、行动处置、数据维护和审计复盘。
- 后端以 `/v3/*` 聚合接口驱动前端，保留 `/v2/*` 审批闭环。

### 3.2 非目标

- 不承诺直接接入真实政务通知网关。
- 不承诺覆盖多城市、多灾种、多事件并行。
- 不把水动力高精度仿真作为本期核心。
- 不允许模型绕过人工审批直接执行高风险动作。

## 4. 主业务链路

```text
事件输入或预置事件
  -> V3 twin overview 聚合
  -> 重点对象空间联动
  -> 智能体会商与追问
  -> proposal 生成
  -> 人工审批闸门
  -> 分众 warning drafts
  -> 审计与复盘
```

## 5. 核心功能

### 5.1 数字孪生主屏

首页 `/` 是演示主舞台，必须包含：

- 真实 Cesium 城市场景。
- 重点对象点位与空间 spotlight。
- proposal/warning 状态图层。
- 左侧态势和对象列表。
- 右侧指挥台、证据阶梯和闭环微流程。
- 智能体对话抽屉入口。

### 5.2 智能体会商

系统需要展示多 Agent 的角色输出，而不是只展示一个大模型结论。

至少包含：

- Impact / Action / Resource / Warning / Audit 角色输出。
- 角色差异与证据对照。
- Supervisor decision path。
- Open questions。
- Blocked by。
- Audit decision。

### 5.3 Proposal 审批

proposal 必须保留人工审批闸门。

前端需要支持：

- 主屏右侧轻量审批。
- `/operations` 详细审批与执行编排。
- 批准、驳回、审批意见。
- 审批后自动刷新主屏状态。

### 5.4 分众预警

approved proposal 可以生成多受众 warning drafts。

默认受众：

- 领导版。
- 部门版。
- 社区版。
- 公众版。

每份 warning draft 需要包含受众、渠道、内容和证据摘要。

### 5.5 审计与可靠性

系统需要向甲方说明为什么可信。

至少展示：

- 审计记录。
- 权限边界。
- Agent council audit。
- proposal -> warning -> audit 闭环。
- SSE / 模型 / 数据降级状态。

## 6. API 要求

前端优先消费 `/v3/*` 聚合接口：

- `GET /v3/events/{event_id}/twin-overview`
- `GET /v3/events/{event_id}/objects/{object_id}`
- `GET /v3/events/{event_id}/agent-council`
- `POST /v3/events/{event_id}/dialog`
- `POST /v3/events/{event_id}/proposals/generate`
- `POST /v3/proposals/{proposal_id}/warnings/generate`
- `GET /v3/events/{event_id}/stream`

`/v2/*` 保留用于审批、通知、执行和审计落库。

## 7. 演示验收标准

- 首页第一屏具备数字孪生指挥台观感。
- 甲方能看到真实 3D 画布和对象空间联动。
- 能从主屏打开智能体对话并基于对象追问。
- 能生成 proposal 并进行人工审批。
- 能从 approved proposal 生成分众 warning drafts。
- `/agents` 能展示会商差异和证据对照。
- `/reliability` 能展示审计和闭环可信性。
- Cesium、LLM、SSE 或部分数据失败时不白屏，并给出降级状态。

## 8. 交付标准

本 demo 交付时应满足：

- 代码可构建。
- 核心定向测试通过。
- 文档与产品口径一致。
- 前端具备面向甲方展示的视觉完成度。
- 后端 V3 聚合接口能驱动主链路。
- 保留后续产品化扩展路径。
