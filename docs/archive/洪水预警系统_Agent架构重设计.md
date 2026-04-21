# 洪水预警系统 Agent 架构重设计

## 1. 设计目标
Agent 架构重设计的目标，不是让系统变成完全自治的多智能体平台，而是在保持防汛业务可控性的前提下，把高认知负载任务交给可检索、可调用工具、可协商的专家 Agent 处理。当前实现遵循以下原则：

- 主流程保持确定性控制
- 专家 Agent 以工具调用方式工作
- 高风险动作仍需审批和规则校验
- 协商采用有限轮次，不做自由辩论
- 记忆与知识增强必须能留痕、可检索、可复用

## 2. 总体架构
当前系统采用“确定性编排层 + 工具调用型专家 Agent + 协商 Agent”两层结构。

### 2.1 确定性编排层
确定性编排层负责：
- 事件状态推进
- 预测结果接入
- 阈值判断与预警等级生成
- 阶段切换
- 审批创建与执行边界控制
- 正式通知与日志管理

编排层当前由 `LangGraph` 主流程和系统内部控制逻辑共同完成，并不由大模型主控。

### 2.2 专家 Agent 层
当前实现的专家 Agent 包括：
- `ForecastAgent`
- `RiskAssessmentAgent`
- `CommunicationAgent`
- `EvacuationPlanningAgent`
- `ResourceCoordinationAgent`
- `CompensationAgent`
- `PostmortemAgent`

这些 Agent 都已经从“系统提前拼装上下文、模型只做生成”升级为“最小种子上下文 + 自主调工具 + 结构化输出”的形式。

## 3. 工具调用式 Agent 设计
### 3.1 运行模式
每个 Agent 启动时只拿到以下种子上下文：
- `incident_id`
- `area_id`
- `stage`
- `risk_level`
- 少量模式标记，例如 `warning/response`

其余事实由 Agent 自主通过工具查询，例如：
- 当前状态
- 最新预测
- 最近观测
- 区域画像
- 资源状态
- 补偿输入
- 近期事件日志
- 前序 Agent 输出
- 冲突卡片
- 三类 RAG
- 长期记忆

### 3.2 工具边界
当前工具分为两类。

只读业务工具：
- `get_incident_state`
- `get_latest_forecast`
- `get_recent_observations`
- `get_area_profile`
- `get_resource_status`
- `get_compensation_inputs`
- `get_recent_event_logs`
- `get_latest_agent_outputs`
- `get_conflict_cards`
- `query_rag`
- `query_long_term_memory`
- `route_plan`

高风险副作用工具仍由确定性流程控制，不开放给专家 Agent 自主触发，例如：
- `send_notification`
- `create_approval_task`
- `run_dno_forecast`

## 4. 专家 Agent 职责分工
### 4.1 ForecastAgent
负责解释 DNO 预测结果、时间窗口和不确定性，不负责发布行动命令。

### 4.2 RiskAssessmentAgent
负责综合预测、观测、区域画像、案例知识和长期记忆，给出风险优先级、重点区域和最低资源需求。

### 4.3 CommunicationAgent
负责根据当前状态、规则文档和已收敛的冲突结论，生成政府端和公众端信息。其信息源优先读取最终冲突卡片，而不是直接使用未收敛的单方意见。

### 4.4 EvacuationPlanningAgent
负责生成路线、安置点匹配、辅助转移建议和清空顺序。

### 4.5 ResourceCoordinationAgent
负责分析可用资源、可同时覆盖能力、资源缺口和无法支撑区域。

### 4.6 CompensationAgent
负责根据住户申报、公共资产受损、核查记录、规则文档和长期记忆生成初步补偿筛查结果。

### 4.7 PostmortemAgent
负责在事件结束后读取日志、冲突卡片、前序 Agent 输出和长期记忆，生成复盘建议和反思记忆。

## 5. 协商型 Agent 设计
为了处理系统中最典型的三类矛盾，当前实现新增了四个协商相关 Agent。

### 5.1 FairnessReviewAgent
用于从脆弱人群、公平性和可达性角度审视疏散方案，识别：
- 未覆盖的脆弱群体
- 不可达路线
- 不适配安置点
- 需要补充的辅助转移措施

### 5.2 CompensationExceptionReviewAgent
用于从个案合理性角度审视补偿初筛结果，识别：
- 困难家庭例外
- 材料不全但核查基本属实的案件
- 证据冲突案件
- 应转人工复核的案件

### 5.3 ResponseConflictResolverAgent
负责响应阶段两类冲突的最终收敛：
- 风险优先级与资源约束冲突
- 疏散效率与脆弱人群公平性冲突

### 5.4 CompensationConflictResolverAgent
负责补偿阶段规则筛查与个案例外之间的最终收敛，输出初步通过、人工复核和暂缓三类队列。

## 6. 三类冲突的协商逻辑
### 6.1 风险优先级与资源约束冲突
参与方：
- `RiskAssessmentAgent`
- `ResourceCoordinationAgent`
- `ResponseConflictResolverAgent`

触发条件：
- 高风险区域数超过可同时覆盖能力
- 存在 `resource_gaps`
- 存在 `unsupported_zones`

输出结果：
- 最终优先区域顺序
- 约束条件下的资源投放建议
- 是否需要人工升级协调

### 6.2 疏散效率与脆弱人群公平性冲突
参与方：
- `EvacuationPlanningAgent`
- `FairnessReviewAgent`
- `ResponseConflictResolverAgent`

触发条件：
- 脆弱群体未被覆盖
- 路线不可达或风险过高
- 安置点不适配重点人群

输出结果：
- 修正后的路线与安置点方案
- 辅助转移要求
- 是否需要人工介入

### 6.3 补偿规则刚性与个案合理性冲突
参与方：
- `CompensationAgent`
- `CompensationExceptionReviewAgent`
- `CompensationConflictResolverAgent`

触发条件：
- 存在个案例外候选
- 存在证据冲突
- 规则结果与困难情形不一致

输出结果：
- 初步通过名单
- 人工复核名单
- 暂缓名单

## 7. 结构化输出设计
为了支持冲突检测和协商，当前 AgentRecommendation 已增加 `structured_payload`。典型字段包括：

- Risk：`priority_zones`、`vulnerable_zones`、`estimated_people_at_risk`、`minimum_resource_needs`
- Resource：`available_capacity`、`dispatchable_units`、`resource_gaps`、`unsupported_zones`
- Evacuation：`route_assignments`、`shelter_assignments`、`assisted_transfer_groups`、`estimated_clearance_order`
- FairnessReview：`fairness_risks`、`unserved_vulnerable_groups`、`accessibility_conflicts`、`recommended_adjustments`
- Compensation：`proposed_approve_ids`、`proposed_manual_review_ids`、`proposed_defer_ids`、`rule_based_reasons`
- CompensationExceptionReview：`exception_candidate_ids`、`hardship_flags`、`evidence_conflicts`、`exception_reasons`
- Resolver：`accepted_positions`、`rejected_positions`、`final_priority_order`、`final_actions`、`escalation_needed`

## 8. 冲突卡片
系统为每次协商生成 `ConflictCard`，作为展示和留痕载体。卡片内容包括：
- 冲突类型与阶段
- 触发原因
- 参与方
- 协商轮次摘要
- 最终收敛结果
- 是否需要人工复核
- 当前状态（resolved / escalated）

操作端主要展示冲突卡片，而不是完整自由对话。完整中间过程保存在事件日志中。

## 9. 与记忆和知识层的关系
协商并不是独立模块，而是与 RAG 和长期记忆联动的。

- Risk 与 Resource 协商会读取历史策略和资源记忆
- 补偿协商会读取历史补偿经验和反思记忆
- 协商结束后，摘要会写回长期记忆库
- Postmortem 会读取冲突卡片并沉淀更高层次的复盘结论

因此，Agent 架构已经从“多角色生成器”演进为“可检索、可调用工具、可协商、可写回记忆”的研究型多智能体系统。

## 10. 当前版本结论
当前版本的 Agent 架构具备以下特点：
- 不是完全自治，而是可控的专家协作
- 不是自由辩论，而是有限轮次协商
- 不是单纯文本生成，而是结构化输出与冲突收敛
- 不是一次性上下文，而是可复用的长期记忆增强

这使得系统更适合灾害业务场景，也更适合作为论文中的“智能体增强型洪水预警响应系统”进行论证与实验分析。
