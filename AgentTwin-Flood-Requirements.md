# AgentTwin-Flood：洪水预警系统智能体化升级需求文档

> 版本：v1.0  
> 编码：UTF-8  
> 适用场景：论文系统设计、项目立项、开发交接、答辩演示  
> 升级重点：影响导向预警、多智能体会商、行动推演决策、可审计 proposal、分众预警生成  

---

## 1. 文档定位

本文用于将现有洪水预警系统升级为一个以大模型智能体为核心的影响导向辅助决策系统。升级目标不是优先提升洪水数值预测精度，也不依赖大量实时传感器数据，而是重点解决以下问题：

- 如何把“天气会怎样”转化为“会造成什么影响”；
- 如何把“影响”进一步转化为“可执行行动”；
- 如何通过多智能体会商提高建议的完整性、可解释性和可审计性；
- 如何面向领导、部门、社区和公众生成不同版本的预警信息；
- 如何保留人在回路审批，避免大模型直接越权决策。

本系统升级后的目标形态可以概括为：

```text
天气预警输入
  → 影响链推理
  → 多智能体会商
  → 行动情景推演
  → 可审计 proposal
  → 人工审批
  → 分众预警发布
  → 复盘学习
```

---

## 2. 现有系统基础

根据当前系统总结，现有系统已经具备较完整的原型基础，包括：

- `frontend/`：主业务控制台；
- `3D_visual/`：Cesium 三维场景展示端；
- `flood_system/api.py`：FastAPI 接口层；
- `flood_system/v2/`：风险、暴露、决策、Copilot、多智能体、通知、记忆等核心模块；
- SQLite：事件、提案、对象画像、资源状态、审计记录等运行时存储；
- RAG 文档：政策、案例、画像等知识来源。

现有系统主链路为：

```text
创建事件
→ 接收观测/模拟数据
→ 风险判定
→ 影响评估
→ 高风险触发 proposal
→ SSE 推送待审批队列
→ 指挥员审批/驳回
→ 生成通知草稿和执行日志
→ 审计与复盘
```

这条链路已经体现出当前系统最重要的设计思想：

```text
风险判断尽量确定化，建议生成尽量智能化，最终动作必须由人确认。
```

本次升级不推倒重来，而是在现有系统基础上，将其从“风险等级 + 区域建议”升级为“影响链 + 行动推演 + 多智能体会商 + 分众预警”。

---

## 3. 升级目标

### 3.1 总体目标

将系统从传统洪水预警原型升级为：

> 面向城市内涝和洪水场景的影响导向、多智能体辅助决策系统。

系统不只回答：

```text
哪里风险高？
```

而是回答：

```text
哪些对象会受影响？
会造成什么后果？
应该采取什么行动？
谁负责执行？
多长时间内执行？
哪些动作需要审批？
公众和部门分别该收到什么信息？
```

### 3.2 本期重点建设内容

本期重点建设 5 类能力：

```text
1. 影响链推理
2. 行动情景推演
3. 多智能体会商
4. 分众预警生成
5. 可审计 proposal 与复盘学习
```

### 3.3 本期暂不重点建设内容

本期暂不把以下能力作为核心：

```text
1. 高精度水动力洪水预测
2. 大规模实时传感器接入
3. 视频水深识别
4. 泵站/闸门自动控制
5. 真实生产级政务系统部署
```

但系统架构需要为这些能力预留接口。

---

## 4. 引入智能体后解决的关键问题

### 4.1 解决问题一：天气信息难以转化为影响信息

传统系统通常可以给出“暴雨黄色预警”“风险等级较高”等结论，但不能稳定回答：

```text
哪些社区会受影响？
哪些学校需要关注？
哪些下穿通道可能出现车辆滞留？
哪些地下空间需要提前提醒？
```

引入智能体后，系统通过“天气—影响—行动知识图谱”和 RAG，将气象预警、对象画像、历史易涝点、应急预案和历史案例关联起来，生成结构化影响链。

---

### 4.2 解决问题二：影响信息难以转化为行动建议

以前系统即使识别出某个对象风险较高，也常停留在“请关注”“建议加强巡查”等泛化表述。升级后，系统要求每条建议必须包含：

```text
对象
影响
行动
责任部门
执行时限
所需资源
审批要求
证据链
```

例如：

```text
对象：XX下穿通道
影响：强降雨期间可能出现车辆滞留
行动：30分钟内安排交警巡查并准备临时警戒
责任部门：交警、街道
审批要求：如实施封控，需指挥员审批
证据：历史易涝点、地势低洼、SOP规则
```

---

### 4.3 解决问题三：多部门会商难以结构化沉淀

传统会商往往依赖人工经验，气象、水务、交通、社区、应急、宣传等部门的判断分散在电话、会议和聊天记录中。系统难以形成统一证据链。

升级后，多智能体分别承担以下角色：

```text
气象解释智能体
影响评估智能体
行动规划智能体
资源约束智能体
公众沟通智能体
审计校验智能体
```

每个智能体输出结构化结论，并记录输入、输出、证据、置信度和校验结果。

---

### 4.4 解决问题四：预警内容不分对象、不可行动

同一条风险事件，对不同接收对象应生成不同内容：

| 接收对象 | 需要的信息 |
|---|---|
| 领导/指挥长 | 风险摘要、影响对象、需审批事项 |
| 应急部门 | 任务清单、责任分工、完成时限 |
| 街道/社区 | 网格员巡查清单、居民提醒话术 |
| 交警 | 重点道路、下穿通道、绕行建议 |
| 学校 | 上下学安排、家长提醒建议 |
| 公众 | 简短明确的避险建议 |

智能体系统通过“结构化预警对象 + 分众改写 + 一致性校验”解决这一问题。

---

### 4.5 解决问题五：大模型直接决策存在幻觉和越权风险

单一大模型直接生成应急建议，容易出现以下问题：

```text
地点不存在
资源不存在
建议缺少依据
高风险动作越权
公众消息夸大
部门任务前后矛盾
```

升级后，系统采用：

```text
角色智能体分工
+ 工具调用
+ 结构化输出
+ 审计智能体验证
+ 人在回路审批
```

所有高风险动作，如封路、转移、停课、正式预警发布、跨区资源调度，必须经过人工审批。

---

### 4.6 解决问题六：复盘经验难以反哺下一次决策

升级后，系统不仅记录日志，还将事件复盘结果反向更新：

```text
影响知识图谱
行动模板库
历史相似案例库
智能体提示词
置信度权重
分众消息模板
```

这样系统可以逐步从“每次重新判断”变为“带有地方经验的智能体决策系统”。

---

## 5. 近一年相关文献与设计依据

本方案的设计依据主要来自近一年关于影响导向预警、灾害智能体、数字风险孪生和 AI 应急管理的研究方向。

### 5.1 影响导向预警

WMO 对 impact-based forecasting and warning services 的核心表述是从“what the weather will be”转向“what the weather will do”，强调预警要提供及时、准确、可行动的信息，支持政府、社区和公众提前准备与响应。

参考：

- WMO Impact-based Forecast and Warning Services  
  https://wmo.int/impact-based-forecast-and-warning-services

### 5.2 LLM 在灾害管理中的作用

2025 年关于 LLM 灾害管理的综述指出，LLM 在灾害管理中的价值包括多源信息融合、响应协调、信息验证和行动建议生成；同时也强调灾害场景中必须避免幻觉、越权和不可追溯决策。

参考：

- Large language models in disaster management review, 2025  
  https://www.sciencedirect.com/science/article/pii/S2212420925004662

### 5.3 Digital Risk Twin 与人在回路

2025 年提出的 Digital Risk Twin 思路强调，灾害风险数字孪生不一定必须依赖完整自动传感器体系，也可以结合人工上报、历史案例、对象画像和现场反馈，并将 human-in-the-loop 决策作为关键机制。

参考：

- Digital risk twin for disaster risk management, 2025  
  https://www.nature.com/articles/s44304-025-00135-x

### 5.4 多风险影响预警

2025 年多风险影响预警研究指出，传统系统如果只关注单一灾害和直接淹没范围，容易忽略道路中断、医疗可达性下降、社区孤立、供电中断等间接影响。因此，系统需要从“灾害范围”转向“影响链”。

参考：

- Multi-hazard impact-based warning research, 2025  
  https://www.nature.com/articles/s44304-025-00157-5

### 5.5 CAP 标准化预警

WMO 将 CAP 描述为适用于多媒体、多灾种、多渠道传播的标准化告警格式。系统可以先实现 CAP-like JSON，以便未来对接正式预警发布渠道。

参考：

- WMO Common Alerting Protocol  
  https://wmo.int/common-alerting-protocol

---

## 6. 升级后总体业务流程

```text
1. 输入天气预警或人工创建风险事件
2. 系统解析天气风险窗口、影响区域和预警等级
3. 影响链引擎识别可能受影响对象
4. 多智能体会商生成影响判断
5. 行动沙盘生成多个候选处置方案
6. 行动评分器对方案进行排序
7. 审计智能体检查证据、权限、一致性和幻觉风险
8. 系统生成待审批 proposal
9. 指挥员审批、修改或驳回
10. 系统生成领导版、部门版、社区版、公众版通知
11. 执行过程留痕
12. 事后复盘并更新案例库、行动模板和影响知识图谱
```

核心变化是从：

```text
风险等级 → 处置建议
```

升级为：

```text
天气触发条件 → 影响链 → 行动方案 → 多智能体校验 → 分众预警 → 人工审批 → 复盘学习
```

---

## 7. 总体技术架构

建议在现有 V2 基础上增加 V3 智能体决策层。

```text
┌──────────────────────────────────────────────┐
│ 1. 大屏展示层                                  │
│ React + Cesium/地图 + ECharts + SSE/WebSocket │
│ 影响态势、智能体会商、行动沙盘、预警草稿           │
└──────────────────────────────────────────────┘
                    ↑
┌──────────────────────────────────────────────┐
│ 2. 业务接口层                                  │
│ FastAPI REST + SSE/WebSocket                  │
│ 事件、影响链、会商、proposal、分众预警             │
└──────────────────────────────────────────────┘
                    ↑
┌──────────────────────────────────────────────┐
│ 3. 智能体会商层                                │
│ Orchestrator + WeatherAgent + ImpactAgent     │
│ ActionAgent + ResourceAgent + CommunicationAgent│
│ AuditAgent                                    │
└──────────────────────────────────────────────┘
                    ↑
┌──────────────────────────────────────────────┐
│ 4. 推演决策层                                  │
│ 影响链推理、行动模板匹配、情景方案生成、评分排序       │
└──────────────────────────────────────────────┘
                    ↑
┌──────────────────────────────────────────────┐
│ 5. 知识与数据层                                │
│ 对象画像、历史案例、SOP、RAG、影响知识图谱、资源清单    │
└──────────────────────────────────────────────┘
                    ↑
┌──────────────────────────────────────────────┐
│ 6. 治理与审计层                                │
│ 权限、审批、证据链、模型输出记录、复盘学习             │
└──────────────────────────────────────────────┘
```

---

## 8. 后端模块修改方案

### 8.1 建议新增目录

```text
flood_system/v3/
├── weather_event_parser.py        # 气象预警/人工输入解析
├── impact_ontology.py             # 天气-对象-影响-行动本体
├── impact_graph.py                # 影响知识图谱
├── impact_chain_engine.py         # 影响链推理
├── action_template_store.py       # 行动模板库
├── scenario_decision_lab.py       # 行动情景推演
├── action_scorer.py               # 行动优先级评分
├── agent_roles.py                 # 智能体角色定义
├── agent_council.py               # 多智能体会商编排
├── agent_memory.py                # 智能体运行记忆
├── evidence_verifier.py           # 证据、权限、幻觉校验
├── audience_warning_center.py     # 分众预警生成
├── cap_like_alert.py              # CAP-like 结构化预警对象
├── message_consistency_checker.py # 多版本消息一致性检查
├── approval_policy.py             # 审批边界和权限策略
└── decision_memory.py             # 事件复盘与案例学习
```

### 8.2 与现有 V2 模块关系

| 现有模块 | 修改方向 |
|---|---|
| `hazard_engine.py` | 保留基础风险判定，输出作为影响链触发条件 |
| `exposure_engine.py` | 升级为影响链推理的数据来源之一 |
| `decision_engine.py` | 与行动模板、评分排序模块结合 |
| `copilot_orchestrator.py` | 升级为影响问答与指挥解释器 |
| `regional_proposals.py` | 从区域提案升级为影响驱动行动 proposal |
| `notification_gateway.py` | 升级为分众预警生成与发布草稿模块 |
| `multi_agent.py` | 扩展为多角色智能体会商机制 |
| `memory_store.py` | 扩展为决策记忆、案例记忆和复盘学习 |
| `security.py` | 增加审批边界和智能体动作权限控制 |
| `reporting.py` | 增加建议质量、行动效果、预警效果评估 |

---

## 9. 前端大屏修改方案

### 9.1 新增页面

```text
frontend/src/pages/DigitalTwinImpactScreen/
├── DigitalTwinImpactScreen.tsx
├── ImpactMapCanvas.tsx
├── ImpactChainPanel.tsx
├── AgentCouncilPanel.tsx
├── ScenarioDecisionLab.tsx
├── ActionPriorityBoard.tsx
├── AudienceWarningPanel.tsx
├── ApprovalQueuePanel.tsx
└── EvidenceTraceDrawer.tsx
```

### 9.2 大屏布局

```text
┌──────────────────────────────────────────────────────────┐
│ 顶部：事件名称、预警等级、时间窗口、当前阶段、审批状态          │
├──────────────┬────────────────────────┬──────────────────┤
│ 左侧天气输入   │ 中央地图/三维影响态势      │ 右侧影响链与会商       │
│ 预警文本       │ 重点对象、影响区域、行动点位 │ 智能体结论、证据链     │
│ 时间窗口       │ proposal 空间分布          │ 行动优先级            │
├──────────────┴────────────────────────┴──────────────────┤
│ 底部：行动时间轴、待审批 proposal、分众预警草稿、执行反馈       │
└──────────────────────────────────────────────────────────┘
```

### 9.3 页面交互

```text
点击影响对象 → 展开影响链
点击行动建议 → 查看评分和证据
点击智能体结论 → 查看输入、输出和证据
点击 proposal → 审批、驳回、编辑
点击分众预警 → 查看领导版、部门版、社区版、公众版
点击证据链 → 展开 RAG 文档、对象画像、历史案例
```

---

## 10. 功能需求

## FR-01：天气事件解析模块

### 功能目标

将气象预警、人工输入、文本描述等转换为系统可处理的结构化事件。

### 输入

```text
气象预警文本
人工录入天气说明
风险区域
预警等级
预计影响时间
已有事件上下文
```

### 输出示例

```json
{
  "event_id": "EVT-20260420-001",
  "hazard_type": "强降雨",
  "warning_level": "orange",
  "affected_area": ["A街道", "B社区"],
  "time_window": {
    "start": "2026-04-20 18:00",
    "end": "2026-04-20 21:00"
  },
  "trigger_summary": "未来3小时强降雨，低洼片区存在城市内涝风险",
  "certainty": "likely"
}
```

### 技术实现

```python
parse_warning_text(text: str) -> WeatherEvent
normalize_warning_level(level: str) -> WarningLevel
extract_time_window(text: str) -> TimeWindow
map_area_names(text: str) -> list[AreaRef]
```

### 解决的问题

解决“天气部门语言无法直接进入应急行动流程”的问题。

---

## FR-02：影响链推理模块

### 功能目标

将天气事件转化为“影响对象—影响后果—建议行动”的结构化链条。

### 影响链定义

```text
天气危险源
→ 影响区域
→ 暴露对象
→ 脆弱性原因
→ 可能影响
→ 次生影响
→ 建议行动
→ 责任部门
→ 时间窗口
→ 证据来源
→ 置信度
```

### 输出示例

```json
{
  "chain_id": "IC-001",
  "hazard": "强降雨",
  "area": "A街道",
  "object": {
    "id": "OBJ-UNDERPASS-001",
    "name": "XX下穿通道",
    "type": "underpass"
  },
  "vulnerability": [
    "历史易涝点",
    "地势低洼",
    "交通流量较大"
  ],
  "primary_impact": "车辆通行受阻",
  "secondary_impact": "救援车辆绕行时间增加",
  "recommended_action": "安排交警提前巡查并准备临时警戒",
  "responsible_department": ["交警", "街道"],
  "time_window": "30分钟内",
  "confidence": 0.78,
  "evidence": [
    {
      "type": "historical_case",
      "ref": "2023-07-XX事件复盘"
    },
    {
      "type": "sop",
      "ref": "城市内涝应急处置预案"
    }
  ]
}
```

### 技术方法

```text
RAG 检索
+ LLM 结构化抽取
+ 天气—影响—行动知识图谱
+ 规则推理
+ 图遍历
+ 置信度评分
```

### 置信度评分示例

```text
ImpactConfidence =
  0.30 × 天气触发强度
+ 0.25 × 对象脆弱性评分
+ 0.20 × 历史案例相似度
+ 0.15 × SOP 匹配度
+ 0.10 × 人工确认/上报权重
```

### 解决的问题

系统以前只能给出风险等级；升级后可以说明哪些对象会受到什么影响、为什么受到影响、应该采取什么行动。

---

## FR-03：行动情景推演模块

### 功能目标

在不依赖复杂洪水预测的情况下，对不同处置方案进行比较，帮助指挥员判断“先做什么”。

### 输入

```text
影响链
资源状态
行动模板库
部门权限
历史案例
事件时间窗口
```

### 输出示例

```json
{
  "scenario_id": "SCN-001",
  "scenario_name": "标准防御方案",
  "actions": [
    {
      "action": "通知A社区网格员巡查低洼楼栋",
      "target": "A社区",
      "department": "街道",
      "deadline": "30分钟内",
      "resource_required": ["网格员", "短信渠道"],
      "expected_effect": "降低居民误入地下车库和低洼区域风险",
      "approval_level": "street_level"
    }
  ],
  "score": {
    "impact_reduction": 0.82,
    "urgency": 0.76,
    "feasibility": 0.91,
    "resource_fit": 0.85,
    "public_disturbance": 0.22,
    "overall": 0.84
  }
}
```

### 行动评分算法

```text
ActionScore =
  0.25 × 影响降低程度
+ 0.20 × 紧急程度
+ 0.15 × 可执行性
+ 0.15 × 资源匹配度
+ 0.10 × 脆弱人群保护
+ 0.10 × 证据置信度
- 0.05 × 公众扰动成本
```

### 方案生成逻辑

```text
Step 1：读取影响链
Step 2：按影响类型匹配行动模板
Step 3：生成候选行动
Step 4：组合成保守方案、标准方案、积极方案
Step 5：检查资源约束和权限边界
Step 6：计算行动评分和方案评分
Step 7：输出推荐方案、备选方案、不推荐方案
```

### 解决的问题

系统以前能生成建议，但难以比较多个方案；升级后可以根据影响、资源、时限、扰动和审批要求对方案进行排序。

---

## FR-04：多智能体会商模块

### 功能目标

将单一 Copilot 升级为多角色智能体会商机制，避免一个大模型直接生成所有结论。

### 智能体角色

| 智能体 | 职责 | 输出 |
|---|---|---|
| 气象解释智能体 | 解析预警等级、时间窗口、影响区域 | 天气触发条件 |
| 影响评估智能体 | 生成影响链 | 受影响对象和后果 |
| 行动规划智能体 | 生成候选行动 | 行动方案 |
| 资源约束智能体 | 检查资源和执行可行性 | 可执行性结论 |
| 公众沟通智能体 | 生成分众预警文案 | 领导/部门/公众消息 |
| 审计校验智能体 | 检查证据、权限、一致性、幻觉 | 是否可提交审批 |

### 智能体调用流程

```text
Orchestrator 接收事件
  → WeatherAgent 解析天气触发条件
  → ImpactAgent 生成影响链
  → ActionAgent 生成候选行动
  → ResourceAgent 检查资源约束
  → CommunicationAgent 生成分众通知
  → AuditAgent 进行证据、权限和一致性校验
  → Orchestrator 汇总成 proposal
```

### 智能体输出格式

所有智能体必须输出结构化 JSON，禁止只输出自然语言。

```json
{
  "agent": "ImpactAgent",
  "task_id": "TASK-001",
  "conclusion": "A街道低洼社区和下穿通道存在较高影响风险",
  "confidence": 0.81,
  "evidence_refs": [
    "PROFILE-A-STREET",
    "SOP-URBAN-FLOOD-001"
  ],
  "recommended_next_step": "生成社区提醒和交警巡查行动"
}
```

### 审计校验规则

每条建议必须通过 5 类检查：

```text
1. 证据校验：是否有 RAG、对象画像、历史案例或人工输入作为依据
2. 权限校验：是否涉及封路、转移、停课、正式预警发布等高风险动作
3. 幻觉校验：地点、部门、资源是否存在于系统数据库
4. 一致性校验：部门任务和公众通知是否互相矛盾
5. 完整性校验：是否包含对象、动作、责任部门、时限、审批级别
```

### 解决的问题

解决单一大模型建议不可控、不可审计、可能越权的问题。

---

## FR-05：可审计 proposal 生成模块

### 功能目标

把智能体会商结果转化为可审批、可修改、可追踪的行动 proposal。

### Proposal 数据结构

```json
{
  "proposal_id": "PROP-001",
  "event_id": "EVT-001",
  "title": "A街道低洼片区预防性提醒与巡查方案",
  "impact_summary": "强降雨可能导致A街道低洼社区出行受阻、地下空间进水和下穿通道车辆滞留。",
  "recommended_actions": [
    {
      "action_id": "ACT-001",
      "target_object": "XX下穿通道",
      "action": "安排交警提前巡查并准备临时警戒",
      "department": "交警",
      "deadline": "30分钟内",
      "approval_required": true,
      "approval_level": "commander"
    }
  ],
  "scenario_score": 0.84,
  "confidence": 0.79,
  "evidence_chain": [
    {
      "source_type": "object_profile",
      "source_id": "OBJ-UNDERPASS-001"
    },
    {
      "source_type": "sop",
      "source_id": "SOP-URBAN-FLOOD-001"
    }
  ],
  "agent_votes": {
    "WeatherAgent": "support",
    "ImpactAgent": "support",
    "ActionAgent": "support",
    "ResourceAgent": "support_with_condition",
    "AuditAgent": "requires_approval"
  },
  "status": "pending_approval"
}
```

### 解决的问题

解决智能体输出难以直接进入审批流程的问题。

升级后，每条建议都有：

```text
建议内容
影响依据
责任部门
执行时限
审批边界
智能体投票
审计结果
```

---

## FR-06：分众预警生成模块

### 功能目标

同一条影响链，自动生成不同受众可理解、可执行的预警信息。

### 分众消息示例

#### 领导版

```text
本次强降雨预计18:00—21:00影响A街道。系统识别3类重点影响：低洼社区出行受阻、下穿通道车辆滞留、地下空间进水风险。建议30分钟内启动社区提醒和重点点位巡查；临时封控建议需指挥员审批。
```

#### 社区版

```text
请A街道网格员在30分钟内巡查低洼楼栋、地下车库入口和排水口，提醒居民避免进入地下空间，发现积水及时上报。
```

#### 公众版

```text
A街道未来3小时有较强降雨，请避开低洼路段、下穿通道和地下空间，车辆不要停放在地下车库低洼区域。
```

#### 交警版

```text
请关注XX下穿通道和XX路低洼路段，建议安排巡查力量，必要时准备临时警戒和绕行提示。
```

### CAP-like 结构化预警对象

```json
{
  "identifier": "ALERT-20260420-001",
  "sender": "AgentTwin-Flood",
  "sent": "2026-04-20T17:30:00+09:00",
  "status": "draft",
  "msgType": "alert",
  "scope": "public",
  "hazard": "强降雨",
  "severity": "orange",
  "urgency": "immediate",
  "certainty": "likely",
  "area": {
    "name": "A街道",
    "polygon": null
  },
  "impact": "低洼社区出行受阻、地下空间进水、下穿通道车辆滞留风险",
  "recommended_actions": [
    "居民避免进入地下空间",
    "社区网格员巡查低洼楼栋",
    "交警关注下穿通道"
  ],
  "effective": "2026-04-20T18:00:00+09:00",
  "expires": "2026-04-20T21:00:00+09:00"
}
```

### 解决的问题

解决传统预警口径单一、公众不可行动、部门任务不明确的问题。

---

## FR-07：大屏影响态势界面

### 功能目标

将现有主业务控制台与三维展示端融合，形成“影响导向”的数字孪生大屏。

### 页面结构

```text
DigitalTwinImpactScreen
├── 中央地图/三维场景
├── 左侧天气触发与事件信息
├── 右侧影响链面板
├── 右侧多智能体会商面板
├── 底部行动时间轴
├── 待审批 proposal 弹窗
└── 分众预警抽屉
```

### 地图图层

```text
重点对象图层
影响链图层
行动建议图层
资源状态图层
审批状态图层
历史案例图层
```

### 解决的问题

解决原三维端偏展示、主控制台偏列表，空间态势和行动决策割裂的问题。

---

## FR-08：复盘学习模块

### 功能目标

将每次事件中的智能体建议、审批结果、执行情况和最终效果沉淀为下一次推理的依据。

### 复盘记录示例

```json
{
  "event_id": "EVT-001",
  "proposal_id": "PROP-001",
  "recommended_action": "通知A社区网格员巡查",
  "approved": true,
  "executed": true,
  "execution_delay_minutes": 12,
  "observed_result": "社区反馈及时，无人员滞留",
  "effectiveness_score": 0.85,
  "lessons_learned": [
    "A社区地下车库提醒应提前至降雨前1小时",
    "公众短信应避免使用过于专业的内涝术语"
  ]
}
```

### 解决的问题

解决事后复盘只形成报告、不能反哺下一次智能体决策的问题。

---

## 11. 算法与技术改进方案

## 11.1 改进一：天气—影响—行动知识图谱推理

### 要解决的难题

```text
天气预警是气象语言；
应急处置是部门语言；
公众提醒是生活语言。
```

以前系统难以自动完成这三者之间的转换。

### 方法

构建本地化知识图谱：

```text
Hazard → Exposure → Vulnerability → Impact → Action → Department → Message
```

### 技术路线

```text
1. 从预案、历史案例、对象画像中抽取实体和关系
2. 用 LLM 生成候选三元组
3. 用规则和人工审核过滤错误关系
4. 形成影响知识图谱
5. 运行时根据天气事件检索相关对象和影响
6. 生成影响链和行动建议
```

### 技术收益

```text
从“风险等级提示”升级为“影响链推理”。
```

---

## 11.2 改进二：行动模板匹配与多指标决策评分

### 要解决的难题

系统不只是要知道“哪里有风险”，还要知道：

```text
先做什么？
谁来做？
资源够不够？
哪个方案收益最大？
哪个方案扰动最小？
```

### 方法

建立行动模板库，并对候选方案进行评分。

### 评分指标

```text
影响降低程度
紧急程度
可执行性
资源匹配度
脆弱人群保护
证据置信度
公众扰动成本
审批复杂度
```

### 技术收益

```text
从“生成一条建议”升级为“比较多个行动方案”。
```

---

## 11.3 改进三：多智能体角色分工与审计校验

### 要解决的难题

单一大模型容易：

```text
幻觉
越权
忽略资源约束
生成不可执行建议
证据不可追溯
```

### 方法

将大模型拆分为多个角色智能体，每个智能体只负责有限任务。

### 核心机制

```text
Orchestrator 统一编排
Role Agent 分工推理
Tool Calling 调用系统数据
Structured Output 保证输出格式
Verifier Agent 审计校验
Human-in-the-loop 最终审批
```

### 技术收益

```text
从“聊天式建议”升级为“可审计行动 proposal”。
```

---

## 11.4 改进四：分众预警生成与一致性校验

### 要解决的难题

同一条风险信息，不同对象需要不同表达：

```text
领导要摘要
部门要任务
社区要操作清单
公众要避险提醒
系统要结构化接口
```

### 方法

先生成统一结构化预警对象，再按受众改写。

### 技术路线

```text
ImpactChain
→ CAP-like Alert Object
→ Audience-specific Prompt
→ Message Draft
→ Consistency Checker
→ Approval
```

### 一致性校验规则

```text
风险等级是否一致
区域名称是否一致
时间窗口是否一致
行动建议是否冲突
公众消息是否夸大
部门任务是否缺少责任人
```

### 技术收益

```text
从“统一模板通知”升级为“分众、可执行、口径一致的预警发布”。
```

---

## 12. 数据需求

## 12.1 必备数据

本期不依赖大量实时传感器，但至少需要以下数据：

| 数据类型 | 用途 |
|---|---|
| 行政区划 | 确定影响区域 |
| 重点对象画像 | 社区、学校、医院、养老院、地下空间、下穿通道 |
| 历史易涝点 | 影响链推理依据 |
| 应急预案/SOP | 行动模板和审批规则 |
| 部门职责表 | 责任部门匹配 |
| 资源清单 | 行动可执行性判断 |
| 历史事件案例 | 相似案例检索和置信度评分 |
| 预警消息模板 | 分众预警生成 |
| 权限规则 | 判断是否需要审批 |

## 12.2 建议数据表

### `impact_objects`

```sql
object_id
object_name
object_type
area_id
location
vulnerability_tags
population_exposure
priority_level
historical_risk_level
```

### `impact_chains`

```sql
chain_id
event_id
hazard_type
area_id
object_id
impact_type
secondary_impact
recommended_action
confidence
evidence_json
created_at
```

### `action_templates`

```sql
template_id
action_type
applicable_impact
required_resource
responsible_department
lead_time_minutes
expected_effect
side_effect
approval_level
message_template_json
```

### `agent_deliberations`

```sql
deliberation_id
event_id
agent_name
task_type
input_json
output_json
confidence
evidence_refs
created_at
```

### `audience_warnings`

```sql
warning_id
proposal_id
audience_type
message_text
cap_like_json
status
approved_by
created_at
```

---

## 13. 接口需求

## 13.1 天气事件解析接口

```http
POST /api/v3/weather-events/parse
```

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
  "hazard_type": "强降雨",
  "warning_level": "yellow",
  "affected_area": ["A街道"],
  "time_window": "未来3小时",
  "certainty": "possible"
}
```

## 13.2 影响链生成接口

```http
POST /api/v3/events/{event_id}/impact-chains/generate
```

响应：

```json
{
  "event_id": "EVT-001",
  "impact_chains": []
}
```

## 13.3 行动方案推演接口

```http
POST /api/v3/events/{event_id}/scenarios/generate
```

响应：

```json
{
  "scenarios": [
    {
      "name": "标准防御方案",
      "score": 0.84,
      "actions": []
    }
  ]
}
```

## 13.4 多智能体会商接口

```http
POST /api/v3/events/{event_id}/agent-council/run
```

响应：

```json
{
  "council_id": "COUNCIL-001",
  "agent_outputs": [],
  "final_recommendation": {},
  "audit_result": {}
}
```

## 13.5 分众预警生成接口

```http
POST /api/v3/proposals/{proposal_id}/warnings/generate
```

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

---

## 14. 非功能需求

## 14.1 可解释性

每条建议必须展示：

```text
影响对象
影响原因
建议动作
证据来源
置信度
责任部门
审批要求
```

## 14.2 安全性

系统必须保留人在回路机制。

以下动作必须由指挥员审批，不允许智能体自动执行：

```text
封路
转移
停课
正式预警发布
资源跨区调度
```

## 14.3 可审计性

每次智能体推理都要记录：

```text
输入
输出
调用工具
引用证据
模型版本
提示词版本
操作人
审批人
时间戳
```

## 14.4 稳定性

大屏页面应支持：

```text
SSE 或 WebSocket 实时刷新
proposal 队列自动更新
智能体会商进度展示
失败重试
接口降级
```

## 14.5 可扩展性

后续应能扩展：

```text
水动力预测模型
视频识别
传感器接入
CAP XML 正式发布
移动端公众预警
真实短信网关
```

---

## 15. 验收标准

## 15.1 功能验收

系统应至少完成以下演示链路：

```text
输入一条强降雨预警
→ 系统生成结构化天气事件
→ 系统识别受影响对象
→ 生成影响链
→ 多智能体会商
→ 生成3个行动方案
→ 系统排序推荐方案
→ 生成待审批 proposal
→ 指挥员审批
→ 生成领导版、社区版、公众版、部门版消息
→ 形成审计记录
```

## 15.2 质量验收

每条 proposal 必须满足：

```text
有明确对象
有明确影响
有明确行动
有责任部门
有执行时限
有证据链
有审批级别
有智能体校验结果
```

## 15.3 展示验收

大屏至少展示：

```text
事件概况
影响对象地图
影响链列表
行动优先级
多智能体会商结论
待审批 proposal
分众预警草稿
证据链详情
```

---

## 16. 为落实开发需要额外设计的文档

## 16.1 产品需求说明书 PRD

用于明确：

```text
用户角色
业务场景
功能范围
页面需求
操作流程
验收标准
```

本文可以作为 PRD 初稿，但还需要补充页面原型和交互细节。

## 16.2 业务流程设计文档

需要画清楚：

```text
事件创建流程
影响链生成流程
智能体会商流程
proposal 审批流程
预警发布流程
复盘学习流程
```

建议使用 BPMN 或泳道图，区分：

```text
系统
值班员
指挥员
部门人员
公众
智能体
```

## 16.3 系统架构设计文档

需要明确：

```text
前端架构
后端架构
v2 与 v3 模块关系
数据库关系
RAG/知识图谱关系
智能体编排方式
部署方式
```

## 16.4 数据模型设计文档

需要详细定义：

```text
事件表
对象画像表
影响链表
行动模板表
proposal 表
智能体运行记录表
分众预警表
审计表
复盘表
```

并说明字段类型、主键、外键、索引和样例数据。

## 16.5 知识图谱与本体设计文档

需要定义：

```text
Hazard
Area
Object
Vulnerability
Impact
SecondaryImpact
Action
Department
Resource
Audience
Message
ApprovalLevel
```

以及实体关系、三元组样例、抽取规则和人工审核规则。

## 16.6 智能体设计文档

需要逐个定义智能体：

```text
角色
职责
输入
输出
可调用工具
禁止行为
提示词模板
输出 JSON Schema
失败处理方式
校验规则
```

尤其要写清楚：

```text
哪些动作智能体不能自动执行
哪些内容必须人工审批
哪些输出必须引用证据
```

## 16.7 Prompt 与工具调用设计文档

需要沉淀：

```text
系统提示词
角色提示词
影响链生成提示词
行动方案生成提示词
审计校验提示词
分众消息生成提示词
工具调用说明
输出格式约束
```

这份文档很重要，因为它直接决定智能体输出是否稳定。

## 16.8 API 接口设计文档

需要定义：

```text
REST API
SSE/WebSocket 事件
请求参数
响应字段
错误码
鉴权方式
分页方式
重试机制
```

## 16.9 前端 UI/UX 原型文档

需要设计：

```text
大屏布局
影响链面板
智能体会商面板
行动沙盘面板
proposal 审批弹窗
分众预警抽屉
证据链抽屉
复盘页面
```

建议先用 Figma、Axure 或直接使用低保真线框图确认交互，再开发。

## 16.10 测试与评测文档

需要设计：

```text
功能测试用例
接口测试用例
智能体输出评测集
幻觉检测用例
权限越权测试
分众消息一致性测试
大屏性能测试
演示场景脚本
```

---

## 17. 建议开发步骤

## 阶段 0：需求冻结与演示场景确定

### 目标

确定系统先演示哪个典型场景。

建议选择一个简化但完整的城市内涝场景：

```text
未来3小时强降雨
A街道低洼社区、学校、下穿通道、地下车库存在影响风险
系统生成行动方案和分众预警
指挥员审批
形成执行留痕
```

### 产出

```text
最终 PRD
演示故事线
角色清单
对象清单
预警输入样例
验收标准
```

---

## 阶段 1：数据与知识准备

### 开发内容

准备最小可用数据集：

```text
5—10个重点对象
3—5个历史易涝点
3—5个应急预案/SOP片段
10—20条行动模板
若干历史案例
部门职责清单
资源清单
```

### 产出

```text
对象画像表
行动模板表
影响知识图谱初版
RAG 文档集
测试事件样例
```

---

## 阶段 2：后端 v3 模块开发

### 优先开发顺序

```text
1. weather_event_parser.py
2. impact_ontology.py
3. impact_chain_engine.py
4. action_template_store.py
5. action_scorer.py
6. scenario_decision_lab.py
7. agent_council.py
8. evidence_verifier.py
9. audience_warning_center.py
10. cap_like_alert.py
```

### 与现有系统集成

保留现有：

```text
regional_proposals.py
notification_gateway.py
memory_store.py
security.py
reporting.py
```

但将它们升级为接收 v3 输出。

---

## 阶段 3：智能体与 Prompt 工程开发

### 开发内容

完成 6 个智能体：

```text
WeatherAgent
ImpactAgent
ActionAgent
ResourceAgent
CommunicationAgent
AuditAgent
```

每个智能体都要实现：

```text
固定输入格式
固定输出 JSON Schema
工具调用权限
失败兜底提示
证据引用要求
```

### 关键要求

智能体输出不能只写自然语言，必须结构化。

审计智能体必须能拦截：

```text
系统中不存在的地点
系统中不存在的资源
无证据建议
越权行动
前后不一致消息
```

---

## 阶段 4：前端大屏开发

### 开发内容

新增：

```text
DigitalTwinImpactScreen
ImpactChainPanel
AgentCouncilPanel
ScenarioDecisionLab
ActionPriorityBoard
AudienceWarningPanel
EvidenceTraceDrawer
```

### 交互重点

```text
点击影响对象 → 展开影响链
点击行动建议 → 查看评分和证据
点击智能体结论 → 查看输入输出
点击 proposal → 审批/驳回/编辑
点击分众预警 → 查看不同版本消息
```

---

## 阶段 5：审批、通知、审计闭环集成

### 开发内容

将 v3 生成的 proposal 接入现有审批链路：

```text
proposal 生成
→ SSE 推送
→ 前端弹窗
→ 指挥员审批
→ 生成分众通知
→ 写入执行日志
→ 写入审计记录
```

### 重点

不要让智能体绕过现有审批链路。

---

## 阶段 6：测试与评测

### 测试内容

```text
影响链是否正确
行动方案是否可执行
智能体是否引用证据
proposal 是否包含完整字段
公众消息是否可理解
部门任务是否明确
高风险动作是否被拦截
审批链路是否完整
```

### 智能体专项评测

建立 20—50 条测试样例：

```text
正常预警样例
缺少对象数据样例
资源不足样例
需要审批样例
地点不存在样例
公众消息夸大样例
部门任务冲突样例
```

---

## 阶段 7：演示与论文材料整理

### 产出

```text
系统架构图
业务流程图
智能体架构图
影响链示例图
数据库 ER 图
接口说明
演示脚本
核心创新点说明
系统截图
测试结果表
```

---

## 18. 最小可开发版本 MVP

如果时间有限，建议优先做一个 MVP：

```text
1. 影响链推理
2. 行动方案排序
3. 四智能体会商
4. 可审计 proposal
5. 分众预警生成
6. 大屏展示
```

四个智能体即可：

```text
气象解释智能体
影响评估智能体
行动规划智能体
审计校验智能体
```

最小闭环为：

```text
输入预警文本
→ 生成影响链
→ 生成行动方案
→ 智能体会商
→ 审计校验
→ 生成 proposal
→ 人工审批
→ 生成分众消息
→ 审计留痕
```

---

## 19. 论文或汇报可用创新点

## 创新点一：天气—影响—行动知识图谱驱动的影响导向预警

系统将预警链路从：

```text
天气 → 风险等级
```

升级为：

```text
天气 → 暴露对象 → 脆弱性 → 影响后果 → 行动建议 → 分众消息
```

技术方法包括：

```text
LLM 信息抽取
RAG
本地影响知识图谱
规则推理
影响链生成
证据追溯
```

解决“天气信息无法自动转化为可行动影响信息”的问题。

---

## 创新点二：面向数据稀缺场景的行动沙盘推演

系统不依赖大量传感器，也不依赖复杂水动力预测，而是做行动后果推演。

技术方法包括：

```text
行动模板库
影响链
资源约束
权限规则
多指标决策评分
What-if 方案比较
```

解决“知道有风险，但不知道先做什么、怎么比较方案”的问题。

---

## 创新点三：多智能体会商与可审计 proposal 生成

系统不让大模型直接替代指挥员，而是由多个角色智能体协作生成经过校验的行动 proposal。

技术方法包括：

```text
角色智能体
总控编排
工具调用
证据校验
权限校验
人在回路审批
审计留痕
```

解决“大模型建议不可控、不可审计、可能越权”的问题。

---

## 20. 最终总结

这次系统升级的本质，是把洪水预警系统从一个“风险展示和建议生成平台”，升级为一个：

```text
影响导向
多智能体会商
行动可推演
结果可审批
过程可审计
事后可学习
```

的智能决策系统。

它解决的核心难题不是“洪水到底能不能预测得更准”，而是：

```text
如何把天气风险转成对象影响，
把对象影响转成行动方案，
把行动方案转成不同角色能执行的信息，
并让整个过程可解释、可审批、可追溯。
```

