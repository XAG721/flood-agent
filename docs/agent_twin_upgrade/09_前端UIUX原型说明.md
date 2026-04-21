# AgentTwin Flood V3 前端 UI/UX 原型说明

## 1. 设计目标

当前前端已经有较完整的指挥控制台框架，`V3` 不建议推翻重做，而是基于现有导航、状态管理和审批弹窗继续扩展。

## 2. 当前前端承接点

可直接承接升级的入口：

- `frontend/src/App.tsx`
- `frontend/src/hooks/useV2OperatorConsole.ts`
- `frontend/src/pages/OverviewPage.tsx`
- `frontend/src/pages/OperationsPage.tsx`
- `frontend/src/pages/AgentsPage.tsx`
- `frontend/src/components/GlobalRegionalProposalDialog.tsx`

## 3. 新增页面/面板建议

## 3.1 影响态势总览屏

新增组件建议：

- `DigitalTwinImpactScreen`
- `ImpactObjectMap`
- `ImpactChainSummaryBoard`

展示重点：

- 当前天气事件
- 影响区域
- 重点对象分布
- 待审批 proposal 数

## 3.2 影响链面板

新增组件：

- `ImpactChainPanel`
- `ImpactEvidenceDrawer`

交互：

- 点击对象 -> 展开影响链
- 点击证据 -> 查看来源、摘要和引用位置

## 3.3 多智能体会商面板

新增组件：

- `AgentCouncilPanel`
- `AgentOpinionCard`

交互：

- 展示各角色观点
- 高亮冲突项
- 展示 AuditAgent 阻断原因

## 3.4 情景推演面板

新增组件：

- `ScenarioDecisionLab`
- `ScenarioScoreBoard`

交互：

- 横向比较多个方案
- 查看各方案得分与取舍

## 3.5 proposal 审批弹窗升级

现有组件：

- `GlobalRegionalProposalDialog.tsx`

升级方向：

- 增加影响链摘要
- 增加责任部门与执行时限
- 增加审批级别
- 增加证据抽屉入口

## 3.6 分众预警面板

新增组件：

- `AudienceWarningPanel`
- `MessageDiffViewer`

展示：

- 领导版
- 部门版
- 社区版
- 公众版

## 4. 页面信息架构建议

### 一级导航

- 总览
- 影响链
- 会商与推演
- 方案审批
- 分众预警
- 数据管理
- 审计与复盘

### 保留的现有导航映射

- `总览` 对应当前 `Overview`
- `会商与推演` 可承接当前 `Operations + Agents`
- `数据管理` 保持当前 `Data`
- `审计与复盘` 可承接当前 `Reliability`

## 5. 典型交互链路

1. 用户进入总览页。
2. 点击重点对象查看影响链。
3. 切换到会商页查看候选方案和智能体观点。
4. 进入 proposal 审批页确认动作。
5. 审批通过后查看分众预警草稿。
6. 在审计页查看证据与留痕。

## 6. 视觉建议

1. 总览页突出“影响对象”和“待决策事项”，弱化纯技术指标。
2. 会商页强调角色分工和冲突点，而不是聊天记录堆叠。
3. 分众预警页突出不同受众版本之间的差异和一致性状态。

## 7. 适配建议

- 桌面端优先，适配答辩大屏展示
- 审批弹窗和证据抽屉支持较大信息密度
- 地图、影响链和方案对比支持联动筛选
