# 洪水预警与协同处置系统 V2

这是一个面向领导演示与防汛业务部门的洪水风险研判系统。项目把“风险总览、影响问答、协同处置”串成一条完整的演示主线，同时保留数据管理、多代理协作、可靠性审计等运行能力。

## 当前前端定位

前端已经收敛为一套“指挥简报式”工作台，核心叙事为：

1. `风险总览`
   用全局风险结论、趋势和重点对象建立当前态势认知。
2. `影响问答`
   围绕重点对象进行专业问答，先解释影响过程和依据，再给出研判结论。
3. `协同处置`
   展示智能体如何拆解任务、调用工具、组织处置方案并进入人工确认。

次级页面仍然保留：

- `数据管理`
- `多代理协作`
- `可靠性与审计`

## 代码架构

### 后端

- `flood_system/api.py`
  暴露 `/v2/*` 接口。
- `flood_system/system.py`
  系统启动与平台装配。
- `flood_system/repository.py`
  SQLite 数据访问层。
- `flood_system/v2/`
  V2 核心能力，包括平台编排、Copilot、区域处置、通知网关、多代理与评测等。

### 前端

前端基于 `React + TypeScript + Vite + react-router-dom + framer-motion`。

#### 目录分层

- [`frontend/src/App.tsx`](/d:/graduation_project/frontend/src/App.tsx)
  只负责路由态判断、控制台状态编排、页面装配和跨页面跳转逻辑。
- [`frontend/src/config/consoleConfig.ts`](/d:/graduation_project/frontend/src/config/consoleConfig.ts)
  集中维护页面元信息、展示文案常量、对象类型映射和运行状态文本。
- [`frontend/src/components`](/d:/graduation_project/frontend/src/components)
  页面级展示组件与功能面板。
- [`frontend/src/pages`](/d:/graduation_project/frontend/src/pages)
  页面布局壳层。
- [`frontend/src/hooks/useV2OperatorConsole.ts`](/d:/graduation_project/frontend/src/hooks/useV2OperatorConsole.ts)
  前端核心状态来源，负责拉取事件、风险、问答、方案、代理、审计等数据，并封装交互动作。
- [`frontend/src/lib/displayText.ts`](/d:/graduation_project/frontend/src/lib/displayText.ts)
  展示文本格式化工具。
- [`frontend/src/lib/consoleFormatting.ts`](/d:/graduation_project/frontend/src/lib/consoleFormatting.ts)
  控制台内部通用格式化逻辑，例如时间、趋势、缓存状态、CSV 处理等。

#### 本轮前端结构优化

为了让 `App.tsx` 更清晰，本轮把原先内联在主文件里的大块 UI 继续拆分成独立模块：

- [`frontend/src/components/CopilotMessageBubble.tsx`](/d:/graduation_project/frontend/src/components/CopilotMessageBubble.tsx)
  负责问答消息的结构化展示。
- [`frontend/src/components/OperationPanel.tsx`](/d:/graduation_project/frontend/src/components/OperationPanel.tsx)
  负责方案审批队列与行动建议面板。
- [`frontend/src/components/MultiAgentDesk.tsx`](/d:/graduation_project/frontend/src/components/MultiAgentDesk.tsx)
  负责多代理协作、任务时间线、共享记忆与评测面板。
- [`frontend/src/components/ReliabilityAuditDesk.tsx`](/d:/graduation_project/frontend/src/components/ReliabilityAuditDesk.tsx)
  负责可靠性、告警、审计与归档面板。

同时，页面元信息与常量不再继续混在主文件里，而是统一进入 [`frontend/src/config/consoleConfig.ts`](/d:/graduation_project/frontend/src/config/consoleConfig.ts)。

## 页面与组件关系

### 风险总览

- [`frontend/src/pages/OverviewPage.tsx`](/d:/graduation_project/frontend/src/pages/OverviewPage.tsx)
- [`frontend/src/components/OverviewHero.tsx`](/d:/graduation_project/frontend/src/components/OverviewHero.tsx)
- [`frontend/src/components/PriorityObjectPanel.tsx`](/d:/graduation_project/frontend/src/components/PriorityObjectPanel.tsx)
- [`frontend/src/components/SignalTimeline.tsx`](/d:/graduation_project/frontend/src/components/SignalTimeline.tsx)

### 影响问答

- [`frontend/src/pages/CopilotPage.tsx`](/d:/graduation_project/frontend/src/pages/CopilotPage.tsx)
- [`frontend/src/components/CopilotMessageBubble.tsx`](/d:/graduation_project/frontend/src/components/CopilotMessageBubble.tsx)
- [`frontend/src/components/CopilotContextPanel.tsx`](/d:/graduation_project/frontend/src/components/CopilotContextPanel.tsx)

### 协同处置

- [`frontend/src/pages/OperationsPage.tsx`](/d:/graduation_project/frontend/src/pages/OperationsPage.tsx)
- [`frontend/src/components/ExecutionFlowBoard.tsx`](/d:/graduation_project/frontend/src/components/ExecutionFlowBoard.tsx)
- [`frontend/src/components/ToolExecutionSummary.tsx`](/d:/graduation_project/frontend/src/components/ToolExecutionSummary.tsx)
- [`frontend/src/components/OperationPanel.tsx`](/d:/graduation_project/frontend/src/components/OperationPanel.tsx)
- [`frontend/src/components/RegionalProposalHistoryPanel.tsx`](/d:/graduation_project/frontend/src/components/RegionalProposalHistoryPanel.tsx)

## 本地运行

### 后端

```powershell
& 'C:\Users\Administrator\anaconda3\python.exe' -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000
```

可用地址：

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 前端

```powershell
Set-Location d:\graduation_project\frontend
npm.cmd install
npm.cmd run dev
```

## 数据构建

如果需要重建碑林区演示数据集：

```powershell
C:\Users\Administrator\anaconda3\python.exe -m flood_system.data_pipeline.beilin_dataset --root d:\graduation_project build --sync-demo-db
```

## 环境变量

- `DASHSCOPE_API_KEY`
- `FLOOD_LLM_MODEL`
- `FLOOD_LLM_API_URL`
- `FLOOD_LLM_PROTOCOL`
- `FLOOD_LLM_TIMEOUT_SECONDS`
- `FLOOD_LLM_PROMPT_PROFILES_PATH`
- `FLOOD_SUPERVISOR_LOOP_ENABLED`
- `FLOOD_SUPERVISOR_LOOP_INTERVAL_SECONDS`
- `FLOOD_HOUSEKEEPING_INTERVAL_SECONDS`

如果本地通过 [`api_key.txt`](/d:/graduation_project/api_key.txt) 管理密钥，请先同步到实际运行环境。

## 验证状态

本次前端结构优化完成后，以下命令已通过：

```powershell
Set-Location d:\graduation_project\frontend
npm.cmd run test
npm.cmd run build
```

说明：

- 测试与构建都通过。
- PowerShell/Conda profile 仍会在命令结束后输出一段本机环境噪声，但不影响前端测试和打包结果。
