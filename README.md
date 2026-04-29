# AgentTwin Flood 生产级 Demo 测试品

`AgentTwin Flood` 是一套面向甲方演示、联调验证和后续产品化扩展的数字孪生智能体洪水预警 demo。当前项目已经统一为一套 AgentTwin 平台：后端仍以 `flood_system` 为唯一服务进程，前端以 `frontend` 为唯一正式入口，`3D_visual` 只作为 Cesium 模型、场景配置和校准逻辑来源。

核心演示链路：

```text
事件 -> 数字孪生态势 -> 重点对象联动 -> 智能体会商 -> proposal 审批 -> 分众 warning -> 执行留痕 -> 审计闭环
```

## 当前能力

- 首页 `/` 已重构为数字孪生智能体指挥主屏，包含左侧态势带、中央 Cesium 三维画布、右侧 proposal / warning 闭环指挥台和智能体对话抽屉。
- 三维画布已接入 `3D_visual` 的 CityEngine GLB 模型资源，并抽出 `frontend/src/lib/cityengineCalibration.ts` 复用源坐标归一化和模型校准逻辑。
- 三维展示层已支持风险热区、动态积水面、水位柱、发光联动路径、proposal / warning 状态标识，以及 6 段式 `Play command story` 指挥叙事镜头。
- 前端支持 `VITE_DEMO_MODE=true` 演示模式，可固定首页事件、对象、proposal、warning 和会商结果，降低现场数据波动对展示的影响。
- 后端提供统一的 AgentTwin 能力入口：`/agent-twin/*` 负责主屏聚合、对象聚焦、智能体会商、对话、proposal 生成、warning 生成和 SSE 实时事件。
- 后端提供统一的平台能力入口：`/platform/*` 负责审批、通知、执行日志、审计、数据维护和可靠性治理。
- 演示主库可通过脚本重建，固定支撑 `event_demo_beilin_primary` 主链路。

## 主要目录与结构边界

- `flood_system/api.py`：FastAPI 统一装配入口，并提供 `/agent-twin/*` 与 `/platform/*` 两类公开能力入口。
- `flood_system/config.py`：运行配置与 `FLOOD_DB_PATH` 解析。
- `flood_system/http/`：AgentTwin HTTP 路由层。
- `flood_system/infrastructure/sse.py`：SSE 编码与流式基础设施。
- `flood_system/schemas/`：HTTP router 使用的 schema import surface。
- `flood_system/storage/schema.py`：SQLite 运行时表结构与索引定义，避免 `repository.py` 继续承载建表大块文本。
- `flood_system/`：承载审批、通知、审计、执行、多智能体和 AgentTwin 聚合读模型等后端能力。
- `frontend/src/api/agentTwinApi.ts`：前端 AgentTwin 主链路 API 门面。
- `frontend/src/api/*Api.ts`：前端平台能力 API 门面。
- `frontend/src/fixtures/agentTwinDemoMode.ts`：前端演示模式固定数据与结构化降级样例。
- `frontend/src/features/dataManagement/dataModels.ts`：数据维护页使用的空档案、空资源状态工厂。
- `frontend/src/state/agentTwinSelectors.ts`：主屏多源态势、影响链图谱和 Agent 差异对照的派生状态。
- `frontend/src/components/DigitalTwinImpactScreen.tsx`：数字孪生智能体主屏。
- `frontend/src/components/DigitalTwinCesiumCanvas.tsx`：Cesium 三维画布与业务点位联动。
- `frontend/src/lib/cityengineCalibration.ts`：CityEngine GLB 源坐标解析、归一化和校准矩阵。
- `3D_visual/`：三维模型校准查看器与资源来源，不作为长期并行前端产品。
- `scripts/rebuild_demo_db.py`：重建生产级 demo 演示主库。
- `scripts/inspect_demo_db.py`：检查演示主库闭环完整性。
- `scripts/start-demo.ps1`：一键重建/检查演示库并启动前后端。
- `docs/agent_twin_upgrade/`：AgentTwin 设计、实施、测试、演示和数据库治理文档。

## 一键演示

推荐现场演示使用：

```powershell
.\scripts\start-demo.ps1
```

脚本会自动：

- 重建 `data/flood_warning_system_demo.db`
- 运行演示库检查
- 设置后端 `FLOOD_DB_PATH`
- 启动后端 `http://127.0.0.1:8000`
- 启动前端 `http://127.0.0.1:5173`
- 打开首页
- 默认设置 `VITE_DEMO_MODE=true`，让前端优先使用固定演示快照

如需保留现有演示库：

```powershell
.\scripts\start-demo.ps1 -SkipRebuild
```

如需关闭前端固定演示态、完全消费实时平台数据：

```powershell
.\scripts\start-demo.ps1 -LiveData
```

## 手动运行

### 1. 重建并检查演示主库

```powershell
C:\Users\Administrator\anaconda3\python.exe scripts\rebuild_demo_db.py --force
C:\Users\Administrator\anaconda3\python.exe scripts\inspect_demo_db.py
```

### 2. 启动后端

```powershell
$env:FLOOD_DB_PATH="D:\graduation_project\data\flood_warning_system_demo.db"
C:\Users\Administrator\anaconda3\python.exe -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000
```

可用地址：

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 3. 启动前端

```powershell
Set-Location d:\graduation_project\frontend
npm.cmd install
$env:VITE_DEMO_MODE="true"
npm.cmd run dev
```

打开：

```text
http://127.0.0.1:5173
```

## 关键接口边界

AgentTwin 主链路：

- `/agent-twin/events/{event_id}/twin-overview`
- `/agent-twin/events/{event_id}/objects/{object_id}`
- `/agent-twin/events/{event_id}/agent-council`
- `/agent-twin/events/{event_id}/dialog`
- `/agent-twin/events/{event_id}/proposals/generate`
- `/agent-twin/proposals/{proposal_id}/warnings/generate`
- `/agent-twin/events/{event_id}/stream`

平台闭环能力：

- proposal 审批/驳回
- notification draft 与 execution log
- audit record
- reliability / closure 追溯
- 数据维护、RAG 维护和运行健康检查

## 验证命令

```powershell
C:\Users\Administrator\anaconda3\python.exe scripts\inspect_demo_db.py
C:\Users\Administrator\anaconda3\python.exe -m pytest
Set-Location d:\graduation_project\frontend
npm.cmd run build
npm.cmd run test -- --run --reporter=basic --testTimeout=10000
```

说明：当前 Cesium 构建仍会提示 chunk 较大，`protobufjs` 也会输出 `eval` 警告，这是三维依赖带来的既有构建警告，不影响当前 demo 功能。

## 文档入口

- [文档索引](./docs/README.md)
- [AgentTwin 文档包](./docs/agent_twin_upgrade/README.md)
- [甲方演示脚本](./docs/agent_twin_upgrade/16_甲方演示脚本.md)
- [可交付重构说明](./docs/agent_twin_upgrade/17_可交付重构说明.md)
