# AgentTwin Flood 生产级 Demo 测试品

`AgentTwin Flood` 是一套面向甲方演示、联调验证和后续产品化扩展的数字孪生智能体洪水预警 demo。当前实现路线是在现有 V2 业务闭环上演进 V3 聚合能力：后端仍以 `flood_system` 为唯一服务进程，前端以 `frontend` 为唯一正式入口，`3D_visual` 作为 Cesium 模型、场景配置和校准逻辑来源。

核心演示链路：

```text
事件 -> 数字孪生态势 -> 重点对象联动 -> 智能体会商 -> proposal 审批 -> 分众 warning -> 执行留痕 -> 审计闭环
```

## 当前能力

- 首页 `/` 已重构为数字孪生智能体指挥主屏，包含左侧态势带、中央 Cesium 三维画布、右侧 proposal / warning 闭环指挥台和智能体对话抽屉。
- 三维画布已接入 `3D_visual` 的 CityEngine GLB 模型资源，并抽出 `frontend/src/lib/cityengineCalibration.ts` 复用源坐标归一化和模型校准逻辑。
- 三维展示层已支持风险热区、扩散圈、发光联动路径、proposal / warning 状态标识，以及 `Command flythrough` 指挥巡航。
- 后端新增 `/v3/*` 聚合接口，承接 twin overview、focus object、agent council、dialog、proposal 生成、warning 生成和 SSE 实时事件。
- `/v2/*` 继续承接人工审批、通知、执行日志和审计落库，避免重写已有闭环内核。
- 演示主库可通过脚本重建，固定支撑 `event_demo_beilin_primary` 主链路。

## 主要目录

- `flood_system/api.py`：FastAPI 统一入口，支持 `FLOOD_DB_PATH` 指向演示库。
- `flood_system/v2/`：现有审批、通知、审计、执行和多智能体运行基础。
- `flood_system/v3/`：AgentTwin 聚合读模型、影响链、会商、proposal 与 warning 前链路。
- `frontend/src/components/DigitalTwinImpactScreen.tsx`：数字孪生智能体主屏。
- `frontend/src/components/DigitalTwinCesiumCanvas.tsx`：Cesium 三维画布与业务点位联动。
- `frontend/src/lib/cityengineCalibration.ts`：CityEngine GLB 源坐标解析、归一化和校准矩阵。
- `3D_visual/`：三维模型校准查看器与资源来源，不作为长期并行前端产品。
- `scripts/rebuild_demo_db.py`：重建生产级 demo 演示主库。
- `scripts/inspect_demo_db.py`：检查演示主库闭环完整性。
- `scripts/start-demo.ps1`：一键重建/检查演示库并启动前后端。
- `docs/agent_twin_upgrade/`：AgentTwin/V3 设计、实施、测试、演示和数据库治理文档。

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

如需保留现有演示库：

```powershell
.\scripts\start-demo.ps1 -SkipRebuild
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
npm.cmd run dev
```

打开：

```text
http://127.0.0.1:5173
```

## 关键接口边界

- `/v3/events/{event_id}/twin-overview`
- `/v3/events/{event_id}/objects/{object_id}`
- `/v3/events/{event_id}/agent-council`
- `/v3/events/{event_id}/dialog`
- `/v3/events/{event_id}/proposals/generate`
- `/v3/proposals/{proposal_id}/warnings/generate`
- `/v3/events/{event_id}/stream`

V2 仍保留并承担：

- proposal 审批/驳回
- notification draft 与 execution log
- audit record
- reliability / closure 追溯

## 验证命令

```powershell
C:\Users\Administrator\anaconda3\python.exe scripts\inspect_demo_db.py
C:\Users\Administrator\anaconda3\python.exe -m pytest tests/test_system.py -k "v3_api_exposes_twin_overview_dialog_and_stream or v3_proposal_generation_and_warning_bridge_reuses_v2_closure"
Set-Location d:\graduation_project\frontend
npm.cmd run build
npx.cmd vitest run src/App.test.tsx -t "总览页" --reporter=basic --testTimeout=10000
npx.cmd vitest run src/App.test.tsx -t "方案处置页" --reporter=basic --testTimeout=10000
```

说明：当前 Cesium 构建仍会提示 chunk 较大，`protobufjs` 也会输出 `eval` 警告，这是三维依赖带来的既有构建警告，不影响当前 demo 功能。

## 文档入口

- [文档索引](./docs/README.md)
- [AgentTwin/V3 文档包](./docs/agent_twin_upgrade/README.md)
- [甲方演示脚本](./docs/agent_twin_upgrade/16_甲方演示脚本.md)
