# AgentTwin Flood 生产级 Demo 测试品文档包

## 1. 文档定位

本目录用于承接 [`AgentTwin-Flood-Requirements.md`](../../AgentTwin-Flood-Requirements.md) 中的升级需求，并把当前项目推进为一个面向甲方演示、联调验证和后续产品化扩展的 **生产级 demo 测试品**。

这里的“生产级 demo”不是最终生产系统，也不是毕业设计原型，而是一个具备真实业务闭环、真实接口边界、真实前端展示效果和明确降级策略的可运行测试品。

统一产品口径如下：

- 产品名称：`AgentTwin Flood`
- 产品形态：数字孪生智能体洪水预警生产级 demo
- 后端定位：在现有 `V2` 业务闭环上演进出 `V3 / AgentTwin` 聚合能力
- 前端定位：面向甲方展示的数字孪生智能体指挥台
- 主链路：`事件 -> 数字孪生态势 -> 对象联动 -> 智能体会商 -> proposal 审批 -> 分众预警 -> 审计闭环`

## 2. 当前代码关系

当前仓库采用“在现有 V2 上演进，不推倒重来”的路线：

- 后端唯一服务入口：`flood_system/api.py`
- 应用装配入口：`flood_system/system.py`
- 运行配置：`flood_system/config.py`
- V3 HTTP 路由：`flood_system/http/v3_router.py`
- SSE 基础设施：`flood_system/infrastructure/sse.py`
- Router schema import surface：`flood_system/schemas/`
- 运行时存储：`flood_system/repository.py`
- 运行时表结构：`flood_system/storage/schema.py`
- 现有审批、通知、审计闭环：`flood_system/v2/`
- V2 平台审计边界：`flood_system/v2/platform_audit.py`
- 新增 AgentTwin 聚合能力：`flood_system/v3/`
- 正式前端入口：`frontend/src/App.tsx`
- V3 API 门面：`frontend/src/api/agentTwinApi.ts`
- 数据维护模型工厂：`frontend/src/features/dataManagement/dataModels.ts`
- AgentTwin 派生状态：`frontend/src/state/agentTwinSelectors.ts`
- AgentTwin 前端编排：`frontend/src/hooks/useAgentTwinConsole.ts`
- 数字孪生主屏：`frontend/src/components/DigitalTwinImpactScreen.tsx`
- Cesium 画布：`frontend/src/components/DigitalTwinCesiumCanvas.tsx`
- 三维校准公共模块：`frontend/src/lib/cityengineCalibration.ts`
- 演示主库重建：`scripts/rebuild_demo_db.py`
- 演示主库检查：`scripts/inspect_demo_db.py`
- 一键演示启动：`scripts/start-demo.ps1`

`3D_visual` 不再作为长期并行产品，而是作为 Cesium 场景、模型资源和空间配置的迁移来源。

## 3. 文档清单

1. [`01_PRD_产品需求说明.md`](./01_PRD_产品需求说明.md)
2. [`02_业务流程设计.md`](./02_业务流程设计.md)
3. [`03_系统架构设计.md`](./03_系统架构设计.md)
4. [`04_数据模型设计.md`](./04_数据模型设计.md)
5. [`05_知识图谱与本体设计.md`](./05_知识图谱与本体设计.md)
6. [`06_智能体设计.md`](./06_智能体设计.md)
7. [`07_Prompt与工具调用设计.md`](./07_Prompt与工具调用设计.md)
8. [`08_API接口设计_V3.md`](./08_API接口设计_V3.md)
9. [`09_前端UIUX原型说明.md`](./09_前端UIUX原型说明.md)
10. [`10_测试与评测方案.md`](./10_测试与评测方案.md)
11. [`11_前端实施拆解.md`](./11_前端实施拆解.md)
12. [`12_开发任务清单.md`](./12_开发任务清单.md)
13. [`13_按模块实施顺序.md`](./13_按模块实施顺序.md)
14. [`14_数据库问题清单与修复建议.md`](./14_数据库问题清单与修复建议.md)
15. [`15_演示主库重建方案.md`](./15_演示主库重建方案.md)
16. [`16_甲方演示脚本.md`](./16_甲方演示脚本.md)
17. [`17_可交付重构说明.md`](./17_可交付重构说明.md)

## 4. 生产级 Demo 验收口径

- 可演示：甲方进入首页后能直观看到城市数字孪生态势、重点对象、智能体会商、待审批动作和分众预警闭环。
- 可操作：主屏内可以完成对象选择、智能体追问、proposal 生成、人工审批和 warning 生成的核心链路。
- 可联调：前端优先消费 `/v3/*` 聚合接口，保留少量 `/v2/*` 审批桥接接口。
- 可降级：Cesium、LLM、SSE 或部分数据失败时，前端不白屏，后端返回结构化降级结果。
- 可追溯：proposal、warning、审计记录和执行留痕以 `proposal_id` 形成闭环。
- 可扩展：后续可以继续接入更多城市对象、更多数据源、真实通知网关和更严格的权限体系。

## 5. 前端展示口径

前端仍然要重点服务甲方展示效果，首页优先级最高：

- 第一眼必须像“数字孪生指挥台”，而不是普通后台。
- 中央必须是空间主画布，当前实现已接入真实 Cesium 画布。
- 三维画布必须复用 `3D_visual` 的模型校准口径，当前已抽出 `cityengineCalibration.ts` 承接 GLB 源坐标解析、归一化矩阵和模型焦点计算。
- 三维展示层需要服务“看懂态势”，当前已增加风险热区、扩散圈、发光联动路径、proposal / warning 状态标识和 `Command flythrough` 指挥巡航。
- 左侧用于态势解释，右侧用于 action / approval / warning 闭环。
- 智能体对话不是普通聊天，而是“会商、解释、追问、生成 proposal”的控制入口。
- `/agents` 展示多智能体会商差异、证据对照、supervisor 编排和治理边界。
- `/reliability` 展示审计、状态、故障恢复、闭环复盘，而不是只做健康监控。

## 6. 当前演示运行方式

推荐使用一键脚本启动：

```powershell
.\scripts\start-demo.ps1
```

该脚本会完成：

- 重建 `data/flood_warning_system_demo.db`
- 运行 `inspect_demo_db.py` 验证闭环数据
- 使用 `FLOOD_DB_PATH` 指向演示库启动后端
- 启动 `frontend` 开发服务
- 打开首页进入数字孪生智能体主屏

如需保留现有演示库，可执行：

```powershell
.\scripts\start-demo.ps1 -SkipRebuild
```

## 7. 推荐阅读顺序

- 产品和演示范围：`01 + 02 + 09`
- 后端和接口实现：`03 + 04 + 06 + 07 + 08`
- 前端重做和展示：`09 + 11 + 13`
- 联调和验收：`08 + 10 + 12 + 14 + 15 + 16`

## 8. 版本说明

- 当前实现目标：`V3 / AgentTwin production-grade demo`
- 当前实现策略：保留 `V2` 审批、通知、审计闭环，新增 `V3` 聚合读模型和智能体会商能力。
- 当前前端策略：`frontend` 为唯一正式展示入口，`3D_visual` 仅作为迁移来源。
