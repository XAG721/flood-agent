# AgentTwin Flood 升级文档包

## 1. 文档定位

本目录用于承接 [`AgentTwin-Flood-Requirements.md`](../../AgentTwin-Flood-Requirements.md) 中提出的最新升级需求，补齐当前仓库里尚未形成体系的 `V3 / AgentTwin` 设计文档。

现有仓库中的以下文档仍然有效，但主要描述的是当前已实现的 `V2` 基线：

- [`docs/v2_baseline/洪水预警系统_API接口说明.md`](../v2_baseline/洪水预警系统_API接口说明.md)
- [`docs/v2_baseline/系统整体框架简要说明.md`](../v2_baseline/系统整体框架简要说明.md)
- [`docs/v2_baseline/洪水预警系统_前后端与总体功能总结.md`](../v2_baseline/洪水预警系统_前后端与总体功能总结.md)
- [`docs/v2_baseline/洪水预警系统_主动预警主动请示需求文档.md`](../v2_baseline/洪水预警系统_主动预警主动请示需求文档.md)

本目录的文档不替代上述 `V2` 文档，而是作为“升级版设计包”使用，供后续开发、联调、答辩和论文材料整理。

## 2. 与当前代码的对应关系

当前代码事实基线如下：

- 后端主入口仍是 `flood_system/api.py`，正式接口前缀为 `/v2/*`
- 运行时存储仍由 `flood_system/repository.py` 中的 `v2_*` 表承担
- 核心业务模型定义在 `flood_system/v2/models.py`
- 多智能体链路已存在于 `flood_system/v2/multi_agent.py`
- Prompt 基线位于 `flood_system/v2/prompt_profiles.json`
- 权限基线位于 `flood_system/v2/security.py`
- 前端控制台入口位于 `frontend/src/App.tsx`
- 前端状态编排入口位于 `frontend/src/hooks/useV2OperatorConsole.ts`
- 全局请示弹窗位于 `frontend/src/components/GlobalRegionalProposalDialog.tsx`

因此，本目录中的设计文档采用“`V2` 基线 + `V3` 升级目标”的写法，避免脱离现有仓库实际情况。

## 3. 本次补充的文档

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

## 4. 建议使用方式

- 立项或答辩时，优先阅读 `01 + 03 + 09`
- 后端开发时，优先阅读 `03 + 04 + 06 + 07 + 08`
- 前端开发时，优先阅读 `01 + 02 + 09 + 08`
- 测试与验收时，优先阅读 `01 + 08 + 10`

## 5. 版本说明

- 当前代码实现状态：`V2`
- 本目录设计目标状态：`V3 / AgentTwin Upgrade`
- 迁移策略：兼容保留 `/v2/*` 主链路，逐步新增 `/api/v3/*` 能力，不一次性推翻现有实现
