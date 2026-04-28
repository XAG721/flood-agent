# 文档索引

## 1. 目录说明

当前仓库的文档按“已实现基线”“AgentTwin/V3 生产级 demo”和“演示运维说明”三层组织：

- [`docs/v2_baseline/`](./v2_baseline/)
  当前代码已经实现并仍然有效的 `V2` 基线说明文档
- [`docs/agent_twin_upgrade/`](./agent_twin_upgrade/)
  面向 `AgentTwin / V3` 升级需求、生产级 demo 测试品和甲方演示交付的设计文档包

## 2. 推荐阅读顺序

### 2.1 了解当前系统

1. [`docs/v2_baseline/洪水预警系统_前后端与总体功能总结.md`](./v2_baseline/洪水预警系统_前后端与总体功能总结.md)
2. [`docs/v2_baseline/系统整体框架简要说明.md`](./v2_baseline/系统整体框架简要说明.md)
3. [`docs/v2_baseline/洪水预警系统_API接口说明.md`](./v2_baseline/洪水预警系统_API接口说明.md)

### 2.2 了解升级目标

1. [`AgentTwin-Flood-Requirements.md`](../AgentTwin-Flood-Requirements.md)
2. [`docs/agent_twin_upgrade/README.md`](./agent_twin_upgrade/README.md)
3. [`docs/agent_twin_upgrade/01_PRD_产品需求说明.md`](./agent_twin_upgrade/01_PRD_产品需求说明.md)
4. [`docs/agent_twin_upgrade/03_系统架构设计.md`](./agent_twin_upgrade/03_系统架构设计.md)
5. [`docs/agent_twin_upgrade/08_API接口设计_V3.md`](./agent_twin_upgrade/08_API接口设计_V3.md)

### 2.3 运行和演示当前生产级 demo

1. [`README.md`](../README.md)
2. [`docs/agent_twin_upgrade/14_数据库问题清单与修复建议.md`](./agent_twin_upgrade/14_数据库问题清单与修复建议.md)
3. [`docs/agent_twin_upgrade/15_演示主库重建方案.md`](./agent_twin_upgrade/15_演示主库重建方案.md)
4. [`docs/agent_twin_upgrade/16_甲方演示脚本.md`](./agent_twin_upgrade/16_甲方演示脚本.md)
5. [`docs/agent_twin_upgrade/17_可交付重构说明.md`](./agent_twin_upgrade/17_可交付重构说明.md)

## 3. 维护约定

- `v2_baseline` 只记录当前代码事实
- `agent_twin_upgrade` 记录升级方案、当前实现说明、演示主库治理和甲方演示脚本
- 废弃文档不再保留在仓库中，避免和当前设计混淆
