# 文档索引

本目录只维护当前系统事实、现行升级合同和可复现验收证据。已完成的阶段性规划、重复导出件和过时接口说明不再保留在当前树中；需要追溯时使用 Git 历史。

## 现行合同

1. [`洪水预警响应系统_渐进式迭代开发与升级设计.md`](../洪水预警响应系统_渐进式迭代开发与升级设计.md)：当前迭代与门禁合同。
2. [`面向区县防办的洪水预警响应系统_升级设计说明V3.md`](../面向区县防办的洪水预警响应系统_升级设计说明V3.md)：产品定位和 V3 业务边界。
3. [`PRODUCT.md`](../PRODUCT.md)：面向区县防办的产品原则。

## 当前实现与验收

- [`releases/v0.2.0.md`](./releases/v0.2.0.md)：本次全仓清理、结构重构、性能和安全更新报告。
- [`progressive_upgrade/`](./progressive_upgrade/)：资产处置、合同、数据字典、运维、安全、评测和阶段验收。
- [`V3_upgrade_acceptance_matrix.md`](./V3_upgrade_acceptance_matrix.md)：V3 功能证据与外部阻塞边界。
- [`openapi.json`](./openapi.json)：由运行中 FastAPI 应用导出的接口快照。
- [`../infra/postgis/README.md`](../infra/postgis/README.md)：PostGIS 影子迁移和对账说明。

## 兼容演示子系统

[`agent_twin_upgrade/`](./agent_twin_upgrade/) 只保留仍用于演示或真实数据接入的材料：甲方演示脚本、真实数据需求和字段字典。旧 AgentTwin 入口是兼容展示层，不是响应域正式状态源。

## 维护规则

- 文档描述必须与当前代码、OpenAPI 或可复现报告一致。
- 生成型逐样本 JSON、DOCX 导出件、构建产物和临时研究笔记不进入版本库。
- `/response/*` 是正式响应状态唯一写入口；兼容 `/platform/*`、`/agent-twin/*` 不得绕过状态机。
- 本地模拟、自动测试和空白人工评测包不得表述为生产验收或专家结论。
