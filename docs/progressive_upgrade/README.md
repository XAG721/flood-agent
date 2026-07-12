# 渐进式升级交付索引

本目录以仓库根目录《洪水预警响应系统_渐进式迭代开发与升级设计.md》为唯一升级合同，记录从旧 AgentTwin 原型到洪水预警响应任务闭环系统的可验证迁移证据。

## 文档入口

- `asset_inventory.md`：原系统资产、真实能力、依赖、写状态风险和 REUSE/WRAP/REFACTOR/REBUILD/RETIRE 结论。
- `implementation_acceptance_matrix.md`：阶段 0—8、最终审查清单和 Go/No-Go 门禁状态。
- `operations_and_rollback.md`：启动、健康检查、功能开关、备份、回滚和人工接管。
- `evaluation_and_gate_report.md`：真实运行的受控模拟指标、Gate 判定和外部 No-Go 边界。
- `../../infra/postgis/README.md`：SQLite 到 PostGIS 单向影子迁移、空间投影、隔离和对账演练。
- `../../output/performance/controlled_performance_report.md`：受控关键 API P50/P95、错误率和冻结预算报告。
- `dependency_security_audit.md`：Python、主前端和 Cesium 依赖漏洞清零及持续 CI 门禁。
- `api_compatibility_mapping.md`：设计概念、新 Core API 路径、旧接口边界和单写约束。

## 当前受控场景

- 区域：单一区县模拟区域；
- 预警：已有暴雨预警的模拟发布、更新与回放；
- 对象：下穿通道及可复用的重点对象登记台账；
- 下发：仅模拟 Outbox，不连接真实跨部门端点；
- 结论边界：系统辅助生成、人工审批、确定性执行、全过程可追溯，不进行洪水预测，也不替代指挥决策。

## 统一入口

- 新版核心业务 API：`/response/*`；兼容版本前缀：`/api/v1/*`。
- 旧 AgentTwin 展示入口：`/agent-twin/*`（内部兼容 `/v3/*`）。
- 旧平台入口：`/platform/*`（内部兼容 `/v2/*`）。
- 运维探针：`/health`、`/ready`、`/metrics`。
