# 渐进式升级交付索引

本目录以仓库根目录《洪水预警响应系统_渐进式迭代开发与升级设计.md》为唯一升级合同，记录当前洪水预警响应任务闭环系统的实现、迁移和验收证据。

## 文档入口

- `../releases/v0.3.0.md`：当前外部接口仿真、场景目录、浏览器验收、性能、安全和兼容性更新报告。
- `asset_inventory.md`：原系统资产、真实能力、依赖、写状态风险和 REUSE/WRAP/REFACTOR/REBUILD/RETIRE 结论。
- `implementation_acceptance_matrix.md`：阶段 0—8、最终审查清单和 Go/No-Go 门禁状态。
- `completion_traceability_audit.md`：按最新设计合同汇总阶段、24 场景、第 18 节切换清单、Gate、交付物和外部 No-Go 的全目标追溯审计。
- `design_contract_evidence_policy.json`：把最新设计中的 89 条显式条件一一绑定到本地、受控、门禁或外部证据组的失败关闭策略。
- `../../output/acceptance/design_contract_audit.md`：从设计原文确定性提取并逐条核验的 89 条合同账本。
- `../../output/acceptance/legacy_baseline_manifest.md`：升级前代码、数据库重建输入、SQLite 与 RAG 索引的离线可复现冻结清单。
- `../../output/acceptance/progressive_completion_audit.md`：CI 可确定性重建的第 18—23 节机器可读审计，固定区分受控首期完成、Gate 2 No-Go 与生产外部 No-Go。
- `operations_and_rollback.md`：启动、健康检查、功能开关、备份、回滚和人工接管。
- `evaluation_and_gate_report.md`：真实运行的受控模拟指标、Gate 判定和外部 No-Go 边界。
- `frc_public_evaluation_protocol.md`：FRC-RAG 公开数据、真实模型、公平基线、配对统计和缺失/冲突/失效难例协议。
- `../../output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md`：HousingQA 冻结真实模型字段/角色权重扫描、参数辨识结果与 Gate 2 边界。
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
