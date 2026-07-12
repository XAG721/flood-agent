# 原系统不可变资产盘点与处置矩阵

盘点基线：Git `2db016a`（`main`，升级前最后一次已提交版本）。升级开发期间不把工作树中的新实现反写为“原系统已有能力”。

| asset_id | layer | code_location | current_capability | state_write | dependencies | reproducibility | test_status | disposition | target_module | rollback |
|---|---|---|---|---|---|---|---|---|---|---|
| LEGACY-FE-01 | 页面 | `frontend/src/components/DigitalTwinCesiumCanvas.tsx` | Cesium 三维场景、对象标注和相机交互 | 否 | Cesium 静态资源、3D 模型 | 部分可复现 | 前端回归 | REUSE/REFACTOR | 新版预警、对象和任务数据访问层 | 关闭响应页开关，保留旧展示入口 |
| LEGACY-FE-02 | 页面 | `frontend/src/features/copilot/` | 智能问答与 AgentTwin 演示 | 间接 | V2 API、模型网关 | 部分可复现 | 前端回归 | WRAP/RETIRE | 证据工作台、结构化草案 | 只读保留，不允许写正式响应状态 |
| LEGACY-BE-01 | 后端 | `flood_system/v2/`、`flood_system/v3/` | 事件展示、智能体编排、方案雏形 | 是，写旧 V2/V3 表 | SQLite、模型网关 | 可复现 | Python 回归 | WRAP/RETIRE | `response_workflow` Core API | 旧入口只读；恢复旧展示但禁止绕过新状态机 |
| LEGACY-RAG-01 | 算法 | `flood_system/rag.py`、`rag_runtime.py` | BM25、Dense、Hybrid 等基础检索 | 否 | 本地文档、可选神经模型 | 可复现 | RAG 专项测试 | REUSE/WRAP | RetrievalStrategy 与 FRC-RAG | `BASELINE_ONLY` |
| LEGACY-DATA-01 | 数据 | `data/flood_warning_system_v2.db` | V2/V3 演示数据和页面状态 | 是 | SQLite | 可复现 | 存储回归 | WRAP/RETIRE | `response_*` 新业务表 | 旧表只读，新数据不双写 |
| LEGACY-OBJ-01 | 数据 | `v2_entity_profiles`、`bootstrap_data` | 重点对象登记台账与部分坐标 | 是 | JSON/SQLite | 可复现 | 候选筛查测试 | WRAP | 风险对象版本与 CandidateRun | 缺字段时降级为区域关联并显示限制 |
| LEGACY-AUTH-01 | 安全 | `v2/security.py` | 旧岗位字符串权限 | 是 | HTTP 头 | 可复现 | 单元测试 | REBUILD | 签名身份断言、AAL2、辖区和职责分离 | 开发身份代理；生产无密钥时失败关闭 |
| LEGACY-AUDIT-01 | 审计 | V2 审计表 | 操作记录但非响应域哈希链 | 是 | SQLite | 可复现 | 部分测试 | REBUILD | 追加时间线与哈希链 | 只读保留旧审计 |
| LEGACY-DEPLOY-01 | 部署 | `scripts/start-demo.ps1` | 本地演示启动 | 否 | Windows、Python、Node | 部分可复现 | 人工 | REUSE/REFACTOR | Docker Compose + 本地脚本 | 本地脚本继续可用 |

## 关键盘点结论

1. 原页面没有资格作为正式业务状态源；正式响应状态只允许由 `/response` Core API 写入。
2. 旧 V2/V3 数据表在迁移期间保留，不执行新旧双向双写。
3. Cesium、基础 RAG、模型网关和已有演示页面可复用，但必须通过新版合同访问数据。
4. 审批、下发、状态迁移、审计、幂等和 Outbox 属于重建范围。
5. 真实 IdP、生产 TLS/KMS、真实预警多边形和异地主机灾备仍是外部环境验收项，不得用本地测试冒充完成。
