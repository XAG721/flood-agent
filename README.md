# 基于多源时空语义关联与 FRC-RAG 的洪水预警响应系统

[![CI](https://github.com/XAG721/flood-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/XAG721/flood-agent/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.3.0-2563eb)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-149eca?logo=react&logoColor=white)

面向区（县）防汛抗旱指挥部办公室，将专业部门发布的预警与风险对象、时空位置、责任台账、预案条款和执行反馈进行关联，形成“对象可确认、任务有依据、过程可审批、结果可核验”的洪水预警响应闭环。

系统研究重点包括两部分：

- **多源时空语义关联**：把预警时空范围、对象坐标与属性、台账有效期、文档版本和业务语义统一到对象级响应事件中；
- **FRC-RAG**：以任务字段和功能角色覆盖为约束组织证据集合，显式处理证据冲突、缺失、失效与跨文档关系，为结构化任务草案提供可定位、可比较、可冻结的依据。

> [!IMPORTANT]
> 当前 `v0.3.0` 已完成单一区域、暴雨预警类型下的**受控模拟闭环**，不进行洪水预测，也不替代防汛指挥决策。FRC-RAG 的工程链路与公开数据实验可复现，但尚未证明稳定优于最强公平基线，Gate 2 保持 `NO-GO / SHADOW`；真实生产部署仍需权威数据、机构 UAT、安全与基础设施验收。

## 研究问题与系统目标

传统预警系统往往停留在信息展示或统一通知层面，难以回答四个直接影响执行的问题：

1. 预警具体影响哪些对象，候选结果为什么被选中？
2. 针对每个对象应由谁、在何时、执行什么动作？
3. 任务依据来自哪个版本、哪一条款，是否存在冲突或缺失？
4. 下发后是否接收、执行、反馈并经过独立核验？

本系统以响应事件为主线，将上述问题组织为确定性业务闭环：

```text
专业预警
  -> 响应事件与不可变预警快照
  -> 多源时空语义关联与候选对象筛查
  -> 人工确认并冻结对象清单
  -> FRC-RAG 九字段证据包
  -> 结构化任务草案与规则校验
  -> 授权人员审批
  -> 模拟下发、接收、执行与反馈
  -> 独立核验、事件关闭与复盘
```

AI 与检索模块只能生成候选对象、证据和草案，不能审批、下发或直接改变正式业务状态。智能服务不可用时，人工证据组装与确定性工作流仍可继续运行。

## 核心技术

### 1. 多源时空语义关联

系统关联预警、风险对象主数据、地理位置、责任关系、预案文档和事件反馈，并保留每一次关联所使用的数据版本。

- 预警保存区域、时间、等级、版本和可选 EPSG:4326 影响范围；
- 风险对象支持 API、UTF-8 CSV、受限 XLSX、JSON 与 Point GeoJSON 导入；
- 候选筛查综合空间、时效、属性、语义和数据质量五类特征，并保存逐候选解释快照；
- 缺少坐标的对象不会冒充空间命中，系统明确记录排除原因或降级为区域台账关联；
- 主数据发生实质变化后，相关候选运行与事件对象进入 `STALE`，重新筛查和人工核验前禁止成案；
- 已确认对象冻结为不可变 `CandidateObjectListVersion`，任务必须绑定清单 ID、版本与内容哈希。

这一机制把“模型给出的风险分”转化为可复核的数据关联过程，候选对象始终需要业务人员确认后才能成为正式处置对象。

### 2. FRC-RAG 证据组织

FRC-RAG（Functional Role Coverage-guided Retrieval-Augmented Generation，功能角色覆盖约束的检索增强生成）不是单一重排器，而是一条从查询分解到证据冻结的完整链路：

```text
任务 Schema
  -> 九字段查询分解
  -> BM25 / Dense / Hybrid 候选召回与重排
  -> 字段支持、功能角色、可信度、适用性与新颖性评分
  -> 冲突惩罚与证据预算约束
  -> SUPPORTED / CONFLICTED / MISSING 三态证据矩阵
  -> 人工裁决、版本比较与证据包冻结
```

证据包覆盖九个任务字段：

| 字段 | 业务含义 |
|---|---|
| `trigger_condition` | 触发条件、预警等级、阈值和适用范围 |
| `risk_object` | 风险对象、位置、脆弱性和影响范围 |
| `responsible_party` | 责任单位、责任岗位、权限和协同主体 |
| `action` | 处置动作、执行步骤、先后顺序和操作要求 |
| `deadline` | 接收、开始、完成、核验时限和数值阈值 |
| `resource_dependency` | 人员、车辆、设备、物资和资源依赖 |
| `feedback_requirement` | 反馈内容、附件、位置、时间和证据要求 |
| `escalation_condition` | 催办、升级、改派和人工接管条件 |
| `exception_condition` | 例外、受阻、终止、替代和特殊情况 |

每条证据保存来源、文档版本、条款、页码、章节路径、表格行和原文定位。核心字段缺失或存在未裁决冲突时阻断审批；资源依赖和例外条件可保持 `MISSING`，但必须给出明确原因，系统不会自动补造条款。

冲突由确定性规则、显式来源关系和可选 NLI 适配器产生候选，最终采用哪个来源必须由人工裁决。默认 NLI 状态为 `nli-unavailable`，不会把未部署的模型伪装成可用能力。

### 3. 确定性响应工作流

- 预警发布、更新、撤销与对象重新核验；
- 任务创建、提交、三档规则校验、职责分离审批与载荷哈希绑定；
- 接收、开始、完成、核验四类时限及催办、升级、改派和人工接管；
- 受阻、资源不足、部分完成、延期、撤回和核验退回整改分支；
- 正常、超时、拒收、部分成功、重复、乱序六类模拟下发场景；
- 回调只追加证据与审计记录，不能直接修改正式任务状态；
- 事件关闭、时间线完整性校验、审计归档和复盘草稿。

### 4. 安全、审计与恢复

- `/response/*` 接受可信身份网关签名的短时身份断言，高风险操作强制 AAL2；
- RBAC 与职责分离阻止越权操作和高风险任务自审；
- 响应域敏感载荷使用 Fernet 认证加密，密钥支持受控轮换；
- 事件台账形成 SHA-256 哈希链，关键事实由数据库触发器禁止更新或删除；
- Core API 写请求由身份绑定的通用幂等账本保护，处理并发、重放和未知写入结果；
- 备份清单使用独立 HMAC 签名，恢复前验证摘要、完整性、密钥标识和实际解密能力；
- 精确位置、联系方式等信息按岗位脱敏，外部模型调用经过个人信息出站过滤；
- 模拟网关只接受 `simulated://` 目标，拒绝连接真实外部下发端点。

## 系统架构

```text
┌──────────────── React 响应工作台 / Cesium 可视化 ────────────────┐
│  事件总览 · 对象核验 · 证据工作台 · 审批 · 执行 · 复盘          │
└──────────────────────────┬───────────────────────────────────────┘
                           │ /response/*  (/api/v1/* 兼容前缀)
┌──────────────────────────▼───────────────────────────────────────┐
│                         FastAPI Core API                         │
│ 身份与幂等边界 · 状态机 · 对象关联 · 文档治理 · FRC-RAG · 审计 │
└───────────────┬──────────────────┬──────────────────┬────────────┘
                │                  │                  │
      ┌─────────▼────────┐ ┌───────▼────────┐ ┌──────▼───────────┐
      │ 加密 SQLite 权威源 │ │ Worker / Outbox │ │ PostGIS 影子投影 │
      │ 版本、任务、证据   │ │ 模拟下发与巡检   │ │ 仅迁移对账，不切流 │
      └──────────────────┘ └────────────────┘ └──────────────────┘
```

接口边界：

- `/response/*`：新版响应域 Core API，也是正式响应状态的唯一写入口；
- `/api/v1/*`：Core API 的兼容版本前缀；
- `/platform/*`、`/agent-twin/*`：旧平台与数字孪生兼容展示层，不得绕过新版状态机写入响应域；
- `/health`、`/ready`、`/metrics`：存活、就绪与运行指标；
- SQLite 当前仍是权威状态源，PostGIS 只用于单向影子迁移和对账。

## 快速开始

### 方式一：Docker Compose 受控模拟环境

需要 Docker Desktop 或兼容的 Docker Compose。仓库根目录执行：

```powershell
docker compose up --build
```

启动后访问：

- 响应工作台：<http://127.0.0.1:8080/response>
- 后端 OpenAPI：<http://127.0.0.1:8000/docs>
- 就绪检查：<http://127.0.0.1:8000/ready>

Compose 会启动 `frontend`、`backend`、`worker` 和模拟专用 `postgis`。其中的身份、数据库和下发配置仅用于本地受控模拟，不能直接用于生产。

### 方式二：Windows 一键演示

需要 Python 3.12、Node.js 22，并先安装依赖：

```powershell
python -m pip install -e ".[test,postgres]"
npm.cmd ci --prefix frontend
```

然后运行：

```powershell
.\scripts\start-demo.ps1
```

脚本会重建并检查演示数据库，启动后端 `http://127.0.0.1:8000` 与前端 `http://127.0.0.1:5173`，并默认启用固定演示快照。常用参数：

```powershell
.\scripts\start-demo.ps1 -SkipRebuild   # 保留现有演示库
.\scripts\start-demo.ps1 -LiveData      # 关闭前端固定演示态
.\scripts\start-demo.ps1 -NoBrowser     # 不自动打开浏览器
```

### 本地开发

```powershell
python -m pip install -e ".[test,postgres]"
python scripts/rebuild_demo_db.py --force
$env:FLOOD_DB_PATH = "$PWD\data\flood_warning_system_demo.db"
python -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000
```

在另一个终端启动前端：

```powershell
npm.cmd ci --prefix frontend
$env:VITE_DEMO_MODE = "true"
npm.cmd run dev --prefix frontend
```

生产模式缺少数据加密密钥、身份断言密钥或备份清单密钥时会拒绝启动。完整的身份、密钥轮换、备份恢复和回滚要求见[运维与回滚手册](docs/progressive_upgrade/operations_and_rollback.md)及[安全与事件处置手册](docs/progressive_upgrade/security_and_incident_manual.md)。

## 复现实验与验证

日常代码验证：

```powershell
python -m pytest -q --basetemp .pytest-tmp/readme
python -m ruff check flood_system scripts tests
npm.cmd test --prefix frontend -- --run
npm.cmd run build --prefix frontend
npm.cmd run build --prefix 3D_visual
```

重建受控验收证据：

```powershell
python scripts/generate_floodagent_bench.py
python scripts/run_candidate_evaluation.py
python scripts/run_rag_evaluation.py
python scripts/run_controlled_performance.py --db tmp/performance.db --output-dir output/performance
python scripts/run_design_contract_audit.py
python scripts/run_progressive_completion_audit.py
```

公开数据、真实神经评分器、消融、敏感性分析和人工标注流程见 [FRC-RAG 公开评测协议](docs/progressive_upgrade/frc_public_evaluation_protocol.md)。仓库不会把小型工程集、空白标注包或跨领域实验表述为真实防汛领域专家结论。

## 当前验证状态

以下为 `2026-07-15` 冻结的仓库验证基线：

| 验证项 | 结果 |
|---|---|
| Python 全量测试 | 248 / 248 通过；Ruff 0 问题；编译通过 |
| 前端单元测试 | 19 / 19 通过 |
| Chromium E2E | 2 / 2 通过 |
| 生产构建 | 主前端与 Cesium 独立构建均通过 |
| 依赖安全审计 | Python 隔离依赖、两套 Node 工程均无已知漏洞 |
| 受控性能预算 | 4 / 4 端点通过，160 次请求错误率为 0 |
| 正常与故障场景 | 24 / 24，`PASS` |
| 设计合同审计 | 89 / 89 已归属：87 条本地/受控证据，2 条外部 No-Go |
| 完成性审计 | 本地要求 12 / 12，`CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO` |
| GitHub Actions | 后端、前端、安全、Compose/Chromium/PostGIS 4 项门禁通过 |

CI 会重新生成关键评测、OpenAPI 和完成性审计并逐字节比较，防止文档结论与代码行为漂移。

## Gate 与研究结论边界

| 门禁 | 当前状态 | 结论 |
|---|---|---|
| Gate 0：模拟数据治理 | `GO`（受控模拟） | 数据来源、版本、种子和清单可复现 |
| Gate 1：候选对象质量 | `GO`（受控模拟） | 达到冻结阈值，仍强制人工确认 |
| Gate 2：FRC-RAG | `NO-GO / SHADOW` | 工程流水线可行，但相对最强公平基线的稳定优势未获证明 |
| Gate 3：安全不变量 | `GO`（自动化） | 越权、未审批、重复写、冲突和非法状态迁移失败关闭 |
| Gate 4：生产验收 | `CONDITIONAL NO-GO` | 缺真实 UAT、生产安全、容量和异地恢复证据 |
| Migration M7：旧链路退役 | `NO-GO` | 需真实旧流量归零、在途事件清空、归档恢复和账号撤权 |

FRC-RAG 当前可证明的是：多字段、功能角色、适用性和冲突约束能够形成可运行、可审计、可复现的证据组织流程。当前实验不能证明 FRC-RAG 已稳定优于最强公平基线，因此正式任务仍保留 Baseline 降级策略，只有重新通过 Gate 2 后才考虑 `CANARY` 或 `DEFAULT`。

## 目录结构

```text
flood_system/                         FastAPI 后端与平台兼容能力
  response_workflow/                 响应域模型、状态机、服务与持久化
  http/response_router.py            Core API 路由
  response_workflow/evidence_governance.py
                                      九字段证据与冲突治理
frontend/                             React 响应工作台
3D_visual/                            Cesium/CityEngine 独立可视化工程
scripts/                              迁移、演示、评测、审计与 Worker 脚本
infra/                                PostGIS 影子迁移与基础设施合同
benchmarks/                           冻结预算与受控基准配置
output/                               可复现评测与验收摘要
docs/                                 当前设计、运维、安全和验收文档
```

## 文档导航

- [产品定义与设计原则](PRODUCT.md)
- [渐进式迭代开发与升级设计](洪水预警响应系统_渐进式迭代开发与升级设计.md)
- [V3 产品与业务设计说明](面向区县防办的洪水预警响应系统_升级设计说明V3.md)
- [当前文档索引](docs/README.md)
- [渐进式升级交付索引](docs/progressive_upgrade/README.md)
- [接口、状态与数据合同](docs/progressive_upgrade/contracts_and_data_dictionary.md)
- [用户手册](docs/progressive_upgrade/user_manual.md)
- [评测与 Gate 报告](docs/progressive_upgrade/evaluation_and_gate_report.md)
- [全目标完成度追溯审计](docs/progressive_upgrade/completion_traceability_audit.md)
- [v0.3.0 版本更新报告](docs/releases/v0.3.0.md)
- [OpenAPI 快照](docs/openapi.json)

## 项目定位

本项目的研究价值不在于让大模型自主发出防汛指令，而在于探索如何把多源、异构、带版本和有效期的业务信息，转换为对象级、证据约束、人工可控的响应任务。系统坚持“事实与建议分层、关键动作人工审批、证据和责任链全程可追溯”的设计原则。
