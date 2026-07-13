# 面向区县防办的洪水预警响应系统

系统面向区（县）防汛抗旱指挥部办公室，在专业部门已经发布预警后，将风险对象筛查、任务草拟、人工审批、部门执行、反馈核实和事件复盘组织为确定、可追溯的业务闭环。`AgentTwin Flood` 继续作为数字孪生与智能辅助原型代号，但不再作为正式业务定位。

核心演示链路：

```text
专业预警 -> 响应事件 -> 对象核验 -> 任务草案 -> 人工审批 -> 部门执行 -> 反馈核实 -> 事件复盘
```

## 当前能力

- 一级入口 `/response` 提供区县防办响应闭环工作台，集中展示预警版本、待核验对象、结构化任务、人工审批、执行反馈、异常升级和统一事件台账。
- 后端 `/response/*` 提供不依赖大模型的确定性工作流：事件和预警快照、区域对象台账筛查、人工对象核验、任务分派/改派、任务全版本、任务状态机、审批隔离、证据校验、时限巡检、升级和事件关闭校验。
- 区域风险对象主数据支持 AAL2 API、UTF-8 CSV、受限 XLSX、JSON 和 Point GeoJSON 导入，逐对象哈希版本、异常行隔离、有效期/重复关系校验和 Cesium 地图联动；主数据实际变化会使相关候选运行与事件对象进入 `STALE`，重新筛查和人工核验前禁止成案。
- 已将证据集合检索接入对象级任务草案：预警快照、对象责任台账和预案文档共同覆盖 `condition`、`object`、`responsibility`、`procedure`、`exception`、`attribution` 六类角色；任一必要角色缺失时拒绝成案并写入阻断台账。
- 生成的任务草案保存检索来源、文档版本、条款号、角色覆盖和 grounding 摘要，前端可展开核对来源；智能辅助不可用或证据不足时仍可使用人工创建任务接口。
- 高风险任务必须由指挥审批员批准，草拟人员不得审批自己提交的高风险任务；AI 生成内容只能进入草稿状态并显示生成版本。
- 任务状态变化、审批、反馈和关闭操作均以追加时间线记录保存，任务完成后必须经过人工核实。
- 现场反馈会自动分类为完成、受阻、资源不足、协同请求或态势更新；相同任务的重复反馈按内容指纹合并，不重复计入过程指标。
- 事件关闭后可运行可重复的区县最小场景评测：保存人工流程参考基线与系统台账实测对照、16 项系统业务指标、V3 十项验收证据、失败案例和数据不足观察项；人工基线明确标记为可配置假设。
- 响应业务载荷使用 Fernet 认证加密后写入 SQLite，事件编号、状态和时间等索引字段保持可查询；密文被修改或密钥不匹配时拒绝读取。
- 事件台账形成逐条 SHA-256 哈希链，并记录操作者、岗位、终端及关键状态变更前后值；原始预警、审批、反馈证据和时间线同时由 SQLite 触发器禁止更新或删除。
- 管理员可创建带 SHA-256 清单和 SQLite 完整性检查的数据库备份，并在不覆盖运行库的隔离副本上执行恢复演练；审计员只能查看备份清单。
- 备份清单使用独立 HMAC 密钥签名并记录源实例；跨主机导入会验证签名、文件摘要、大小、SQLite 完整性、加密密钥标识和实际解密能力，再允许隔离恢复。
- 数据加密密钥支持在线重加密和受控激活：轮换前自动备份，生产模式写入 `pending_activation` 状态；部署未提升新密钥时重启会拒绝服务，防止回退旧钥或继续产生混合密文。
- `/response/*` 只接受可信身份网关签名的短时身份断言；签名绑定操作者、岗位、终端、AAL、时间、nonce、方法和路径，并通过一次性 nonce 防重放。高风险审批、任务豁免、备份和恢复强制 AAL2。
- `/response/*` 和 `/api/v1/*` 的全部写请求由通用幂等账本保护：作用域绑定身份、终端、方法和路径，同键同请求重放原 2xx 响应，同键异请求或并发占位返回 409；异常结果进入不可盲重试的 `INDETERMINATE`。记录加密落盘并随数据密钥轮换。
- 风险对象的联系方式、特殊人群说明和精确位置带数据分类；读取时按签名岗位生成脱敏视图。所有真实外部模型调用统一经过个人信息出站过滤，并只把脱敏计数和摘要哈希写入审计记录。
- 受阻、资源不足和协同请求会生成升级记录与备选处置动作；事件关闭时自动生成基于完整台账的复盘草稿。
- 首页 `/` 已重构为数字孪生智能体指挥主屏，包含左侧态势带、中央 Cesium 三维画布、右侧 proposal / warning 闭环指挥台和智能体对话抽屉。
- 三维画布已接入 `3D_visual` 的 CityEngine GLB 模型资源，并抽出 `frontend/src/lib/cityengineCalibration.ts` 复用源坐标归一化和模型校准逻辑。
- 三维展示层已支持风险热区、动态积水面、水位柱、发光联动路径、proposal / warning 状态标识，以及 6 段式 `Play command story` 指挥叙事镜头。
- 前端支持 `VITE_DEMO_MODE=true` 演示模式，可固定首页事件、对象、proposal、warning 和会商结果，降低现场数据波动对展示的影响。
- 后端提供统一的 AgentTwin 能力入口：`/agent-twin/*` 负责主屏聚合、对象聚焦、智能体会商、对话、proposal 生成、warning 生成和 SSE 实时事件。
- 后端提供统一的平台能力入口：`/platform/*` 负责审批、通知、执行日志、审计、数据维护和可靠性治理。
- 演示主库可通过脚本重建，固定支撑 `event_demo_beilin_primary` 主链路。

## 渐进式迭代 V1 入口

最新《洪水预警响应系统_渐进式迭代开发与升级设计》对应的实现和验收资料位于 [`docs/progressive_upgrade/`](docs/progressive_upgrade/README.md)，OpenAPI 快照位于 [`docs/openapi.json`](docs/openapi.json)。本轮新增：

- 不可变预警与风险对象版本；预警更新后对象进入 `STALE`，重新核验前不可成案；
- CandidateRun、原始评分、校准置信度、算法/特征/数据版本和缺失特征；
- 文档版本、条款索引、生效/替代关系和失效版本检索隔离；
- FRC-RAG 证据包字段三态、冲突原子组、人工裁决与冻结版本；
- 固定任务 JSON Schema、`PASS/SOFT_WARNING/HARD_BLOCK` 规则结果、乐观锁；
- 审批载荷/证据哈希、幂等 Outbox、接收/开始/完成/核验四类时限；
- 身份绑定的通用写请求幂等账本、原响应重放、并发预留和失败不确定态；
- 受阻、部分完成、延期、改派、撤回、人工接管和独立核验分支；
- 数据库结构校验和账本、Legacy Adapter 只读审计和无法映射旧数据隔离；
- 可重复 FloodAgent-Bench v2 生成器、24 项正式场景目录、候选关联评测、模拟端点网络隔离；
- 下发通道正常/超时/拒收/部分成功/重复/乱序故障矩阵，以及不改变正式状态的幂等异步回调；
- `/health`、`/ready`、`/metrics`、Docker Compose 和功能开关回滚。

常用命令：

```powershell
python scripts/migrate_response_schema.py --db data/flood_warning_system_v2.db
python scripts/run_legacy_migration_inventory.py --db data/flood_warning_system_v2.db
python scripts/generate_floodagent_bench.py
python scripts/run_candidate_evaluation.py
python scripts/run_rag_evaluation.py
python scripts/run_housing_weight_sensitivity.py
python scripts/run_response_worker.py --once
python scripts/reset_simulation_environment.py --confirm RESET-SIMULATION
python scripts/export_openapi.py --output docs/openapi.json
docker compose up --build
```

Compose 同时启动模拟专用 PostGIS。SQLite 响应域仍是权威源；执行和验证单向影子迁移的命令见 `infra/postgis/README.md`，在生产迁移 Gate 通过前不得切换权威存储。

运行受控 API 性能回归预算：

```powershell
python scripts/run_controlled_performance.py --db tmp/performance.db --output-dir output/performance
```

该命令只验证仓库内单进程回归预算，不能代替生产网络、并发容量或真实数据规模压测。

冻结并验证旧系统可复现基线（实际 SQLite/RAG 二进制只保存在忽略的 `.cache/`，Git 仅保存哈希、Schema、行数和 Git blob 身份）：

```powershell
python scripts/freeze_legacy_baseline.py
```

逐条重建第 18.3—18.6、22、23 节的 89 条显式设计合同，再重建汇总完成性审计：

```powershell
python scripts/run_design_contract_audit.py
python scripts/run_progressive_completion_audit.py
```

逐条报告写入 `output/acceptance/design_contract_audit.json` 和 `.md`，汇总报告写入 `output/acceptance/progressive_completion_audit.json` 和 `.md`。CI 会重建并逐字节比较；当前 87 条本地/受控条件有证据，`18.4-10` 和 `22.4-10` 的真实旧流量归零与旧链路退役仍为外部 No-Go。本地受控闭环通过不会改变 FRC-RAG Gate 2 或真实生产/UAT 的 No-Go 状态。

## 主要目录与结构边界

- `flood_system/api.py`：FastAPI 统一装配入口，并提供 `/agent-twin/*` 与 `/platform/*` 两类公开能力入口。
- `flood_system/config.py`：运行配置与 `FLOOD_DB_PATH` 解析。
- `flood_system/http/`：AgentTwin HTTP 路由层。
- `flood_system/response_workflow/`：区县防办确定性响应工作流模型与服务。
- `flood_system/rag_evaluation.py`：BM25、哈希向量 Dense、混合、MMR、Rerank、覆盖贪心代理和 FRC-Select 的可重复工程评测、w/o Role / w/o Field 消融与机器可读 Gate 2 判定；覆盖贪心代理不是 SetR 复现。
- `flood_system/frc_public_evidence.py`：导入真实 BGE/reranker 公共数据产物，重算逐样本配对置信区间，并执行缺失证据压力切片。
- `flood_system/frc_housing_weight_sensitivity.py`：复用 HousingQA 冻结真实模型分数，执行字段/角色权重单因素扫描并生成可失败关闭的逐例工件；该跨领域诊断不改变 Gate 2。
- `flood_system/design_contract_audit.py`：从最新设计原文提取 89 条显式合同，校验证据唯一归属、仓库内路径、内容标记和旧基线哈希，并保留外部 No-Go。
- `flood_system/http/response_router.py`：`/response/*` 业务闭环 API。
- `flood_system/http/idempotency.py`：Core API 写请求指纹、原子预留、响应重放与不确定结果失败关闭。
- `flood_system/infrastructure/sse.py`：SSE 编码与流式基础设施。
- `flood_system/schemas/`：HTTP router 使用的 schema import surface。
- `flood_system/storage/schema.py`：SQLite 运行时表结构与索引定义，避免 `repository.py` 继续承载建表大块文本。
- `flood_system/`：承载审批、通知、审计、执行、多智能体和 AgentTwin 聚合读模型等后端能力。
- `frontend/src/api/agentTwinApi.ts`：前端 AgentTwin 主链路 API 门面。
- `frontend/src/api/responseWorkflowApi.ts`：响应闭环 API 门面。
- `frontend/src/pages/ResponseWorkflowPage.tsx`：区县防办响应事件工作台。
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
- `docs/progressive_upgrade/`：当前架构、实施、运维、安全和验收资料。
- `docs/agent_twin_upgrade/`：仅保留 AgentTwin 兼容演示脚本和真实数据接入字典。

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
$env:FLOOD_ENVIRONMENT="production"
$env:FLOOD_DATA_ENCRYPTION_KEY="<Fernet key from your secret manager>"
$env:FLOOD_TRUSTED_IDENTITY_SECRET="<HMAC secret shared only with your identity gateway>"
$env:FLOOD_BACKUP_MANIFEST_SECRET="<independent HMAC secret for backup manifests>"
$env:FLOOD_INSTANCE_ID="beilin-primary-a"
$env:FLOOD_REQUIRE_HTTPS="1"
$env:FLOOD_TRUST_PROXY_HEADERS="1"
C:\Users\Administrator\anaconda3\python.exe -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000
```

生产模式缺少 `FLOOD_DATA_ENCRYPTION_KEY`、`FLOOD_TRUSTED_IDENTITY_SECRET` 或 `FLOOD_BACKUP_MANIFEST_SECRET` 时拒绝启动。未设置生产模式时，系统仅为本地开发生成数据库同目录的 `*.db.key`，Vite 开发代理使用明确标记的本地签名密钥；这些开发默认值不得用于部署。密钥文件和 `backups/` 已排除版本控制。备份恢复需要相同的加密密钥，清单会记录非敏感的密钥标识用于匹配校验。

生产身份网关必须在完成密码加 MFA、WebAuthn 或等价身份验证后签发 `X-Identity-*` 请求头。AAL2 断言经过 HMAC 校验、两分钟时效校验和 nonce 防重放后，才允许执行高风险审批。`FLOOD_TRUST_PROXY_HEADERS=1` 只能在后端仅接受可信反向代理流量时启用。

### 生产加密密钥轮换

1. 保持 `FLOOD_DATA_ENCRYPTION_KEY` 为当前密钥，将新 Fernet 密钥写入 `FLOOD_DATA_ENCRYPTION_KEY_NEXT`。
2. 使用管理员 AAL2 身份调用 `POST /response/security/keys/rotate`。系统先创建签名备份，再重加密响应域载荷并返回新旧密钥标识。
3. 返回 `pending_activation` 后，将新密钥提升为 `FLOOD_DATA_ENCRYPTION_KEY`，并把旧密钥加入逗号分隔的 `FLOOD_DATA_DECRYPTION_KEYS`。
4. 重启服务。系统验证待激活密钥标识后完成激活，并继续允许读取轮换前备份。
5. 备份保留期结束且确认不再需要旧密文后，才能从 `FLOOD_DATA_DECRYPTION_KEYS` 移除旧密钥。

若数据库已完成重加密但部署配置尚未提升新密钥，服务会以 `pending activation` 错误拒绝启动，不会自动回退。

### 跨主机灾备导入

将来源主机生成的 `.db` 和 `.manifest.json` 文件复制到目标实例的 `backups/incoming/`，保证目标实例配置相同的 `FLOOD_BACKUP_MANIFEST_SECRET`，并在密钥环中保留备份使用的加密密钥。随后由管理员 AAL2 调用 `POST /response/security/backups/import`，验证通过后再调用恢复演练接口。导入接口拒绝路径穿越、签名篡改、摘要不符、数据库损坏、未知密钥和解密失败。

### 备份保留与审计日志归档

管理员以 AAL2 身份调用 `POST /response/security/backups/retention`。建议先使用 `dry_run=true` 查看候选，再以 `dry_run=false` 执行。`keep_latest` 始终保留最新恢复点，`max_age_days` 控制超过保留期的旧备份；所有加密密钥轮换恢复点会被强制保护。清理只删除备份文件，签名备份元数据和不可变执行记录继续留在数据库中。

调用 `POST /response/events/{event_id}/audit-archives` 会在时间线哈希链验证通过后，生成包含事件看板、任务全版本和完整性报告的加密归档，同时生成独立 HMAC 签名清单。默认留存期为 2555 天。审计员可列出并验证归档，但只有管理员可以创建归档；归档元数据和保留策略执行记录均不可覆盖或删除。

### 预警空间范围关联

预警载荷可选传入 `affected_geometry`，格式为闭合的 EPSG:4326 外环坐标。对象画像同时具备经纬度时，候选筛查使用边界包含的点落多边形判断；缺少坐标的对象不会冒充精确匹配，并在结果中单独计数。预警未提供多边形时，系统明确降级为 `area_id` 区域台账关联。空间关系只决定待核验候选，不直接形成正式风险结论，也不替代水动力分析。

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
http://127.0.0.1:5173/response
```

## 关键接口边界

区县防办响应闭环：

- `POST /response/events`、`GET /response/events/{event_id}`
- `POST /response/events/{event_id}/alerts`
- `POST /response/events/{event_id}/risk-objects`
- `POST /response/events/{event_id}/risk-objects/discover`
- `POST /response/events/{event_id}/risk-objects/{object_id}/verify`
- `POST /response/events/{event_id}/tasks`
- `POST /response/events/{event_id}/risk-objects/{object_id}/task-draft`
- `POST /response/tasks/{task_id}/submit`
- `POST /response/tasks/{task_id}/decision`
- `POST /response/tasks/{task_id}/acknowledge`、`start`、`feedback`、`verify-completion`
- `POST /response/tasks/{task_id}/assign`、`GET /response/tasks/{task_id}/versions`
- `POST /response/events/{event_id}/deadline-sweep`
- `POST /response/events/{event_id}/close`
- `POST /response/events/{event_id}/review-draft`
- `POST /response/events/{event_id}/scenario-evaluation`
- `GET /response/events/{event_id}/scenario-reports`
- `GET /response/events/{event_id}/integrity`
- `POST /response/security/backups`、`GET /response/security/backups`
- `POST /response/security/backups/restore`
- `POST /response/security/backups/import`
- `POST /response/security/backups/retention`
- `POST /response/events/{event_id}/audit-archives`、`GET /response/events/{event_id}/audit-archives`
- `POST /response/security/audit-archives/{archive_id}/verify`
- `POST /response/security/keys/rotate`

运行 V3 本地 RAG 方法对比与 FRC-Select 消融：

```powershell
python scripts/run_rag_evaluation.py
```

结果写入 `output/rag_evaluation/rag_evaluation_report.json` 和 `rag_evaluation_report.md`。该工程验收明确使用小规模仓库内标注集；Dense 为无外部模型依赖的哈希 n-gram 向量基线，不能替代真实神经向量模型或公开 benchmark 复现。

真实神经向量评测应在干净的 Python 3.12+ 虚拟环境中运行，避免 Anaconda 基础环境的 MKL/OpenMP 与 PyTorch DLL 冲突：

```powershell
python -m venv .venv-rag-evaluation
.\.venv-rag-evaluation\Scripts\python.exe -m pip install -e ".[rag-evaluation]"
.\.venv-rag-evaluation\Scripts\python.exe scripts\run_neural_rag_evaluation.py --model BAAI/bge-small-zh-v1.5
```

报告写入 `output/rag_evaluation/neural_rag_evaluation_report.json` 和 `.md`，记录模型、池化方式、设备、向量维度、PyTorch/Transformers 版本、逐用例结果和限制。报告和模型文件均为可重复生成产物，不纳入版本控制。

公开 benchmark 固定子集复现：

```powershell
.\.venv-rag-evaluation\Scripts\python.exe scripts\fetch_public_rag_benchmarks.py
.\.venv-rag-evaluation\Scripts\python.exe scripts\run_public_rag_benchmarks.py --model BAAI/bge-small-en-v1.5 --sample-size 100 --top-k 4 --seed 20260712
# 全量可评测范围（CPU 参考耗时约 52 分钟）
.\.venv-rag-evaluation\Scripts\python.exe scripts\run_public_rag_benchmarks.py --model BAAI/bge-small-en-v1.5 --sample-size 7405 --top-k 4 --seed 20260712 --output-dir output/rag_evaluation/full_public
```

获取脚本只使用 MultiHop-RAG、ConditionalQA、HotpotQA 官方 GitHub 仓库以及 `hotpotqa/hotpot_qa` 官方 Hugging Face 数据集。报告保存源仓库 revision 和数据文件 SHA-256。快速报告使用每数据集 100 条；`full_public/` 报告覆盖 MultiHop-RAG 2255 条可映射证据查询、ConditionalQA 271 条可回答且证据可解析的开发样本，以及 HotpotQA 全部 7405 条 distractor validation。被排除样本数量和规则写入报告；该结果仍不是官方 leaderboard 提交。逐样本 JSON 可由命令重建且不纳入版本控制，仓库只保留全量摘要 Markdown。

导入 `D:\RAG_test` 已完成的真实 BGE Large、BGE reranker 和本地 Qwen 公共数据实验，并独立重算配对统计：

```powershell
$bgeSnapshot = (Get-ChildItem -Directory D:\RAG_test\.hf_cache\hub\models--BAAI--bge-large-en-v1.5\snapshots | Select-Object -First 1).FullName
$rerankerSnapshot = (Get-ChildItem -Directory D:\RAG_test\.hf_cache\hub\models--BAAI--bge-reranker-large\snapshots | Select-Object -First 1).FullName
python scripts/run_frc_wo_reranker_ablation.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --model-name $bgeSnapshot --device cuda
python scripts/run_frc_chunk_length_sensitivity.py `
  --source-role-scores D:\RAG_test\frc-select\outputs\role_scores\role_scores_conditionalqa.jsonl `
  --output output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --reranker-model-path $rerankerSnapshot --device cuda `
  --chunk-lengths 64,128,256 --overlap-ratio 0.2
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_conflicts_frc_ablation `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_housing_frc_ablation `
  --hf-home D:\RAG_test\.hf_cache `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_lawshift_temporal_ablation `
  --hf-home D:\RAG_test\.hf_cache
D:\anaconda3\envs\rag_exp\python.exe -m scripts.prepare_eurlex_temporal_source `
  --refresh-query --refresh-documents
D:\anaconda3\envs\rag_exp\python.exe -m scripts.run_eurlex_temporal_ablation `
  --hf-home D:\RAG_test\.hf_cache
python scripts/import_frc_public_reference.py `
  --reference-root D:\RAG_test\frc-select `
  --conflicts-path output\rag_evaluation\conflicts_frc\conflicts_frc_report.json `
  --supplemental-ablation output\rag_evaluation\public_frc_reference\wo_reranker_conditionalqa.json `
  --chunk-length-sensitivity output\rag_evaluation\public_frc_reference\chunk_length_sensitivity_conditionalqa.json `
  --controlled-domain-sensitivity output\rag_evaluation\controlled_domain_sensitivity\controlled_domain_sensitivity.json `
  --conflicts-ablation output\rag_evaluation\conflicts_frc_ablation\conflicts_frc_ablation.json `
  --housing-ablation output\rag_evaluation\housing_frc_ablation\housing_frc_ablation.json `
  --lawshift-ablation output\rag_evaluation\lawshift_temporal_ablation\lawshift_temporal_ablation.json `
  --eurlex-ablation output\rag_evaluation\eurlex_temporal_ablation\eurlex_temporal_ablation.json
```

字段/角色权重和冲突阈值的受控诊断可独立复现：

```powershell
python scripts/run_frc_controlled_sensitivity.py
```

该诊断使用仓库构造的 `SYNTHETIC` 小型领域基准和确定性选择器，不使用神经模型；它只证明参数可审计、阈值行为可辨识，不能替代公开数据真实模型实验或解除 Gate 2。

`w/o Reranker` 会同时用 BGE 双编码器重算相关性和角色分，不复用 Cross-Encoder 角色分；当前 285 例 Evidence F1 为 0.697327，完整 FRC 为 0.705754。Google CONFLICTS 的 458 例冻结真实模型候选池完成 `w/o Conflict`，Full/消融准确率为 0.334061/0.338428。HousingQA 的 40 个公开专家复合用例完成 `w/o Field` 与辖区适用性消融：Full 相对 `w/o Field` 字段覆盖提高 0.050000（95% CI [+0.018750, +0.087500]），但与最强字段分解基线持平；LawShift 的 124 例、31 类专家审阅修订完成版本替换消融，Full 相对 `w/o Applicability` 精确版本证据提高 0.137097（95% CI [+0.080645, +0.201613]），但与公平的适用性过滤 Cross-Encoder 基线持平。EUR-Lex/CELLAR 的 30 对权威废止边界形成 60 例生效/失效日期用例，Full 相对无适用性过滤的精确证据提高 0.516667（95% CI [+0.383333, +0.650000]），但与获得相同日期元数据的公平过滤 Cross-Encoder 同为 1.000000。9/9 命名消融与适用性三部分构造均已有执行工件；总体覆盖仍为 `PARTIAL`，因为公开真实模型字段/角色权重扫描、同一防汛领域复现和双专家评判未完成，结果不支持 Gate 2 放行。

分块长度敏感性对 ConditionalQA 全部 285 例执行 64/128/256-token、20% overlap 的真实 reranker 重评分，并按唯一父证据 ID 评价。FRC Evidence F1 为 0.614475/0.632662/0.679077，相对每档最强覆盖贪心代理均略低且 95% CI 跨 0；FRC 重复父证据率由 64-token 的 0.324912 降至 256-token 的 0.122807。

该导入只读参考目录，关键输入写入 SHA-256；历史 `setr_style` 在报告中统一正名为 `coverage_greedy_proxy`。协议与 Gate 2 解释见 `docs/progressive_upgrade/frc_public_evaluation_protocol.md`。

Google CONFLICTS 冲突与过时信息全量评测：

```powershell
conda run -n rag_exp python -m scripts.run_conflicts_frc_evaluation `
  --hf-home D:\RAG_test\.hf_cache `
  --generator-model-path D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4
```

该流程对 458 例官方数据执行六方法同预算选择与本地 Qwen 五类冲突分类，缓存逐样本分数和检查点，只提交汇总报告。结果位于 `output/rag_evaluation/conflicts_frc/`；它不是论文 expected-behavior adherence 的官方复现。

`scripts.run_conflicts_frc_ablation` 复用同一真实模型候选池，并仅对证据序列发生变化的提示增量调用 Qwen；0.25/0.40/0.55/0.70/0.85 冲突角色阈值的准确率为 0.336245/0.336245/0.334061/0.334061/0.325328。该扫描为事后诊断，不能用于重新挑选测试参数或解除 Gate 2。

区县证据集独立双人标注和第三方裁决：

```powershell
# 生成不含 gold 标签、检索分数、可信度和冲突信息的盲化包及两份空白表单
python scripts\rag_annotation_workflow.py prepare --output-dir output\rag_annotation\round-1 --seed 20260712

# 两名标注员各自完成表单后计算一致性和逐题分歧
python scripts\rag_annotation_workflow.py compare `
  --package output\rag_annotation\round-1\annotation_package.json `
  --first output\rag_annotation\round-1\annotator-a.json `
  --second output\rag_annotation\round-1\annotator-b.json `
  --output output\rag_annotation\round-1\comparison.json

# 第三名独立裁决员完成裁决表后生成带哈希溯源的最终 benchmark
python scripts\rag_annotation_workflow.py finalize `
  --package output\rag_annotation\round-1\annotation_package.json `
  --first output\rag_annotation\round-1\annotator-a.json `
  --second output\rag_annotation\round-1\annotator-b.json `
  --adjudication output\rag_annotation\round-1\adjudication-template.json `
  --output output\rag_annotation\round-1\adjudicated_benchmark.json
```

`output/rag_annotation/round-1/PROTOCOL.md` 是分发协议。工具强制两名标注员身份不同、裁决员不得兼任、每题恰好一条决定、每个已选证据都有合法角色，并输出 Cohen's kappa、证据 Jaccard、角色/答案一致率和提交哈希。仓库中的表单当前保持空白，只有实际人员独立完成并裁决后才能作为正式人工标注结论。

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
python scripts\inspect_demo_db.py
python -m pytest
Set-Location d:\graduation_project\frontend
npm.cmd run build
npm.cmd run test -- --run
npm.cmd run test:e2e
```

说明：当前 Cesium 构建仍会提示 chunk 较大，`protobufjs` 也会输出 `eval` 警告，这是三维依赖带来的既有构建警告，不影响当前 demo 功能。

## 文档入口

- [文档索引](./docs/README.md)
- [v0.3.0 更新报告](./docs/releases/v0.3.0.md)
- [v0.2.0 清理与结构重构报告](./docs/releases/v0.2.0.md)
- [渐进式升级与验收](./docs/progressive_upgrade/README.md)
- [全目标完成度追溯审计](./docs/progressive_upgrade/completion_traceability_audit.md)
- [89 条设计合同逐条审计](./output/acceptance/design_contract_audit.md)
- [旧系统可复现基线冻结报告](./output/acceptance/legacy_baseline_manifest.md)
- [机器可读完成性审计](./output/acceptance/progressive_completion_audit.md)
- [AgentTwin 兼容演示资料](./docs/agent_twin_upgrade/README.md)
- [甲方演示脚本](./docs/agent_twin_upgrade/16_甲方演示脚本.md)
