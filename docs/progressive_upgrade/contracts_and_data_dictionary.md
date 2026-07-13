# 核心合同、权限矩阵与数据字典

## 写入边界

`/response/*` 是响应域正式状态唯一写入口。`/platform/*`、`/agent-twin/*` 和 `/response/legacy/*` 只承担旧展示、只读兼容或受控映射；Legacy Adapter 不写审批、任务和状态表。AI/RAG 只产生候选、证据包和草案，不具备审批、下发或状态迁移权限。

## 角色权限矩阵

| 动作 | 值班员 | 业务复核 | 授权审批 | 联络员 | 现场执行 | 审计 | 管理员 | 外部模拟服务 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 创建预警事件/候选 | 是 | 是 | 是 | 否 | 否 | 否 | 是 | 否 |
| 导入风险对象主数据（AAL2） | 否 | 是 | 否 | 否 | 否 | 否 | 是 | 否 |
| 确认或排除对象 | 是 | 是 | 是 | 否 | 否 | 否 | 是 | 否 |
| 审查证据/裁决冲突 | 否 | 是 | 是 | 否 | 否 | 只读 | 是 | 否 |
| 编辑并提交草案 | 是 | 是 | 是 | 否 | 否 | 否 | 是 | 否 |
| 一般审批 | 否 | 是 | 是 | 否 | 否 | 否 | 否 | 否 |
| 高风险审批/AAL2 | 否 | 否 | 是 | 否 | 否 | 否 | 否 | 否 |
| 接收/分派 | 否 | 是 | 是 | 是 | 否 | 否 | 否 | 否 |
| 执行反馈 | 否 | 否 | 人工接管时 | 是 | 仅本人任务 | 否 | 否 | 否 |
| 完成核验 | 是 | 是 | 是 | 否 | 不得核验本人结果 | 只读 | 否 | 否 |
| 延期审批 | 否 | 是 | 是 | 只能申请 | 只能申请 | 否 | 否 | 否 |
| 撤回任务 | 否 | 否 | 是/AAL2 | 否 | 否 | 否 | 否 | 否 |
| 功能开关/迁移 | 否 | 否 | 否 | 否 | 否 | 只读 | 是/AAL2 | 否 |
| 提交模拟下发回调 | 否 | 否 | 否 | 否 | 否 | 否 | 否 | 仅专用入口、Mock Token 与幂等键匹配 |

执行人和核验人必须分离；高风险草案起草人不得自批；延期申请人不得审批本人申请。

## 状态转换表

| 当前状态 | 允许进入 |
|---|---|
| `draft` | `pending_approval`、`waived`、`cancelled` |
| `pending_approval` | `draft`、`issued`、`waived`、`cancelled` |
| `issued` | `acknowledged`、`blocked`、`escalated`、`taken_over`、`waived`、`cancelled` |
| `acknowledged` | `in_progress`、`blocked`、`escalated`、`taken_over`、`waived`、`cancelled` |
| `in_progress` | `blocked`、`partially_completed`、`pending_verification`、`escalated`、`taken_over`、`waived`、`cancelled` |
| `blocked` | `acknowledged`、`in_progress`、`escalated`、`taken_over`、`waived`、`cancelled` |
| `partially_completed` | `pending_verification`、`blocked`、`escalated`、`taken_over`、`waived`、`cancelled` |
| `pending_verification` | `completed`、`in_progress`、`escalated`、`waived`、`cancelled` |
| `escalated` | `acknowledged`、`in_progress`、`blocked`、`taken_over`、`waived`、`cancelled` |
| `taken_over` | `pending_verification`、`blocked`、`waived`、`cancelled` |
| `completed` / `waived` / `cancelled` | 无；终态 |

代码权威表位于 `flood_system/response_workflow/state_machine.py`，测试穷举所有状态对并验证终态无出边。

## 核心实体

| 实体 | 主键/版本 | 关键不变量 |
|---|---|---|
| ResponseEvent | `event_id`、固定 `workflow_engine_version` | 一个响应事件对应一条预警处置主线；关闭后只读 |
| AlertSnapshot | `snapshot_id`、事件内 `version` | 原始载荷、来源和 SHA-256 不可覆盖；更新使对象 STALE |
| EventRiskObject | `event_id + object_id`、`version` | 来源、数据版本、模拟标记、缺失项、校准置信度；STALE 不可成案 |
| RiskObjectVersionSnapshot | `snapshot_id` | 候选、更新、STALE 和人工决定全版本不可变 |
| RiskObjectRegistryRecord | `area_id + object_id`、`registry_version` | CSV/XLSX/JSON/GeoJSON/API 共用主数据；逐对象内容哈希控制版本；重复记录只指向规范对象；状态、有效期、坐标和敏感字段受控 |
| RiskObjectRegistryVersionSnapshot | `snapshot_id`、`area_id + object_id + registry_version` | 每次实际对象内容/来源版本变化形成不可更新、不可删除的完整快照；密钥轮换只能通过授权事务重加密 |
| RiskObjectRegistryImportResult | `import_id`、源文件 SHA-256 | 保存格式、字节数、新增/更新/未变/隔离数量、隔离原因、受影响事件和 STALE 运行数；不保存文件明文 |
| CandidateRunRecord | `run_id` | 绑定预警、对象数据、算法、特征、参数和运行模式版本 |
| DocumentVersionRecord | `version_id` | 发布单位、辖区、生效时间、替代关系、条款、源哈希和索引版本 |
| EvidencePackageVersion | `package_id + version` | 字段 `SUPPORTED/CONFLICTED/MISSING`；冲突裁决生成新版本；冻结后哈希稳定 |
| ResponseTask | `task_id + version` | 对象、责任、动作、四时限、依据、证据、审批和下发哈希完整 |
| RuleEvaluationRecord | `evaluation_id` | `PASS/SOFT_WARNING/HARD_BLOCK`；HARD_BLOCK 不得审批 |
| ApprovalRecord | `approval_id` | 绑定任务版本、载荷哈希、证据哈希和规则集版本；不可修改删除 |
| OutboxMessage | `message_id`、唯一 `idempotency_key` | 发送前重算审批载荷哈希；不一致失败关闭 |
| DispatchCallbackRecord | `callback_id`、唯一 `idempotency_key`、外部 `version` | 重复回调幂等合并，乱序只记录；任何回调均不得直接修改正式任务状态 |
| IdempotencyRecord | `scope + idempotency_key`、`request_hash`、`status` | scope 绑定身份/终端/方法/路径；同键异请求拒绝；2xx 可原样重放；异常结果标记 `INDETERMINATE`，不得盲重试；载荷加密并纳入密钥轮换 |
| DeadlineExtensionRecord | `extension_id` | 独立申请与审批，不修改已审批基础载荷 |
| TimelineEntry | `entry_id`、哈希链 | 追加写；记录操作者、岗位、终端、前后状态和原因 |

## 统一错误码

| HTTP | code | 含义 |
|---:|---|---|
| 400 | `VALIDATION_ERROR` | 参数、业务字段或数据质量不合法 |
| 401 | `UNAUTHORIZED` | 签名身份缺失、过期、重放或无效 |
| 403 | `FORBIDDEN` | 岗位无权执行动作 |
| 403 | `ASSURANCE_REQUIRED` | 操作要求 AAL2 |
| 403 | `IDENTITY_MISMATCH` / `TERMINAL_MISMATCH` | 载荷身份与可信断言不一致 |
| 404 | `NOT_FOUND` | 事件、任务、证据包或旧资源不存在 |
| 409 | `VERSION_CONFLICT` | 乐观锁版本不一致 |
| 409 | `STATE_CONFLICT` | 状态机不允许当前转换 |
| 409 | `RULE_HARD_BLOCK` | 规则硬阻断或未决关键冲突 |
| 409 | `IDEMPOTENCY_KEY_REUSED` | 同一身份与路由作用域中的幂等键已用于不同请求 |
| 409 | `IDEMPOTENCY_REQUEST_IN_PROGRESS` | 等价写请求正在处理，可使用同一请求稍后重试 |
| 409 | `IDEMPOTENCY_OUTCOME_INDETERMINATE` | 原请求可能已写入但未安全完成，必须先人工对账 |
| 409 | `IDEMPOTENCY_RESPONSE_NOT_REPLAYABLE` | 原请求已完成但响应超过重放上限，按记录的资源引用查询 |
| 413 | `REQUEST_TOO_LARGE` | 风险对象文件导入请求超过服务端请求体上限，在 JSON/Base64 解析前拒绝 |

错误响应统一为 `detail: { code, message, retryable }`。

## 风险对象主数据文件合同

- `POST /response/risk-objects/imports` 接受结构化 JSON；`POST /response/risk-objects/file-imports` 接受文件名、MIME、Base64 和 SHA-256 组成的 JSON 封装。两个写入口都经过可信身份、AAL2、RBAC 和通用幂等账本。
- 文件只允许 UTF-8 CSV、无宏/无外链/无公式 XLSX、JSON 和 Point GeoJSON；默认解码后不超过 10 MiB、5000 行、100 列和每单元格 20000 字符。ZIP 成员路径、成员数、解压总量和压缩比均在 openpyxl 读取前校验。
- 非法行进入 `RiskObjectImportQuarantineItem`，合法行仍可在同一事务导入；同批重复 ID、悬空/自引用/链式 `duplicate_of` 不会污染主数据。
- 逐对象 `content_hash` 决定是否增加 `registry_version`；整个源文件 SHA-256 只用于来源溯源，文件中其他行变化不会令未变化对象虚增版本。
- 主数据实际变化使同区域既有 CandidateRun 进入 `stale`，已形成的相同事件对象也进入 `STALE` 并生成不可变版本；重新筛查和人工核验前禁止成案。
