# 核心合同、权限矩阵与数据字典

## 写入边界

`/response/*` 是响应域正式状态唯一写入口。`/platform/*`、`/agent-twin/*` 和 `/response/legacy/*` 只承担旧展示、只读兼容或受控映射；Legacy Adapter 不写审批、任务和状态表。AI/RAG 只产生候选、证据包和草案，不具备审批、下发或状态迁移权限。

## 角色权限矩阵

| 动作 | 值班员 | 业务复核 | 授权审批 | 联络员 | 现场执行 | 审计 | 管理员 | 外部模拟服务 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 创建预警事件/候选 | 是 | 是 | 是 | 否 | 否 | 否 | 是 | 否 |
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
| CandidateRunRecord | `run_id` | 绑定预警、对象数据、算法、特征、参数和运行模式版本 |
| DocumentVersionRecord | `version_id` | 发布单位、辖区、生效时间、替代关系、条款、源哈希和索引版本 |
| EvidencePackageVersion | `package_id + version` | 字段 `SUPPORTED/CONFLICTED/MISSING`；冲突裁决生成新版本；冻结后哈希稳定 |
| ResponseTask | `task_id + version` | 对象、责任、动作、四时限、依据、证据、审批和下发哈希完整 |
| RuleEvaluationRecord | `evaluation_id` | `PASS/SOFT_WARNING/HARD_BLOCK`；HARD_BLOCK 不得审批 |
| ApprovalRecord | `approval_id` | 绑定任务版本、载荷哈希、证据哈希和规则集版本；不可修改删除 |
| OutboxMessage | `message_id`、唯一 `idempotency_key` | 发送前重算审批载荷哈希；不一致失败关闭 |
| DispatchCallbackRecord | `callback_id`、唯一 `idempotency_key`、外部 `version` | 重复回调幂等合并，乱序只记录；任何回调均不得直接修改正式任务状态 |
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

错误响应统一为 `detail: { code, message, retryable }`。
