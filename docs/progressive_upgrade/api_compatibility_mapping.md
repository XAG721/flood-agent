# 新旧接口与概念映射

新版公开前缀为 `/api/v1`，中间件将其映射到相同契约的 `/response` Core API。下表记录设计文档概念名与当前实现名；未列出的旧 `/platform`、`/agent-twin` 入口不能写响应域新表。

| 设计概念 | Core API 实现 | 说明 |
|---|---|---|
| Warning / revision | `POST /events`、`POST /events/{event_id}/alerts` | Event 固定工作流版本；AlertSnapshot 保存原始内容、哈希和生命周期。撤销/过期通过新 revision 的 `lifecycle_status` 表达，不覆盖旧版。 |
| Risk-object master registry | `GET /risk-objects`、`GET /risk-objects/versions`、`GET /risk-objects/imports`、`POST /risk-objects/imports`、`POST /risk-objects/file-imports` | API/CSV/XLSX/JSON/Point GeoJSON 导入到区域主数据；AAL2、逐对象哈希版本、隔离区、字段脱敏和不可变历史。 |
| CandidateRun | `POST /events/{event_id}/risk-objects/discover`、`GET /events/{event_id}/candidate-runs` | 优先读取 Core 主数据，空区域才使用旧 EntityProfile 只读适配器；持久化算法、特征、主数据指纹、校准和空间关联模式。主数据变化后运行进入 STALE。 |
| Candidate confirm/exclude | `POST /events/{event_id}/risk-objects/{object_id}/verify` | 人工决定生成不可变对象版本；预警更新后对象变为 STALE。 |
| Confirmed candidate list | `GET /events/{event_id}/candidate-object-lists`、`POST /events/{event_id}/candidate-object-lists/freeze` | 业务复核/授权审批冻结最新预警下已确认对象；清单按版本追加、哈希稳定且不可更新删除，后续任务绑定最新清单 ID、版本和哈希。 |
| Event risk-object supplement | `POST /events/{event_id}/risk-objects` | 人工补充到事件上下文，不替代区域主数据导入；仍需人工核验并形成事件对象版本。 |
| Document/version/parse/publish | `POST /documents`、`GET /documents` | 单次命令完成源哈希、条款解析、版本登记和索引；相同内容幂等，替代版本令旧索引失效。 |
| Evidence run/package | `POST /events/{event_id}/risk-objects/{object_id}/task-draft`、`GET /evidence-packages/{package_id}` | 草案请求先独立生成 EvidencePackageVersion；证据不足时不创建任务。 |
| Manual evidence/freeze | `POST /evidence-packages/{id}/manual-evidence`、`POST /evidence-packages/{id}/freeze` | 人工补证和冻结均创建新版本；缺失或未决冲突时拒绝冻结。 |
| Conflict resolve | `POST /evidence-packages/{id}/conflicts/{conflict_id}/resolve` | 选中来源、理由和裁决人写入新证据包版本。 |
| Task draft/version/validate/submit | `POST .../task-draft`、`PATCH /tasks/{id}`、`GET /tasks/{id}/versions`、`POST /tasks/{id}/submit` | 更新携带 `expected_version`；提交前运行规则与证据门禁。 |
| Approval approve/reject/revision | `POST /tasks/{id}/decision` | 决定绑定任务、证据和规则哈希；拒绝返回草稿并形成新版本路径。 |
| Issue/ack/start/feedback/blocked/verify | `/tasks/{id}/decision`、`acknowledge`、`assign`、`start`、`feedback`、`verify-completion` | 只有明确命令，无通用 status 修改接口。 |
| Extension/cancel/takeover/escalate | `deadline-extensions`、`cancel`、`take-over`、`deadline-sweep` | 延期独立审批；撤回和人工接管要求授权及理由；四时限巡检幂等。 |
| Simulated dispatch/callback | `POST /dispatch/outbox/process`、`GET /dispatch/outbox/{id}/callbacks`、`POST /simulation/dispatch-callbacks` | 支持正常、超时、拒收、部分成功、重复和乱序；外部服务身份仅能访问回调入口，且回调不推进正式任务状态。 |
| Timeline/audit/transitions/replay/report | `events/{id}/timeline`、`audit-trail`、`tasks/{id}/transitions`、`events/{id}/replay`、`events/{id}/reports` | 回放只读且先验证哈希链；报告来自持久化事件证据。 |
| Legacy event read | `GET /legacy/events/{legacy_event_id}` | 只读投影，记录接口、映射版本、Trace、操作人和终端。 |

## 写入与兼容约束

1. `/api/v1` 与 `/response` 是同一 Core API 的路径别名，不是两个写服务，不构成双写。
2. Legacy Adapter 不提供 POST/PATCH/DELETE；旧事件继续在旧表续办时，不反写新 ResponseEvent。
3. 影子检索只记录 Baseline/FRC 差异，`SHADOW` 的正式证据选择保持 Baseline，不产生外部下发副作用。
4. 所有正式状态变更都经过身份断言、RBAC、身份/终端/路由绑定的通用请求幂等账本、业务幂等/版本检查、集中状态机和追加审计；2xx 原响应可重放，未知结果失败关闭。
