# 运行、功能开关与回滚手册

## 本地运行

后端：`uvicorn flood_system.api:app --host 127.0.0.1 --port 8000`。

前端：在 `frontend` 目录执行 `npm run dev`。

容器模拟环境：`docker compose up --build`，前端位于 `http://127.0.0.1:8080`，后端位于 `http://127.0.0.1:8000`，PostGIS 影子迁移实例默认绑定 `127.0.0.1:5432`。

Compose 中的 `worker` 独立执行幂等 Outbox 发送和四时限巡检；单次诊断可运行 `python scripts/run_response_worker.py --once`。Worker 只调用确定性工作流服务，异常保持失败关闭并在下一周期重试。

SQLite 到 PostGIS 的演练使用 `scripts/migrate_response_to_postgis.py`，只写 `flood_simulation`，并生成批次、隔离、数量/哈希和空间投影报告。映射 v5 包含风险对象当前主数据/不可变版本/导入记录、CandidateObjectListVersion，以及文档版本、加密源快照、解析、生命周期和索引构建共 21 类记录；对象位置继续投影为 EPSG:4326 Point/GiST，非模拟记录失败关闭到隔离区。迁移失败或出现隔离项时 SQLite 继续保持权威源，不允许切换。

Compose 演示环境显式设置 `FLOOD_ALLOW_DEV_IDENTITY_HEADERS=1`，让同一容器网络内的前端代理使用模拟岗位头；该分支在 `production/prod` 环境无条件禁用。试点和生产必须删除此配置，并由可信 IdP 网关签发短时 HMAC/OIDC 身份断言。Core API 所有写请求必须带 `Idempotency-Key`，前端同时生成 `X-Correlation-ID`。

通用幂等账本默认保留 24 小时（`FLOOD_IDEMPOTENCY_TTL_SECONDS=86400`，允许范围 60 秒至 7 天），成功响应默认最多加密保存 8 MiB 用于原样重放（`FLOOD_IDEMPOTENCY_MAX_REPLAY_BYTES=8388608`，允许范围 1 KiB 至 64 MiB）。客户端重试必须复用业务请求的 `Idempotency-Key`，但身份断言 nonce 每次重新签发。同键异请求不得换请求体重试；出现 `IDEMPOTENCY_OUTCOME_INDETERMINATE` 时，先按事件/任务/记录 ID 和审计时间线对账，再由授权人员决定补偿或使用新键发起新命令。不得直接删除账本来强行重试。

受控重建模拟环境：`python scripts/reset_simulation_environment.py --confirm RESET-SIMULATION`。命令拒绝生产环境、拒绝工作区外数据库路径，并重新生成 FloodAgent-Bench。开发管理员单次签名断言可用 `python scripts/build_dev_admin_assertion.py --method GET --path /response/configuration/feature-flags` 生成；生产管理员必须由真实 IdP/MFA 提供，不能使用该脚本。

风险对象文件导入默认解码上限由 `FLOOD_INGESTION_MAX_FILE_BYTES=10485760` 控制，安全范围为 1 KiB—50 MiB；请求体上限按 Base64 膨胀和 1 MiB JSON 余量自动计算。调大前必须同时评估反向代理限制、进程内存、行数上限和上传超时，不得靠取消 ZIP/公式/哈希检查提高吞吐。

探针：`/health` 检查进程，`/ready` 检查数据库，`/metrics` 暴露响应事件、Outbox、风险对象主数据 active/inactive 数、被冻结候选清单数、被台账变更失效的 CandidateRun、累计隔离行和按 `processing/completed/indeterminate` 分组的幂等账本指标。指标查询只读加密账本，不产生清理写入。

回滚旧应用时保留 `response_candidate_object_lists`。它是成案来源凭证，不得删表或修改历史版本来绕过绑定检查；恢复新版本后应先核对最新预警、最新冻结版本、对象核验哈希和已有任务绑定，再恢复写流量。

回滚不删除 `response_document_versions`、`response_document_sources`、`response_document_parses`、`response_document_lifecycle_events`、`response_index_builds` 或 `response_contract_versions`。旧应用不认识这些表时保持只读；恢复新版本后先核对源文件 SHA-256、最后生命周期、最近成功索引、替代链以及任务/审批绑定的 Schema 和规则集版本。索引重建失败时继续使用最后成功版本，不得把失败构建标为已发布。

## 功能开关

通过 `PUT /response/configuration/feature-flags` 管理。请求必须使用管理员签名身份和 AAL2，且提供变更理由。支持 `environment`、`event_id`、`role`、`scenario` 作用域；更具体的作用域优先。

生产环境未配置的高风险功能默认关闭。开发模拟环境允许当前已验证的确定性闭环功能默认开启。

## 回滚

1. 代码：恢复上一个可运行镜像或提交；不删除升级后产生的数据。
2. 配置：把事件级功能开关恢复到前一版本，保留操作者和原因。
3. 检索：FRC-RAG 未通过门禁时切回 `BASELINE_ONLY`，证据包保留只读。
4. 下发：失败消息保持在 Outbox；哈希不一致不得重试为成功，转人工接管。
5. 数据：优先向前修复；审批、任务、反馈、时间线和归档不得因技术回滚删除。
6. 业务：状态机不可用时暂停受影响功能，不允许旧页面直接修改新版正式状态。

## 生产前必检

- 设置独立 `FLOOD_DATA_ENCRYPTION_KEY`，禁止使用开发侧车密钥；
- 部署可信 HTTPS 终止代理并启用 `FLOOD_REQUIRE_HTTPS=1`；
- 配置真实身份提供方和 AAL2；
- 审核功能开关初始状态；
- 执行备份、跨主机导入、隔离恢复和 Outbox 对账；
- 确认模拟端点与未来真实端点网络隔离。

## 旧链路退役（M7）

旧写接口关闭后先保持只读，不因受控测试通过而删除。只有生产观测证明旧接口流量持续归零、没有在途旧事件，生产数据归档已完成独立恢复演练，并且旧服务账号与写权限均已撤销，才允许批准 M7 和移除旧链路。任一条件缺少真实运行证据时均保持 `NO-GO_EXTERNAL`；仓库中的模拟流量、截图或自动测试不能替代该批准。
