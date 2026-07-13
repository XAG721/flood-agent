# 运行、功能开关与回滚手册

## 本地运行

后端：`uvicorn flood_system.api:app --host 127.0.0.1 --port 8000`。

前端：在 `frontend` 目录执行 `npm run dev`。

容器模拟环境：`docker compose up --build`，前端位于 `http://127.0.0.1:8080`，后端位于 `http://127.0.0.1:8000`，PostGIS 影子迁移实例默认绑定 `127.0.0.1:5432`。

Compose 中的 `worker` 独立执行幂等 Outbox 发送和四时限巡检；单次诊断可运行 `python scripts/run_response_worker.py --once`。Worker 只调用确定性工作流服务，异常保持失败关闭并在下一周期重试。

SQLite 到 PostGIS 的演练使用 `scripts/migrate_response_to_postgis.py`，只写 `flood_simulation`，并生成批次、隔离、数量/哈希和空间投影报告。迁移失败或出现隔离项时 SQLite 继续保持权威源，不允许切换。

Compose 演示环境显式设置 `FLOOD_ALLOW_DEV_IDENTITY_HEADERS=1`，让同一容器网络内的前端代理使用模拟岗位头；该分支在 `production/prod` 环境无条件禁用。试点和生产必须删除此配置，并由可信 IdP 网关签发短时 HMAC/OIDC 身份断言。Core API 所有写请求必须带 `Idempotency-Key`，前端同时生成 `X-Correlation-ID`。

受控重建模拟环境：`python scripts/reset_simulation_environment.py --confirm RESET-SIMULATION`。命令拒绝生产环境、拒绝工作区外数据库路径，并重新生成 FloodAgent-Bench。开发管理员单次签名断言可用 `python scripts/build_dev_admin_assertion.py --method GET --path /response/configuration/feature-flags` 生成；生产管理员必须由真实 IdP/MFA 提供，不能使用该脚本。

探针：`/health` 检查进程，`/ready` 检查数据库，`/metrics` 暴露响应事件和 Outbox 指标。

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
