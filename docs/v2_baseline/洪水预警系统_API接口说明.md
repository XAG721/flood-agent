# 洪水预警系统 API 接口说明

## 1. 说明

当前系统全部以 `/v2/*` 作为正式接口前缀。旧版 `/incidents/*` 和早期原型接口不再作为现行文档维护对象。

基础地址默认是：

```text
http://127.0.0.1:8000
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

## 2. 通用约定

- 请求与响应默认使用 `application/json`
- 时间字段默认使用 `ISO 8601`
- 大模型生成阶段失败时，相关接口会返回 `503`
- 常规错误响应格式：

```json
{
  "detail": "error message"
}
```

## 3. 健康与运维

### `GET /health`

基础健康检查。

### `GET /v2/supervisor/status`

查询 supervisor loop 状态，包括：

- `running`
- `interval_seconds`
- `consecutive_failures`
- `circuit_state`
- `last_success_at`
- `last_failure_at`
- `last_retry_at`
- `last_completed_at`
- `pending_trigger_count`
- `recent_replay_count`
- `recent_timeline_failure_count`

### `GET /v2/alerts`

查询站内运行告警。

支持过滤参数：

- `event_id`
- `severity`
- `source_type`
- `from_ts`
- `to_ts`
- `limit`

### `GET /v2/audit/records`

查询审计记录，过滤参数与告警接口一致。

### `GET /v2/archive/status`

查询归档状态。

### `POST /v2/archive/run`

手动触发一次归档周期，需要具备 `archive_run` 权限。

## 4. 事件、观测与模拟风险

### `POST /v2/events`

创建事件。

示例：

```json
{
  "area_id": "beilin_10km2",
  "title": "碑林区积涝联动处置事件",
  "trigger_reason": "前端控制台创建",
  "operator": "frontend_console"
}
```

### `POST /v2/events/{event_id}/observations`

写入观测数据并驱动常规风险更新。

### `POST /v2/events/{event_id}/simulation-updates`

写入洪水模拟结果。该接口是区域级主动请示的正式入口。

请求示例：

```json
{
  "generated_at": "2026-04-10T10:30:00Z",
  "depth_threshold_m": 0.5,
  "flow_threshold_mps": 1.5,
  "cells": [
    {
      "cell_id": "grid_a01",
      "label": "东关南街北段",
      "water_depth_m": 0.72,
      "flow_velocity_mps": 1.11
    },
    {
      "cell_id": "grid_a02",
      "label": "建国路口",
      "water_depth_m": 0.38,
      "flow_velocity_mps": 0.96
    }
  ]
}
```

响应示例：

```json
{
  "event_id": "event_xxx",
  "overall_risk_level": "Red",
  "risk_stage_key": "risk_stage_xxx",
  "trigger_id": "trigger_xxx",
  "supervisor_run_id": "run_xxx",
  "queue_version": "b7f4c8d5a3e912ef",
  "llm_status": "ok",
  "llm_error": null
}
```

### `GET /v2/events/{event_id}/hazard-state`

查询事件当前 hazard state。

### `GET /v2/entities/{entity_id}/impact`

查询对象级 impact。

### `POST /v2/advisories/generate`

生成对象级 advisory。

说明：

- 当前 advisory 由 LLM 生成
- 默认不进入区域级可确认 proposal 队列

## 5. 区域级主动请示

### `GET /v2/proposals/pending`

返回跨事件的待确认区域 proposal 快照。

返回结构：

- `queue_version`
- `generated_at`
- `items`

### `GET /v2/events/{event_id}/regional-proposals`

返回当前事件的区域 proposal 历史。

可选参数：

- `status`

### `PATCH /v2/proposals/{proposal_id}/draft`

更新区域 proposal 草稿，仅允许 `commander` 编辑 `action_scope`。

### `POST /v2/proposals/{proposal_id}/approve`

批准区域 proposal。批准后会调用执行材料生成链，落通知稿与执行日志。

### `POST /v2/proposals/{proposal_id}/reject`

驳回区域 proposal。

### `GET /v2/proposals/stream`

SSE 推送跨事件待确认区域 proposal 队列。前端全局对话框依赖此接口。

### 区域 proposal 关键字段

当前 `ActionProposal` 已支持以下区域级能力字段：

- `proposal_scope`
- `action_type`
- `execution_mode`
- `action_display_name`
- `action_display_tagline`
- `action_display_category`
- `high_risk_object_ids`
- `action_scope`
- `risk_stage_key`
- `system_version_hash`
- `generation_source`
- `model_name`
- `prompt_profile`
- `grounding_summary`

说明：

- `action_type` 当前允许开放式值，不再限制为固定三类
- 前端优先展示 `action_display_name/tagline/category`
- `execution_mode` 用于决定进入通知、转移、调度或通用行动闭环

## 6. Copilot 会话接口

### `POST /v2/copilot/sessions/bootstrap`

创建或初始化 Copilot 会话。

### `GET /v2/copilot/sessions/{session_id}`

获取会话视图，包含：

- `messages`
- `latest_answer`
- `proposals`
- `notification_drafts`
- `execution_logs`
- `memory_snapshot`
- `plan_runs`
- `recent_tool_executions`

### `GET /v2/copilot/sessions/{session_id}/memory`

获取会话级 memory bundle。

### `POST /v2/copilot/sessions/{session_id}/messages`

向 Copilot 发送消息。当前最终回答正文由 LLM 生成。

### 会话级 proposal 接口

- `POST /v2/copilot/sessions/{session_id}/proposals/{proposal_id}/approve`
- `POST /v2/copilot/sessions/{session_id}/proposals/{proposal_id}/reject`
- `POST /v2/copilot/sessions/{session_id}/proposals/batch-approve`
- `POST /v2/copilot/sessions/{session_id}/proposals/batch-reject`

说明：

- 这些接口仍保留兼容能力
- 当前区域级主动请示主路径不依赖这些 session proposal 接口

## 7. Runtime Admin

### 对象画像

- `GET /v2/admin/entity-profiles`
- `GET /v2/admin/entity-profiles/{entity_id}`
- `POST /v2/admin/entity-profiles`
- `PUT /v2/admin/entity-profiles/{entity_id}`
- `DELETE /v2/admin/entity-profiles/{entity_id}`

### 资源状态

- `GET /v2/admin/areas/{area_id}/resource-status`
- `PUT /v2/admin/areas/{area_id}/resource-status`
- `GET /v2/admin/events/{event_id}/resource-status`
- `PUT /v2/admin/events/{event_id}/resource-status`
- `DELETE /v2/admin/events/{event_id}/resource-status`

### RAG 文档

- `GET /v2/admin/rag-documents`
- `POST /v2/admin/rag-documents/import`
- `POST /v2/admin/rag-documents/reload`

## 8. Dataset Pipeline

### `GET /v2/admin/dataset/status`

查询数据管线状态，包含：

- dataset source 列表
- `raw_ready`
- `raw_completeness_percent`
- `missing_required_sources`
- `stale_sources`
- `raw_cache_health`
- `active_job`
- `recent_jobs`

### 数据任务接口

- `POST /v2/admin/dataset/fetch`
- `POST /v2/admin/dataset/build`
- `POST /v2/admin/dataset/validate`
- `POST /v2/admin/dataset/sync-demo-db`
- `GET /v2/admin/dataset/jobs`
- `POST /v2/admin/dataset/jobs/{job_id}/cancel`
- `POST /v2/admin/dataset/jobs/{job_id}/retry`

## 9. Multi-Agent、经验与评测

### 多代理与协作

- `GET /v2/events/{event_id}/agent-status`
- `GET /v2/events/{event_id}/agent-tasks`
- `GET /v2/events/{event_id}/agent-timeline`
- `GET /v2/events/{event_id}/trigger-events`
- `GET /v2/events/{event_id}/shared-memory`
- `GET /v2/events/{event_id}/supervisor-runs`
- `POST /v2/supervisor/tick`
- `POST /v2/events/{event_id}/supervisor/run`
- `POST /v2/agent-tasks/{task_id}/replay`

### 经验与评测

- `GET /v2/events/{event_id}/experience-context`
- `GET /v2/entities/{entity_id}/strategy-history`
- `GET /v2/agent-metrics`
- `GET /v2/events/{event_id}/decision-report`
- `GET /v2/evaluation/benchmarks`
- `POST /v2/evaluation/run`
- `GET /v2/evaluation/reports/{report_id}`
- `POST /v2/evaluation/reports/{report_id}/replay`

## 10. RBAC

当前系统内置角色：

- `observer`
- `street_operator`
- `district_operator`
- `commander`

### `GET /v2/security/capabilities`

返回当前角色的能力矩阵，用于前端区块和动作级权限控制。

当前常见动作标签包括：

- `创建事件`
- `导入监测数据`
- `导入模拟结果`
- `处理区域请示`
- `编辑请示草稿`
- `修改运行期数据`
- `管理数据集任务`
- `控制后台巡检`
- `重放智能体任务`
- `执行归档清理`
- `运行评测任务`
