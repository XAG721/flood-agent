# FloodAgent-Bench 24 场景自动化验收追踪

- 数据集版本：`floodagent-bench-v2`
- 场景覆盖：24/24
- 判定：`PASS`
- 说明：覆盖表示场景已绑定到全量 CI 执行的自动测试；不替代真实岗位或生产环境验收。

| 场景 | 名称 | 自动化证据 | 状态 |
|---|---|---|---|
| SIMSCENARIO-01 | 正常预警—对象—任务闭环 | `tests/test_response_workflow.py::test_full_deterministic_response_loop_and_close` | covered |
| SIMSCENARIO-02 | 对象坐标缺失 | `tests/test_response_workflow.py::test_candidate_discovery_excludes_missing_coordinates_with_explicit_limitation` | covered |
| SIMSCENARIO-03 | 对象坐标偏移或坐标系错误 | `tests/test_response_policies.py::test_geo_polygon_rejects_wrong_crs_and_out_of_bounds_coordinates` | covered |
| SIMSCENARIO-04 | 对象别名、重名和重复台账 | `tests/test_response_workflow.py::test_risk_object_alias_duplicate_is_merged_and_expired_registry_is_blocked` | covered |
| SIMSCENARIO-05 | 预警覆盖边界对象 | `tests/test_response_policies.py::test_point_in_polygon_includes_boundary_and_excludes_outside_point` | covered |
| SIMSCENARIO-06 | 台账过期 | `tests/test_response_workflow.py::test_risk_object_alias_duplicate_is_merged_and_expired_registry_is_blocked` | covered |
| SIMSCENARIO-07 | 台账遗漏后人工补充 | `tests/test_response_workflow.py::test_missing_registry_object_can_be_manually_added_and_confirmed` | covered |
| SIMSCENARIO-08 | 新旧预案同时存在 | `tests/test_response_workflow.py::test_document_versions_are_clause_indexed_and_superseded_versions_leave_current_retrieval` | covered |
| SIMSCENARIO-09 | 辖区适用性不一致 | `tests/test_response_workflow.py::test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution` | covered |
| SIMSCENARIO-10 | 数值或时限冲突 | `tests/test_frc_evaluation.py::test_conflict_detection_benchmark_reports_gate_two_metrics` | covered |
| SIMSCENARIO-11 | 责任主体冲突 | `tests/test_response_workflow.py::test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution` | covered |
| SIMSCENARIO-12 | 条款包含例外条件 | `tests/test_response_workflow.py::test_grounded_task_draft_covers_all_roles_and_keeps_plan_attribution` | covered |
| SIMSCENARIO-13 | 源文件没有某项任务依据 | `tests/test_response_workflow.py::test_grounded_task_draft_is_blocked_when_policy_evidence_is_missing` | covered |
| SIMSCENARIO-14 | OCR错误或文档解析失败 | `tests/test_response_workflow.py::test_document_parse_and_index_dependency_failures_are_isolated` | covered |
| SIMSCENARIO-15 | 向量索引、模型或规则服务中断 | `tests/test_response_workflow.py::test_document_parse_and_index_dependency_failures_are_isolated`<br>`tests/test_system.py::test_supervisor_loop_retries_and_records_warning` | covered |
| SIMSCENARIO-16 | AI输出非法JSON或虚构依据编号 | `tests/test_system.py::test_platform_api_returns_explicit_llm_errors_without_rule_fallback` | covered |
| SIMSCENARIO-17 | 审批驳回和重新生成 | `tests/test_response_workflow.py::test_task_versions_keep_complete_immutable_snapshots_and_approval_history` | covered |
| SIMSCENARIO-18 | 并发审批 | `tests/test_response_workflow.py::test_rule_engine_hard_blocks_ungrounded_ai_draft_and_optimistic_lock_conflict` | covered |
| SIMSCENARIO-19 | 重复下发和重复回调 | `tests/test_response_workflow.py::test_simulated_dispatch_fault_matrix_is_audited_without_mutating_task_state`<br>`tests/test_response_workflow.py::test_external_service_callback_is_token_bound_idempotent_and_state_safe` | covered |
| SIMSCENARIO-20 | 下发部分成功 | `tests/test_response_workflow.py::test_simulated_dispatch_fault_matrix_is_audited_without_mutating_task_state` | covered |
| SIMSCENARIO-21 | 执行受阻、申请延期和改派 | `tests/test_response_workflow.py::test_liaison_assignment_and_reassignment_control_the_field_executor`<br>`tests/test_response_workflow.py::test_deadline_extension_is_independently_approved_without_mutating_approved_payload`<br>`tests/test_response_workflow.py::test_resource_shortage_feedback_creates_escalation_with_alternatives` | covered |
| SIMSCENARIO-22 | 四类时限超时升级 | `tests/test_response_workflow.py::test_all_four_deadline_types_escalate_with_explicit_audit` | covered |
| SIMSCENARIO-23 | 核验退回整改 | `tests/test_response_workflow.py::test_verification_can_return_completion_for_rework_without_losing_feedback` | covered |
| SIMSCENARIO-24 | 通信中断后人工接管和恢复补录 | `tests/test_response_workflow.py::test_manual_takeover_and_commander_cancellation_are_audited_terminal_branches`<br>`tests/test_response_workflow.py::test_simulated_dispatch_timeout_stays_pending_and_recovers_idempotently` | covered |
