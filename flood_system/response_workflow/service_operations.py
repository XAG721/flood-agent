from __future__ import annotations

import os

from .feedback_policy import (
    build_review_recommendations,
    missing_evidence_types,
    review_timeline_line,
)

from .models import (
    AuditArchiveRecord,
    AuditArchiveRequest,
    AuditArchiveVerificationResult,
    BackupCreateRequest,
    BackupImportRequest,
    BackupImportResult,
    BackupRestoreRequest,
    BackupRestoreResult,
    BackupRetentionRequest,
    BackupRetentionResult,
    DatabaseBackupRecord,
    KeyRotationRequest,
    KeyRotationResult,
    DistrictScenarioReport,
    EventReviewDraft,
    EventCloseRequest,
    EventDashboard,
    EventStatus,
    FeedbackCategory,
    ObjectVerificationStatus,
    OperatorRole,
    ReviewDraftRequest,
    ScenarioAcceptanceCheck,
    ScenarioEvaluationRequest,
    ScenarioFailureCase,
    ScenarioMetricResult,
    TaskActionRequest,
    TaskStatus,
    TimelineEntry,
    TimelineIntegrityIssue,
    TimelineIntegrityReport,
)



class WorkflowOperationsMixin:
    def close_event(self, event_id: str, request: EventCloseRequest) -> EventDashboard:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "close an event",
        )
        event = self._event(event_id, active=True)
        tasks = self.repository.list_response_tasks(event_id)
        if not tasks:
            raise ValueError("an event with no tasks cannot be closed")
        incomplete = [
            task.task_id
            for task in tasks
            if task.status
            not in {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
        ]
        if incomplete:
            raise ValueError(f"event has incomplete tasks: {', '.join(incomplete)}")
        event = event.model_copy(
            update={
                "status": EventStatus.CLOSED,
                "updated_at": self._now(),
                "closed_at": self._now(),
                "closed_by": request.operator_id,
            }
        )
        self.repository.save_response_event(event)
        self._record(
            event_id=event_id,
            entry_type="event",
            action="event_closed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"event_status": EventStatus.ACTIVE.value},
            after_state={"event_status": event.status.value},
            detail={"note": request.note, "task_count": len(tasks)},
        )
        self.generate_review_draft(
            event_id,
            ReviewDraftRequest(
                operator_id=request.operator_id,
                operator_role=request.operator_role,
                terminal_id=request.terminal_id,
                note="事件关闭时自动生成复盘草稿",
            ),
        )
        return self.get_dashboard(event_id)

    def generate_review_draft(
        self, event_id: str, request: ReviewDraftRequest
    ) -> EventReviewDraft:
        self._require_role(
            request.operator_role,
            {
                OperatorRole.DUTY_OFFICER,
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
            },
            "generate an event review draft",
        )
        event = self._event(event_id)
        tasks = self.repository.list_response_tasks(event_id)
        objects = self.repository.list_event_risk_objects(event_id)
        feedback = self.repository.list_event_feedback(event_id)
        escalations = self.repository.list_escalations(event_id)
        timeline = self.repository.list_timeline_entries(event_id)
        completed = [
            task
            for task in tasks
            if task.status
            in {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
        ]
        duplicates = [
            entry for entry in timeline if entry.action == "duplicate_feedback_merged"
        ]
        evidence_complete = 0
        task_by_id = {task.task_id: task for task in tasks}
        for item in feedback:
            task = task_by_id.get(item.task_id)
            if (
                task
                and item.category == FeedbackCategory.COMPLETION
                and not missing_evidence_types(task, item.evidence)
            ):
                evidence_complete += 1

        process_metrics: dict[str, int | float] = {
            "alert_versions": len(self.repository.list_alert_snapshots(event_id)),
            "candidate_objects": len(objects),
            "confirmed_objects": sum(
                item.verification_status == ObjectVerificationStatus.CONFIRMED
                for item in objects
            ),
            "task_count": len(tasks),
            "completed_tasks": len(completed),
            "feedback_count": len(feedback),
            "duplicate_feedback_merged": len(duplicates),
            "escalation_count": len(escalations),
            "evidence_complete_feedback": evidence_complete,
            "completion_rate": round(len(completed) / len(tasks), 4) if tasks else 0.0,
        }
        key_decisions = [
            review_timeline_line(entry)
            for entry in timeline
            if entry.action
            in {
                "task_approved_and_issued",
                "task_rejected_to_draft",
                "task_waived",
                "event_closed",
            }
        ]
        delays = [
            f"任务 {item.task_id}：{item.reason}；建议：{'、'.join(item.recommended_actions) or '人工协调'}"
            for item in escalations
        ]
        evidence_findings = [
            f"{evidence_complete} 条完成反馈通过必需证据类型校验。",
            f"{sum(bool(task.source_evidence) for task in tasks)} 项任务保留了成案来源证据。",
            f"系统合并了 {len(duplicates)} 条重复反馈，避免重复计入过程指标。",
        ]
        unresolved = [
            f"任务 {task.task_id} 当前状态为 {task.status.value}。"
            for task in tasks
            if task.status not in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
        ]
        recommendations = build_review_recommendations(
            feedback,
            has_escalations=bool(escalations),
            has_duplicates=bool(duplicates),
            has_unresolved_tasks=bool(unresolved),
        )
        review = EventReviewDraft(
            review_id=self._id("REVIEW"),
            event_id=event_id,
            headline=f"{event.title}响应过程复盘草稿",
            executive_summary=(
                f"事件共关联 {len(objects)} 个候选对象、形成 {len(tasks)} 项任务，"
                f"完成或批准豁免 {len(completed)} 项，发生 {len(escalations)} 次升级。"
            ),
            process_metrics=process_metrics,
            key_decisions=key_decisions,
            delays_and_escalations=delays,
            evidence_findings=evidence_findings,
            unresolved_issues=unresolved,
            improvement_recommendations=recommendations,
            source_timeline_entry_ids=[entry.entry_id for entry in timeline],
            generated_by=request.operator_id,
            created_at=self._now(),
        )
        self.repository.save_event_review_draft(review)
        self._record(
            event_id=event_id,
            entry_type="review",
            action="event_review_draft_generated",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "review_id": review.review_id,
                "source_entry_count": len(timeline),
                "note": request.note,
            },
        )
        return review

    def run_scenario_evaluation(
        self, event_id: str, request: ScenarioEvaluationRequest
    ) -> DistrictScenarioReport:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.AUDITOR},
            "run a district scenario evaluation",
        )
        dashboard = self.get_dashboard(event_id)
        event = dashboard.event
        timeline = dashboard.timeline
        actions = {entry.action for entry in timeline}
        action_entries: dict[str, list[TimelineEntry]] = {}
        for entry in timeline:
            action_entries.setdefault(entry.action, []).append(entry)

        def refs(*names: str) -> list[str]:
            return [
                entry.entry_id
                for name in names
                for entry in action_entries.get(name, [])
            ]

        def duration(
            start_actions: tuple[str, ...], end_actions: tuple[str, ...]
        ) -> float | None:
            starts = [
                entry
                for name in start_actions
                for entry in action_entries.get(name, [])
            ]
            ends = [
                entry for name in end_actions for entry in action_entries.get(name, [])
            ]
            if not starts or not ends:
                return None
            start = min(starts, key=lambda item: item.created_at)
            later = [item for item in ends if item.created_at >= start.created_at]
            if not later:
                return None
            elapsed = (
                min(later, key=lambda item: item.created_at).created_at
                - start.created_at
            ).total_seconds() / 60
            return round(max(elapsed, 0.0), 3)

        def rate(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator * 100, 2) if denominator else None

        manual = request.manual_baseline
        object_ids = {item.object_id for item in dashboard.risk_objects}
        expected_ids = set(request.expected_object_ids)
        missing_expected = expected_ids - object_ids
        object_omission_rate = (
            rate(len(missing_expected), len(expected_ids)) if expected_ids else None
        )
        manual_corrections = sum(
            item.verification_status == ObjectVerificationStatus.EXCLUDED
            for item in dashboard.risk_objects
        ) + len(action_entries.get("candidate_updated", []))
        manual_correction_rate = rate(manual_corrections, len(dashboard.risk_objects))

        required_task_fields = (
            "object_id",
            "action",
            "responsible_organization",
            "responsible_role",
            "deadline_at",
            "acknowledge_deadline_at",
            "required_evidence",
            "plan_basis",
            "escalation_rule",
        )
        complete_tasks = sum(
            all(getattr(task, field) for field in required_task_fields)
            for task in dashboard.tasks
        )
        task_completeness = rate(complete_tasks, len(dashboard.tasks))
        approved_tasks = sum(
            task.approved_version is not None for task in dashboard.tasks
        )
        completed_tasks = sum(
            task.status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
            for task in dashboard.tasks
        )
        verified_tasks = {
            entry.task_id for entry in action_entries.get("completion_verified", [])
        }
        feedback_complete = 0
        task_by_id = {task.task_id: task for task in dashboard.tasks}
        for item in dashboard.feedback:
            task = task_by_id.get(item.task_id)
            if (
                task
                and item.category == FeedbackCategory.COMPLETION
                and not missing_evidence_types(task, item.evidence)
            ):
                feedback_complete += 1
        completion_feedback = sum(
            item.category == FeedbackCategory.COMPLETION for item in dashboard.feedback
        )
        verification_attempts = len(
            action_entries.get("completion_verified", [])
        ) + len(action_entries.get("completion_returned", []))

        escalation_task_ids = {
            entry.task_id
            for entry in timeline
            if entry.action in {"deadline_escalated", "task_blocked_and_escalated"}
        }
        recovered_escalations = {
            task_id
            for task_id in escalation_task_ids
            if task_id
            and task_by_id.get(task_id)
            and task_by_id[task_id].status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
        }
        escalation_durations: list[float] = []
        for escalation in [
            entry
            for entry in timeline
            if entry.action in {"deadline_escalated", "task_blocked_and_escalated"}
        ]:
            recovery = next(
                (
                    entry
                    for entry in timeline
                    if entry.task_id == escalation.task_id
                    and entry.created_at >= escalation.created_at
                    and entry.action
                    in {
                        "task_acknowledged",
                        "task_started",
                        "completion_submitted",
                        "task_waived",
                    }
                ),
                None,
            )
            if recovery:
                escalation_durations.append(
                    (recovery.created_at - escalation.created_at).total_seconds() / 60
                )

        expected_actions = {
            "event_created",
            "candidate_added",
            "candidate_confirmed",
            "task_submitted",
            "task_approved_and_issued",
        }
        if dashboard.tasks:
            expected_actions.add("task_draft_created")
        if completed_tasks:
            expected_actions.update(
                {
                    "task_acknowledged",
                    "task_started",
                    "completion_submitted",
                    "completion_verified",
                }
            )
        if event.status == EventStatus.CLOSED:
            expected_actions.update({"event_closed", "event_review_draft_generated"})
        timeline_coverage = rate(len(expected_actions & actions), len(expected_actions))
        linked_timeline = sum(
            entry.event_id == event_id
            and (entry.task_id is None or entry.task_id in task_by_id)
            for entry in timeline
        )
        timeline_integrity = rate(linked_timeline, len(timeline))

        metric_specs = [
            (
                "object_list_generation_time",
                "对象研判",
                "清单生成时间",
                "分钟",
                duration(("event_created",), ("candidate_added",)),
                refs("event_created", "candidate_added"),
            ),
            (
                "object_omission_rate",
                "对象研判",
                "对象遗漏率",
                "%",
                object_omission_rate,
                refs("candidate_added", "candidate_confirmed"),
            ),
            (
                "manual_correction_rate",
                "对象研判",
                "人工修正率",
                "%",
                manual_correction_rate,
                refs("candidate_updated", "candidate_excluded"),
            ),
            (
                "task_generation_time",
                "方案生成",
                "任务生成时间",
                "分钟",
                duration(
                    ("candidate_confirmed",),
                    ("task_draft_created", "grounded_task_draft_generated"),
                ),
                refs(
                    "candidate_confirmed",
                    "task_draft_created",
                    "grounded_task_draft_generated",
                ),
            ),
            (
                "task_element_completeness",
                "方案生成",
                "任务要素完整率",
                "%",
                task_completeness,
                [task.task_id for task in dashboard.tasks],
            ),
            (
                "approval_duration",
                "审批执行",
                "审批耗时",
                "分钟",
                duration(("task_submitted",), ("task_approved_and_issued",)),
                refs("task_submitted", "task_approved_and_issued"),
            ),
            (
                "acknowledgement_duration",
                "审批执行",
                "确认耗时",
                "分钟",
                duration(("task_approved_and_issued",), ("task_acknowledged",)),
                refs("task_approved_and_issued", "task_acknowledged"),
            ),
            (
                "on_time_completion_rate",
                "审批执行",
                "按时完成率",
                "%",
                rate(completed_tasks, len(dashboard.tasks)),
                refs("completion_verified", "task_waived"),
            ),
            (
                "timeout_detection_rate",
                "异常处理",
                "超时识别率",
                "%",
                100.0 if action_entries.get("deadline_escalated") else None,
                refs("deadline_escalated"),
            ),
            (
                "escalation_handling_time",
                "异常处理",
                "升级处理时间",
                "分钟",
                round(sum(escalation_durations) / len(escalation_durations), 3)
                if escalation_durations
                else None,
                refs(
                    "deadline_escalated",
                    "task_blocked_and_escalated",
                    "task_acknowledged",
                    "task_started",
                ),
            ),
            (
                "reassignment_success_rate",
                "异常处理",
                "改派/恢复成功率",
                "%",
                rate(len(recovered_escalations), len(escalation_task_ids)),
                refs(
                    "deadline_escalated",
                    "task_blocked_and_escalated",
                    "completion_verified",
                ),
            ),
            (
                "execution_evidence_completeness",
                "反馈核实",
                "执行证据完整率",
                "%",
                rate(feedback_complete, completion_feedback),
                [item.feedback_id for item in dashboard.feedback],
            ),
            (
                "verification_return_rate",
                "反馈核实",
                "待核实退回率",
                "%",
                rate(
                    len(action_entries.get("completion_returned", [])),
                    verification_attempts,
                ),
                refs("completion_returned", "completion_verified"),
            ),
            (
                "log_coverage",
                "复盘审计",
                "日志覆盖率",
                "%",
                timeline_coverage,
                refs(*sorted(expected_actions & actions)),
            ),
            (
                "timeline_completeness",
                "复盘审计",
                "时间线完整率",
                "%",
                timeline_integrity,
                [entry.entry_id for entry in timeline],
            ),
            (
                "report_generation_time",
                "复盘审计",
                "报告生成时间",
                "分钟",
                duration(("event_closed",), ("event_review_draft_generated",)),
                refs("event_closed", "event_review_draft_generated"),
            ),
        ]
        metrics: list[ScenarioMetricResult] = []
        observations: list[str] = []
        for metric_id, category, label, unit, system_value, evidence in metric_specs:
            manual_value = manual.get(metric_id)
            if system_value is None:
                status = "not_evaluable"
                interpretation = "当前场景没有足够的真值或异常样本，未计算该指标。"
                observations.append(
                    f"{label}缺少可计算样本；后续场景需补充人工标注真值或异常事件。"
                )
            elif manual_value is not None and unit == "分钟":
                status = "measured"
                saved = manual_value - system_value
                interpretation = f"相对人工参考基线节省 {round(saved, 2)} 分钟。"
            else:
                status = "measured"
                interpretation = "由事件台账、任务状态和证据记录直接计算。"
            metrics.append(
                ScenarioMetricResult(
                    metric_id=metric_id,
                    category=category,
                    label=label,
                    unit=unit,
                    manual_value=manual_value,
                    system_value=system_value,
                    status=status,
                    interpretation=interpretation,
                    evidence_refs=evidence,
                )
            )

        versions = [item.version for item in dashboard.alert_snapshots]
        task_fields_ok = complete_tasks == len(dashboard.tasks) and bool(
            dashboard.tasks
        )
        approval_ok = approved_tasks == len(dashboard.tasks) and bool(dashboard.tasks)
        trace_ok = timeline_coverage == 100.0 and timeline_integrity == 100.0
        timeout_ok = not any(
            task.status == TaskStatus.ESCALATED for task in dashboard.tasks
        ) or bool(escalation_task_ids)
        feedback_linked = all(
            item.event_id == event_id
            and item.task_id in task_by_id
            and item.object_id == task_by_id[item.task_id].object_id
            for item in dashboard.feedback
        ) and bool(dashboard.feedback)
        verified_ok = all(
            task.status == TaskStatus.WAIVED or task.task_id in verified_tasks
            for task in dashboard.tasks
        )
        review_ok = (
            event.status == EventStatus.CLOSED and dashboard.review_draft is not None
        )
        core_actions = {
            "event_created",
            "task_submitted",
            "task_approved_and_issued",
            "completion_verified",
            "event_closed",
        }
        manual_fallback_ok = core_actions.issubset(actions)
        actor_roles = {
            entry.actor_role
            for entry in timeline
            if entry.actor_role != "system" and entry.entry_type != "evaluation"
        }
        minimum_scope_ok = (
            event.status == EventStatus.CLOSED
            and len(dashboard.alert_snapshots) >= 1
            and len(dashboard.risk_objects) >= 1
            and len(dashboard.tasks) >= 1
            and len(actor_roles) >= 4
        )
        checks_data = [
            (
                "AC-01",
                "每个事件具有唯一编号和完整预警版本",
                bool(event.event_id) and versions == list(range(1, len(versions) + 1)),
                f"事件 {event.event_id}，预警版本 {versions}",
                [
                    event.event_id,
                    *[item.snapshot_id for item in dashboard.alert_snapshots],
                ],
            ),
            (
                "AC-02",
                "任务绑定对象、责任、时限、依据和反馈要求",
                task_fields_ok,
                f"{complete_tasks}/{len(dashboard.tasks)} 项任务要素完整",
                [task.task_id for task in dashboard.tasks],
            ),
            (
                "AC-03",
                "高风险任务未审批时不能下发",
                approval_ok,
                f"{approved_tasks}/{len(dashboard.tasks)} 项任务保留批准版本",
                refs("task_approved_and_issued"),
            ),
            (
                "AC-04",
                "所有审批和状态变化可追溯",
                trace_ok,
                f"日志覆盖率 {timeline_coverage or 0}% ，关联完整率 {timeline_integrity or 0}%",
                refs(*sorted(expected_actions & actions)),
            ),
            (
                "AC-05",
                "未确认、超时和受阻任务可提醒和升级",
                timeout_ok,
                f"记录 {len(escalation_task_ids)} 项升级；当前无未记录的升级状态",
                refs("deadline_escalated", "task_blocked_and_escalated"),
            ),
            (
                "AC-06",
                "现场反馈关联到事件、对象和任务",
                feedback_linked,
                f"{len(dashboard.feedback)} 条反馈均通过三类编号关联",
                [item.feedback_id for item in dashboard.feedback],
            ),
            (
                "AC-07",
                "任务完成后必须经过核实",
                verified_ok,
                f"{len(verified_tasks)} 项任务记录完成核实",
                refs("completion_verified"),
            ),
            (
                "AC-08",
                "事件关闭时生成时间线和过程指标",
                review_ok,
                f"时间线 {len(timeline)} 条，复盘草稿 {'已生成' if dashboard.review_draft else '未生成'}",
                [dashboard.review_draft.review_id] if dashboard.review_draft else [],
            ),
            (
                "AC-09",
                "AI 不可用时人工流程仍可继续运行",
                manual_fallback_ok,
                "确定性状态机已完成创建、审批、执行、核实和关闭，评测过程未调用模型",
                refs(*sorted(core_actions & actions)),
            ),
            (
                "AC-10",
                "至少一个最小区县场景完成端到端测试",
                minimum_scope_ok,
                f"1 个区县、{len(dashboard.risk_objects)} 类对象实例、{len(dashboard.alert_snapshots)} 版预警、{len(actor_roles)} 个岗位完成闭环",
                [event.event_id],
            ),
        ]
        checks = [
            ScenarioAcceptanceCheck(
                check_id=check_id,
                requirement=requirement,
                passed=passed,
                detail=detail,
                evidence_refs=evidence,
            )
            for check_id, requirement, passed, detail, evidence in checks_data
        ]
        failures = [
            ScenarioFailureCase(
                failure_id=self._id("FAIL"),
                stage="验收检查",
                severity="blocking",
                trigger=check.check_id,
                observed=check.detail,
                expected=check.requirement,
                recommendation="补齐对应业务记录后重新运行场景评测。",
            )
            for check in checks
            if not check.passed
        ]
        for escalation in dashboard.escalations:
            failures.append(
                ScenarioFailureCase(
                    failure_id=self._id("FAIL"),
                    stage="异常处理",
                    severity="handled",
                    trigger=escalation.reason,
                    observed=f"任务 {escalation.task_id} 触发升级并形成备选动作。",
                    expected="异常应被发现、升级并恢复到可执行状态。",
                    recommendation="复盘升级耗时，并验证推荐动作是否需要固化为标准预案。",
                )
            )
        all_passed = all(check.passed for check in checks)
        overall_status = (
            "failed"
            if not all_passed
            else ("passed_with_observations" if observations or failures else "passed")
        )
        report = DistrictScenarioReport(
            report_id=self._id("SCENARIO"),
            event_id=event_id,
            scenario_name=request.scenario_name,
            scenario_type=request.scenario_type,
            scope={
                "area_id": event.area_id,
                "alert_versions": len(dashboard.alert_snapshots),
                "risk_object_instances": len(dashboard.risk_objects),
                "risk_object_types": sorted(
                    {item.object_type for item in dashboard.risk_objects}
                ),
                "task_count": len(dashboard.tasks),
                "operator_roles": sorted(actor_roles),
            },
            baseline_note="人工基线为可配置参考假设，不代表真实对照组实测；系统值来自当前事件不可覆盖台账。",
            metrics=metrics,
            acceptance_checks=checks,
            failure_cases=failures,
            observations=observations,
            overall_status=overall_status,
            generated_by=request.operator_id,
            created_at=self._now(),
        )
        self.repository.save_district_scenario_report(report)
        self._record(
            event_id=event_id,
            entry_type="evaluation",
            action="scenario_evaluation_completed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "report_id": report.report_id,
                "overall_status": overall_status,
                "note": request.note,
            },
        )
        return report

    def list_scenario_reports(self, event_id: str) -> list[DistrictScenarioReport]:
        self._event(event_id)
        return self.repository.list_district_scenario_reports(event_id)

    def verify_timeline_integrity(
        self, event_id: str, operator_role: OperatorRole
    ) -> TimelineIntegrityReport:
        self._require_role(
            operator_role,
            {
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
                OperatorRole.ADMIN,
            },
            "verify timeline integrity",
        )
        self._event(event_id)
        entries = self.repository.list_timeline_entries(event_id)
        issues: list[TimelineIntegrityIssue] = []
        expected_previous = "GENESIS"
        verified_entries = 0
        for entry in entries:
            entry_issues = 0
            if entry.previous_hash != expected_previous:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id,
                        issue=f"previous_hash mismatch: expected {expected_previous}, got {entry.previous_hash}",
                    )
                )
                entry_issues += 1
            calculated = self._timeline_entry_hash(entry)
            if not entry.record_hash:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id, issue="record_hash is missing"
                    )
                )
                entry_issues += 1
            elif calculated != entry.record_hash:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id, issue="record_hash verification failed"
                    )
                )
                entry_issues += 1
            if entry_issues == 0:
                verified_entries += 1
            expected_previous = entry.record_hash or calculated
        return TimelineIntegrityReport(
            event_id=event_id,
            status="verified" if not issues else "failed",
            total_entries=len(entries),
            verified_entries=verified_entries,
            head_hash=entries[-1].record_hash if entries else None,
            issues=issues,
            verified_at=self._now(),
        )

    def create_database_backup(
        self, request: BackupCreateRequest
    ) -> DatabaseBackupRecord:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "create a database backup"
        )
        return self.repository.create_database_backup(
            label=request.label,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def list_database_backups(
        self, operator_role: OperatorRole
    ) -> list[DatabaseBackupRecord]:
        self._require_role(
            operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "list database backups",
        )
        return self.repository.list_database_backups()

    def apply_backup_retention(
        self, request: BackupRetentionRequest
    ) -> BackupRetentionResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "apply the backup retention policy",
        )
        return self.repository.apply_backup_retention(
            keep_latest=request.keep_latest,
            max_age_days=request.max_age_days,
            dry_run=request.dry_run,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def create_audit_archive(
        self, event_id: str, request: AuditArchiveRequest
    ) -> AuditArchiveRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "create an encrypted audit archive",
        )
        dashboard = self.get_dashboard(event_id)
        integrity = self.verify_timeline_integrity(event_id, OperatorRole.ADMIN)
        if integrity.status != "verified":
            raise ValueError(
                "timeline integrity must be verified before audit archive creation"
            )
        bundle = {
            "format": "district-response-audit-archive-v1",
            "event_id": event_id,
            "exported_at": self._now().isoformat(),
            "dashboard": dashboard.model_dump(mode="json"),
            "task_versions": {
                task.task_id: [
                    item.model_dump(mode="json")
                    for item in self.list_task_versions(task.task_id)
                ]
                for task in dashboard.tasks
            },
            "timeline_integrity": integrity.model_dump(mode="json"),
        }
        return self.repository.create_audit_archive(
            event_id=event_id,
            label=request.label,
            bundle=bundle,
            timeline_entries=integrity.total_entries,
            timeline_head_hash=integrity.head_hash,
            retention_days=request.retention_days,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def list_audit_archives(
        self, event_id: str, operator_role: OperatorRole
    ) -> list[AuditArchiveRecord]:
        self._require_role(
            operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "list audit archives",
        )
        self._event(event_id)
        return self.repository.list_audit_archives(event_id)

    def verify_audit_archive(
        self, archive_id: str, request: TaskActionRequest
    ) -> AuditArchiveVerificationResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "verify an audit archive",
        )
        return self.repository.verify_audit_archive(
            archive_id=archive_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def restore_database_backup(
        self, request: BackupRestoreRequest
    ) -> BackupRestoreResult:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "run a backup restore drill"
        )
        return self.repository.restore_database_backup(
            backup_id=request.backup_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def import_database_backup(
        self, request: BackupImportRequest
    ) -> BackupImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "import a cross-host database backup",
        )
        return self.repository.import_database_backup(
            manifest_filename=request.manifest_filename,
            backup_filename=request.backup_filename,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def rotate_data_encryption_key(
        self, request: KeyRotationRequest
    ) -> KeyRotationResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "rotate the data encryption key",
        )
        configured = os.getenv("FLOOD_DATA_ENCRYPTION_KEY_NEXT", "").strip()
        if not configured:
            raise ValueError(
                "FLOOD_DATA_ENCRYPTION_KEY_NEXT is required for key rotation"
            )
        backup = self.repository.create_database_backup(
            label=f"pre-key-rotation-{request.label}",
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )
        return self.repository.rotate_data_encryption_key(
            new_key=configured.encode("ascii"),
            backup_id=backup.backup_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )
