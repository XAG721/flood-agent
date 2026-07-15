from __future__ import annotations

from datetime import datetime, timedelta

from .feedback_policy import (
    classify_feedback,
    feedback_dedupe_key,
    missing_evidence_types,
    recommend_feedback_alternatives,
)
from .evidence_governance import (
    REQUIRED_TASK_FIELDS,
    build_field_evidence_map,
    infer_task_field_support,
)
from .state_machine import ensure_transition

from .models import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalRequest,
    CandidateRunRecord,
    DeadlineExtensionDecisionRequest,
    DeadlineExtensionRecord,
    DeadlineExtensionRequest,
    DeadlineExtensionStatus,
    EvidenceRole,
    FeedbackCategory,
    FeedbackRequest,
    ObjectVerificationStatus,
    OperatorRole,
    ResponseTask,
    RetrievalMode,
    RuleCheckResult,
    RuleEvaluationRecord,
    RuleOutcome,
    TaskActionRequest,
    TaskAssignmentRecord,
    TaskAssignmentRequest,
    TaskCreateRequest,
    TaskFeedback,
    TaskDraftGenerationRequest,
    TaskDraftGenerationResult,
    TaskEvidenceRef,
    TaskStatus,
    TaskUpdateRequest,
    TimelineEntry,
)

from .service_shared import APPROVAL_ROLES, EDIT_ROLES, EXECUTION_ROLES


class TaskWorkflowMixin:
    def create_task(self, event_id: str, request: TaskCreateRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "create a task draft")
        self._event(event_id, active=True)
        if request.idempotency_key:
            existing_task = next(
                (
                    item
                    for item in self.repository.list_response_tasks(event_id)
                    if item.creation_idempotency_key == request.idempotency_key
                ),
                None,
            )
            if existing_task is not None:
                return existing_task
        risk_object = self.repository.get_event_risk_object(event_id, request.object_id)
        if (
            risk_object is None
            or risk_object.verification_status != ObjectVerificationStatus.CONFIRMED
            or risk_object.stale
        ):
            raise ValueError(
                "tasks may only be created for confirmed event risk objects"
            )
        candidate_list = self._resolve_candidate_object_list(
            event_id,
            request.object_id,
            request.candidate_object_list_version,
        )
        if candidate_list is not None and (
            request.candidate_object_list_id != candidate_list.list_id
            or request.candidate_object_list_version != candidate_list.version
            or request.candidate_object_list_hash != candidate_list.content_hash
        ):
            raise ValueError(
                "task must bind the latest frozen candidate object list id, version and hash"
            )
        now = self._now()
        task = ResponseTask(
            **request.model_dump(
                exclude={
                    "operator_id",
                    "operator_role",
                    "terminal_id",
                    "idempotency_key",
                }
            ),
            task_id=self._id("TASK"),
            event_id=event_id,
            drafted_by=request.operator_id,
            created_at=now,
            updated_at=now,
            task_schema_version=self.TASK_SCHEMA_VERSION,
            rule_set_version=self.RULE_SET_VERSION,
            creation_idempotency_key=request.idempotency_key,
        )
        self.repository.save_response_task(task)
        self._save_task_version_snapshot(
            task,
            change_type="created",
            changed_fields=list(TaskCreateRequest.model_fields),
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
            note="任务草案创建",
        )
        self._record(
            event_id=event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            entry_type="task",
            action="task_draft_created",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={"status": task.status.value, "version": task.version},
            detail={"generated_by_ai": task.generated_by_ai, "version": task.version},
        )
        return task

    def generate_task_draft(
        self,
        event_id: str,
        object_id: str,
        request: TaskDraftGenerationRequest,
    ) -> TaskDraftGenerationResult:
        self._require_role(
            request.operator_role, EDIT_ROLES, "generate a grounded task draft"
        )
        self._event(event_id, active=True)
        risk_object = self.repository.get_event_risk_object(event_id, object_id)
        if risk_object is None:
            raise LookupError(f"risk object not found: {object_id}")
        if risk_object.verification_status != ObjectVerificationStatus.CONFIRMED:
            raise ValueError(
                "task drafts may only be generated for confirmed event risk objects"
            )
        if risk_object.stale:
            raise ValueError(
                "task drafts cannot use a stale risk object; rerun discovery and verification"
            )
        candidate_object_list = self._resolve_candidate_object_list(
            event_id,
            object_id,
            request.candidate_object_list_version,
        )

        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts:
            raise ValueError("event has no alert snapshot")
        alert = alerts[-1]
        query = request.query.strip() or self._build_task_query(alert, risk_object)
        if request.retrieval_mode in {
            RetrievalMode.CANARY,
            RetrievalMode.DEFAULT,
        } and not self.is_feature_enabled(
            "feature.frc_rag_formal_enabled",
            event_id=event_id,
            role=request.operator_role,
        ):
            raise ValueError(
                "FRC-RAG formal retrieval is gated off; use SHADOW/BASELINE_ONLY/REVIEW until Gate 2 is approved"
            )
        policy_documents, retrieval_trace = self._run_retrieval_strategy(
            query, request.retrieval_mode
        )
        evidence = self._build_task_evidence(
            alert, risk_object, policy_documents, request=request
        )
        conflicts, nli_assessments = self._analyze_evidence_conflicts(evidence)
        covered_roles = {
            role.value
            for item in evidence
            if item.source_type != "workflow_contract"
            for role in item.roles
        }
        role_coverage = {
            role.value: role.value in covered_roles for role in EvidenceRole
        }
        missing_roles = [role for role, covered in role_coverage.items() if not covered]
        field_evidence_map = build_field_evidence_map(evidence)
        blocking_missing_fields = [
            field for field in REQUIRED_TASK_FIELDS if not field_evidence_map[field]
        ]
        plan_basis = self._build_plan_basis(policy_documents)
        action = self._extract_supported_action(policy_documents, risk_object)
        validation_errors = [f"缺少 {role} 证据" for role in missing_roles]
        validation_errors.extend(
            f"关键任务字段缺少证据：{field}"
            for field in blocking_missing_fields
        )
        if not plan_basis:
            validation_errors.append("缺少带版本和条款号的预案引用")
        if not action:
            validation_errors.append("检索证据不足以形成可执行处置动作")
        if conflicts:
            validation_errors.extend(
                f"存在未决{item.conflict_type}冲突：{item.field_name}"
                for item in conflicts
            )

        grounding_summary = self._build_grounding_summary(
            evidence, role_coverage, validation_errors
        )
        evidence_package = self._build_evidence_package(
            event_id=event_id,
            object_id=object_id,
            evidence=evidence,
            role_coverage=role_coverage,
            validation_errors=validation_errors,
            conflicts=conflicts,
            nli_assessments=nli_assessments,
            retrieval_mode=request.retrieval_mode,
            retrieval_trace=retrieval_trace,
            action=action,
            plan_basis=plan_basis,
            operator_id=request.operator_id,
        )
        self.repository.save_evidence_package(evidence_package)
        if validation_errors:
            self._record(
                event_id=event_id,
                object_id=object_id,
                entry_type="task_draft",
                action="task_draft_blocked_by_evidence_gate",
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                detail={
                    "missing_roles": missing_roles,
                    "validation_errors": validation_errors,
                    "evidence_source_ids": [item.source_id for item in evidence],
                    "generation_version": self.GENERATION_VERSION,
                },
            )
            return TaskDraftGenerationResult(
                event_id=event_id,
                object_id=object_id,
                status="insufficient_evidence",
                evidence=evidence,
                role_coverage=role_coverage,
                missing_roles=missing_roles,
                validation_errors=validation_errors,
                grounding_summary=grounding_summary,
                generation_version=self.GENERATION_VERSION,
                evidence_package=evidence_package,
            )

        now = self._now()
        task = self.create_task(
            event_id,
            TaskCreateRequest(
                object_id=object_id,
                title=f"{risk_object.name}现场核查与响应处置",
                action=action,
                responsible_organization=risk_object.responsible_organization,
                responsible_role=risk_object.responsible_role,
                cooperate_roles=self._cooperate_roles(risk_object.object_type),
                deadline_at=now + timedelta(minutes=request.deadline_minutes),
                acknowledge_deadline_at=now
                + timedelta(minutes=request.acknowledge_minutes),
                required_evidence=self._required_evidence(risk_object.object_type),
                plan_basis=plan_basis,
                approval_policy=self._approval_policy(alert.level, action),
                escalation_rule="未确认时提醒成员单位联络员；超时或受阻时升级至防办审核员并记录异常单。",
                operator_id=request.operator_id,
                operator_role=request.operator_role,
                terminal_id=request.terminal_id,
                generated_by_ai=True,
                generation_version=self.GENERATION_VERSION,
                grounding_summary=grounding_summary,
                evidence_role_coverage=role_coverage,
                source_evidence=evidence,
                evidence_package_id=evidence_package.package_id,
                evidence_package_version=evidence_package.version,
                evidence_package_hash=evidence_package.content_hash,
                candidate_object_list_id=(
                    candidate_object_list.list_id if candidate_object_list else None
                ),
                candidate_object_list_version=(
                    candidate_object_list.version if candidate_object_list else None
                ),
                candidate_object_list_hash=(
                    candidate_object_list.content_hash
                    if candidate_object_list
                    else None
                ),
                idempotency_key=f"draft:{event_id}:{object_id}:{evidence_package.content_hash}",
            ),
        )
        self._record(
            event_id=event_id,
            task_id=task.task_id,
            object_id=object_id,
            entry_type="task_draft",
            action="grounded_task_draft_generated",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={"status": task.status.value, "version": task.version},
            detail={
                "evidence_source_ids": [item.source_id for item in evidence],
                "role_coverage": role_coverage,
                "generation_version": self.GENERATION_VERSION,
                "candidate_object_list_id": (
                    candidate_object_list.list_id if candidate_object_list else None
                ),
                "candidate_object_list_version": (
                    candidate_object_list.version if candidate_object_list else None
                ),
                "candidate_object_list_hash": (
                    candidate_object_list.content_hash
                    if candidate_object_list
                    else None
                ),
            },
        )
        return TaskDraftGenerationResult(
            event_id=event_id,
            object_id=object_id,
            status="ready",
            task=task,
            evidence=evidence,
            role_coverage=role_coverage,
            grounding_summary=grounding_summary,
            generation_version=self.GENERATION_VERSION,
            evidence_package=evidence_package,
        )

    def list_candidate_runs(self, event_id: str) -> list[CandidateRunRecord]:
        self._event(event_id)
        return self.repository.list_candidate_runs(event_id)

    def list_task_transitions(self, task_id: str) -> list[TimelineEntry]:
        task = self._task(task_id)
        return [
            item
            for item in self.repository.list_timeline_entries(task.event_id)
            if item.task_id == task_id
        ]

    def replay_event(self, event_id: str, request: TaskActionRequest) -> dict:
        self._require_role(
            request.operator_role,
            {
                OperatorRole.AUDITOR,
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.ADMIN,
            },
            "replay an event",
        )
        integrity = self.verify_timeline_integrity(event_id, request.operator_role)
        if integrity.status != "verified":
            raise ValueError("event timeline integrity must verify before replay")
        return {
            "mode": "read_only_replay",
            "integrity": integrity,
            "dashboard": self.get_dashboard(event_id),
        }

    @staticmethod
    def _normalize_evidence_refs(
        evidence: list[TaskEvidenceRef],
    ) -> list[TaskEvidenceRef]:
        normalized: list[TaskEvidenceRef] = []
        for item in evidence:
            support = item.field_support or infer_task_field_support(
                item.excerpt,
                [role.value for role in item.roles],
            )
            source_locator = item.source_locator or (
                f"{item.source_type or 'evidence'}://{item.source_id}"
            )
            normalized.append(
                item.model_copy(
                    update={
                        "field_support": support,
                        "source_locator": source_locator,
                    }
                )
            )
        return normalized

    def update_task(self, task_id: str, request: TaskUpdateRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "edit a task draft")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.DRAFT:
            raise ValueError(
                "only draft tasks can be edited; return the task to draft first"
            )
        self._assert_expected_version(task, request.expected_version)
        changes = request.model_dump(
            exclude={
                "operator_id",
                "operator_role",
                "note",
                "terminal_id",
                "expected_version",
            },
            exclude_none=True,
        )
        if not changes:
            return task
        before_state = {field: getattr(task, field) for field in changes}
        changes.update({"version": task.version + 1, "updated_at": self._now()})
        task = task.model_copy(update=changes)
        self.repository.save_response_task(task)
        self._save_task_version_snapshot(
            task,
            change_type="revised",
            changed_fields=sorted(
                field for field in changes if field not in {"updated_at", "version"}
            ),
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
            note=request.note,
        )
        self._record(
            event_id=task.event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            entry_type="task",
            action="task_draft_revised",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state=before_state,
            after_state={
                field: getattr(task, field)
                for field in changes
                if field not in {"updated_at", "version"}
            },
            detail={
                "version": task.version,
                "changed_fields": sorted(changes),
                "note": request.note,
            },
        )
        return task

    def submit_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "submit a task")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.DRAFT:
            raise ValueError("only draft tasks can be submitted")
        self._assert_expected_version(task, request.expected_version)
        known_ids = {
            item.task_id for item in self.repository.list_response_tasks(task.event_id)
        }
        missing = [item for item in task.dependencies if item not in known_ids]
        if missing:
            raise ValueError(f"unknown task dependencies: {', '.join(missing)}")
        evaluation = self._evaluate_task_rules(task, request)
        if evaluation.overall_outcome == RuleOutcome.HARD_BLOCK:
            raise ValueError(
                "task is blocked by rules: "
                + "; ".join(
                    item.message
                    for item in evaluation.checks
                    if item.outcome == RuleOutcome.HARD_BLOCK
                )
            )
        warnings = [
            item.message
            for item in evaluation.checks
            if item.outcome == RuleOutcome.SOFT_WARNING
        ]
        ensure_transition(task.status, TaskStatus.PENDING_APPROVAL)
        task = task.model_copy(
            update={
                "status": TaskStatus.PENDING_APPROVAL,
                "last_rule_evaluation_id": evaluation.evaluation_id,
                "validation_warnings": warnings,
                "updated_at": self._now(),
            }
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_submitted",
            request,
            {"version": task.version, "previous_status": TaskStatus.DRAFT.value},
        )
        return task

    def decide_task(self, task_id: str, request: ApprovalRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, APPROVAL_ROLES, "approve or reject a task"
        )
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.PENDING_APPROVAL:
            raise ValueError("task is not pending approval")
        self._assert_expected_version(task, request.expected_version)
        if (
            task.approval_policy == ApprovalPolicy.COMMANDER_REQUIRED
            and request.operator_role != OperatorRole.COMMANDER
        ):
            raise PermissionError("this high-risk task requires commander approval")
        if (
            task.approval_policy == ApprovalPolicy.COMMANDER_REQUIRED
            and task.drafted_by == request.operator_id
        ):
            raise PermissionError("the drafter cannot approve their own high-risk task")
        evaluation = self._evaluate_task_rules(task, request)
        if (
            request.decision == ApprovalDecision.APPROVED
            and evaluation.overall_outcome == RuleOutcome.HARD_BLOCK
        ):
            raise ValueError("task approval is blocked by the current rule evaluation")
        task_payload_hash = self._task_payload_hash(task)
        evidence_package_hash = self._evidence_package_hash(task)
        record = ApprovalRecord(
            approval_id=self._id("APR"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            decision=request.decision,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            note=request.note,
            created_at=self._now(),
            task_payload_hash=task_payload_hash,
            evidence_package_hash=evidence_package_hash,
            task_schema_version=task.task_schema_version,
            rule_set_version=self.RULE_SET_VERSION,
            rule_evaluation_id=evaluation.evaluation_id,
        )
        previous_status = task.status
        outbox = None
        if request.decision == ApprovalDecision.APPROVED:
            ensure_transition(task.status, TaskStatus.ISSUED)
            outbox = self._build_dispatch_outbox(task, record)
            task = task.model_copy(
                update={
                    "status": TaskStatus.ISSUED,
                    "approved_version": task.version,
                    "approval_payload_hash": task_payload_hash,
                    "evidence_package_hash": evidence_package_hash,
                    "rule_set_version": self.RULE_SET_VERSION,
                    "dispatch_message_id": outbox.message_id,
                    "updated_at": self._now(),
                }
            )
            action = "task_approved_and_issued"
        else:
            ensure_transition(task.status, TaskStatus.DRAFT)
            task = task.model_copy(
                update={
                    "status": TaskStatus.DRAFT,
                    "version": task.version + 1,
                    "updated_at": self._now(),
                }
            )
            action = "task_rejected_to_draft"
        self.repository.commit_task_decision(record, task, outbox)
        if outbox is not None and self.is_feature_enabled(
            "feature.simulated_dispatch", event_id=task.event_id
        ):
            outbox = self._dispatch_outbox_message(outbox)
        if request.decision == ApprovalDecision.REJECTED:
            self._save_task_version_snapshot(
                task,
                change_type="approval_rejected",
                changed_fields=["status", "version"],
                operator_id=request.operator_id,
                terminal_id=request.terminal_id,
                note=request.note,
            )
        self._task_event(
            task,
            action,
            request,
            {
                "approval_id": record.approval_id,
                "task_payload_hash": record.task_payload_hash,
                "evidence_package_hash": record.evidence_package_hash,
                "dispatch_message_id": task.dispatch_message_id,
                "note": request.note,
                "previous_status": previous_status.value,
            },
        )
        return task

    def acknowledge_task(
        self, task_id: str, request: TaskActionRequest
    ) -> ResponseTask:
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if (
            task.approval_payload_hash
            and self._task_payload_hash(task) != task.approval_payload_hash
        ):
            raise ValueError(
                "approved task payload hash does not match the dispatch payload"
            )
        return self._transition(
            task_id,
            request,
            {TaskStatus.ISSUED, TaskStatus.ESCALATED, TaskStatus.BLOCKED},
            TaskStatus.ACKNOWLEDGED,
            "task_acknowledged",
        )

    @staticmethod
    def _assert_expected_version(
        task: ResponseTask, expected_version: int | None
    ) -> None:
        if expected_version is not None and expected_version != task.version:
            raise ValueError(
                f"task version conflict: expected {expected_version}, current {task.version}"
            )

    def _evaluate_task_rules(
        self, task: ResponseTask, request: TaskActionRequest
    ) -> RuleEvaluationRecord:
        risk_object = self.repository.get_event_risk_object(
            task.event_id, task.object_id
        )
        packages = self.repository.list_evidence_packages(
            task.event_id, object_id=task.object_id
        )
        package = next(
            (
                item
                for item in reversed(packages)
                if item.package_id == task.evidence_package_id
            ),
            None,
        )
        checks: list[RuleCheckResult] = []

        def add(
            rule_id: str,
            outcome: RuleOutcome,
            message: str,
            field_name: str | None = None,
        ):
            checks.append(
                RuleCheckResult(
                    rule_id=rule_id,
                    outcome=outcome,
                    message=message,
                    field_name=field_name,
                )
            )

        add(
            "RULE-OBJECT-CONFIRMED",
            RuleOutcome.PASS
            if risk_object
            and risk_object.verification_status == ObjectVerificationStatus.CONFIRMED
            else RuleOutcome.HARD_BLOCK,
            "风险对象已人工确认"
            if risk_object
            and risk_object.verification_status == ObjectVerificationStatus.CONFIRMED
            else "风险对象未人工确认",
            "object_id",
        )
        deadline_valid = (
            task.acknowledge_deadline_at
            <= (task.start_deadline_at or task.deadline_at)
            <= task.deadline_at
            <= (task.verification_deadline_at or task.deadline_at)
        )
        add(
            "RULE-DEADLINES-ORDERED",
            RuleOutcome.PASS if deadline_valid else RuleOutcome.HARD_BLOCK,
            "四类时限顺序有效"
            if deadline_valid
            else "接收、开始、完成和核验时限顺序无效",
            "deadline_at",
        )
        required_complete = bool(
            task.title.strip()
            and task.action.strip()
            and task.required_evidence
            and task.plan_basis
        )
        add(
            "RULE-REQUIRED-FIELDS",
            RuleOutcome.PASS if required_complete else RuleOutcome.HARD_BLOCK,
            "任务必填字段完整"
            if required_complete
            else "任务标题、动作、证据要求或预案依据缺失",
        )
        if task.generated_by_ai:
            unresolved_conflicts = (
                [
                    item
                    for item in package.conflicts
                    if item.resolution_status != "resolved"
                ]
                if package
                else []
            )
            package_ok = bool(
                package
                and package.status == "frozen"
                and not unresolved_conflicts
                and not package.blocking_missing_fields
                and package.content_hash == task.evidence_package_hash
            )
            add(
                "RULE-EVIDENCE-PACKAGE-FROZEN",
                RuleOutcome.PASS if package_ok else RuleOutcome.HARD_BLOCK,
                "冻结证据包与草案哈希一致"
                if package_ok
                else "AI 草案缺少冻结证据包、存在未决冲突/关键字段缺失或哈希不一致",
                "evidence_package_id",
            )
            critical_field_sources = (
                package.field_evidence_map if package is not None else {}
            )
            evidence_by_id = {
                item.source_id: item for item in task.source_evidence
            }
            bound = bool(task.source_evidence) and all(
                critical_field_sources.get(field)
                and any(
                    source_id in evidence_by_id
                    and evidence_by_id[source_id].source_locator
                    and evidence_by_id[source_id].field_support.get(field, 0) >= 0.18
                    for source_id in critical_field_sources[field]
                )
                for field in REQUIRED_TASK_FIELDS
            )
            add(
                "RULE-EVIDENCE-BOUND",
                RuleOutcome.PASS if bound else RuleOutcome.HARD_BLOCK,
                "关键字段已绑定来源证据" if bound else "关键字段未绑定可定位来源证据",
                "source_evidence",
            )
        if task.task_id in task.dependencies:
            add(
                "RULE-NO-SELF-DEPENDENCY",
                RuleOutcome.HARD_BLOCK,
                "任务不能依赖自身",
                "dependencies",
            )
        else:
            add(
                "RULE-NO-SELF-DEPENDENCY",
                RuleOutcome.PASS,
                "任务依赖不包含自身",
                "dependencies",
            )
        add(
            "RULE-COOPERATION-DEFINED",
            RuleOutcome.PASS if task.cooperate_roles else RuleOutcome.SOFT_WARNING,
            "已定义协同岗位"
            if task.cooperate_roles
            else "未定义协同岗位，请审批人员确认是否为单岗位任务",
            "cooperate_roles",
        )
        overall = (
            RuleOutcome.HARD_BLOCK
            if any(item.outcome == RuleOutcome.HARD_BLOCK for item in checks)
            else RuleOutcome.SOFT_WARNING
            if any(item.outcome == RuleOutcome.SOFT_WARNING for item in checks)
            else RuleOutcome.PASS
        )
        record = RuleEvaluationRecord(
            evaluation_id=self._id("RULEEVAL"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            rule_set_version=self.RULE_SET_VERSION,
            overall_outcome=overall,
            checks=checks,
            evaluated_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_rule_evaluation(record)
        self._task_event(
            task,
            "task_rules_evaluated",
            request,
            {
                "evaluation_id": record.evaluation_id,
                "outcome": overall.value,
                "hard_blocks": [
                    item.rule_id
                    for item in checks
                    if item.outcome == RuleOutcome.HARD_BLOCK
                ],
                "soft_warnings": [
                    item.rule_id
                    for item in checks
                    if item.outcome == RuleOutcome.SOFT_WARNING
                ],
            },
        )
        return record

    def start_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        task = self._task(task_id)
        if task.assignee_id is None:
            raise ValueError(
                "task must be assigned to a field operator before execution starts"
            )
        if (
            request.operator_role != OperatorRole.FIELD_OPERATOR
            or request.operator_id != task.assignee_id
        ):
            raise PermissionError(
                "only the assigned field operator may start this task"
            )
        return self._transition(
            task_id,
            request,
            {TaskStatus.ACKNOWLEDGED, TaskStatus.ESCALATED, TaskStatus.BLOCKED},
            TaskStatus.IN_PROGRESS,
            "task_started",
        )

    def assign_task(self, task_id: str, request: TaskAssignmentRequest) -> ResponseTask:
        self._require_role(
            request.operator_role,
            {OperatorRole.LIAISON, OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "assign a field operator",
        )
        if request.assignee_role != OperatorRole.FIELD_OPERATOR:
            raise ValueError("response tasks may only be assigned to a field operator")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.ESCALATED,
            TaskStatus.BLOCKED,
            TaskStatus.PARTIALLY_COMPLETED,
            TaskStatus.TAKEN_OVER,
        }:
            raise ValueError("task cannot be assigned in the current status")
        previous_assignee = task.assignee_id
        if previous_assignee == request.assignee_id:
            return task
        assignment_version = task.assignment_version + 1
        task = task.model_copy(
            update={
                "assignee_id": request.assignee_id,
                "assignee_name": request.assignee_name,
                "assignee_role": request.assignee_role,
                "assignment_version": assignment_version,
                "updated_at": self._now(),
            }
        )
        assignment = TaskAssignmentRecord(
            assignment_id=self._id("ASSIGN"),
            event_id=task.event_id,
            task_id=task.task_id,
            assignment_version=assignment_version,
            previous_assignee_id=previous_assignee,
            assignee_id=request.assignee_id,
            assignee_name=request.assignee_name,
            assignee_role=request.assignee_role,
            reason=request.reason,
            assigned_by=request.operator_id,
            assigned_by_role=request.operator_role,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_response_task(task)
        self.repository.save_task_assignment(assignment)
        action = "task_reassigned" if previous_assignee else "task_assigned"
        self._task_event(
            task,
            action,
            request,
            {
                "assignment_id": assignment.assignment_id,
                "assignment_version": assignment_version,
                "previous_assignee_id": previous_assignee,
                "assignee_id": request.assignee_id,
                "reason": request.reason,
            },
        )
        return task

    def submit_feedback(self, task_id: str, request: FeedbackRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, EXECUTION_ROLES, "submit task feedback"
        )
        task = self._task(task_id)
        if (
            task.assignee_id
            and request.operator_role == OperatorRole.FIELD_OPERATOR
            and request.operator_id != task.assignee_id
        ):
            raise PermissionError(
                "only the assigned field operator may submit execution feedback"
            )
        self._event(task.event_id, active=True)
        previous_status = task.status
        category, classification_reason = classify_feedback(request)
        dedupe_key = feedback_dedupe_key(request)
        duplicate = next(
            (
                item
                for item in self.repository.list_task_feedback(task_id)
                if item.dedupe_key == dedupe_key
            ),
            None,
        )
        if duplicate is not None:
            self._task_event(
                task,
                "duplicate_feedback_merged",
                request,
                {
                    "duplicate_of": duplicate.feedback_id,
                    "category": duplicate.category.value,
                    "dedupe_key": dedupe_key,
                },
            )
            return task
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.ESCALATED,
            TaskStatus.BLOCKED,
            TaskStatus.PARTIALLY_COMPLETED,
            TaskStatus.TAKEN_OVER,
        }:
            raise ValueError("feedback cannot be submitted in the current task status")
        escalation_reason = request.blocked_reason or request.resource_gap
        if category == FeedbackCategory.COORDINATION_REQUEST:
            escalation_reason = request.summary
        if category == FeedbackCategory.COMPLETION:
            missing_types = missing_evidence_types(task, request.evidence)
            if missing_types:
                raise ValueError(
                    f"required evidence missing: {', '.join(missing_types)}"
                )
        feedback = TaskFeedback(
            feedback_id=self._id("FDB"),
            event_id=task.event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            summary=request.summary,
            evidence=request.evidence,
            blocked_reason=request.blocked_reason,
            resource_gap=request.resource_gap,
            category=category,
            classification_reason=classification_reason,
            dedupe_key=dedupe_key,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            created_at=self._now(),
        )
        self.repository.save_task_feedback(feedback)
        if escalation_reason:
            previous = task.status
            ensure_transition(task.status, TaskStatus.BLOCKED)
            task = task.model_copy(
                update={"status": TaskStatus.BLOCKED, "updated_at": self._now()}
            )
            escalation = self._save_escalation(
                task,
                previous,
                escalation_reason,
                "duty_officer",
                recommended_actions=recommend_feedback_alternatives(category),
            )
            action = "task_blocked_and_escalated"
        elif category == FeedbackCategory.COMPLETION:
            ensure_transition(task.status, TaskStatus.PENDING_VERIFICATION)
            task = task.model_copy(
                update={
                    "status": TaskStatus.PENDING_VERIFICATION,
                    "updated_at": self._now(),
                }
            )
            action = "completion_submitted"
        elif category == FeedbackCategory.PARTIAL_COMPLETION:
            ensure_transition(task.status, TaskStatus.PARTIALLY_COMPLETED)
            task = task.model_copy(
                update={
                    "status": TaskStatus.PARTIALLY_COMPLETED,
                    "updated_at": self._now(),
                }
            )
            action = "task_partially_completed"
        else:
            action = "situation_feedback_recorded"
        self.repository.save_response_task(task)
        self._task_event(
            task,
            action,
            request,
            {
                "feedback_id": feedback.feedback_id,
                "summary": request.summary,
                "category": category.value,
                "escalation_id": escalation.escalation_id
                if escalation_reason
                else None,
                "previous_status": previous_status.value,
            },
        )
        return task

    def verify_completion(
        self, task_id: str, approved: bool, request: TaskActionRequest
    ) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "verify task completion")
        task = self._task(task_id)
        if task.status != TaskStatus.PENDING_VERIFICATION:
            raise ValueError("task is not pending verification")
        completion_feedback = [
            item
            for item in self.repository.list_task_feedback(task_id)
            if item.category == FeedbackCategory.COMPLETION
        ]
        if (
            completion_feedback
            and completion_feedback[-1].operator_id == request.operator_id
        ):
            raise PermissionError(
                "the completion submitter cannot verify their own result"
            )
        if task.assignee_id and task.assignee_id == request.operator_id:
            raise PermissionError("the assigned executor cannot verify their own task")
        target = TaskStatus.COMPLETED if approved else TaskStatus.IN_PROGRESS
        ensure_transition(task.status, target)
        previous_status = task.status
        task = task.model_copy(update={"status": target, "updated_at": self._now()})
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "completion_verified" if approved else "completion_returned",
            request,
            {"note": request.note, "previous_status": previous_status.value},
        )
        return task

    def request_deadline_extension(
        self, task_id: str, request: DeadlineExtensionRequest
    ) -> DeadlineExtensionRecord:
        self._require_role(
            request.operator_role, EXECUTION_ROLES, "request a deadline extension"
        )
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.ESCALATED,
            TaskStatus.PARTIALLY_COMPLETED,
        }:
            raise ValueError(
                "deadline extension cannot be requested in the current task status"
            )
        current_completion = task.effective_completion_deadline_at or task.deadline_at
        if request.proposed_completion_deadline_at <= current_completion:
            raise ValueError(
                "proposed completion deadline must be later than the current effective deadline"
            )
        record = DeadlineExtensionRecord(
            extension_id=self._id("EXT"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            status=DeadlineExtensionStatus.REQUESTED,
            original_start_deadline_at=task.effective_start_deadline_at
            or task.start_deadline_at,
            original_completion_deadline_at=current_completion,
            original_verification_deadline_at=task.effective_verification_deadline_at
            or task.verification_deadline_at,
            proposed_start_deadline_at=request.proposed_start_deadline_at,
            proposed_completion_deadline_at=request.proposed_completion_deadline_at,
            proposed_verification_deadline_at=request.proposed_verification_deadline_at,
            request_reason=request.reason,
            requested_by=request.operator_id,
            requested_by_role=request.operator_role,
            created_at=self._now(),
        )
        self.repository.save_deadline_extension(record)
        self._task_event(
            task,
            "deadline_extension_requested",
            request,
            {"extension_id": record.extension_id, "reason": request.reason},
        )
        return record

    def decide_deadline_extension(
        self, extension_id: str, request: DeadlineExtensionDecisionRequest
    ) -> DeadlineExtensionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "decide a deadline extension",
        )
        record = self.repository.get_deadline_extension(extension_id)
        if record is None:
            raise LookupError(f"deadline extension not found: {extension_id}")
        if record.status != DeadlineExtensionStatus.REQUESTED:
            raise ValueError("deadline extension has already been decided")
        if record.requested_by == request.operator_id:
            raise PermissionError(
                "the requester cannot decide their own deadline extension"
            )
        task = self._task(record.task_id)
        now = self._now()
        status = (
            DeadlineExtensionStatus.APPROVED
            if request.approved
            else DeadlineExtensionStatus.REJECTED
        )
        record = record.model_copy(
            update={
                "status": status,
                "decided_by": request.operator_id,
                "decision_reason": request.reason,
                "decided_at": now,
            }
        )
        self.repository.save_deadline_extension(record)
        if request.approved:
            task = task.model_copy(
                update={
                    "effective_start_deadline_at": record.proposed_start_deadline_at
                    or task.effective_start_deadline_at
                    or task.start_deadline_at,
                    "effective_completion_deadline_at": record.proposed_completion_deadline_at,
                    "effective_verification_deadline_at": record.proposed_verification_deadline_at
                    or task.effective_verification_deadline_at
                    or task.verification_deadline_at,
                    "active_extension_id": record.extension_id,
                    "updated_at": now,
                }
            )
            self.repository.save_response_task(task)
        self._task_event(
            task,
            "deadline_extension_approved"
            if request.approved
            else "deadline_extension_rejected",
            request,
            {"extension_id": record.extension_id, "reason": request.reason},
        )
        return record

    def cancel_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, {OperatorRole.COMMANDER}, "cancel a task"
        )
        if len(request.note.strip()) < 4:
            raise ValueError("task cancellation requires a reason")
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.WAIVED,
            TaskStatus.CANCELLED,
        }:
            raise ValueError("terminal tasks cannot be cancelled")
        previous = task.status
        ensure_transition(task.status, TaskStatus.CANCELLED)
        task = task.model_copy(
            update={"status": TaskStatus.CANCELLED, "updated_at": self._now()}
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_cancelled",
            request,
            {"previous_status": previous.value, "reason": request.note},
        )
        return task

    def take_over_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "take over a task",
        )
        if len(request.note.strip()) < 4:
            raise ValueError("manual takeover requires a reason")
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.ESCALATED,
            TaskStatus.PARTIALLY_COMPLETED,
        }:
            raise ValueError("task cannot be taken over in the current status")
        previous = task.status
        ensure_transition(task.status, TaskStatus.TAKEN_OVER)
        task = task.model_copy(
            update={
                "status": TaskStatus.TAKEN_OVER,
                "assignee_id": request.operator_id,
                "assignee_name": "人工接管岗位",
                "assignee_role": request.operator_role,
                "updated_at": self._now(),
            }
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_taken_over",
            request,
            {"previous_status": previous.value, "reason": request.note},
        )
        return task

    def waive_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, {OperatorRole.COMMANDER}, "waive a task"
        )
        task = self._task(task_id)
        if task.status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}:
            return task
        previous_status = task.status
        ensure_transition(task.status, TaskStatus.WAIVED)
        task = task.model_copy(
            update={"status": TaskStatus.WAIVED, "updated_at": self._now()}
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_waived",
            request,
            {"reason": request.note, "previous_status": previous_status.value},
        )
        return task

    def run_deadline_sweep(
        self, event_id: str, request: TaskActionRequest, *, now: datetime | None = None
    ) -> list[ResponseTask]:
        self._require_role(request.operator_role, EDIT_ROLES, "run deadline escalation")
        self._event(event_id, active=True)
        moment = now or self._now()
        escalated: list[ResponseTask] = []
        for task in self.repository.list_response_tasks(event_id):
            reason = None
            deadline_type = None
            start_deadline = task.effective_start_deadline_at or task.start_deadline_at
            completion_deadline = (
                task.effective_completion_deadline_at or task.deadline_at
            )
            verification_deadline = (
                task.effective_verification_deadline_at or task.verification_deadline_at
            )
            if (
                task.status == TaskStatus.ISSUED
                and task.acknowledge_deadline_at < moment
            ):
                reason = "任务超过确认时限仍未确认"
                deadline_type = "acknowledge"
            elif (
                task.status == TaskStatus.ACKNOWLEDGED
                and start_deadline
                and start_deadline < moment
            ):
                reason = "任务超过开始时限仍未开始"
                deadline_type = "start"
            elif (
                task.status
                in {
                    TaskStatus.IN_PROGRESS,
                    TaskStatus.PARTIALLY_COMPLETED,
                    TaskStatus.BLOCKED,
                }
                and completion_deadline < moment
            ):
                reason = "任务超过完成时限"
                deadline_type = "completion"
            elif (
                task.status == TaskStatus.PENDING_VERIFICATION
                and verification_deadline
                and verification_deadline < moment
            ):
                reason = "任务超过核验时限"
                deadline_type = "verification"
            if reason:
                previous = task.status
                ensure_transition(task.status, TaskStatus.ESCALATED)
                task = task.model_copy(
                    update={"status": TaskStatus.ESCALATED, "updated_at": self._now()}
                )
                self.repository.save_response_task(task)
                escalation = self._save_escalation(task, previous, reason, "reviewer")
                self._task_event(
                    task,
                    "deadline_escalated",
                    request,
                    {
                        "reason": reason,
                        "deadline_type": deadline_type,
                        "escalation_id": escalation.escalation_id,
                        "previous_status": previous.value,
                    },
                )
                escalated.append(task)
        return escalated
